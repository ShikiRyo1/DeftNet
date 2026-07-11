"""Core DeftNet / DEFT-Net architecture.

The implementation is intentionally self-contained and notebook-free. It keeps
the research idea used in the manuscript:

1. train heterogeneous U-Net-style experts independently,
2. discard expert decoders and freeze the expert encoders,
3. fuse same-scale encoder features with depth-banded HSAF gates,
4. feed the fused pyramid into a single trainable decoder.

The default depth-band policy follows the current public method specification. Legacy
checkpoints can be loaded with a custom `band_policy` if their exact admission
sets differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LayerName = str
ExpertName = str

CANONICAL_EXPERT_NAMES: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
LEGACY_EXPERT_NAMES: Tuple[str, ...] = ("E5", "E7", "E9", "E11", "E12")
LEGACY_TO_CANONICAL = dict(zip(LEGACY_EXPERT_NAMES, CANONICAL_EXPERT_NAMES))
CANONICAL_TO_LEGACY = {value: key for key, value in LEGACY_TO_CANONICAL.items()}


@dataclass
class DeftNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 16
    norm: str = "in"
    expert_names: Tuple[str, ...] = CANONICAL_EXPERT_NAMES
    band_policy: Dict[LayerName, Tuple[ExpertName, ...]] = field(
        default_factory=lambda: {
            "e1": ("E1", "E2", "E3"),
            "e2": ("E1", "E2", "E3"),
            "e3": ("E1", "E2", "E3", "E4", "E5"),
            "e4": ("E4", "E5"),
            "e5": ("E4", "E5"),
        }
    )
    fusion_mode: str = "hsaf"
    use_deep_supervision: bool = False
    use_feature_adapters: bool = True
    adapter_dropout: float = 0.05
    adapter_residual: bool = True
    hsaf_gate_dropout: float = 0.10
    hsaf_temperature: float = 1.5
    e11_depths: Tuple[int, ...] = (0, 0, 0, 1, 1)
    e11_heads: Tuple[int, ...] = (0, 0, 0, 4, 8)
    swin_depths: Tuple[int, ...] = (0, 0, 1, 1, 1)
    swin_heads: Tuple[int, ...] = (0, 0, 4, 8, 8)
    swin_window: int = 8
    dense_layers: Tuple[int, ...] = (3, 3, 3, 3, 3)
    dense_growth_div: int = 8
    freeze_experts: bool = True

    def __post_init__(self) -> None:
        names = tuple(self.expert_names)
        legacy_roster = names == LEGACY_EXPERT_NAMES or (
            any(name in {"E7", "E9", "E11", "E12"} for name in names)
            and set(names).issubset(set(LEGACY_EXPERT_NAMES))
        )
        if legacy_roster:
            self.expert_names = tuple(LEGACY_TO_CANONICAL[name] for name in names)
            self.band_policy = {
                layer: tuple(LEGACY_TO_CANONICAL.get(name, name) for name in admitted)
                for layer, admitted in self.band_policy.items()
            }
        else:
            self.expert_names = names
            self.band_policy = {
                layer: tuple(admitted) for layer, admitted in self.band_policy.items()
            }

        unknown = set(self.expert_names) - set(CANONICAL_EXPERT_NAMES)
        if unknown:
            raise ValueError(f"Unknown expert names: {sorted(unknown)}")
        for layer in ("e1", "e2", "e3", "e4", "e5"):
            if layer not in self.band_policy:
                raise ValueError(f"band_policy is missing {layer!r}")
            invalid = set(self.band_policy[layer]) - set(self.expert_names)
            if invalid:
                raise ValueError(f"band_policy[{layer!r}] contains unavailable experts: {sorted(invalid)}")


def _make_norm(channels: int, norm: str) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "gn":
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization: {norm}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(out_ch, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(out_ch, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _make_norm(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _make_norm(channels, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class EncoderBase(nn.Module):
    def __init__(self, cfg: DeftNetConfig):
        super().__init__()
        self.cfg = cfg
        b = cfg.base_channels
        self.filters = [b, b * 2, b * 4, b * 8, b * 16]
        self.pool = nn.MaxPool2d(2, 2)


class EncoderSemantic(EncoderBase):
    """E1: residual semantic CNN encoder."""

    def __init__(self, cfg: DeftNetConfig):
        super().__init__(cfg)
        f = self.filters
        self.c1 = nn.Sequential(
            ConvBlock(cfg.in_channels, f[0], cfg.norm),
            ResidualBlock(f[0], cfg.norm),
            ResidualBlock(f[0], cfg.norm),
        )
        self.c2 = nn.Sequential(
            ConvBlock(f[0], f[1], cfg.norm),
            ResidualBlock(f[1], cfg.norm),
            ResidualBlock(f[1], cfg.norm),
        )
        self.c3 = nn.Sequential(
            ConvBlock(f[1], f[2], cfg.norm),
            ResidualBlock(f[2], cfg.norm),
            ResidualBlock(f[2], cfg.norm),
        )
        self.c4 = nn.Sequential(
            ConvBlock(f[2], f[3], cfg.norm),
            ResidualBlock(f[3], cfg.norm),
            ResidualBlock(f[3], cfg.norm),
        )
        self.c5 = nn.Sequential(
            ConvBlock(f[3], f[4], cfg.norm),
            ResidualBlock(f[4], cfg.norm),
            ResidualBlock(f[4], cfg.norm),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = self.c1(x)
        e2 = self.c2(self.pool(e1))
        e3 = self.c3(self.pool(e2))
        e4 = self.c4(self.pool(e3))
        e5 = self.c5(self.pool(e4))
        return [e1, e2, e3, e4, e5]


class EncoderHRNetLite(EncoderBase):
    """E2: lightweight HRNet-style multi-resolution refinement."""

    def __init__(self, cfg: DeftNetConfig):
        super().__init__(cfg)
        f = self.filters
        self.c1 = ConvBlock(cfg.in_channels, f[0], cfg.norm)
        self.c2 = ConvBlock(f[0], f[1], cfg.norm)
        self.c3 = ConvBlock(f[1], f[2], cfg.norm)
        self.c4 = ConvBlock(f[2], f[3], cfg.norm)
        self.c5 = ConvBlock(f[3], f[4], cfg.norm)
        self.proj = nn.ModuleDict(
            {
                f"{src}->{dst}": nn.Conv2d(f[src - 1], f[dst - 1], 1, bias=False)
                for src in range(2, 6)
                for dst in range(1, src)
            }
        )
        self.refine = nn.ModuleList([ConvBlock(f[i], f[i], cfg.norm) for i in range(4)])

    def _to(self, feat: torch.Tensor, src: int, dst: int, shape: Sequence[int]) -> torch.Tensor:
        y = self.proj[f"{src}->{dst}"](feat)
        return F.interpolate(y, size=shape, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.c1(x)
        f2 = self.c2(self.pool(f1))
        f3 = self.c3(self.pool(f2))
        f4 = self.c4(self.pool(f3))
        f5 = self.c5(self.pool(f4))
        e1 = self.refine[0](f1 + self._to(f2, 2, 1, f1.shape[-2:]) + self._to(f3, 3, 1, f1.shape[-2:]) + self._to(f4, 4, 1, f1.shape[-2:]) + self._to(f5, 5, 1, f1.shape[-2:]))
        e2 = self.refine[1](f2 + self._to(f3, 3, 2, f2.shape[-2:]) + self._to(f4, 4, 2, f2.shape[-2:]) + self._to(f5, 5, 2, f2.shape[-2:]))
        e3 = self.refine[2](f3 + self._to(f4, 4, 3, f3.shape[-2:]) + self._to(f5, 5, 3, f3.shape[-2:]))
        e4 = self.refine[3](f4 + self._to(f5, 5, 4, f4.shape[-2:]))
        return [e1, e2, e3, e4, f5]


class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int, norm: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, growth, 1, bias=False),
            _make_norm(growth, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth, growth, 3, padding=1, bias=False),
            _make_norm(growth, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.net(x)], dim=1)


class DenseStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, growth_div: int, norm: str):
        super().__init__()
        growth = max(4, out_ch // max(1, growth_div))
        cur = in_ch
        blocks = []
        for _ in range(layers):
            blocks.append(DenseLayer(cur, growth, norm))
            cur += growth
        self.dense = nn.Sequential(*blocks)
        self.compress = nn.Sequential(
            nn.Conv2d(cur, out_ch, 1, bias=False),
            _make_norm(out_ch, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(self.dense(x))


class EncoderDenseNet(EncoderBase):
    """E3: DenseNet-style feature-reuse encoder."""

    def __init__(self, cfg: DeftNetConfig):
        super().__init__(cfg)
        f = self.filters
        layers = cfg.dense_layers
        self.s1 = DenseStage(cfg.in_channels, f[0], layers[0], cfg.dense_growth_div, cfg.norm)
        self.s2 = DenseStage(f[0], f[1], layers[1], cfg.dense_growth_div, cfg.norm)
        self.s3 = DenseStage(f[1], f[2], layers[2], cfg.dense_growth_div, cfg.norm)
        self.s4 = DenseStage(f[2], f[3], layers[3], cfg.dense_growth_div, cfg.norm)
        self.s5 = DenseStage(f[3], f[4], layers[4], cfg.dense_growth_div, cfg.norm)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = self.s1(x)
        e2 = self.s2(self.pool(e1))
        e3 = self.s3(self.pool(e2))
        e4 = self.s4(self.pool(e3))
        e5 = self.s5(self.pool(e4))
        return [e1, e2, e3, e4, e5]


def _partition_windows(x: torch.Tensor, window: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        h += pad_h
        w += pad_w
    x = x.view(b, c, h // window, window, w // window, window)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window * window, c)
    return x, (h, w, pad_h, pad_w)


def _unpartition_windows(xw: torch.Tensor, window: int, meta: Tuple[int, int, int, int], batch: int, channels: int) -> torch.Tensor:
    h, w, pad_h, pad_w = meta
    x = xw.view(batch, h // window, w // window, window, window, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch, channels, h, w)
    if pad_h or pad_w:
        x = x[..., : h - pad_h, : w - pad_w]
    return x


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, shift: bool = False, mlp_ratio: float = 4.0):
        super().__init__()
        self.window = int(window)
        self.shift = self.window // 2 if shift and self.window > 1 else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=max(1, heads), batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(2, 3))
        xw, meta = _partition_windows(x, self.window)
        y = self.norm1(xw)
        y, _ = self.attn(y, y, y, need_weights=False)
        xw = xw + y
        xw = xw + self.mlp(self.norm2(xw))
        x = _unpartition_windows(xw, self.window, meta, b, c)
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(2, 3))
        return x


class WindowTransformerStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                WindowAttentionBlock(dim, heads, window, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PyramidAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=max(1, heads), batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        query = self.norm1(tokens)
        attended, _ = self.attn(query, query, query, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


class PyramidTransformerStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, norm: str, mlp_ratio: float = 4.0):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            _make_norm(dim, norm),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[PyramidAttentionBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pre(x))


class EncoderPyramidTransformer(EncoderBase):
    """E4: pyramid transformer encoder with attention only at deep stages."""

    def __init__(self, cfg: DeftNetConfig):
        super().__init__(cfg)
        f = self.filters
        self.c1 = ConvBlock(cfg.in_channels, f[0], cfg.norm)
        self.c2 = ConvBlock(f[0], f[1], cfg.norm)
        self.c3 = ConvBlock(f[1], f[2], cfg.norm)
        self.c4 = ConvBlock(f[2], f[3], cfg.norm)
        self.c5 = ConvBlock(f[3], f[4], cfg.norm)
        self.t = nn.ModuleList(
            [
                nn.Identity()
                if d <= 0
                else PyramidTransformerStage(f[i], d, cfg.e11_heads[i], cfg.norm, 4.0)
                for i, d in enumerate(cfg.e11_depths)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = self.t[0](self.c1(x))
        e2 = self.t[1](self.c2(self.pool(e1)))
        e3 = self.t[2](self.c3(self.pool(e2)))
        e4 = self.t[3](self.c4(self.pool(e3)))
        e5 = self.t[4](self.c5(self.pool(e4)))
        return [e1, e2, e3, e4, e5]


class EncoderSwinLite(EncoderBase):
    """E5: shifted-window hierarchical encoder."""

    def __init__(self, cfg: DeftNetConfig):
        super().__init__(cfg)
        f = self.filters
        self.c1 = ConvBlock(cfg.in_channels, f[0], cfg.norm)
        self.c2 = ConvBlock(f[0], f[1], cfg.norm)
        self.c3 = ConvBlock(f[1], f[2], cfg.norm)
        self.c4 = ConvBlock(f[2], f[3], cfg.norm)
        self.c5 = ConvBlock(f[3], f[4], cfg.norm)
        self.t = nn.ModuleList(
            [
                nn.Identity()
                if d <= 0
                else WindowTransformerStage(f[i], d, cfg.swin_heads[i], cfg.swin_window, 4.0)
                for i, d in enumerate(cfg.swin_depths)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = self.t[0](self.c1(x))
        e2 = self.t[1](self.c2(self.pool(e1)))
        e3 = self.t[2](self.c3(self.pool(e2)))
        e4 = self.t[3](self.c4(self.pool(e3)))
        e5 = self.t[4](self.c5(self.pool(e4)))
        return [e1, e2, e3, e4, e5]


def build_encoder(name: str, cfg: DeftNetConfig) -> EncoderBase:
    if name == "E1":
        return EncoderSemantic(cfg)
    if name == "E2":
        return EncoderHRNetLite(cfg)
    if name == "E3":
        return EncoderDenseNet(cfg)
    if name == "E4":
        return EncoderPyramidTransformer(cfg)
    if name == "E5":
        return EncoderSwinLite(cfg)
    raise ValueError(f"Unknown expert name: {name}")


class FeatureAdapter(nn.Module):
    def __init__(self, channels: int, cfg: DeftNetConfig):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(channels, channels, 1, bias=False),
            _make_norm(channels, cfg.norm),
            nn.ReLU(inplace=True),
        ]
        if cfg.adapter_dropout > 0:
            layers.append(nn.Dropout2d(cfg.adapter_dropout))
        self.block = nn.Sequential(*layers)
        self.residual = bool(cfg.adapter_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.residual else y


class HSAF(nn.Module):
    """Hierarchical spatial adaptive fusion at one feature scale."""

    def __init__(self, channels: int, expert_names: Sequence[str], layer_name: str, cfg: DeftNetConfig):
        super().__init__()
        self.expert_names = tuple(expert_names)
        self.layer_name = layer_name
        self.cfg = cfg
        self.compress = nn.Sequential(
            nn.Conv2d(channels * len(expert_names), channels, 1, bias=False),
            _make_norm(channels, cfg.norm),
            nn.ReLU(inplace=True),
        )
        self.weight_head = nn.Conv2d(channels, len(expert_names), 1, bias=True)
        self.dropout = nn.Dropout2d(cfg.hsaf_gate_dropout) if cfg.hsaf_gate_dropout > 0 else nn.Identity()
        self.last_weights: torch.Tensor | None = None

    def _allow_mask(self, logits: torch.Tensor) -> torch.Tensor:
        allowed = set(self.cfg.band_policy.get(self.layer_name, self.expert_names))
        vals = [name in allowed for name in self.expert_names]
        if not any(vals):
            vals = [True] * len(self.expert_names)
        mask = torch.tensor(vals, device=logits.device, dtype=torch.bool)
        return mask.view(1, len(vals), 1, 1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(list(feats), dim=1)
        logits = self.weight_head(self.dropout(self.compress(x)))
        logits = logits.masked_fill(~self._allow_mask(logits), -10000.0)
        if self.cfg.hsaf_temperature != 1.0:
            logits = logits / float(self.cfg.hsaf_temperature)
        weights = torch.softmax(logits, dim=1)
        self.last_weights = weights.detach()
        fused = torch.zeros_like(feats[0])
        for idx, feat in enumerate(feats):
            fused = fused + weights[:, idx : idx + 1] * feat
        return fused


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(out_ch, norm),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(out_ch * 2, out_ch, norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class Decoder(nn.Module):
    def __init__(self, filters: Sequence[int], cfg: DeftNetConfig):
        super().__init__()
        f = list(filters)
        self.use_deep_supervision = cfg.use_deep_supervision
        self.up4 = UpBlock(f[4], f[3], cfg.norm)
        self.up3 = UpBlock(f[3], f[2], cfg.norm)
        self.up2 = UpBlock(f[2], f[1], cfg.norm)
        self.up1 = UpBlock(f[1], f[0], cfg.norm)
        self.head1 = nn.Conv2d(f[0], cfg.out_channels, 1)
        if self.use_deep_supervision:
            self.head2 = nn.Conv2d(f[1], cfg.out_channels, 1)
            self.head3 = nn.Conv2d(f[2], cfg.out_channels, 1)
            self.head4 = nn.Conv2d(f[3], cfg.out_channels, 1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor | List[torch.Tensor]:
        e1, e2, e3, e4, e5 = feats
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        if not self.use_deep_supervision:
            return self.head1(d1)
        return [self.head1(d1), self.head2(d2), self.head3(d3), self.head4(d4)]


class ExpertSegmentor(nn.Module):
    """Phase-I U-Net segmentor used to specialize one expert encoder."""

    def __init__(self, expert_name: str, cfg: DeftNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or DeftNetConfig()
        if expert_name not in CANONICAL_EXPERT_NAMES:
            raise ValueError(f"Unknown Phase-I expert {expert_name!r}")
        self.expert_name = expert_name
        self.encoder = build_encoder(expert_name, self.cfg)
        self.decoder = Decoder(self.encoder.filters, self.cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        return self.decoder(self.encoder(x))


class DeftNet(nn.Module):
    def __init__(self, cfg: DeftNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or DeftNetConfig()
        self.encoders = nn.ModuleDict({name: build_encoder(name, self.cfg) for name in self.cfg.expert_names})
        filters = next(iter(self.encoders.values())).filters
        self.expert_names = tuple(self.cfg.expert_names)
        self.hsaf = nn.ModuleDict(
            {
                "e1": HSAF(filters[0], self.expert_names, "e1", self.cfg),
                "e2": HSAF(filters[1], self.expert_names, "e2", self.cfg),
                "e3": HSAF(filters[2], self.expert_names, "e3", self.cfg),
                "e4": HSAF(filters[3], self.expert_names, "e4", self.cfg),
                "e5": HSAF(filters[4], self.expert_names, "e5", self.cfg),
            }
        )
        self.use_adapters = bool(self.cfg.use_feature_adapters)
        if self.use_adapters:
            self.adapters = nn.ModuleDict(
                {
                    f"{name}_e{idx + 1}": FeatureAdapter(filters[idx], self.cfg)
                    for name in self.expert_names
                    for idx in range(5)
                }
            )
        self.decoder = Decoder(filters, self.cfg)
        if self.cfg.freeze_experts:
            self.freeze_experts()

    def freeze_experts(self) -> None:
        for encoder in self.encoders.values():
            for param in encoder.parameters():
                param.requires_grad_(False)

    def unfreeze_experts(self) -> None:
        for encoder in self.encoders.values():
            for param in encoder.parameters():
                param.requires_grad_(True)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_expert_checkpoints(
        self,
        checkpoints: Mapping[str, str | Path],
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> Dict[str, str]:
        """Load Phase-I encoder weights and return the resolved checkpoint paths.

        Accepted files may contain an ``encoder_state_dict``, a Phase-I
        ``model``/``state_dict`` with ``encoder.`` keys, or an older full
        DEFT-Net checkpoint with ``encoders.<expert>.`` keys.
        """

        normalized = _normalize_expert_checkpoint_mapping(checkpoints)
        missing = [name for name in self.expert_names if name not in normalized]
        if missing:
            raise ValueError(f"Missing Phase-I checkpoints for: {', '.join(missing)}")

        resolved: Dict[str, str] = {}
        for name in self.expert_names:
            path = Path(normalized[name]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Phase-I checkpoint for {name} not found: {path}")
            checkpoint = torch.load(path, map_location=map_location)
            state = _extract_encoder_state(checkpoint, name)
            incompatible = self.encoders[name].load_state_dict(state, strict=strict)
            if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
                raise RuntimeError(
                    f"Incompatible Phase-I checkpoint for {name}: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
            resolved[name] = str(path)
        self.freeze_experts()
        return resolved

    def _adapt(self, name: str, level: int, feat: torch.Tensor) -> torch.Tensor:
        if not self.use_adapters:
            return feat
        return self.adapters[f"{name}_e{level}"](feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        feats_by_expert: MutableMapping[str, List[torch.Tensor]] = {}
        for name, encoder in self.encoders.items():
            feats = encoder(x)
            feats_by_expert[name] = [self._adapt(name, idx + 1, feat) for idx, feat in enumerate(feats)]

        fused: List[torch.Tensor] = []
        for level_idx, layer in enumerate(("e1", "e2", "e3", "e4", "e5")):
            level_feats = [feats_by_expert[name][level_idx] for name in self.expert_names]
            if self.cfg.fusion_mode == "mean":
                allowed = set(self.cfg.band_policy.get(layer, self.expert_names))
                selected = [feat for name, feat in zip(self.expert_names, level_feats) if name in allowed]
                if not selected:
                    selected = level_feats
                fused.append(torch.stack(selected, dim=0).mean(dim=0))
            elif self.cfg.fusion_mode == "hsaf":
                fused.append(self.hsaf[layer](level_feats))
            else:
                raise ValueError(f"Unknown fusion mode: {self.cfg.fusion_mode}")
        return self.decoder(fused)


def build_deftnet(**kwargs) -> DeftNet:
    cfg = DeftNetConfig(**kwargs)
    return DeftNet(cfg)


def _normalize_expert_checkpoint_mapping(
    checkpoints: Mapping[str, str | Path],
) -> Dict[str, str | Path]:
    keys = tuple(checkpoints)
    legacy = any(key in {"E7", "E9", "E11", "E12"} for key in keys)
    if legacy:
        return {LEGACY_TO_CANONICAL.get(key, key): value for key, value in checkpoints.items()}
    return dict(checkpoints)


def _unwrap_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping of parameter tensors")
    for key in ("encoder_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _extract_encoder_state(checkpoint: object, expert_name: str) -> Dict[str, torch.Tensor]:
    state = _unwrap_state_dict(checkpoint)
    legacy_name = CANONICAL_TO_LEGACY[expert_name]
    prefixes = (
        f"encoders.{expert_name}.",
        f"encoders.{legacy_name}.",
        "encoder.",
        "module.encoder.",
    )
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
        }
        if selected:
            return selected
    direct = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not direct:
        raise ValueError(f"No encoder tensors found for {expert_name}")
    return direct


def remap_legacy_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Translate v0.1.x expert identifiers without colliding on legacy E5."""

    legacy_detected = any(
        isinstance(key, str) and ("encoders.E7." in key or "adapters.E7_" in key)
        for key in state
    )
    if not legacy_detected:
        return dict(state)

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for index, legacy_name in enumerate(LEGACY_EXPERT_NAMES):
            token = f"__DEFT_LEGACY_{index}__"
            new_key = new_key.replace(f"encoders.{legacy_name}.", f"encoders.{token}.")
            new_key = new_key.replace(f"adapters.{legacy_name}_", f"adapters.{token}_")
        for index, canonical_name in enumerate(CANONICAL_EXPERT_NAMES):
            token = f"__DEFT_LEGACY_{index}__"
            new_key = new_key.replace(f"encoders.{token}.", f"encoders.{canonical_name}.")
            new_key = new_key.replace(f"adapters.{token}_", f"adapters.{canonical_name}_")
        remapped[new_key] = value
    return remapped


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    return model.load_state_dict(remap_legacy_state_dict(state), strict=strict)
