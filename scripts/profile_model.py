from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from deftnet.models import DeftNet, DeftNetConfig


def conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_ch, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_ch = module.in_channels // module.groups
    return int(batch * out_ch * out_h * out_w * in_ch * kernel_h * kernel_w)


def linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def estimate_macs(model: nn.Module, sample: torch.Tensor) -> int:
    total = 0
    hooks = []

    def add_macs(module: nn.Module, _inputs, output):
        nonlocal total
        if isinstance(output, (list, tuple)):
            return
        if isinstance(module, nn.Conv2d):
            total += conv2d_macs(module, output)
        elif isinstance(module, nn.Linear):
            total += linear_macs(module, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(add_macs))
    with torch.no_grad():
        model(sample)
    for hook in hooks:
        hook.remove()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DEFT-Net parameters, approximate MACs, latency, and CUDA VRAM.")
    parser.add_argument("--config", default="configs/deftnet_cmig.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    image_size = args.image_size or int(cfg_dict.get("train", {}).get("image_size", 512))
    device = torch.device(args.device)
    model = DeftNet(DeftNetConfig(**cfg_dict.get("model", {}))).to(device).eval()
    sample = torch.zeros(args.batch_size, cfg_dict.get("model", {}).get("in_channels", 1), image_size, image_size, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = estimate_macs(model, sample)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    result = {
        "config": args.config,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs_conv_linear_only": macs,
        "gmacs_conv_linear_only": macs / 1e9,
        "latency_ms_mean": (elapsed / args.repeats) * 1000.0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cuda_peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "notes": "MACs count Conv2d and Linear modules only; use the same script/hardware for all compared models.",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
