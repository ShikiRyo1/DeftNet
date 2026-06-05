from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from deftnet.models import DeftNet, DeftNetConfig


def make_synthetic_vessel(size: int = 128) -> tuple[Image.Image, Image.Image]:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    points = [(8, 78), (30, 62), (52, 66), (78, 48), (110, 42), (124, 35)]
    draw.line(points, fill=255, width=6, joint="curve")
    draw.line([(56, 64), (72, 82), (96, 100)], fill=255, width=4)
    draw.line([(78, 49), (95, 65), (118, 70)], fill=255, width=3)
    image = mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    noise = np.random.default_rng(2026).normal(18, 8, (size, size)).astype(np.float32)
    arr = np.clip(np.asarray(image, dtype=np.float32) * 0.65 + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), mask


def main() -> None:
    out_dir = Path("examples_output")
    out_dir.mkdir(exist_ok=True)
    image, mask = make_synthetic_vessel()
    image.save(out_dir / "synthetic_image.png")
    mask.save(out_dir / "synthetic_mask.png")

    cfg = DeftNetConfig(base_channels=4, use_feature_adapters=False, hsaf_gate_dropout=0.0)
    model = DeftNet(cfg).eval()
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).view(1, 1, image.height, image.width)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].numpy()
    Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).save(out_dir / "untrained_probability.png")
    print(f"Wrote synthetic demo outputs to {out_dir.resolve()}")
    print("The probability map is from an untrained tiny model and is only an API smoke demo.")


if __name__ == "__main__":
    main()
