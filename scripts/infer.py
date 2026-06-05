from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from deftnet.models import DeftNet, DeftNetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run binary vessel segmentation on one image.")
    parser.add_argument("--config", default="configs/dca_five_expert.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    image_size = int(cfg_dict.get("train", {}).get("image_size", 512))
    model = DeftNet(DeftNetConfig(**cfg_dict.get("model", {}))).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt.get("model", ckpt.get("state_dict", ckpt)), strict=False)
    model.eval()

    tf = T.Compose([T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR), T.ToTensor()])
    image = Image.open(args.image).convert("L")
    x = tf(image).unsqueeze(0).to(args.device)
    with torch.no_grad():
        outputs = model(x)
        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        mask = (torch.sigmoid(logits)[0, 0].cpu() >= args.threshold).byte().numpy() * 255
    out = Image.fromarray(mask).resize(image.size, resample=Image.Resampling.NEAREST)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)


if __name__ == "__main__":
    main()
