from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from deftnet.data import VesselSegmentationDataset
from deftnet.metrics import evaluate_binary_batch
from deftnet.models import DeftNet, DeftNetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeftNet/DEFT-Net at a fixed operating point.")
    parser.add_argument("--config", default="configs/dca_five_expert.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "true_test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    model = DeftNet(DeftNetConfig(**cfg_dict.get("model", {}))).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt.get("model", ckpt.get("state_dict", ckpt)), strict=False)
    model.eval()

    image_size = int(cfg_dict.get("train", {}).get("image_size", 512))
    ds = VesselSegmentationDataset.from_root(args.data_root, args.split, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_metrics = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            all_metrics.append(evaluate_binary_batch(logits, y, threshold=args.threshold))
    summary = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
