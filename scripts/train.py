from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from deftnet.data import VesselSegmentationDataset
from deftnet.losses import CombinedSegLoss
from deftnet.models import DeftNet, DeftNetConfig
from deftnet.metrics import evaluate_binary_batch


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeftNet/DEFT-Net fusion head.")
    parser.add_argument("--config", default="configs/dca_five_expert.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="runs/deftnet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    train_cfg = cfg_dict.get("train", {})
    model_cfg = DeftNetConfig(**cfg_dict.get("model", {}))
    epochs = args.epochs or int(train_cfg.get("epochs", 60))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 4))
    lr = args.lr or float(train_cfg.get("lr", 1e-4))
    image_size = int(train_cfg.get("image_size", 512))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = VesselSegmentationDataset.from_root(args.data_root, "train", image_size=image_size, augment=True)
    val_ds = VesselSegmentationDataset.from_root(args.data_root, "val", image_size=image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeftNet(model_cfg).to(device)
    loss_fn = CombinedSegLoss(**train_cfg.get("loss", {}))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-5)))

    best_dice = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss, _ = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * x.size(0)

        model.eval()
        metric_accum = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                metric_accum.append(evaluate_binary_batch(logits, y, threshold=0.5))
        mean_metrics = {k: sum(m[k] for m in metric_accum) / len(metric_accum) for k in metric_accum[0]}
        record = {"epoch": epoch, "train_loss": total_loss / len(train_ds), **mean_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        if mean_metrics["dice"] > best_dice:
            best_dice = mean_metrics["dice"]
            torch.save({"model": model.state_dict(), "config": cfg_dict, "epoch": epoch, "metrics": mean_metrics}, out_dir / "best.pth")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
