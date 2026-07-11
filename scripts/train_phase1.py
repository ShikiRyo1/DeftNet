from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from deftnet.data import VesselSegmentationDataset
from deftnet.losses import CombinedSegLoss
from deftnet.metrics import evaluate_binary_batch
from deftnet.models import CANONICAL_EXPERT_NAMES, DeftNetConfig, ExpertSegmentor
from deftnet.training import git_commit, make_grad_scaler, seed_everything, weighted_metric_mean, write_json


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_assignments(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    assignments = {str(row["sample_id"]): int(row["perspective_fold"]) for row in records}
    if not assignments:
        raise ValueError("Perspective manifest contains no records")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-I specialization of one DEFT-Net expert.")
    parser.add_argument("--config", default="configs/deftnet_cmig.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--perspective-manifest", required=True)
    parser.add_argument("--expert", required=True, choices=CANONICAL_EXPERT_NAMES)
    parser.add_argument("--omit-fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/phase1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg_dict = load_yaml(args.config)
    train_cfg = cfg_dict.get("phase1", cfg_dict.get("train", {}))
    model_cfg = DeftNetConfig(**cfg_dict.get("model", {}))
    omitted_fold = args.omit_fold if args.omit_fold is not None else CANONICAL_EXPERT_NAMES.index(args.expert)
    n_folds = int(cfg_dict.get("protocol", {}).get("perspective_folds", 5))
    if not 0 <= omitted_fold < n_folds:
        raise ValueError(f"omit-fold must be in [0, {n_folds - 1}]")

    seed_everything(args.seed)
    device = torch.device(args.device)
    image_size = int(train_cfg.get("image_size", cfg_dict.get("train", {}).get("image_size", 512)))
    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 60))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    full_train = VesselSegmentationDataset.from_root(
        args.data_root, "train", image_size=image_size, augment=True
    )
    assignments = load_assignments(args.perspective_manifest)
    missing = sorted(set(full_train.sample_ids) - set(assignments))
    if missing:
        raise ValueError(f"Perspective manifest is missing {len(missing)} training samples")
    train_indices = [
        index
        for index, sample_id in enumerate(full_train.sample_ids)
        if assignments[sample_id] != omitted_fold
    ]
    if not train_indices:
        raise ValueError("Complementary Phase-I training subset is empty")
    train_ds = Subset(full_train, train_indices)
    val_ds = VesselSegmentationDataset.from_root(
        args.data_root, "val", image_size=image_size, augment=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ExpertSegmentor(args.expert, model_cfg).to(device)
    loss_fn = CombinedSegLoss(**train_cfg.get("loss", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("scheduler_t_max", epochs)),
        eta_min=float(train_cfg.get("scheduler_eta_min", 1e-6)),
    )
    scaler = make_grad_scaler(amp_enabled)
    clip_norm = float(train_cfg.get("gradient_clip_norm", 5.0))

    out_dir = Path(args.output_dir) / f"{args.expert}_omit_fold{omitted_fold}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "phase": "I",
        "expert": args.expert,
        "omitted_perspective_fold": omitted_fold,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "config": str(Path(args.config).resolve()),
        "perspective_manifest": str(Path(args.perspective_manifest).resolve()),
        "git_commit": git_commit(),
    }
    write_json(out_dir / "run_manifest.json", run_manifest)

    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Phase I {args.expert} {epoch}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(x)
                loss, _ = loss_fn(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * x.size(0)

        model.eval()
        validation = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                validation.append((x.size(0), evaluate_binary_batch(logits, y, threshold=0.5)))
        metrics = weighted_metric_mean(validation)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_ds),
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {
                    "phase": "I",
                    "expert": args.expert,
                    "omitted_perspective_fold": omitted_fold,
                    "seed": args.seed,
                    "model": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "config": cfg_dict,
                    "epoch": epoch,
                    "metrics": metrics,
                    "git_commit": git_commit(),
                },
                out_dir / "best.pth",
            )
        scheduler.step()
        write_json(out_dir / "history.json", history)


if __name__ == "__main__":
    main()
