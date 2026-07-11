from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from deftnet.metrics import (
    betti0_error,
    betti1_error,
    branch_skeleton_recovery,
    cbdice_score,
    centerline_break_count,
    centerline_continuity,
    cldice_score,
    component_count_deviation,
    component_count_ratio,
    hd95_assd,
    metrics_from_confusion,
    skeleton_false_negative_rate,
    thin_structure_recall,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def collect_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def confusion_np(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    return {
        "tp": int(np.logical_and(pred_b, target_b).sum()),
        "tn": int(np.logical_and(~pred_b, ~target_b).sum()),
        "fp": int(np.logical_and(pred_b, ~target_b).sum()),
        "fn": int(np.logical_and(~pred_b, target_b).sum()),
    }


def safe_mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fixed-threshold per-image segmentation metrics.")
    parser.add_argument("--pred-dir", required=True, help="Directory of predicted masks or probability maps.")
    parser.add_argument("--mask-dir", required=True, help="Directory of binary ground-truth masks.")
    parser.add_argument("--output", required=True, help="Output per-image CSV.")
    parser.add_argument("--model", default="DEFT-Net")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    preds = collect_by_stem(Path(args.pred_dir))
    masks = collect_by_stem(Path(args.mask_dir))
    sample_ids = sorted(set(preds) & set(masks))
    if not sample_ids:
        raise SystemExit("No matching prediction/mask filenames by stem.")

    rows: list[dict[str, object]] = []
    metric_names = [
        "dice",
        "iou",
        "sen",
        "pre",
        "spec",
        "mcc",
        "cldice",
        "cbdice",
        "hd95",
        "assd",
        "branch_recovery",
        "thin_recall",
        "continuity",
        "break_count",
        "betti0_error",
        "betti1_error",
        "cc_ratio",
        "cc_deviation",
        "skel_fnr",
    ]

    for sample_id in sample_ids:
        pred_prob = load01(preds[sample_id])
        target = load01(masks[sample_id]) >= 0.5
        pred = pred_prob >= args.threshold
        c = confusion_np(pred, target)
        metrics = metrics_from_confusion(**c)
        metrics["cldice"] = cldice_score(pred.astype(np.float32), target.astype(np.float32))
        metrics["cbdice"] = cbdice_score(pred, target)
        hd95, assd = hd95_assd(pred.astype(np.float32), target.astype(np.float32))
        metrics["hd95"] = hd95
        metrics["assd"] = assd
        metrics["branch_recovery"] = branch_skeleton_recovery(pred, target)
        metrics["thin_recall"] = thin_structure_recall(pred, target)
        metrics["continuity"] = centerline_continuity(pred, target)
        metrics["break_count"] = centerline_break_count(pred, target)
        metrics["betti0_error"] = betti0_error(pred, target)
        metrics["betti1_error"] = betti1_error(pred, target)
        metrics["cc_ratio"] = component_count_ratio(pred, target)
        metrics["cc_deviation"] = component_count_deviation(pred, target)
        metrics["skel_fnr"] = skeleton_false_negative_rate(pred, target)
        rows.append(
            {
                "model": args.model,
                "split": args.split,
                "sample_id": sample_id,
                "threshold": args.threshold,
                **{k: metrics[k] for k in metric_names},
                **c,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "split", "sample_id", "threshold", *metric_names, "tp", "tn", "fp", "fn"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "split": args.split,
        "n": len(rows),
        "threshold": args.threshold,
        "means": {name: safe_mean([float(r[name]) for r in rows]) for name in metric_names},
        "output": str(out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
