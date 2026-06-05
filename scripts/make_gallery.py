from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def load01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def overlay_errors(image: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    base = np.stack([image, image, image], axis=-1)
    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target
    out = base.copy()
    out[tp] = 0.35 * out[tp] + 0.65 * np.array([0.0, 0.85, 0.1])
    out[fp] = 0.35 * out[fp] + 0.65 * np.array([1.0, 0.0, 0.0])
    out[fn] = 0.35 * out[fn] + 0.65 * np.array([0.0, 0.25, 1.0])
    return np.clip(out, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a qualitative TP/FP/FN gallery from images, masks, and predictions.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    images = collect_by_stem(Path(args.image_dir))
    masks = collect_by_stem(Path(args.mask_dir))
    preds = collect_by_stem(Path(args.pred_dir))
    sample_ids = sorted(set(images) & set(masks) & set(preds))[: args.max_samples]
    if not sample_ids:
        raise SystemExit("No matching image/mask/prediction filenames by stem.")

    rows = int(np.ceil(len(sample_ids) / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(3.0 * args.cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sample_id in zip(axes.ravel(), sample_ids):
        image = load01(images[sample_id])
        target = load01(masks[sample_id]) >= 0.5
        pred = load01(preds[sample_id]) >= args.threshold
        ax.imshow(overlay_errors(image, pred, target))
        ax.set_title(sample_id, fontsize=8)
        ax.axis("off")
    fig.suptitle("TP green / FP red / FN blue", fontsize=10)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(out)


if __name__ == "__main__":
    main()
