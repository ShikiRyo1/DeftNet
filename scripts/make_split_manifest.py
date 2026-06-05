from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a checksum split manifest from a DeftNet data_root.")
    parser.add_argument("--data-root", required=True, help="Root with train_images/train_masks/etc.")
    parser.add_argument("--output", required=True, help="Output CSV manifest path.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "true_test"])
    parser.add_argument("--strict", action="store_true", help="Fail if any requested split folder is missing.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for split in args.splits:
        image_dir = data_root / f"{split}_images"
        mask_dir = data_root / f"{split}_masks"
        if not image_dir.exists() or not mask_dir.exists():
            missing.append(split)
            continue
        images = collect_by_stem(image_dir)
        masks = collect_by_stem(mask_dir)
        common = sorted(set(images) & set(masks))
        for sample_id in common:
            image = images[sample_id]
            mask = masks[sample_id]
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "image_relpath": image.relative_to(data_root).as_posix(),
                    "mask_relpath": mask.relative_to(data_root).as_posix(),
                    "image_sha256": sha256_file(image),
                    "mask_sha256": sha256_file(mask),
                }
            )

    if args.strict and missing:
        raise SystemExit(f"Missing requested split folders: {', '.join(missing)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample_id", "image_relpath", "mask_relpath", "image_sha256", "mask_sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows": len(rows), "missing_splits": missing, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
