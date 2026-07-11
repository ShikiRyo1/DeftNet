from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from deftnet.data import VesselSegmentationDataset
from deftnet.training import assign_perspective_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase-I complementary-view fold assignments.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Output JSON manifest.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dataset = VesselSegmentationDataset.from_root(
        args.data_root, "train", image_size=args.image_size, augment=False
    )
    assignments = assign_perspective_folds(dataset.sample_ids, args.folds, args.seed)
    fold_sizes = {
        str(fold): sum(value == fold for value in assignments.values())
        for fold in range(args.folds)
    }
    payload = {
        "schema_version": 1,
        "description": "Each Phase-I expert omits one fold and trains on the complementary samples.",
        "seed": args.seed,
        "n_folds": args.folds,
        "n_samples": len(assignments),
        "fold_sizes": fold_sizes,
        "records": [
            {"sample_id": sample_id, "perspective_fold": fold}
            for sample_id, fold in sorted(assignments.items())
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "perspective_fold"])
        writer.writeheader()
        writer.writerows(payload["records"])
    print(json.dumps({"json": str(out), "csv": str(csv_path), "fold_sizes": fold_sizes}, indent=2))


if __name__ == "__main__":
    main()
