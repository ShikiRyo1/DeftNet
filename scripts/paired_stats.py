from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


DEFAULT_METRICS = ["dice", "iou", "sen", "pre", "cldice", "hd95", "assd"]


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def paired_rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def bootstrap_ci(diff: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diff)
    means = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(float(diff[idx].mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm_adjust(pairs: list[tuple[str, str, float]]) -> dict[tuple[str, str], float]:
    valid = [(metric, baseline, p) for metric, baseline, p in pairs if not math.isnan(p)]
    valid.sort(key=lambda x: x[2])
    m = len(valid)
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    for i, (metric, baseline, p) in enumerate(valid):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[(metric, baseline)] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired statistics from long-form per-image metrics CSV files.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more CSVs with model/sample_id metric columns.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default="DeftNet")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--model-column", default="model")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    table: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in read_rows(args.input):
        model = row[args.model_column]
        sample_id = row[args.id_column]
        table[model][sample_id] = {m: float(row[m]) for m in args.metrics if row.get(m, "") not in ("", "nan")}

    if args.reference not in table:
        raise SystemExit(f"Reference model not found: {args.reference}")

    baselines = sorted(m for m in table if m != args.reference)
    results: dict[str, object] = {"reference": args.reference, "bootstrap": args.bootstrap, "comparisons": {}}
    p_values: list[tuple[str, str, float]] = []

    for baseline in baselines:
        common_ids = sorted(set(table[args.reference]) & set(table[baseline]))
        baseline_result: dict[str, object] = {"n_paired": len(common_ids), "metrics": {}}
        for metric in args.metrics:
            paired = [
                (table[args.reference][sid].get(metric), table[baseline][sid].get(metric))
                for sid in common_ids
                if metric in table[args.reference][sid] and metric in table[baseline][sid]
            ]
            if not paired:
                continue
            ref = np.asarray([p[0] for p in paired], dtype=np.float64)
            base = np.asarray([p[1] for p in paired], dtype=np.float64)
            diff = ref - base
            try:
                stat, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="zsplit")
                stat_value = float(stat)
                p_value = float(p)
            except ValueError:
                stat_value = float("nan")
                p_value = float("nan")
            ci_low, ci_high = bootstrap_ci(diff, args.bootstrap, args.seed)
            p_values.append((metric, baseline, p_value))
            baseline_result["metrics"][metric] = {
                "reference_mean": float(ref.mean()),
                "baseline_mean": float(base.mean()),
                "mean_diff": float(diff.mean()),
                "bootstrap95_mean_diff": [ci_low, ci_high],
                "wilcoxon_stat": stat_value,
                "p_value": p_value,
                "rank_biserial": paired_rank_biserial(diff),
            }
        results["comparisons"][baseline] = baseline_result

    adjusted = holm_adjust(p_values)
    for baseline, baseline_result in results["comparisons"].items():
        for metric, metric_result in baseline_result["metrics"].items():
            metric_result["p_holm_all_tests"] = adjusted.get((metric, baseline), float("nan"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "baselines": baselines}, indent=2))


if __name__ == "__main__":
    main()
