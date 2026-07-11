from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def assign_perspective_folds(
    sample_ids: list[str], n_folds: int = 5, seed: int = 2026
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    shuffled = sorted(sample_ids)
    random.Random(seed).shuffle(shuffled)
    return {sample_id: index % n_folds for index, sample_id in enumerate(shuffled)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_grad_scaler(enabled: bool):
    active = bool(enabled and torch.cuda.is_available())
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=active)
    return torch.cuda.amp.GradScaler(enabled=active)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def weighted_metric_mean(records: Iterable[tuple[int, dict[str, float]]]) -> dict[str, float]:
    items = list(records)
    total = sum(weight for weight, _ in items)
    if total <= 0:
        raise ValueError("No validation samples were evaluated")
    keys = items[0][1]
    return {
        key: float(sum(weight * metrics[key] for weight, metrics in items) / total)
        for key in keys
    }


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
