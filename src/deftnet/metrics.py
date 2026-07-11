from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from scipy import ndimage as ndi


def binary_confusion(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.bool()
    target = target.bool()
    tp = torch.logical_and(pred, target).sum().item()
    tn = torch.logical_and(~pred, ~target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_confusion(tp: float, tn: float, fp: float, fn: float, eps: float = 1e-7) -> Dict[str, float]:
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    sen = (tp + eps) / (tp + fn + eps)
    pre = (tp + eps) / (tp + fp + eps)
    spec = (tn + eps) / (tn + fp + eps)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), eps))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return {"dice": dice, "iou": iou, "sen": sen, "pre": pre, "spec": spec, "mcc": mcc}


def hard_skeletonize(mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask > 0.5).astype(np.float32)
    except Exception:
        cur = (mask > 0.5).astype(np.float32)
        out = np.zeros_like(cur)
        struct = ndi.generate_binary_structure(2, 1)
        for _ in range(32):
            eroded = ndi.binary_erosion(cur, structure=struct).astype(np.float32)
            opened = ndi.binary_dilation(eroded, structure=struct).astype(np.float32)
            out = np.clip(out + np.clip(cur - opened, 0, 1), 0, 1)
            cur = eroded
            if cur.sum() < 1:
                break
        return out


def cldice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    sk_pred = hard_skeletonize(pred)
    sk_target = hard_skeletonize(target)
    tprec = (sk_pred * target).sum() / (sk_pred.sum() + eps)
    tsens = (sk_target * pred).sum() / (sk_target.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def _square(radius: int) -> np.ndarray:
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def cbdice_score(pred: np.ndarray, target: np.ndarray, radius: int = 2, eps: float = 1e-7) -> float:
    """Hard centerline-boundary Dice used by the fixed structural evaluator."""

    pred = pred.astype(bool)
    target = target.astype(bool)
    sk_pred = hard_skeletonize(pred).astype(bool)
    sk_target = hard_skeletonize(target).astype(bool)
    if not sk_pred.any() and not sk_target.any():
        return 1.0
    if not sk_pred.any() or not sk_target.any():
        return 0.0
    target_band = ndi.binary_dilation(sk_target, structure=_square(radius))
    pred_band = ndi.binary_dilation(sk_pred, structure=_square(radius))
    precision = (sk_pred & target_band).sum() / max(int(sk_pred.sum()), 1)
    recall = (sk_target & pred_band).sum() / max(int(sk_target.sum()), 1)
    return float(2 * precision * recall / (precision + recall + eps))


def centerline_continuity(pred: np.ndarray, target: np.ndarray, radius: int = 2) -> float:
    sk_target = hard_skeletonize(target).astype(bool)
    if not sk_target.any():
        return 1.0
    covered = ndi.binary_dilation(pred.astype(bool), structure=_square(radius))
    return float((sk_target & covered).sum() / max(int(sk_target.sum()), 1))


def centerline_break_count(pred: np.ndarray, target: np.ndarray, radius: int = 2) -> float:
    sk_target = hard_skeletonize(target).astype(bool)
    if not sk_target.any():
        return 0.0
    covered = ndi.binary_dilation(pred.astype(bool), structure=_square(radius))
    missing = sk_target & ~covered
    return float(ndi.label(missing, structure=np.ones((3, 3), dtype=bool))[1])


def branch_skeleton_recovery(pred: np.ndarray, target: np.ndarray, radius: int = 2) -> float:
    sk_target = hard_skeletonize(target).astype(bool)
    if not sk_target.any():
        return 1.0
    degree = ndi.convolve(sk_target.astype(np.int8), np.ones((3, 3), dtype=np.int8))
    degree = degree - sk_target.astype(np.int8)
    branches = sk_target & (degree >= 3)
    if not branches.any():
        branches = sk_target
    covered = ndi.binary_dilation(pred.astype(bool), structure=_square(radius))
    return float((branches & covered).sum() / max(int(branches.sum()), 1))


def thin_structure_recall(
    pred: np.ndarray,
    target: np.ndarray,
    width_percentile: float = 35.0,
    radius: int = 1,
) -> float:
    target = target.astype(bool)
    sk_target = hard_skeletonize(target).astype(bool)
    if not sk_target.any():
        return 1.0
    diameter = 2.0 * ndi.distance_transform_edt(target)
    threshold = float(np.percentile(diameter[sk_target], width_percentile))
    thin = sk_target & (diameter <= threshold)
    if not thin.any():
        thin = sk_target
    covered = ndi.binary_dilation(pred.astype(bool), structure=_square(radius))
    return float((thin & covered).sum() / max(int(thin.sum()), 1))


def betti0_error(pred: np.ndarray, target: np.ndarray) -> float:
    structure = np.ones((3, 3), dtype=bool)
    pred_count = ndi.label(pred.astype(bool), structure=structure)[1]
    target_count = ndi.label(target.astype(bool), structure=structure)[1]
    return float(abs(int(pred_count) - int(target_count)))


def component_count_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    structure = np.ones((3, 3), dtype=bool)
    pred_count = ndi.label(pred.astype(bool), structure=structure)[1]
    target_count = ndi.label(target.astype(bool), structure=structure)[1]
    if target_count == 0:
        return 1.0 if pred_count == 0 else float(pred_count)
    return float(pred_count / target_count)


def component_count_deviation(pred: np.ndarray, target: np.ndarray) -> float:
    return abs(component_count_ratio(pred, target) - 1.0)


def skeleton_false_negative_rate(pred: np.ndarray, target: np.ndarray, radius: int = 2) -> float:
    sk_target = hard_skeletonize(target).astype(bool)
    if not sk_target.any():
        return 0.0
    covered = ndi.binary_dilation(pred.astype(bool), structure=_square(radius))
    return float((sk_target & ~covered).sum() / max(int(sk_target.sum()), 1))


def betti1_error(pred: np.ndarray, target: np.ndarray) -> float:
    def holes(mask: np.ndarray) -> int:
        _, background_count = ndi.label(
            ~mask.astype(bool), structure=np.ones((3, 3), dtype=bool)
        )
        return max(int(background_count) - 1, 0)

    return float(abs(holes(pred) - holes(target)))


def hd95_assd(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan"), float("nan")
    try:
        from medpy.metric.binary import assd, hd95

        return float(hd95(pred, target)), float(assd(pred, target))
    except Exception:
        pred_edge = pred ^ ndi.binary_erosion(pred)
        target_edge = target ^ ndi.binary_erosion(target)
        if pred_edge.sum() == 0 or target_edge.sum() == 0:
            return float("nan"), float("nan")
        d_to_target = ndi.distance_transform_edt(~target_edge)
        d_to_pred = ndi.distance_transform_edt(~pred_edge)
        dist = np.concatenate([d_to_target[pred_edge], d_to_pred[target_edge]])
        return float(np.percentile(dist, 95)), float(dist.mean())


def evaluate_binary_batch(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).detach().cpu()
    target_cpu = (target >= 0.5).detach().cpu()
    c = binary_confusion(pred, target_cpu)
    values = metrics_from_confusion(**c)
    cl_values = []
    cb_values = []
    hd_values = []
    assd_values = []
    branch_values = []
    thin_values = []
    continuity_values = []
    break_values = []
    betti0_values = []
    cc_ratio_values = []
    cc_deviation_values = []
    for p, t in zip(pred.numpy(), target_cpu.numpy()):
        p2 = p[0].astype(np.float32)
        t2 = t[0].astype(np.float32)
        cl_values.append(cldice_score(p2, t2))
        cb_values.append(cbdice_score(p2, t2))
        hd, assd = hd95_assd(p2, t2)
        hd_values.append(hd)
        assd_values.append(assd)
        branch_values.append(branch_skeleton_recovery(p2, t2))
        thin_values.append(thin_structure_recall(p2, t2))
        continuity_values.append(centerline_continuity(p2, t2))
        break_values.append(centerline_break_count(p2, t2))
        betti0_values.append(betti0_error(p2, t2))
        cc_ratio_values.append(component_count_ratio(p2, t2))
        cc_deviation_values.append(component_count_deviation(p2, t2))
    values["cldice"] = float(np.nanmean(cl_values))
    values["cbdice"] = float(np.nanmean(cb_values))
    values["hd95"] = float(np.nanmean(hd_values))
    values["assd"] = float(np.nanmean(assd_values))
    values["branch_recovery"] = float(np.nanmean(branch_values))
    values["thin_recall"] = float(np.nanmean(thin_values))
    values["continuity"] = float(np.nanmean(continuity_values))
    values["break_count"] = float(np.nanmean(break_values))
    values["betti0_error"] = float(np.nanmean(betti0_values))
    values["cc_ratio"] = float(np.nanmean(cc_ratio_values))
    values["cc_deviation"] = float(np.nanmean(cc_deviation_values))
    return values
