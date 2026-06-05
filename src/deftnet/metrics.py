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
    hd_values = []
    assd_values = []
    for p, t in zip(pred.numpy(), target_cpu.numpy()):
        p2 = p[0].astype(np.float32)
        t2 = t[0].astype(np.float32)
        cl_values.append(cldice_score(p2, t2))
        hd, assd = hd95_assd(p2, t2)
        hd_values.append(hd)
        assd_values.append(assd)
    values["cldice"] = float(np.nanmean(cl_values))
    values["hd95"] = float(np.nanmean(hd_values))
    values["assd"] = float(np.nanmean(assd_values))
    return values
