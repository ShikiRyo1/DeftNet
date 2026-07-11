from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class SoftSkeletonize2D(nn.Module):
    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = int(num_iter)

    @staticmethod
    def soft_erode(x: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
        return torch.minimum(p1, p2)

    @staticmethod
    def soft_dilate(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, 3, stride=1, padding=1)

    def soft_open(self, x: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        opened = self.soft_open(x)
        skel = F.relu(x - opened)
        for _ in range(self.num_iter):
            x = self.soft_erode(x)
            opened = self.soft_open(x)
            delta = F.relu(x - opened)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class SoftClDiceLoss2D(nn.Module):
    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.skeletonize = SoftSkeletonize2D(num_iter)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        skel_pred = self.skeletonize(prob)
        with torch.no_grad():
            skel_true = self.skeletonize((target > 0.5).float())
        tprec = (skel_pred * target).sum() / (skel_pred.sum() + self.smooth)
        tsens = (skel_true * prob).sum() / (skel_true.sum() + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
        return 1.0 - cldice


class CombinedSegLoss(nn.Module):
    """BCE-Dice objective used in both manuscript training phases.

    ``w_cldice`` is retained only to load legacy configs. The current protocol
    fixes it to zero; clDice and cbDice are evaluation-only endpoints.
    """

    def __init__(self, w_bce: float = 0.50, w_dice: float = 0.50, w_cldice: float = 0.0):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.w_cldice = float(w_cldice)
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = SoftClDiceLoss2D(num_iter=10)

    def forward(self, outputs, target: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            main = outputs[0]
            aux = 0.0
            for item in outputs[1:]:
                if item.shape[-2:] != target.shape[-2:]:
                    item = F.interpolate(item, size=target.shape[-2:], mode="bilinear", align_corners=False)
                aux = aux + self.bce(item, target) + dice_loss(item, target)
            aux = 0.4 * aux / max(1, len(outputs) - 1)
        else:
            main = outputs
            aux = 0.0
        bce = self.bce(main, target)
        dsc = dice_loss(main, target)
        cld = self.cldice(main, target) if self.w_cldice > 0 else main.new_zeros(())
        total = self.w_bce * bce + self.w_dice * dsc + self.w_cldice * cld + aux
        return total, {"bce": float(bce.detach()), "dice": float(dsc.detach()), "cldice": float(cld.detach())}
