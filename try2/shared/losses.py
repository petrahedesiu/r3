
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            pred, target, weight=self.class_weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)
        nClasses = pred_soft.shape[1]

        target_one_hot = (
            F.one_hot(target, nClasses).permute(0, 3, 1, 2).float()
        )

        losses = []
        for c in range(1, nClasses):
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]

            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1.0 - target_c)).sum()
            fn = ((1.0 - pred_c) * target_c).sum()

            tversky_index = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
            )
            losses.append(1.0 - tversky_index)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == "all" else torch.unique(labels)

    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=probas.device, requires_grad=True)


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
    per_image: bool = False,
) -> torch.Tensor:
    if per_image:
        loss = torch.stack(
            [
                lovasz_softmax_flat(
                    prob.unsqueeze(0)
                    .permute(0, 2, 3, 1)
                    .reshape(-1, prob.size(0)),
                    lab.unsqueeze(0).reshape(-1),
                    classes=classes,
                )
                for prob, lab in zip(probas, labels)
            ]
        ).mean()
    else:
        loss = lovasz_softmax_flat(
            probas.permute(0, 2, 3, 1).reshape(-1, probas.size(1)),
            labels.reshape(-1),
            classes=classes,
        )
    return loss


class BoundaryLoss(nn.Module):

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def compute_distance_map(
        mask: np.ndarray,
        num_classes: int = 3,
    ) -> np.ndarray:
        H, W = mask.shape
        distance_map = np.zeros((num_classes, H, W), dtype=np.float32)

        # loop over classes
        for c in range(num_classes):
            binary = (mask == c).astype(np.uint8)

            if binary.sum() == 0:
                distance_map[c] = np.ones((H, W), dtype=np.float32)
                continue
            if binary.sum() == H * W:
                distance_map[c] = -np.ones((H, W), dtype=np.float32)
                continue

            pos_dist = distance_transform_edt(1 - binary).astype(np.float32)
            neg_dist = distance_transform_edt(binary).astype(np.float32)

            distance_map[c] = pos_dist - neg_dist

        return distance_map

    def forward(
        self,
        pred_softmax: torch.Tensor,
        distance_map: torch.Tensor,
    ) -> torch.Tensor:
        assert pred_softmax.shape == distance_map.shape, (
            f"Shape mismatch: pred {pred_softmax.shape} vs "
            f"dist_map {distance_map.shape}"
        )

        losses = []
        for c in range(1, self.num_classes):
            loss_c = (pred_softmax[:, c] * distance_map[:, c]).mean()
            losses.append(loss_c)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pred_softmax.device, requires_grad=True)


class CompoundLoss(nn.Module):

    def __init__(
        self,
        focal_weight: float = 0.35,
        tversky_weight: float = 0.35,
        lovasz_weight: float = 0.30,
        boundary_weight: float = 0.10,
        class_weights: Optional[torch.Tensor] = None,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_boundary: bool = False,
        epoch_for_boundary_rampup: int = 15,
        num_classes: int = 3,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary
        self.epoch_for_boundary_rampup = max(epoch_for_boundary_rampup, 1)

        self.focal = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
        )
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

        if self.use_boundary:
            self.boundary = BoundaryLoss(num_classes=num_classes)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        distance_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)

        pred_soft = F.softmax(pred, dim=1)
        lovasz_loss = lovasz_softmax(pred_soft, target, classes="present")

        total = (
            self.focal_weight * focal_loss
            + self.tversky_weight * tversky_loss
            + self.lovasz_weight * lovasz_loss
        )

        metrics: Dict[str, float] = {
            "focal": focal_loss.item(),
            "tversky": tversky_loss.item(),
            "lovasz": lovasz_loss.item(),
        }

        if self.use_boundary and distance_map is not None:
            boundary_loss = self.boundary(pred_soft, distance_map)

            ramp = min(1.0, epoch / self.epoch_for_boundary_rampup)
            effective_bw = self.boundary_weight * ramp

            total = total + effective_bw * boundary_loss
            metrics["boundary"] = boundary_loss.item()
            metrics["boundary_weight_effective"] = effective_bw

        return total, metrics
