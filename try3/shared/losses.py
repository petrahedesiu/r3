
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from scipy.ndimage import distance_transform_edt


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce_loss).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        nClasses = pred_soft.shape[1]
        target_oh = F.one_hot(target, nClasses).permute(0, 3, 1, 2).float()
        losses = []
        for c in range(1, nClasses):
            pred_c = pred_soft[:, c]
            tgt_c = target_oh[:, c]
            tp = (pred_c * tgt_c).sum()
            fp = (pred_c * (1.0 - tgt_c)).sum()
            fn = ((1.0 - pred_c) * tgt_c).sum()
            ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            losses.append(1.0 - ti)
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == "all" else torch.unique(labels)
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, c] if C > 1 else probas[:, 0]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=probas.device, requires_grad=True)


def lovasz_softmax(probas, labels, classes="present"):
    return lovasz_softmax_flat(
        probas.permute(0, 2, 3, 1).reshape(-1, probas.size(1)),
        labels.reshape(-1),
        classes=classes,
    )


class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def compute_distance_map(mask, num_classes=3):
        H, W = mask.shape
        dm = np.zeros((num_classes, H, W), dtype=np.float32)
        for c in range(num_classes):
            binary = (mask == c).astype(np.uint8)
            if binary.sum() == 0:
                dm[c] = np.ones((H, W), dtype=np.float32)
                continue
            if binary.sum() == H * W:
                dm[c] = -np.ones((H, W), dtype=np.float32)
                continue
            pos_dist = distance_transform_edt(1 - binary).astype(np.float32)
            neg_dist = distance_transform_edt(binary).astype(np.float32)
            dm[c] = pos_dist - neg_dist
        return dm

    def forward(self, pred_softmax, distance_map):
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
        focal_weight=0.35,
        tversky_weight=0.35,
        lovasz_weight=0.30,
        boundary_weight=0.15,
        class_weights=None,
        tversky_alpha=0.2,
        tversky_beta=0.8,
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_boundary=False,
        boundary_rampup_epochs=15,
        num_classes=3,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary
        self.boundary_rampup_epochs = max(boundary_rampup_epochs, 1)

        self.focal = FocalLoss(focal_alpha, focal_gamma, class_weights)
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta)
        if self.use_boundary:
            self.boundary = BoundaryLoss(num_classes)

    def forward(self, pred, target, epoch=0, distance_map=None):
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        pred_soft = F.softmax(pred, dim=1)
        lov_loss = lovasz_softmax(pred_soft, target, classes="present")

        total = (self.focal_weight * focal_loss
                 + self.tversky_weight * tversky_loss
                 + self.lovasz_weight * lov_loss)

        metrics = {
            "focal": focal_loss.item(),
            "tversky": tversky_loss.item(),
            "lovasz": lov_loss.item(),
        }

        if self.use_boundary and distance_map is not None:
            b_loss = self.boundary(pred_soft, distance_map)
            ramp = min(1.0, epoch / self.boundary_rampup_epochs)
            effective_bw = self.boundary_weight * ramp
            total = total + effective_bw * b_loss
            metrics["boundary"] = b_loss.item()
            metrics["boundary_weight_effective"] = effective_bw

        return total, metrics
