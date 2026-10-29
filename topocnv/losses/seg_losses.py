"""Segmentation losses used by TopoCNV training protocols."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        weight_bce: float = 0.5,
        weight_dice: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.include_background = include_background
        self.smooth = smooth

    def _loss_single(
        self,
        logits: torch.Tensor,
        targets_one_hot: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.shape[2:] != targets_one_hot.shape[2:]:
            targets_one_hot = F.interpolate(
                targets_one_hot, size=logits.shape[2:], mode="nearest"
            )

        log_probs = torch.log_softmax(logits, dim=1)
        bce = -(log_probs * targets_one_hot).sum(dim=1).mean()

        probs = torch.softmax(logits, dim=1)
        targets = targets_one_hot
        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            targets = targets[:, 1:]

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        denom = torch.sum(probs, dims) + torch.sum(targets, dims) + self.smooth
        dice_per_class = 2.0 * intersection / denom
        dice_loss = 1.0 - dice_per_class.mean()
        return self.weight_bce * bce + self.weight_dice * dice_loss

    def forward(self, logits, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            preds: Sequence[torch.Tensor] = list(logits)
        else:
            preds = [logits]

        num_classes = preds[0].shape[1]
        if targets.ndim == 4:
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        elif targets.ndim == 5:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
                targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
                targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
            else:
                targets_one_hot = targets.float()
        else:
            raise ValueError(f"Unexpected target shape {targets.shape}")

        weights = [1.0] + [0.5] * (len(preds) - 1)
        weight_sum = float(sum(weights))
        loss = 0.0
        for weight, pred in zip(weights, preds):
            loss = loss + weight * self._loss_single(pred, targets_one_hot)
        return loss / weight_sum


class WeightedBCEDiceLoss(nn.Module):
    """Class-weighted focal/CE plus Dice loss for nerve segmentation."""

    def __init__(
        self,
        weight_bce: float = 0.5,
        weight_dice: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
    ) -> None:
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.include_background = include_background
        self.smooth = smooth
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if class_weights is None:
            self.class_weights = torch.tensor([0.1, 10.0, 8.0, 6.0, 3.0, 3.0])
        else:
            self.class_weights = class_weights

    def _loss_single(
        self,
        logits: torch.Tensor,
        targets_one_hot: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            logits = logits.float()
            mask = mask.to(device=logits.device, dtype=torch.float32)
            if mask.ndim == 5 and mask.shape[1] == 1:
                mask = mask[:, 0]
            if mask.ndim != 4:
                raise ValueError(f"Unexpected mask shape {mask.shape}")
            if mask.shape[1:] != logits.shape[2:]:
                mask = F.interpolate(mask[:, None], size=logits.shape[2:], mode="nearest")[:, 0]
        min_ch = getattr(
            self, "_min_ch_fallback", min(logits.shape[1], targets_one_hot.shape[1])
        )
        if logits.shape[2:] != targets_one_hot.shape[2:]:
            targets_one_hot = F.interpolate(
                targets_one_hot, size=logits.shape[2:], mode="nearest"
            )

        if torch.isnan(targets_one_hot).any() or torch.isinf(targets_one_hot).any():
            raise ValueError("targets_one_hot contains NaN or Inf values")

        class_weight_source = self.class_weights
        if targets_one_hot.shape[1] != len(class_weight_source):
            if targets_one_hot.shape[1] < len(class_weight_source):
                class_weight_source = class_weight_source[: targets_one_hot.shape[1]]
            else:
                pad = torch.ones(targets_one_hot.shape[1] - len(class_weight_source))
                class_weight_source = torch.cat([class_weight_source, pad])
        else:
            class_weight_source = self.class_weights
        class_weights = class_weight_source.to(logits.device)

        voxel_weight_expanded = None
        if voxel_weight is not None:
            if voxel_weight.ndim == 4:
                voxel_weight = voxel_weight.unsqueeze(1)
            elif voxel_weight.ndim != 5:
                raise ValueError(f"Unexpected voxel_weight shape {voxel_weight.shape}")
            if voxel_weight.shape[1] != 1:
                raise ValueError(
                    f"voxel_weight channel must be 1, got {voxel_weight.shape[1]}"
                )
            voxel_weight_expanded = voxel_weight.to(logits.device).expand(
                -1, logits.shape[1], -1, -1, -1
            )

        if self.use_focal:
            bce = self._focal_loss(
                logits, targets_one_hot, class_weights, voxel_weight_expanded, mask=mask
            )
        else:
            log_probs = torch.log_softmax(logits, dim=1)
            target = targets_one_hot
            aligned_log_probs = log_probs
            if aligned_log_probs.shape[1] != target.shape[1]:
                min_ch = min(aligned_log_probs.shape[1], target.shape[1])
                aligned_log_probs = aligned_log_probs[:, :min_ch]
                target = target[:, :min_ch]
                expanded_class_weights = class_weights[:min_ch]
            else:
                expanded_class_weights = class_weights
            ce_per_voxel = -(aligned_log_probs * target).sum(dim=1)
            class_weights_expanded = expanded_class_weights.view(1, -1, 1, 1, 1)
            weighted_ce = ce_per_voxel * (
                target * class_weights_expanded
            ).sum(dim=1)
            if voxel_weight_expanded is not None:
                voxel_weight_3d = (
                    voxel_weight_expanded[:, 0, :, :, :]
                    if voxel_weight_expanded.shape[1] > 1
                    else voxel_weight_expanded.squeeze(1)
                )
                weighted_ce = weighted_ce * voxel_weight_3d
            if mask is None:
                bce = weighted_ce.mean()
            else:
                bce = (weighted_ce * mask).sum() / mask.sum().clamp(min=1.0)

        probs = torch.softmax(logits, dim=1)
        targets = targets_one_hot
        min_ch = min(probs.shape[1], targets.shape[1])
        if probs.shape[1] != targets.shape[1]:
            probs = probs[:, :min_ch]
            targets = targets[:, :min_ch]

        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            targets = targets[:, 1:]
            n_fg = probs.shape[1]
            if "min_ch" in dir() and min_ch > 1:
                weight_start = 1
            else:
                weight_start = 1
            if n_fg <= len(class_weights) - weight_start:
                weights_for_dice = class_weights[weight_start : weight_start + n_fg]
            else:
                weights_for_dice = class_weights[weight_start:]
        else:
            weights_for_dice = class_weights[: probs.shape[1]]

        if mask is not None:
            probs = probs * mask[:, None]
            targets = targets * mask[:, None]
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        denom = torch.sum(probs, dims) + torch.sum(targets, dims) + self.smooth
        dice_per_class = 2.0 * intersection / denom
        weighted_dice = (
            (dice_per_class * weights_for_dice).sum() / weights_for_dice.sum()
        )
        dice_loss = 1.0 - weighted_dice
        return self.weight_bce * bce + self.weight_dice * dice_loss

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        if probs.shape[1] != targets.shape[1]:
            target_channels = targets.shape[1]
            if probs.shape[1] > target_channels:
                probs = probs[:, :target_channels]
                logits = logits[:, :target_channels]
                log_probs = log_probs[:, :target_channels]
            else:
                targets = targets[:, : probs.shape[1]]

        p_t = (probs * targets).sum(dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
        focal_weight = (1.0 - p_t) ** self.focal_gamma
        ce = -(log_probs * targets).sum(dim=1)
        class_weight_per_voxel = (
            targets * class_weights.view(1, -1, 1, 1, 1)
        ).sum(dim=1)
        focal_loss = (
            self.focal_alpha * focal_weight * ce * class_weight_per_voxel
        )
        if voxel_weight is not None:
            if voxel_weight.shape[1] == 1:
                voxel_weight = voxel_weight.squeeze(1)
            elif voxel_weight.shape[1] > 1:
                voxel_weight = voxel_weight[:, 0, :, :, :]
            focal_loss = focal_loss * voxel_weight
        if mask is not None:
            return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return focal_loss.mean()

    def forward(
        self,
        logits,
        targets: torch.Tensor,
        voxel_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            preds: Sequence[torch.Tensor] = list(logits)
        else:
            preds = [logits]

        num_classes = preds[0].shape[1]
        if targets.ndim == 4:
            min_value = targets.min().item()
            max_value = targets.max().item()
            if min_value < 0 or max_value >= num_classes:
                raise ValueError(
                    f"Label values out of range: min={min_value}, max={max_value}, "
                    f"but num_classes={num_classes} (valid range: 0-{num_classes-1}). "
                    f"Target shape: {targets.shape}"
                )
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        elif targets.ndim == 5:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
                min_value = targets.min().item()
                max_value = targets.max().item()
                if min_value < 0 or max_value >= num_classes:
                    raise ValueError(
                        f"Label values out of range: min={min_value}, max={max_value}, "
                        f"but num_classes={num_classes} (valid range: 0-{num_classes-1}). "
                        f"Target shape: {targets.shape}"
                    )
                targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
                targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
            else:
                targets_one_hot = targets.float()
        else:
            raise ValueError(f"Unexpected target shape {targets.shape}")

        weights = [1.0] + [0.5] * (len(preds) - 1)
        weight_sum = float(sum(weights))
        loss = 0.0
        for weight, pred in zip(weights, preds):
            loss = loss + weight * self._loss_single(
                pred, targets_one_hot, voxel_weight=voxel_weight, mask=mask
            )
        return loss / weight_sum


class MaskedWeightedBCEDice(WeightedBCEDiceLoss):
    """复用统一损失，保留 v4 的构造参数、ROI 位置参数及权重 buffer。"""

    def __init__(self, class_weights, weight_bce=0.3, weight_dice=0.7, focal_alpha=0.25,
                 focal_gamma=3.0, smooth=1e-5):
        super().__init__(weight_bce=weight_bce, weight_dice=weight_dice,
                         include_background=False, smooth=smooth,
                         class_weights=torch.as_tensor(class_weights, dtype=torch.float32),
                         focal_alpha=focal_alpha, focal_gamma=focal_gamma, use_focal=True)
        weights = self.class_weights
        del self.class_weights
        self.register_buffer("class_weights", weights)

    def forward(self, logits, target, mask=None):
        return super().forward(logits.float(), target, mask=mask)


def multiclass_dice_score(
    logits,
    targets: torch.Tensor,
    include_background: bool = False,
    smooth: float = 1e-5,
    per_class: bool = False,
) -> float:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    num_classes = logits.shape[1]
    if targets.ndim == 4:
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
    elif targets.ndim == 5:
        if targets.shape[1] == 1:
            targets = targets.squeeze(1)
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            targets_one_hot = targets.float()
    else:
        raise ValueError(f"Unexpected target shape {targets.shape}")

    probs = torch.softmax(logits, dim=1)
    if probs.shape[1] != targets_one_hot.shape[1]:
        min_ch = min(probs.shape[1], targets_one_hot.shape[1])
        probs = probs[:, :min_ch]
        targets_one_hot = targets_one_hot[:, :min_ch]
    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets_one_hot, dims)
    denom = torch.sum(probs, dims) + torch.sum(targets_one_hot, dims) + smooth
    dice_per_class = 2.0 * intersection / denom
    if per_class:
        return dice_per_class.cpu().numpy()
    return float(dice_per_class.mean().item())


class HierarchicalTreeLoss(nn.Module):
    """Multi-level loss for the Hierarchical Soft-Tree Stage2 model."""

    def __init__(
        self,
        final_loss: nn.Module,
        group_loss: nn.Module,
        branch_loss: nn.Module,
        lambda_group: float = 0.3,
        lambda_branch: float = 0.2,
        lambda_aux: float = 0.3,
    ) -> None:
        super().__init__()
        self.final_loss = final_loss
        self.group_loss = group_loss
        self.branch_loss = branch_loss
        self.lambda_group = lambda_group
        self.lambda_branch = lambda_branch
        self.lambda_aux = lambda_aux

    @staticmethod
    def _labels_to_group(labels: torch.Tensor) -> torch.Tensor:
        group = torch.zeros_like(labels)
        group[(labels >= 1) & (labels <= 3)] = 1
        group[labels == 4] = 2
        return group.long()

    @staticmethod
    def _labels_to_branch_a(labels: torch.Tensor) -> torch.Tensor:
        sub = torch.zeros_like(labels)
        mask = (labels >= 1) & (labels <= 3)
        sub[mask] = labels[mask] - 1
        return sub.long()

    @staticmethod
    def _binary_aux_loss(
        aux_logits: torch.Tensor, aux_target: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(aux_logits.squeeze(1))
        target = aux_target.float()
        bce = F.binary_cross_entropy_with_logits(
            aux_logits.squeeze(1), target, reduction="mean"
        )
        smooth = 1e-5
        intersection = (pred * target).sum()
        denom = pred.sum() + target.sum() + smooth
        dice_loss = 1.0 - 2.0 * intersection / denom
        return 0.4 * bce + 0.6 * dice_loss

    def _soft_masked_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft_mask: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(targets.long(), num_classes).permute(
            0, 4, 1, 2, 3
        ).float()
        log_probs = torch.log_softmax(logits, dim=1)
        ce_per_voxel = -(log_probs * target_one_hot).sum(dim=1)
        if class_weights is not None:
            weights = class_weights[:num_classes].to(logits.device)
            class_weight_per_voxel = (
                target_one_hot * weights.view(1, -1, 1, 1, 1)
            ).sum(dim=1)
            ce_per_voxel = ce_per_voxel * class_weight_per_voxel
        ce = (ce_per_voxel * soft_mask).sum() / (soft_mask.sum() + 1e-8)

        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = (
            probs * target_one_hot * soft_mask.unsqueeze(1)
        ).sum(dims)
        denom = (
            (probs * soft_mask.unsqueeze(1)).sum(dims)
            + (target_one_hot * soft_mask.unsqueeze(1)).sum(dims)
            + 1e-5
        )
        dice_per_class = 2.0 * intersection / denom
        if class_weights is not None:
            weighted_dice = (dice_per_class * weights).sum() / weights.sum()
            dice_loss = 1.0 - weighted_dice
        else:
            dice_loss = 1.0 - dice_per_class.mean()
        return 0.4 * ce + 0.6 * dice_loss

    def forward(
        self,
        final_probs: torch.Tensor,
        group_logits: torch.Tensor,
        branch_a_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        labels: torch.Tensor,
        aux_target: torch.Tensor,
        p_group: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        final_logits = torch.log(final_probs.clamp(1e-7, 1.0))
        loss_final = self.final_loss(final_logits, labels)
        group_labels = self._labels_to_group(labels)
        loss_group = self.group_loss(group_logits, group_labels)
        if p_group is not None:
            mask_a = p_group[:, 1]
        else:
            mask_a = torch.softmax(group_logits, dim=1)[:, 1]
        branch_a_labels = self._labels_to_branch_a(labels)
        branch_a_weights = torch.tensor([3.0, 2.0, 1.0], device=labels.device)
        loss_branch_a = self._soft_masked_loss(
            branch_a_logits,
            branch_a_labels,
            mask_a,
            class_weights=branch_a_weights,
        )
        loss_aux = self._binary_aux_loss(aux_logits, aux_target)
        total = (
            loss_final
            + self.lambda_group * loss_group
            + self.lambda_branch * loss_branch_a
            + self.lambda_aux * loss_aux
        )
        return {
            "total": total,
            "l_final": loss_final,
            "l_group": loss_group,
            "l_branch_a": loss_branch_a,
            "l_aux": loss_aux,
        }


__all__ = [
    "BCEDiceLoss",
    "WeightedBCEDiceLoss",
    "MaskedWeightedBCEDice",
    "multiclass_dice_score",
    "HierarchicalTreeLoss",
]
