"""Binary segmentation and evidence-gated memory-alignment losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GATE_TAU = 0.5
DEFAULT_GATE_ALPHA_FIXED = 0.5
DEFAULT_BONE_NORM_THR = 0.97
DEFAULT_EVIDENCE_TEMP = 0.01
DEFAULT_EVIDENCE_SMOOTH = 3
DEFAULT_LAMBDA_L = 1.0
DEFAULT_LAMBDA_MEM = 0.5
DEFAULT_MEM_SUP_WEIGHT = 0.3


class BinaryYLoss(nn.Module):
    """BCE plus soft Dice on binary Y logits with optional ignored voxels."""

    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, y_logits, y_target, ignore_mask=None):
        logits = y_logits.float()
        if logits.ndim == 4:
            logits = logits.unsqueeze(1)
        target = y_target.float()
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if ignore_mask is None:
            mask = torch.ones_like(target)
        else:
            mask = (~ignore_mask.bool()).float()
            if mask.ndim == 4:
                mask = mask.unsqueeze(1)
        probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        bce = (bce * mask).sum() / mask.sum().clamp(min=1.0)
        intersection = (probability * target * mask).sum()
        denominator = (probability * mask).sum() + (target * mask).sum()
        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        total = self.weight_bce * bce + self.weight_dice * dice
        return {"total": total, "bce": bce.detach(), "dice": dice.detach()}


class EvidenceGatedYLoss(nn.Module):
    """Evidence-gated binary Y loss with low-evidence and memory terms."""

    def __init__(
        self,
        lambda_L=DEFAULT_LAMBDA_L,
        lambda_mem=DEFAULT_LAMBDA_MEM,
        mem_sup_weight=DEFAULT_MEM_SUP_WEIGHT,
        tau=DEFAULT_GATE_TAU,
        mem_align="low",
        weight_bce=0.4,
        weight_dice=0.6,
    ):
        super().__init__()
        if mem_align not in ("low", "global", "off"):
            raise ValueError(
                f"mem_align must be low|global|off, got {mem_align}"
            )
        self.seg = BinaryYLoss(
            weight_bce=weight_bce, weight_dice=weight_dice
        )
        self.lambda_L = float(lambda_L)
        self.lambda_mem = float(lambda_mem)
        self.mem_sup_weight = float(mem_sup_weight)
        self.tau = float(tau)
        self.mem_align = mem_align

    def forward(
        self,
        y_logits,
        y_target,
        ignore_mask=None,
        g=None,
        y_mem_logits=None,
    ):
        ignore = (
            torch.zeros_like(y_target, dtype=torch.bool)
            if ignore_mask is None
            else ignore_mask.bool()
        )
        base = self.seg(y_logits, y_target, ignore)
        total = base["total"]
        output = {"seg": base["total"].detach()}
        low = None
        if g is not None:
            low = (g[:, 0] if g.ndim == 5 else g) < self.tau
        if self.lambda_L > 0 and low is not None and low.any():
            low_loss = self.seg(y_logits, y_target, ignore | ~low)["total"]
            total = total + self.lambda_L * low_loss
            output["low"] = low_loss.detach()
        if y_mem_logits is not None:
            if self.mem_sup_weight > 0:
                supervised_memory_loss = self.seg(
                    y_mem_logits, y_target, ignore
                )["total"]
                total = total + self.mem_sup_weight * supervised_memory_loss
                output["mem_sup"] = supervised_memory_loss.detach()
            if self.lambda_mem > 0 and self.mem_align != "off":
                region = ~ignore
                if self.mem_align == "low":
                    region = (
                        region & low
                        if low is not None
                        else torch.zeros_like(region)
                    )
                if region.any():
                    probability = torch.sigmoid(y_logits.float())[:, 0]
                    memory_probability = torch.sigmoid(
                        y_mem_logits.float()
                    )[:, 0].detach()
                    region_float = region.float()
                    alignment_loss = (
                        ((probability - memory_probability) ** 2) * region_float
                    ).sum() / region_float.sum().clamp(min=1.0)
                    total = total + self.lambda_mem * alignment_loss
                    output["align"] = alignment_loss.detach()
        output["total"] = total
        return output


__all__ = [
    "BinaryYLoss",
    "EvidenceGatedYLoss",
    "DEFAULT_GATE_TAU",
    "DEFAULT_GATE_ALPHA_FIXED",
    "DEFAULT_BONE_NORM_THR",
    "DEFAULT_EVIDENCE_TEMP",
    "DEFAULT_EVIDENCE_SMOOTH",
    "DEFAULT_LAMBDA_L",
    "DEFAULT_LAMBDA_MEM",
    "DEFAULT_MEM_SUP_WEIGHT",
]
