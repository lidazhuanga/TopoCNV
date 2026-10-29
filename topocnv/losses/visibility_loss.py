"""v4 可见度辅助头：背景与类别 1–4 的交叉熵及软 Dice。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def visibility_target(raw_label):
    """将原标签 0–5 映射到可见度 0–4，类别 5 作为背景或由调用者屏蔽。"""
    lab = raw_label.long()
    return torch.where((lab >= 1) & (lab <= 4), lab, torch.zeros_like(lab))


class VisibilityLoss(nn.Module):
    def __init__(self, class5="background", smooth=1.0):
        super().__init__()
        if class5 not in ("background", "ignore"):
            raise ValueError(f"class5 must be 'background' or 'ignore', got {class5!r}")
        self.class5 = class5
        self.smooth = smooth

    def forward(self, logits5, raw_label, mask=None):
        lab = raw_label.long()
        valid = torch.ones_like(lab, dtype=torch.bool) if mask is None else mask.bool()
        if self.class5 == "ignore":
            valid = valid & (lab != 5)
        tgt = visibility_target(lab)
        v = valid.float()
        log_probs = torch.log_softmax(logits5.float(), dim=1)
        ce = -log_probs.gather(1, tgt[:, None])[:, 0]
        ce = (ce * v).sum() / v.sum().clamp(min=1.0)
        probs = log_probs.exp()[:, 1:] * v[:, None]
        oh = F.one_hot(tgt, 5).permute(0, 4, 1, 2, 3)[:, 1:].float() * v[:, None]
        dims = (0, 2, 3, 4)
        dice = (2.0 * (probs * oh).sum(dims) + self.smooth) / (probs.sum(dims) + oh.sum(dims) + self.smooth)
        return ce + (1.0 - dice.mean())


__all__ = ["VisibilityLoss", "visibility_target"]
