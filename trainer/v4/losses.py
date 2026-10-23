"""v4 losses.

MaskedWeightedBCEDice: the SOTA DynUNet loss (npc_losses.WeightedBCEDiceLoss with use_focal=True,
include_background=False, single output), with an optional ROI mask: focal term averaged over ROI
voxels, Dice sums restricted to ROI voxels. With mask=None it equals the original (tests/test_v4_losses.py).

VisibilityLoss: auxiliary 5-class head (background + classes 1/2/3/4), CE + soft Dice; class 5 is
either background or ignored.

skeleton_recall_loss: Skeleton Recall Loss (Kirchhoff et al., ECCV 2024) on the binary Y probability
against the precomputed tubed skeleton of GT Y.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedWeightedBCEDice(nn.Module):
    def __init__(self, class_weights, weight_bce=0.3, weight_dice=0.7, focal_alpha=0.25,
                 focal_gamma=3.0, smooth=1e-5):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth

    def forward(self, logits, target, mask=None):
        n_cls = logits.shape[1]
        logits = logits.float()
        onehot = F.one_hot(target.long(), n_cls).permute(0, 4, 1, 2, 3).float()
        cw = self.class_weights.to(logits.device)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)

        p_t = (probs * onehot).sum(dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
        focal_w = (1.0 - p_t) ** self.focal_gamma
        ce = -(log_probs * onehot).sum(dim=1)
        cw_vox = (onehot * cw.view(1, -1, 1, 1, 1)).sum(dim=1)
        focal = self.focal_alpha * focal_w * ce * cw_vox
        pf, tf = probs[:, 1:], onehot[:, 1:]
        if mask is None:
            focal = focal.mean()
        else:
            m = mask.float()
            focal = (focal * m).sum() / m.sum().clamp(min=1.0)
            pf = pf * m[:, None]
            tf = tf * m[:, None]

        dims = (0, 2, 3, 4)
        dice_c = 2.0 * (pf * tf).sum(dims) / (pf.sum(dims) + tf.sum(dims) + self.smooth)
        w = cw[1:]
        dice_loss = 1.0 - (dice_c * w).sum() / w.sum()
        return self.weight_bce * focal + self.weight_dice * dice_loss


def visibility_target(raw_label):
    """Raw labels 0..5 -> visibility classes 0..4 (class 5 -> 0; callers mask it out for 'ignore')."""
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


def skeleton_recall_loss(prob_fg, skel, smooth=1e-5):
    """1 - soft recall of the tubed GT skeleton: 1 - (sum p*S + eps) / (sum S + eps). 0 if S is empty."""
    s = skel.float()
    return 1.0 - ((prob_fg.float() * s).sum() + smooth) / (s.sum() + smooth)
