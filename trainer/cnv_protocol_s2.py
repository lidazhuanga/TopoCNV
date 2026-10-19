#!/usr/bin/env python3
"""
CNV protocol S2 helpers (Phase 2/3 wiring).

Naming lock:
  bony_roi      ≡ class4 only
  bypass_roi / class5_bypass ≡ class5 (绕行) — NOT in Y, NOT in S2 target/loss
  Y = 1∪2∪3∪4

Primary S2 head = binary Y. SoftTree kept as legacy behind use_soft_tree=True.

NO TRAINING launch in this module.
# CNV protocol v0.3
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L1_DILATE_R_MM = 3.0
BONY_ROI_LABEL = 4
BYPASS_ROI_LABEL = 5
DEFAULT_SPACING_ZYX = (2.5, 1.1328120231628418, 1.1328120231628418)


def dilate_mm(mask, spacing_zyx, r_mm=3.0):
    m = np.asarray(mask, dtype=bool)
    if not m.any() or r_mm <= 0:
        return m.copy()
    try:
        from scipy.ndimage import binary_dilation
        rz = max(1, int(np.ceil(r_mm / float(spacing_zyx[0]))))
        ry = max(1, int(np.ceil(r_mm / float(spacing_zyx[1]))))
        rx = max(1, int(np.ceil(r_mm / float(spacing_zyx[2]))))
        zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
        fp = (zz / max(rz, 1)) ** 2 + (yy / max(ry, 1)) ** 2 + (xx / max(rx, 1)) ** 2 <= 1.0
        return binary_dilation(m, structure=fp)
    except ImportError:
        out = m.copy()
        n_iter = max(1, int(np.ceil(r_mm / min(spacing_zyx))))
        for _ in range(n_iter):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            acc = padded[1:-1, 1:-1, 1:-1].copy()
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dz == dy == dx == 0:
                            continue
                        acc |= padded[1 + dz:out.shape[0] + 1 + dz,
                                      1 + dy:out.shape[1] + 1 + dy,
                                      1 + dx:out.shape[2] + 1 + dx]
            out = acc
        return out


def convert_label_to_Y(label: np.ndarray) -> np.ndarray:
    """Binary Y: 1 where label in {1,2,3,4}; class5_bypass -> 0."""
    lab = np.asarray(label)
    return ((lab >= 1) & (lab <= 4)).astype(np.int64)


def class5_bypass_ignore_mask(label: np.ndarray) -> np.ndarray:
    return np.asarray(label) == BYPASS_ROI_LABEL


def stage1_softmax_to_cascade_prior(
    s1_soft: np.ndarray,
    spacing_zyx: Sequence[float] = DEFAULT_SPACING_ZYX,
    r_mm: float = L1_DILATE_R_MM,
) -> np.ndarray:
    """[P(c4), dilate(bony_roi=class4)] from S1 softmax. class5_bypass not in dilate base."""
    s1 = np.asarray(s1_soft, dtype=np.float32)
    if s1.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {s1.shape}")
    C = s1.shape[0]
    if C == 6:
        p_c4 = s1[BONY_ROI_LABEL].astype(np.float32)
        hard_c4 = np.argmax(s1, axis=0) == BONY_ROI_LABEL
        dil = dilate_mm(hard_c4, spacing_zyx, r_mm=r_mm).astype(np.float32)
        return np.stack([p_c4, dil], axis=0)
    if C == 3:
        return s1[1:].astype(np.float32).copy()  # legacy
    if C == 2:
        return s1.astype(np.float32).copy()
    raise ValueError(f"Unsupported S1 softmax C={C}")


class BinaryYLoss(nn.Module):
    """BCE + soft Dice on binary Y logits; voxels in ignore_mask are excluded.

    Computed in float32 on logits: F.binary_cross_entropy on probabilities is
    rejected inside CUDA autocast.
    """

    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, y_logits, y_target, ignore_mask=None):
        logits = y_logits.float()
        if logits.ndim == 4:
            logits = logits.unsqueeze(1)
        tgt = y_target.float()
        if tgt.ndim == 4:
            tgt = tgt.unsqueeze(1)
        if ignore_mask is None:
            m = torch.ones_like(tgt)
        else:
            m = (~ignore_mask.bool()).float()
            if m.ndim == 4:
                m = m.unsqueeze(1)
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        bce = (bce * m).sum() / m.sum().clamp(min=1.0)
        inter = (prob * tgt * m).sum()
        denom = (prob * m).sum() + (tgt * m).sum()
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        total = self.weight_bce * bce + self.weight_dice * dice
        return {"total": total, "bce": bce.detach(), "dice": dice.detach()}


# ---- Claim-M★ evidence map and gate (protocol §2) ----
DEFAULT_GATE_TAU = 0.5
DEFAULT_GATE_ALPHA_FIXED = 0.5
# Normalized-CT value of the raw-HU bone window (HU >= 200) under the cache's
# per-case 1-99 percentile normalization. On the 154 training CTs the 99th
# percentile is ~200 HU (median normalized value of HU=200: 0.993, p10 0.964;
# 35% of cases saturate at 1.0), so bone sits at the top of the range.
# Reproduce with tools/calibrate_bone_threshold.py.
DEFAULT_BONE_NORM_THR = 0.97
DEFAULT_EVIDENCE_TEMP = 0.01
DEFAULT_EVIDENCE_SMOOTH = 3
DEFAULT_LAMBDA_L = 1.0
DEFAULT_LAMBDA_MEM = 0.5
DEFAULT_MEM_SUP_WEIGHT = 0.3


def evidence_map_g1(x, bone_thr=DEFAULT_BONE_NORM_THR, temp=DEFAULT_EVIDENCE_TEMP,
                    smooth=DEFAULT_EVIDENCE_SMOOTH):
    """G1 evidence map g(x) in [0,1]; high = local CT evidence is sufficient.

    x: [B,C,D,H,W] with channel 0 = normalized CT and, when C >= 2, the cascade
    prior [P(class4), dilate(class4, 3 mm)] in channels 1..2.
    g = local mean of max(bone-window proxy, class4 canal proximity).
    """
    ct = x[:, :1].float()
    g = torch.sigmoid((ct - bone_thr) / temp)
    if x.shape[1] >= 2:
        c4 = x[:, 1:min(3, x.shape[1])].float().clamp(0, 1).amax(dim=1, keepdim=True)
        g = torch.maximum(g, c4)
    if smooth > 1:
        g = F.avg_pool3d(g, kernel_size=smooth, stride=1, padding=smooth // 2,
                         count_include_pad=False)
    return g.clamp(0, 1)


class EvidenceGatedYLoss(nn.Module):
    """Protocol §2.3 loss on binary Y.

    L = L_seg(Y) + lambda_L * L_seg(Y; w=1_L)
        + lambda_mem * ||sigmoid(Y_hat) - sg(sigmoid(Y_mem))||^2 on the align region
        + mem_sup_weight * L_seg(Y_mem)
    with L = {g < tau}. mem_align: 'low' (Full; only L), 'global' (A_gate), 'off'.
    """

    def __init__(self, lambda_L=DEFAULT_LAMBDA_L, lambda_mem=DEFAULT_LAMBDA_MEM,
                 mem_sup_weight=DEFAULT_MEM_SUP_WEIGHT, tau=DEFAULT_GATE_TAU,
                 mem_align="low", weight_bce=0.4, weight_dice=0.6):
        super().__init__()
        if mem_align not in ("low", "global", "off"):
            raise ValueError(f"mem_align must be low|global|off, got {mem_align}")
        self.seg = BinaryYLoss(weight_bce=weight_bce, weight_dice=weight_dice)
        self.lambda_L = float(lambda_L)
        self.lambda_mem = float(lambda_mem)
        self.mem_sup_weight = float(mem_sup_weight)
        self.tau = float(tau)
        self.mem_align = mem_align

    def forward(self, y_logits, y_target, ignore_mask=None, g=None, y_mem_logits=None):
        ignore = (torch.zeros_like(y_target, dtype=torch.bool) if ignore_mask is None
                  else ignore_mask.bool())
        base = self.seg(y_logits, y_target, ignore)
        total = base["total"]
        out = {"seg": base["total"].detach()}
        low = None
        if g is not None:
            low = (g[:, 0] if g.ndim == 5 else g) < self.tau
        if self.lambda_L > 0 and low is not None and low.any():
            l_low = self.seg(y_logits, y_target, ignore | ~low)["total"]
            total = total + self.lambda_L * l_low
            out["low"] = l_low.detach()
        if y_mem_logits is not None:
            if self.mem_sup_weight > 0:
                l_sup = self.seg(y_mem_logits, y_target, ignore)["total"]
                total = total + self.mem_sup_weight * l_sup
                out["mem_sup"] = l_sup.detach()
            if self.lambda_mem > 0 and self.mem_align != "off":
                region = ~ignore
                if self.mem_align == "low":
                    region = region & low if low is not None else torch.zeros_like(region)
                if region.any():
                    p = torch.sigmoid(y_logits.float())[:, 0]
                    pm = torch.sigmoid(y_mem_logits.float())[:, 0].detach()
                    r = region.float()
                    l_align = (((p - pm) ** 2) * r).sum() / r.sum().clamp(min=1.0)
                    total = total + self.lambda_mem * l_align
                    out["align"] = l_align.detach()
        out["total"] = total
        return out


class BinaryYStage2Model(nn.Module):
    """Primary protocol S2: CT + cascade prior -> binary Y (class5_bypass excluded).

    Evidence-gated anatomical memory (Claim-M★): the backbone fuses local and
    memory-retrieved encoder tokens as f = g*f_loc + (1-g)*f_mem, where g is the
    G1 evidence map. gate_mode: 'evidence' (Full), 'fixed' (A_gate, g = alpha),
    'none' (plain memory). A memory decoder head yields Y_mem for the alignment
    term. Returns (y_prob, y_logits, g, y_mem_logits).
    """

    def __init__(self, net, hier_decoder, feat_dim: int, use_confidence_gate: bool = True,
                 in_channels: int = 3, gate_alpha_fixed=None,
                 bone_thr: float = DEFAULT_BONE_NORM_THR,
                 evidence_temp: float = DEFAULT_EVIDENCE_TEMP,
                 mem_dim=None):
        super().__init__()
        self.net = net
        self.hier_decoder = hier_decoder
        self.in_channels = in_channels
        self.use_memory_bank = getattr(net, "memory_bank", None) is not None
        if gate_alpha_fixed is not None:
            self.gate_mode = "fixed"
        elif use_confidence_gate:
            self.gate_mode = "evidence"
        else:
            self.gate_mode = "none"
        self.gate_alpha_fixed = gate_alpha_fixed
        self.bone_thr = float(bone_thr)
        self.evidence_temp = float(evidence_temp)
        self.y_head = nn.Conv3d(feat_dim, 1, kernel_size=1)
        self.mem_head = (nn.Linear(mem_dim, 1)
                         if self.use_memory_bank and mem_dim and hasattr(net, "evidence_gate")
                         else None)
        print(f"✓ BinaryY Stage2 (Y=1∪2∪3∪4; class5_bypass not in target) "
              f"memory={self.use_memory_bank} gate={self.gate_mode} "
              f"mem_head={self.mem_head is not None}")

    def evidence(self, x):
        return evidence_map_g1(x, bone_thr=self.bone_thr, temp=self.evidence_temp)

    def _memory_logits(self, target_shape):
        tok = getattr(self.net, "_mem_tok0", None)
        if self.mem_head is None or tok is None:
            return None
        t, grid = tok
        B = t.shape[0]
        logit = self.mem_head(t).transpose(1, 2).reshape(B, 1, *grid)
        padded = getattr(self.net, "_patch_padded", tuple(target_shape))
        logit = F.interpolate(logit.float(), size=padded, mode="trilinear", align_corners=False)
        D, H, W = target_shape
        return logit[:, :, :D, :H, :W]

    def forward(self, x, labels=None, coord_info=None):
        g = self.evidence(x)
        if hasattr(self.net, "evidence_gate"):
            if not self.use_memory_bank or self.gate_mode == "none":
                self.net.evidence_gate = None
            elif self.gate_mode == "fixed":
                self.net.evidence_gate = float(self.gate_alpha_fixed)
            else:
                self.net.evidence_gate = g
        if hasattr(self.net, "evidence_gate"):
            feats, stem_skips, target_shape = self.net.forward_features(x, labels, coord_info)
        else:
            feats, stem_skips, target_shape = self.net.forward_features(x, labels)

        if getattr(self.net, "provides_fullres_features", False):
            feat = self.hier_decoder(feats)
        else:
            feat = self.hier_decoder(feats, stem_skips, target_shape)

        y_logits = self.y_head(feat)
        y_prob = torch.sigmoid(y_logits)
        y_mem_logits = self._memory_logits(tuple(x.shape[2:]))
        return y_prob, y_logits, g, y_mem_logits


# ---- A2 scramble (protocol §13 ③) ----


def scramble_cascade_prior(prior, mode="spatial_roll", rng=None):
    """Destroy semantic alignment of cascade prior (A2).

    prior: [C,D,H,W]. mode: spatial_roll | channel_permute | drop | 0/none/off
    """
    import numpy as np
    if mode in (None, "", "0", "none", "off", False):
        return prior
    out = np.asarray(prior, dtype=np.float32).copy()
    rng = rng or np.random.RandomState(0)
    if mode == "drop":
        return np.zeros_like(out)
    if mode == "channel_permute" and out.shape[0] > 1:
        return out[rng.permutation(out.shape[0])]
    if mode == "spatial_roll":
        for ax in range(1, out.ndim):
            shift = int(rng.randint(0, max(out.shape[ax], 1)))
            out = np.roll(out, shift, axis=ax)
        return out
    raise ValueError(f"Unknown prior_scramble mode: {mode}")

