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
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topocnv.losses.evidence_losses import (
    BinaryYLoss,
    DEFAULT_BONE_NORM_THR,
    DEFAULT_EVIDENCE_SMOOTH,
    DEFAULT_EVIDENCE_TEMP,
    DEFAULT_GATE_ALPHA_FIXED,
    DEFAULT_GATE_TAU,
    DEFAULT_LAMBDA_L,
    DEFAULT_LAMBDA_MEM,
    DEFAULT_MEM_SUP_WEIGHT,
    EvidenceGatedYLoss,
)

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


# ---- Claim-M★ evidence map and gate (protocol §2) ----
# Loss classes and their defaults are re-exported from topocnv.losses.


from topocnv.models.evidence_gate import BinaryYStage2Model, evidence_map_g1


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

