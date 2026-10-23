#!/usr/bin/env python3
"""
CNV protocol v0.3-confirmed evaluation helpers (Phase 1).

Primary mask Y = labels in {1,2,3,4}.  # CNV protocol v0.3
Primary metrics: Dice_Y + HD95_Y (optional ASSD_Y).
L1: Dice_L, Precision_L, Recall_L, |L|/|Y| (+ Dice_L_legacy, Dice_H for audit).
v0.4 (protocol v0.4-draft §6): SurfDice_Y (surface Dice @ 2 mm on Y);
  invisible segment L_vis = GT classes 2∪3 → Recall_vis = |P∩L_vis|/|L_vis|,
  Dice_vis = Dice(P∩R_vis, Y∩R_vis), R_vis = dilate(L_vis, 3 mm), Lvis_over_Y;
  optional visibility-head output (0..4) → Dice_tier1..Dice_tier4.
Secondary: Dice_123, Dice_c4. Binary-Y cascade arms: Stage-2 Y split by Stage-1
posteriors (partition_Y_by_posterior, source "s1_inY").
Secondary_source: "pred" | "s1" | "N/A" (binary-only without multiclass source).
FORBIDDEN as Primary: 5-class mean / soft-tree mean / class4-only.
class5 = bypass_roi / class5_bypass
(绕行/肌肉) is NOT in Y and NOT in L1.

L1 (low-evidence subset):
  CNV_GT = Y
  H_bone = bone_window_high_contrast(raw_HU) ∪ dilate(bony_roi=class4, r=3mm)
  L      = CNV_GT \\ H_bone
  Scoring (v2, 2026-10-21; L itself unchanged) — both masks restricted to ¬H_bone:
    Dice_L        = Dice(P \\ H_bone, L)                 counts FP outside H_bone
    Precision_L   = |P ∩ L| / |P \\ H_bone|
    Recall_L      = |P ∩ L| / |L|
    Dice_L_legacy = Dice(P ∩ L, L) = 2·Recall_L / (1 + Recall_L)   (v1, FN-only)
    Dice_H        = Dice(P ∩ H_bone, Y ∩ H_bone)          complementary subset
  P = predicted Y. v1 reported Dice_L_legacy under the name Dice_L.

HU SOURCE (CRITICAL):
  Preprocess cache (`*_image.npy`) is percentile-normalized to [0,1] — NEVER
  use it for bone-window / HU thresholds.
  Raw CT NIfTI lives at:
    {DATA_ROOT}/image/{case_id}.nii.gz
  default DATA_ROOT = <repo>/npc_data/train (override with CNV_DATA_ROOT)
  Load with nibabel get_fdata(), transpose (2,0,1) to (D,H,W) to match cache
  labels, then optionally resample to target_spacing (order=1 for HU).

Bone-window thresholds (frozen defaults for P0; document any change):
  BONE_HU_LOW  = 200   # cortical / dense bone floor (HU)
  BONE_HU_HIGH = 3000  # clip metal / extreme outliers
  High-contrast proxy: voxels with HU in [BONE_HU_LOW, BONE_HU_HIGH].
  (Local-gradient enrichment optional via bone_window_high_contrast(..., use_grad=True).)

Naming lock (cnv-实验):
  bony_roi      ≡ class4 only
  bypass_roi    ≡ class5 (FORBIDDEN: bony_* for class5)
  class5_bypass ≡ alias of class5

This module is additive / CPU-safe; no training.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# ---- frozen defaults (protocol §4.2 / naming lock) ----
Y_LABELS: Tuple[int, ...] = (1, 2, 3, 4)
BONY_ROI_LABEL: int = 4          # bony_roi ≡ class4 only
BYPASS_ROI_LABEL: int = 5        # class5_bypass / bypass_roi — not in Y / L1
L1_DILATE_R_MM: float = 3.0
BONE_HU_LOW: float = 200.0
BONE_HU_HIGH: float = 3000.0
SURF_TOL_MM: float = 2.0         # surface Dice tolerance (v0.4)
VIS_LABELS: Tuple[int, ...] = (2, 3)   # invisible segment: MRI-guided ∪ possible course
VIS_DILATE_R_MM: float = 3.0
TIER_LABELS: Tuple[int, ...] = (1, 2, 3, 4)
V04_KEYS: Tuple[str, ...] = ("SurfDice_Y", "Recall_vis", "Dice_vis", "Lvis_over_Y") + tuple(
    f"Dice_tier{k}" for k in TIER_LABELS)
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "CNV_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "npc_data" / "train")))
DEFAULT_SPACING_ZYX: Tuple[float, float, float] = (2.5, 1.1328120231628418, 1.1328120231628418)


ArrayLike = Union[np.ndarray]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def mask_Y(labels: ArrayLike) -> np.ndarray:
    """Binary mask: labels ∈ {1,2,3,4}.  Excludes class5_bypass."""
    lab = np.asarray(labels)
    return (lab >= 1) & (lab <= 4)


def mask_123(labels: ArrayLike) -> np.ndarray:
    """Binary mask: labels ∈ {1,2,3}."""
    lab = np.asarray(labels)
    return (lab >= 1) & (lab <= 3)


def mask_c4(labels: ArrayLike) -> np.ndarray:
    """Binary mask: bony_roi ≡ class4 only."""
    return np.asarray(labels) == BONY_ROI_LABEL


def mask_bypass(labels: ArrayLike) -> np.ndarray:
    """Binary mask: class5_bypass / bypass_roi (绕行). Not in Y/L1."""
    return np.asarray(labels) == BYPASS_ROI_LABEL


def pred_to_Y(pred: ArrayLike, *, binary: bool = False) -> np.ndarray:
    """Convert prediction to Y mask.

    If binary=True, treat pred as already a Y probability/logit mask (>0.5).
    Else assume multiclass integer labels and merge {1,2,3,4}.
    """
    p = np.asarray(pred)
    if binary:
        if p.dtype == np.bool_ or p.max() <= 1.0:
            return p > 0.5
        return p > 0
    return mask_Y(p)


# ---------------------------------------------------------------------------
# Geometry / bone window
# ---------------------------------------------------------------------------
def dilate_mm(
    mask: ArrayLike,
    spacing_zyx: Sequence[float],
    r_mm: float = L1_DILATE_R_MM,
) -> np.ndarray:
    """Anisotropic binary dilation by physical radius r_mm.

    Ball radius in voxels = r_mm / spacing per axis (ceil).
    Uses scipy.ndimage if available; else a separable approx via iterate max-pool.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any() or r_mm <= 0:
        return m.copy()
    rz = max(1, int(np.ceil(r_mm / float(spacing_zyx[0]))))
    ry = max(1, int(np.ceil(r_mm / float(spacing_zyx[1]))))
    rx = max(1, int(np.ceil(r_mm / float(spacing_zyx[2]))))
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure

        # Approximate ellipsoid via iterative structure of size proportional to r
        # Build an ellipsoid footprint
        zz, yy, xx = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
        footprint = (zz / max(rz, 1)) ** 2 + (yy / max(ry, 1)) ** 2 + (xx / max(rx, 1)) ** 2 <= 1.0
        return binary_dilation(m, structure=footprint)
    except ImportError:
        # Fallback: axis-separated box dilation (slightly larger than ball)
        out = m.copy()
        # Simple iterative 3x3x3 expansion rz/ry/rx times (crude but dependency-free)
        n_iter = max(rz, ry, rx)
        kernel_iters = n_iter
        for _ in range(kernel_iters):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            # 3x3x3 OR
            acc = padded[1:-1, 1:-1, 1:-1].copy()
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dz == dy == dx == 0:
                            continue
                        acc |= padded[1 + dz : out.shape[0] + 1 + dz,
                                      1 + dy : out.shape[1] + 1 + dy,
                                      1 + dx : out.shape[2] + 1 + dx]
            out = acc
        return out


def bone_window_high_contrast(
    hu: ArrayLike,
    hu_low: float = BONE_HU_LOW,
    hu_high: float = BONE_HU_HIGH,
    *,
    use_grad: bool = False,
    grad_percentile: float = 90.0,
) -> np.ndarray:
    """Bone-window high-contrast voxels from raw HU (not cache [0,1]).

    Default: HU ∈ [hu_low, hu_high].
    Optional: intersect with high local |grad| (percentile threshold).
    """
    x = np.asarray(hu, dtype=np.float32)
    bone = (x >= hu_low) & (x <= hu_high)
    if not use_grad or not bone.any():
        return bone
    # Simple finite-difference magnitude
    gz = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx = np.zeros_like(x)
    gz[1:-1] = x[2:] - x[:-2]
    gy[:, 1:-1] = x[:, 2:] - x[:, :-2]
    gx[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]
    g = np.sqrt(gz * gz + gy * gy + gx * gx)
    thr = np.percentile(g[bone], grad_percentile) if bone.any() else 0.0
    return bone & (g >= thr)


def build_H_bone(
    hu: ArrayLike,
    labels: ArrayLike,
    spacing_zyx: Sequence[float],
    r_mm: float = L1_DILATE_R_MM,
    *,
    hu_low: float = BONE_HU_LOW,
    hu_high: float = BONE_HU_HIGH,
) -> np.ndarray:
    """H_bone = bone_window_high_contrast(HU) ∪ dilate(bony_roi=class4, r_mm).

    class5_bypass is intentionally excluded from the dilate base.
    """
    bw = bone_window_high_contrast(hu, hu_low=hu_low, hu_high=hu_high)
    c4 = mask_c4(labels)
    c4_dil = dilate_mm(c4, spacing_zyx, r_mm=r_mm)
    return bw | c4_dil


def build_L1(
    labels: ArrayLike,
    hu: ArrayLike,
    spacing_zyx: Sequence[float],
    r_mm: float = L1_DILATE_R_MM,
    **bone_kw,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Y, H_bone, L) where L = Y \\ H_bone. class5_bypass never enters."""
    Y = mask_Y(labels)
    H = build_H_bone(hu, labels, spacing_zyx, r_mm=r_mm, **bone_kw)
    L = Y & ~H
    return Y, H, L


# ---------------------------------------------------------------------------
# Dice / recall
# ---------------------------------------------------------------------------
def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def _recall(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    n = gt.sum()
    if n == 0:
        return float("nan")
    return float(tp / n)


def _precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    n = pred.sum()
    if n == 0:
        return float("nan")
    return float((pred & gt).sum() / n)


def _hd95(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Sequence[float]) -> float:
    """95th-percentile Hausdorff distance (mm). NaN if either empty."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() or not gt.any():
        return float("nan")
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        return float("nan")
    sp = tuple(float(s) for s in spacing_zyx)

    def _boundary(m):
        return m & ~binary_erosion(m, iterations=1, border_value=0)

    b_gt, b_pred = _boundary(gt), _boundary(pred)
    dt_pred = distance_transform_edt(~pred, sampling=sp)
    dt_gt = distance_transform_edt(~gt, sampling=sp)
    if not b_gt.any() or not b_pred.any():
        d1, d2 = dt_pred[gt], dt_gt[pred]
    else:
        d1, d2 = dt_pred[b_gt], dt_gt[b_pred]
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def _assd(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Sequence[float]) -> float:
    """Average symmetric surface distance (mm). Optional."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() or not gt.any():
        return float("nan")
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        return float("nan")
    sp = tuple(float(s) for s in spacing_zyx)

    def _boundary(m):
        return m & ~binary_erosion(m, iterations=1, border_value=0)

    b_gt, b_pred = _boundary(gt), _boundary(pred)
    if not b_gt.any() or not b_pred.any():
        return float("nan")
    dt_pred = distance_transform_edt(~pred, sampling=sp)
    dt_gt = distance_transform_edt(~gt, sampling=sp)
    return float(0.5 * (dt_pred[b_gt].mean() + dt_gt[b_pred].mean()))


def partition_Y_by_posterior(pred_Y: ArrayLike, s1_softmax: ArrayLike) -> np.ndarray:
    """Secondary labels for a binary-Y cascade: split the Stage-2 Y by Stage-1 posteriors.

    Inside pred_Y each voxel gets argmax_{c in 1..4} P_S1(c); outside it, 0. The union of
    classes 1..4 therefore equals pred_Y exactly. s1_softmax is [C>=5, D, H, W] (may be an
    mmap; only the bounding box of pred_Y is read).
    """
    pY = np.asarray(pred_Y, dtype=bool)
    out = np.zeros(pY.shape, dtype=np.int16)
    if not pY.any():
        return out
    idx = np.argwhere(pY)
    bb = tuple(slice(a, b + 1) for a, b in zip(idx.min(0), idx.max(0)))
    sub = np.asarray(s1_softmax[(slice(1, 5),) + bb])
    out[bb] = np.where(pY[bb], np.argmax(sub, axis=0) + 1, 0)
    return out


def _surface_dice(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Sequence[float],
                  tol_mm: float = SURF_TOL_MM) -> float:
    """Surface Dice at tolerance tol_mm (Nikolov et al. 2021), voxel-boundary version.

    Boundaries are the mask voxels removed by one 6-connected erosion (same as HD95).
    SD = (#∂P within tol of ∂G + #∂G within tol of ∂P) / (#∂P + #∂G).
    Both empty → nan; exactly one empty → 0.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return float("nan")
    if not pred.any() or not gt.any():
        return 0.0
    from scipy.ndimage import binary_erosion, distance_transform_edt

    sp = tuple(float(s) for s in spacing_zyx)
    # crop to the union bounding box plus the tolerance margin (distances beyond tol are irrelevant)
    idx = np.argwhere(pred | gt)
    pad = [int(np.ceil(tol_mm / s)) + 2 for s in sp]
    bb = tuple(slice(max(lo - m, 0), min(hi + m + 1, n))
               for lo, hi, m, n in zip(idx.min(0), idx.max(0), pad, pred.shape))
    p, g = pred[bb], gt[bb]
    bp = p & ~binary_erosion(p, iterations=1, border_value=0)
    bg = g & ~binary_erosion(g, iterations=1, border_value=0)
    d_to_g = distance_transform_edt(~bg, sampling=sp)
    d_to_p = distance_transform_edt(~bp, sampling=sp)
    hit = (d_to_g[bp] <= tol_mm).sum() + (d_to_p[bg] <= tol_mm).sum()
    return float(hit / (bp.sum() + bg.sum()))


def score_v04(
    pY: np.ndarray,
    gt: np.ndarray,
    spacing_zyx: Sequence[float],
    vis_pred: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """v0.4 metrics: SurfDice_Y, Recall_vis, Dice_vis, Lvis_over_Y, Dice_tier1..4.

    R_vis depends only on the reference, so it is the same for every method compared.
    Dice_tier* are nan unless vis_pred (visibility-head labels 0..4) is given.
    """
    gY = mask_Y(gt)
    out: Dict[str, float] = {"SurfDice_Y": _surface_dice(pY, gY, spacing_zyx)}
    L_vis = np.isin(gt, VIS_LABELS)
    if L_vis.any():
        R_vis = dilate_mm(L_vis, spacing_zyx, r_mm=VIS_DILATE_R_MM)
        out["Recall_vis"] = _recall(pY, L_vis)
        out["Dice_vis"] = _dice(pY & R_vis, gY & R_vis)
        out["Lvis_over_Y"] = float(L_vis.sum() / gY.sum())
    else:
        out["Recall_vis"] = out["Dice_vis"] = out["Lvis_over_Y"] = float("nan")
    v = None if vis_pred is None else np.asarray(vis_pred)
    for k in TIER_LABELS:
        out[f"Dice_tier{k}"] = float("nan") if v is None else _dice(v == k, gt == k)
    return out


def score_protocol(
    pred: ArrayLike,
    gt_labels: ArrayLike,
    hu: Optional[ArrayLike] = None,
    spacing_zyx: Sequence[float] = DEFAULT_SPACING_ZYX,
    *,
    pred_binary_Y: bool = False,
    secondary_pred: Optional[ArrayLike] = None,
    secondary_source: str = "s1",
    vis_pred: Optional[ArrayLike] = None,
    r_mm: float = L1_DILATE_R_MM,
    hu_low: float = BONE_HU_LOW,
    hu_high: float = BONE_HU_HIGH,
) -> Dict[str, float]:
    """Compute protocol metrics from pred / gt arrays.

    Returns dict with:
      Primary: Dice_Y, HD95_Y (ASSD_Y optional)
      Secondary: Dice_123, Dice_c4 (+ Secondary_source)
      L1: Dice_L, Precision_L, Recall_L, L_over_Y, Dice_L_legacy, Dice_H
          (nan if hu is None)
      v0.4: SurfDice_Y, Recall_vis, Dice_vis, Lvis_over_Y, Dice_tier1..4 (see score_v04)

    Secondary policy (protocol §13.1 / methods):
      - binary-only Primary (B0 / SOTA / S2 binary_y) without a multiclass
        secondary source → Dice_123/Dice_c4 = nan, Secondary_source = "N/A"
      - cascade arms (main table, decided 2026-10-22): secondary_pred =
        partition_Y_by_posterior(Stage-2 Y, Stage-1 softmax), secondary_source = "s1_inY";
        a raw Stage-1 argmax with the default "s1" is kept for audit only
        (FORBIDDEN: never pass GT labels as secondary_pred)
      - multiclass Primary **model** pred → Secondary from pred, Secondary_source = "pred"
        (FORBIDDEN: GT masquerading as prediction)
    """
    gt = np.asarray(gt_labels)
    pY = pred_to_Y(pred, binary=pred_binary_Y)
    gY = mask_Y(gt)
    out: Dict[str, float] = {"Dice_Y": _dice(pY, gY)}
    out["HD95_Y"] = _hd95(pY, gY, spacing_zyx)
    out["ASSD_Y"] = _assd(pY, gY, spacing_zyx)
    if secondary_pred is not None:
        sp = np.asarray(secondary_pred)
        out["Dice_123"] = _dice(mask_123(sp), mask_123(gt))
        out["Dice_c4"] = _dice(mask_c4(sp), mask_c4(gt))
        out["Secondary_source"] = secondary_source  # type: ignore[assignment]
    elif pred_binary_Y:
        # binary-only arm: Secondary explicitly N/A (nan for numeric aggregation)
        out["Dice_123"] = float("nan")
        out["Dice_c4"] = float("nan")
        out["Secondary_source"] = "N/A"  # type: ignore[assignment]
    else:
        p = np.asarray(pred)
        out["Dice_123"] = _dice(mask_123(p), mask_123(gt))
        out["Dice_c4"] = _dice(mask_c4(p), mask_c4(gt))
        out["Secondary_source"] = "pred"  # type: ignore[assignment]
    out.update(score_v04(pY, gt, spacing_zyx, vis_pred=vis_pred))

    if hu is None:
        for k in ("Dice_L", "Precision_L", "Recall_L", "L_over_Y", "Dice_L_legacy", "Dice_H"):
            out[k] = float("nan")
        return out

    Y, H, L = build_L1(gt, hu, spacing_zyx, r_mm=r_mm, hu_low=hu_low, hu_high=hu_high)
    # Both masks restricted to the low-evidence complement ¬H: L = Y \ H on the GT
    # side, P \ H on the prediction side, so FP outside H_bone count against Dice_L.
    pred_L = pY & ~H
    out["Dice_L"] = _dice(pred_L, L)
    out["Precision_L"] = _precision(pred_L, L)
    out["Recall_L"] = _recall(pY, L)
    # v1 score: prediction clipped to L itself, so FP never count (= 2R/(1+R)).
    out["Dice_L_legacy"] = _dice(pY & L, L)
    out["Dice_H"] = _dice(pY & H, Y & H)
    y_n = int(Y.sum())
    out["L_over_Y"] = float(L.sum() / y_n) if y_n > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Raw HU I/O
# ---------------------------------------------------------------------------
def load_raw_hu_dhw(
    case_id: str,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    target_spacing_zyx: Optional[Sequence[float]] = None,
    reference_shape_dhw: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load raw CT HU as (D,H,W) float32 matching cache orientation.

    Returns (hu_dhw, orig_spacing_zyx).
    If target_spacing_zyx is set, resamples HU (order=1) to that spacing.
    If reference_shape_dhw is set after resample, crop/pad to match.
    """
    import nibabel as nib

    data_root = Path(data_root)
    ct_path = data_root / "image" / f"{case_id}.nii.gz"
    if not ct_path.exists():
        ct_path = data_root / "image" / f"{case_id}.nii"
    if not ct_path.exists():
        raise FileNotFoundError(f"Raw CT not found for {case_id} under {data_root}/image/")

    nii = nib.load(str(ct_path))
    hu = nii.get_fdata().astype(np.float32)
    hu = np.transpose(hu, (2, 0, 1))  # (H,W,D)->(D,H,W) — matches trainer cache
    zooms = nii.header.get_zooms()[:3]
    orig_sp = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

    if target_spacing_zyx is not None:
        try:
            import npc_preprocess as pp
            hu = pp.resample_image_to_spacing(
                hu, orig_sp, tuple(target_spacing_zyx), order=1
            ).astype(np.float32)
            spacing = tuple(float(s) for s in target_spacing_zyx)
        except Exception:
            # Fallback nearest-neighbor zoom
            from scipy.ndimage import zoom
            factors = [orig_sp[i] / float(target_spacing_zyx[i]) for i in range(3)]
            hu = zoom(hu, factors, order=1).astype(np.float32)
            spacing = tuple(float(s) for s in target_spacing_zyx)
    else:
        spacing = orig_sp

    if reference_shape_dhw is not None and hu.shape != tuple(reference_shape_dhw):
        out = np.zeros(reference_shape_dhw, dtype=np.float32)
        slices = tuple(slice(0, min(hu.shape[i], reference_shape_dhw[i])) for i in range(3))
        out[slices] = hu[slices]
        hu = out
    return hu, spacing


def score_case_from_paths(
    case_id: str,
    pred: ArrayLike,
    gt_labels: ArrayLike,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    spacing_zyx: Sequence[float] = DEFAULT_SPACING_ZYX,
    *,
    pred_binary_Y: bool = False,
    secondary_pred: Optional[ArrayLike] = None,
    secondary_source: str = "s1",
    vis_pred: Optional[ArrayLike] = None,
    load_hu: bool = True,
) -> Dict[str, float]:
    """Score one case; optionally load raw HU for L1.

    secondary_pred: optional multiclass labels (typically S1 argmax) used for
    Dice_123 / Dice_c4 when Primary pred is binary Y (cascade arms).
    """
    hu = None
    sp = tuple(spacing_zyx)
    if load_hu:
        hu, sp = load_raw_hu_dhw(
            case_id,
            data_root=data_root,
            target_spacing_zyx=spacing_zyx,
            reference_shape_dhw=tuple(np.asarray(gt_labels).shape),
        )
    row = score_protocol(
        pred, gt_labels, hu=hu, spacing_zyx=sp,
        pred_binary_Y=pred_binary_Y, secondary_pred=secondary_pred,
        secondary_source=secondary_source, vis_pred=vis_pred,
    )
    row["case_id"] = case_id  # type: ignore[assignment]
    return row


# ---------------------------------------------------------------------------
# Tiny self-check (CPU, no GPU)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.RandomState(0)
    gt = np.zeros((20, 32, 32), dtype=np.int16)
    gt[5:8, 10:14, 10:14] = 1
    gt[8:10, 12:16, 12:16] = 2
    gt[10:12, 14:18, 14:18] = 3
    gt[12:15, 8:20, 8:20] = 4   # bony_roi
    gt[16:18, 20:24, 20:24] = 5  # class5_bypass — must not enter Y
    assert mask_Y(gt).sum() == (gt >= 1).sum() - mask_bypass(gt).sum()
    assert not (mask_Y(gt) & mask_bypass(gt)).any()

    hu = rng.randn(*gt.shape).astype(np.float32) * 50
    hu[mask_c4(gt)] = 800  # bone-like
    Y, H, L = build_L1(gt, hu, DEFAULT_SPACING_ZYX, r_mm=3.0)
    assert (L & mask_bypass(gt)).sum() == 0
    assert (Y & mask_bypass(gt)).sum() == 0

    pred = gt.copy()
    pred[mask_bypass(gt)] = 0
    m = score_protocol(pred, gt, hu=hu, spacing_zyx=DEFAULT_SPACING_ZYX)
    assert m["Dice_Y"] > 0.99
    assert m.get("Secondary_source") == "pred"
    assert "Dice_L" in m and "Recall_L" in m and "L_over_Y" in m
    assert "HD95_Y" in m and "ASSD_Y" in m

    # binary-only → Secondary N/A
    m_bin = score_protocol(mask_Y(pred), gt, hu=hu, spacing_zyx=DEFAULT_SPACING_ZYX,
                           pred_binary_Y=True)
    assert m_bin["Secondary_source"] == "N/A"
    assert m_bin["Dice_123"] != m_bin["Dice_123"]  # nan

    # cascade: Primary binary Y + S1 multiclass secondary
    m_cas = score_protocol(mask_Y(pred), gt, hu=hu, spacing_zyx=DEFAULT_SPACING_ZYX,
                           pred_binary_Y=True, secondary_pred=pred)
    assert m_cas["Secondary_source"] == "s1"
    assert m_cas["Dice_123"] > 0.99 and m_cas["Dice_c4"] > 0.99

    _self_test_L1(gt, hu)
    _self_test_partition(gt)
    _self_test_v04(gt)

    print("[metrics_protocol] self-test OK", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


def _self_test_L1(gt: np.ndarray, hu: np.ndarray) -> None:
    """L1 v2 scoring: FP outside H_bone lower Dice_L; FP inside H_bone do not."""
    sp = DEFAULT_SPACING_ZYX
    Y, H, L = build_L1(gt, hu, sp, r_mm=3.0)
    assert L.any() and (Y & H).any(), "toy case must have both L and Y∩H voxels"

    def run(p):
        return score_protocol(p.astype(np.uint8), gt, hu=hu, spacing_zyx=sp, pred_binary_Y=True)

    close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731

    # 1) perfect prediction
    m = run(Y)
    for k in ("Dice_L", "Precision_L", "Recall_L", "Dice_L_legacy", "Dice_H"):
        assert close(m[k], 1.0), (k, m[k])

    # 2) FP blob in soft tissue (outside H): only the new Dice_L / Precision_L see it
    hu2 = hu.copy()
    blob = np.zeros_like(Y)
    blob[1:4, 24:30, 1:7] = True
    hu2[blob] = 40.0
    assert not (blob & Y).any()
    Y2, H2, L2 = build_L1(gt, hu2, sp, r_mm=3.0)
    assert not (blob & H2).any()
    m = score_protocol((Y | blob).astype(np.uint8), gt, hu=hu2, spacing_zyx=sp, pred_binary_Y=True)
    n_l, n_fp = int(L2.sum()), int(blob.sum())
    assert close(m["Recall_L"], 1.0) and close(m["Dice_L_legacy"], 1.0)
    assert close(m["Precision_L"], n_l / (n_l + n_fp)), m["Precision_L"]
    assert close(m["Dice_L"], 2 * n_l / (2 * n_l + n_fp)), m["Dice_L"]
    assert close(m["Dice_H"], 1.0)

    # 3) FP inside H_bone (dilated class4 rim, not in Y): Dice_L untouched, Dice_H drops
    rim = H & ~Y
    assert rim.any()
    m = run(Y | rim)
    assert close(m["Dice_L"], 1.0) and close(m["Precision_L"], 1.0)
    assert m["Dice_H"] < 1.0 and m["Dice_Y"] < 1.0

    # 4) half of L missed, no FP: new Dice_L equals legacy = 2R/(1+R)
    miss = L.copy()
    idx = np.flatnonzero(miss)
    miss.flat[idx[: len(idx) // 2]] = False   # keep second half of L
    p = (Y & ~L) | miss
    m = run(p)
    r = miss.sum() / L.sum()
    assert close(m["Recall_L"], r)
    assert close(m["Dice_L_legacy"], 2 * r / (1 + r))
    assert close(m["Dice_L"], m["Dice_L_legacy"]) and close(m["Precision_L"], 1.0)

    # 5) legacy identity holds with FP too; empty prediction gives 0 / nan
    m = run(Y | blob)
    assert close(m["Dice_L_legacy"], 2 * m["Recall_L"] / (1 + m["Recall_L"]))
    m = run(np.zeros_like(Y))
    assert m["Dice_L"] == 0.0 and m["Recall_L"] == 0.0
    assert m["Precision_L"] != m["Precision_L"]  # nan: no predicted voxels outside H

    # 6) no HU → all L1 fields nan
    m = score_protocol(Y.astype(np.uint8), gt, hu=None, pred_binary_Y=True)
    assert all(m[k] != m[k] for k in ("Dice_L", "Precision_L", "Recall_L", "Dice_L_legacy", "Dice_H"))
    print("[metrics_protocol] L1 v2 self-test OK")


def _self_test_partition(gt: np.ndarray) -> None:
    """A_inY Secondary: classes follow the Stage-1 posterior, union equals the Stage-2 Y."""
    onehot = np.stack([(gt == c) for c in range(6)]).astype(np.float32)
    Y = mask_Y(gt)
    # Stage-1 posterior = GT one-hot; Stage-2 Y = GT Y → perfect Secondary
    lab = partition_Y_by_posterior(Y, onehot)
    assert np.array_equal(lab > 0, Y) and np.array_equal(lab[Y], gt[Y])
    m = score_protocol(Y.astype(np.uint8), gt, pred_binary_Y=True, secondary_pred=lab,
                       secondary_source="s1_inY")
    assert m["Secondary_source"] == "s1_inY" and m["Dice_123"] == 1.0 and m["Dice_c4"] == 1.0
    # Stage-1 says background/class5 everywhere: voxels in Y still get a class in 1..4
    bg = np.zeros_like(onehot)
    bg[0] = 0.9
    bg[5] = 0.1
    bg[2, Y] = 0.05
    lab = partition_Y_by_posterior(Y, bg)
    assert np.array_equal(lab > 0, Y) and set(np.unique(lab[Y])) == {2}
    # Extra Stage-2 voxels outside the GT are labelled, voxels outside the Stage-2 Y are not
    p = Y.copy()
    p[0, 0, 0] = True
    lab = partition_Y_by_posterior(p, onehot)
    assert np.array_equal(lab > 0, p)
    assert not partition_Y_by_posterior(np.zeros_like(Y), onehot).any()
    print("[metrics_protocol] A_inY partition self-test OK")


def _self_test_v04(gt: np.ndarray) -> None:
    """Surface Dice @ 2 mm, invisible-segment metrics and per-tier Dice."""
    sp = DEFAULT_SPACING_ZYX            # (2.5, 1.133, 1.133) mm
    close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
    Y = mask_Y(gt)
    L_vis = np.isin(gt, VIS_LABELS)
    R_vis = dilate_mm(L_vis, sp, r_mm=VIS_DILATE_R_MM)

    def run(p, vis=None):
        return score_protocol(p.astype(np.uint8), gt, spacing_zyx=sp, pred_binary_Y=True, vis_pred=vis)

    m = run(Y)
    for k in ("SurfDice_Y", "Recall_vis", "Dice_vis"):
        assert close(m[k], 1.0), (k, m[k])
    assert close(m["Lvis_over_Y"], L_vis.sum() / Y.sum())
    assert all(m[f"Dice_tier{k}"] != m[f"Dice_tier{k}"] for k in TIER_LABELS)  # nan without vis head

    # surface Dice: 1-voxel x shift (1.13 mm <= 2 mm) stays 1; 3-voxel shift (3.4 mm) drops; empty -> 0
    assert close(run(np.roll(Y, 1, axis=2))["SurfDice_Y"], 1.0)
    sd3 = run(np.roll(Y, 3, axis=2))["SurfDice_Y"]
    assert 0.0 < sd3 < 1.0, sd3
    assert run(np.zeros_like(Y))["SurfDice_Y"] == 0.0
    nan_sd = _surface_dice(np.zeros_like(Y), np.zeros_like(Y), sp)
    assert nan_sd != nan_sd
    # one slice shift in z (2.5 mm > 2 mm): faces normal to z move out of tolerance
    assert run(np.roll(Y, 1, axis=0))["SurfDice_Y"] < 1.0

    # FP next to class 2 (inside R_vis, outside Y): Dice_vis drops, Recall_vis stays
    fp = np.zeros_like(Y)
    fp[9, 11, 13] = True
    assert R_vis[9, 11, 13] and not Y[9, 11, 13]
    m = run(Y | fp)
    n = int((Y & R_vis).sum())
    assert close(m["Recall_vis"], 1.0) and close(m["Dice_vis"], 2 * n / (2 * n + 1)), m["Dice_vis"]
    # FP far away (outside R_vis): Dice_vis unchanged, Dice_Y drops
    far = np.zeros_like(Y)
    far[1:4, 24:30, 1:7] = True
    assert not (far & R_vis).any()
    m = run(Y | far)
    assert close(m["Dice_vis"], 1.0) and m["Dice_Y"] < 1.0
    # miss class 3 entirely: Recall_vis = |class2| / |class2 ∪ class3|
    m = run(Y & ~(gt == 3))
    assert close(m["Recall_vis"], (gt == 2).sum() / L_vis.sum()) and m["Dice_vis"] < 1.0

    # visibility head: perfect labels -> every tier 1; swap tiers 2/3 -> those two drop to 0
    vis = np.where(gt == 5, 0, gt)
    m = run(Y, vis)
    assert all(close(m[f"Dice_tier{k}"], 1.0) for k in TIER_LABELS)
    sw = vis.copy()
    sw[vis == 2] = 3
    sw[vis == 3] = 2
    m = run(Y, sw)
    assert m["Dice_tier2"] == 0.0 and m["Dice_tier3"] == 0.0 and close(m["Dice_tier1"], 1.0)

    # no class 2/3 in the reference -> vis metrics nan
    g2 = np.where(np.isin(gt, VIS_LABELS), 1, gt)
    m = score_protocol(mask_Y(g2).astype(np.uint8), g2, spacing_zyx=sp, pred_binary_Y=True)
    assert m["Recall_vis"] != m["Recall_vis"] and m["Dice_vis"] != m["Dice_vis"]
    print("[metrics_protocol] v0.4 self-test OK")


if __name__ == "__main__":
    _self_test()
