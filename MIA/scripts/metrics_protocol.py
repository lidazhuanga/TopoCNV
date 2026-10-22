#!/usr/bin/env python3
"""
CNV protocol v0.3-confirmed evaluation helpers (Phase 1).

Primary mask Y = labels in {1,2,3,4}.  # CNV protocol v0.3
Primary metrics: Dice_Y + HD95_Y (optional ASSD_Y).
L1: Dice_L, Precision_L, Recall_L, |L|/|Y| (+ Dice_L_legacy, Dice_H for audit).
Secondary: Dice_123, Dice_c4.
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


def score_protocol(
    pred: ArrayLike,
    gt_labels: ArrayLike,
    hu: Optional[ArrayLike] = None,
    spacing_zyx: Sequence[float] = DEFAULT_SPACING_ZYX,
    *,
    pred_binary_Y: bool = False,
    secondary_pred: Optional[ArrayLike] = None,
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

    Secondary policy (protocol §13.1 / methods):
      - binary-only Primary (B0 / SOTA / S2 binary_y) without a multiclass
        secondary source → Dice_123/Dice_c4 = nan, Secondary_source = "N/A"
      - cascade arms: pass Stage-1 **model** multiclass pred/argmax as secondary_pred
        → Secondary_source = "s1"  (FORBIDDEN: never pass GT labels as secondary_pred)
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
        out["Secondary_source"] = "s1"  # type: ignore[assignment]
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


if __name__ == "__main__":
    _self_test()
