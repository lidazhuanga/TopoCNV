"""ROI cascade (protocol v0.4 §4, V4_PLAN §4.1).

Stage-1 landmarks (argmax in {class 1, class 4}) -> split at the head midline (from the CT, not from
the landmarks) -> per side, drop far false positives by growing a cluster from the largest component
(components of >= min_vox voxels within link_mm of the cluster join it) -> per-side bounding box,
expanded by m mm. The ROI is the union of the two side boxes. A side without landmarks gets the other
side's box mirrored about the midline; with no landmarks at all there is no ROI (whole volume).
"""
import numpy as np
from scipy import ndimage as ndi

from .common import LR_AXIS, MARGINS_MM, SPACING, expand_box, mask_bbox

BODY_THR = 0.3  # percentile-normalised CT: air ~0, soft tissue ~0.63
MIN_VOX = 10
LINK_MM = 30.0
_CONN26 = np.ones((3, 3, 3), dtype=bool)


def head_midline(image, z_range=None):
    """Left-right centre (axis-1 index) of the body mask, optionally over a z slab."""
    img = np.asarray(image if z_range is None else image[z_range[0]:z_range[1]])
    prof = (img > BODY_THR).sum(axis=(0, 2)).astype(np.float64)
    if prof.sum() == 0:
        return image.shape[LR_AXIS] / 2.0
    return float((prof * np.arange(prof.size)).sum() / prof.sum())


def side_slices(shape, mid):
    """Axis-1 index ranges of side 0 (lower indices) and side 1."""
    cut = int(round(mid))
    cut = min(max(cut, 0), shape[LR_AXIS])
    return [(0, cut), (cut, shape[LR_AXIS])]


def clean_cluster(mask, spacing=SPACING, min_vox=MIN_VOX, link_mm=LINK_MM):
    """Largest component plus every component (>= min_vox) chained to it within link_mm."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    bb = mask_bbox(mask)
    if bb is None:
        return out
    sl = tuple(slice(b0, b1) for b0, b1 in bb)
    sub = mask[sl]
    lab, n = ndi.label(sub, structure=_CONN26)
    sizes = ndi.sum(sub, lab, index=np.arange(1, n + 1))
    ids = [i + 1 for i in range(n) if sizes[i] >= min_vox]
    if not ids:
        return out
    seed = int(np.argmax(sizes)) + 1
    cluster = lab == seed
    remaining = [i for i in ids if i != seed]
    while remaining:
        dist = ndi.distance_transform_edt(~cluster, sampling=spacing)
        dmin = ndi.minimum(dist, lab, index=remaining)
        add = [i for i, d in zip(remaining, np.atleast_1d(dmin)) if d <= link_mm]
        if not add:
            break
        cluster |= np.isin(lab, add)
        remaining = [i for i in remaining if i not in add]
    out[sl] = cluster
    return out


def margin_vox(m_mm, spacing=SPACING):
    return [int(np.ceil(m_mm / s)) for s in spacing]


def mirror_box(box, mid, shape):
    """Mirror a box about the axis-1 midline (voxel centres i -> 2*mid - i)."""
    b = [list(x) for x in box]
    lo, hi = b[LR_AXIS]
    new_lo = int(np.floor(2 * mid - (hi - 1)))
    new_hi = int(np.ceil(2 * mid - lo)) + 1
    b[LR_AXIS] = [max(new_lo, 0), min(new_hi, shape[LR_AXIS])]
    return b


def landmark_mask_from_probs(p1, p4):
    """argmax over (background, class 1, class 4) from probabilities in [0, 1]."""
    p1 = np.asarray(p1, dtype=np.float32)
    p4 = np.asarray(p4, dtype=np.float32)
    p0 = 1.0 - p1 - p4
    is1 = (p1 > p0) & (p1 >= p4)
    is4 = (p4 > p0) & (p4 > p1)
    return is1 | is4, is4


def compute_case_roi(landmarks, is_c4, image, spacing=SPACING, margins=MARGINS_MM,
                     min_vox=MIN_VOX, link_mm=LINK_MM):
    """Per-case ROI record plus the cleaned landmark / class-4 masks (used by the priors)."""
    shape = landmarks.shape
    zz = np.nonzero(landmarks.any(axis=(1, 2)))[0]
    z_range = (int(zz.min()), int(zz.max()) + 1) if zz.size else None
    mid = head_midline(image, z_range)
    clean = np.zeros(shape, dtype=bool)
    sides = []
    for s, (a0, a1) in enumerate(side_slices(shape, mid)):
        part = clean_cluster(landmarks[:, a0:a1], spacing, min_vox, link_mm)
        clean[:, a0:a1] = part
        n_raw = int(landmarks[:, a0:a1].sum())
        bb = mask_bbox(part)
        if bb is not None:
            bb[LR_AXIS] = [bb[LR_AXIS][0] + a0, bb[LR_AXIS][1] + a0]
        sides.append({"side": s, "n_raw": n_raw, "n_clean": int(part.sum()), "bbox": bb,
                      "source": "own" if bb is not None else "none"})
    for s in (0, 1):
        o = sides[1 - s]
        if sides[s]["bbox"] is None and o["source"] == "own":
            sides[s]["bbox"] = mirror_box(o["bbox"], mid, shape)
            sides[s]["source"] = "mirror"
    for sd in sides:
        sd["boxes"] = {str(m): (expand_box(sd["bbox"], margin_vox(m, spacing), shape)
                                if sd["bbox"] is not None else None) for m in margins}
    rec = {"shape": list(shape), "midline": mid, "z_range": z_range, "sides": sides,
           "roi_found": any(sd["bbox"] is not None for sd in sides),
           "params": {"min_vox": min_vox, "link_mm": link_mm, "body_thr": BODY_THR}}
    return rec, clean, clean & is_c4


def roi_boxes(rec, m):
    """Side boxes for margin m; [] means no ROI (use the whole volume)."""
    return [sd["boxes"][str(m)] for sd in rec["sides"] if sd["boxes"][str(m)] is not None]


def coverage(rec, m, gt_y):
    """Fraction of GT Y voxels inside the ROI, and ROI volume fraction."""
    from .common import boxes_mask
    shape = gt_y.shape
    boxes = roi_boxes(rec, m)
    full = [[0, n] for n in shape]
    roi = boxes_mask(boxes, full) if boxes else np.ones(shape, dtype=bool)
    n_y = int(gt_y.sum())
    cov = float((gt_y & roi).sum() / n_y) if n_y else float("nan")
    return cov, float(roi.mean())
