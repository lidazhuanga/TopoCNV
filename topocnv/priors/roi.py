"""基于 Stage-1 路标的双侧 ROI 级联，保留 v4 协议的聚类与镜像回退。"""

import numpy as np
from scipy import ndimage as ndi

from topocnv.data.utils import LR_AXIS, MARGINS_MM, SPACING
from topocnv.utils.box_ops import boxes_mask, expand_box, mask_bbox

BODY_THR = 0.3
MIN_VOX = 10
LINK_MM = 45.0
_CONN26 = np.ones((3, 3, 3), dtype=bool)


def head_midline(image, z_range=None):
    """从 CT 身体掩膜计算左右中线，可限制在 z 轴区间内。"""
    img = np.asarray(image if z_range is None else image[z_range[0]:z_range[1]])
    prof = (img > BODY_THR).sum(axis=(0, 2)).astype(np.float64)
    if prof.sum() == 0:
        return image.shape[LR_AXIS] / 2.0
    return float((prof * np.arange(prof.size)).sum() / prof.sum())


def side_slices(shape, mid):
    """返回左右两侧在 axis-1 上的索引范围。"""
    cut = int(round(mid))
    cut = min(max(cut, 0), shape[LR_AXIS])
    return [(0, cut), (cut, shape[LR_AXIS])]


def clean_cluster(mask, spacing=SPACING, min_vox=MIN_VOX, link_mm=LINK_MM):
    """保留最大连通域及经 link_mm 距离链式连接的足够大连通域。"""
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
    """围绕 axis-1 中线镜像包围盒，按体素中心定义反射。"""
    b = [list(x) for x in box]
    lo, hi = b[LR_AXIS]
    new_lo = int(np.floor(2 * mid - (hi - 1)))
    new_hi = int(np.ceil(2 * mid - lo)) + 1
    b[LR_AXIS] = [max(new_lo, 0), min(new_hi, shape[LR_AXIS])]
    return b


def landmark_mask_from_probs(p1, p4):
    """从背景、类别 1、类别 4 的概率获得 argmax 路标掩膜。"""
    p1 = np.asarray(p1, dtype=np.float32)
    p4 = np.asarray(p4, dtype=np.float32)
    p0 = 1.0 - p1 - p4
    is1 = (p1 > p0) & (p1 >= p4)
    is4 = (p4 > p0) & (p4 > p1)
    return is1 | is4, is4


def compute_case_roi(landmarks, is_c4, image, spacing=SPACING, margins=MARGINS_MM,
                     min_vox=MIN_VOX, link_mm=LINK_MM):
    """返回病例 ROI 记录以及清洁后的路标和类别 4 掩膜。"""
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
    """返回指定边距的双侧包围盒；空列表表示全体积回退。"""
    return [sd["boxes"][str(m)] for sd in rec["sides"] if sd["boxes"][str(m)] is not None]


def coverage(rec, m, gt_y):
    """返回 GT Y 覆盖率及 ROI 体积比例。"""
    shape = gt_y.shape
    boxes = roi_boxes(rec, m)
    full = [[0, n] for n in shape]
    roi = boxes_mask(boxes, full) if boxes else np.ones(shape, dtype=bool)
    n_y = int(gt_y.sum())
    cov = float((gt_y & roi).sum() / n_y) if n_y else float("nan")
    return cov, float(roi.mean())


__all__ = [
    "BODY_THR", "MIN_VOX", "LINK_MM", "head_midline", "side_slices", "clean_cluster",
    "margin_vox", "mirror_box", "landmark_mask_from_probs", "compute_case_roi", "roi_boxes", "coverage",
]
