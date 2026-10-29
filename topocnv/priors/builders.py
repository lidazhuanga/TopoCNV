"""v4 概率、距离与图谱先验的构建及 ROI 记录工厂。"""

import numpy as np
from scipy import ndimage as ndi

from topocnv.data.utils import MARGINS_MM, SPACING
from topocnv.priors.atlas import ATLAS_ROI_THR
from topocnv.priors.roi import margin_vox, mirror_box, roi_boxes
from topocnv.utils.box_ops import box_slices, expand_box, union_box

DIST_TRUNC_MM = 30.0


def dist_prior(c4_clean_box, spacing=SPACING, trunc_mm=DIST_TRUNC_MM):
    if not c4_clean_box.any():
        return np.zeros(c4_clean_box.shape, dtype=np.float32)
    d = ndi.distance_transform_edt(~c4_clean_box, sampling=spacing)
    return (1.0 - np.minimum(d, trunc_mm) / trunc_mm).astype(np.float32)


def to_uint8(x):
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def prior_box_from_rec(rec, margin):
    boxes = roi_boxes(rec, margin)
    return union_box(boxes) if boxes else None


def build_case_priors(p4, c4_clean, image, prior_box, atlas=None, exclude=None, transforms=None):
    """返回 prior_box 内的三通道 uint8 先验及图谱配准信息。"""
    sl = box_slices(prior_box)
    a = np.asarray(p4[sl], dtype=np.float32)
    b = dist_prior(c4_clean[sl])
    if atlas is not None:
        c, info = atlas.warp_to_target(c4_clean, image, prior_box, exclude=exclude, transforms=transforms)
    else:
        c, info = np.zeros_like(a), []
    return np.stack([to_uint8(a), to_uint8(b), to_uint8(c)]), info


def atlas_roi_record(rec, atlas, c4_clean, image, exclude=None, thr=ATLAS_ROI_THR, margins=None):
    """将路标 ROI 替换为图谱支撑区 ROI，缺失侧使用对侧镜像。"""
    margins = margins or MARGINS_MM
    shape = tuple(rec["shape"])
    transforms = atlas.target_transforms(c4_clean, image)
    bbs = atlas.support_bboxes(shape, transforms, exclude, thr)
    srcs = ["atlas" if b is not None else "none" for b in bbs]
    for s in (0, 1):
        if bbs[s] is None and bbs[1 - s] is not None:
            bbs[s] = mirror_box(bbs[1 - s], transforms[0], shape)
            srcs[s] = "mirror"
    for s, sd in enumerate(rec["sides"]):
        sd["landmark_boxes"] = sd["boxes"]
        sd["landmark_source"] = sd["source"]
        sd["atlas_bbox"] = bbs[s]
        sd["source"] = srcs[s]
        sd["boxes"] = {str(m): (expand_box(bbs[s], margin_vox(m), shape) if bbs[s] is not None else None)
                       for m in margins}
    rec["roi_found"] = any(b is not None for b in bbs)
    rec["roi_method"] = {"name": "atlas_support", "thr": thr}
    return rec, transforms


__all__ = ["DIST_TRUNC_MM", "dist_prior", "to_uint8", "prior_box_from_rec", "build_case_priors", "atlas_roi_record"]
