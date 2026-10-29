"""v4 管线的半开区间包围盒与数组区域工具。"""

import numpy as np


def mask_Y(lab):
    lab = np.asarray(lab)
    return (lab >= 1) & (lab <= 4)


def box_shape(box):
    return tuple(int(b1 - b0) for b0, b1 in box)


def box_slices(box):
    return tuple(slice(int(b0), int(b1)) for b0, b1 in box)


def expand_box(box, margin_vox, shape):
    return [[max(int(b0) - int(m), 0), min(int(b1) + int(m), int(n))]
            for (b0, b1), m, n in zip(box, margin_vox, shape)]


def mask_bbox(mask):
    """返回布尔掩膜的半开包围盒；空掩膜返回 None。"""
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        return None
    return [[int(i.min()), int(i.max()) + 1] for i in idx]


def union_box(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [[min(b[a][0] for b in boxes), max(b[a][1] for b in boxes)] for a in range(3)]


def region_from_box(arr, box, region, full_shape=None, shift=None):
    """读取仅存储于 box 内的数组区域，框外补零，可按病例循环平移。"""
    lead = arr.shape[:-3]
    out = np.zeros(lead + box_shape(region), dtype=arr.dtype)
    if shift is None:
        src, dst = [], []
        for (r0, r1), (b0, b1) in zip(region, box):
            lo, hi = max(r0, b0), min(r1, b1)
            if hi <= lo:
                return out
            src.append(slice(lo - b0, hi - b0))
            dst.append(slice(lo - r0, hi - r0))
        out[(Ellipsis,) + tuple(dst)] = arr[(Ellipsis,) + tuple(src)]
        return out
    idx, valid = [], []
    for ax, ((r0, r1), (b0, b1)) in enumerate(zip(region, box)):
        p = (np.arange(r0, r1) - int(shift[ax])) % int(full_shape[ax])
        v = (p >= b0) & (p < b1)
        if not v.any():
            return out
        idx.append(p[v] - b0)
        valid.append(np.nonzero(v)[0])
    out[(Ellipsis,) + np.ix_(*valid)] = arr[(Ellipsis,) + np.ix_(*idx)]
    return out


def boxes_mask(boxes, region):
    """在 region 上构造 boxes 并集的布尔掩膜。"""
    m = np.zeros(box_shape(region), dtype=bool)
    for b in boxes:
        sl = []
        for (r0, r1), (b0, b1) in zip(region, b):
            lo, hi = max(r0, b0), min(r1, b1)
            if hi <= lo:
                break
            sl.append(slice(lo - r0, hi - r0))
        else:
            m[tuple(sl)] = True
    return m


def case_shift(case_id, shape):
    """为打乱先验对照生成固定的逐病例平移量。"""
    import zlib
    rng = np.random.RandomState(zlib.crc32(case_id.encode()) & 0x7FFFFFFF)
    return [int(rng.randint(0, max(int(n), 1))) for n in shape]


__all__ = [
    "mask_Y", "box_shape", "box_slices", "expand_box", "mask_bbox",
    "union_box", "region_from_box", "boxes_mask", "case_shift",
]
