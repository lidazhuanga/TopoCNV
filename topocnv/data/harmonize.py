"""参考标签厚度统一化：仅在平面内扩展薄 Y 标签，保留原标签及类别 5。"""

import numpy as np
from scipy import ndimage as ndi

from topocnv.data.utils import SPACING
from topocnv.utils.box_ops import mask_Y

SIGMA = (0.0, 0.8, 0.8)
THIN_THR = 8.5
T_GRID = tuple(round(0.10 + 0.01 * i, 2) for i in range(21))
PLANE4 = np.zeros((3, 3, 3), bool)
PLANE4[1] = ndi.generate_binary_structure(2, 1)
_PAD = (1, 6, 6)


def _crop(y):
    nz = np.nonzero(y)
    return tuple(slice(max(0, int(i.min()) - p), min(n, int(i.max()) + 1 + p))
                 for i, p, n in zip(nz, _PAD, y.shape))


def harmonise(lab, t, sigma=SIGMA):
    """返回标签副本，按平面内平滑概率阈值 t 生长 Y。"""
    lab = np.asarray(lab)
    out = np.array(lab, copy=True)
    y_full = mask_Y(lab)
    if not y_full.any():
        return out
    sl = _crop(y_full)
    sub, y = lab[sl], y_full[sl]
    g = ndi.gaussian_filter(y.astype(np.float32), sigma)
    grow = (g >= t) & ~y & ndi.binary_dilation(y, PLANE4) & (sub != 5)
    if grow.any():
        _, idx = ndi.distance_transform_edt(~y, sampling=SPACING, return_indices=True)
        nz = np.nonzero(grow)
        o = out[sl]
        o[nz] = sub[idx[0][nz], idx[1][nz], idx[2][nz]]
    return out


def skeleton(y):
    """计算二值掩膜的三维骨架，先裁剪以减少计算量。"""
    from skimage.morphology import skeletonize
    sk = np.zeros(y.shape, bool)
    if y.any():
        sl = _crop(y)
        sk[sl] = skeletonize(y[sl])
    return sk


def thickness(lab):
    """Y 体素数与骨架体素数之比。"""
    y = mask_Y(lab)
    return float(y.sum() / max(1, skeleton(y).sum()))


def run_length(y, axis):
    """计算指定轴上的平均连续段长度。"""
    prev = np.zeros_like(y)
    idx = [slice(None)] * 3
    src = [slice(None)] * 3
    idx[axis], src[axis] = slice(1, None), slice(None, -1)
    prev[tuple(idx)] = y[tuple(src)]
    return float(y.sum() / max(1, (y & ~prev).sum()))


__all__ = ["SIGMA", "THIN_THR", "T_GRID", "PLANE4", "harmonise", "skeleton", "thickness", "run_length"]
