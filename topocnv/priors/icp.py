"""Umeyama 相似变换、Trimmed ICP 与双侧类别 4 点云工具。"""

import numpy as np
from scipy.spatial import cKDTree

from topocnv.data.utils import SPACING
from topocnv.priors.roi import side_slices

MIN_ICP_POINTS = 20
MAX_ICP_POINTS = 4000
SCALE_BOUNDS = (0.8, 1.25)
TARGET_SCALE_BOUNDS = (1.0, 1.0)


def umeyama(src, dst):
    """最小二乘相似变换 dst ≈ s * R @ src + t（Umeyama 1991）。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    cov = Y.T @ X / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_s = (X ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def apply_sim(T, pts):
    s, R, t = T
    return s * pts @ R.T + t


def icp_similarity(src, dst, n_iter=60, trim=0.9, scale_bounds=SCALE_BOUNDS, tol=1e-5, seed=0):
    """带相似变换的截尾点对点 ICP，返回变换和毫米 RMSE。"""
    rng = np.random.RandomState(seed)
    if len(src) > MAX_ICP_POINTS:
        src = src[rng.choice(len(src), MAX_ICP_POINTS, replace=False)]
    tree = cKDTree(dst)
    T = (1.0, np.eye(3), dst.mean(0) - src.mean(0))
    prev = np.inf
    rmse = np.inf
    for _ in range(n_iter):
        d, idx = tree.query(apply_sim(T, src))
        keep = d <= np.quantile(d, trim)
        s, R, t = umeyama(src[keep], dst[idx[keep]])
        if not scale_bounds[0] <= s <= scale_bounds[1]:
            s = float(np.clip(s, *scale_bounds))
            t = dst[idx[keep]].mean(0) - s * R @ src[keep].mean(0)
        T = (s, R, t)
        rmse = float(np.sqrt((d[keep] ** 2).mean()))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return T, rmse


def mask_points_mm(mask, spacing=SPACING, offset=(0, 0, 0)):
    idx = np.argwhere(mask)
    return (idx + np.asarray(offset)) * np.asarray(spacing, dtype=np.float64)


def side_points(mask_c4, mid, spacing=SPACING):
    """返回全体积毫米坐标系内的双侧类别 4 点云。"""
    out = []
    for a0, a1 in side_slices(mask_c4.shape, mid):
        sub = np.zeros_like(mask_c4)
        sub[:, a0:a1] = mask_c4[:, a0:a1]
        out.append(mask_points_mm(sub, spacing))
    return out


def bilateral_c4_distance_mm(c4_mask, mid, spacing=SPACING):
    """双侧清洁类别 4 质心的左右物理距离；缺侧或中线未知时返回 None。"""
    if mid is None:
        return None
    pts = side_points(c4_mask, mid, spacing)
    cents = []
    for p in pts:
        if len(p) < 1:
            return None
        cents.append(p.mean(0))
    return abs(float(cents[1][1] - cents[0][1]))


__all__ = [
    "MIN_ICP_POINTS", "MAX_ICP_POINTS", "SCALE_BOUNDS", "TARGET_SCALE_BOUNDS",
    "umeyama", "apply_sim", "icp_similarity", "mask_points_mm", "side_points", "bilateral_c4_distance_mm",
]
