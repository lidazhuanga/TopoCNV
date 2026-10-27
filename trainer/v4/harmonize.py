"""Label-thickness harmonisation of the reference standard (V4.2, amendment V4-AM-20261027-02).

About 45 of the 154 labels follow a thin drawing convention. Y thickness (Y voxels per skeleton voxel) is about
6.4 for these cases against about 11.4 for the rest. The nerve length and the z extent are the same; the thin
labels are about 0.7 voxel narrower on each in-plane side.

harmonise() grows a thin label in-plane by sub-voxel resampling:
  1. Smooth the binary Y in-plane with a Gaussian (sigma, in voxels, z untouched).
  2. Re-threshold at t.
  3. Keep only voxels that are in-plane 4-neighbours of Y and are not class 5.
  4. Give each added voxel the class of the nearest original Y voxel, using physical spacing.
Original voxels are never changed or removed, so growth is at most one voxel in-plane.

Torch-free; thickness() needs scikit-image (nnU-Net venv).
"""
import numpy as np
from scipy import ndimage as ndi

from .common import SPACING, mask_Y

SIGMA = (0.0, 0.8, 0.8)  # voxels: no smoothing along z (the conventions differ in-plane only)
THIN_THR = 8.5           # thickness below this = thin convention (gap of the bimodal distribution, 2026-10-27)
T_GRID = tuple(round(0.10 + 0.01 * i, 2) for i in range(21))  # 0.10 .. 0.30
PLANE4 = np.zeros((3, 3, 3), bool)
PLANE4[1] = ndi.generate_binary_structure(2, 1)
_PAD = (1, 6, 6)  # crop margin: >= Gaussian radius (truncate 4 * 0.8 -> 3 px) + 1-voxel growth


def _crop(y):
    nz = np.nonzero(y)
    return tuple(slice(max(0, int(i.min()) - p), min(n, int(i.max()) + 1 + p))
                 for i, p, n in zip(nz, _PAD, y.shape))


def harmonise(lab, t, sigma=SIGMA):
    """Return a copy of lab with Y grown in-plane where the smoothed Y reaches t (see module docstring)."""
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
    """3-D skeleton of a binary mask (cropped for speed)."""
    from skimage.morphology import skeletonize
    sk = np.zeros(y.shape, bool)
    if y.any():
        sl = _crop(y)
        sk[sl] = skeletonize(y[sl])
    return sk


def thickness(lab):
    """Y voxels per skeleton voxel."""
    y = mask_Y(lab)
    return float(y.sum() / max(1, skeleton(y).sum()))


def run_length(y, axis):
    """Mean length of the runs of y along one axis (voxels)."""
    prev = np.zeros_like(y)
    idx = [slice(None)] * 3
    src = [slice(None)] * 3
    idx[axis], src[axis] = slice(1, None), slice(None, -1)
    prev[tuple(idx)] = y[tuple(src)]
    return float(y.sum() / max(1, (y & ~prev).sum()))
