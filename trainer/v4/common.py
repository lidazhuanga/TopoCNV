"""Shared paths, constants and small array helpers for the v4 pipeline (protocol v0.4-draft).

Torch-free on purpose: tools/v4_skeletons.py runs this module under the nnU-Net venv.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "MIA" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS = Path(os.environ.get("CNV_RESULTS_ROOT", str(REPO / "results")))
CACHE_DIR = RESULTS / "preprocess_cache" / "sp_2.500_1.133_1.133"
SPACING = (2.5, 1.1328120231628418, 1.1328120231628418)  # == metrics_protocol.DEFAULT_SPACING_ZYX
LR_AXIS = 1  # left-right is axis 1 of the cache arrays (z, lr, ap); checked 2026-10-22
MARGINS_MM = (15, 25, 35)
PRIOR_BOX_MARGIN = 35  # priors are stored on the union of the m = 35 mm side boxes
PRIOR_NAMES = ("a", "b", "c")  # a: Stage-1 class-4 prob, b: distance to foramina, c: atlas

STAGE1_DIR = RESULTS / "v4_stage1"
PRIORS_DIR = Path(os.environ.get("V4_PRIORS_DIR", str(RESULTS / "v4_priors")))  # override: smoke tests
SKEL_DIR = RESULTS / "v4_skel"
RUNS_DIR = RESULTS / "v4_runs"


def load_config(path):
    """configs/*.yaml are written in JSON flow style (valid YAML 1.2): topocnv has no PyYAML."""
    with open(path) as f:
        return json.load(f)


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def load_image(case_id, mmap=True):
    return np.load(CACHE_DIR / f"{case_id}_image.npy", mmap_mode="r" if mmap else None)


def load_label(case_id, mmap=True):
    return np.load(CACHE_DIR / f"{case_id}_label.npy", mmap_mode="r" if mmap else None)


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
    """Half-open bounding box [[z0, z1], [y0, y1], [x0, x1]] of a boolean mask, or None."""
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
    """Values of a field stored only on `box` (arr has shape [..., *box_shape]) read on `region`.

    Zero outside the box. With `shift`, the full-volume field is rolled first:
    out[p] = field[(p - shift) mod full_shape] (the v4 shuffled-prior control).
    """
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
    """Boolean mask on `region` of the union of `boxes` (all in full-volume voxel coordinates)."""
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
    """Fixed per-case roll for the shuffled-prior arm (uniform over each axis, as old A2 spatial_roll)."""
    import zlib
    rng = np.random.RandomState(zlib.crc32(case_id.encode()) & 0x7FFFFFFF)
    return [int(rng.randint(0, max(int(n), 1))) for n in shape]
