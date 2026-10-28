#!/usr/bin/env python3
"""V4.3 (V4_PLAN §12.3): on which side of the MRI field of view the missing Y lies, in cases with y_cov < 0.95.

Data property only (frozen MRI field-of-view masks, harmonised labels, CT headers); reads no model output.
Column-wise rule: a missing Y voxel (classes 1-4 of results/v4_harm_v1, outside _mri_fov) lies 'cranial' if it is
above the highest field-of-view voxel of its (lr, ap) column, 'caudal' if below the lowest, 'lateral' if its
column has no field of view at all. The tilted edge of an oblique MR slab is thus counted on its side. Cache
axis 0 = NIfTI slice index (sitk); its direction (LPS, +z = cranial) is read from every CT header.

  python tools/v4_mm43_trunc_stats.py --out results/v4_mm43_trunc_side.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.v4.common import CACHE_DIR, RESULTS  # noqa: E402
from trainer.v4.splits import all_cases  # noqa: E402

MRI = Path.home() / "cnv" / "mri" / "cache"
HARM = RESULTS / "v4_harm_v1" / CACHE_DIR.name
COV_MAX = 0.95       # 'incomplete coverage', as the MRI README §4 (review: coverage 50-95%)
DOMINANT = 0.7       # a side is a case's dominant side if it holds >= 70% of the case's missing Y voxels


def z_direction(c):
    import SimpleITK as sitk
    r = sitk.ImageFileReader()
    r.SetFileName(str(ROOT / "npc_data" / "train" / "image" / f"{c}.nii.gz"))
    r.ReadImageInformation()
    return int(round(r.GetDirection()[8]))


def one(c):
    fov = np.load(MRI / f"{c}_mri_fov.npy") > 0
    lab = np.load(HARM / f"{c}_label.npy")
    assert fov.shape == lab.shape, (c, fov.shape, lab.shape)
    y = (lab >= 1) & (lab <= 4)
    miss = y & ~fov
    z, a, b = np.nonzero(miss)
    col = fov.any(0)
    n = fov.shape[0]
    top = np.where(col, n - 1 - np.argmax(fov[::-1], 0), -1)
    bot = np.where(col, np.argmax(fov, 0), n)
    inc = col[a, b]
    cr = int(((z > top[a, b]) & inc).sum())
    ca = int(((z < bot[a, b]) & inc).sum())
    la = int(len(z) - cr - ca)
    side = "none"
    if len(z):
        best = max((cr, "cranial"), (ca, "caudal"), (la, "lateral"))
        side = best[1] if best[0] >= DOMINANT * len(z) else "mixed"
    return {"case": c, "y": int(y.sum()), "y_cov": float(1 - len(z) / max(int(y.sum()), 1)),
            "cranial": cr, "caudal": ca, "lateral": la, "side": side, "z_direction": z_direction(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (MRI / "FROZEN").exists():
        raise SystemExit(f"{MRI}/FROZEN missing: the MRI cache is not frozen")
    from multiprocessing import Pool
    with Pool(12) as p:
        rows = p.map(one, [c for c in all_cases() if (MRI / f"{c}_mri_fov.npy").exists()])   # no MR: no fov file
    inc = [r for r in rows if r["y_cov"] < COV_MAX]
    out = {"rule": __doc__.split("\n\n")[0], "frozen": (MRI / "FROZEN").read_text().strip(),
           "z_direction_counts": dict(Counter(r["z_direction"] for r in rows)),
           "n_cases": len(rows), "n_incomplete": len(inc),
           "dominant_side": dict(Counter(r["side"] for r in inc)),
           "missing_voxels": {s: sum(r[s] for r in inc) for s in ("cranial", "caudal", "lateral")},
           "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
