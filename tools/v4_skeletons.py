#!/usr/bin/env python3
"""Tubed skeletons of GT Y for the Skeleton Recall Loss (Kirchhoff et al., ECCV 2024).

skeletonize (skimage, 3D Lee thinning) -> one dilation with a 3x3x3 structuring element -> AND Y.
Saved on the Y bounding box: results/v4_skel/{case}_skel.npy (uint8) + {case}_skel.json (box).
Labels only; no fold dependence (the loss uses training cases' own GT).

topocnv has no scikit-image, so this runs under the nnU-Net venv (numpy/scipy/skimage only):
  ~/cnv/envs/nnunetv2_cnv/bin/python tools/v4_skeletons.py --workers 16
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.v4.common import SKEL_DIR, expand_box, load_label, mask_bbox, mask_Y  # noqa: E402
from trainer.v4.splits import all_cases  # noqa: E402


def tubed_skeleton(y):
    from skimage.morphology import skeletonize
    sk = skeletonize(y.astype(bool))
    return ndi.binary_dilation(sk > 0, structure=np.ones((3, 3, 3), bool)) & y


def one(case_id):
    y = mask_Y(load_label(case_id))
    box = expand_box(mask_bbox(y), (2, 2, 2), y.shape)
    sl = tuple(slice(a, b) for a, b in box)
    tube = tubed_skeleton(y[sl])
    np.save(SKEL_DIR / f"{case_id}_skel.npy", tube.astype(np.uint8))
    (SKEL_DIR / f"{case_id}_skel.json").write_text(json.dumps({"box": box}))
    return case_id, int(y.sum()), int(tube.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    SKEL_DIR.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as p:
        res = p.map(one, all_cases())
    frac = [t / y for _, y, t in res]
    print(f"{len(res)} cases; tube / Y voxels: mean {np.mean(frac):.3f}, min {np.min(frac):.3f}")


if __name__ == "__main__":
    main()
