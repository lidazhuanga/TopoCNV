"""Calibrate the bone threshold of the G1 evidence map on normalized CT.

The preprocessing cache normalizes each case to [0, 1] by its own 1st/99th
intensity percentiles (no HU clipping), so a fixed HU level maps to a
case-dependent normalized value.  This script reports, for every training
case, the normalized value of ``--hu`` (default 200 HU, cortical-bone onset)
and summarizes the distribution.  ``DEFAULT_BONE_NORM_THR`` in
``trainer/cnv_protocol_s2.py`` (0.97) sits between the 10th percentile and the
median of this distribution on the 154-case training cohort.

Usage::

    python tools/calibrate_bone_threshold.py [--data_root npc_data/train] [--hu 200] [--per_case]
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npc_data import DATA_ROOT_DEFAULT, _get_all_cases  # noqa: E402


def normalized_value(image: np.ndarray, hu: float) -> float:
    p1, p99 = np.percentile(image, [1.0, 99.0])
    return float(np.clip((hu - p1) / max(p99 - p1, 1e-6), 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--hu", type=float, default=200.0)
    ap.add_argument("--per_case", action="store_true", help="print one line per case")
    args = ap.parse_args()

    vals = []
    for i, (_, img_path, _) in enumerate(_get_all_cases(args.data_root)):
        x = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
        t = normalized_value(x, args.hu)
        vals.append(t)
        if args.per_case:
            print(f"case {i:03d}: t(HU={args.hu:.0f})={t:.3f}", flush=True)
    if not vals:
        sys.exit(f"no cases found under {args.data_root}")
    v = np.asarray(vals)
    print(f"n={len(v)} median={np.median(v):.3f} mean={v.mean():.3f} "
          f"p10={np.percentile(v, 10):.3f} p90={np.percentile(v, 90):.3f} "
          f"frac_saturated={(v >= 1.0).mean():.3f}")


if __name__ == "__main__":
    main()
