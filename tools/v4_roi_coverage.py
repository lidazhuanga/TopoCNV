#!/usr/bin/env python3
"""ROI coverage of GT Y for the v4 cascade (protocol v0.4 §4: target >= 99% of Y voxels).

  python tools/v4_roi_coverage.py --folds 0,1 --set ival --variant dev       # development: V_0 and V_1
  python tools/v4_roi_coverage.py --folds 0,1 --set train --variant dev      # T_k - V_k
  python tools/v4_roi_coverage.py --folds 0,1,2,3,4 --set report --variant report   # after freezing

--set report is refused until configs/v4_frozen.yaml exists. Writes results/v4_roi_coverage/<set>_folds<..>.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.v4.common import MARGINS_MM, REPO, RESULTS, load_label, mask_Y  # noqa: E402
from trainer.v4.data import load_roi_record  # noqa: E402
from trainer.v4.roi import coverage  # noqa: E402
from trainer.v4.splits import internal_split, report_cases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--set", choices=["ival", "train", "report"], default="ival")
    ap.add_argument("--variant", choices=["dev", "report"], default="dev",
                    help="prior/ROI build: dev (atlas from T_k - V_k) or report (atlas from T_k)")
    args = ap.parse_args()
    if args.set == "report" and not (REPO / "configs" / "v4_frozen.yaml").exists():
        raise SystemExit("report-fold coverage only after configs/v4_frozen.yaml is committed")
    rows = []
    for k in [int(x) for x in args.folds.split(",")]:
        tr, iv = internal_split(k)
        cases = {"ival": iv, "train": tr, "report": report_cases(k)}[args.set]
        for c in cases:
            rec = load_roi_record(f"fold{k}_{args.variant}", c)
            gt = mask_Y(load_label(c))
            row = {"fold": k, "patient_id": c, "roi_found": rec["roi_found"],
                   "sides": "/".join(s["source"] for s in rec["sides"])}
            for m in MARGINS_MM:
                cov, vol = coverage(rec, m, gt)
                row[f"cov_m{m}"], row[f"roivol_m{m}"] = cov, vol
            rows.append(row)
    out = RESULTS / "v4_roi_coverage"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.variant}_{args.set}_folds{args.folds.replace(',', '')}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for m in MARGINS_MM:
        v = np.array([r[f"cov_m{m}"] for r in rows])
        summ[m] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "frac_ge_0.99": float(np.mean(v >= 0.99)),
                   "roi_vol_frac": float(np.mean([r[f"roivol_m{m}"] for r in rows]))}
    summ["n"] = len(rows)
    summ["no_roi"] = [r["patient_id"] for r in rows if not r["roi_found"]]
    summ["mirrored"] = [r["patient_id"] for r in rows if "mirror" in r["sides"]]
    (path.with_suffix(".json")).write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
