#!/usr/bin/env python3
"""D1-AM1: freeze the per-fold QC gate (user confirmed 2026-10-25; see ~/cnv/kimi_d1_answer_20261025.log).

Per outer fold k, using ONLY the dev-tier Stage-1 predictions of the fold's training cases
(T_k - V_k, inner-OOF; no GT, no V_k, no report fold):
  1. margin m_k: from results/v4_roi_coverage/dev_train_folds{k}.csv — smallest m in
     {15, 25, 35} whose MINIMUM coverage over the 105 training cases is >= 0.99; if none,
     m_k = 35 (deterministic closure of the infeasible case, NOT a lowered threshold);
  2. D signal: left-right distance (mm) between the per-side clean class-4 point-cloud
     centroids (trainer/v4/priors.py::bilateral_c4_distance_mm);
  3. bounds L_k = Q1 - 3*IQR, U_k = Q3 + 3*IQR (linear quantiles over the valid D values).

Writes configs/v4_qc/fold{k}.json (write-once: refuses to overwrite without --force, and
records the per-case D values + the coverage CSV sha256 so the freeze is auditable).
Structural failures (D uncomputable) are recorded as triggers and excluded from the quantiles.
The gate is applied by tools/v4_build_priors.py --qc; the report phase never recalibrates.

  python tools/v4_qc_d1.py                # all folds
  python tools/v4_qc_d1.py --folds 0,1
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from topocnv.data.utils import STAGE1_DIR, load_image  # noqa: E402
from topocnv.priors import bilateral_c4_distance_mm  # noqa: E402
from topocnv.priors.roi import compute_case_roi, head_midline, landmark_mask_from_probs  # noqa: E402
from topocnv.data.v4_splits import N_INNER, internal_split, stage1_source  # noqa: E402

MARGINS = (15, 25, 35)
MIN_VALID_D = 90


def load_s1_dev(k, case_id):
    """Dev-tier Stage-1 input of fold k for case_id (inner-OOF inside T_k, ensemble for V_k).
    Must stay byte-identical to tools/v4_build_priors.load_s1 (dev variant)."""
    src, j = stage1_source(k, case_id, tier="dev")
    root = STAGE1_DIR / f"fold{k}_dev"
    if src == "inner":
        d = np.load(root / "pred" / f"{case_id}_s1prob.npz")
        return d["p1"] / np.float32(255), d["p4"] / np.float32(255)
    p1 = p4 = None
    for jj in range(N_INNER):
        d = np.load(root / f"inner{jj}" / "val_pred" / f"{case_id}_s1prob.npz")
        a, b = d["p1"].astype(np.float32), d["p4"].astype(np.float32)
        p1, p4 = (a, b) if p1 is None else (p1 + a, p4 + b)
    return p1 / (255.0 * N_INNER), p4 / (255.0 * N_INNER)


def case_d(k, case_id):
    p1, p4 = load_s1_dev(k, case_id)
    lm, is4 = landmark_mask_from_probs(p1, p4)
    image = load_image(case_id)
    _, _, c4_clean = compute_case_roi(lm, is4, image)
    return bilateral_c4_distance_mm(c4_clean, head_midline(image, None))


def margin_from_coverage(k, cases):
    """Frozen D1 rule on the fold's training-case coverage CSV; 35 if infeasible."""
    path = ROOT / "results/v4_roi_coverage" / f"dev_train_folds{k}.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(cases), f"fold {k}: expected {len(cases)} training rows, got {len(rows)}"
    assert {r["patient_id"] for r in rows} == set(cases), f"fold {k}: CSV patient set mismatch"
    assert all(int(r["fold"]) == k for r in rows), f"fold {k}: CSV fold column mismatch"
    for m in MARGINS:
        mn = min(float(r[f"cov_m{m}"]) for r in rows)
        if mn >= 0.99:
            return m, mn, True
    return MARGINS[-1], min(float(r[f"cov_m{MARGINS[-1]}"]) for r in rows), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen config")
    args = ap.parse_args()
    import multiprocessing as mp
    out_dir = ROOT / "configs" / "v4_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("fold | m_k | min_cov@mk | feasible | Q1 | Q3 | L | U | train triggers")
    for k in [int(x) for x in args.folds.split(",")]:
        out = out_dir / f"fold{k}.json"
        if out.exists() and not args.force:
            raise SystemExit(f"{out} already frozen (write-once); pass --force to override")
        cases = internal_split(k)[0]
        m_k, min_cov, feasible = margin_from_coverage(k, cases)
        with mp.get_context("fork").Pool(args.workers) as p:
            ds = p.starmap(case_d, [(k, c) for c in cases])
        invalid = [c for c, d in zip(cases, ds) if d is None or not math.isfinite(d)]
        valid = [d for d in ds if d is not None and math.isfinite(d)]
        assert len(valid) >= MIN_VALID_D, f"fold {k}: only {len(valid)} valid D values"
        q1, q3 = (float(x) for x in np.quantile(np.array(valid), [0.25, 0.75], method="linear"))
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        trig = [c for c, d in zip(cases, ds)
                if d is None or not math.isfinite(d) or not (lo <= d <= hi)]
        csv_path = ROOT / "results/v4_roi_coverage" / f"dev_train_folds{k}.csv"
        cfg = {"amendment": "D1-AM1", "fold": k, "margin": m_k, "min_cov_at_margin": min_cov,
               "rule_feasible": feasible, "L": lo, "U": hi, "q1": q1, "q3": q3,
               "n_train": len(cases), "n_valid_d": len(valid), "d_invalid": invalid,
               "train_triggers": trig, "train_d_max": max(valid),
               "per_case_D": {c: (None if d is None else round(float(d), 3))
                              for c, d in zip(cases, ds)},
               "coverage_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
               "computed": datetime.now().isoformat(),
               "source": "tools/v4_qc_d1.py: dev-tier inner-OOF Stage 1 of T_k-V_k only "
                         "(no GT beyond the frozen D1 coverage rule, no V_k, no report fold); "
                         "thresholds never recalibrated"}
        out.write_text(json.dumps(cfg, indent=1))
        print(f"f{k}   | {m_k:2d} | {min_cov:.4f}    | {str(feasible):5s}    | {q1:.3f} | {q3:.3f} "
              f"| {lo:.3f} | {hi:.3f} | {trig}")
    print(f"frozen -> {out_dir}/fold*.json")


if __name__ == "__main__":
    main()
