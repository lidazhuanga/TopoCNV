#!/usr/bin/env python3
"""Development-phase summary on the internal validation sets only (V_0 ∪ V_1; protocol v0.4 §5).

Reads results/v4_runs/<name>/fold{0,1}/ival_metrics.csv (never report-fold files) and prints, per
candidate: pooled mean Dice_Y, HD95_Y mean / median, per-fold means, and the paired difference to a
reference candidate (case-level bootstrap 95% CI, 10,000 resamples).

  python tools/v4_dev_summary.py r1_m25 r1p_abc r1p_bc r1p_ac r1p_ab --ref r1_m25
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.v4.common import RUNS_DIR  # noqa: E402

DEV_FOLDS = (0, 1)


def load(name):
    rows = {}
    for k in DEV_FOLDS:
        p = RUNS_DIR / name / f"fold{k}" / "ival_metrics.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                rows[(k, r["patient_id"])] = (float(r["Dice_Y"]), float(r["HD95_Y"]))
    return rows


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    data = {n: load(n) for n in args.names + ([args.ref] if args.ref and args.ref not in args.names else [])}
    out = {}
    print(f"{'candidate':<16}{'folds':>6}{'n':>5}{'Dice_Y':>9}{'f0':>8}{'f1':>8}{'HD95 mean':>11}{'HD95 med':>10}"
          f"{'dDice vs ref [95% CI]':>32}")
    for n in args.names:
        r = data[n]
        if not r:
            print(f"{n:<16} (no completed folds)")
            continue
        folds = sorted({k for k, _ in r})
        dice = np.array([v[0] for v in r.values()])
        hd = np.array([v[1] for v in r.values()])
        per = {k: float(np.mean([v[0] for (kk, _), v in r.items() if kk == k])) for k in folds}
        rec = {"folds": folds, "n": len(r), "Dice_Y": float(dice.mean()), "per_fold": per,
               "HD95_mean": float(np.nanmean(hd)), "HD95_median": float(np.nanmedian(hd))}
        ref_txt = ""
        if args.ref and n != args.ref and data[args.ref]:
            common = sorted(set(r) & set(data[args.ref]))
            d = np.array([r[c][0] - data[args.ref][c][0] for c in common])
            lo, hi = boot_ci(d)
            rec.update(ref=args.ref, n_paired=len(common), dDice=float(d.mean()), dDice_ci=[lo, hi])
            ref_txt = f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(common)}"
        out[n] = rec
        f0 = f"{per.get(0, float('nan')):.4f}"
        f1 = f"{per.get(1, float('nan')):.4f}"
        print(f"{n:<16}{''.join(map(str, folds)):>6}{len(r):>5}{dice.mean():>9.4f}{f0:>8}{f1:>8}"
              f"{np.nanmean(hd):>11.2f}{np.nanmedian(hd):>10.2f}{ref_txt:>32}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
