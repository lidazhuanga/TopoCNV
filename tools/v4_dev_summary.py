#!/usr/bin/env python3
"""Development-phase scoring and summary on the internal validation sets only (V_0 ∪ V_1; protocol §5).

1) score: eval_protocol_metrics.py --pred_binary_Y on results/v4_dev_preds/<name>/fold{k}/ (V_k
   predictions only; the report folds are never in these folders) -> results/v4_dev_eval/<name>/fold{k}.csv
2) summary: pooled mean Dice_Y, SurfDice_Y, HD95_Y mean / median, per fold, and the paired difference to a
   reference candidate (case-level bootstrap 95% CI, 10,000 resamples). The reference for every
   comparison is R0-dev (stock nnU-Net v2 trained on T_k - V_k; main session 2026-10-23), never the
   5-fold report numbers.

  python tools/v4_dev_summary.py score r0_dev r1_m25
  python tools/v4_dev_summary.py summary r1_m25 r1p_abc r1p_bc --ref r0_dev
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.v4.common import CACHE_DIR, RESULTS  # noqa: E402
from trainer.v4.splits import internal_split  # noqa: E402

DEV_FOLDS = (0, 1)
PRED = RESULTS / "v4_dev_preds"
EVAL = RESULTS / "v4_dev_eval"
KEYS = ("Dice_Y", "HD95_Y", "SurfDice_Y", "Recall_vis", "Dice_vis")


def score(names):
    for n in names:
        for k in DEV_FOLDS:
            pdir = PRED / n / f"fold{k}"
            if not (pdir / "EXPORTED").exists():
                print(f"{n} fold {k}: not exported yet")
                continue
            have = {p.name.split("_stage2_label")[0] for p in pdir.glob("*_stage2_label.nrrd")}
            assert have == set(internal_split(k)[1]), f"{pdir} must hold exactly V_{k}"
            out = EVAL / n / f"fold{k}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(ROOT / "MIA" / "scripts" / "eval_protocol_metrics.py"),
                            "--pred_dir", str(pdir), "--gt_cache", str(CACHE_DIR), "--pred_binary_Y",
                            "--no_hu", "--csv", str(out)], check=True, stdout=subprocess.DEVNULL)
            print(f"scored {n} fold {k} -> {out}")


def load(name):
    rows = {}
    for k in DEV_FOLDS:
        p = EVAL / name / f"fold{k}.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["case_id"] == "MEAN":
                    continue
                rows[(k, r["case_id"])] = {key: float(r[key]) for key in KEYS if key in r and r[key] != ""}
    return rows


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(names, ref, out_json=None):
    data = {n: load(n) for n in set(names) | ({ref} if ref else set())}
    res = {}
    hdr = f"{'candidate':<22}{'folds':>6}{'n':>4}{'Dice_Y':>8}{'f0':>8}{'f1':>8}{'SurfD':>8}{'HD95':>8}{'HD95md':>8}"
    print(hdr + (f"   dDice vs {ref} [95% CI]" if ref else ""))
    for n in names:
        r = data[n]
        if not r:
            print(f"{n:<22} (not scored)")
            continue
        folds = sorted({k for k, _ in r})
        col = lambda key: np.array([v.get(key, np.nan) for v in r.values()])  # noqa: E731
        per = {k: float(np.mean([v["Dice_Y"] for (kk, _), v in r.items() if kk == k])) for k in folds}
        rec = {"folds": folds, "n": len(r), "per_fold": per,
               **{key: float(np.nanmean(col(key))) for key in KEYS}, "HD95_median": float(np.nanmedian(col("HD95_Y")))}
        txt = ""
        if ref and n != ref and data[ref]:
            common = sorted(set(r) & set(data[ref]))
            d = np.array([r[c]["Dice_Y"] - data[ref][c]["Dice_Y"] for c in common])
            dh = np.array([r[c]["HD95_Y"] - data[ref][c]["HD95_Y"] for c in common])
            lo, hi = boot_ci(d)
            hlo, hhi = boot_ci(dh[np.isfinite(dh)])
            rec.update(ref=ref, n_paired=len(common), dDice=float(d.mean()), dDice_ci=[lo, hi],
                       dHD95=float(np.nanmean(dh)), dHD95_ci=[hlo, hhi])
            txt = f"   {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  dHD95 {np.nanmean(dh):+.2f} n={len(common)}"
        res[n] = rec
        print(f"{n:<22}{''.join(map(str, folds)):>6}{len(r):>4}{rec['Dice_Y']:>8.4f}{per.get(0, np.nan):>8.4f}"
              f"{per.get(1, np.nan):>8.4f}{rec['SurfDice_Y']:>8.4f}{rec['HD95_Y']:>8.2f}{rec['HD95_median']:>8.2f}" + txt)
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "summary"])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ref", default="r0_dev")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.names)
    else:
        summary(a.names, a.ref, a.json)


if __name__ == "__main__":
    main()
