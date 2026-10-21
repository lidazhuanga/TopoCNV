#!/usr/bin/env python3
"""
P0 main-table builder (protocol v0.3, binary Y). CPU only, no training.

For every P0 arm, scores the out-of-fold Stage-2 labels in <exp>_label_predictions with
MIA/scripts/eval_protocol_metrics.py (Primary Dice_Y/HD95_Y, L1, Secondary), then writes
a markdown table with mean ± sd, folds completed, and paired Wilcoxon p (Full vs arm, Dice_Y).

  python tools/p0_summary.py                 # all arms; re-scores only when preds changed
  python tools/p0_summary.py --arms Full,C1  # subset
Outputs: results/p0_binary_y_eval/<arm>.csv and results/p0_binary_y_eval/P0_main_table.md
"""
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
import ablation_configs as ac  # noqa: E402

OUT = Path(ac.RESULTS_ROOT) / "p0_binary_y_eval"
GT_CACHE = Path(ac.RESULTS_ROOT) / "preprocess_cache" / "sp_2.500_1.133_1.133"
COLS = ["Dice_Y", "HD95_Y", "Dice_L", "Recall_L", "L_over_Y", "Dice_123", "Dice_c4"]


def pred_dir(arm):
    return Path(ac.RESULTS_ROOT) / f"{ac.stage2_exp(arm)}_label_predictions"


def folds_done(arm):
    return sum((Path(ac.RESULTS_ROOT) / ac.stage2_exp(arm) / f"fold{f}" / "COMPLETED").exists()
               for f in range(ac.NUM_FOLDS))


def score(arm, force=False):
    pdir = pred_dir(arm)
    preds = sorted(pdir.glob("*_stage2_label.nrrd")) if pdir.exists() else []
    if not preds:
        return None
    out_csv = OUT / f"{arm}.csv"
    newest = max(p.stat().st_mtime for p in preds)
    if force or not out_csv.exists() or out_csv.stat().st_mtime < newest:
        cmd = [sys.executable, str(ROOT / "MIA/scripts/eval_protocol_metrics.py"),
               "--pred_dir", str(pdir), "--gt_cache", str(GT_CACHE),
               "--data_root", str(os.environ.get("CNV_DATA_ROOT", ROOT / "npc_data/train")),
               "--pred_binary_Y", "--csv", str(out_csv)]
        if not ac.CONFIGS[arm]["b0_single_stage"]:
            cmd += ["--s1_pred_dir", ac.STAGE1_CV_DIR]
        with open(OUT / f"{arm}.log", "w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with open(out_csv) as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ac.ROW_ORDER))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = {a: score(a, args.force) for a in args.arms.split(",")}
    lines = [f"# P0 main table (binary Y, protocol v0.3) — {datetime.now():%Y-%m-%d %H:%M}", "",
             "Out-of-fold Stage-2 predictions; mean ± sd over scored cases. "
             "p = paired Wilcoxon vs Full on Dice_Y (cases scored in both).", "",
             "| Arm | folds | n | " + " | ".join(COLS) + " | p (Dice_Y vs Full) |",
             "|---|---|---|" + "---|" * len(COLS) + "---|"]
    full = rows.get("Full")
    for arm, r in rows.items():
        if not r:
            lines.append(f"| {arm} | {folds_done(arm)}/5 | 0 |" + " — |" * len(COLS) + " — |")
            continue
        cells = []
        for c in COLS:
            v = np.array([fnum(x[c]) for x in r.values()])
            v = v[~np.isnan(v)]
            cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else "N/A")
        p = "—"
        if full and arm != "Full":
            common = sorted(set(full) & set(r))
            if len(common) >= 10:
                from scipy.stats import wilcoxon
                a = [fnum(full[k]["Dice_Y"]) for k in common]
                b = [fnum(r[k]["Dice_Y"]) for k in common]
                p = f"{wilcoxon(a, b).pvalue:.3g} (n={len(common)})"
        lines.append(f"| {arm} | {folds_done(arm)}/5 | {len(r)} | " + " | ".join(cells) + f" | {p} |")
    (OUT / "P0_main_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
