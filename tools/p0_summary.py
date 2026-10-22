#!/usr/bin/env python3
"""
P0 main-table builder (protocol v0.3, binary Y). CPU only, no training.

For every P0 arm, scores the out-of-fold Stage-2 labels in <exp>_label_predictions with
MIA/scripts/eval_protocol_metrics.py (Primary Dice_Y/HD95_Y, L1, Secondary), then writes
a markdown table with mean ± sd, folds completed, and paired Wilcoxon p vs Full
(Dice_Y and Dice_L), plus a second table restricted to the cases scored in every arm.

L1 columns follow the v2 scoring (docs/METRICS_FIX_NOTES.md): Dice_L counts false
positives outside H_bone; Dice_L_legacy is the v1 FN-only score (= 2R/(1+R)).

  python tools/p0_summary.py                 # all arms; re-scores only when preds changed
  python tools/p0_summary.py --arms Full,C1  # subset
Outputs: results/p0_binary_y_eval_v2/<arm>.csv and results/p0_binary_y_eval_v2/P0_main_table.md
(results/p0_binary_y_eval/ holds the v1 tables and is no longer written.)
"""
import argparse
import csv
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
import ablation_configs as ac  # noqa: E402

OUT = Path(ac.RESULTS_ROOT) / "p0_binary_y_eval_v2"
GT_CACHE = Path(ac.RESULTS_ROOT) / "preprocess_cache" / "sp_2.500_1.133_1.133"
COLS = ["Dice_Y", "HD95_Y", "Dice_L", "Precision_L", "Recall_L", "L_over_Y",
        "Dice_L_legacy", "Dice_123", "Dice_c4"]
COMMON_COLS = ["Dice_Y", "HD95_Y", "Dice_L", "Precision_L", "Recall_L", "Dice_L_legacy"]
P_COLS = ["Dice_Y", "Dice_L"]


def pred_dir(arm):
    return Path(ac.RESULTS_ROOT) / f"{ac.stage2_exp(arm)}_label_predictions"


def folds_done(arm):
    return sum((Path(ac.RESULTS_ROOT) / ac.stage2_exp(arm) / f"fold{f}" / "COMPLETED").exists()
               for f in range(ac.NUM_FOLDS))


def _read(out_csv):
    with open(out_csv) as f:
        # eval_protocol_metrics appends a MEAN row; keep per-case rows only
        return {r["case_id"]: r for r in csv.DictReader(f) if r["case_id"] != "MEAN"}


def _stale(out_csv, newest):
    if not out_csv.exists() or out_csv.stat().st_mtime < newest:
        return True
    with open(out_csv) as f:
        header = next(csv.reader(f), [])
    return not set(COLS) <= set(header)  # written by an older eval version


def score(arm, out_dir, force=False):
    pdir = pred_dir(arm)
    preds = sorted(pdir.glob("*_stage2_label.nrrd")) if pdir.exists() else []
    if not preds:
        return None
    out_csv = out_dir / f"{arm}.csv"
    if force or _stale(out_csv, max(p.stat().st_mtime for p in preds)):
        # Unique tmp name: the P0 queue may run this script while a manual run is going.
        tmp = out_dir / f".{arm}.{os.getpid()}.csv"
        cmd = [sys.executable, str(ROOT / "MIA/scripts/eval_protocol_metrics.py"),
               "--pred_dir", str(pdir), "--gt_cache", str(GT_CACHE),
               "--data_root", str(os.environ.get("CNV_DATA_ROOT", ROOT / "npc_data/train")),
               "--pred_binary_Y", "--csv", str(tmp)]
        if not ac.CONFIGS[arm]["b0_single_stage"]:
            cmd += ["--s1_pred_dir", ac.STAGE1_CV_DIR]
        try:
            with open(out_dir / f"{arm}.log", "w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            os.replace(tmp, out_csv)
        except Exception:
            print(f"[p0_summary] scoring {arm} failed (see {out_dir / (arm + '.log')}); "
                  f"keeping previous CSV if any", file=sys.stderr)
            traceback.print_exc()
            tmp.unlink(missing_ok=True)
            if not out_csv.exists():
                return None
    try:
        return _read(out_csv)
    except Exception:
        traceback.print_exc()
        return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _cell(rows, col):
    v = np.array([fnum(x.get(col)) for x in rows])
    v = v[~np.isnan(v)]
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else "N/A"


def _wilcoxon(full, r, col, common):
    if len(common) < 10:
        return "—"
    from scipy.stats import wilcoxon
    a = np.array([fnum(full[k].get(col)) for k in common])
    b = np.array([fnum(r[k].get(col)) for k in common])
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10 or np.all(a[ok] == b[ok]):
        return "—"
    return f"{wilcoxon(a[ok], b[ok]).pvalue:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ac.ROW_ORDER))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {a: score(a, out_dir, args.force) for a in args.arms.split(",")}
    full = rows.get("Full")
    lines = [f"# P0 main table (binary Y, protocol v0.3, L1 scoring v2) — {datetime.now():%Y-%m-%d %H:%M}", "",
             "Out-of-fold Stage-2 predictions; mean ± sd over scored cases. "
             "p = paired Wilcoxon vs Full on the cases scored in both (n in brackets). "
             "Dice_L = Dice(P \\ H, Y \\ H) counts false positives outside H; "
             "Dice_L_legacy = v1 score Dice(P ∩ L, L) = 2·Recall_L/(1+Recall_L). "
             "Only rows with folds 5/5 are final.", "",
             "| Arm | folds | n | " + " | ".join(COLS) + " | " + " | ".join(f"p {c}" for c in P_COLS) + " |",
             "|---|---|---|" + "---|" * (len(COLS) + len(P_COLS))]
    for arm, r in rows.items():
        if not r:
            lines.append(f"| {arm} | {folds_done(arm)}/5 | 0 |" + " — |" * (len(COLS) + len(P_COLS)))
            continue
        cells = [_cell(r.values(), c) for c in COLS]
        ps = ["—"] * len(P_COLS)
        if full and arm != "Full":
            common = sorted(set(full) & set(r))
            ps = [f"{_wilcoxon(full, r, c, common)} (n={len(common)})" for c in P_COLS]
        lines.append(f"| {arm} | {folds_done(arm)}/5 | {len(r)} | " + " | ".join(cells)
                     + " | " + " | ".join(ps) + " |")

    scored = {a: r for a, r in rows.items() if r}
    if len(scored) > 1:
        common = sorted(set.intersection(*(set(r) for r in scored.values())))
        lines += ["", f"## Common cases (scored in all {len(scored)} arms with results, n={len(common)})", "",
                  "| Arm | " + " | ".join(COMMON_COLS) + " |", "|---|" + "---|" * len(COMMON_COLS)]
        for arm, r in scored.items():
            lines.append(f"| {arm} | " + " | ".join(_cell([r[k] for k in common], c) for c in COMMON_COLS) + " |")

    md = out_dir / "P0_main_table.md"
    tmp = out_dir / f".P0_main_table.{os.getpid()}.md"
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
