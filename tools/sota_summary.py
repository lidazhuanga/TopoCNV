#!/usr/bin/env python3
"""
SOTA comparison table (protocol v0.3 §14.4, binary Y). CPU only, no training.

For every comparison method, scores the out-of-fold labels in
results/sota_binary_y_<method>_label_predictions with MIA/scripts/eval_protocol_metrics.py
(read-only call; Primary Dice_Y/HD95_Y, L1, v0.4 SurfDice_Y / Recall_vis / Dice_vis). Single-stage binary methods have no multiclass
output, so Secondary (Dice_123, Dice_c4) is N/A, as for P0 B0. The table adds the P0 Full and
B0 rows from the P0 evaluation (read only) and gives paired Wilcoxon p vs P0 Full.

  python tools/sota_summary.py                     # all methods; re-scores only when preds changed
  python tools/sota_summary.py --methods nnunetv2
Outputs: results/sota_binary_y_eval/<method>.csv and results/sota_binary_y_eval/SOTA_main_table.md
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
RESULTS = Path(os.environ.get("CNV_RESULTS_ROOT", ROOT / "results"))
OUT = RESULTS / "sota_binary_y_eval"
GT_CACHE = RESULTS / "preprocess_cache" / "sp_2.500_1.133_1.133"
P0_EVAL = [RESULTS / "p0_binary_y_eval_v2", RESULTS / "p0_binary_y_eval"]  # newest scoring first
METHODS = ["nnunetv2", "nnunet", "segresnet", "attention_unet", "unetr", "swin_unetr", "nnformer"]
LABELS = {"nnunetv2": "nnU-Net v2 (3d_fullres, 300 epochs)", "nnunet": "DynUNet (nnU-Net-style)",
          "segresnet": "SegResNet", "attention_unet": "Attention U-Net", "unetr": "UNETR",
          "swin_unetr": "SwinUNETR", "nnformer": "nnFormer"}
COLS = ["Dice_Y", "HD95_Y", "SurfDice_Y", "Dice_L", "Precision_L", "Recall_L", "L_over_Y",
        "Recall_vis", "Dice_vis", "Dice_123", "Dice_c4", "wDSC_Y_0.50", "Recall_GT3"]   # C3W secondaries (V4-AM-C3W-01)
P_COLS = ["Dice_Y", "HD95_Y", "SurfDice_Y", "Dice_L", "Dice_vis"]
# Columns a CSV must have to count as current; CSVs from an older eval version are re-scored.
REQUIRED = set(COLS) | {"Lvis_over_Y", "Empty_pred_Y", "wDSC_Y_0.50", "wDSC_Y_0.25",  # noqa
               "wDSC_Y_0.75", "Recall_GT3"}   # v2 2026-10-24; C3W 2026-10-26


def pred_dir(method):
    return RESULTS / f"sota_binary_y_{method}_label_predictions"


def folds_done(method):
    return sum((RESULTS / f"sota_binary_y_{method}" / f"fold{f}" / "COMPLETED").exists() for f in range(5))


def _read(path):
    with open(path) as f:
        # eval_protocol_metrics appends a MEAN row; keep per-case rows only
        return {r["case_id"]: r for r in csv.DictReader(f) if r["case_id"] != "MEAN"}


def _stale(out_csv, newest):
    if not out_csv.exists() or out_csv.stat().st_mtime < newest:
        return True
    with open(out_csv) as f:
        header = next(csv.reader(f), [])
    return not REQUIRED <= set(header)


def score(method, out_dir, force=False):
    pdir = pred_dir(method)
    preds = sorted(pdir.glob("*_stage2_label.nrrd")) if pdir.exists() else []
    if not preds:
        return None
    out_csv = out_dir / f"{method}.csv"
    if force or _stale(out_csv, max(p.stat().st_mtime for p in preds)):
        tmp = out_dir / f".{method}.{os.getpid()}.csv"
        cmd = [sys.executable, str(ROOT / "MIA/scripts/eval_protocol_metrics.py"),
               "--pred_dir", str(pdir), "--gt_cache", str(GT_CACHE),
               "--data_root", str(os.environ.get("CNV_DATA_ROOT", ROOT / "npc_data/train")),
               "--pred_binary_Y", "--csv", str(tmp)]
        try:
            with open(out_dir / f"{method}.log", "w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            os.replace(tmp, out_csv)
        except Exception:
            print(f"[sota_summary] scoring {method} failed (see {out_dir / (method + '.log')}); "
                  f"keeping previous CSV if any", file=sys.stderr)
            traceback.print_exc()
            tmp.unlink(missing_ok=True)
            if not out_csv.exists():
                return None
    return _read(out_csv)


def p0_rows(arm):
    for d in P0_EVAL:
        if (d / f"{arm}.csv").exists():
            return _read(d / f"{arm}.csv"), d
    return None, None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _cell(rows, col):
    v = np.array([fnum(x.get(col)) for x in rows])
    v = v[~np.isnan(v)]
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else "N/A"


def _wilcoxon(ref, r, col, common):
    from scipy.stats import wilcoxon
    a = np.array([fnum(ref[k].get(col)) for k in common])
    b = np.array([fnum(r[k].get(col)) for k in common])
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10 or np.all(a[ok] == b[ok]):
        return "—"
    return f"{wilcoxon(a[ok], b[ok]).pvalue:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = {m: score(m, OUT, args.force) for m in args.methods.split(",")}
    full, p0_dir = p0_rows("Full")
    b0, _ = p0_rows("B0")
    lines = [f"# SOTA comparison (binary Y, protocol v0.3 §14.4) — {datetime.now():%Y-%m-%d %H:%M}", "",
             "Out-of-fold predictions on the preprocess cache grid; mean ± sd over scored cases. "
             "p = paired Wilcoxon vs P0 Full on the cases scored in both (n in brackets). "
             "Secondary (Dice_123, Dice_c4) is N/A for single-stage binary methods. "
             f"P0 rows are read from {p0_dir.name if p0_dir else '—'}/ and follow that table's fold count. "
             "Only rows with folds 5/5 are final.", "",
             "| Method | folds | n | " + " | ".join(COLS) + " | " + " | ".join(f"p {c}" for c in P_COLS) + " |",
             "|---|---|---|" + "---|" * (len(COLS) + len(P_COLS))]
    for name, r in (("P0 Full (proposed)", full), ("P0 B0 (single-stage)", b0)):
        if r:
            lines.append(f"| {name} | — | {len(r)} | " + " | ".join(_cell(r.values(), c) for c in COLS)
                         + " |" + " — |" * len(P_COLS))
    for m, r in rows.items():
        if not r:
            lines.append(f"| {LABELS[m]} | {folds_done(m)}/5 | 0 |" + " — |" * (len(COLS) + len(P_COLS)))
            continue
        ps = ["—"] * len(P_COLS)
        if full:
            common = sorted(set(full) & set(r))
            ps = [f"{_wilcoxon(full, r, c, common)} (n={len(common)})" for c in P_COLS]
        lines.append(f"| {LABELS[m]} | {folds_done(m)}/5 | {len(r)} | "
                     + " | ".join(_cell(r.values(), c) for c in COLS) + " | " + " | ".join(ps) + " |")

    md = OUT / "SOTA_main_table.md"
    tmp = OUT / f".SOTA_main_table.{os.getpid()}.md"
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
