#!/usr/bin/env python3
"""
Secondary (Dice_123 / Dice_c4) diagnosis and candidate definitions. CPU only, analysis only:
nothing here feeds the P0 main table until the definition is chosen.

Stage-1 diagnosis (per case, all Stage-1 OOF cases):
  S1 Y volume vs GT Y volume, share of S1 Y voxels > 10 / 30 mm from GT Y,
  S1 Y Dice over the whole volume and inside a 10 mm corridor around GT Y.

Candidate Secondary definitions (per cascade arm, per scored case; B0 stays N/A):
  S0_full   current: Stage-1 argmax over the whole volume
  A_inY     Stage-2 Y partitioned by Stage-1 posteriors: inside the Stage-2 Y,
            class = argmax_{c in 1..4} P_S1(c); outside it, background.
            Union of 1..4 equals the Primary prediction by construction.
  B_roiY    Stage-1 argmax kept only inside dilate(Stage-2 Y, 10 mm)
  C_gtROI   Stage-1 argmax kept only inside dilate(GT Y, 10 mm)  (uses GT: diagnostic only)

  python tools/secondary_options.py --workers 12
Outputs: results/p0_binary_y_eval_v2/secondary/{s1_diagnosis.csv, options_<arm>.csv, summary.md}
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
import ablation_configs as ac  # noqa: E402
from topocnv.metrics.protocol import DEFAULT_SPACING_ZYX, _dice, dilate_mm  # noqa: E402

GT_CACHE = Path(ac.RESULTS_ROOT) / "preprocess_cache" / "sp_2.500_1.133_1.133"
S1_DIR = Path(ac.STAGE1_CV_DIR)
OUT = Path(ac.RESULTS_ROOT) / "p0_binary_y_eval_v2" / "secondary"
SP = DEFAULT_SPACING_ZYX
ROI_MM = 10.0
OPTIONS = ["S0_full", "A_inY", "B_roiY", "C_gtROI"]


def _nrrd(path):
    import nrrd
    return nrrd.read(str(path))[0]


def _bbox(mask, pad):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def diagnose(case_id):
    gt = np.load(GT_CACHE / f"{case_id}_label.npy")
    s1 = _nrrd(S1_DIR / f"{case_id}_stage1_label.nrrd")
    gY, sY = (gt >= 1) & (gt <= 4), (s1 >= 1) & (s1 <= 4)
    dist = distance_transform_edt(~gY, sampling=SP)
    roi = dist <= ROI_MM
    d_s = dist[sY]
    n_cc = cc_label(sY)[1]
    return {
        "case_id": case_id, "gt_Y": int(gY.sum()), "s1_Y": int(sY.sum()),
        "s1_over_gt": float(sY.sum() / max(gY.sum(), 1)),
        "frac_s1Y_gt10mm": float((d_s > 10).mean()) if d_s.size else float("nan"),
        "frac_s1Y_gt30mm": float((d_s > 30).mean()) if d_s.size else float("nan"),
        "s1_Y_components": int(n_cc),
        "s1_DiceY_full": _dice(sY, gY), "s1_DiceY_gtROI": _dice(sY & roi, gY),
        "s1_Dice123_full": _dice((s1 >= 1) & (s1 <= 3), (gt >= 1) & (gt <= 3)),
        "s1_Dice123_gtROI": _dice((s1 >= 1) & (s1 <= 3) & roi, (gt >= 1) & (gt <= 3)),
        "s1_Dicec4_full": _dice(s1 == 4, gt == 4),
        "s1_Dicec4_gtROI": _dice((s1 == 4) & roi, gt == 4),
    }


def options(args):
    arm, case_id, pred_path = args
    gt = np.load(GT_CACHE / f"{case_id}_label.npy")
    s1 = _nrrd(S1_DIR / f"{case_id}_stage1_label.nrrd")
    pY = _nrrd(pred_path) > 0
    g123, gc4 = (gt >= 1) & (gt <= 3), gt == 4
    lab = {"S0_full": s1}
    # A: partition the Stage-2 Y by Stage-1 posteriors over classes 1..4
    a = np.zeros_like(s1)
    if pY.any():
        bb = _bbox(pY, 0)
        sm = np.load(S1_DIR / f"{case_id}_stage1_softmax.npy", mmap_mode="r")
        sub = np.asarray(sm[1:5, bb[0], bb[1], bb[2]])
        part = (np.argmax(sub, axis=0) + 1).astype(s1.dtype)
        a[bb] = np.where(pY[bb], part, 0)
    lab["A_inY"] = a
    lab["B_roiY"] = np.where(dilate_mm(pY, SP, ROI_MM), s1, 0)
    lab["C_gtROI"] = np.where(distance_transform_edt(~((gt >= 1) & (gt <= 4)), sampling=SP) <= ROI_MM, s1, 0)
    row = {"arm": arm, "case_id": case_id, "pred_Y": int(pY.sum())}
    for k, m in lab.items():
        row[f"{k}_Dice_123"] = _dice((m >= 1) & (m <= 3), g123)
        row[f"{k}_Dice_c4"] = _dice(m == 4, gc4)
    # A_inY consistency check: union of its classes must equal the Primary prediction
    row["A_inY_equals_predY"] = bool(np.array_equal((a >= 1) & (a <= 4), pY))
    return row


def _write(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _ms(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default=",".join(n for n in ac.ROW_ORDER if not ac.CONFIGS[n]["b0_single_stage"]))
    ap.add_argument("--skip_diag", action="store_true", help="reuse s1_diagnosis.csv")
    ap.add_argument("--summary_only", action="store_true", help="rebuild summary.md from existing CSVs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Secondary diagnosis and candidate definitions", ""]

    with Pool(args.workers) as pool:
        diag = None
        if not (args.skip_diag or args.summary_only):
            cases = sorted(p.name.replace("_stage1_label.nrrd", "") for p in S1_DIR.glob("*_stage1_label.nrrd"))
            diag = pool.map(diagnose, cases)
            _write(OUT / "s1_diagnosis.csv", diag)
        elif (OUT / "s1_diagnosis.csv").exists():
            with open(OUT / "s1_diagnosis.csv") as f:
                diag = list(csv.DictReader(f))
        if diag:
            md += [f"## Stage-1 OOF, full volume (n={len(diag)})", "", "| metric | mean ± sd | median |", "|---|---|---|"]
            for k in list(diag[0])[1:]:
                v = np.array([r[k] for r in diag], dtype=float)
                md.append(f"| {k} | {_ms(v)} | {np.nanmedian(v):.4f} |")
            md.append("")

        md += [f"## Candidate Secondary per arm (ROI = {ROI_MM:g} mm)", "",
               "| Arm | n | " + " | ".join(f"{o} Dice_123 | {o} Dice_c4" for o in OPTIONS) + " | A_inY ≡ pred Y |",
               "|---|---|" + "---|---|" * len(OPTIONS) + "---|"]
        for arm in args.arms.split(","):
            pdir = Path(ac.RESULTS_ROOT) / f"{ac.stage2_exp(arm)}_label_predictions"
            jobs = [(arm, p.name.replace("_stage2_label.nrrd", ""), p) for p in sorted(pdir.glob("*_stage2_label.nrrd"))]
            if not jobs:
                continue
            if args.summary_only:
                if not (OUT / f"options_{arm}.csv").exists():
                    continue
                with open(OUT / f"options_{arm}.csv") as f:
                    rows = [{k: (v == "True" if k == "A_inY_equals_predY" else v) for k, v in r.items()}
                            for r in csv.DictReader(f)]
            else:
                rows = pool.map(options, jobs)
                _write(OUT / f"options_{arm}.csv", rows)
            cells = []
            for o in OPTIONS:
                cells += [_ms([r[f"{o}_Dice_123"] for r in rows]), _ms([r[f"{o}_Dice_c4"] for r in rows])]
            ok = sum(r["A_inY_equals_predY"] for r in rows)
            md.append(f"| {arm} | {len(rows)} | " + " | ".join(cells) + f" | {ok}/{len(rows)} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
