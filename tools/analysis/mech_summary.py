#!/usr/bin/env python3
"""Tables for the mechanism analysis from results/analysis_mech/cases/*.json.

    python tools/analysis/mech_summary.py [--in results/analysis_mech]

Writes summary.md (tables) and long-format CSVs next to the inputs.
Per-case values are summarised as mean ± sd and median; paired comparisons
use the Wilcoxon signed-rank test on cases present in both groups.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=None)
A = ap.parse_args()
ROOT = Path(A.inp or Path(__file__).resolve().parents[2] / "results" / "analysis_mech")
REGS = ["YH", "L", "Yglow", "Yghigh", "ring5", "bg20"]
ARMS = ["Full", "A_gate", "A_mem"]

rows = [json.loads(p.read_text()) for p in sorted((ROOT / "cases").glob("*.json"))]
by = defaultdict(dict)                       # arm -> case -> row
for r in rows:
    by[r["arm"]][r["case"]] = r


def folds_of(arm):
    return sorted({r["fold"] for r in by[arm].values()})


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return "—"
    return f"{v.mean():.3f} ± {v.std(ddof=1) if v.size > 1 else 0:.3f}"


def med(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]" if v.size else "—"


def pval(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
        return "—"
    return f"{wilcoxon(a[ok], b[ok]).pvalue:.2g}"


def get(r, *keys):
    for k in keys:
        if r is None:
            return np.nan
        r = r.get(k) if isinstance(r, dict) else None
    return np.nan if r is None else r


L_ = []
L_.append(f"# Mechanism analysis — summary tables\n\nSource: `{ROOT}/cases` ({len(rows)} case-runs).\n")
L_.append("| Arm | folds | n cases | best epochs |\n|---|---|---|---|")
for arm in ARMS:
    if by[arm]:
        eps = sorted({(r['fold'], r['best_epoch']) for r in by[arm].values()})
        L_.append(f"| {arm} | {','.join(map(str, folds_of(arm)))} | {len(by[arm])} | "
                  + ", ".join(f"f{f}:{e}" for f, e in eps) + " |")

# ---------- 1. segmentation metrics per condition ----------
MET = ["Dice_Y", "HD95_Y", "Recall_L", "Dice_LnotH", "Recall_YH", "Recall_YE", "Precision_Y"]
CONDS = ["orig", "nomem", "E1", "E1_nomem", "E2", "E2_nomem", "orig_memreadout"]
L_.append("\n## 1. Segmentation metrics by condition (mean ± sd over cases; HD95 median [IQR])\n")
L_.append("| Arm | condition | n | " + " | ".join(MET) + " |\n|---|---|---|" + "---|" * len(MET))
for arm in ARMS:
    for c in CONDS:
        vals = [r["metrics"].get(c) for r in by[arm].values()]
        vals = [v for v in vals if v]
        if not vals:
            continue
        cells = [med([v[m] for v in vals]) if m == "HD95_Y" else ms([v.get(m) for v in vals]) for m in MET]
        L_.append(f"| {arm} | {c} | {len(vals)} | " + " | ".join(cells) + " |")

# ---------- 2. effect of removing memory at inference ----------
L_.append("\n## 2. Removing the memory branch at inference (orig − nomem), per case\n")
L_.append("| Arm | n | ΔDice_Y | p | ΔRecall_L | p | ΔRecall_YH | p | ΔDice_LnotH | p |\n|---|---|---|---|---|---|---|---|---|---|")
for arm in ("Full", "A_gate"):
    R = [r for r in by[arm].values() if "nomem" in r["metrics"]]
    if not R:
        continue
    cells = []
    for m in ("Dice_Y", "Recall_L", "Recall_YH", "Dice_LnotH"):
        a = [r["metrics"]["orig"][m] for r in R]
        b = [r["metrics"]["nomem"][m] for r in R]
        cells += [ms(np.subtract(a, b)), pval(a, b)]
    L_.append(f"| {arm} | {len(R)} | " + " | ".join(cells) + " |")

# ---------- 3. memory reliance by region ----------
L_.append("\n## 3. Memory reliance by region (orig input)\n")
L_.append("Token statistics are mapped back to voxels (nearest upsampling, sliding-window weights) and "
          "averaged within each region. `gate` = effective g at the level; `share` = ||(1−g)f_mem|| / "
          "(||g f_loc|| + ||(1−g)f_mem||); `shift` = ||f − f_loc|| / ||f_loc||; `dmem` = ||f_mem − f_loc|| / "
          "||f_loc|| (what memory proposes before gating); `pfg` = retrieval mass on CN V prototypes. "
          "|Δp| and flip compare output probabilities with and without the memory branch.\n")
STAT = ["gate", "share", "shift", "dmem", "pfg"]
for arm in ("Full", "A_gate"):
    R = list(by[arm].values())
    if not R:
        continue
    L_.append(f"\n### {arm} (n = {len(R)})\n")
    hdr = ["region"] + [f"{s}{lv}" for s in STAT for lv in range(3)] + ["|Δp|", "Δp signed", "flip"]
    L_.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for reg in REGS:
        cells = []
        for s in STAT:
            for lv in range(3):
                cells.append(f"{np.nanmean([get(r, 'tok', 'orig', reg, f'{s}{lv}') for r in R]):.3f}")
        d = [get(r, "delta", "orig-nomem", reg) for r in R]
        for k in ("abs", "signed", "flip"):
            cells.append(f"{np.nanmean([get(x, k) if isinstance(x, dict) else np.nan for x in d]):.4f}")
        L_.append(f"| {reg} | " + " | ".join(cells) + " |")
    # paired L vs YH contrast
    for q, key in (("shift0", ("tok", "orig")), ("abs", ("delta", "orig-nomem"))):
        a = [get(r, *key, "L", q) for r in R]
        b = [get(r, *key, "YH", q) for r in R]
        ratio = np.divide(a, b)
        L_.append(f"\n{arm}: {q} in L vs Y∩H — L/YH ratio median {np.nanmedian(ratio):.2f}, "
                  f"L > YH in {int(np.nansum(np.greater(a, b)))}/{len(R)} cases, Wilcoxon p = {pval(a, b)}")

# ---------- 4. L2 erasure ----------
L_.append("\n## 4. L2 evidence erasure (drop = orig − erased; positive = worse after erasure)\n")
L_.append("| Arm | erasure | memory at inference | n | drop Dice_Y | drop Recall_YE | drop Recall_YH | drop Recall_L |\n|---|---|---|---|---|---|---|---|")
DROPS = {}
for arm in ARMS:
    R = list(by[arm].values())
    for e in ("E1", "E2"):
        for base, cond, memlab in (("orig", e, "on"), ("nomem", f"{e}_nomem", "off")):
            R2 = [r for r in R if cond in r["metrics"]]
            if not R2:
                continue
            if arm == "A_mem":
                memlab = "none (arm)"
            cells = []
            for m in ("Dice_Y", "Recall_YE", "Recall_YH", "Recall_L"):
                d = [r["metrics"][base][m] - r["metrics"][cond][m] for r in R2]
                DROPS[(arm, e, memlab, m)] = {r["case"]: x for r, x in zip(R2, d)}
                cells.append(ms(d))
            L_.append(f"| {arm} | {e} | {memlab} | {len(R2)} | " + " | ".join(cells) + " |")

L_.append("\nPaired arm comparisons of the drop (cases present in both arms):\n")
L_.append("| comparison | erasure | metric | n | mean drop (arm 1) | mean drop (arm 2) | p |\n|---|---|---|---|---|---|---|")
for e in ("E1", "E2"):
    for m in ("Dice_Y", "Recall_YE", "Recall_L"):
        for a1, a2 in (("Full", "A_mem"), ("Full", "A_gate"), ("A_gate", "A_mem")):
            k1 = (a1, e, "on" if a1 != "A_mem" else "none (arm)", m)
            k2 = (a2, e, "on" if a2 != "A_mem" else "none (arm)", m)
            if k1 not in DROPS or k2 not in DROPS:
                continue
            cs = sorted(set(DROPS[k1]) & set(DROPS[k2]))
            x = [DROPS[k1][c] for c in cs]
            y = [DROPS[k2][c] for c in cs]
            L_.append(f"| {a1} vs {a2} | {e} | {m} | {len(cs)} | {np.mean(x):.4f} | {np.mean(y):.4f} | {pval(x, y)} |")
L_.append("\nMemory × erasure interaction within an arm (drop with memory − drop without memory; "
          "negative = memory reduces the damage):\n")
L_.append("| Arm | erasure | metric | n | interaction | p |\n|---|---|---|---|---|---|")
for arm in ("Full", "A_gate"):
    for e in ("E1", "E2"):
        for m in ("Dice_Y", "Recall_YE", "Recall_L"):
            k1, k2 = (arm, e, "on", m), (arm, e, "off", m)
            if k1 in DROPS and k2 in DROPS:
                cs = sorted(set(DROPS[k1]) & set(DROPS[k2]))
                x = [DROPS[k1][c] for c in cs]
                y = [DROPS[k2][c] for c in cs]
                L_.append(f"| {arm} | {e} | {m} | {len(cs)} | {ms(np.subtract(x, y))} | {pval(x, y)} |")

# gate / token change inside E under erasure
L_.append("\nEffective gate and memory shift inside Y∩E before and after erasure (level 0):\n")
L_.append("| Arm | condition | g (voxel) | gate0 | share0 | shift0 | pfg0 |\n|---|---|---|---|---|---|---|")
for arm in ("Full", "A_gate"):
    R = list(by[arm].values())
    for c in ("orig", "E1", "E2"):
        vals = [np.nanmean([get(r, "tok", c, "YE", f"{s}0") for r in R]) for s in ("gate", "share", "shift", "pfg")]
        L_.append(f"| {arm} | {c} | — | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

(ROOT / "summary.md").write_text("\n".join(L_) + "\n")
print("\n".join(L_))
