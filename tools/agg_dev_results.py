#!/usr/bin/env python
"""Aggregate v4 dev eval CSVs: per-fold means + paired per-case deltas vs r0_dev."""
import csv, math, os
from collections import defaultdict

BASE = "results/v4_dev_preds"
KEYS = ["Dice_Y", "HD95_Y", "wDSC_Y_0.50", "Recall_GT3"]

def load(run):
    out = {}
    for fd in sorted(os.listdir(os.path.join(BASE, run))):
        p = os.path.join(BASE, run, fd, "eval.csv")
        if not os.path.exists(p):
            continue
        rows = {}
        with open(p) as f:
            for r in csv.DictReader(f):
                rows[r["case_id"]] = r
        out[fd] = rows
    return out

def fnum(r, k):
    v = r.get(k, "")
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

runs = {r: load(r) for r in ["r0_dev", "r1", "r1p_abc", "r1p_bc"] if os.path.isdir(os.path.join(BASE, r))}

print("=== per-fold means ===")
print(f"{'run':10s} {'fold':7s} {'n':>3s} " + " ".join(f"{k:>12s}" for k in KEYS) + " empty")
per_run = defaultdict(dict)
for run, folds in runs.items():
    for fd, rows in sorted(folds.items()):
        vals = {k: mean([fnum(r, k) for r in rows.values()]) for k in KEYS}
        emp = sum(1 for r in rows.values() if r.get("Empty_pred_Y", "0") not in ("0", "0.0", "", "False"))
        per_run[run][fd] = vals
        cells = " ".join(f"{vals[k]:12.4f}" if vals[k] is not None else f"{'--':>12s}" for k in KEYS)
        print(f"{run:10s} {fd:7s} {len(rows):3d} {cells} {emp}")

print("\n=== run means over available folds ===")
for run, folds in per_run.items():
    cells = " ".join(f"{mean([v[k] for v in folds.values()]):12.4f}" if mean([v[k] for v in folds.values()]) is not None else f"{'--':>12s}" for k in KEYS)
    print(f"{run:10s} ({len(folds)} folds) {cells}")

print("\n=== paired deltas vs r0_dev (common cases per fold) ===")
for run in ["r1", "r1p_abc", "r1p_bc"]:
    if run not in runs:
        continue
    for fd in sorted(runs[run]):
        if fd not in runs.get("r0_dev", {}):
            continue
        base = runs["r0_dev"][fd]
        cur = runs[run][fd]
        common = sorted(set(base) & set(cur))
        for k in ["Dice_Y", "HD95_Y", "wDSC_Y_0.50"]:
            d = [fnum(cur[c], k) - fnum(base[c], k) for c in common if fnum(cur[c], k) is not None and fnum(base[c], k) is not None]
            if not d:
                continue
            win = sum(1 for x in d if x > 0)
            print(f"{run:10s} {fd} d{k:12s} n={len(d):2d} mean={sum(d)/len(d):+.4f} win={win}/{len(d)}")

print("\n=== paired deltas r1p_* vs r1 (common cases per fold) ===")
for run in ["r1p_abc", "r1p_bc"]:
    if run not in runs or "r1" not in runs:
        continue
    for fd in sorted(runs[run]):
        if fd not in runs["r1"]:
            continue
        base = runs["r1"][fd]
        cur = runs[run][fd]
        common = sorted(set(base) & set(cur))
        for k in ["Dice_Y", "HD95_Y", "wDSC_Y_0.50"]:
            d = [fnum(cur[c], k) - fnum(base[c], k) for c in common if fnum(cur[c], k) is not None and fnum(base[c], k) is not None]
            if not d:
                continue
            win = sum(1 for x in d if x > 0)
            print(f"{run:10s} {fd} d{k:12s} n={len(d):2d} mean={sum(d)/len(d):+.4f} win={win}/{len(d)}")

print("\n=== worst 5 cases by Dice_Y (r1, across evaluated folds) ===")
allc = []
for fd, rows in runs.get("r1", {}).items():
    for c, r in rows.items():
        d = fnum(r, "Dice_Y")
        if d is not None:
            b = fnum(runs["r0_dev"].get(fd, {}).get(c, {}), "Dice_Y")
            allc.append((d, b, fd, c))
for d, b, fd, c in sorted(allc)[:5]:
    bs = f"{b:.4f}" if b is not None else "--"
    print(f"  {fd} {c}: r1={d:.4f} r0_dev={bs}")
