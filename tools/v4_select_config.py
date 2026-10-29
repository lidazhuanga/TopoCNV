#!/usr/bin/env python3
"""Per-fold nested config selection (V4_PLAN §6 as revised 2026-10-24, nested CV).

After the dev candidates of outer fold k are trained on T_k - V_k and scored on V_k with
eval_protocol_metrics.py --pred_binary_Y (one CSV per candidate, e.g.
results/v4_dev_preds/<name>/fold<k>/eval.csv), this tool applies the pre-registered rules and
freezes fold k's configuration:

  1. Completeness gate (hard): every candidate CSV must contain exactly the V_k cases,
     no more, no less. Any mismatch aborts — no selection from partial results.
  2. Primary criterion: mean Dice_Y on V_k. The best candidate wins.
  3. Simplicity rule: a candidate whose mean Dice_Y is within TOL (default 0.003) of the best
     wins instead if its simplicity rank ("simple", 0 = simplest) is smaller.
  4. HD95_Y is computed for the record only; it never decides the winner.

Outputs:
  configs/v4_frozen/fold{k}.json   chosen candidate + its config dict, verbatim for the report tier
  log/v4/selection_fold{k}.json    the full table (all candidates, all metrics, the rule trace)

The report tier (tools/v4_nnunet.py train --variant report) refuses to run without the frozen file.

  python tools/v4_select_config.py --fold 0 --manifest configs/v4_dev/fold0_candidates.json

Manifest format (JSON):
  {
   "tol": 0.003,
   "candidates": [
     {"name": "r1p_bc",  "csv": "results/v4_dev_preds/r1p_bc/fold0/eval.csv", "simple": 2,
      "config": {"trainer": "nnUNetTrainerV4_Pbc", "margin": 25}},
     ...
   ]
  }
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from topocnv.data.v4_splits import internal_split  # noqa: E402


def read_dice_csv(path):
    """-> dict case_id -> (dice_y, hd95_y). Rows with a non-numeric Dice_Y abort."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("case_id") or "").strip()
            if not cid or cid.upper() == "MEAN":   # safety: no summary rows as data
                continue
            if cid in out:
                raise SystemExit(f"{path}: duplicate case_id {cid}")
            try:
                d = float(row["Dice_Y"])
            except (TypeError, ValueError):
                raise SystemExit(f"{path}: non-numeric Dice_Y for case {cid}")
            h = row.get("HD95_Y", "")
            try:
                h = float(h)
            except (TypeError, ValueError):
                h = None
            out[cid] = (d, h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tol", type=float, default=None, help="override the manifest tolerance")
    args = ap.parse_args()
    k = args.fold
    man = json.loads(Path(args.manifest).read_text())
    tol = args.tol if args.tol is not None else float(man.get("tol", 0.003))
    want = set(internal_split(k)[1])

    table = []
    for cand in man["candidates"]:
        path = ROOT / cand["csv"]
        per_case = read_dice_csv(path)
        got = set(per_case)
        if got != want:
            raise SystemExit(f"completeness gate failed for {cand['name']} ({path}):\n"
                             f"  missing {sorted(want - got)}\n  extra {sorted(got - want)}")
        dice = [per_case[c][0] for c in sorted(want)]
        hd = [per_case[c][1] for c in sorted(want) if per_case[c][1] is not None]
        table.append({"name": cand["name"], "simple": int(cand.get("simple", 0)),
                      "config": cand.get("config", {}), "csv": cand["csv"], "n": len(dice),
                      "mean_dice_y": sum(dice) / len(dice),
                      "mean_hd95_y": (sum(hd) / len(hd)) if hd else None,
                      "n_hd95_nan": len(want) - len(hd)})

    best_dice = max(t["mean_dice_y"] for t in table)
    winners = [t for t in table if best_dice - t["mean_dice_y"] < tol]   # strict <, per V4_PLAN §6
    chosen = min(winners, key=lambda t: (t["simple"], -t["mean_dice_y"]))
    trace = {"rule": "max mean Dice_Y on V_k; within tol -> smaller simplicity rank wins",
             "tol": tol, "best_mean_dice_y": best_dice, "within_tol": [t["name"] for t in winners],
             "chosen": chosen["name"]}

    frozen = {"fold": k, "chosen": chosen["name"], "chosen_config": chosen["config"],
              "selected_at": datetime.now().isoformat(), "manifest": args.manifest,
              "rule": trace["rule"], "tol": tol, "candidates": table}
    out_yaml = ROOT / "configs" / "v4_frozen" / f"fold{k}.json"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(json.dumps(frozen, indent=1))
    log = ROOT / "log" / "v4" / f"selection_fold{k}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(frozen, indent=1))

    print(f"fold {k}: V_k n={len(want)}  tol={tol}")
    for t in sorted(table, key=lambda t: -t["mean_dice_y"]):
        mark = " <- chosen" if t is chosen else ""
        hd = f"{t['mean_hd95_y']:.2f}" if t["mean_hd95_y"] is not None else "NA"
        print(f"  {t['name']:<16} simple={t['simple']}  Dice_Y={t['mean_dice_y']:.4f}  HD95={hd}{mark}")
    print(f"frozen -> {out_yaml}")


if __name__ == "__main__":
    main()
