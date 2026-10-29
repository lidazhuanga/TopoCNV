#!/usr/bin/env python3
"""V4.3 development selection per outer fold (V4_PLAN §12.5, fixed before any multimodal result).

  python tools/v4_select43.py select --fold 0      # MMd-1 vs MMd-2 on V_k
  python tools/v4_select43.py freeze --folds 0,1,2,3,4 --confirmed "<main session's words>"   # after the report

S_with / S_none = mean Dice_Y over V_k (harm_v1 GT) with each case's own MRI / with no MRI for any case;
R = v42_r0u's mean Dice_Y over V_k. Eligible: S_none >= R - 0.01. Among the eligible, the highest S_with; if
both are eligible and their S_with differ by < 0.003, MMd-1. If none is eligible: the higher S_none (tie: MMd-1),
robustness_met = false. HD95 and the other metrics are recorded only. Every hard check aborts on failure.
Manifest results/v4_select43/fold{k}.json is write-once; configs/v4_frozen43/fold{k}.json only via `freeze`.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import v4_mm43 as M  # noqa: E402
import v4_nnunet as N  # noqa: E402
import v4_select42 as S42  # noqa: E402
from topocnv.data.v4_splits import internal_split  # noqa: E402

PROTOCOL = M.PROTOCOL
TOL = 0.003
MARGIN = 0.01
SELECT43 = N.RESULTS / "v4_select43"
# (role, name, trainer); the list order is the tie order (MMd-1 = the milder candidate first)
CANDS = [("MMd-1", "v43_mmd1", "nnUNetTrainerV43_MMd1"), ("MMd-2", "v43_mmd2", "nnUNetTrainerV43_MMd2")]
R0U = ("v42_r0u", N.R0_TRAINER)


def choose43(s_with, s_none, r, order, tol=TOL, margin=MARGIN):
    """§12.5 rule. s_with / s_none: {name: float}; r: R0u's score; order: names, milder first."""
    assert set(s_with) == set(s_none) == set(order) and len(order) == len(set(order))
    vals = list(s_with.values()) + list(s_none.values()) + [r]
    assert all(math.isfinite(v) for v in vals), "non-finite score"
    eligible = [n for n in order if s_none[n] >= r - margin]
    if eligible:
        best = max(s_with[n] for n in eligible)
        near = [n for n in eligible if best - s_with[n] < tol]
        chosen = near[0] if len(near) > 1 else next(n for n in eligible if s_with[n] == best)
        met = True
    else:
        top = max(s_none.values())
        chosen = next(n for n in order if s_none[n] == top)
        best, near, met = None, [], False
    return chosen, {"eligible": eligible, "threshold": r - margin, "best_with_mri_among_eligible": best,
                    "within_tol": near, "robustness_met": met}


def select(k):
    head = S42.common_checks()
    head["protocol"] = PROTOCOL
    did, dname = M.V43["mm"]
    meta = M.check_dataset43(dname)
    split = N.v42_split_index("binY", k, "dev")
    train, want = internal_split(k)

    def run_ok(r):
        mm = r.get("mm43") or {}
        if mm.get("protocol") != PROTOCOL or mm.get("dataset") != dname or mm.get("mri_sums_sha256") != meta["mri_sums_sha256"]:
            raise SystemExit(f"run record {r['time']}: V4.3 settings {mm} do not match {dname} as built now")

    r540, n540 = N.V42["binY"]
    ref = S42.candidate(k, R0U[0], R0U[1], r540, n540, split, train, want, want)
    entries = []
    for role, name, trainer in CANDS:
        e_with = S42.candidate(k, name, trainer, did, dname, split, train, want, want, extra_check=run_ok)
        e_none = S42.candidate(k, f"{name}_nomri", trainer, did, dname, split, train, want, want, extra_check=run_ok)
        ck = {(N.RESULTS / "v4_dev_preds" / n / f"fold{k}" / "EXPORTED").read_text().split("checkpoint ")[-1].strip()
              for n in (name, f"{name}_nomri")}
        cur = N._file_sha(Path(e_with["fold_dir"]) / "checkpoint_final.pth")
        if ck != {cur}:
            raise SystemExit(f"{name} fold {k}: the two conditions come from checkpoints {ck}, current {cur}")
        conds = {"with_mri": "condition with_mri", "no_mri": "condition no_mri"}
        for n, c in ((name, "with_mri"), (f"{name}_nomri", "no_mri")):
            if conds[c] not in (N.RESULTS / "v4_dev_preds" / n / f"fold{k}" / "EXPORTED").read_text():
                raise SystemExit(f"{n} fold {k}: EXPORTED is not the {c} condition")
        entries.append({"role": role, "name": name, "trainer": trainer, "checkpoint_sha256": cur,
                        "with_mri": e_with, "no_mri": e_none})
    order = [n for _, n, _ in CANDS]
    chosen, trace = choose43({e["name"]: e["with_mri"]["mean_dice_y"] for e in entries},
                             {e["name"]: e["no_mri"]["mean_dice_y"] for e in entries}, ref["mean_dice_y"], order)
    c = next(e for e in entries if e["name"] == chosen)
    man = dict(head, step="V4.3 dev", fold=k, rule="§12.5", tol=TOL, margin=MARGIN, reference=ref,
               mri={"sums_sha256": meta["mri_sums_sha256"], "digest": meta["mri_digest"], "frozen": meta["mri_frozen"]},
               candidates=entries, trace=trace,
               chosen={"role": c["role"], "name": chosen, "trainer": c["trainer"],
                       "case_trainer": c["trainer"].replace("MMd", "MMdcase")},
               tool_sha256=N._file_sha(Path(__file__).resolve()))
    print(f"V4.3 fold {k}: R0u {ref['mean_dice_y']:.4f}, threshold {trace['threshold']:.4f}, eligible {trace['eligible']}")
    for e in entries:
        print(f"  {e['role']:<6} with MRI {e['with_mri']['mean_dice_y']:.4f}  no MRI {e['no_mri']['mean_dice_y']:.4f}"
              + ("  <- chosen" if e["name"] == chosen else ""))
    if not trace["robustness_met"]:
        print("  robustness criterion NOT met by either candidate (reported as is, V4_PLAN §12.5 point 3)")
    S42.write_manifest(SELECT43 / f"fold{k}.json", man)


def freeze(folds, confirmed, by):
    if not confirmed.strip():
        raise SystemExit("--confirmed needs the main session's confirmation (V4_PLAN §12.5)")
    if sorted(folds) != [0, 1, 2, 3, 4]:
        raise SystemExit("freeze all five folds at once (V4_PLAN §12.5)")
    missing = [k for k in folds if not (SELECT43 / f"fold{k}.json").exists()]
    if missing:
        raise SystemExit(f"selection manifests missing for folds {missing}")
    have = [k for k in folds if (M.FROZEN43 / f"fold{k}.json").exists()]
    if have:
        raise SystemExit(f"folds {have} already frozen (write-once)")
    for k in folds:
        p = SELECT43 / f"fold{k}.json"
        m = json.loads(p.read_text())
        cfg = {"protocol": PROTOCOL, "fold": k, "chosen": m["chosen"], "robustness_met": m["trace"]["robustness_met"],
               "select_sha256": N._file_sha(p), "confirmed": confirmed, "confirmed_by": by,
               "frozen": datetime.now().isoformat()}
        M.FROZEN43.mkdir(parents=True, exist_ok=True)
        N._atomic_text(M.FROZEN43 / f"fold{k}.json", json.dumps(cfg, indent=1))
        print(f"frozen fold {k}: {m['chosen']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "freeze"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--confirmed", default="")
    ap.add_argument("--by", default="main session")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze([int(x) for x in a.folds.split(",")], a.confirmed, a.by)
    elif a.fold is None or not 0 <= a.fold < 5:
        raise SystemExit("--fold 0..4")
    else:
        select(a.fold)


if __name__ == "__main__":
    main()
