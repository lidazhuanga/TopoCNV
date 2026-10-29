#!/usr/bin/env python3
"""V4.2 development selection per outer fold (V4_PLAN §11.6, rules fixed in §11.6.1 before any D4'/D2' result).

  python tools/v4_select42.py d4 --fold 0      # w*_k from {S0 = v42_r0u, S0.5 = v42_s05, S1.0 = v42_s10}
  python tools/v4_select42.py d2 --fold 0      # P*_k from {B = the D4' winner, v42_d2ab/ac/bc/abc} on B(w*_k)
  python tools/v4_select42.py freeze --folds 0,1,2,3,4 --confirmed "<main session's words>"   # after the report

Score = mean Dice_Y over V_k (harmonised GT, results/v4_dev_eval_harm_v1/<name>/fold{k}.csv), float64.
Eligible = best - score < 0.003; lowest simplicity level; then highest score; then the fixed order. HD95 and the
other metrics are recorded only. Every hard check of §11.6.1 aborts on failure. Manifests
results/v4_select42/fold{k}/d4.json and d2.json are write-once: a rerun must give the same content.
d2.json holds the proposed m_k + P*_k + w*_k; configs/v4_frozen42/fold{k}.json is written only by `freeze`.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import v4_dev_summary as DS  # noqa: E402
import v4_nnunet as N  # noqa: E402
from topocnv.data.v4_splits import N_INNER, internal_split  # noqa: E402

PROTOCOL = "V4.2 / V4-AM-20261027-02"
TOL = 0.003
GT = "harm_v1"
FROZEN42 = ROOT / "configs" / "v4_frozen42"
# (role, name, simplicity level, trainer); order of the list = the fixed tie order
D4 = [("S0", "v42_r0u", 0, N.R0_TRAINER), ("S05", "v42_s05", 1, "nnUNetTrainerV4_noP_S05"),
      ("S10", "v42_s10", 2, "nnUNetTrainerV4_noP_S10")]
D2_PRIORS = [("ab", 1), ("ac", 1), ("bc", 1), ("abc", 2)]


def choose(scores, levels, order, tol=TOL):
    """§11.6.1 rule. scores/levels: {name: float/int}; order: names in the fixed tie order."""
    assert set(scores) == set(levels) == set(order) and len(order) == len(set(order))
    assert all(math.isfinite(s) for s in scores.values()), "non-finite candidate score"
    best = max(scores.values())
    eligible = [n for n in order if best - scores[n] < tol]
    low = min(levels[n] for n in eligible)
    same = [n for n in eligible if levels[n] == low]
    top = max(scores[n] for n in same)
    chosen = next(n for n in order if n in same and scores[n] == top)
    return chosen, {"best": best, "eligible": eligible, "lowest_level": low, "level_candidates": same}


def per_case_dice(path, want):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c = (r.get("case_id") or "").strip()
            if not c or c.upper() == "MEAN":
                continue
            if c in rows:
                raise SystemExit(f"{path}: duplicate case {c}")
            d = float(r["Dice_Y"])
            if not math.isfinite(d):
                raise SystemExit(f"{path}: non-finite Dice_Y for {c}")
            h = r.get("HD95_Y", "")
            rows[c] = (d, float(h) if h not in ("", None) else float("nan"))
    if set(rows) != set(want):
        raise SystemExit(f"{path}: cases != V_k (missing {sorted(set(want) - set(rows))[:5]}, "
                         f"extra {sorted(set(rows) - set(want))[:5]})")
    return rows


def run_record(fd):
    p = fd / "v42_runs.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} missing: no run record")
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return recs[-1], recs


def candidate(k, name, trainer, did, dname, split, train, val, want, extra_check=None):
    """Hard checks 3-5 and 8 for one candidate; returns its manifest entry."""
    pdir = N.RESULTS / "v4_dev_preds" / name / f"fold{k}"
    mk = pdir / "EXPORTED"
    if not mk.exists() or (pdir / "EXPORTED_PARTIAL").exists():
        raise SystemExit(f"{name} fold {k}: needs {mk} and no EXPORTED_PARTIAL next to it")
    m = re.search(r" from (\S+) trainer (\S+)", mk.read_text())
    if not m:
        raise SystemExit(f"{mk}: cannot read the run folder and trainer")
    fd = Path(m.group(1))
    if m.group(2) != trainer or fd != N.results_dir(did, dname, trainer, split):
        raise SystemExit(f"{name} fold {k}: exported from {fd} by {m.group(2)}, expected {trainer} on {dname} "
                         f"split {split}")
    if not (fd / "checkpoint_final.pth").exists():
        raise SystemExit(f"{fd}: no checkpoint_final.pth")
    last, recs = run_record(fd)
    for r in recs:
        cmd = r["cmd"]
        if r["trainer"] != trainer or cmd[1:4] != [str(did), N.CONFIG, str(split)] or cmd[cmd.index("-tr") + 1] != trainer:
            raise SystemExit(f"{fd}: run record {r['time']} ran {cmd}, not {trainer} on {did} split {split}")
        if extra_check:
            extra_check(r)
    sp = json.loads((N.NNUNET_HOME / "nnUNet_preprocessed" / dname / "splits_final.json").read_text())[split]
    if sorted(sp["train"]) != sorted(train) or sorted(sp["val"]) != sorted(val):
        raise SystemExit(f"{dname} split {split}: train/val are not T_{k} - V_{k} / {'V_k copies' if val != want else 'V_k'}")
    preds = sorted(pdir.glob("*_stage2_label.nrrd"))
    if sorted(p.name.split("_stage2_label")[0] for p in preds) != sorted(want):
        raise SystemExit(f"{pdir}: predictions are not exactly V_{k}")
    csv_path = DS.GT[GT][1] / name / f"fold{k}.csv"
    rows = per_case_dice(csv_path, want)
    dice = np.array([rows[c][0] for c in sorted(want)], np.float64)
    hd = np.array([rows[c][1] for c in sorted(want)], np.float64)
    return {"name": name, "trainer": trainer, "dataset": dname, "split": split, "fold_dir": str(fd), "n": len(want),
            "mean_dice_y": float(dice.mean()), "mean_hd95_y": float(np.nanmean(hd)) if np.isfinite(hd).any() else None,
            "csv": str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else str(csv_path),
            "csv_sha256": N._file_sha(csv_path),
            "predictions_sha256": N._digest({p.name: N._file_sha(p) for p in preds}),
            "run_git_head": last.get("git_head"), "run_dataset": last.get("dataset")}


def common_checks():
    """Hard check 1: harm_v1 direction / config / labels, and the eval directory's reference tag."""
    sha, digest = N.check_direction()
    DS.check_tag(GT)
    tag = DS.GT[GT][1] / "GT.json"
    if not tag.exists():
        raise SystemExit(f"{tag} missing: nothing scored against {GT}")
    return {"protocol": PROTOCOL, "harm_config_sha256": sha, "harm_label_digest": digest,
            "eval_gt": json.loads(tag.read_text())}


def write_manifest(path, man):
    """Write-once: an existing manifest must equal the new one apart from 'written'."""
    body = {k: v for k, v in man.items() if k != "written"}
    if path.exists():
        have = json.loads(path.read_text())
        if {k: v for k, v in have.items() if k != "written"} != json.loads(json.dumps(body)):
            raise SystemExit(f"{path} exists with different content: refusing (write-once)")
        print(f"{path} reproduced")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    N._atomic_text(path, json.dumps(dict(body, written=datetime.now().isoformat()), indent=1))
    print(f"wrote {path}")


def show(step, k, entries, chosen, trace):
    print(f"{step} fold {k}: best {trace['best']:.4f}, eligible {trace['eligible']}")
    for e in entries:
        hd = f"{e['mean_hd95_y']:.2f}" if e["mean_hd95_y"] is not None else "NA"
        print(f"  {e['role']:<4} {e['name']:<12} level {e['level']}  Dice_Y {e['mean_dice_y']:.4f}  HD95 {hd}"
              + ("  <- chosen" if e["name"] == chosen else ""))


def d4(k):
    head = common_checks()
    train, want = internal_split(k)
    did, dname = N.V42["binY"]
    split = N.v42_split_index("binY", k, "dev")
    entries = []
    for role, name, level, trainer in D4:
        e = candidate(k, name, trainer, did, dname, split, train, want, want)
        entries.append(dict(e, role=role, level=level))
    chosen, trace = choose({e["name"]: e["mean_dice_y"] for e in entries}, {e["name"]: e["level"] for e in entries},
                           [n for _, n, _, _ in D4])
    w = next(r for r, n, _, _ in D4 if n == chosen)
    man = dict(head, step="D4'", fold=k, rule="§11.6.1", tol=TOL, candidates=entries, trace=trace,
               chosen={"name": chosen, "w": w, "trainer": next(t for _, n, _, t in D4 if n == chosen)},
               tool_sha256=N._file_sha(Path(__file__).resolve()))
    show("D4'", k, entries, chosen, trace)
    write_manifest(N.SELECT42 / f"fold{k}" / "d4.json", man)


def check_priors(meta):
    """Hard checks 6-7: D1-AM2 config, priors DONE consistent with its files, Stage 1 inputs recorded."""
    pdir = Path(meta["priors_dir"])
    rec = json.loads((pdir / "DONE").read_text())
    if N._digest({i: v["sha"] for i, v in rec["ids"].items()}) != rec["digest"]:
        raise SystemExit(f"{pdir}/DONE: digest does not match its own per-file hashes")
    for i, v in rec["ids"].items():
        with np.load(pdir / f"{i}_prior.npz") as d:
            if N.prior_sha(d["prior"], d["bbox"]) != v["sha"]:
                raise SystemExit(f"{pdir}/{i}_prior.npz changed after the priors were built")
        if not v.get("stage1_sha256"):
            raise SystemExit(f"{pdir}/DONE: no Stage 1 input hash for {i}")
    if not rec.get("stage1_checkpoints_sha256") or len(rec["stage1_checkpoints_sha256"]) != N_INNER:
        raise SystemExit(f"{pdir}/DONE: Stage 1 checkpoints not recorded")
    qc = json.loads((N.QC42 / f"fold{meta['fold']}.json").read_text())
    if qc.get("amendment") != "D1-AM2" or len({int(qc["margin"]), int(meta["margin"]), int(rec["margin"])}) != 1:
        raise SystemExit("D1-AM2 margin differs between the config, the priors and the dataset")
    return rec, qc


def d2(k):
    head = common_checks()
    p4 = N.SELECT42 / f"fold{k}" / "d4.json"
    if not p4.exists():
        raise SystemExit(f"{p4} missing: run d4 first")
    m4 = json.loads(p4.read_text())
    d4sha = N._file_sha(p4)
    w = m4["chosen"]["w"]
    train, want = internal_split(k)
    did, dname = N.V42["binY"]
    b_name, b_trainer = m4["chosen"]["name"], m4["chosen"]["trainer"]
    assert (w, b_name, b_trainer) in {(r, n, t) for r, n, _, t in D4}, "d4.json names no D4' candidate"
    entries = [dict(candidate(k, b_name, b_trainer, did, dname, N.v42_split_index("binY", k, "dev"), train, want, want),
                    role="B", level=0)]
    pdid, pname = N.v42p(k, "dev")
    meta = N.check_dataset42p(pname)
    prec, qc = check_priors(meta)
    copies = [N.copy_id(c, j) for c in want for j in range(N_INNER)]

    def run_ok(combo):
        def f(r):
            d = r.get("d2") or {}
            if d.get("priors") != combo or d.get("w") != w or d.get("d4_sha256") != d4sha:
                raise SystemExit(f"run record {r['time']}: D2' settings {d} != priors {combo}, w {w}, this d4.json")
            if (r.get("dataset") or {}).get("priors_digest") != meta["priors_digest"]:
                raise SystemExit(f"run record {r['time']}: trained on other priors than {pname} has now")
        return f

    for combo, level in D2_PRIORS:
        e = candidate(k, f"v42_d2{combo}", N.prior_trainer(combo, w), pdid, pname, 0, train, copies, want,
                      extra_check=run_ok(combo))
        entries.append(dict(e, role=combo, level=level))
    order = [b_name] + [f"v42_d2{c}" for c, _ in D2_PRIORS]
    chosen, trace = choose({e["name"]: e["mean_dice_y"] for e in entries}, {e["name"]: e["level"] for e in entries},
                           order)
    p_star = next(e["role"] for e in entries if e["name"] == chosen)
    p_star = "" if p_star == "B" else p_star
    man = dict(head, step="D2'", fold=k, rule="§11.6.1", tol=TOL, d4_sha256=d4sha, w=w, candidates=entries,
               trace=trace, chosen={"name": chosen, "P": p_star},
               priors={"dir": meta["priors_dir"], "digest": meta["priors_digest"], "qc42_sha256": meta["qc42_sha256"],
                       "stage1_checkpoints_sha256": prec["stage1_checkpoints_sha256"]},
               proposed={"fold": k, "m": int(qc["margin"]), "P": p_star, "w": w,
                         "trainer": N.prior_trainer(p_star, w) if p_star else b_trainer},
               tool_sha256=N._file_sha(Path(__file__).resolve()))
    show("D2'", k, entries, chosen, trace)
    write_manifest(N.SELECT42 / f"fold{k}" / "d2.json", man)


def freeze(folds, confirmed, by):
    if not confirmed.strip():
        raise SystemExit("--confirmed needs the main session's confirmation (V4_PLAN §11.6.1)")
    if sorted(folds) != [0, 1, 2, 3, 4]:
        raise SystemExit("freeze all five folds at once, after all five d2.json exist (V4_PLAN §11.6.1)")
    missing = [k for k in folds if not (N.SELECT42 / f"fold{k}" / "d2.json").exists()]
    if missing:
        raise SystemExit(f"d2.json missing for folds {missing}")
    for k in folds:
        p2 = N.SELECT42 / f"fold{k}" / "d2.json"
        if not p2.exists():
            raise SystemExit(f"{p2} missing")
        m2 = json.loads(p2.read_text())
        out = FROZEN42 / f"fold{k}.json"
        cfg = {"protocol": PROTOCOL, "fold": k, "chosen_config": m2["proposed"], "d2_sha256": N._file_sha(p2),
               "d4_sha256": m2["d4_sha256"], "confirmed": confirmed, "confirmed_by": by,
               "frozen": datetime.now().isoformat()}
        if out.exists():
            raise SystemExit(f"{out} already frozen (write-once)")
        out.parent.mkdir(parents=True, exist_ok=True)
        N._atomic_text(out, json.dumps(cfg, indent=1))
        print(f"frozen fold {k}: {m2['proposed']} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["d4", "d2", "freeze"])
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
        {"d4": d4, "d2": d2}[a.cmd](a.fold)


if __name__ == "__main__":
    main()
