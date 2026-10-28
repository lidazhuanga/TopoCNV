#!/usr/bin/env python3
"""Append jobs to log/v4/jobs.json for tools/v4_queue.py (idempotent by key; order = launch priority).

  python tools/v4_jobs.py stage1 --folds 0,1 --tier dev      # s1dev_* keys, fold{k}_dev outputs
  python tools/v4_jobs.py stage1 --folds 0,1                # report tier (s1_*)
  python tools/v4_jobs.py priors --folds 0,1
  python tools/v4_jobs.py stage2 --config configs/v4_dev/r0.yaml --folds 0,1
  python tools/v4_jobs.py nnprep --folds 0,1 --variant dev --margin 15
  python tools/v4_jobs.py nntrain --folds 0,1 --variant dev --trainer nnUNetTrainerV4_Pabc --name r1p_abc
  python tools/v4_jobs.py list
V4.2 (V4_PLAN §11; full image, harmonised labels; tools/v4_nnunet.py prep42 / train42):
  python tools/v4_jobs.py nnprep42 --what42 binY          # Dataset540 (then --what42 s1: Dataset541)
  python tools/v4_jobs.py nntrain42 --folds 0,1,2,3,4 --variant dev --name v42_r0u                 # stock R0u
  python tools/v4_jobs.py nntrain42 --folds 0,1,2,3,4 --variant dev --trainer nnUNetTrainerV4_noP_S05 --name v42_s05
  python tools/v4_jobs.py s1train42 --folds 0,1,2,3,4 --variant dev                                # 3 inner each
  python tools/v4_jobs.py smoke42p --after s142_dev_f0_i2   # GPU smoke of the prior path, gates the D2' chain
  python tools/v4_jobs.py d2chain42 --folds 0,1,2,3,4    # Stage 1 holdout prediction -> priors -> Dataset542+k,
                                                         # D4' scores + selection, D2' arms + scores + selection
V4.3 (V4_PLAN §12; tools/v4_mm43.py):
  python tools/v4_jobs.py mm43 --folds 0,1,2,3,4 --after smoke43    # Dataset557, GPU smoke smoke43, MMd-1/MMd-2
                                                         # dev runs, scores (with / without MRI), selection
Jobs are inserted next to their fold's existing jobs (priority = list order); jobs.json is replaced atomically.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.v4.common import load_config  # noqa: E402
from trainer.v4.splits import N_INNER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "log" / "v4" / "jobs.json"
PY = sys.executable
NNPY = str(Path.home() / "cnv" / "envs" / "nnunetv2_cnv" / "bin" / "python")


def load():
    return json.loads(JOBS.read_text()) if JOBS.exists() else []


def add(jobs, job, after=None):
    """Append, or insert right after the job `after` (and after jobs inserted there before)."""
    if any(j["key"] == job["key"] for j in jobs):
        print(f"exists: {job['key']}")
        return
    if after is None:
        jobs.append(job)
    else:
        idx = [n for n, j in enumerate(jobs) if j["key"] == after]
        if not idx:
            raise SystemExit(f"cannot insert {job['key']}: no job {after}")
        n = idx[0] + 1
        while n < len(jobs) and jobs[n].get("_after") == after:
            n += 1
        jobs.insert(n, dict(job, _after=after))
    print(f"added: {job['key']}" + (f" (after {after})" if after else ""))


def save(jobs):
    """Atomic replace: the running queue re-reads jobs.json every poll."""
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOBS.with_name(f".{JOBS.name}.tmp")
    tmp.write_text(json.dumps(jobs, indent=1))
    os.replace(tmp, JOBS)


def d2chain42(jobs, k):
    """Fold k's D2' chain (V4_PLAN §11.6.1-11.6.2). GPU jobs follow the queue's GPU rules like every other job.
    Needs the smoke42p job (added first by `v4_jobs.py smoke42p`): the whole chain waits for it."""
    if not any(j["key"] == "smoke42p" for j in jobs):
        raise SystemExit("add the smoke42p job first (python tools/v4_jobs.py smoke42p --after <key>)")
    ev = "results/v4_dev_eval_harm_v1/{n}/fold%d.csv" % k

    def score(name, dep, after):
        add(jobs, {"key": f"sc42_{name}_f{k}", "kind": "cpu", "deps": [dep],
                   "cmd": [PY, "tools/v4_dev_summary.py", "score", name, "--gt", "harm_v1", "--folds", str(k)],
                   "done": ev.format(n=name)}, after)

    s1 = [f"s142_dev_f{k}_i{j}" for j in range(N_INNER)]
    for j in range(N_INNER):        # short GPU jobs: right behind the fold's Stage 1 runs
        add(jobs, {"key": f"s1p42_dev_f{k}_i{j}", "kind": "gpu", "deps": [s1[j], "smoke42p"],
                   "cmd": [NNPY, "tools/v4_nnunet.py", "predict42", "--fold", str(k), "--variant", "dev",
                           "--inner", str(j)],
                   "done": f"log/v4/s1p42_dev_f{k}_i{j}.done"}, s1[-1])
    tail = f"nn42_v42_s10_dev_f{k}"         # the fold's last D4' run: the rest of the chain goes behind it
    add(jobs, {"key": f"pri42_dev_f{k}", "kind": "cpu", "deps": s1 + [f"s1p42_dev_f{k}_i{j}" for j in range(N_INNER)],
               "cmd": [PY, "tools/v4_build_priors42.py", "--fold", str(k), "--workers", "12"],
               "done": f"results/v4_priors42/fold{k}_dev/DONE"}, tail)
    add(jobs, {"key": f"nnprep42p_dev_f{k}", "kind": "cpu", "deps": [f"pri42_dev_f{k}"],
               "cmd": [NNPY, "tools/v4_nnunet.py", "prep42p", "--fold", str(k), "--variant", "dev"],
               "done": f"log/v4/nnprep_{542 + k}.done"}, tail)
    d4 = ["v42_r0u", "v42_s05", "v42_s10"]
    for n in d4:
        score(n, f"nn42_{n}_dev_f{k}", tail)
    add(jobs, {"key": f"d4sel42_f{k}", "kind": "cpu", "deps": [f"sc42_{n}_f{k}" for n in d4],
               "cmd": [PY, "tools/v4_select42.py", "d4", "--fold", str(k)],
               "done": f"results/v4_select42/fold{k}/d4.json"}, tail)
    for combo in ("ab", "ac", "bc", "abc"):
        n = f"v42_d2{combo}"
        add(jobs, {"key": f"nn42_{n}_dev_f{k}", "kind": "gpu",
                   "deps": [f"nnprep42p_dev_f{k}", f"d4sel42_f{k}", "smoke42p"],
                   "cmd": [NNPY, "tools/v4_nnunet.py", "train42", "--dataset", "prior", "--fold", str(k),
                           "--variant", "dev", "--priors", combo, "--name", n],
                   "done": f"results/v4_dev_preds/{n}/fold{k}/EXPORTED"}, tail)
        score(n, f"nn42_{n}_dev_f{k}", tail)
    add(jobs, {"key": f"d2sel42_f{k}", "kind": "cpu",
               "deps": [f"d4sel42_f{k}"] + [f"sc42_v42_d2{c}_f{k}" for c in ("ab", "ac", "bc", "abc")],
               "cmd": [PY, "tools/v4_select42.py", "d2", "--fold", str(k)],
               "done": f"results/v4_select42/fold{k}/d2.json"}, tail)


def mm43(jobs, folds, after):
    """V4.3 dev phase (V4_PLAN §12.5), inserted after `after` so it runs ahead of the D2' prior arms.
    Selection also needs the fold's v42_r0u score (sc42_v42_r0u_f{k}, from d2chain42)."""
    if after is None:
        raise SystemExit("mm43 needs --after <key> (its place in the priority order)")
    add(jobs, {"key": "nnprep43_mm", "kind": "cpu", "deps": ["nnprep42_binY"],
               "cmd": [NNPY, "tools/v4_mm43.py", "prep43", "--what", "mm"],
               "done": "log/v4/nnprep_557.done"}, after)
    add(jobs, {"key": "smoke43", "kind": "gpu", "deps": [], "cmd": ["bash", "tests/v4_nnunet_smoke43.sh", "queue"],
               "done": "log/v4/smoke43.done"}, after)
    ev = "results/v4_dev_eval_harm_v1/{n}/fold%d.csv"
    for k in folds:
        for c in ("1", "2"):
            n = f"v43_mmd{c}"
            add(jobs, {"key": f"nn43_{n}_dev_f{k}", "kind": "gpu", "deps": ["nnprep43_mm", "smoke43"],
                       "cmd": [NNPY, "tools/v4_mm43.py", "train43", "--what", "mm", "--fold", str(k), "--variant", "dev",
                               "--trainer", f"nnUNetTrainerV43_MMd{c}", "--name", n],
                       "done": f"log/v4/nn43_{n}_dev_f{k}.done"}, after)
    for k in folds:
        names = [f"v43_mmd{c}{s}" for c in ("1", "2") for s in ("", "_nomri")]
        for n in names:
            add(jobs, {"key": f"sc43_{n}_f{k}", "kind": "cpu", "deps": [f"nn43_{n.replace('_nomri', '')}_dev_f{k}"],
                       "cmd": [PY, "tools/v4_dev_summary.py", "score", n, "--gt", "harm_v1", "--folds", str(k)],
                       "done": ev.format(n=n) % k}, after)
        add(jobs, {"key": f"sel43_f{k}", "kind": "cpu", "deps": [f"sc43_{n}_f{k}" for n in names] + [f"sc42_v42_r0u_f{k}"],
                   "cmd": [PY, "tools/v4_select43.py", "select", "--fold", str(k)],
                   "done": f"results/v4_select43/fold{k}.json"}, after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage1", "priors", "r0dev", "stage2", "nnprep", "nntrain", "list",
                                     "nnprep42", "nntrain42", "s1train42", "d2chain42", "smoke42p", "mm43"])
    # stage2 = legacy DynUNet appendix path only (V4_PLAN §5); mainline arms are nnprep/nntrain
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--tier", choices=["report", "dev"], default="report",
                    help="stage1: which Stage-1 tier to schedule (V4_PLAN §5)")
    ap.add_argument("--config")
    ap.add_argument("--variant", choices=["dev", "report"], default="dev")
    ap.add_argument("--margin", type=int,
                    help="nnprep: default = the frozen D1-AM1 margin in configs/v4_qc/fold{k}.json")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--trainer")
    ap.add_argument("--name")
    ap.add_argument("--after", default=None, help="smoke42p / mm43: insert after this job key")
    ap.add_argument("--what42", dest="what42", choices=["binY", "s1"], default="binY",
                    help="nnprep42: which V4.2 dataset (540 binY, 541 s1)")
    args = ap.parse_args()
    jobs = load()
    folds = [int(f) for f in args.folds.split(",")]
    if args.what == "list":
        for j in jobs:
            print(j["key"], "<-", j.get("deps", []))
        return
    if args.what == "nnprep42":
        did = {"binY": 540, "s1": 541}[args.what42]
        add(jobs, {"key": f"nnprep42_{args.what42}", "kind": "cpu",
                   "deps": ["nnprep42_binY"] if args.what42 == "s1" else [],   # 541 links 540's raw CT
                   "cmd": [NNPY, "tools/v4_nnunet.py", "prep42", "--what", args.what42],
                   "done": f"log/v4/nnprep_{did}.done"})
        folds = []
    if args.what == "smoke42p":     # GPU smoke of the prior path; gates the whole D2' chain
        add(jobs, {"key": "smoke42p", "kind": "gpu", "cmd": ["bash", "tests/v4_nnunet_smoke42p.sh", "queue"],
                   "done": "log/v4/smoke42p.done"}, args.after)
        folds = []
    if args.what == "mm43":
        mm43(jobs, folds, args.after)
        folds = []
    for k in folds:
        if args.what == "stage1":
            pfx = "s1_" if args.tier == "report" else "s1dev_"
            sub = "fold{k}" if args.tier == "report" else "fold{k}_dev"
            for j in range(N_INNER):
                add(jobs, {"key": f"{pfx}f{k}_i{j}", "kind": "gpu", "resume_arg": "--resume",
                           "cmd": [PY, "-m", "trainer.v4.train_stage1", "--fold", str(k), "--inner", str(j),
                                   "--tier", args.tier],
                           "done": f"results/v4_stage1/{sub.format(k=k)}/inner{j}/COMPLETED"})
        elif args.what == "priors":
            for v in ("dev", "report"):
                pfx = "s1_" if v == "report" else "s1dev_"
                add(jobs, {"key": f"priors_{v}_f{k}", "kind": "cpu",
                           "deps": [f"{pfx}f{k}_i{j}" for j in range(N_INNER)],
                           "cmd": [PY, "tools/v4_build_priors.py", "--fold", str(k), "--variant", v,
                                   "--workers", "24", "--qc", f"configs/v4_qc/fold{k}.json"],
                           "done": f"results/v4_priors/fold{k}_{v}/atlas_meta.json"})
        elif args.what == "r0dev":
            add(jobs, {"key": f"r0dev_f{k}", "kind": "gpu",
                       "cmd": [NNPY, "tools/v4_nnunet.py", "train", "--fold", str(k), "--variant", "dev",
                               "--r0", "--name", "r0_dev"],
                       "done": f"results/v4_dev_preds/r0_dev/fold{k}/EXPORTED"})
        elif args.what == "nnprep":
            if args.variant == "dev" and args.shuffle:
                raise SystemExit("--shuffle is only defined for the report variant")
            sh = "shuf" if args.shuffle else ""
            dname = {"dev": 520, "report": 530 if args.shuffle else 510}[args.variant] + k
            margin = args.margin
            if margin is None:
                qcp = ROOT / "configs" / "v4_qc" / f"fold{k}.json"
                if not qcp.exists():
                    raise SystemExit(f"nnprep needs --margin or {qcp}")
                margin = int(json.loads(qcp.read_text())["margin"])
            add(jobs, {"key": f"nnprep_{args.variant}{sh}_f{k}", "kind": "cpu", "deps": [f"priors_{args.variant}_f{k}"],
                       "cmd": [NNPY, "tools/v4_nnunet.py", "prep", "--fold", str(k), "--variant", args.variant,
                               "--margin", str(margin)] + (["--shuffle"] if args.shuffle else []),
                       "done": f"log/v4/nnprep_{dname}.done"})
        elif args.what == "nntrain":
            sh = "shuf" if args.shuffle else ""
            out = (f"results/v4_dev_preds/{args.name}/fold{k}/EXPORTED" if args.variant == "dev"
                   else f"results/v4_{args.name}_label_predictions/EXPORTED_fold{k}")
            add(jobs, {"key": f"nn_{args.name}_{args.variant}_f{k}", "kind": "gpu",
                       "deps": [f"nnprep_{args.variant}{sh}_f{k}"],
                       "cmd": [NNPY, "tools/v4_nnunet.py", "train", "--fold", str(k), "--variant", args.variant,
                               "--trainer", args.trainer, "--name", args.name] + (["--shuffle"] if args.shuffle else []),
                       "done": out})
        elif args.what == "nntrain42":
            if not args.name:
                raise SystemExit("nntrain42 needs --name (v42_* for dev candidates)")
            out = (f"results/v4_dev_preds/{args.name}/fold{k}/EXPORTED" if args.variant == "dev"
                   else f"results/v4_{args.name}_label_predictions/EXPORTED_fold{k}")
            add(jobs, {"key": f"nn42_{args.name}_{args.variant}_f{k}", "kind": "gpu", "deps": ["nnprep42_binY"],
                       "cmd": [NNPY, "tools/v4_nnunet.py", "train42", "--dataset", "binY", "--fold", str(k),
                               "--variant", args.variant, "--name", args.name]
                              + (["--trainer", args.trainer] if args.trainer else []),
                       "done": out})
        elif args.what == "d2chain42":
            d2chain42(jobs, k)
        elif args.what == "s1train42":
            for j in range(N_INNER):
                add(jobs, {"key": f"s142_{args.variant}_f{k}_i{j}", "kind": "gpu", "deps": ["nnprep42_s1"],
                           "cmd": [NNPY, "tools/v4_nnunet.py", "train42", "--dataset", "s1", "--fold", str(k),
                                   "--variant", args.variant, "--inner", str(j)],
                           "done": f"log/v4/s142_{args.variant}_f{k}_i{j}.done"})
        else:
            cfg = load_config(ROOT / args.config)
            add(jobs, {"key": f"{cfg['name']}_f{k}", "kind": "gpu", "resume_arg": "--resume",
                       "cmd": [PY, "-m", "trainer.v4.train_stage2", "--config", args.config, "--fold", str(k)],
                       "done": f"results/v4_runs/{cfg['name']}/fold{k}/COMPLETED"})
            print(f"note: {cfg['name']} uses the legacy DynUNet stage2 path (appendix only); "
                  "priors deps must be resolved manually")
    save(jobs)


if __name__ == "__main__":
    main()
