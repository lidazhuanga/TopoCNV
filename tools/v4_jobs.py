#!/usr/bin/env python3
"""Append jobs to log/v4/jobs.json for tools/v4_queue.py (idempotent by key; order = launch priority).

  python tools/v4_jobs.py stage1 --folds 0,1 --tier dev      # s1dev_* keys, fold{k}_dev outputs
  python tools/v4_jobs.py stage1 --folds 0,1                # report tier (s1_*)
  python tools/v4_jobs.py priors --folds 0,1
  python tools/v4_jobs.py stage2 --config configs/v4_dev/r0.yaml --folds 0,1
  python tools/v4_jobs.py nnprep --folds 0,1 --variant dev --margin 15
  python tools/v4_jobs.py nntrain --folds 0,1 --variant dev --trainer nnUNetTrainerV4_Pabc --name r1p_abc
  python tools/v4_jobs.py list
"""
import argparse
import json
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


def add(jobs, job):
    if any(j["key"] == job["key"] for j in jobs):
        print(f"exists: {job['key']}")
        return
    jobs.append(job)
    print(f"added: {job['key']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage1", "priors", "r0dev", "stage2", "nnprep", "nntrain", "list"])
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
    args = ap.parse_args()
    jobs = load()
    folds = [int(f) for f in args.folds.split(",")]
    if args.what == "list":
        for j in jobs:
            print(j["key"], "<-", j.get("deps", []))
        return
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
        else:
            cfg = load_config(ROOT / args.config)
            add(jobs, {"key": f"{cfg['name']}_f{k}", "kind": "gpu", "resume_arg": "--resume",
                       "cmd": [PY, "-m", "trainer.v4.train_stage2", "--config", args.config, "--fold", str(k)],
                       "done": f"results/v4_runs/{cfg['name']}/fold{k}/COMPLETED"})
            print(f"note: {cfg['name']} uses the legacy DynUNet stage2 path (appendix only); "
                  "priors deps must be resolved manually")
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=1))


if __name__ == "__main__":
    main()
