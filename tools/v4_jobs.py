#!/usr/bin/env python3
"""Append jobs to log/v4/jobs.json for tools/v4_queue.py (idempotent by key; order = launch priority).

  python tools/v4_jobs.py stage1 --folds 0,1
  python tools/v4_jobs.py priors --folds 0,1
  python tools/v4_jobs.py stage2 --config configs/v4_dev/r0.yaml --folds 0,1
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
    ap.add_argument("what", choices=["stage1", "priors", "stage2", "list"])
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--config")
    args = ap.parse_args()
    jobs = load()
    folds = [int(f) for f in args.folds.split(",")]
    if args.what == "list":
        for j in jobs:
            print(j["key"], "<-", j.get("deps", []))
        return
    for k in folds:
        if args.what == "stage1":
            for j in range(N_INNER):
                add(jobs, {"key": f"s1_f{k}_i{j}", "kind": "gpu", "resume_arg": "--resume",
                           "cmd": [PY, "-m", "trainer.v4.train_stage1", "--fold", str(k), "--inner", str(j)],
                           "done": f"results/v4_stage1/fold{k}/inner{j}/COMPLETED"})
        elif args.what == "priors":
            add(jobs, {"key": f"priors_f{k}", "kind": "cpu", "deps": [f"s1_f{k}_i{j}" for j in range(N_INNER)],
                       "cmd": [PY, "tools/v4_build_priors.py", "--fold", str(k), "--workers", "24"],
                       "done": f"results/v4_priors/fold{k}/atlas_meta.json"})
        else:
            cfg = load_config(ROOT / args.config)
            needs_priors = bool(cfg.get("roi", {}).get("enabled") or cfg.get("priors"))
            add(jobs, {"key": f"{cfg['name']}_f{k}", "kind": "gpu", "resume_arg": "--resume",
                       "deps": [f"priors_f{k}"] if needs_priors else [],
                       "cmd": [PY, "-m", "trainer.v4.train_stage2", "--config", args.config, "--fold", str(k)],
                       "done": f"results/v4_runs/{cfg['name']}/fold{k}/COMPLETED"})
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=1))


if __name__ == "__main__":
    main()
