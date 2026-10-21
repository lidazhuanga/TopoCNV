#!/usr/bin/env python3
"""
P0 Stage-2 queue runner (protocol v0.3, binary Y). Runs on the training host under nohup.

Launches the 6 P0 arms x 5 folds from MIA/scripts/ablation_configs.py, fold-major so every
arm's fold 0 finishes first. Runs are spread across GPUs (at most --per_gpu of ours per card,
fewest-ours then most-free first) and start only on cards with --min_free_mb free; the queue
never signals any process, ours or other users'. A run is done when its fold dir has
COMPLETED; a run that died is relaunched with --resume (latest_checkpoint.pth), up to
MAX_ATTEMPTS times. Each time more runs complete, tools/p0_summary.py refreshes the main table
in the background. State: log/<RUN_TAG>/queue_state.json, one log per run.

  nohup python tools/p0_queue.py >> log/p0_binary_y_20261020/queue.log 2>&1 &
  touch log/p0_binary_y_20261020/STOP   # stop launching new runs (running ones continue)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
import ablation_configs as ac  # noqa: E402

RUN_TAG = "p0_binary_y_20261020"
LOG_DIR = ROOT / "log" / RUN_TAG
ARM_ORDER = ["Full", "A_gate", "A_mem", "C1", "B0", "A2"]
MAX_ATTEMPTS = 3


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def gpu_free_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    free = {}
    for line in out.strip().splitlines():
        idx, total, used = [int(x) for x in line.split(",")]
        free[str(idx)] = total - used
    return free


def fold_dir(arm, fold):
    return Path(ac.RESULTS_ROOT) / ac.stage2_exp(arm) / f"fold{fold}"


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # zombie children count as dead
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma-separated GPU ids")
    ap.add_argument("--per_gpu", type=int, default=2, help="max of our jobs per GPU")
    ap.add_argument("--max_jobs", type=int, default=8)
    ap.add_argument("--min_free_mb", type=int, default=40000, help="launch only if the GPU has this much free")
    ap.add_argument("--job_mb", type=int, default=16000, help="expected peak memory of one run")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    gpus = args.gpus.split(",")
    jobs = [(a, int(f)) for f in args.folds.split(",") for a in args.arms.split(",")]
    running = {}  # pid -> (key, Popen)
    summary, summarized = None, 0

    print(f"[{now()}] queue start: {len(jobs)} runs on GPUs {gpus}", flush=True)
    while True:
        for pid, (key, proc) in list(running.items()):
            if proc.poll() is not None:
                print(f"[{now()}] {key} exited rc={proc.returncode} on GPU {state[key]['gpu']}", flush=True)
                state[key]["status"] = "exited"
                state[key]["rc"] = proc.returncode
                del running[pid]

        pending = []
        for arm, fold in jobs:
            key = f"{arm}_fold{fold}"
            st = state.setdefault(key, {"attempts": 0, "status": "pending"})
            if (fold_dir(arm, fold) / "COMPLETED").exists():
                st["status"] = "completed"
                continue
            if st.get("pid") and alive(st["pid"]) and key not in [k for k, _ in running.values()]:
                st["status"] = "running_orphan"  # launched by an earlier queue instance
                continue
            if any(k == key for k, _ in running.values()):
                continue
            if st["attempts"] >= MAX_ATTEMPTS:
                st["status"] = "failed"
                continue
            pending.append((arm, fold, key))

        stop = (LOG_DIR / "STOP").exists()
        if pending and not stop:
            ours = {g: 0 for g in gpus}
            for key, _ in running.values():
                ours[state[key]["gpu"]] += 1
            for s in state.values():
                if s["status"] == "running_orphan" and s.get("gpu") in ours:
                    ours[s["gpu"]] += 1
            free = gpu_free_mb()
            while pending and sum(ours.values()) < args.max_jobs:
                # spread: fewest of our jobs first, then most free memory; never touch other users' jobs
                cands = [g for g in gpus if ours[g] < args.per_gpu and free.get(g, 0) >= args.min_free_mb]
                if not cands:
                    break
                gpu = min(cands, key=lambda g: (ours[g], -free[g]))
                ours[gpu] += 1
                free[gpu] -= args.job_mb
                arm, fold, key = pending.pop(0)
                cmd = ac.train_command(arm, fold).split()
                cmd[0] = sys.executable
                if (fold_dir(arm, fold) / "latest_checkpoint.pth").exists():
                    cmd.append("--resume")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                log = open(LOG_DIR / f"{key}.log", "a")
                log.write(f"\n# [{now()}] GPU {gpu} attempt {state[key]['attempts'] + 1}\n# {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[proc.pid] = (key, proc)
                state[key].update(status="running", pid=proc.pid, gpu=gpu,
                                  attempts=state[key]["attempts"] + 1, started=now())
                print(f"[{now()}] launched {key} on GPU {gpu} pid={proc.pid}", flush=True)

        state_path.write_text(json.dumps(state, indent=1))
        # refresh the P0 main table (CPU) whenever more runs have completed
        n_done = sum(s["status"] == "completed" for s in state.values())
        if n_done > summarized and (summary is None or summary.poll() is not None):
            summary = subprocess.Popen([sys.executable, str(ROOT / "tools" / "p0_summary.py")], cwd=ROOT,
                                       stdout=open(LOG_DIR / "summary.log", "w"),
                                       stderr=subprocess.STDOUT, start_new_session=True)
            summarized = n_done
            print(f"[{now()}] summary refresh started ({n_done} runs completed)", flush=True)
        unfinished = [k for k, s in state.items() if s["status"] not in ("completed", "failed")]
        if not running and (not unfinished or stop):
            if summary is not None:
                summary.wait()
            if n_done > summarized:
                subprocess.run([sys.executable, str(ROOT / "tools" / "p0_summary.py")], cwd=ROOT,
                               stdout=open(LOG_DIR / "summary.log", "w"), stderr=subprocess.STDOUT)
            print(f"[{now()}] queue exit: unfinished={unfinished} stop={stop}", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
