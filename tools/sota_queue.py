#!/usr/bin/env python3
"""
SOTA comparison queue (protocol v0.3 §14.4, binary Y). Runs on the training host under nohup.

Runs 7 methods x 5 folds: standard nnU-Net v2 (tools/sota_nnunet.py, separate venv) and the
six registry networks in trainer/npc_sota_train.py. Every run is capped at 300 epochs.

GPU rules (2026-10-22): as many cards as are free, at most one of our runs per card. A card is
used only if it has >= --min_free_mb (40 GB) free. P0 keeps priority: while P0 still has runs
waiting to launch, a card P0 could still use (fewer than P0_PER_GPU P0 runs on it) must keep
--min_free_mb free after our run. The queue never signals any process, ours or other users'.

nnU-Net runs start only after its data preparation has finished (splits_final.json exists).
A run is done when results/sota_binary_y_<method>/fold<k>/COMPLETED exists; a run that died
is relaunched (--resume / nnU-Net --c) up to MAX_ATTEMPTS times. Each time more runs
complete, tools/sota_summary.py refreshes the table in the background.

  nohup python tools/sota_queue.py >> log/sota_binary_y_20261021/queue.log 2>&1 &
  touch log/sota_binary_y_20261021/STOP   # stop launching new runs (running ones continue)
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
RUN_TAG = "sota_binary_y_20261021"
LOG_DIR = ROOT / "log" / RUN_TAG
P0_STATE = ROOT / "log" / "p0_binary_y_20261020" / "queue_state.json"  # read only
P0_PER_GPU = 2
EPOCHS = 300
NNUNET_PY = Path.home() / "cnv" / "envs" / "nnunetv2_cnv" / "bin" / "python"
NNUNET_PREP = Path.home() / "cnv" / "nnunet_cnv" / "nnUNet_preprocessed" / "Dataset501_CNVbinaryY"
METHOD_ORDER = ["nnunetv2", "nnunet", "segresnet", "attention_unet", "unetr", "swin_unetr", "nnformer"]
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


def alive(pid):
    try:
        os.kill(pid, 0)  # signal 0: existence check only, nothing is delivered
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def p0_load():
    """(P0 runs alive per GPU, number of P0 runs still waiting to launch)."""
    try:
        st = json.loads(P0_STATE.read_text())
    except (OSError, ValueError):
        return {}, 0
    per_gpu, pending = {}, 0
    for s in st.values():
        if s.get("status") in ("running", "running_orphan") and s.get("pid") and alive(s["pid"]):
            per_gpu[str(s.get("gpu"))] = per_gpu.get(str(s.get("gpu")), 0) + 1
        elif s.get("status") in ("pending", "exited"):
            pending += 1
    return per_gpu, pending


def nnunet_ready():
    return (NNUNET_PREP / "splits_final.json").exists()


def fold_dir(method, fold):
    return ROOT / "results" / f"sota_binary_y_{method}" / f"fold{fold}"


def command(method, fold, gpu):
    if method == "nnunetv2":
        return [str(NNUNET_PY), "tools/sota_nnunet.py", "train", "--fold", str(fold)]
    cmd = [sys.executable, "trainer/npc_sota_train.py", "--model", method, "--fold", str(fold),
           "--gpu", gpu, "--epochs", str(EPOCHS), "--select_by_sliding_window", "--random_crop_prob", "0.33"]
    if (fold_dir(method, fold) / "latest_checkpoint.pth").exists():
        cmd.append("--resume")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma-separated GPU ids")
    ap.add_argument("--min_free_mb", type=int, default=40000)
    ap.add_argument("--job_mb", type=int, default=16000, help="expected peak memory of one run")
    ap.add_argument("--max_jobs", type=int, default=8, help="upper bound; one run per card anyway")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    gpus = args.gpus.split(",")
    jobs = [(m, int(f)) for f in args.folds.split(",") for m in args.methods.split(",")]
    running = {}  # pid -> (key, Popen)
    summary, summarized = None, 0

    print(f"[{now()}] queue start: {len(jobs)} runs on GPUs {gpus}, {EPOCHS} epochs", flush=True)
    while True:
        for pid, (key, proc) in list(running.items()):
            if proc.poll() is not None:
                print(f"[{now()}] {key} exited rc={proc.returncode} on GPU {state[key]['gpu']}", flush=True)
                state[key]["status"] = "exited"
                state[key]["rc"] = proc.returncode
                del running[pid]

        pending = []
        for method, fold in jobs:
            key = f"{method}_fold{fold}"
            st = state.setdefault(key, {"attempts": 0, "status": "pending"})
            if (fold_dir(method, fold) / "COMPLETED").exists():
                st["status"] = "completed"
                continue
            if any(k == key for k, _ in running.values()):
                continue
            if st.get("pid") and alive(st["pid"]):
                st["status"] = "running_orphan"  # launched by an earlier queue instance
                continue
            if st["status"] in ("running", "running_orphan"):
                st["status"] = "exited"  # died while no queue instance was watching it
            if st["attempts"] >= MAX_ATTEMPTS:
                st["status"] = "failed"
                continue
            if method == "nnunetv2" and not nnunet_ready():
                st["status"] = "waiting_prep"
                continue
            pending.append((method, fold, key))

        stop = (LOG_DIR / "STOP").exists()
        active = [k for k, _ in running.values()] + [k for k, s in state.items()
                                                     if s["status"] == "running_orphan"]
        if pending and not stop and len(active) < args.max_jobs:
            busy = {g: 0 for g in gpus}
            for key in active:
                if state[key].get("gpu") in busy:
                    busy[state[key]["gpu"]] += 1
            p0_gpu, p0_pending = p0_load()
            free = gpu_free_mb()

            def fits(g):
                if busy[g] >= 1 or free.get(g, 0) < args.min_free_mb:
                    return False
                if p0_pending and p0_gpu.get(g, 0) < P0_PER_GPU:
                    return free[g] - args.job_mb >= args.min_free_mb  # leave P0 its launch room
                return True

            while pending and len(active) < args.max_jobs:
                cands = [g for g in gpus if fits(g)]
                if not cands:
                    break
                # cards P0 is not using first, then the card with the most free memory
                gpu = min(cands, key=lambda g: (p0_gpu.get(g, 0), -free[g]))
                method, fold, key = pending.pop(0)
                busy[gpu] += 1
                free[gpu] -= args.job_mb
                cmd = command(method, fold, gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                log = open(LOG_DIR / f"{key}.log", "a")
                log.write(f"\n# [{now()}] GPU {gpu} attempt {state[key]['attempts'] + 1}\n# {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[proc.pid] = (key, proc)
                active.append(key)
                state[key].update(status="running", pid=proc.pid, gpu=gpu,
                                  attempts=state[key]["attempts"] + 1, started=now())
                print(f"[{now()}] launched {key} on GPU {gpu} pid={proc.pid} "
                      f"(free {free[gpu] + args.job_mb} MB, P0 on card {p0_gpu.get(gpu, 0)}, "
                      f"P0 pending {p0_pending})", flush=True)

        state_path.write_text(json.dumps(state, indent=1))
        n_done = sum(s["status"] == "completed" for s in state.values())
        if n_done > summarized and (summary is None or summary.poll() is not None):
            summary = subprocess.Popen([sys.executable, str(ROOT / "tools" / "sota_summary.py")], cwd=ROOT,
                                       stdout=open(LOG_DIR / "summary.log", "w"),
                                       stderr=subprocess.STDOUT, start_new_session=True)
            summarized = n_done
            print(f"[{now()}] summary refresh started ({n_done} runs completed)", flush=True)
        unfinished = [k for k, s in state.items() if s["status"] not in ("completed", "failed")]
        if not running and (not unfinished or stop):
            if summary is not None:
                summary.wait()
            print(f"[{now()}] queue exit: unfinished={unfinished} stop={stop}", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
