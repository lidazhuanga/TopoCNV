#!/usr/bin/env python3
"""v4 job queue (runs on the training host under nohup).

Jobs come from log/v4/jobs.json (re-read every poll, so jobs can be appended while it runs):
  {"key": str, "cmd": [argv...], "done": "path relative to repo", "deps": [keys], "kind": "gpu"|"cpu",
   "resume_arg": "--resume" (optional, appended on relaunch)}
A job is done when its `done` file exists; it starts when all deps are done.

GPU rules (protocol v0.4 §8; main session 2026-10-22): at most --max_gpu_jobs (4) v4 runs, at most one
v4 run per card (counted for this session only; sharing a card with cnv-sota is allowed), only on cards
with >= --min_free_mb free. nnU-Net v2 keeps priority: while it still has runs waiting in the SOTA queue,
a card must keep --min_free_mb free after our run starts. The queue never signals any process.

  nohup python tools/v4_queue.py >> log/v4/queue.log 2>&1 &
  touch log/v4/STOP     # stop launching (running jobs continue)
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log" / "v4"
SOTA_STATE = ROOT / "log" / "sota_binary_y_20261021" / "queue_state.json"  # read only
MAX_ATTEMPTS = 3


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def gpu_free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, tot, used = [int(x) for x in line.split(",")]
        free[str(i)] = tot - used
    return free


def alive(pid):
    try:
        os.kill(pid, 0)  # existence check only, no signal is delivered
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def nnunetv2_waiting():
    try:
        st = json.loads(SOTA_STATE.read_text())
    except (OSError, ValueError):
        return False
    return any(k.startswith("nnunetv2_") and s.get("status") in ("pending", "waiting_prep", "exited")
               for k, s in st.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--min_free_mb", type=int, default=40000)
    ap.add_argument("--job_mb", type=int, default=30000, help="expected peak memory of one v4 run")
    ap.add_argument("--max_gpu_jobs", type=int, default=4)
    ap.add_argument("--max_cpu_jobs", type=int, default=2)
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state_path = LOG_DIR / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    gpus = args.gpus.split(",")
    running = {}
    print(f"[{now()}] v4 queue start", flush=True)
    while True:
        jobs = json.loads((LOG_DIR / "jobs.json").read_text())
        byk = {j["key"]: j for j in jobs}
        for pid, (key, proc) in list(running.items()):
            if proc.poll() is not None:
                state[key].update(status="exited", rc=proc.returncode, ended=now())
                print(f"[{now()}] {key} exited rc={proc.returncode}", flush=True)
                del running[pid]

        def done(key):
            return (ROOT / byk[key]["done"]).exists()

        pending = []
        for j in jobs:
            key = j["key"]
            st = state.setdefault(key, {"attempts": 0, "status": "pending"})
            if done(key):
                st["status"] = "completed"
                continue
            if any(k == key for k, _ in running.values()):
                continue
            if st.get("pid") and alive(st["pid"]):
                st["status"] = "running_orphan"
                continue
            if st["status"] in ("running", "running_orphan"):
                st["status"] = "exited"
            if st["attempts"] >= MAX_ATTEMPTS:
                st["status"] = "failed"
                continue
            if not all(d in byk and done(d) for d in j.get("deps", [])):
                st["status"] = "waiting_deps"
                continue
            st["status"] = "pending"
            pending.append(j)

        stop = (LOG_DIR / "STOP").exists()
        active = [k for k, _ in running.values()] + [k for k, s in state.items() if s["status"] == "running_orphan"]
        act_gpu = [k for k in active if state[k].get("gpu") is not None]
        act_cpu = [k for k in active if state[k].get("gpu") is None]
        if pending and not stop:
            free = gpu_free_mb()
            ours = {state[k]["gpu"] for k in act_gpu}
            protect = nnunetv2_waiting()
            for j in pending:
                key = j["key"]
                gpu = None
                if j.get("kind", "gpu") == "gpu":
                    if len(act_gpu) >= args.max_gpu_jobs:
                        continue
                    need = args.min_free_mb + (args.job_mb if protect else 0)
                    cands = [g for g in gpus if g not in ours and free.get(g, 0) >= max(need, args.min_free_mb)]
                    if not cands:
                        continue
                    gpu = max(cands, key=lambda g: free[g])
                elif len(act_cpu) >= args.max_cpu_jobs:
                    continue
                cmd = list(j["cmd"])
                if state[key]["attempts"] > 0 and j.get("resume_arg"):
                    cmd.append(j["resume_arg"])
                env = dict(os.environ, PYTHONUNBUFFERED="1")
                env["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
                log = open(LOG_DIR / f"{key}.log", "a")
                log.write(f"\n# [{now()}] GPU {gpu} attempt {state[key]['attempts'] + 1}\n# {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[proc.pid] = (key, proc)
                state[key].update(status="running", pid=proc.pid, gpu=gpu, attempts=state[key]["attempts"] + 1,
                                  started=now())
                if gpu is not None:
                    act_gpu.append(key)
                    ours.add(gpu)
                    free[gpu] -= args.job_mb
                else:
                    act_cpu.append(key)
                print(f"[{now()}] launched {key} on GPU {gpu} pid={proc.pid} (nnUNetv2 protected: {protect})",
                      flush=True)
        state_path.write_text(json.dumps(state, indent=1))
        unfinished = [j["key"] for j in jobs if state[j["key"]]["status"] not in ("completed", "failed")]
        if not running and (not unfinished or stop):
            print(f"[{now()}] queue exit: unfinished={unfinished} stop={stop}", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
