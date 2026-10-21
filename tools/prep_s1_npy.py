#!/usr/bin/env python3
"""
Build the mmap .npy cache of every Stage-1 out-of-fold softmax before Stage-2 runs start,
so parallel P0 runs never decompress the gzip NRRDs themselves.

  python tools/prep_s1_npy.py [--workers 6]
"""
import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
import ablation_configs as ac  # noqa: E402


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "s2trainer", ROOT / "trainer" / "npc_vesselFM3D_cascade_MIA_v3_vit_memory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def convert(case_id):
    return _trainer()._ensure_stage1_softmax_npy(Path(ac.STAGE1_CV_DIR), case_id).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    d = Path(ac.STAGE1_CV_DIR)
    todo = [p.name[: -len("_stage1_softmax.nrrd")] for p in sorted(d.glob("*_stage1_softmax.nrrd"))
            if not p.with_suffix(".npy").exists()]
    print(f"{len(todo)} to convert in {d}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for name in ex.map(convert, todo):
            print(name, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
