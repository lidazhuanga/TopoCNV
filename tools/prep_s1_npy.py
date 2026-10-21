#!/usr/bin/env python3
"""
Build the Stage-2 input caches for every Stage-1 out-of-fold prediction before Stage-2 runs
start, so parallel P0 runs never rebuild them per sample:
  <case>_stage1_softmax.npy   mmap copy of the gzip NRRD softmax
  <case>_cascade_prior.npy    [P(c4), dilate(hard c4)] used as the Stage-2 prior

  python tools/prep_s1_npy.py [--workers 16]
"""
import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
sys.path.insert(0, str(ROOT / "trainer"))  # trainer imports cnv_protocol_s2 from its own dir
import ablation_configs as ac  # noqa: E402

_T = None


def _trainer():
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location(
            "s2trainer", ROOT / "trainer" / "npc_vesselFM3D_cascade_MIA_v3_vit_memory.py")
        _T = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_T)
        assert _T._HAS_PROTOCOL_S2, "cnv_protocol_s2 not importable; prior cache would be wrong"
    return _T


def build(case_id):
    prior = _trainer()._load_cascade_prior(Path(ac.STAGE1_CV_DIR), case_id)
    return f"{case_id} prior={tuple(prior.shape)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    d = Path(ac.STAGE1_CV_DIR)
    todo = [p.name[: -len("_stage1_softmax.nrrd")] for p in sorted(d.glob("*_stage1_softmax.nrrd"))
            if not (d / p.name.replace("_stage1_softmax.nrrd", "_cascade_prior.npy")).exists()]
    print(f"{len(todo)} to build in {d}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(build, todo):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
