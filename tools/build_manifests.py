#!/usr/bin/env python
"""Build D2 candidate manifests for the given folds (r1 + 4 prior arms)."""
import json, sys
from pathlib import Path

ROOT = Path("/cfs/zzkj/home/dkq/cnv/code/topocnv_remote")
ARMS = [
    ("r1",      "nnUNetTrainerV4_R1",  1),
    ("r1p_abc", "nnUNetTrainerV4_Pabc", 3),
    ("r1p_bc",  "nnUNetTrainerV4_Pbc",  2),
    ("r1p_ac",  "nnUNetTrainerV4_Pac",  2),
    ("r1p_ab",  "nnUNetTrainerV4_Pab",  2),
]

for k in [int(a) for a in sys.argv[1:]]:
    margin = json.loads((ROOT / f"configs/v4_qc/fold{k}.json").read_text())["margin"]
    cands = []
    for name, trainer, simple in ARMS:
        cfg = {"trainer": trainer}
        if name != "r1":
            cfg["margin"] = margin
        cands.append({"name": name, "simple": simple,
                      "csv": f"results/v4_dev_preds/{name}/fold{k}/eval.csv",
                      "config": cfg})
    man = {"tol": 0.003, "candidates": cands}
    out = ROOT / f"configs/v4_dev/fold{k}_candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1))
    print("wrote", out, "margin", margin)
