#!/bin/bash
# V4.2 prior-path smoke (V4_PLAN §11.6.1-11.6.2): tiny Dataset540/541 copies in a scratch nnU-Net home, then on
# one GPU, 2 epochs x 5 iterations each:
#   1. Stage 1 stock smoke model (inner 0 of dev fold 0) -> predict42's sliding window on its own held-out case
#      reproduces nnU-Net's validation/*.npz (the inputs of the priors); predict42 on a V_0 case (holdout42/)
#   2. fake priors (from the GT; values meaningless) -> prep42p: 4-channel Dataset542 with 3 copies of the V_0 case
#   3. train42 --dataset prior with a smoke prior arm -> validation_v4, p_Y of the 3 copies, export = mean > 0.5
# No production dataset, marker or prior file is touched; on success writes log/v4/smoke42p.done, the gate of
# the D2' chain (Stage 1 holdout prediction onwards).
#
#  bash tests/v4_nnunet_smoke42p.sh <gpu_id>      # or 'queue': use the CUDA_VISIBLE_DEVICES the queue set
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/topocnv/bin/python
NNPY=$HOME/cnv/envs/nnunetv2_cnv/bin/python
GPU=${1:?usage: bash tests/v4_nnunet_smoke42p.sh <gpu_id>|queue}
[ "$GPU" = queue ] && GPU=${CUDA_VISIBLE_DEVICES:?queue mode needs CUDA_VISIBLE_DEVICES}
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
if [ "$FREE" -lt 40960 ]; then echo "GPU $GPU has $FREE MiB free (< 40 GB): not starting"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU
SCRATCH=$PWD/results/v4_smoke_nn42p
REAL_HOME=$HOME/cnv/nnunet_cnv
HARM=$PWD/results/v4_harm_v1/sp_2.500_1.133_1.133
export CNV_NNUNET_HOME=$SCRATCH/nnunet V42_PRIORS_DIR=$SCRATCH/priors42 V42_QC_DIR=$SCRATCH/qc42
rm -rf "$SCRATCH" results/v4_dev_preds/smoke42p
rm -f log/v4/smoke42p.done                  # a failed rerun must not leave an earlier pass in place
mkdir -p "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY" "$SCRATCH/nnunet/nnUNet_raw" \
         "$SCRATCH/nnunet/nnUNet_results"
cp "$REAL_HOME/nnUNet_preprocessed/Dataset501_CNVbinaryY/nnUNetPlans.json" \
   "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY/"
# production paths only the D2' chain writes; its jobs wait for this smoke, so none may appear while it runs
# (other queue jobs keep writing their own markers meanwhile, so log/v4 as a whole is not compared)
marks() { for p in log/v4/s1p42_dev_f0_i0.done log/v4/nnprep_542.done results/v4_priors42/fold0_dev                    configs/v4_qc42/fold0.json results/v4_dev_preds/v42_d2abc; do [ -e "$p" ] && echo "$p"; done; true; }
MARKS_BEFORE=$(marks)

echo "== [smoke42p 1/5] case list =="
read -r CASES VCASE S1VAL < <($PY -c "
import sys; sys.path.insert(0, '.')
from trainer.v4.splits import inner_folds, internal_split
inner = [inner_folds(0, tier='dev')[j][0] for j in range(3)]
v = internal_split(0)[1][0]
print(','.join(inner + [v]), v, inner[0])")
echo "cases $CASES (V_0 case $VCASE, Stage 1 held-out case $S1VAL)"
$NNPY tools/v4_nnunet.py prep42 --what binY --cases "$CASES" --np 4 | tail -1
$NNPY tools/v4_nnunet.py prep42 --what s1 --cases "$CASES" --np 4 | tail -1

echo "== [smoke42p 2/5] Stage 1 smoke model: predict42 vs nnU-Net validation, holdout prediction =="
$NNPY tools/v4_nnunet.py train42 --dataset s1 --fold 0 --variant dev --inner 0 --trainer nnUNetTrainer_SMOKE_2x5 | tail -1
$NNPY tools/v4_nnunet.py predict42 --fold 0 --variant dev --inner 0 --trainer nnUNetTrainer_SMOKE_2x5 --cases "$VCASE" | tail -1
$NNPY - "$S1VAL" "$VCASE" <<'EOF'
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")
import v4_nnunet as N

c, v = sys.argv[1], sys.argv[2]
fd, split = N.s1_dir(0, "dev", 0, N.SMOKE_STOCK)
out = N._predict42(fd, split, N.V42["s1"][1], [c], Path(N.NNUNET_HOME) / "parity", "s1")
a = np.load(out / f"{c}.npz")["probabilities"]
b = np.load(fd / "validation" / f"{c}.npz")["probabilities"]
assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
diff = float(np.abs(a - b).max())
agree = float((a.argmax(0) == b.argmax(0)).mean())
assert diff < 1e-3 and agree > 0.9999, f"predict42 differs from nnU-Net validation: max |dp| {diff}, argmax {agree}"
print(f"predict42 parity OK: max |dp| {diff:.2e}, argmax agreement {agree:.6f}")
h = np.load(fd / "holdout42" / f"{v}.npz")["probabilities"]
shape = np.load(N.CACHE_DIR / f"{v}_label.npy", mmap_mode="r").shape
assert h.shape == (5,) + shape and np.allclose(h.sum(0), 1, atol=1e-3), h.shape
print(f"holdout42 OK: {h.shape}")
EOF

echo "== [smoke42p 3/5] fake priors -> prep42p (Dataset542, 3 copies of $VCASE) =="
$NNPY - "$CASES" "$VCASE" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")
import v4_nnunet as N

cases, v = sys.argv[1].split(","), sys.argv[2]
cache, sha, digest = N.harm_cache(verify=False)
N.QC42.mkdir(parents=True, exist_ok=True)
qc = N.QC42 / "fold0.json"
qc.write_text(json.dumps({"amendment": "D1-AM2", "fold": 0, "margin": 25, "smoke": True}))
out = N.PRIORS42 / "fold0_dev"
out.mkdir(parents=True, exist_ok=True)
ids = {}
for c in cases:
    for i in ([N.copy_id(c, j) for j in range(3)] if c == v else [c]):
        y = (np.load(cache / f"{c}_label.npy") % 5) > 0
        idx = np.argwhere(y)
        lo, hi = idx.min(0), idx.max(0) + 1
        box = np.stack([np.maximum(lo - 4, 0), np.minimum(hi + 4, y.shape)], 1)
        sl = tuple(slice(a, b) for a, b in box)
        rng = np.random.default_rng(len(ids))
        pri = np.stack([y[sl] * 255, rng.integers(0, 256, y[sl].shape), rng.integers(0, 256, y[sl].shape)]).astype(np.uint8)
        N._save_npz(out / f"{i}_prior.npz", prior=pri, bbox=box)
        ids[i] = {"case": c, "sha": N.prior_sha(pri, box), "stage1_sha256": "smoke"}
(out / "DONE").write_text(json.dumps({"fold": 0, "variant": "dev", "margin": 25, "qc42_sha256": N._file_sha(qc),
                                      "harm_config_sha256": sha, "harm_label_digest": digest, "ids": ids,
                                      "digest": N._digest({i: r["sha"] for i, r in ids.items()})}))
print(f"fake priors for {sorted(ids)}")
EOF
$NNPY tools/v4_nnunet.py prep42p --fold 0 --cases "$CASES" --np 4 | tail -1

echo "== [smoke42p 4/5] prior arm: train, 3-copy p_Y, export =="
$NNPY tools/v4_nnunet.py train42 --dataset prior --fold 0 --variant dev --priors abc \
    --trainer nnUNetTrainerV4_SMOKE_Pabc_Vign_S10 --name smoke42p --cases "$CASES" | tail -1
$NNPY - "$VCASE" <<'EOF'
import sys

import nrrd
import numpy as np

sys.path.insert(0, "tools")
import v4_nnunet as N

v = sys.argv[1]
did, name = N.v42p(0, "dev")
fd = N.results_dir(did, name, "nnUNetTrainerV4_SMOKE_Pabc_Vign_S10", 0)
ps = [np.load(fd / "pY42" / f"{N.copy_id(v, j)}.npz")["pY"] for j in range(3)]
pred, _ = nrrd.read(str(N.RESULTS / "v4_dev_preds" / "smoke42p" / "fold0" / f"{v}_stage2_label.nrrd"))
want = ((ps[0] + ps[1] + ps[2]) / np.float32(3) > 0.5)
assert pred.shape == want.shape and np.array_equal(pred > 0, want), "export != mean of the 3 copies > 0.5"
assert (N.RESULTS / "v4_dev_preds" / "smoke42p" / "fold0" / "EXPORTED_PARTIAL").exists()
assert not (N.RESULTS / "v4_dev_preds" / "smoke42p" / "fold0" / "EXPORTED").exists()
y = np.load(fd / "validation_v4" / f"{N.copy_id(v, 0)}.npz")["Y"]
agree = float(((ps[0] > 0.5) == (y > 0)).mean())
assert agree > 0.9999, f"copy 0 p_Y > 0.5 vs validation_v4 Y agree on {agree}"
print(f"3-copy export OK; copy 0 p_Y vs validation_v4 Y agree on {agree:.6f}")
EOF

echo "== [smoke42p 5/5] production markers untouched =="
if [ "$(marks)" != "$MARKS_BEFORE" ]; then echo "SMOKE42P FAIL: a production marker or prior file changed"; exit 1; fi
echo "$(date -Iseconds) smoke42p OK on GPU $GPU, HEAD $(git rev-parse --short HEAD)" > log/v4/smoke42p.done
echo "SMOKE42P_OK (values meaningless; plumbing validated)"
