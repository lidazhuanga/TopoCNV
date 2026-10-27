#!/bin/bash
# V4.2 real-path smoke (V4_PLAN §11.7 step 0; qoder review 6 P1-5). Tiny copies of Dataset540/541 (4 cases,
# harm_v1 labels) in a scratch nnU-Net home, then on one GPU, 2 epochs x 5 iterations each:
#   1. nnUNetTrainerV4_SMOKE_noP_S10 (Skeleton Recall) on binY dev fold 0 -> validation_v4 -> export42
#   2. nnUNetTrainer_SMOKE_2x5 (stock) on binY dev fold 0 -> validation/*.nii.gz -> export42 (stock path)
#   3. nnUNetTrainer_SMOKE_2x5 on s1 dev fold 0 inner 0 with --npz -> 5-class probabilities on the cache grid,
#      argmax == nnU-Net's NIfTI after the (2,1,0) transpose; no queue marker written
#   4. evaluator on both exports against harm_v1 (exactly the one validation case scored)
# Values are meaningless; this proves the plumbing only. The production datasets and markers are not touched.
#
#  bash tests/v4_nnunet_smoke42.sh <gpu_id>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/topocnv/bin/python
NNPY=$HOME/cnv/envs/nnunetv2_cnv/bin/python
GPU=${1:?usage: bash tests/v4_nnunet_smoke42.sh <gpu_id>}
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
if [ "$FREE" -lt 40960 ]; then echo "GPU $GPU has $FREE MiB free (< 40 GB): not starting"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU
SCRATCH=results/v4_smoke_nn42
REAL_HOME=$HOME/cnv/nnunet_cnv
HARM=$PWD/results/v4_harm_v1/sp_2.500_1.133_1.133
export CNV_NNUNET_HOME=$PWD/$SCRATCH/nnunet
rm -rf "$SCRATCH" results/v4_dev_preds/smoke42_v4 results/v4_dev_preds/smoke42_stock
mkdir -p "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY" "$SCRATCH/nnunet/nnUNet_raw" \
         "$SCRATCH/nnunet/nnUNet_results"
cp "$REAL_HOME/nnUNet_preprocessed/Dataset501_CNVbinaryY/nnUNetPlans.json" \
   "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY/"
marks() { ls log/v4/ | grep -E '^(nnprep_54|s142_)' | sort | md5sum; }
MARKS_BEFORE=$(marks)

echo "== [smoke42 1/5] case list =="
# binY dev fold 0: train = the first case of each dev-tier inner fold, val = a thin-label case of V_0;
# s1 dev fold 0 inner 0: train = inner folds 1, 2, val = inner fold 0 (V_0 is held out of Stage 1)
read -r CASES VCASE S1VAL < <($PY -c "
import json, sys; sys.path.insert(0, '.')
from trainer.v4.splits import inner_folds, internal_split
man = json.load(open('$HARM/manifest.json'))['cases']
inner = [inner_folds(0, tier='dev')[j][0] for j in range(3)]
v = next(c for c in internal_split(0)[1] if man[c]['thin'])
print(','.join(inner + [v]), v, inner[0])")
echo "cases $CASES (binY val $VCASE, s1 val $S1VAL)"

echo "== [smoke42 2/5] tiny Dataset540 / Dataset541 =="
$NNPY tools/v4_nnunet.py prep42 --what binY --cases "$CASES" --np 4 | tail -1
$NNPY tools/v4_nnunet.py prep42 --what s1 --cases "$CASES" --np 4 | tail -1

echo "== [smoke42 3/5] binY dev fold 0: V4 trainer (S1.0) and stock trainer -> export42 =="
$NNPY tools/v4_nnunet.py train42 --dataset binY --fold 0 --variant dev \
    --trainer nnUNetTrainerV4_SMOKE_noP_S10 --name smoke42_v4 --cases "$CASES" | tail -1
$NNPY tools/v4_nnunet.py train42 --dataset binY --fold 0 --variant dev \
    --trainer nnUNetTrainer_SMOKE_2x5 --name smoke42_stock --cases "$CASES" | tail -1

echo "== [smoke42 4/5] s1 dev fold 0 inner 0 (stock, --npz) =="
$NNPY tools/v4_nnunet.py train42 --dataset s1 --fold 0 --variant dev --inner 0 \
    --trainer nnUNetTrainer_SMOKE_2x5 | tail -1
$NNPY - "$S1VAL" <<'EOF'
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from trainer.v4.common import CACHE_DIR

c = sys.argv[1]
fd = (Path(os.environ["CNV_NNUNET_HOME"]) / "nnUNet_results" / "Dataset541_CNVv42s1"
      / "nnUNetTrainer_SMOKE_2x5__nnUNetPlans__3d_fullres" / "fold_0" / "validation")
p = np.load(fd / f"{c}.npz")["probabilities"]
shape = tuple(np.load(CACHE_DIR / f"{c}_label.npy", mmap_mode="r").shape)
assert p.shape == (5,) + shape, (p.shape, shape)
assert np.allclose(p.sum(0), 1, atol=1e-3), "probabilities do not sum to 1"
seg = np.asarray(nib.load(str(fd / f"{c}.nii.gz")).dataobj).transpose(2, 1, 0)
agree = float((seg == p.argmax(0)).mean())
assert agree > 0.9999, f"argmax(npz) vs transposed NIfTI agree on only {agree:.6f}"
print(f"s1 npz OK: {p.shape} {p.dtype}, argmax == transposed NIfTI on {agree:.6f} of voxels (cache axis order)")
EOF
if [ "$(marks)" != "$MARKS_BEFORE" ]; then echo "SMOKE42 FAIL: a production marker was written"; exit 1; fi
for d in "$CNV_NNUNET_HOME"/nnUNet_results/Dataset54*/*/fold_*; do
  test -s "$d/v42_runs.jsonl" || { echo "SMOKE42 FAIL: no run record in $d"; exit 1; }
done

echo "== [smoke42 5/5] evaluator against harm_v1 =="
for n in smoke42_v4 smoke42_stock; do
  test -f "results/v4_dev_preds/$n/fold0/${VCASE}_stage2_label.nrrd"
  $PY MIA/scripts/eval_protocol_metrics.py --pred_dir "results/v4_dev_preds/$n/fold0" --gt_cache "$HARM" \
      --pred_binary_Y --no_hu --csv "$SCRATCH/eval_$n.csv" | tail -1
  N=$(awk -F, 'NR>1 && $1!="MEAN"' "$SCRATCH/eval_$n.csv" | wc -l)   # evaluator appends a MEAN row
  if [ "$N" -ne 1 ]; then echo "SMOKE42 FAIL: $n scored $N cases, expected 1"; exit 1; fi
done
echo "SMOKE42_OK (values meaningless; plumbing validated)"
