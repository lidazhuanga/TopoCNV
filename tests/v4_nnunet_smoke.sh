#!/bin/bash
# Real nnU-Net v2 path smoke (V4_PLAN §8 precondition; review P0-3): dev-tier Stage 1 -> dev priors
# -> tiny Dataset52k -> 2-epoch trainer (2+5 outputs, class5 ignore, Skeleton, Channel0Only) ->
# export (ROI paste-back, _vis.nrrd) -> evaluator read-back. Values are meaningless (2 epochs,
# 3 training cases); this only proves the plumbing. Uses scratch V4_PRIORS_DIR / CNV_NNUNET_HOME.
#
#  bash tests/v4_nnunet_smoke.sh [gpu_id]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/topocnv/bin/python
NNPY=$HOME/cnv/envs/nnunetv2_cnv/bin/python
GPU=${1:-4}
export CUDA_VISIBLE_DEVICES=$GPU
SCRATCH=results/v4_smoke_nn
REAL_HOME=$HOME/cnv/nnunet_cnv
export V4_PRIORS_DIR=$PWD/$SCRATCH/priors
export CNV_NNUNET_HOME=$PWD/$SCRATCH/nnunet
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY" "$SCRATCH/nnunet/nnUNet_raw" \
         "$SCRATCH/nnunet/nnUNet_results" "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset519_CNVv4R0dev"
cp "$REAL_HOME/nnUNet_preprocessed/Dataset501_CNVbinaryY/nnUNetPlans.json" \
   "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY/"

echo "== [smoke 1/5] case list =="
CASES=$($PY -c "import sys; sys.path.insert(0,'.'); from topocnv.data.v4_splits import inner_folds, internal_split; \
print(','.join([inner_folds(0, tier='dev')[j][0] for j in range(3)] + [internal_split(0)[1][0]]))")
VCASE=$($PY -c "import sys; sys.path.insert(0,'.'); from topocnv.data.v4_splits import internal_split; print(internal_split(0)[1][0])")
echo "cases: $CASES (val case: $VCASE)"

echo "== [smoke 2/5] dev-tier Stage 1 (3 inner x 3 cases x 2 epochs) =="
for j in 0 1 2; do
  $PY -m trainer.v4.train_stage1 --fold 0 --inner "$j" --tier dev --smoke 3 --epochs 2 \
      --smoke_cases "$CASES" | tail -2
done

echo "== [smoke 3/5] dev priors (atlas from real labels; 4 smoke cases) =="
$PY tools/v4_build_priors.py --fold 0 --variant dev \
    --stage1_root results/v4_stage1/smoke/fold0_dev --cases "$CASES" --workers 8 | tail -2

echo "== [smoke 4/5] tiny dataset prep + 2-epoch train + export =="
$NNPY tools/v4_nnunet.py prep --fold 0 --variant dev --margin 25 --cases "$CASES" --np 4 | tail -3
$NNPY tools/v4_nnunet.py train --fold 0 --variant dev --trainer nnUNetTrainerV4_SMOKE_Pabc_Vign_S10 \
    --name smoke_nn --cases "$CASES" | tail -5

echo "== [smoke 5/5] evaluator read-back =="
test -f "results/v4_dev_preds/smoke_nn/fold0/${VCASE}_stage2_label.nrrd"
test -f "results/v4_dev_preds/smoke_nn/fold0/${VCASE}_vis.nrrd"
$PY MIA/scripts/eval_protocol_metrics.py --pred_dir results/v4_dev_preds/smoke_nn/fold0 \
    --gt_cache "$PWD/results/preprocess_cache/sp_2.500_1.133_1.133" \
    --pred_binary_Y --no_hu --csv "$SCRATCH/smoke_eval.csv" | tail -4
N=$(awk -F, 'NR>1 && $1!="MEAN"' "$SCRATCH/smoke_eval.csv" | wc -l)   # evaluator appends a MEAN row
if [ "$N" -ne 1 ]; then echo "SMOKE FAIL: expected exactly 1 scored case, got $N"; exit 1; fi
grep -q "Empty_pred_Y" "$SCRATCH/smoke_eval.csv" || { echo "SMOKE FAIL: no Empty_pred_Y column"; exit 1; }
grep -q "wDSC_Y_0.50" "$SCRATCH/smoke_eval.csv" || { echo "SMOKE FAIL: no wDSC_Y_0.50 column"; exit 1; }
grep -q "Recall_GT3" "$SCRATCH/smoke_eval.csv" || { echo "SMOKE FAIL: no Recall_GT3 column"; exit 1; }
echo "SMOKE_OK (values meaningless; plumbing validated)"
