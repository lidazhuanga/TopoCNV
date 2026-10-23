#!/usr/bin/env bash
# v4 end-to-end smoke test on one GPU: Stage 1 (3 inner models, 2 epochs) -> priors for 5 cases ->
# Stage 2 R0 and Full-v4 (2 epochs) -> OOF label readable by eval_protocol_metrics.py --pred_binary_Y.
# Writes only to results/v4_stage1/smoke, results/v4_smoke, results/v4_runs/*_smoke.
#   CUDA_VISIBLE_DEVICES=4 bash tests/v4_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export V4_PRIORS_DIR=$PWD/results/v4_smoke/priors
CASES=$(python -c "
from trainer.v4.splits import internal_split, report_cases
tr, iv = internal_split(0); print(','.join(tr[:2] + iv[:2] + report_cases(0)[:1]))")
echo "smoke cases: $CASES"
for j in 0 1 2; do
  python -m trainer.v4.train_stage1 --fold 0 --inner $j --smoke 4 --epochs 2 --num_workers 2 --smoke_cases "$CASES"
done
python tools/v4_build_priors.py --fold 0 --workers 5 --cases "$CASES" --stage1_root results/v4_stage1/smoke/fold0
python -m trainer.v4.train_stage2 --config configs/v4_dev/r0.yaml --fold 0 --smoke 2 --num_workers 2
python -m trainer.v4.train_stage2 --config tests/smoke_full.yaml --fold 0 --smoke 2 --num_workers 2
python - <<'EOF'
import torch
from pathlib import Path
from trainer.v4.common import RESULTS, load_config
from trainer.v4.model import build_dynunet, stage2_channels
from trainer.v4.splits import internal_split
from trainer.v4.train_stage2 import score_cases
cfg = load_config("tests/smoke_full.yaml")
dev = torch.device("cuda")
m = build_dynunet(*stage2_channels(cfg)).to(dev)
m.load_state_dict(torch.load(RESULTS / "v4_runs/smoke_full_smoke/fold0/best_model.pth")["model_state"])
out = RESULTS / "v4_smoke/labels"
out.mkdir(parents=True, exist_ok=True)
c = internal_split(0)[1][0]
print(score_cases(m, [c], 0, cfg, dev, False, label_dir=out, vis_dir=out))
EOF
python MIA/scripts/eval_protocol_metrics.py --pred_dir results/v4_smoke/labels --pred_binary_Y --no_hu \
  --gt_cache results/preprocess_cache/sp_2.500_1.133_1.133 --csv results/v4_smoke/eval.csv
cat results/v4_smoke/eval.csv
echo SMOKE_OK
