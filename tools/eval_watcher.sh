#!/bin/bash
# Watch v4_dev_preds: auto-evaluate each fold as soon as EXPORTED appears; exit when all 25 wave-1 folds have eval.csv.
cd /cfs/zzkj/home/dkq/cnv/code/topocnv_remote
PY=~/miniconda3/envs/topocnv/bin/python
while true; do
  alldone=1
  for run in r1 r1p_abc r1p_bc r1p_ac r1p_ab; do
    for k in 0 1 2 3 4; do
      fd=results/v4_dev_preds/$run/fold$k
      if [ ! -d "$fd" ]; then alldone=0; continue; fi
      if [ -f "$fd/EXPORTED" ] && [ ! -f "$fd/eval.csv" ]; then
        echo "$(date +%m-%d\ %H:%M) evaluating $run fold$k"
        $PY MIA/scripts/eval_protocol_metrics.py --pred_dir "$fd" \
          --gt_cache results/preprocess_cache/sp_2.500_1.133_1.133 \
          --pred_binary_Y --no_hu --csv "$fd/eval.csv" >> log/v4/eval_watcher.log 2>&1
        echo "$(date +%m-%d\ %H:%M) done $run fold$k"
      fi
      [ -f "$fd/eval.csv" ] || alldone=0
    done
  done
  if [ "$alldone" = "1" ]; then echo "$(date +%m-%d\ %H:%M) ALL_25_DEV_EVALS_DONE"; break; fi
  sleep 600
done
