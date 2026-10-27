#!/bin/bash
# Watch v4_dev_preds for the given arms (space-separated in $ARMS): auto-evaluate each fold as soon
# as EXPORTED appears; exit when every fold of every arm has eval.csv. Usage: ARMS="a b c" bash tools/eval_watcher2.sh
cd /cfs/zzkj/home/dkq/cnv/code/topocnv_remote
PY=~/miniconda3/envs/topocnv/bin/python
ARMS=${ARMS:-"r1v_bg_s05 r1v_bg_s10 r1v_ign_s05 r1v_ign_s10"}
LOG=log/v4/eval_watcher2_main.log
while true; do
  alldone=1
  for run in $ARMS; do
    for k in 0 1 2 3 4; do
      fd=results/v4_dev_preds/$run/fold$k
      if [ ! -d "$fd" ]; then alldone=0; continue; fi
      if [ -f "$fd/EXPORTED" ] && [ ! -f "$fd/eval.csv" ]; then
        echo "$(date +%m-%d\ %H:%M) evaluating $run fold$k" >> $LOG
        $PY MIA/scripts/eval_protocol_metrics.py --pred_dir "$fd" \
          --gt_cache results/preprocess_cache/sp_2.500_1.133_1.133 \
          --pred_binary_Y --no_hu --csv "$fd/eval.csv" >> $LOG 2>&1
        echo "$(date +%m-%d\ %H:%M) done $run fold$k" >> $LOG
      fi
      [ -f "$fd/eval.csv" ] || alldone=0
    done
  done
  if [ "$alldone" = "1" ]; then echo "$(date +%m-%d\ %H:%M) WATCHER2_ALL_DONE arms=$ARMS" >> $LOG; break; fi
  sleep 600
done
