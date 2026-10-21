# P0 Stage-2 运行手册（protocol v0.3，binary Y）

启动于 2026-10-21 08:35（ztest2，`/cfs/zzkj/home/dkq/cnv/code/topocnv_remote`，conda env `topocnv`）。

## 跑什么

6 个臂 × 5 折 = 30 个 run，臂定义见 `MIA/scripts/ablation_configs.py`：
Full | A_mem | A_gate | B0 | C1 | A2。所有臂的 Stage-1 先验都来自
`..._stage1_s1_6class_protocol_v03_20261016_cv_predictions`（154 例 OOF，5 折均已完成）。
B0 只为数据加载器读取这个目录；`--b0_single_stage 1` 下网络输入只有 CT（`in_channels=1`）。

每个 run 最多 500 epoch，每 5 个 epoch 验证一次，early stop 耐心 50 epoch；
结束时写出验证折的 OOF 标签到 `results/<exp>_label_predictions/`，并写 `fold*/COMPLETED`。
实测（单卡、无并发）：每个训练 epoch 约 2.7 min，31 例全量滑窗预测约 10 min。

## 队列

```bash
cd /cfs/zzkj/home/dkq/cnv/code/topocnv_remote
source ~/miniconda3/etc/profile.d/conda.sh && conda activate topocnv && source env_cnv.sh
nohup python tools/p0_queue.py >> log/p0_binary_y_20261020/queue.log 2>&1 &
```

- 按折优先：所有臂的 fold 0 先跑，尽早得到可比的单折结果。
- GPU 均分：先给每张卡放 1 个任务，再放第 2 个；每卡最多 2 个（`--per_gpu`），总数最多 8 个（`--max_jobs`）；只在空闲显存 ≥ 40 GB 的卡上启动新任务（`--min_free_mb`）。队列不会向任何进程（包括别人的）发信号。
- 断点续训：run 进程死掉但没写 COMPLETED 时，带 `--resume` 从 `latest_checkpoint.pth` 重启，最多 3 次。
- 队列进程可以随时重启：仍在运行的 run 会被识别为 `running_orphan`，计入每卡配额。
  停掉队列只能按 PID：`kill $(pgrep -u dkq -f "^python tools/p0_queue")`。不要用
  `pkill -f p0_queue`，这个模式会把发起命令的 SSH shell 也匹配上。
- `touch log/p0_binary_y_20261020/STOP`：不再启动新 run，已在跑的继续跑。
- 每当完成的 run 增加，队列会在后台调用 `tools/p0_summary.py` 刷新主表。

状态：`log/p0_binary_y_20261020/queue_state.json`；每个 run 的日志：`log/p0_binary_y_20261020/<Arm>_fold<k>.log`。

## 主表

`python tools/p0_summary.py` 对每个臂的 OOF 标签调用 `MIA/scripts/eval_protocol_metrics.py`
（Primary：Dice_Y、HD95_Y；L1：Dice_L、Recall_L、|L|/|Y|；Secondary 用 Stage-1 多类预测，B0 为 N/A），
输出 `results/p0_binary_y_eval/<Arm>.csv` 和 `results/p0_binary_y_eval/P0_main_table.md`，
其中 p 值为各臂与 Full 在 Dice_Y 上的配对 Wilcoxon 检验。只有当 `folds` 一列为 5/5 时，这一行才是最终结果。

## 本轮修复

- Stage-2 验证和推理改为通过 mmap 的 `.npy` 缓存读取 Stage-1 softmax，不再每次验证都对整折的 gzip NRRD
  （每例约 870 MB）重新解压。只改了读取路径，读到的值完全相同。可用 `tools/prep_s1_npy.py` 预先生成缓存。
- 补上 B0 的 `--stage1_pred_dir`（此前 B0 因找不到 Stage-1 目录而无法启动）。
