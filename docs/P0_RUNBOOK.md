# P0 Stage-2 运行手册（protocol v0.3，binary Y）

启动于 2026-10-21 08:35，08:52 带先验缓存重启，09:08 以无缓冲日志从头重跑（ztest2，`/cfs/zzkj/home/dkq/cnv/code/topocnv_remote`，conda env `topocnv`）。

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
  （每例约 870 MB）重新解压。只改了读取路径，读到的值完全相同。
- 级联先验 [P(c4), dilate(hard c4)] 每例只计算一次，缓存为 `<case>_cascade_prior.npy`。原来训练时每取一个
  patch 都要读入整个 6 通道 softmax（约 1 GB），再做一次全体积的 argmax 和 3 mm 膨胀；8 个 run 并发时，
  GPU 利用率一直是 0%。改用缓存后，单个样本的读取从数秒降到 0.07 s。已抽查 02609977：缓存与直接从 NRRD
  计算的结果逐体素完全相同。两种缓存都由 `tools/prep_s1_npy.py` 预先生成；缓存缺失时也会在首次访问时原子写出。
- 第一次启动（08:35）的 9 个 run 停在第 1 个 epoch，尚无 checkpoint；已全部停掉，于 08:52 用先验缓存重新启动。
  旧的队列状态保存在 `queue_state.pre_prior_cache.json`。
- 队列启动的 run 设置了 `PYTHONUNBUFFERED=1`。此前 stdout 是块缓冲的，日志里看不到 epoch 行。
  08:52 那批 run 只跑了 2 个 epoch，09:08 删掉输出后从头重跑，让所有 run 的训练过程一致；
  旧日志在 `log/p0_binary_y_20261020/aborted_0835_0852/`。
  按这批 run 估算：首个 epoch（含冷缓存和验证）约 11 min，之后每个 epoch 约 1.1 min（8 个 run 并发）。

## 解读注意

- 训练日志里的 `val_Dice_Y` 是在中心 patch 上算的，只用于选 best checkpoint，会明显高估。
  主表只用全体积 OOF 预测（`p0_summary.py`）。
- Stage-1 的全体积 OOF 预测假阳性很多。例如 01501509 预测 102,585 个 Y 体素，真值只有 6,029 个；
  全体积 Y Dice 为 0.08–0.39（抽查 3 例），而 Stage-1 日志里的 patch 验证 Dice 约 0.58。
  已排除 NRRD 方向问题（交换 H/W 后 Dice 降到约 0.01）。所以 Secondary（取自 Stage-1）的数值会偏低。
- A2 的打乱方式：训练时在 patch 内做 `spatial_roll`，验证和推理时在整个体积上做。两者都破坏了空间对齐，
  但不完全相同，写 Methods 时需要说明。
- 补上 B0 的 `--stage1_pred_dir`（此前 B0 因找不到 Stage-1 目录而无法启动）。

## 2026-10-24 终止：methods §17.1 的七臂 fold0 重跑（v4 路线取代）

2026-10-24 11:10 按 methods §17.1 重跑的 P0 fold0 七臂确认放弃：协议 v0.4-draft 明确
"P0 结果作为历史记录保留，不进新稿的主表"，GPU 优先级也在 v4 之后；P0 5/5 主表早已完成。
full/b0（500 epoch，fold0）已跑完的结果只覆盖部分臂、无下游用途，目录内已放
ABANDONED_2026-10-24.md 标记（results/ 下 *_p0_full_protocol_v03_20261020* 与
*_p0_b0_protocol_v03_20261020* 四个目录及 log/p0_protocol_v03_20261020/）；a_mem/a_gate/c1/a2
四臂取消，不再排队。同日 nnunet 臂的坏结果（启动丢 flag 导致）已按原版配方重跑修复
（fold0 Dice_Y 0.6475 / HD95 12.79，sota_binary_y_nnunet/fold0 与 10-22 的 5/5 序列恢复一致）。
上述 §17.1 出处为 Mac 侧 methods 文档，服务器与本地仓库均无该节。
