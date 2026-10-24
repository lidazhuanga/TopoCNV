# SOTA 对照运行手册（binary Y，protocol v0.3 §14.4）

对照方法在 binary Y 终点（Y = 1∪2∪3∪4，class5_bypass 记为背景）上做 5 折重跑。主基线是标准 nnU-Net v2；
其余方法来自 `trainer/npc_sota_train.py` 的 registry。全部方法用与 P0 相同的数据、折划分、缓存和指标，
不再使用五类均值。所有 SOTA 方法统一为 **300 epoch** 上限（P0 为 500 epoch，见下文"与 P0 的差异"）。

## 方法清单

| 方法 | 来源 | 输入 / 输出 | 说明 |
|---|---|---|---|
| nnU-Net v2 | 原版 `nnunetv2==2.5.2`（PyPI），独立 venv `~/cnv/envs/nnunetv2_cnv` | CT 缓存 → 1 通道 Y 概率 | `3d_fullres`，默认 plans，`nnUNetTrainer_300epochs`，通道 `nonorm`；由 `tools/sota_nnunet.py` 驱动 |
| DynUNet（nnU-Net 风格） | registry `nnunet` | 同上 | 6 层 DynUNet，filters 32–320，残差块 |
| SegResNet | registry `segresnet` | 同上 | init_filters 16 |
| Attention U-Net | registry `attention_unet` | 同上 | channels 16–256 |
| UNETR | registry `unetr` | 同上 | hidden 768，12 头，patch 128³ |
| SwinUNETR | registry `swin_unetr` | 同上 | feature_size 48 |
| nnFormer（简化版） | registry `nnformer`（`NNFormer3D`） | 同上 | 卷积 + 窗口注意力编码器，U-Net 解码器 |

registry 方法在 `trainer/npc_sota_train.py` 中实现（`build_model`），本次只改了验证和保存部分。nnFormer 是仓内的简化实现，
不是官方代码，手册和论文里要这样写。

**UNETR 没有收敛，是欠训练的弱基线。** 5 折的训练 Dice 最后一个 epoch 都在 0.27–0.30 左右，验证 Dice_Y 约 0.29，远低于其他方法。
这不是评价的问题，而是这套配方（Adam 5e-4、300 epoch、ViT 骨干）对 UNETR 不适用。按现有配方报告为弱基线，正文中要注明它没有收敛，
不能作为「ViT 类方法不如 CNN」的证据。没有为它单独调参。

## 共同配方（registry 方法）

- 数据：`results/preprocess_cache/sp_2.500_1.133_1.133` 中的 `{case}_image.npy`（百分位归一化 CT）和 `{case}_label.npy`；
  标签经 `convert_label_to_binary_Y` 转为 Y；训练 patch 128³，batch 2，前景偏置裁剪，`--random_crop_prob 0.33`
  （约 1/3 的 patch 为随机位置）。
- 优化：Adam，lr 5e-4，cosine 退火到 1%（T_max = 300），`WeightedBCEDiceLoss`（BCE 0.3 + Dice 0.7，focal γ = 3），
  梯度裁剪 1.0，AMP bf16；最多 300 epoch。
- 选模型：`--select_by_sliding_window`。每 25 epoch 对当前模型做全体积滑窗推理，按全体积 Dice_Y 选 best；
  连续 6 次检查（150 epoch）没有提升则早停。
- 全体积滑窗：`npc_sota_train.validate_binary_y_cache_sw`，在缓存空间做 MONAI 滑窗（128³，重叠 50%，gaussian 权重，
  sigma_scale 0.125），softmax 前景 > 0.5 即 Y。

## 公平性设定（与 P0 对齐）

- 数据：154 例训练队列，同一缓存网格（2.5 / 1.133 / 1.133 mm，z/y/x）和同一 Y 标签。
- 折划分：`extras/splits.json`，即 `_create_folds(seed=42)`（2026-10-21 已逐折核对一致）。
- 预测格式：OOF 标签保存为 `<case>_stage2_label.nrrd`（int16，gzip，与 P0 相同），
  `results/sota_binary_y_<method>_label_predictions/`。
- 评价：`MIA/scripts/eval_protocol_metrics.py --pred_binary_Y`（只读调用，不改代码）。单阶段二值方法没有多类输出，
  Secondary（Dice_123、Dice_c4）记为 N/A，与 P0 的 B0 一致。
- 汇总：`tools/sota_summary.py` 写 `results/sota_binary_y_eval/SOTA_main_table.md`，并把 P0 Full 与 B0 作为参照行
  （只读 `results/p0_binary_y_eval_v2/`），用配对 Wilcoxon 检验 vs P0 Full。

## 评价缺例（已修正，2026-10-23）

nnU-Net v2 的初版评价只有 149 例，缺了第 4 折的 5 例：`00816783`、`02270253`、`03460775`、`03490781`、`04837013`。
原因：汇总脚本在第 4 折的标签导出（01:47–01:48）完成之前就启动了 nnU-Net 的评价，评价按顺序读病例，读到这 5 例时文件还没有写出来，
报了「no pred」。这 5 例正是第 4 折按 ID 排序的前 5 例。标签文件本身完整（形状与缓存一致），不需要重新推断。
重算时对全部 7 个方法强制重算，nnU-Net 恢复为 154 例。其他方法的评价都是在第 4 折完成后才开始的，原本就是 154 例。

## 与 P0 的已知差异（写 Methods 时需说明）

1. **训练预算不同。** P0 上限 500 epoch，SOTA 统一 300 epoch（2026-10-22 按用户要求改定）。SOTA 的训练量少于 P0，
   若 SOTA 不如 P0，审稿人可能质疑是否训练不足；需在正文中说明，或在附录补一组 500 epoch 的对照。
2. **选模型的依据不同。** P0 用中心 patch 的 Dice 选 best checkpoint，SOTA registry 方法用全体积 Dice_Y 选
   （`--select_by_sliding_window`）。选择依据更接近最终指标，对对照方法略为有利。
3. **nnU-Net v2 自带预处理。** nnU-Net 会按自己的 plans 重采样（目标间距由数据集中位数决定）并自选 patch 与 batch。
   输入仍是同一份缓存 CT，但 nnU-Net 的预处理和 patch 不同于 P0。
4. **nnU-Net 不做 CT 归一化。** 缓存本身已是百分位归一化，通道名设为 `nonorm`，避免二次归一化。
5. **nnU-Net 的训练器是本仓库补的。** 2.5.2 没有 300 epoch 的官方变体，`tools/nnUNetTrainer_300epochs.py` 是一个
   只改 `num_epochs` 的子类，安装到 venv 的 `variants/training_length/`。nnU-Net 自带的 5 折验证结果
   （`validation/summary.json`）只作参考，正式结果以本仓库的评价为准。
6. **nnFormer 为简化实现**（见上）。

## 运行

```bash
cd ~/cnv/code/topocnv_remote
source ~/miniconda3/etc/profile.d/conda.sh && conda activate topocnv && source env_cnv.sh
# nnU-Net 数据准备（CPU，一次性）
~/cnv/envs/nnunetv2_cnv/bin/python tools/sota_nnunet.py prep
# 队列（尽量多用空闲卡，每卡最多 1 个我们的 run；数据准备完成前不启动 nnU-Net）
nohup python tools/sota_queue.py >> log/sota_binary_y_20261021/queue.log 2>&1 &
touch log/sota_binary_y_20261021/STOP   # 停止启动新 run（在跑的继续）
# 汇总（CPU）
python tools/sota_summary.py
```

GPU 规则（2026-10-22 用户指示）：P0 优先。尽量多用卡，但每张卡最多只跑一个我们的 run。只在空闲显存 ≥40 GB 的卡上启动。
若 P0 在该卡上还有待启动的 run，要求启动后仍留有 40 GB。队列不向任何进程发信号。

## 运行状态（2026-10-23）

7 个方法 × 5 折 = 35 个 run 全部完成（`COMPLETED` 标记齐全），队列 07:38 正常退出，没有失败的 run。
前一版汇总的 nnU-Net 缺 5 例已修正（见上文「评价缺例」）。

| 方法 | 完成折数 | 备注 |
|---|---|---|
| nnU-Net v2（3d_fullres，300 epoch） | 5/5 | 154 例 |
| DynUNet | 5/5 | 300 epoch，报告折选模型（按现有指示保留） |
| SegResNet | 5/5 | 同上 |
| Attention U-Net | 5/5 | 同上 |
| UNETR | 5/5 | 未收敛，欠训练的弱基线（见上文） |
| SwinUNETR | 5/5 | 同上 |
| nnFormer | 5/5 | 简化实现，同上 |

## 最终表

`results/sota_binary_y_eval/SOTA_main_table.md`（由 `tools/sota_summary.py` 生成，已于 2026-10-23 用 v0.4 指标强制重算）。
Dice_Y（mean ± sd）与 HD95_Y（mm）：

| 方法 | Dice_Y | HD95_Y | p Dice_Y vs P0 Full |
|---|---|---|---|
| P0 Full | 0.6049 ± 0.1262 | 51.27 ± 98.09 | — |
| P0 B0 | 0.6030 ± 0.1186 | 10.12 ± 7.06 | — |
| nnU-Net v2（300 epoch） | 0.6579 ± 0.1249 | 4.99 ± 7.30 | 7.8e-24 |
| DynUNet | 0.6355 ± 0.1276 | 9.50 ± 7.39 | 2.1e-20 |
| SegResNet | 0.5780 ± 0.1058 | 16.57 ± 6.81 | 2.0e-06 |
| Attention U-Net | 0.5706 ± 0.1209 | 28.27 ± 36.85 | 2.1e-08 |
| UNETR（未收敛） | 0.2978 ± 0.0633 | 114.89 ± 62.14 | 5.1e-27 |
| SwinUNETR | 0.5725 ± 0.1065 | 22.69 ± 18.16 | 1.9e-10 |
| nnFormer | 0.5749 ± 0.1041 | 27.48 ± 16.26 | 2.6e-09 |

只有 5/5 折、154 例的行才是最终结果。上表的 p 值是与 P0 Full 的配对 Wilcoxon 检验，只用于参考，正式比较以 v0.4 的 bootstrap 和 Holm 校正为准。
