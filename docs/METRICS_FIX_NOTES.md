# 评价指标修正记录（L1 打分 v2 与 Secondary 诊断）

2026-10-22，cnv-metrics 会话。涉及文件：`MIA/scripts/metrics_protocol.py`、`MIA/scripts/eval_protocol_metrics.py`、
`tools/p0_summary.py`、`tools/secondary_options.py`（新）。结果目录：`results/p0_binary_y_eval_v2/`。
用户已于 2026-10-22 确认采用下面的 Dice_L 新定义。

## 1. Dice_L 的问题

旧代码（v1）：`Dice_L = Dice(P ∩ L, L)`，P 为预测的 Y。因为 P ∩ L ⊆ L，这个式子里不可能出现假阳性，
所以恒有 `Dice_L = 2·Recall_L / (1 + Recall_L)`，它只是 Recall_L 的单调变换。在 4 个臂 × 154 例的真实数据上核对过，
两者最大差 1.1e-16。旧主表里 Dice_L（约 0.78–0.82）高于 Dice_Y（约 0.59–0.61），原因就在这里。
低证据区内以及 H 之外软组织里的假阳性，旧 Dice_L 完全不计。

## 2. 新定义（v2）

L 的定义不动：`H = {HU ∈ [200, 3000]} ∪ dilate(参考标注 class4, 3 mm)`，`L = Y \ H`。
H 只由参考标注和原始 CT 决定，与预测无关，所以各方法用的是同一个 H。只改打分方式：预测和参考都去掉 H。

| 列名 | 公式 | 说明 |
|---|---|---|
| `Dice_L` | 2\|P ∩ L\| / (\|P \ H\| + \|L\|) = Dice(P \ H, Y \ H) | 主报；H 外的假阳性计入 |
| `Precision_L` | \|P ∩ L\| / \|P \ H\| | 新增 |
| `Recall_L` | \|P ∩ L\| / \|L\| | 不变 |
| `L_over_Y` | \|L\| / \|Y\| | 不变 |
| `Dice_L_legacy` | Dice(P ∩ L, L) = 2R/(1+R) | v1 的 Dice_L，只作核对，不进正文 |
| `Dice_H` | Dice(P ∩ H, Y ∩ H) | 高证据区对照，只在 CSV 里，主表不列 |

H 内的预测体素不进入 Dice_L；它们只体现在全局 Dice_Y 和 Dice_H 里。

**考虑过、没有采用的方案：限制在走廊 ROI 内（例如 dilate(Y, 10 mm) \ H）。** 好处是远处孤立假阳性不会拉低 Dice_L，
指标更“局部”。没采用的原因：多一个自由参数（ROI 半径）；ROI 由参考标注决定，等于把定位信息交给了评价；
另外 Dice_Y 本身也计入全体积假阳性，v2 的 Dice_L 与它口径一致。如果审稿人要求，可以作为敏感性分析补算。

自测：`python MIA/scripts/eval_protocol_metrics.py --self-test` 覆盖 6 种情形：完美预测；H 外软组织假阳性（新 Dice_L 和
Precision_L 下降，legacy 不变）；H 内假阳性（Dice_L 不变，Dice_H 和 Dice_Y 下降）；漏掉一半 L 且无假阳性（新旧相等）；
空预测（Dice_L = 0，Precision_L = NaN）；无 HU（全部 NaN）。

### Methods 可直接使用的英文描述

> **Low-evidence subset (L1).** For each case, a high-evidence region H was defined as the union of voxels with raw
> CT attenuation between 200 and 3000 HU and the reference foramen/canal label (class 4) dilated by 3 mm with an
> ellipsoidal structuring element in physical units. The low-evidence part of the reference was L = Y \ H. H depends
> only on the reference label and the CT, so it is identical for every method compared. Both the prediction P and
> the reference were restricted to the complement of H before scoring: Dice_L = 2|P ∩ L| / (|P \ H| + |L|),
> precision_L = |P ∩ L| / |P \ H| and recall_L = |P ∩ L| / |L|. Predicted voxels outside both H and Y therefore count
> as false positives in Dice_L and precision_L; predicted voxels inside H are scored only by the global Dice_Y.
> We also report |L|/|Y|, the share of the reference in the low-evidence region.

## 3. P0 新旧数值对比（v2 目录，6 臂 × 5 折，各 154 例 OOF）

| 臂 | n | Dice_Y | Dice_L 旧 (=Dice_L_legacy) | Dice_L 新 | Precision_L | Recall_L | Dice_H | HD95_Y 中位数 [IQR] | HD95_Y>50 mm 例数 |
|---|---|---|---|---|---|---|---|---|---|
| Full | 154 | 0.605 ± 0.126 | 0.815 ± 0.054 | 0.600 ± 0.138 | 0.552 ± 0.177 | 0.691 ± 0.076 | 0.614 ± 0.109 | 5.1 [3.6–10.0] | 29 |
| A_mem | 154 | 0.606 ± 0.125 | 0.818 ± 0.049 | 0.600 ± 0.136 | 0.548 ± 0.174 | 0.695 ± 0.069 | 0.617 ± 0.111 | 6.9 [5.0–13.9] | 37 |
| A_gate | 154 | 0.584 ± 0.122 | 0.810 ± 0.051 | 0.577 ± 0.133 | 0.519 ± 0.168 | 0.683 ± 0.070 | 0.596 ± 0.107 | 7.6 [5.1–219.7] | 58 |
| B0 | 154 | 0.603 ± 0.119 | 0.805 ± 0.047 | 0.595 ± 0.129 | 0.552 ± 0.170 | 0.676 ± 0.065 | 0.616 ± 0.105 | 10.1 [6.6–12.0] | 2 |
| C1 | 154 | 0.609 ± 0.119 | 0.787 ± 0.048 | 0.598 ± 0.129 | 0.575 ± 0.179 | 0.652 ± 0.064 | 0.626 ± 0.106 | 8.5 [6.1–12.8] | 20 |
| A2 | 154 | 0.595 ± 0.117 | 0.772 ± 0.058 | 0.587 ± 0.127 | 0.572 ± 0.177 | 0.632 ± 0.075 | 0.609 ± 0.104 | 10.9 [8.9–13.7] | 2 |

与 Full 的配对 Wilcoxon（n=154）：Dice_L 新定义下，A_mem p=0.62、B0 p=0.105、C1 p=0.393，与 Full 无显著差异；
A_gate p=4.5e-13、A2 p=4.2e-05（均低于 Full）。Dice_Y 的 p 值依次为 0.594、0.503、0.0791、3.4e-13、3.1e-4。

**对主张的影响（只陈述数字，结论由主会话定）**：旧 Dice_L 下 Full（0.815）明显高于 C1（0.787）和 A2（0.772），
这个差异全部来自 Recall_L。新定义计入假阳性后，Full 的 Precision_L（0.552）低于 C1（0.575）和 A2（0.572），
Dice_L 变成 0.600 vs 0.598 vs 0.587，Full 与 C1、B0 之间的差异不再显著。也就是说，“记忆在低证据区更好”现在只能落在
Recall_L 上（Full 0.691 vs C1 0.652），代价是那里的假阳性更多。

Dice_L 旧 = v1 主表里的 Dice_L。均值 ± 标准差，按病例计。HD95_Y 有明显的远处假阳性离群值，正文建议报中位数 [IQR]；
HD95_Y > 50 mm 的例数列在最后一列。配对 Wilcoxon 和主表全文见 `results/p0_binary_y_eval_v2/P0_main_table.md`。

## 4. Secondary（Dice_123、Dice_c4）

### 诊断

Secondary 原来直接取 Stage-1 的全体积 argmax。Stage-1 OOF 全体积预测（154 例，`secondary/s1_diagnosis.csv`）：

| 指标 | 均值 ± sd | 中位数 |
|---|---|---|
| Stage-1 Y 体积 / 真值 Y 体积 | 7.20 ± 5.85 | 5.96 |
| Stage-1 Y 体素中距真值 Y > 30 mm 的比例 | 0.782 ± 0.100 | 0.803 |
| Stage-1 Y 连通域个数 | 776 ± 485 | 683 |
| Stage-1 Y Dice，全体积 | 0.200 ± 0.088 | 0.186 |
| Stage-1 Y Dice，真值 Y 外扩 10 mm 的走廊内 | 0.591 ± 0.109 | 0.606 |
| Stage-1 Dice_123，全体积 / 走廊内 | 0.196 / 0.574 | |
| Stage-1 Dice_c4，全体积 / 走廊内 | 0.237 / 0.578 | |

结论：神经附近 Stage-1 的表现与它的 patch 验证 Dice（约 0.58）一致，问题出在远处。约 78% 的 Stage-1 Y 体素离真值超过
30 mm，分散在几百个小连通域里。原因是 Stage-1 的训练设置：README 里的 Stage-1 命令没有传 `--random_crop_prob`，默认值是
0，所以训练 patch 全部以前景为中心，模型没见过不含神经的区域。验证也是以前景为中心的 patch（`validate_stage1_patch`
沿用了同一个参数），所以这个问题在训练日志里看不出来。滑窗推理本身没有问题，方向问题已在 runbook 里排除。

另外，旧取法下所有级联臂的 Secondary 完全相同（都来自同一个 Stage-1），不能用来比较各臂。

Stage-2 训练用了 `--random_crop_prob 0.33`，见过背景 patch，所以最终 Y 没有受这个问题拖累。但 Stage-2 的先验
[P(c4), dilate(hard c4)] 里含有这些远处假阳性，这一点 Methods 或局限里可能需要交代。是否重训 Stage-1 由主会话决定。

### 方案（`tools/secondary_options.py`，结果在 `results/p0_binary_y_eval_v2/secondary/`）

- **S0_full（现状）**：Stage-1 全体积 argmax。
- **A_inY**：在 Stage-2 预测的 Y 内部，按 Stage-1 后验在类别 1–4 中取 argmax；Y 外为背景。1–4 的并集按构造恰好等于主预测 Y
  （154/154 例核对通过）。
- **B_roiY**：Stage-1 argmax，只保留 dilate(Stage-2 Y, 10 mm) 内的部分。
- **C_gtROI**：Stage-1 argmax，只保留 dilate(真值 Y, 10 mm) 内的部分。用到了真值，只能当诊断。

各级联臂 154 例均值（Dice_123 / Dice_c4；B0 无 Stage-1，仍为 N/A）：

| 臂 | S0_full | A_inY | B_roiY | C_gtROI |
|---|---|---|---|---|
| Full | 0.196 / 0.237 | 0.587 / 0.564 | 0.569 / 0.541 | 0.574 / 0.578 |
| A_mem | 0.196 / 0.237 | 0.588 / 0.561 | 0.569 / 0.520 | 0.574 / 0.578 |
| A_gate | 0.196 / 0.237 | 0.570 / 0.539 | 0.567 / 0.533 | 0.574 / 0.578 |
| C1 | 0.196 / 0.237 | 0.585 / 0.583 | 0.568 / 0.546 | 0.574 / 0.578 |
| A2 | 0.196 / 0.237 | 0.569 / 0.579 | 0.571 / 0.572 | 0.574 / 0.578 |

| 方案 | 优点 | 缺点 |
|---|---|---|
| S0_full | 不改；定义最简单 | 主要反映 Stage-1 远处假阳性；各臂相同，无法比较 |
| A_inY（建议） | 不用真值；与主预测 Y 一致（Secondary 是对主结果的分解）；各臂不同，可比较；符合临床用法（最终交付的是一个多类勾画） | Dice_123/c4 同时受 Stage-2 的 Y 边界和 Stage-1 的类别划分影响，要在 Methods 里写清 |
| B_roiY | 不用真值；保留 Stage-1 自己的边界 | 多一个参数（10 mm）；多类并集与主预测 Y 不一致；仍留有 Y 附近的 Stage-1 假阳性 |
| C_gtROI | 最直接地反映 Stage-1 本身的分类能力 | 用了真值定位，不能作为方法结果报告；各臂相同 |

我的建议是 A_inY 作为 Secondary 主报，C_gtROI 作为“Stage-1 在神经附近的分类能力”放补充材料。这一点等主会话和用户拍板，
目前主表里的 Dice_123 / Dice_c4 仍是 S0_full。

## 5. 目录和运行说明

- `results/p0_binary_y_eval_v2/`：v2 评分，主表 `P0_main_table.md`（含与 Full 的配对 Wilcoxon：Dice_Y 和 Dice_L，以及全部臂共有病例的子表）。
  `tools/p0_summary.py` 现在默认写这里，P0 队列刷新主表时也写这里。
- `results/p0_binary_y_eval/`：v1 的冻结快照（部分折，Dice_L 为旧定义），不再更新。2026-10-21 晚和 10-22 上午有两次进程
  在部署新代码后往这个目录写过 v2 列，已按 `Dice_L := Dice_L_legacy` 还原为 v1 口径。下游统计（`tools/p0_stats.py`）应改读 v2 目录。
- `p0_summary.py` 的改动：新列；评分失败不再中断（保留旧 CSV 并打印错误）；CSV 和主表用临时文件 + 原子替换写出，
  多个进程同时运行也不会读到半个文件；CSV 缺新列时自动重算。
- SOTA 对照网络用的是同一个 `eval_protocol_metrics.py`，重新评分即得到 v2 的 Dice_L；之前按旧代码算出的 Dice_L 都不能用。

## 6. 顺带发现、未处理

- HD95_Y 离群：Full 29 例、A_mem 37 例、A_gate 58 例、C1 20 例的 HD95_Y > 50 mm（远处孤立假阳性连通域），B0 和 A2 各 2 例。这是模型行为，不是评分错误。
  正文 HD95 应报中位数 [IQR]，并考虑是否报告“去掉远处连通域”的后处理结果（需要另定规则，不能事后挑）。
- HD95 在预测为空时返回 NaN（目前 P0 中没有出现），汇总规则需要在 Methods 里写明。
- 类别 1–3 是可见度分层（1 = CT 可见神经，2 = MRI 引导神经，3 = 可能走行），代码注释和 README 里的 “V1/V2/V3” 是旧说法，本次没有改（不在本会话的文件范围内）。
