# 机制分析：证据门控记忆是否在低证据区起作用

只做推理和分析，没有训练新模型。

**数据范围（本版）**：第 1 节用全部 154 例。第 2–3 节目前只有 **fold 0**：Full 31 例、A_mem 31 例、A_gate 30 例（A_gate 第 31 例在写本版时还在跑）。fold 1、fold 2 的推理在后台按顺序进行（`results/analysis_mech/run.log`），跑完后用同一套脚本刷新表格和图，数字以刷新后的 `results/analysis_mech/summary.md` 为准。

代码在 `tools/analysis/`：`mech_common.py`（公共加载）、`gate_stats.py`（第 1 节）、`mech_infer.py`（第 2–3 节推理）、`mech_summary.py`（汇总表）、`mech_figs.py`（出图）、`check_oof.py`（用 trainer 自带滑窗复现 OOF 标签）。
结果在 `results/analysis_gate/`、`results/analysis_mech/`（每个病例×臂一个 JSON；`summary.md` 是汇总表；`maps/` 存每折前 3 例的 ROI 体数据），图在 `results/analysis_figs/`。
训练脚本、模型文件和 `MIA/scripts/` 指标代码只 import、没改。记录 memory 支路时，是在模型实例上替换 `_gate_fuse`，类本身没动。

**正确性核对**：fold 0 的 92 个病例×臂，本脚本 orig 条件的 Dice_Y 和 HD95_Y 与 P0 主表的逐例 CSV（`results/p0_binary_y_eval/<Arm>.csv`）完全一致，差值为 0。

---

## 0. 关键结论

1. **g 基本实现了协议的 H/L 划分，但骨窗阈值不是 200 HU。** 在神经外扩 20 mm 的范围内，{g ≥ 0.5} 与 H 的 Dice 为 0.89；L 中 92% 的体素 g < 0.5。归一化 CT 阈值 0.97 换算回 HU 的中位数是 125 HU（IQR 94–163），87% 的病例低于 200 HU。有 9/154 例的阈值落在软组织范围（−8 到 52 HU），这些病例的 L 有超过 30% 被判成高证据。
2. **Full 推理时确实在用 memory，而且 L 比 Y∩H 用得多。** 推理时去掉 memory 支路，Full 的 Dice_Y 降 0.017、Recall_L 降 0.013、Recall_YH 降 0.007（31 例，p 均 < 0.001）。第 0 个 memory 层上，融合特征被 memory 改动的幅度在 L 是 Y∩H 的 4.4 倍（31/31 例 L 更大）。
3. **但「低证据区更依赖 memory」不是门控独有的。** A_gate 的门控恒为 0.5，同样指标在 L 也是 Y∩H 的 3.0 倍，原因是 memory 支路本身在软组织处提出的修改就更大（门控前修改量：A_gate 0.72 vs 0.24，Full 0.80 vs 0.43）。到输出层，去掉 memory 后概率变化 |Δp| 的 L/Y∩H 比值，Full 为 1.50，A_gate 为 1.51，几乎相同。检索到 CN V 原型的权重在 L 和 Y∩H 也一样（0.78 vs 0.79）。较深的两层（1/8、1/16 分辨率）g 被池化得很粗，L 与 Y∩H 的门控差别很小（第 2 层 0.28 vs 0.36）。
4. **L2 抹证实验不支持「证据缺失时 memory 能补回来」。** 抹掉神经附近高证据区的 CT 后，三个臂都明显变差，Full 没有更稳：
   - 只抹 CT（E1）：Full 的 Dice_Y 下降 0.105，比 A_mem（0.118）少，但比 A_gate（0.099）多；被抹区域内神经的召回下降，Full 0.154 与 A_mem 0.157 相同（p = 0.32），A_gate 0.120 最少。
   - CT 和先验一起抹（E2）：Full 在被抹区域内的门控从 0.75 降到 0.14，memory 占比从 0.26 升到 0.87，路由确实切到了 memory；但这部分神经的召回从 0.73 掉到 0.06。Full 的 Dice_Y 下降（0.394）与 A_mem（0.398）没有差别（p = 0.79），被抹区域召回的下降反而比 A_mem 大（0.671 vs 0.646，p < 0.001）。
   - 在 Full 内部比较「抹除后有/无 memory」：E1 下有 memory 时下降更多（被抹区域召回多降 0.024，p < 0.001），E2 下差别不显著。
5. **memory-only 读出很弱**：Full 的 Ŷ_mem 单独做分割，Dice_Y 0.31、Recall_L 0.47。这和第 4 条一致：memory 检索用的是局部 token 做查询，局部证据被抹掉后，查询本身也坏了，检索不到可替代的解剖信息。

综合：机制的「路由」部分成立，g 按设计把低证据区交给 memory，Full 也确实依赖这部分输出；「低证据区更依赖原型」在特征层可以展示，但同样的空间偏好在固定门控的 A_gate 里也有，到输出层两者没有区别；「局部证据缺失时靠解剖记忆补全」这一层，L2 实验给出的是阴性结果。写论文时建议把机制表述限制在「门控把软组织段的特征来源切换到 memory，Full 在推理时依赖这一支路」，不要写「memory 能在证据缺失时恢复神经」。以上是 fold 0 的结果，等 fold 1–2 跑完再确认。

---

## 1. 证据图 g 与协议 H 的一致性（154 例，全部病例）

### 方法

- g 按训练时的定义在整个体积上计算：`g = avgpool3(max(σ((CT_norm − 0.97)/0.01), P(c4), dilate(ĉ4, 3 mm)))`，先验用 Stage-1 的 OOF 预测，和 Stage-2 推理时一致。
- H 按协议：`H = (原始 HU ∈ [200, 3000]) ∪ dilate(GT class4, 3 mm)`，HU 从原始 NIfTI 重采样到 cache 网格（`metrics_protocol.load_raw_hu_dhw`）；`L = Y \ H`。
- 归一化 CT 与 HU 的对应：cache 是每例重采样 CT 的 1–99 百分位仿射映射再截断到 [0,1]，所以在未截断体素上拟合 `HU = a·CT_norm + b`，得到每例 0.97 对应的 HU。
- 背景分两层：Y 外 0–5 mm（ring5）、5–20 mm（bg20）。体素一致性在 Y 外扩 20 mm 的 ROI 内算。

### 结果

| 指标 | 中位数 [IQR]（154 例） | 范围 |
|---|---|---|
| 归一化 CT = 0.97 对应的 HU | 125 [94, 163] | −8 – 541 |
| 归一化 CT = 1.0 对应的 HU（即每例 p99） | 223 [190, 262] | 85 – 589 |
| \|L\|/\|Y\| | 0.597 [0.576, 0.621] | 0.472 – 0.759 |
| Y∩H 中 g ≥ 0.5 的比例 | 0.923 [0.893, 0.940] | 0.644 – 0.982 |
| L 中 g ≥ 0.5 的比例（g 判为高证据，协议判为低证据） | 0.084 [0.047, 0.130] | 0.006 – 0.545 |
| ROI 内 {g ≥ 0.5} 与 H 的 Dice | 0.890 [0.849, 0.922] | 0.444 – 0.948 |
| ROI 内 {CT_norm > 0.97} 覆盖 {HU ≥ 200} 的比例 | 1.000 [1.000, 1.000] | 0.655 – 1.000 |
| ROI 内 {CT_norm > 0.97} 中 HU < 200 的比例 | 0.126 [0.056, 0.221] | 0 – 0.735 |
| ROI 内 预测 c4 扩张 与 GT c4 扩张 的 Dice | 0.681 [0.652, 0.708] | 0.542 – 0.774 |

g 的区域分布（每例均值的中位数）：Y∩H 0.856，L 0.194，ring5 0.428，bg20 0.230。L 中 g < 0.5 的体素占 91.6%（中位数），Y∩H 中只占 7.7%。

### 不一致之处及影响

1. **骨窗阈值不是 200 HU。** 0.97 在 134/154 例（87%）中对应低于 200 HU 的值，中位数 125 HU；49 例低于 100 HU，9 例低于 50 HU（已进入软组织/肌肉的 CT 值）。原因是 cache 按每例 p99 归一化，而多数病例 p99 只有 190–260 HU，0.97 落在 p99 以下约 100 HU 处。README 里「200 HU 对应归一化值中位数 0.993」这句话本身没错，但它说的是 200 HU 映到哪里，不是 0.97 映回哪里。
2. **影响集中在少数病例。** 每例阈值 HU 与「L 中被 g 判成高证据的比例」的 Spearman 相关为 −0.91（p < 10⁻⁵⁰）。L 中超过 30% 被判为高证据的 9 例，阈值都在 −8 到 52 HU 之间。这些病例里，软组织段神经本应走 memory，实际上大部分被门控交给了局部特征。对大多数病例，两种定义在神经附近基本一致：{CT_norm > 0.97} 包含了全部 HU ≥ 200 的体素，多出来的 12.6% 主要是骨边缘的部分容积体素。
3. **孔道项用的是预测 c4，H 用的是 GT c4。** 两者扩张后的 Dice 只有 0.68，预测 c4 只覆盖 GT c4 扩张区的 62%。但 Y∩H 中 g 仍有 92% ≥ 0.5，因为骨窗项和平滑补上了大部分。L 中由预测 c4 造成的高证据只占 1.8%（中位数），骨窗项占 6.0%。
4. **平滑和 sigmoid 软边界。** 3×3×3 平均和温度 0.01（约等于 30 HU）让 g 在骨-软组织交界处连续过渡，ring5 的 g 均值 0.43，正好落在两者之间。
5. **memory 层上的有效门控更粗。** g 在每个 memory 层被平均池化到 token 网格（输入的 1/4、1/8、1/16）。fold 0 的 Full 中，L 与 Y∩H 的有效门控在第 0 层为 0.25 vs 0.75，第 1 层 0.28 vs 0.57，第 2 层 0.28 vs 0.36。第 2 层一个 token 在 z 方向覆盖约 40 mm，高低证据基本被平均掉了。

建议 Methods 写清：g 的骨窗阈值按每例归一化 CT 定义，实际约 125 HU（IQR 94–163），不是 200 HU；g 的孔道项用 Stage-1 预测（推理时拿不到 GT）；约 6%（9/154）的病例因 p99 偏低，阈值落进软组织范围。

图：`results/analysis_figs/fig_gate_vs_H.png`（左：四个区域的 g 直方图；中：Y∩H 与 L 中 g ≥ 0.5 的比例；右：0.97 对应 HU 的分布，虚线为 200 HU）。

---

## 2. 记忆依赖度（Full vs A_gate）

### 方法

对每个臂、每个已完成折的验证病例，用和训练脚本相同的滑窗（patch 40×192×192，步长 0.5，高斯加权，fp32）推理。

在三个 memory 层（编码器第 2/3/4 层，token 网格为输入的 1/4、1/8、1/16，下称第 0/1/2 层）上，对每个 token 记录：

- `gate`：该层实际使用的门控值（Full 为 g 池化到 token 网格；A_gate 恒为 0.5）
- `share = ‖(1−g)·f_mem‖ / (‖g·f_loc‖ + ‖(1−g)·f_mem‖)`：任务里建议的 memory 占比
- `shift = ‖f − f_loc‖ / ‖f_loc‖`：融合特征相对纯局部特征的变化，等于 `(1−g)·‖f_mem − f_loc‖/‖f_loc‖`
- `dmem = ‖f_mem − f_loc‖ / ‖f_loc‖`：memory 支路本身提出的修改量（门控前）
- `pfg`：检索权重落在 CN V 原型块上的总质量（记忆库 2 个块 × 32 个原型，块 0 为背景）

token 值按最近邻上采样回体素，按滑窗权重平均，再在 Y∩H、L、ring5、bg20 中求均值。

输出层做反事实：推理时去掉 memory 支路（每层 f = f_loc，等价于 g ≡ 1），比较概率变化 |Δp|，以及 Dice_Y、Recall_L、Recall_YH 的变化。

两点说明：

- `share` 在 Full 里基本由 g 决定（memory 支路是残差式交叉注意力，f_mem 与 f_loc 范数接近），低证据区 share 高是构造使然，不能单独当证据。能回答「低证据区是否更依赖原型」的是 `shift`、`dmem` 和输出反事实 |Δp|。
- 对 A_gate，「去掉 memory」把训练时固定的 0.5 混合改成纯局部特征，属于分布外操作。两臂之间比较 L 与 Y∩H 的**相对**差异，不比较绝对值。

### 结果（fold 0；Full n = 31，A_gate n = 30）

**推理时去掉 memory 支路的影响**（orig − nomem，逐例配对，Wilcoxon）：

| 臂 | n | ΔDice_Y | ΔRecall_L | ΔRecall_YH | ΔDice_LnotH |
|---|---|---|---|---|---|
| Full | 31 | +0.017 ± 0.012（p = 2e-8） | +0.013 ± 0.015（p = 4e-5） | +0.007 ± 0.010（p = 2e-4） | +0.022 ± 0.015（p = 7e-8） |
| A_gate | 30 | −0.005 ± 0.013（p = 0.07） | +0.007 ± 0.008（p = 2e-4） | +0.009 ± 0.008（p = 5e-7） | −0.011 ± 0.020（p = 0.015） |

正值表示有 memory 更好。`Dice_LnotH = Dice(pred \ H, L)`，会计入低证据区外的假阳性；协议的 Dice_L 恒等于 2R/(1+R)，信息和 Recall_L 重复。

**按区域的 memory 依赖度**（各例区域均值再取均值）：

| 区域 | 臂 | gate 0/1/2 | shift 0/1/2 | dmem 0/1/2 | pfg 0 | \|Δp\| | 翻转率 |
|---|---|---|---|---|---|---|---|
| Y∩H | Full | 0.75 / 0.57 / 0.36 | 0.14 / 0.13 / 0.54 | 0.43 / 0.30 / 0.84 | 0.79 | 0.046 | 0.047 |
| L | Full | 0.25 / 0.28 / 0.28 | 0.61 / 0.22 / 0.58 | 0.80 / 0.30 / 0.81 | 0.78 | 0.068 | 0.068 |
| 0–5 mm | Full | 0.40 / 0.36 / 0.30 | 0.38 / 0.21 / 0.59 | 0.63 / 0.33 / 0.84 | 0.75 | 0.024 | 0.024 |
| 5–20 mm | Full | 0.23 / 0.24 / 0.23 | 0.45 / 0.27 / 0.68 | 0.63 / 0.36 / 0.87 | 0.60 | 0.0003 | 0.0002 |
| Y∩H | A_gate | 0.5（恒定） | 0.12 / 0.16 / 0.17 | 0.24 / 0.31 / 0.35 | 0.78 | 0.031 | 0.031 |
| L | A_gate | 0.5 | 0.36 / 0.20 / 0.16 | 0.72 / 0.39 / 0.32 | 0.76 | 0.049 | 0.049 |
| 0–5 mm | A_gate | 0.5 | 0.27 / 0.21 / 0.16 | 0.53 / 0.41 / 0.33 | 0.73 | 0.018 | 0.018 |
| 5–20 mm | A_gate | 0.5 | 0.32 / 0.28 / 0.18 | 0.63 / 0.56 / 0.37 | 0.57 | 0.0002 | 0.0001 |

L 对 Y∩H 的逐例比较：

| 指标 | 臂 | L/Y∩H 比值（中位数） | L > Y∩H 的病例 | Wilcoxon p |
|---|---|---|---|---|
| shift（第 0 层） | Full | 4.41 | 31/31 | 9e-10 |
| shift（第 0 层） | A_gate | 3.02 | 30/30 | 2e-9 |
| \|Δp\|（去掉 memory） | Full | 1.50 | 30/31 | 5e-9 |
| \|Δp\|（去掉 memory） | A_gate | 1.51 | 29/30 | 4e-9 |

解读：

- 第 0 层上，Full 在 L 的特征改动是 Y∩H 的 4.4 倍。这里一半来自门控（L 的有效 gate 0.25，Y∩H 0.75），另一半来自 memory 支路本身：门控前的修改量 dmem 在 L 就比 Y∩H 大（0.80 vs 0.43）。A_gate 没有空间门控，比值也有 3.0。
- 第 1、2 层上，Full 的门控在 L 与 Y∩H 之间差别小，shift 也接近（第 2 层 0.58 vs 0.54）。
- 到输出层，去掉 memory 后 L 的概率变化比 Y∩H 大约一半，但 Full 和 A_gate 的比值一样（1.50 vs 1.51）。也就是说，输出对 memory 的依赖偏向软组织段，这个偏向在固定门控模型里同样存在，不能归功于证据门控。
- 检索到 CN V 原型的权重 pfg 在 L 与 Y∩H 相同（0.78 vs 0.79），只在远离神经的背景处降低（0.60）。检索本身不区分高低证据。
- Full 的 Ŷ_mem（memory-only 读出）Dice_Y 0.31 ± 0.08，Recall_L 0.47 ± 0.07，单独使用远不如主输出（Dice_Y 0.62）。
- 顺带的观察：fold 0 中 Full 的 HD95_Y 中位数 4.1 mm，A_gate 208 mm，A_mem 209 mm（后两者有大量远处假阳性连通域）；Full 推理时去掉 memory 后 HD95_Y 仍是 4.2 mm。Full 抑制远处假阳性的能力来自训练出的权重，不依赖推理时的 memory 支路。

图：`results/analysis_figs/fig_memory_reliance.png`（三层 shift 与 |Δp| 按区域，Full vs A_gate）；`fig_case_<病例号>.png`（Full 的典型病例：CT、g、Y∩H/L、预测、第 0 层 memory shift、CN V 原型检索权重、Δp）。

---

## 3. L2 抹证实验（附录）

### 设计（运行前确定，写在 `mech_infer.py` 里）

- **抹除区域** `E = H ∩ {到 Y 的距离 ≤ 10 mm}`：只抹神经附近的高证据区（骨窗高对比和孔道邻域），远处颅骨不动，避免整颅去骨造成大幅分布偏移。H 用协议定义（含 GT class4 扩张）；GT 只用来决定在哪里扰动，不进入模型。因为 Y∩H 全部落在 Y 外 10 mm 以内，所以被抹掉的神经部分 Y∩E 就等于 Y∩H，下文「被抹区域召回」即 Recall_YH。
- **替换值**：E 内的归一化 CT 换成该例软组织强度加同分布噪声。软组织参照取 `{到 Y ≤ 10 mm} \ H` 中 HU ∈ [0, 100] 的体素，用其归一化 CT 的均值和标准差；随机种子按病例号固定。加噪声是为了不留下「完全平坦」的纹理线索。
- **两个强度**：
  - E1：只抹 CT，级联先验不动。Full 的骨窗项下降，孔道先验项（预测 c4）仍在。
  - E2：抹 CT，同时把 E 内两个先验通道置 0。g 在 E 内整体下降，Full 会把这些位置交给 memory。
- **对比**：Full、A_mem、A_gate 在同一批病例上的下降量（orig − 抹除后）：Dice_Y、被抹区域召回、Recall_L。有 memory 的两臂再做「抹除 × 去掉 memory」2×2：交互项 = 有 memory 时的下降 − 无 memory 时的下降，负值表示 memory 减轻了损失。
- **实现**：每个滑窗是独立前向，不与 E 相交的窗口抹除前后输出完全相同，所以 E1/E2 只重算与 E 相交的窗口（每例约 25%），其余复用参照条件的输出，结果与整卷重算一致。
- **预期**：如果主张成立，Full 在 E2 下被抹区域召回的下降应小于 A_mem，且 Full 的交互项为负。

### 结果（fold 0；Full 31 例，A_mem 31 例，A_gate 30 例）

**各臂的下降量**（orig − 抹除后，正值 = 变差；mean ± sd）：

| 臂 | 抹除 | 推理时 memory | Dice_Y 下降 | 被抹区域召回下降 | Recall_L 下降 |
|---|---|---|---|---|---|
| Full | E1 | 开 | 0.105 ± 0.045 | 0.154 ± 0.047 | 0.258 ± 0.048 |
| Full | E1 | 关 | 0.094 ± 0.039 | 0.130 ± 0.042 | 0.232 ± 0.047 |
| A_gate | E1 | 开 | 0.099 ± 0.035 | 0.120 ± 0.040 | 0.237 ± 0.043 |
| A_gate | E1 | 关 | 0.088 ± 0.032 | 0.107 ± 0.038 | 0.218 ± 0.039 |
| A_mem | E1 | （无 memory） | 0.118 ± 0.051 | 0.157 ± 0.045 | 0.270 ± 0.055 |
| Full | E2 | 开 | 0.394 ± 0.141 | 0.671 ± 0.085 | 0.504 ± 0.105 |
| Full | E2 | 关 | 0.386 ± 0.137 | 0.667 ± 0.081 | 0.500 ± 0.099 |
| A_gate | E2 | 开 | 0.366 ± 0.125 | 0.614 ± 0.075 | 0.470 ± 0.094 |
| A_gate | E2 | 关 | 0.374 ± 0.131 | 0.612 ± 0.070 | 0.473 ± 0.100 |
| A_mem | E2 | （无 memory） | 0.398 ± 0.148 | 0.646 ± 0.092 | 0.487 ± 0.102 |

E1 后的绝对值：Full Dice_Y 0.518、Recall_L 0.479；E2 后 Full Dice_Y 0.229、被抹区域召回 0.061（抹除前 0.732）。Recall_L 也明显下降，说明抹掉骨性线索后，软组织段的定位也跟着失准，L 并不是独立于 H 被识别的。

**臂间配对比较**（下降量，Wilcoxon）：

| 比较 | 抹除 | 指标 | n | 臂 1 | 臂 2 | p |
|---|---|---|---|---|---|---|
| Full vs A_mem | E1 | Dice_Y | 31 | 0.105 | 0.118 | 5e-8 |
| Full vs A_mem | E1 | 被抹区域召回 | 31 | 0.154 | 0.157 | 0.32 |
| Full vs A_mem | E1 | Recall_L | 31 | 0.258 | 0.270 | 0.007 |
| Full vs A_gate | E1 | Dice_Y | 30 | 0.108 | 0.099 | 0.001 |
| Full vs A_gate | E1 | 被抹区域召回 | 30 | 0.156 | 0.120 | 1e-6 |
| Full vs A_mem | E2 | Dice_Y | 31 | 0.394 | 0.398 | 0.79 |
| Full vs A_mem | E2 | 被抹区域召回 | 31 | 0.671 | 0.646 | 4e-4 |
| Full vs A_mem | E2 | Recall_L | 31 | 0.504 | 0.487 | 0.012 |
| Full vs A_gate | E2 | Dice_Y | 30 | 0.405 | 0.366 | 3e-7 |
| Full vs A_gate | E2 | 被抹区域召回 | 30 | 0.675 | 0.614 | 6e-8 |

**memory × 抹除交互**（有 memory 时的下降 − 无 memory 时的下降；负值 = memory 减轻损失）：

| 臂 | 抹除 | Dice_Y | 被抹区域召回 | Recall_L |
|---|---|---|---|---|
| Full | E1 | +0.011 ± 0.010（p = 2e-7） | +0.024 ± 0.012（p = 9e-10） | +0.026 ± 0.015（p = 3e-8） |
| Full | E2 | +0.008 ± 0.015（p = 0.008） | +0.004 ± 0.013（p = 0.14） | +0.005 ± 0.021（p = 0.26） |
| A_gate | E1 | +0.011 ± 0.006（p = 6e-9） | +0.013 ± 0.009（p = 5e-8） | +0.019 ± 0.011（p = 4e-9） |
| A_gate | E2 | −0.008 ± 0.015（p = 0.013） | +0.002 ± 0.020（p = 0.11） | −0.003 ± 0.021（p = 0.22） |

**被抹区域内的门控与 memory 依赖**（第 0 层，Y∩E 内均值）：

| 臂 | 条件 | gate | share | shift | pfg |
|---|---|---|---|---|---|
| Full | orig | 0.745 | 0.259 | 0.142 | 0.791 |
| Full | E1 | 0.515 | 0.492 | 0.274 | 0.774 |
| Full | E2 | 0.138 | 0.871 | 0.471 | 0.676 |
| A_gate | orig | 0.5 | 0.520 | 0.120 | 0.778 |
| A_gate | E2 | 0.5 | 0.580 | 0.340 | 0.665 |

解读：E2 下 Full 的门控按设计把被抹区域交给了 memory（gate 0.75 → 0.14，share 0.26 → 0.87），但被抹神经的召回几乎归零，而且下降比无 memory 的 A_mem 更多。交互项为正或不显著，没有一项显示 memory 减轻了抹除的损失。L2 结果不支持「局部证据缺失时由解剖记忆补全」。一个可能的原因是检索的查询来自局部 token：局部证据被抹掉后，查询本身也失真，CN V 原型的检索权重从 0.79 降到 0.68。

图：`results/analysis_figs/fig_L2_erasure.png`（三个臂在 E1/E2 下 Dice_Y、被抹区域召回、Recall_L 的下降）。

---

## 4. 使用的折与病例数

| 分析 | 臂 | 折 | 病例数 | checkpoint（best epoch） |
|---|---|---|---|---|
| 第 1 节 g vs H | — | 全部 | 154 | Stage-1 OOF 先验 |
| 第 2–3 节 | Full | 0 | 31 | 325 |
| 第 2–3 节 | A_mem | 0 | 31 | 275 |
| 第 2–3 节 | A_gate | 0 | 30（第 31 例写本版时未完成） | 230 |

fold 1–2 正在推理（Full、A_mem 的 fold 2 已训练完成；A_gate fold 2 只有训练完成后才会纳入，脚本按 `COMPLETED` 标记判断）。

## 5. 英文段落草稿

**Methods (Supplement) — mechanism analysis.**
We analysed the trained Stage-2 models at inference only. The evidence map g was compared with the protocol high-evidence region H = (HU ≥ 200) ∪ dilate(class 4, 3 mm) in all 154 cases. For each memory level we recorded the effective gate, the relative feature change introduced by memory, ‖f − f_loc‖/‖f_loc‖, and the retrieval weight assigned to CN V prototypes, and mapped these token-level quantities back to voxels. Reliance on memory was further measured counterfactually by removing the memory branch at inference (f = f_loc at every memory level). In an evidence-erasure experiment, normalized CT values inside E = H ∩ {distance to Y ≤ 10 mm} were replaced with the patient's soft-tissue intensity plus matched noise, either alone (E1) or together with the cascade prior (E2). Out-of-fold validation cases of fold 0 were used (Full, n = 31; A_mem, n = 31; A_gate, n = 30); paired differences were tested with the Wilcoxon signed-rank test.

**Results (Supplement) — does the gate route low-evidence voxels to memory?**
The evidence map reproduced the protocol split closely: within 20 mm of the nerve, the Dice overlap between {g ≥ 0.5} and H was 0.89 (median over 154 cases), and 92% of low-evidence voxels (L = Y \ H) had g < 0.5. Because the bone term of g is defined on per-patient normalized CT, its threshold corresponded to a median of 125 HU (IQR 94–163) rather than 200 HU; in 9 of 154 patients the threshold fell below 55 HU and more than 30% of L was classified as high evidence. In the Full model, removing the memory branch at inference lowered Dice_Y by 0.017 ± 0.012 and Recall_L by 0.013 ± 0.015 (both P < 0.001). At the finest memory level, memory changed the fused features 4.4 times more in L than in Y ∩ H (31/31 patients). However, the model with a constant gate (A_gate) showed a similar spatial preference (3.0-fold), and at the output level the change caused by removing memory was 1.5 times larger in L than in Y ∩ H for both Full and A_gate. Retrieval weights on CN V prototypes did not differ between L and Y ∩ H (0.78 vs 0.79).

**Results (Supplement) — evidence erasure.**
Erasing high-evidence CT near the nerve degraded all models, and the Full model was not more robust. With CT erased (E1), the drop in Dice_Y was 0.105 for Full, 0.118 for A_mem and 0.099 for A_gate; the recall of the erased nerve segment fell by 0.154, 0.157 (P = 0.32 vs Full) and 0.120, respectively. When the cascade prior was erased as well (E2), the gate of the Full model switched the erased region to memory (mean gate 0.75 → 0.14), yet recall of the erased segment fell from 0.73 to 0.06, a larger drop than in A_mem (0.671 vs 0.646, P < 0.001). Within the Full model, keeping the memory branch did not reduce the damage caused by erasure (interaction +0.024 for E1, P < 0.001; +0.004 for E2, P = 0.14). Thus the gate routes low-evidence regions to memory as designed, but the memory branch does not recover anatomy when local image evidence is removed.

（以上英文按 fold 0 写，fold 1–2 跑完后数字需要替换。）
