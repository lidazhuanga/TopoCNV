# V4 实现计划（协议 v0.4-draft）

- 日期：2026-10-22
- 依据：`D:\CNV\methods_redesign\CNV_protocol_v0.4-draft.md`（以它为准）、`METHOD_REDESIGN_v1.md`
- 负责会话：cnv-v4（由主会话管理）
- 本文件只写"怎么做"。所有试过的配置和结果记在 `docs/V4_DEV_LOG.md`。

## 0. 开工前查到的事实（只读分析，未训练）

1. **缓存轴向**：`<case>_image.npy` 形状 (z, a1, a2)，间距 2.5 / 1.133 / 1.133 mm。**左右方向是第 1 轴**（类别 5 在第 1 轴上是双峰、中间为零），第 2 轴是前后。左右分侧一律按第 1 轴。
2. **ROI 覆盖率的上限（用 GT 路标框，154 例）**：

   | 路标 | m = 15 mm | 25 mm | 35 mm |
   |---|---|---|---|
   | 只用类别 4 | 均值 91.1%，最低 85.4% | 97.3%，最低 91.0% | 99.75%，最低 96.3%，92% 的病例 ≥ 99% |
   | 类别 1∪4 | 99.4%，最低 91.5% | 99.97%，最低 98.8% | 100% |

   ROI 体积只占全体积的 2–4%。
   结论：只用类别 4，即使框是对的，m = 35 也有 8% 的病例达不到 99%。所以 **Stage 1 学类别 1 和 4**（协议 §4 写了"类别 4，可加类别 1"）。类别 4 概率仍单独作为先验通道 (a)。
3. **SOTA DynUNet fold 0**（cnv-sota，按报告折选模型）：Dice_Y 0.6423，HD95_Y 12.6 mm，best epoch 275，单卡约 3.5 小时（其中每 25 epoch 一次 31 例的全体积验证）。
4. 训练环境 `topocnv` 里没有 scikit-image（骨架化要用），也没有 PyYAML 和 pytest。不往共享环境里装：骨架预计算 `tools/v4_skeletons.py` 用 nnU-Net 的 venv（`~/cnv/envs/nnunetv2_cnv`，有 skimage 0.24）跑，只用到 numpy/scipy/skimage；`configs/*.yaml` 写成 JSON 风格（这也是合法的 YAML），用 `json` 读；测试用 `unittest`。

## 1. 文件结构

```
trainer/v4/
  splits.py        折、内部验证集、Stage 1 内层折（全部由种子确定）
  data.py          Stage 1 / Stage 2 数据集（缓存 CT + 先验 + ROI + 标签 + 骨架）
  model.py         DynUNet（与 npc_sota_train.build_model('nnunet') 同配置，输入/输出通道可变）
  losses.py        ROI 掩码版 WeightedBCEDice、可见度辅助损失、Skeleton Recall Loss
  roi.py           由 Stage 1 概率求左右两侧外接框，清理远处假阳性
  priors.py        先验通道 (a)(b)(c)：类别 4 概率、距离图、群体概率图谱（含相似变换 ICP）
  infer.py         ROI 内滑窗推理、OOF 保存
  train_stage1.py  Stage 1：3 类（背景 / 类别 1 / 类别 4）
  train_stage2.py  Stage 2：R0 … Full-v4 全部臂共用一个入口，用配置开关部件
tools/
  v4_build_priors.py   某一折 Stage 1 完成后，生成该折全部 154 例的先验和 ROI
  v4_roi_coverage.py   ROI 对 GT Y 的覆盖率
  v4_queue.py          GPU 队列（规则见 §6）
  v4_dev_summary.py    汇总内部验证结果到 V4_DEV_LOG
tests/test_v4_*.py     单元测试和冒烟测试
configs/v4_dev/*.yaml  开发阶段每个候选配置一份
configs/v4_frozen.yaml 冻结配置
results/v4_*/  log/v4_*/  docs/V4_*.md
```

R0 复现核对：主会话 2026-10-22 同意只看 R0 一个臂在报告折 fold 0 上的 Dice_Y/HD95，与 cnv-sota 的 DynUNet fold 0 对比。R0 是固定基线，不参与任何设计选择；这次访问记进 `docs/V4_DEV_LOG.md` 和 `log/v4/report_access.log`。其他臂冻结前不看报告折。

不改 `MIA/scripts/`、`trainer/npc_sota_train.py`、`tools/sota_*`、`tools/p0_*`。只 import 其中的函数（`metrics_protocol._dice/_hd95/mask_Y`、`npc_losses.WeightedBCEDiceLoss`、`npc_preprocess` 的增强函数）。

## 2. 划分

- **报告折**：`extras/splits.json` 的 5 折（31/31/31/31/30 例），与 P0、SOTA 相同。
- **内部验证集 V_k**：第 k 折的训练病例 T_k（123 或 124 例）按 ID 排序，用 `numpy.random.default_rng(20261022 + k)` 打乱，取前 round(0.15·|T_k|) 例（18 或 19 例）。Stage 2 只在 T_k − V_k 上训练，只按 V_k 选 checkpoint。
- **Stage 1 内层折**：T_k 再分成 3 个内层折（种子 `20261022 + 100k`，V_k 也分在内层折里）。

## 3. Stage 1 怎么做到嵌套

第 k 折的 Stage 1 一共训练 3 个模型，每个模型用 T_k 中另外两个内层折的病例训练：

- **T_k 里的每一例**（包括 V_k）：用没见过它的那个内层模型预测，即内层 OOF。Stage 2 的训练和内部验证都用这些预测。
- **报告折病例**：3 个内层模型的概率取平均。3 个模型都没见过报告折。

所以任何一例的 Stage 1 输入都来自没见过它的模型，报告折更是连 Stage 1 都没碰过。代价是报告折用的是 3 模型平均，比单模型略好、略平滑，这一点会写进 Methods。

Stage 1 配方：DynUNet，3 类（背景、类别 1、类别 4；类别 2、3、5 当背景），128³ patch，`random_crop_prob 0.33`（旧 Stage 1 是 0，远处假阳性多半由此而来），WeightedBCEDice（类别权重 0.1/10/10），**固定 150 epoch、取最后一个 checkpoint，不做选择**，这样 Stage 1 不需要再嵌一层验证集。

每折 3 个 run，5 折共 15 个，每个约 1–1.5 小时（含预测约 70 例全体积）。

## 4. 部件实现

### 4.1 ROI 级联（R1）

1. Stage 1 概率 → 路标掩码 L = argmax ∈ {1, 4}。
2. 左右分侧：头部中线取自 CT（颅底层面的体轮廓在第 1 轴上的中点），不依赖 L，避免单侧假阳性带偏。
3. 每侧去掉远处假阳性：取最大连通域，再并入与它最近距离 ≤ 30 mm、且 ≥ 10 体素的连通域，直到不再增加。这一步的参数在开发阶段只用 **T_0、T_1 训练病例的内层 OOF 预测**调，不看 Stage 2 结果，定了就写进冻结配置。
4. 每侧外接框外扩 m mm，两个框的并集就是 ROI。某一侧没有路标时，用另一侧的框按中线镜像；两侧都没有就不裁剪（全体积），并记录。
5. Stage 2 的做法：输入仍是全体积 CT（保留 128³ 的上下文），**patch 中心只在 ROI 内采样，损失只在 ROI 内的体素上计算**，推理只在 ROI 外接区域上滑窗，ROI 外一律置 0。这样 R0 和 R1 的骨干、patch、配方完全一样，差别只在 ROI。
6. 覆盖率：`tools/v4_roi_coverage.py` 对每例输出 Y 体素覆盖率、ROI 体积。开发阶段报 V_0 ∪ V_1，报告阶段报全部 154 例的 OOF。

m 的选法（预先写定）：在 V_0 ∪ V_1 上，取 {15, 25, 35} 中**平均覆盖率 ≥ 99% 的最小 m**；都达不到就取 35，并照实报告。不为每个 m 训练 Stage 2。

### 4.2 先验通道（P）

都在第 k 折内构建，只用 T_k 的标签：

- (a) Stage 1 的类别 4 概率（ROI 外置 0）。
- (b) 到预测骨孔的距离：对清理后的类别 4 掩码做各向异性 EDT（mm），截断在 30 mm，归一化为 1 − d/30。
- (c) 群体概率图谱：
  - 模板病例：T_k 里两侧类别 4 质心间距处于中位数的那一例。
  - 每个 T_k 病例用 **GT 类别 4** 的点云，左右两侧分别做相似变换 ICP（旋转 + 各向同性缩放 + 平移，缩放限制在 0.8–1.25），把它的 Y 映到模板空间，取平均。
  - 对目标病例，用 **Stage 1 预测的类别 4** 做同样的 ICP，把图谱映回目标空间。训练病例和测试病例都用预测的类别 4 对齐，训练和测试一致。
  - 留一：T_k 里的病例（包括 V_k）用的图谱不含它自己的 Y；报告折用 T_k 全部病例的图谱。
- 先验只存 ROI（m = 35）外接框内的部分，uint8 压缩，框外按 0 处理。

组合的选法：见 §5 的 D2。

### 4.3 可见度辅助头（V）

DynUNet 输出 2 + 5 = 7 个通道（同一个 1×1×1 输出卷积）：前 2 个是主输出 Y，后 5 个是背景 + 类别 1/2/3/4。辅助损失 = CE + soft Dice（不含背景），权重 λ_v 固定为 0.5（协议没有规定，我事先定下，不调）。类别 5 的两种处理：当背景，或在辅助损失里忽略（CE 和 Dice 都去掉这些体素）；在 D3 里选。主输出 Y 里类别 5 一律是背景（终点定义不变）。报告阶段另存辅助头的 argmax（ROI 内）供 Secondary。

### 4.4 Skeleton Recall Loss（S）

按 Kirchhoff et al. (ECCV 2024)：对每例 GT Y 做 3D 骨架化，用 3×3×3 结构元膨胀一次成"管状骨架"，再与 Y 取交，预先存盘。损失 = 1 − Σ(p_Y · S) / (Σ S + ε)，在 batch 和空间上求和。总损失 = 主损失 + λ_v·辅助损失 + w·骨架损失，w 在 D4 里从 {0.5, 1.0} 选。

### 4.5 打乱先验（Full-v4 + 打乱先验）

沿用旧 A2 的 `spatial_roll`：先验通道在整个体积上按每例固定的随机位移循环平移（种子由病例 ID 决定），**训练和推理用同一个位移**（旧 A2 训练时在 patch 内打乱、推理时在全体积打乱，两者不一致，这里改掉）。ROI 不打乱。

### 4.6 测试

| 部件 | 测试 |
|---|---|
| splits | 5 折与 `extras/splits.json` 一致；V_k ⊂ T_k；报告折与 V_k、Stage 1 训练集都不相交；内层折覆盖 T_k |
| losses | ROI 掩码全为 1 时与 `WeightedBCEDiceLoss` 数值一致；骨架损失在完美预测时为 0、全背景时为 1 |
| roi | 合成体积：远处假团块被去掉；单侧缺失时镜像；外扩量按 mm 换算正确 |
| priors | ICP 能恢复已知相似变换（误差 < 1 mm）；图谱留一不含自身；EDT 截断和归一化正确 |
| model / train | 2 例、2 epoch 的冒烟训练跑通（R0、R1、Full-v4），输出能被 `eval_protocol_metrics.py --pred_binary_Y` 读取 |

## 5. 开发阶段（协议 §5）

只用 fold 0、fold 1，只看 V_0、V_1。判据统一为：**V_0 ∪ V_1 合并后（约 37 例）的平均 Dice_Y**；与最好的候选差 < 0.003 时取更简单的那个；HD95_Y 一并记录，只作参考。按顺序贪心选择，每一步都在前一步选定的配置上做：

| 步 | 选什么 | 候选 | run 数 |
|---|---|---|---|
| D0 | 基线 | R0 | 2 |
| D1 | ROI 外扩 m、ROI 清理参数 | {15, 25, 35}，按 §4.1 的覆盖率规则 | 0（只用 Stage 1）+ R1 2 个 |
| D2 | 先验组合 | {a,b,c}、{b,c}、{a,c}、{a,b} | 8 |
| D3 | 类别 5 在辅助头里的处理 | 当背景 / 忽略 | 4 |
| D4 | 骨架损失权重 | 0.5 / 1.0 | 4 |

再加 Stage 1 共 6 个 run。开发阶段约 26 个 run。所有候选，包括没选上的，都记进 `docs/V4_DEV_LOG.md`。开发阶段不读报告折的任何指标。

冻结配置下 fold 0、1 的 Stage 2 run 和报告阶段的 run 完全相同（同样在 T_k − V_k 上训练、按 V_k 选），报告阶段直接沿用，不重训。

## 6. GPU 和时间

- 队列规则：最多同时 4 个 v4 run；每张卡最多 1 个 v4 run（按会话分别计，可以和 cnv-sota 共用一张卡，主会话 2026-10-22 确认）；只在空闲显存 ≥ 40 GB 的卡上启动；不向任何进程发信号。nnU-Net v2 还有 run 在 SOTA 队列里等着时，v4 启动后该卡仍要留 40 GB。主会话已让 cnv-sota 暂停启动新的非 nnU-Net run。
- 估算（单 run：Stage 1 约 1.2 小时，Stage 2 约 2.5–3 小时；内部验证只有 18 例，ROI 臂只在 ROI 内滑窗，比 SOTA 的全体积验证快）：

  | 阶段 | run 数 | GPU 小时 | 4 卡墙钟 |
  |---|---|---|---|
  | 开发 | 6 个 Stage 1 + 20 个 Stage 2 | 约 65 | 约 1 天（D0–D4 有先后，实际 1.5–2 天） |
  | 报告 | 9 个 Stage 1 + 约 25 个 Stage 2 | 约 85 | 约 1 天 |

  这是 4 张卡都能用的情况。现在能用的卡都被 cnv-sota 占着，实际时间取决于卡什么时候空出来。

## 7. 报告阶段

冻结后，协议 §4 的 7 个臂 × 5 折。OOF 存为 `results/v4_<arm>_label_predictions/<case>_stage2_label.nrrd`（int16 gzip，0/1，缓存网格，与 SOTA/P0 同格式），评价只调用 `eval_protocol_metrics.py --pred_binary_Y`，报告折只评一次。臂名：`r0`、`r1`、`r1p`、`r1pv`、`full`、`full_noP`、`full_shufP`。
