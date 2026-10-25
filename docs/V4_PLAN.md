# V4 实现计划（协议 v0.4-draft，2026-10-23 修订版）

- 依据：`D:\CNV\methods_redesign\CNV_protocol_v0.4-draft.md`（以它为准，§3 已于 2026-10-23 修订）、`METHOD_REDESIGN_v1.md`
- 负责会话：cnv-v4（由主会话管理）
- 本文件只写"怎么做"。所有试过的配置和结果记在 `docs/V4_DEV_LOG.md`。

## 修订记录

| 日期 | 改动 | 原因 |
|---|---|---|
| 2026-10-22 | 初版：DynUNet 骨干 | — |
| 2026-10-23 | **主线改为 nnU-Net v2 框架**。R0 = 与对照完全相同的 nnU-Net v2；其他臂都是 nnU-Net v2 的自定义 trainer 和数据集。DynUNet 版只保留已在跑的 R0 fold 0，作附录里的骨干对照；DynUNet 的 R0 fold 1 已从队列撤掉 | SOTA 5 折：nnU-Net v2 Dice_Y 0.659、HD95 5.0 mm，DynUNet 0.636、9.5 mm。验收线约 0.679，以 DynUNet 为地基要涨 0.04 以上 |
| 2026-10-23 | **ROI 改由图谱定义（待确认，见 §9 第 2 条）** | 真实 Stage 1 的路标框覆盖率只有 80–88%，见 §0 第 5 条 |
| 2026-10-24 | **开发/报告改为 nested CV**：D1–D4 在每个外层折 k 的 T_k−V_k 内独立完成，报告阶段用该折冻结的配置（configs/v4_frozen/fold{k}.json，train --variant report 以此门控）；开发期 Stage 1 单独一套（tier=dev，内层折只分 T_k−V_k，V_k 由三模型 ensemble 预测） | kimi k3.1-pro 审查（~/cnv/v4_kimi_review_final_20261024.md）P0-1/P0-2：同一套五折既调参又报告、开发期 Stage 1 未嵌套；用户 2026-10-24 决定采用 nested CV |

## 0. 已查到的事实

1. **缓存轴向**：`<case>_image.npy` 形状 (z, a1, a2)，间距 2.5 / 1.133 / 1.133 mm。左右方向是第 1 轴。
2. **nnU-Net v2 对照（Dataset501_CNVbinaryY）**：输入就是这份缓存（`nonorm`），plans 的目标间距与缓存相同，不重采样；3d_fullres，patch 56×192×192，batch 2，6 级 PlainConvUNet（32–320）；每 epoch 约 38 s，300 epoch 约 3.2 小时；取最后一个 checkpoint。预处理数据 39 GB。
3. **用 GT 路标框的 ROI 覆盖率上限**：类别 1∪4 + 25 mm 为 99.97%，只用类别 4 + 35 mm 为 99.75%（最低 96.3%）。所以 Stage 1 学类别 1 和 4。
4. **ROI 清理参数**：GT 路标每侧的连通域最多相距约 40 mm（T_0 − V_0：P99 39.5 mm），连接距离定为 45 mm（commit bceed15）。
5. **真实 Stage 1 的质量（fold 0，DynUNet，150 epoch，内层 OOF，T_0 − V_0 的 105 例）**：类别 1 Dice 0.35，类别 4 Dice 0.47；预测的路标体积约为 GT 的 60%；预测框在 z 低端比 GT 框短了中位数 55 mm（P90 67 mm），漏掉了一整段路标。按协议原定义（路标外接框 + m）得到的 ROI 覆盖率：

   | m | V_0（18 例）均值 / 最低 | T_0 − V_0（105 例）均值 |
   |---|---|---|
   | 15 | 67.9% / 46.5% | 66.1% |
   | 25 | 79.7% / 64.4% | 78.2% |
   | 35 | 87.6% / 78.4% | 86.0% |

   远达不到 99% 的目标。
6. **图谱定义的 ROI**（§4.1 的方案 B）：用预测的类别 4 点做 ICP，把训练折 Y 图谱（阈值 0.02）的支撑区映到当前病例，每侧取外接框 + m。同一份 Stage 1 预测下：

   | m | V_0 均值 / 最低 | T_0 − V_0 均值 / 最低 | ROI 体积 |
   |---|---|---|---|
   | 15 | 99.96% / 99.3% | 99.65% / 93.6% | 3.3% |
   | 25 | 100% / 100% | 99.98% / 99.5% | 4.7% |
   | 35 | 100% / 100% | 100% / 100% | 6.3% |

7. 训练环境 `topocnv` 没有 scikit-image、PyYAML、pytest：骨架化放在 nnU-Net 的 venv 里做；`configs/*.yaml` 用 JSON 风格写；测试用 `unittest`。

## 1. 总体结构

```
Stage 1（DynUNet 3 类，背景 / 类别 1 / 类别 4；开发/报告两个层级，见 §5）
   └→ tools/v4_build_priors.py：每折 154 例的 ROI 记录 + 先验 (a)(b)(c)      [已实现，fold 0 已完成]
        └→ tools/v4_nnunet.py prep --fold k：写 nnU-Net 数据集 Dataset51k_CNVv4f{k}
             （ROI 裁剪后的 CT + 3 个先验通道，标签为原始 0–5）
             └→ nnUNetv2_train 51k 3d_fullres <split> -tr nnUNetTrainerV4_<arm>
                  └→ tools/v4_nnunet.py export：贴回全体积、ROI 外清零 → results/v4_<arm>_label_predictions/
```

新增文件（全部在我的目录内）：

```
trainer/v4/nnunet/
  trainer_v4.py     nnUNetTrainerV4 及各臂子类（通道开关、可见度头、骨架损失、按 Y 采样、导出）
  skeleton.py       SkeletonTransform（按 Kirchhoff 2024 官方 nnU-Net 实现：增强后的 seg 骨架化 + 管状膨胀）
  losses.py         Y 主损失（nnU-Net 原版 DC_and_CE）+ 可见度 DC_and_CE + Skeleton Recall
tools/v4_nnunet.py  prep / train / export / install（把 trainer 链接进 venv 的 nnunetv2 包）
tests/test_v4_nnunet_*.py
```

nnU-Net 的原始数据和结果放在 `~/cnv/nnunet_cnv/` 下，新 dataset ID 用 510–515（510 = Stage 1 备用，511–515 对应 fold 0–4）。**不动 Dataset501 及其结果。**

## 2. 划分

- 报告折：`extras/splits.json` 的 5 折。
- 内部验证集 V_k：T_k 的 15%，种子 `20261022 + k`（实现不变）。
- Stage 1 内层折：报告层 3 个内层折分整个 T_k，种子 `20261022 + 100k`；开发层 3 个内层折只分 T_k − V_k，种子 `20261022 + 100k + 5000`（trainer/v4/splits.py 的 tier 参数）。
- 每个 Dataset51k 的 `splits_final.json` 写两条：条目 0 = 报告用（train = T_k，val = 报告折 k），条目 1 = 开发用（train = T_k − V_k，val = V_k）。开发阶段只跑条目 1。

## 3. 选模型（已确认 2026-10-23：取最后一个 epoch，见 §9 决策记录）

建议**沿用 nnU-Net 默认：取最后一个 epoch**（协议 §3 允许二选一）：

- 开发阶段：每个候选在 T_k − V_k 上训练，用最后一个 epoch 预测 V_k，在 V_k 上比较。
- 报告阶段：冻结的配置在 T_k 上重新训练，用最后一个 epoch 预测报告折。
- 好处：R0 就是 cnv-sota 已经跑完的 nnU-Net v2 5 折，训练数据、plans、epoch、选模型方式完全一致，不用重跑；Full-v4 和对照的训练数据也一样多。
- 代价：开发阶段的 run 不能直接用于报告阶段，每个臂报告阶段都要重训 5 折。
- 另一个选项（每 25 epoch 在 V_k 上按 Dice_Y 选）：所有臂（包括 R0）只用 T_k − V_k 训练，比对照少 15% 的训练数据，对 Full-v4 不利；R0 也要重跑 5 折。

## 4. 部件在 nnU-Net v2 里的实现

所有 Stage 2 臂共用 `nnUNetTrainerV4`（继承 `nnUNetTrainer_300epochs`，plans 从 Dataset501 复制，只把通道相关字段改成 4 通道 `NoNormalization`，patch、网络、batch、优化器、学习率、增强、epoch 全部不变）。各臂只在下面几处不同：

### 4.1 ROI 级联（R1 及之后所有臂）

- 预处理裁剪：每例只把 ROI 外接区域写进 nnU-Net 数据集，推理后贴回全体积，ROI 外为 0。ROI 小于 patch 的方向由 nnU-Net 自己补零。
- ROI 的定义，两种方案：
  - **方案 A（协议原文）**：Stage 1 路标外接框 + m。覆盖率 80–88%（§0 第 5 条），达不到 99%。
  - **方案 B（建议）**：路标仍然决定位置——预测的类别 4 点和模板做 ICP——但外接框取的是映过来的训练折 Y 图谱支撑区（阈值 0.02，留一），每侧外接框 + m。覆盖率 99.98–100%（§0 第 6 条）。阈值 0.02 和 m 一起在内部验证上定：取平均覆盖率 ≥ 99% 的最小 m。
- 这一步在开发阶段做完就定下 m，不为每个 m 训练 Stage 2。

### 4.2 先验通道（P）

- 数据集里固定写 3 个先验通道 (a)(b)(c)（实现与 fold 0 已生成的相同）。哪个组合由 trainer 决定：没选的通道在训练和推理时都置 0。网络结构对所有臂一样（4 个输入通道），R1、Full-v4 − P 就是 3 个先验通道全为 0。
- 打乱先验：同一个数据集，先验通道换成按病例固定位移循环平移后的版本（另写一份数据集，ID 后缀用 `_shuf`）。
- 强度类增强只作用于 CT 通道 0（2026-10-24 决定）：`nnUNetTrainerV4.Channel0Only` 包装 stock 的 Gamma/亮度/对比度/模糊/低分辨率/噪声变换，先验通道不参与；空间变换仍作用于全部通道，保持 CT/先验/标签对齐。R0 单通道不受影响。

### 4.3 可见度辅助头（V）

- 网络输出通道 2 + 5（每个深监督层都有），前 2 个是 Y，后 5 个是背景 + 类别 1/2/3/4。
- 标签存原始 0–5。trainer 从中生成 Y 目标和可见度目标；类别 5 当背景或忽略（nnU-Net 的 ignore 机制）。
- 损失：Y 用 nnU-Net 原版 DC_and_CE（含深监督权重），可见度头用同样的 DC_and_CE，权重 λ_v = 0.5（事先定死，不调）。
- 采样：nnU-Net 默认按类别采前景。标签有 6 类时，类别 5 也会被当成前景采样，所以 trainer 改为把类别 1–4 合并成 Y 再采样，与对照（只有 Y 一类前景）一致。

### 4.4 Skeleton Recall Loss（S）

照 Kirchhoff et al. (ECCV 2024) 的官方 nnU-Net 实现：在数据增强流水线里，空间变换之后对 Y 做 3D 骨架化，再膨胀成管状（`do_tube`），随深监督一起下采样；损失为 Y 概率在管状骨架上的软召回，权重 w ∈ {0.5, 1.0}。之前预计算的骨架（`results/v4_skel/`）只给 DynUNet 版用。

### 4.5 导出

trainer 的最终验证改为：logits 的前 2 个通道 → Y（softmax argmax），后 5 个 → 可见度标签；用 nnU-Net 自己的导出函数还原 nonzero 裁剪，再由 `tools/v4_nnunet.py export` 按 ROI 框贴回全体积，写 `{case}_stage2_label.nrrd`（int16，缓存网格）。

## 5. Stage 1（嵌套，两个层级）

DynUNet 3 类（背景 / 类别 1 / 类别 4），150 epoch，取最后一个 checkpoint，每层 3 个内层模型；被整体预测的一侧永远用 3 个模型的平均概率：

- **开发层（`--tier dev`）**：内层折只划分 T_k − V_k。内层模型 j 在 (T_k − V_k) − 内层折 j 上训练；预测自己的内层折（inner-OOF，供 Stage 2 训练病例建先验）和 V_k（存 `inner{j}/val_pred/`，三模型平均＝开发层的 held-out 推理，与报告折同口径）。**V_k 不出现在任何开发层 Stage 1 的训练集里。**
- **报告层（`--tier report`）**：内层折划分整个 T_k（V_k 在这一层是普通训练数据——报告层没有选择动作）。预测自己的内层折和报告折 k（`inner{j}/report_pred/`，三模型平均）。

产物目录：`results/v4_stage1/fold{k}_dev/`（开发层）、`results/v4_stage1/fold{k}/`（报告层）。fold 0/1 已有的报告层 run 继续有效；**此前用报告层 Stage 1 构建的 fold{k}_dev 先验（2026-10-23）作废**，须由开发层 Stage 1 重建（开发层 Stage 1 共 15 个 run，全部新跑）。

## 6. 开发阶段＝嵌套选择（协议 §5，2026-10-24 改为 per-fold）

选择在每个外层折 k（k = 0…4）**内部独立**完成：只用 T_k − V_k 训练、只在 V_k 上比较；报告折 k 从头到尾不参与该折的任何选择。判据不变：V_k 平均 Dice_Y；与最好候选差 < 0.003 时取更简单的（简单度排序写进候选清单）；HD95_Y 只作参考。候选训练数据 = 数据集 52k（dev 条目：train = T_k − V_k，val = V_k）。

每折流程：

1. 开发层 Stage 1 ×3 → `tools/v4_build_priors.py --variant dev`（读 fold{k}_dev，重建 fold{k}_dev 先验）→ 按 §6.1 规则定 m（只用本折训练病例与 V_k 的覆盖率）。
2. 下表候选在 T_k − V_k 上训练，V_k 上导出预测，`eval_protocol_metrics.py --pred_binary_Y` 出逐例 CSV。
3. `tools/v4_select_config.py --fold k` 汇总：**完整性硬校验**（每候选病例集合恰为 V_k，缺一例即中止）→ 选择规则 → 写 `configs/v4_frozen/fold{k}.json`（该折冻结配置 + 全部候选数字留痕）。
4. R0-dev / R1-dev 为参照（不进选择规则），但每折都跑，用于判断"有没有候选打平/超过基线"。

| 步 | 内容 | run/折 |
|---|---|---|
| D0-ref | R0-dev（数据集 519，无级联） | 1 |
| D1 | m 按覆盖率规则定（不训练）；R1-dev 参照 | 1 |
| D2 | 先验组合 {a,b,c}、{b,c}、{a,c}、{a,b} | 4 |
| D3+D4 | 类别 5 {当背景, 忽略} × 骨架权重 {0.5, 1.0} | 4 |
| 链闭合 | R1+P+V（先验组合 = 本折 D2 的选择；类别 5 处理 = 本折 D3 的选择；无骨架） | 1 |
| 合计 | 每折 11；共 55，fold 0/1 的 R0-dev 已有 2 个 | 新启动 53 |

R1+P+V（用户 2026-10-25 采纳，kimi P1-8）在该折 D2/D3 结果出来后启动，使 R1 → R1+P → R1+P+V → Full-v4 归因链每一环都有 V_k 数据，并验证先验组合在有无骨架损失时排序稳定。

### 6.1 D1-AM1：ROI margin 规则与解剖安全回退（用户确认 2026-10-25）

依据 kimi k3.1-pro 咨询（`~/cnv/kimi_d1_answer_20261025.log`，会话 3b2fb650）与实测覆盖率（`results/v4_roi_coverage/dev_train_folds*.csv`）：

1. **m 选择（原规则 + 不可行闭合）**：每折只用 dev 层 T_k − V_k 的 105 个留一/inner-OOF 训练病例，取未回退图谱 ROI 最低覆盖率 ≥ 99% 的最小 m∈{15,25,35}；若不存在，取上限 m=35。回退后的覆盖率不参与 m 的重新选择。冻结结果：**f0=35、f1=25、f2=35、f3=15、f4=35**。
2. **安全门（GT 无关，全折全病例统一）**：D = 双侧清洁类别 4 点云质心的左右物理距离（真实质心，非框中心）。在本折 105 个训练病例上冻结 L_k=Q1−3·IQR、U_k=Q3+3·IQR（线性分位）。结构性失败（任一侧非 own-side ICP、变换非有限/图谱支撑空/框缺失、D 不可算）或 D∉[L_k,U_k] → **全体积回退：boxes 清空、a/b/c 先验置零、导出不做 ROI 外清零**，四个通道形状不变，所有级联臂共享同一回退决定。冻结值见 `configs/v4_qc/fold{k}.json`（`tools/v4_qc_d1.py` 生成；当前训练集触发仅 f0 的 2200498367，D≈347mm，其余 524 例最大 58mm）。
3. **诚实定位**：本修订由训练失败促成（f0 单例图谱映射整体错误、f4 三例尾部 97.7-98.8% 不可达 99%），是对不完整规则的确定性闭合而非降阈值；f4 三例不特判、如实披露；冻结后不因 V_k/报告折结果回调。报告阶段沿用 dev 冻结的 m 与阈值，不重算；每折报告触发例数与回退前后覆盖率，回退病例不剔除出主分析。
4. 空预测处理（评价器已实现，2026-10-24）：HD95_Y 对空预测（GT 非空）返回该病例的物理对角线作为惩罚值并计入汇总均值，不再被 nanmean 静默剔除；逐例输出 Empty_pred_Y 标记，汇总报告空预测数/率（ASSD_Y 为可选指标，维持 NaN 语义）。

## 7. 报告阶段（每折用自己的冻结配置）

1. 5 份 `configs/v4_frozen/fold{k}.json` 齐全后报告阶段才开跑（`tools/v4_nnunet.py train --variant report` 以此为门控）。**不同折允许选出不同配置**——被评价的对象是"带内部选择流程的整套方法"，这正是 nested CV 的语义。
2. 每折：报告层 Stage 1 ×3（fold 0/1 已有，fold 2–4 新跑 9 个）→ `--variant report` 先验 → 6 个臂（R1、R1+P、R1+P+V、Full-v4、Full-v4 − P、Full-v4 + 打乱先验）按 fold{k}.json 配置各训练 1 份（共 30 run）→ export → `eval_protocol_metrics.py --pred_binary_Y` 只评一次。
3. R0 直接用 cnv-sota 的 nnU-Net v2 五折 OOF（训练数据、plans、epoch、选模型方式一致）。
4. 评价前完整性硬校验：每臂恰 154 例、各折病例集合精确一致、无重复（SOTA 曾发生 149 例被当结果的事故，SOTA_RUNBOOK §49-54）。
5. 统计口径（已确认，2026-10-24）：主比较预注册为 Full-v4 vs R0 的逐例配对 ΔDice/ΔHD95 + bootstrap CI，其余臂为次要比较（Holm 校正）。不做重复 seed（用户决定：审稿人不关注 seed 设置）。

**必须写进论文的局限**：主干选择（DynUNet→nnU-Net v2）依据过完整 SOTA 五折 OOF，nested CV 消除不了这一曝光；确认性结论需要未参与设计的新测试队列。重跑同一批 154 例无济于事——泄漏在“设计用过这些数据”这一事实层面，不在任何单次 run 的质量（2026-10-24 用户问询后确认：不重跑）。

### 7.1 数据范围（用户 2026-10-25 确认）

绿皮稿件的数据范围 = **医生标注与医生修改过的病例**，与稿件口径一致：178 例连续筛查 → 21 例排除（影像不全/缺 MRI）→ 157 例有配准 MRI → **154 例完成四层标注**（五折交叉验证）+ 3 例无标注（剂量学可行性专用）。约 250 例的完整存量与 2026-10-24 新到的 120 例无标签病例**一律不使用**：不进训练、验证、测试，也不进任何配准或先验产物。2026-10-24 拟定的"其标注到位后作锁定确认集"方向**取消**。

120 例无标签 CT 维持隔离（`npc_data/train/image_orphans_no_label_20261024/`）；其 MR 虽存于 `mri/raw/` 但不在 selection 清单内，配准循环不处理。主干曝光（DynUNet→nnU-Net v2 依据过全量 SOTA 五折）的最终处理即 §7 末段的**局限声明**，本稿不设外部确认集。

## 8. GPU 和时间

队列规则不变：最多 4 个 v4 run；每卡最多 1 个 v4 run；空闲显存 ≥ 40 GB 才启动；不向任何进程发信号。单个 Stage 2 run 约 3.5–4 小时；Stage 1（DynUNet）每个约 40 分钟。

| 内容 | run 数 | GPU 小时 |
|---|---|---|
| 开发层 Stage 1（5 折 × 3） | 15 | ~10 |
| 开发候选（每折 11；fold 0/1 的 R0-dev 已有 2） | 55（新 53） | ~210 |
| 报告层 Stage 1（fold 2–4 × 3） | 9 | ~6 |
| 报告臂（6 臂 × 5 折） | 30 | ~120 |
| 合计 | ~109 | ~350 |

四卡理想满载约 82 小时纯训练；按波次串行和共享资源加 30–50% 余量，现实预期 **5–6 天**。若采纳重复 seed（+~10 run）或 R1+P+V（+5）相应增加。

**大规模排队的前置条件（kimi P0-3）**：先用当前 nnU-Net 路径完成 R1、Full-v4、打乱先验的真实端到端冒烟（2+5 输出、class 5 ignore、在线骨架、ROI 贴回、`_vis.nrrd`、evaluator 读回），吞吐不做专门实测（用户 2026-10-24：符合一般要求即可），时间预算按上表加余量执行。`tests/v4_smoke.sh` 目前仍是旧 DynUNet 路径，需要改写。另：`tools/v4_nnunet.py` 曾有一处 f-string 语法错误（2026-10-24 修复），说明 prep 自该行引入后从未执行过——第一次 prep 时留意预处理输出。

## 9. 决策记录（2026-10-24 更新）

1. ~~选模型：最后一个 epoch~~（2026-10-23 确认）。
2. ~~ROI 定义：方案 B（图谱支撑区 ICP 映射 + m）~~（2026-10-23 确认）。
3. ~~Stage 1 骨干：保持 DynUNet~~（2026-10-23 确认）。
4. ~~评价设计：nested CV（每折独立选择）~~（2026-10-24 确认，§5–§8 已按此修订）。
5. ~~D1 的 m 规则：下尾规则（训练病例最低覆盖率 ≥ 99%）~~；空预测 HD95 = 病例物理对角线惩罚、另报空预测率（2026-10-24 确认，评价器已实现）。
6. ~~统计预注册：主比较 Full-v4 vs R0 逐例配对 ΔDice/ΔHD95 + bootstrap CI，其余 Holm~~；不做重复 seed（2026-10-24 确认）。
7. ~~先验通道增强：强度类增强只作用于 CT 通道 0~~（2026-10-24 确认，trainer 已实现，纳入冒烟验证范围）。
8. ~~冒烟与吞吐：按一般要求执行标准端到端冒烟，不做专门吞吐实测~~（2026-10-24 确认）。
9. ~~消融链闭合（kimi P1-8）~~：**采纳**（2026-10-25 用户确认）——开发期每折加 R1+P+V 1 个 run（+5 run ≈ +20 GPU 小时），归因链每一环都有 V_k 数据；与已被否决的重复 seed 无关（配置覆盖 vs 随机性）。
10. ~~D1 修订 D1-AM1（kimi 建议，详见 §6.1）~~：**确认采纳**（2026-10-25 用户确认）——各折 m=f0:35/f1:25/f2:35/f3:15/f4:35 + 双侧 c4 质心间距安全门（每折 Q1/Q3±3IQR 冻结，触发→全体积+先验置零），`configs/v4_qc/fold{k}.json` 为冻结载体，报告阶段不重算。
