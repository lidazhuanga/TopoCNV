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
Stage 1（嵌套，照原计划：DynUNet 3 类，背景 / 类别 1 / 类别 4）
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
- Stage 1 内层折：T_k 的 3 个内层折，种子 `20261022 + 100k`（实现不变）。
- 每个 Dataset51k 的 `splits_final.json` 写两条：条目 0 = 报告用（train = T_k，val = 报告折 k），条目 1 = 开发用（train = T_k − V_k，val = V_k）。开发阶段只跑条目 1。

## 3. 选模型（待确认，见 §9 第 1 条）

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

### 4.3 可见度辅助头（V）

- 网络输出通道 2 + 5（每个深监督层都有），前 2 个是 Y，后 5 个是背景 + 类别 1/2/3/4。
- 标签存原始 0–5。trainer 从中生成 Y 目标和可见度目标；类别 5 当背景或忽略（nnU-Net 的 ignore 机制）。
- 损失：Y 用 nnU-Net 原版 DC_and_CE（含深监督权重），可见度头用同样的 DC_and_CE，权重 λ_v = 0.5（事先定死，不调）。
- 采样：nnU-Net 默认按类别采前景。标签有 6 类时，类别 5 也会被当成前景采样，所以 trainer 改为把类别 1–4 合并成 Y 再采样，与对照（只有 Y 一类前景）一致。

### 4.4 Skeleton Recall Loss（S）

照 Kirchhoff et al. (ECCV 2024) 的官方 nnU-Net 实现：在数据增强流水线里，空间变换之后对 Y 做 3D 骨架化，再膨胀成管状（`do_tube`），随深监督一起下采样；损失为 Y 概率在管状骨架上的软召回，权重 w ∈ {0.5, 1.0}。之前预计算的骨架（`results/v4_skel/`）只给 DynUNet 版用。

### 4.5 导出

trainer 的最终验证改为：logits 的前 2 个通道 → Y（softmax argmax），后 5 个 → 可见度标签；用 nnU-Net 自己的导出函数还原 nonzero 裁剪，再由 `tools/v4_nnunet.py export` 按 ROI 框贴回全体积，写 `{case}_stage2_label.nrrd`（int16，缓存网格）。

## 5. Stage 1（嵌套，照原计划）

DynUNet 3 类，150 epoch，取最后一个 checkpoint，每折 3 个内层模型；报告折用 3 个模型的平均概率。fold 0 已完成，fold 1 在跑。

这个 Stage 1 偏弱（§0 第 5 条）。如果采用方案 B，ROI 不再依赖路标是否找全，Stage 1 的质量只影响先验 (a)(b) 和 ICP。要不要把 Stage 1 也换成 nnU-Net（3 类，每折 3 个 run，每个约 3.5 小时），见 §9 第 3 条。

## 6. 开发阶段（协议 §5）

只用 fold 0、fold 1，只看 V_0、V_1。判据：V_0 ∪ V_1 合并（37 例）的平均 Dice_Y；与最好的候选差 < 0.003 时取更简单的；HD95_Y 只作参考。

| 步 | 选什么 | 候选 | run 数 |
|---|---|---|---|
| D0 | 参照 | R0（nnU-Net v2 原版，Dataset501 的开发条目） | 2 |
| D1 | ROI 的 m（及方案 B 的阈值） | 覆盖率规则，不训练 | 0；另跑 R1 2 个作参照 |
| D2 | 先验组合 | {a,b,c}、{b,c}、{a,c}、{a,b}（R1+P） | 8 |
| D3+D4 | 类别 5 处理 × 骨架权重 | {当背景, 忽略} × {0.5, 1.0}（Full-v4），2×2 一次跑完 | 8 |

D3 和 D4 合成一个 2×2，是为了少一轮先后等待。R1+P+V 臂用 D3+D4 选出的类别 5 处理。开发阶段共 20 个 Stage 2 run。

## 7. 报告阶段

冻结后，协议 §4 的臂中 R0 直接用 cnv-sota 的 nnU-Net v2 OOF（若 §3 采用"最后一个 epoch"）；其余 6 个臂（R1、R1+P、R1+P+V、Full-v4、Full-v4 − P、Full-v4 + 打乱先验）各在 T_k 上训练 5 折，共 30 个 run。另需 fold 2–4 的 Stage 1（9 个 run）和先验。OOF 存到 `results/v4_<arm>_label_predictions/`，评价只调用 `eval_protocol_metrics.py --pred_binary_Y`，报告折只评一次。

## 8. GPU 和时间

- 队列规则不变：最多 4 个 v4 run；每卡最多 1 个 v4 run（可与 cnv-sota 共卡）；空闲显存 ≥ 40 GB 才启动；nnU-Net v2 还有对照 run 在等时，启动后该卡仍要留 40 GB；不向任何进程发信号。
- 单个 Stage 2 run：nnU-Net 300 epoch 约 3.2 小时，加骨架增强和导出，按 3.5–4 小时算。Stage 1（DynUNet）每个约 40 分钟。

| 阶段 | 内容 | GPU 小时 | 4 卡墙钟 |
|---|---|---|---|
| 实现 | nnU-Net trainer、数据集转换、单元测试、冒烟 | — | 约 1 天 |
| 开发 | 3 个 Stage 1（fold 1 剩余）+ 20 个 Stage 2 | 约 75 | 约 1 天（两轮：R0/R1/D2 12 个，再 D3+D4 8 个） |
| 报告 | 9 个 Stage 1 + 30 个 Stage 2 | 约 115 | 约 1.5 天 |

合计约 3.5–4 天，前提是 4 张卡都有空位。如果 Stage 1 改用 nnU-Net（§9 第 3 条），再加约 50 GPU 小时，约半天到 1 天。

## 9. 需要主会话 / 用户确认

1. **选模型**：建议取最后一个 epoch，报告阶段在 T_k 上重训，R0 直接用 cnv-sota 的 nnU-Net v2 结果（§3）。
2. **ROI 定义**：建议用方案 B（路标做 ICP 定位，外接框取映过来的训练折 Y 图谱支撑区 + m）。协议原文的方案 A 在真实 Stage 1 下覆盖率只有 80–88%（§0 第 5、6 条）。这会改协议 §4 的 ROI 定义。
3. **Stage 1 骨干**：建议保持 DynUNet（照原计划，已在跑）。只有在 D2 显示先验 (a)(b) 明显受 Stage 1 质量拖累时，才换成 nnU-Net。
