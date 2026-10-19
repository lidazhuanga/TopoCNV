# TopoCNV

Code for automated multi-branch trigeminal nerve (CN V) delineation on planning CT, intended to support perineural-invasion-oriented CTV and OAR definition in nasopharyngeal carcinoma radiotherapy.

Associated manuscript (under review):

> Automated Trigeminal Nerve Delineation on Planning CT to Support Perineural-Invasion-Oriented Target Definition in Nasopharyngeal Carcinoma Radiotherapy

This repository contains the training, inference and evaluation code used for the manuscript. Imaging data and trained weights are not included.

## Method

The intracranial and skull-base course of CN V is only partly visible on planning CT. Near bony foramina and canals the image carries strong evidence; along soft-tissue segments it often does not. The model therefore switches its inference source by local image evidence (evidence-gated anatomical memory).

1. **Stage 1** segments six classes (background, V1, V2, V3, foramen/canal anchor = class 4, muscle bypass = class 5) with a ViT-UNETR backbone. The class-4 probability and a 3 mm dilation of the hard class-4 mask form the cascade prior.
2. **Stage 2** predicts one binary CN V mask `Y = 1∪2∪3∪4` from `[CT, P(c4), dilate(c4)]` with a DynUNet backbone that carries an anatomical memory bank at three encoder levels.
3. **Evidence map** `g(x)` combines a bone-window response on normalized CT (`σ((CT − 0.97)/0.01)`) with the foramen prior, smoothed by a 3×3×3 average. At every memory level the features are fused as `f = g·f_local + (1 − g)·f_memory`: high-evidence voxels follow the image, low-evidence voxels follow the anatomical memory.
4. **Loss** on `Y`: `L_seg + λ_L·L_seg|_{g<τ} + λ_mem·‖σ(Ŷ) − sg(σ(Ŷ_mem))‖²|_{g<τ} + 0.3·L_seg(Ŷ_mem)`, where `Ŷ_mem` is a memory-only readout, `sg` is stop-gradient, and λ_L = 1.0, λ_mem = 0.5, τ = 0.5.

The bone threshold 0.97 is calibrated on the training cohort: the cache normalizes each CT by its own 1st–99th percentiles, which maps 200 HU to a median of 0.993 (10th percentile 0.964). `tools/calibrate_bone_threshold.py` reproduces this.

Implementation: `trainer/cnv_protocol_s2.py` (evidence map, loss, Stage-2 model), `npc_nnunet_mla.py` (memory-level gating), `trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py` (training and inference driver).

## Evaluation

The primary endpoint is the merged CN V mask `Y = 1∪2∪3∪4`; class 5 (bypass) is not part of the target. All metrics are in `MIA/scripts/metrics_protocol.py`.

| Role | Metrics |
|------|---------|
| Primary | Dice_Y, HD95_Y (mm) |
| Mechanism (low-evidence region L) | Dice_L, Recall_L, \|L\|/\|Y\| |
| Secondary | Dice on V1∪V2∪V3; Dice on class 4 |

`L` is the part of `Y` outside the high-evidence region `H = (raw HU ≥ 200) ∪ dilate(class 4, 3 mm)`; HU is read from the original NIfTI, not from the normalized cache.

## Ablation arms (P0)

Each arm removes or replaces one component. A constant gate defines no low-evidence region, so A_gate applies the memory-alignment term to the whole volume. `python MIA/scripts/ablation_configs.py` prints the table; `--commands FOLD` prints the training commands.

| Arm | Memory | Gate | λ_L | Cascade prior | Tests |
|-----|--------|------|-----|---------------|-------|
| Full | on | g(x) | 1.0 | Stage-1 class 4 | proposed model |
| A_mem | off | — | 1.0 | Stage-1 class 4 | contribution of memory |
| A_gate | on | fixed 0.5 | 1.0 | Stage-1 class 4 | contribution of evidence gating |
| C1 | off | — | 0 | Stage-1 class 4 | cascade only |
| B0 | off | — | 0 | none (single stage) | cascade vs. single stage |
| A2 | off | — | 0 | spatially scrambled | semantic content of the prior |

## Installation

Training needs Python ≥ 3.9 and a CUDA GPU. The development environment was Python 3.9, PyTorch 2.8 (CUDA 12.8) and MONAI 1.5.

```bash
pip install -r requirements.txt
```

## Data layout

```
npc_data/train/
  image/<case>.nii.gz      planning CT
  label/<case>.nrrd        or <case>.nii.gz; labels 0 = background, 1 = V1, 2 = V2, 3 = V3, 4 = foramen/canal, 5 = bypass
```

Default locations are relative to the repository root. Override them with `CNV_DATA_ROOT` (data) and `CNV_RESULTS_ROOT` (cache, checkpoints, predictions).

## Usage

Run all commands from the repository root.

```bash
# Stage 1, folds 0-4 (the first run also builds the preprocessing cache)
python trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py --stage 1 --fold 0 \
    --epochs 300 --exp_suffix _s1_6class_protocol_v03_20261016 --gpu 0

# Out-of-fold Stage-1 predictions for every training case (after all five folds)
python trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py --stage generate_cv \
    --exp_suffix _s1_6class_protocol_v03_20261016 --gpu 0

# Stage 2: training commands for every arm of one fold
python MIA/scripts/ablation_configs.py --commands 0

# Inference on new CTs with the Full arm (other arms: use the model flags printed by ablation_configs.py)
python trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py --stage inference \
    --input_dir /path/to/ct --output_dir /path/to/out \
    --exp_suffix <stage-2 suffix> --stage1_exp_suffix _s1_6class_protocol_v03_20261016 \
    --backbone nnunet_mla --memory_type anatomical_v4 --s2_head binary_y \
    --use_memory_bank 1 --use_confidence_gate 1 --ensemble   # five-fold ensemble; or --fold k

# Protocol metrics on saved predictions
python MIA/scripts/eval_protocol_metrics.py --pred_dir /path/to/preds \
    --gt_cache results/preprocess_cache/sp_2.500_1.133_1.133 --pred_binary_Y --csv metrics.csv
```

Comparison networks (nnU-Net-style DynUNet, SegResNet, Attention U-Net, UNETR, SwinUNETR, nnFormer) are trained on the same binary `Y` target; checkpoint selection and reporting use the same sliding-window Dice_Y and HD95_Y as the proposed model:

```bash
python trainer/npc_sota_train.py --model segresnet --fold 0 --use_cache 1 --gpu 0
```

## Tests

```bash
python tests/test_evidence_gate.py        # CPU, about 30 s
python MIA/scripts/eval_protocol_metrics.py --self-test
```

The tests check the evidence map, each loss term, that `g = 1` reproduces the memory-free path while `g = 0` uses memory, that gradients reach the memory readout, and that the ablation arms are distinct. The real-case test is skipped when no data are present.

## Data availability

Imaging data are not deposited publicly. They are available from the corresponding author upon reasonable request.

## Contact

Senxiang Yan (corresponding author), Department of Radiation Oncology, The First Affiliated Hospital, Zhejiang University School of Medicine
Email: yansenxiang@zju.edu.cn

## Citation

A BibTeX entry will be added once the paper is published.

---

# TopoCNV（中文）

本仓库为鼻咽癌放疗计划 CT 上三叉神经（CN V）多分支自动勾画的代码，用于支持面向神经周围侵犯的 CTV 与危及器官定义。

相关论文（审稿中）：

> Automated Trigeminal Nerve Delineation on Planning CT to Support Perineural-Invasion-Oriented Target Definition in Nasopharyngeal Carcinoma Radiotherapy

仓库包含论文所用的训练、推理与评价代码，不含影像数据和模型权重。

## 方法

CN V 在颅底的走行在计划 CT 上只部分可见：骨性孔道附近证据强，软组织段常常缺乏证据。模型因此按局部影像证据切换推断来源（证据门控解剖记忆）。

1. **第一阶段**：ViT-UNETR 六类分割（背景、V1、V2、V3、孔道锚点 = 第 4 类、肌肉绕行 = 第 5 类）。第 4 类概率与其硬掩膜 3 mm 扩张作为级联先验。
2. **第二阶段**：输入 `[CT, P(c4), dilate(c4)]`，输出二值 CN V 掩膜 `Y = 1∪2∪3∪4`；骨干为 DynUNet，在三个编码层接入解剖记忆库。
3. **证据图** `g(x)`：归一化 CT 上的骨窗响应 `σ((CT − 0.97)/0.01)` 与孔道先验取最大，再做 3×3×3 平均平滑。每个记忆层按 `f = g·f_local + (1 − g)·f_memory` 融合：高证据处跟随影像，低证据处跟随解剖记忆。
4. **损失**：`L_seg + λ_L·L_seg|_{g<τ} + λ_mem·‖σ(Ŷ) − sg(σ(Ŷ_mem))‖²|_{g<τ} + 0.3·L_seg(Ŷ_mem)`，λ_L = 1.0，λ_mem = 0.5，τ = 0.5。

骨窗阈值 0.97 在训练队列上标定：缓存按每例 1–99 百分位归一化，200 HU 对应的归一化值中位数为 0.993（第 10 百分位 0.964）。可用 `tools/calibrate_bone_threshold.py` 复现。

## 评价

主终点为合并 CN V 掩膜 `Y = 1∪2∪3∪4`，第 5 类（绕行）不计入目标。

- 主指标：Dice_Y、HD95_Y（mm）
- 机制指标（低证据区 L）：Dice_L、Recall_L、|L|/|Y|
- 次要指标：V1∪V2∪V3 的 Dice；第 4 类 Dice

`L` 为 `Y` 中位于高证据区 `H = (原始 HU ≥ 200) ∪ dilate(第 4 类, 3 mm)` 之外的部分；HU 取自原始 NIfTI。

消融臂、安装、数据格式与运行命令见上方英文部分。

## 数据可用性

影像数据不公开存放。如需使用，请联系通讯作者（yansenxiang@zju.edu.cn）。

## 联系

严森祥（通讯作者），浙江大学医学院附属第一医院放疗科
邮箱：yansenxiang@zju.edu.cn

## 引用

论文正式发表后，将在此补充 BibTeX。
