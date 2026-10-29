"""baseline/SOTA 共用分割循环与历史全量滑窗验证。

保留旧入口的损失统计、AMP、空类别与异常处理口径；不替换为
通用 trainer 中不同的梯度累积或非有限梯度策略。
"""

import csv
import json
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch

from topocnv.data.dataset import _get_all_cases, _create_folds
from topocnv.data.labels import load_label_volume
from topocnv.data import preprocess as pp
from topocnv.losses import multiclass_dice_score
from .amp import legacy_amp_settings

AMP_DTYPE, USE_GRAD_SCALER = legacy_amp_settings()


def compute_dice_per_class(pred: np.ndarray, label: np.ndarray, num_classes: int, smooth: float = 1e-5):
    """历史硬标签 Dice：跳过背景、双方为空记 NaN、均值忽略 NaN。"""
    dice_scores = []
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        label_c = (label == c).astype(np.float32)
        intersection = np.sum(pred_c * label_c)
        union = np.sum(pred_c) + np.sum(label_c)
        if union < smooth:
            dice = np.nan
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    valid_scores = [s for s in dice_scores if not np.isnan(s)]
    mean_dice = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
    return {'mean': mean_dice, 'per_class': dice_scores}


def train_supervised_epoch(model, loader, optimizer, loss_fn, device, scaler=None, *,
                           amp_dtype=AMP_DTYPE, dice_fn=multiclass_dice_score) -> Tuple[float, float]:
    """单阶段分割 epoch；异常损失仍计入分母，不检查或改变梯度策略。"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    use_amp = (amp_dtype is not None)
    for batch_idx, (images, labels) in enumerate(loader):
        try:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] NaN/Inf loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                num_batches += 1
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            with torch.no_grad():
                logits_for_dice = logits[0] if isinstance(logits, (list, tuple)) else logits
                dice = dice_fn(logits_for_dice.float(), labels, include_background=False)
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
        except Exception as e:
            print(f"\n[ERROR] Exception in batch {batch_idx}:")
            print(f"  images.shape: {images.shape if images is not None else 'None'}")
            print(f"  labels.shape: {labels.shape if labels is not None else 'None'}")
            print(f"  labels.min: {labels.min().item() if labels is not None else 'None'}")
            print(f"  labels.max: {labels.max().item() if labels is not None else 'None'}")
            print(f"  labels unique values: {torch.unique(labels).cpu().numpy() if labels is not None else 'None'}")
            raise e
    if num_batches == 0:
        return 0.0, 0.0
    return total_loss / num_batches, total_dice / num_batches


def evaluate_supervised_epoch(model, loader, loss_fn, device, *,
                              amp_dtype=AMP_DTYPE, dice_fn=multiclass_dice_score) -> Tuple[float, float]:
    """单阶段验证 epoch；保持历史非有限损失传播和批次平均策略。"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    use_amp = (amp_dtype is not None)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
            logits_for_dice = logits[0] if isinstance(logits, (list, tuple)) else logits
            dice = dice_fn(logits_for_dice.float(), labels, include_background=False)
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    if num_batches == 0:
        return 0.0, 0.0
    return total_loss / num_batches, total_dice / num_batches


def validate_with_sliding_window(
    model,
    data_root: str,
    fold: int,
    output_dir: str,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    device: str,
    spacing_params_path: str = None,
    step_size: float = 0.5,
    num_folds: int = 5,
    seed: int = 42,
    compute_metrics: bool = True,
    normalization: str = 'ct',
):
    """旧多类验证：保留空间轴、输出文件、平滑 Dice 和汇总日志。

    二值 Y 主要终点的 cache-space 验证仍由协议训练入口负责，
    不以此多类验证替代其指标或模型选择口径。
    """
    print("\n" + "=" * 60)
    print("开始滑窗推理验证集...")
    print("=" * 60)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")
    target_spacing = None
    clip_lower = -200.0
    clip_upper = 300.0
    if spacing_params_path and Path(spacing_params_path).exists():
        try:
            with open(spacing_params_path, 'r') as f:
                params = json.load(f)
            target_spacing = tuple(params['target_spacing'])
            clip_lower = params.get('clip_lower', clip_lower)
            clip_upper = params.get('clip_upper', clip_upper)
            print(f"加载target_spacing: {target_spacing}")
            print(f"加载clip_range: [{clip_lower}, {clip_upper}] HU")
        except Exception as e:
            print(f"警告: 无法加载spacing参数: {e}")
    cases = _get_all_cases(data_root)
    patient_ids = [c[0] for c in cases]
    folds = _create_folds(patient_ids, num_folds=num_folds, seed=seed)
    val_ids = set(folds[fold])
    val_cases = [(pid, img_path, label_path) for pid, img_path, label_path in cases if pid in val_ids]
    print(f"\n验证集病例数: {len(val_cases)}")
    print(f"Patch size: {patch_size}")
    print(f"Step size: {step_size}")
    print(f"Num classes: {num_classes}\n")
    model.eval()
    all_dice_scores = []
    class_dice_scores = [[] for _ in range(num_classes - 1)]
    for idx, (patient_id, image_path, label_path) in enumerate(val_cases):
        print(f"[{idx + 1}/{len(val_cases)}] 处理病例: {patient_id}")
        try:
            image_nii = nib.load(str(image_path))
            image = image_nii.get_fdata().astype(np.float32)
            original_spacing = None
            if target_spacing is not None:
                nii_spacing = image_nii.header.get_zooms()[:3]
                original_spacing = (float(nii_spacing[2]), float(nii_spacing[1]), float(nii_spacing[0]))
                print(f"  原始spacing: {original_spacing}")
            image = np.transpose(image, (2, 0, 1))
            print(f"  图像shape: {image.shape} (z, x, y)")
            print(f"  原始NIfTI shape: {image_nii.shape} (x, y, z)")
            prediction = pp.sliding_window_inference(
                image=image, model=model, patch_size=patch_size,
                num_classes=num_classes, step_size=step_size, device=device,
                use_gaussian=True, original_spacing=original_spacing,
                target_spacing=target_spacing, clip_lower=clip_lower,
                clip_upper=clip_upper, normalization=normalization,
            )
            print(f"  预测shape: {prediction.shape} (z, x, y)")
            prediction = np.transpose(prediction, (1, 2, 0))
            print(f"  转换后shape: {prediction.shape} (x, y, z)")
            if prediction.shape != image_nii.shape:
                print(f"  ⚠️  警告: 预测shape {prediction.shape} 与原始图像shape {image_nii.shape} 不匹配！")
                print(f"       这可能是由于spacing重采样导致的差异")
            pred_nii = nib.Nifti1Image(prediction, affine=image_nii.affine, header=image_nii.header)
            output_file = output_path / f"{patient_id}_pred.nii.gz"
            nib.save(pred_nii, str(output_file))
            print(f"  保存到: {output_file}")
            if compute_metrics:
                label = load_label_volume(label_path)
                dice_scores = compute_dice_per_class(prediction, label, num_classes)
                all_dice_scores.append(dice_scores['mean'])
                for c in range(num_classes - 1):
                    class_dice_scores[c].append(dice_scores['per_class'][c])
                print(f"  Dice分数:")
                print(f"    平均Dice: {dice_scores['mean']:.4f}")
                print(f"    Class 1 (CN V):        {dice_scores['per_class'][0]:.4f}")
                print(f"    Class 2 (CN V 0.5):    {dice_scores['per_class'][1]:.4f}")
                print(f"    Class 3 (CN V 0.1):    {dice_scores['per_class'][2]:.4f}")
                print(f"    Class 4 (through):     {dice_scores['per_class'][3]:.4f}")
                print(f"    Class 5 (bypass):      {dice_scores['per_class'][4]:.4f}")
                nerve_dice = np.mean([dice_scores['per_class'][i] for i in range(3)])
                print(f"    神经类(1,2,3)平均: {nerve_dice:.4f}")
            print()
        except Exception as e:
            print(f"  错误: 处理病例 {patient_id} 时出错: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    print("=" * 60)
    print(f"滑窗推理完成！所有结果已保存到: {output_path}")
    print("=" * 60)
    if compute_metrics and len(all_dice_scores) > 0:
        print("\n" + "=" * 60)
        print("验证集整体指标")
        print("=" * 60)
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        print(f"\n平均Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"\n各类别Dice：")
        class_names = ["CN V", "CN V 0.5", "CN V 0.1", "through", "bypass"]
        for c, name in enumerate(class_names):
            class_scores = [s for s in class_dice_scores[c] if not np.isnan(s)]
            if len(class_scores) > 0:
                mean_c = np.mean(class_scores)
                std_c = np.std(class_scores)
                print(f"  Class {c+1} ({name:12s}): {mean_c:.4f} ± {std_c:.4f}  (n={len(class_scores)})")
            else:
                print(f"  Class {c+1} ({name:12s}): N/A (无有效样本)")
        nerve_scores = []
        for i in range(len(all_dice_scores)):
            scores = [class_dice_scores[c][i] for c in range(3) if not np.isnan(class_dice_scores[c][i])]
            if len(scores) > 0:
                nerve_scores.append(np.mean(scores))
        if len(nerve_scores) > 0:
            mean_nerve = np.mean(nerve_scores)
            std_nerve = np.std(nerve_scores)
            print(f"\n神经类(1,2,3)平均Dice: {mean_nerve:.4f} ± {std_nerve:.4f}")
        metrics_file = output_path / "validation_metrics.csv"
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "mean_dice", "class1_CN_V", "class2_CN_V_0.5",
                             "class3_CN_V_0.1", "class4_through", "class5_bypass", "nerve_mean"])
            for i, (patient_id, _, _) in enumerate(val_cases):
                if i < len(all_dice_scores):
                    nerve_mean = np.mean([class_dice_scores[c][i] for c in range(3)
                                          if not np.isnan(class_dice_scores[c][i])])
                    writer.writerow([
                        patient_id, f"{all_dice_scores[i]:.4f}",
                        f"{class_dice_scores[0][i]:.4f}", f"{class_dice_scores[1][i]:.4f}",
                        f"{class_dice_scores[2][i]:.4f}", f"{class_dice_scores[3][i]:.4f}",
                        f"{class_dice_scores[4][i]:.4f}", f"{nerve_mean:.4f}",
                    ])
        cv_results = {
            "fold": fold, "num_cases": len(val_cases), "num_valid": len(all_dice_scores),
            "mean_dice": float(np.mean(all_dice_scores)) if all_dice_scores else 0.0,
            "std_dice": float(np.std(all_dice_scores)) if all_dice_scores else 0.0,
            "per_class_dice": {},
            "per_case_dice": {str(pid): float(s) for (pid, _, _), s in zip(val_cases, all_dice_scores)},
        }
        for c in range(num_classes - 1):
            scores_c = [s for s in class_dice_scores[c] if not np.isnan(s)]
            cv_results["per_class_dice"][f"class_{c+1}"] = {
                "mean": float(np.mean(scores_c)) if scores_c else 0.0,
                "std": float(np.std(scores_c)) if scores_c else 0.0, "n": len(scores_c),
            }
        cv_json_path = output_path / "cv_results.json"
        with open(cv_json_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"\n指标已保存到: {metrics_file}, {cv_json_path}")
        print("=" * 60 + "\n")
    else:
        print("\n")
    if compute_metrics and len(all_dice_scores) > 0:
        return float(np.mean(all_dice_scores))
    return None


__all__ = [
    "compute_dice_per_class", "train_supervised_epoch", "evaluate_supervised_epoch",
    "validate_with_sliding_window",
]
