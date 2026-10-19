import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Tuple

# 确保可以从项目根目录导入模块，无论从哪里运行该脚本
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    import nrrd as _nrrd
except ImportError:
    _nrrd = None

from npc_data import DATA_ROOT_DEFAULT, get_train_val_datasets, _get_all_cases, _create_folds
from npc_losses import BCEDiceLoss, WeightedBCEDiceLoss, multiclass_dice_score
from npc_model import UNet3D
import npc_preprocess as pp

# Fast cached dataset (reuses preprocessed 31GB cache)
try:
    from npc_data_cached import CachedSOTADataset, DEFAULT_CACHE_DIR as _CACHE_DIR
    HAS_CACHED_DATASET = True
except ImportError:
    HAS_CACHED_DATASET = False
    _CACHE_DIR = None


# ==================== Config ====================
RESULTS_ROOT = os.environ.get("CNV_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
EXPERIMENT_NAME = "npc_baseline"
PATCH_SIZE: Tuple[int, int, int] = (16, 196, 196)
BATCH_SIZE = 2
NUM_CLASSES = 6
NUM_FOLDS = 5
EARLY_STOP_PATIENCE = 50  # Stop if no val Dice improvement for 50 epochs

# AMP configuration: bfloat16 for numerical stability (no GradScaler needed)
# bf16 has same exponent range as fp32, preventing attention score overflow
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
    USE_GRAD_SCALER = False
else:
    AMP_DTYPE = torch.float16
    USE_GRAD_SCALER = True

# 对于小目标神经分割,需要特制训练策略:
# 1. 类别权重: 根据重要性设置不同权重
# 2. Focal Loss: 处理小目标和类别不平衡
# 3. 高前景采样率: 确保更多神经结构被采样


def compute_dice_per_class(pred: np.ndarray, label: np.ndarray, num_classes: int, smooth: float = 1e-5):
    """
    计算每个类别的Dice分数
    
    Args:
        pred: 预测结果 (H, W, D)，值为类别索引
        label: 真实标签 (H, W, D)，值为类别索引
        num_classes: 类别数（包括背景）
        smooth: 平滑项
        
    Returns:
        包含mean和per_class的字典
    """
    dice_scores = []
    
    # 跳过背景类（class 0）
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        label_c = (label == c).astype(np.float32)
        
        intersection = np.sum(pred_c * label_c)
        union = np.sum(pred_c) + np.sum(label_c)
        
        if union < smooth:
            # 如果该类别在预测和标签中都不存在，则dice为NaN
            dice = np.nan
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
        
        dice_scores.append(dice)
    
    # 计算平均Dice（忽略NaN值）
    valid_scores = [s for s in dice_scores if not np.isnan(s)]
    mean_dice = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
    
    return {
        'mean': mean_dice,
        'per_class': dice_scores,  # 长度为num_classes-1
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    use_amp = (AMP_DTYPE is not None)

    for batch_idx, (images, labels) in enumerate(loader):
        try:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)

            # NaN/Inf detection
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
                # Handle models that return lists (e.g., deep supervision outputs)
                logits_for_dice = logits[0] if isinstance(logits, (list, tuple)) else logits
                dice = multiclass_dice_score(logits_for_dice.float(), labels, include_background=False)

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


def evaluate(model, loader, loss_fn, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    use_amp = (AMP_DTYPE is not None)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)

            # Handle models that return lists (e.g., deep supervision outputs)
            logits_for_dice = logits[0] if isinstance(logits, (list, tuple)) else logits
            dice = multiclass_dice_score(logits_for_dice.float(), labels, include_background=False)

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
    normalization: str = 'ct',  # 'ct' (z-score) or 'percentile' (1-99% -> [0,1])
):
    """
    使用滑窗推理对验证集进行完整预测并保存结果
    
    Args:
        model: 训练好的模型
        data_root: 数据根目录
        fold: 验证折数
        output_dir: 输出目录
        patch_size: patch大小
        num_classes: 类别数
        device: 设备
        spacing_params_path: spacing参数文件路径
        step_size: 滑窗步长比例
        num_folds: 总折数
        seed: 随机种子
        compute_metrics: 是否计算Dice指标
    """
    print("\n" + "=" * 60)
    print("开始滑窗推理验证集...")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")
    
    # 加载spacing参数
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
    
    # 获取验证集病例
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
    
    # 用于存储指标
    all_dice_scores = []
    class_dice_scores = [[] for _ in range(num_classes - 1)]  # 不包括背景
    
    # 对每个病例进行推理
    for idx, (patient_id, image_path, label_path) in enumerate(val_cases):
        print(f"[{idx + 1}/{len(val_cases)}] 处理病例: {patient_id}")
        
        try:
            # 加载图像
            image_nii = nib.load(str(image_path))
            image = image_nii.get_fdata().astype(np.float32)
            
            # 获取原始spacing
            original_spacing = None
            if target_spacing is not None:
                nii_spacing = image_nii.header.get_zooms()[:3]  # (x, y, z)
                original_spacing = (float(nii_spacing[2]), float(nii_spacing[1]), float(nii_spacing[0]))  # (z, y, x)
                print(f"  原始spacing: {original_spacing}")
            
            # 转换维度顺序 (x, y, z) -> (z, x, y)
            # 注意: transpose(2, 0, 1) 产生的是 (z, x, y) 而非 (z, y, x)
            image = np.transpose(image, (2, 0, 1))
            print(f"  图像shape: {image.shape} (z, x, y)")
            print(f"  原始NIfTI shape: {image_nii.shape} (x, y, z)")
            
            # 滑窗推理
            prediction = pp.sliding_window_inference(
                image=image,
                model=model,
                patch_size=patch_size,
                num_classes=num_classes,
                step_size=step_size,
                device=device,
                use_gaussian=True,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                normalization=normalization,
            )
            
            print(f"  预测shape: {prediction.shape} (z, x, y)")
            
            # 转换回 (x, y, z) 顺序，与NIfTI格式一致
            # 从 (z, x, y) -> (x, y, z)，正确的转换是 transpose(1, 2, 0)
            prediction = np.transpose(prediction, (1, 2, 0))
            print(f"  转换后shape: {prediction.shape} (x, y, z)")
            
            # 验证shape是否与原始图像一致
            if prediction.shape != image_nii.shape:
                print(f"  ⚠️  警告: 预测shape {prediction.shape} 与原始图像shape {image_nii.shape} 不匹配！")
                print(f"       这可能是由于spacing重采样导致的差异")
            
            # 保存为NIfTI格式
            pred_nii = nib.Nifti1Image(prediction, affine=image_nii.affine, header=image_nii.header)
            
            # 保存文件
            output_file = output_path / f"{patient_id}_pred.nii.gz"
            nib.save(pred_nii, str(output_file))
            print(f"  保存到: {output_file}")
            
            # 计算Dice指标
            if compute_metrics:
                # 加载label (supports NIfTI and NRRD)
                if str(label_path).endswith('.nrrd') and _nrrd is not None:
                    _data, _ = _nrrd.read(str(label_path))
                    if _data.ndim == 4:
                        label = np.zeros(_data.shape[1:], dtype=np.int16)
                        for _ci in range(_data.shape[0]):
                            label[_data[_ci] > 0] = _ci + 1
                        label = np.clip(label, 0, 5)
                    else:
                        label = _data.astype(np.int16)
                else:
                    label_nii = nib.load(str(label_path))
                    label = label_nii.get_fdata().astype(np.int16)
                
                # 计算Dice
                dice_scores = compute_dice_per_class(prediction, label, num_classes)
                
                # 存储结果
                all_dice_scores.append(dice_scores['mean'])
                for c in range(num_classes - 1):
                    class_dice_scores[c].append(dice_scores['per_class'][c])
                
                # 打印当前病例的Dice
                print(f"  Dice分数:")
                print(f"    平均Dice: {dice_scores['mean']:.4f}")
                print(f"    Class 1 (CN V):        {dice_scores['per_class'][0]:.4f}")
                print(f"    Class 2 (CN V 0.5):    {dice_scores['per_class'][1]:.4f}")
                print(f"    Class 3 (CN V 0.1):    {dice_scores['per_class'][2]:.4f}")
                print(f"    Class 4 (through):     {dice_scores['per_class'][3]:.4f}")
                print(f"    Class 5 (bypass):      {dice_scores['per_class'][4]:.4f}")
                
                # 计算神经类(1,2,3)的平均Dice
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
    
    # 计算并显示总体指标
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
        
        # 神经类(1,2,3)的整体指标
        nerve_scores = []
        for i in range(len(all_dice_scores)):
            scores = [class_dice_scores[c][i] for c in range(3) if not np.isnan(class_dice_scores[c][i])]
            if len(scores) > 0:
                nerve_scores.append(np.mean(scores))
        
        if len(nerve_scores) > 0:
            mean_nerve = np.mean(nerve_scores)
            std_nerve = np.std(nerve_scores)
            print(f"\n神经类(1,2,3)平均Dice: {mean_nerve:.4f} ± {std_nerve:.4f}")
        
        # 保存指标到CSV + JSON
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
                        patient_id,
                        f"{all_dice_scores[i]:.4f}",
                        f"{class_dice_scores[0][i]:.4f}",
                        f"{class_dice_scores[1][i]:.4f}",
                        f"{class_dice_scores[2][i]:.4f}",
                        f"{class_dice_scores[3][i]:.4f}",
                        f"{class_dice_scores[4][i]:.4f}",
                        f"{nerve_mean:.4f}",
                    ])
        
        # Save cv_results.json with aggregated per-class metrics
        cv_results = {
            "fold": fold,
            "num_cases": len(val_cases),
            "num_valid": len(all_dice_scores),
            "mean_dice": float(np.mean(all_dice_scores)) if all_dice_scores else 0.0,
            "std_dice": float(np.std(all_dice_scores)) if all_dice_scores else 0.0,
            "per_class_dice": {},
            "per_case_dice": {str(pid): float(s) for (pid, _, _), s in zip(val_cases, all_dice_scores)},
        }
        for c in range(num_classes - 1):
            scores_c = [s for s in class_dice_scores[c] if not np.isnan(s)]
            cv_results["per_class_dice"][f"class_{c+1}"] = {
                "mean": float(np.mean(scores_c)) if scores_c else 0.0,
                "std": float(np.std(scores_c)) if scores_c else 0.0,
                "n": len(scores_c),
            }
        cv_json_path = output_path / "cv_results.json"
        with open(cv_json_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"\n指标已保存到: {metrics_file}, {cv_json_path}")
        print("=" * 60 + "\n")
    else:
        print("\n")

    # Return the headline metric so callers (e.g. best-model selection in
    # npc_sota_train) can use full-volume sliding-window Dice as the criterion.
    if compute_metrics and len(all_dice_scores) > 0:
        return float(np.mean(all_dice_scores))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="NPC baseline 3D nnUNet-like training")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for 5-fold cross validation (0-4)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (推荐500+用于极小目标)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (降低初始LR)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--spacing_params", type=str, default=None, help="Path to spacing parameters JSON file")
    parser.add_argument("--step_size", type=float, default=0.5, help="Sliding window step size ratio")
    parser.add_argument("--skip_sliding_window", action="store_true", help="Skip sliding window validation at the end")
    parser.add_argument("--gpu", type=str, default="0", help="Specify GPU device(s), e.g., '0' or '0,1'. Default uses all available GPUs.")
    parser.add_argument("--val_only", action="store_true", help="Only perform validation with sliding window inference (skip training)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint for validation (default: best_model.pth in fold directory)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    args = parser.parse_args()

    if args.fold < 0 or args.fold >= NUM_FOLDS:
        raise ValueError(f"fold must be in [0, {NUM_FOLDS - 1}], got {args.fold}")

    # 设置GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"使用GPU: {args.gpu}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 加载spacing参数（如果提供）
    target_spacing = None
    clip_lower = -200.0
    clip_upper = 300.0
    if args.spacing_params and Path(args.spacing_params).exists():
        try:
            with open(args.spacing_params, 'r') as f:
                params = json.load(f)
            target_spacing = tuple(params['target_spacing'])
            clip_lower = params.get('clip_lower', clip_lower)
            clip_upper = params.get('clip_upper', clip_upper)
            print(f"使用target_spacing: {target_spacing}")
            print(f"使用clip_range: [{clip_lower}, {clip_upper}] HU")
        except Exception as e:
            print(f"警告: 无法加载spacing参数: {e}")

    # Use cached dataset if available (3-5x faster data loading)
    cache_dir = _CACHE_DIR
    use_cache = HAS_CACHED_DATASET and Path(cache_dir).exists() if cache_dir else False
    normalization = 'percentile' if use_cache else 'ct'  # Match training normalization
    if use_cache:
        from npc_data_cached import CachedSOTADataset
        cases = _get_all_cases(DATA_ROOT_DEFAULT)
        patient_ids = [c[0] for c in cases]
        folds = _create_folds(patient_ids, num_folds=NUM_FOLDS, seed=42)
        val_ids = set(folds[args.fold])
        train_cases = [(p, i, l) for p, i, l in cases if p not in val_ids]
        val_cases = [(p, i, l) for p, i, l in cases if p in val_ids]
        train_dataset = CachedSOTADataset(train_cases, patch_size=PATCH_SIZE,
                                          cache_dir=cache_dir, training=True)
        val_dataset = CachedSOTADataset(val_cases, patch_size=PATCH_SIZE,
                                        cache_dir=cache_dir, training=False)
        print(f"Using CACHED dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    else:
        train_dataset, val_dataset, folds = get_train_val_datasets(
            DATA_ROOT_DEFAULT,
            patch_size=PATCH_SIZE,
            fold=args.fold,
            num_folds=NUM_FOLDS,
            seed=42,
            target_spacing=target_spacing,
            ensure_foreground=True,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        print(f"Using NPCDataset (on-the-fly): train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = UNet3D(in_channels=1, num_classes=NUM_CLASSES, base_channels=48).to(device)  # 增大base_channels
    
    exp_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_NAME)
    fold_dir = os.path.join(exp_dir, f"fold{args.fold}")
    os.makedirs(fold_dir, exist_ok=True)

    best_model_path = os.path.join(fold_dir, "best_model.pth")
    last_model_path = os.path.join(fold_dir, "last_model.pth")
    
    # 仅验证模式：加载已有模型进行滑窗推理
    if args.val_only:
        print("\n" + "=" * 70)
        print("仅验证模式：加载模型进行滑窗推理")
        print("=" * 70)
        
        # 确定模型路径
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = best_model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 兼容不同的checkpoint格式
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"加载模型 (Epoch: {epoch})")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # 统一使用 val_predictions_best 目录
        val_output_dir = os.path.join(fold_dir, "val_predictions_best")
        
        # 进行滑窗推理
        validate_with_sliding_window(
            model=model,
            data_root=DATA_ROOT_DEFAULT,
            fold=args.fold,
            output_dir=val_output_dir,
            patch_size=PATCH_SIZE,
            num_classes=NUM_CLASSES,
            device=str(device),
            spacing_params_path=args.spacing_params,
            step_size=args.step_size,
            num_folds=NUM_FOLDS,
            seed=42, normalization=normalization,
        )
        
        print("\n" + "=" * 70)
        print("验证完成！")
        print("=" * 70)
        return
    
    # 训练模式
    print("\n" + "=" * 70)
    print("训练模式")
    print("=" * 70)
    
    # 使用带类别权重的Focal Loss + Dice Loss
    # 类别权重: [背景, CN V, CN V 0.5, CN V 0.1, through, bypass]
    # 针对极度不平衡数据(前景仅0.027%),使用激进权重
    loss_fn = WeightedBCEDiceLoss(
        weight_bce=0.3,  # 降低BCE权重
        weight_dice=0.7,  # 提高Dice权重(更关注前景重叠)
        include_background=False,
        class_weights=torch.tensor([0.1, 10.0, 8.0, 6.0, 3.0, 3.0]),  # 激进权重
        focal_alpha=0.25,
        focal_gamma=3.0,  # 提高gamma,更关注难样本
        use_focal=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 使用学习率调度器 - CosineAnnealingLR有助于收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01  # 最小学习率为初始值的1%
    )

    log_path = os.path.join(fold_dir, "train_log.csv")
    best_val_dice = 0.0
    best_epoch = 0
    no_improve = 0

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "lr"])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_dice = evaluate(model, val_loader, loss_fn, device)
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice, current_lr])
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | "
                f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f} | "
                f"lr={current_lr:.6f}"
            )

            # 保存最佳模型
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                no_improve = 0
                torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_model_path)
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n早停: epoch {epoch} (连续{no_improve}个epoch无提升)")
                break

    # 保存最后一个epoch的模型
    torch.save({"epoch": args.epochs, "model_state": model.state_dict()}, last_model_path)
    print(f"最后模型已保存到: {last_model_path}")
    
    print(f"\n训练完成！最佳验证Dice: {best_val_dice:.4f}")
    
    # 训练结束后，对验证集进行完整的滑窗推理
    if not args.skip_sliding_window:
        print("\n加载最佳模型进行滑窗推理验证...")
        
        # 加载最佳模型
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"加载最佳模型 (Epoch {checkpoint['epoch']})")
        
        # 统一使用 val_predictions_best 目录
        val_output_dir = os.path.join(fold_dir, "val_predictions_best")
        
        # 进行滑窗推理
        validate_with_sliding_window(
            model=model,
            data_root=DATA_ROOT_DEFAULT,
            fold=args.fold,
            output_dir=val_output_dir,
            patch_size=PATCH_SIZE,
            num_classes=NUM_CLASSES,
            device=str(device),
            spacing_params_path=args.spacing_params,
            step_size=args.step_size,
            num_folds=NUM_FOLDS,
            seed=42,
        )

        # Copy cv_results.json to fold directory
        import shutil
        cv_src = Path(val_output_dir) / "cv_results.json"
        cv_dst = Path(fold_dir) / "cv_results.json"
        if cv_src.exists():
            shutil.copy(cv_src, cv_dst)
            print(f"cv_results.json saved to: {cv_dst}")

    # Write COMPLETED marker
    completed_marker = Path(fold_dir) / "COMPLETED"
    completed_marker.write_text(f"Training completed at {__import__('datetime').datetime.now().isoformat()}\n"
                                f"Best val_dice: {best_val_dice:.4f}\n"
                                f"Best epoch: {best_epoch}\n")
    print(f"COMPLETED marker written: {completed_marker}")


if __name__ == "__main__":
    main()
