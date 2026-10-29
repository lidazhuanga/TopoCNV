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

from topocnv.config.loader import load_json
from topocnv.data.dataset import DATA_ROOT_DEFAULT, get_train_val_datasets, _get_all_cases, _create_folds
from topocnv.losses import BCEDiceLoss, WeightedBCEDiceLoss
from topocnv.engine.segmentation import train_supervised_epoch, evaluate_supervised_epoch, validate_with_sliding_window, compute_dice_per_class, multiclass_dice_score
from topocnv.models.unet3d import UNet3D
from topocnv.data import preprocess as pp
from topocnv.engine.amp import legacy_amp_settings

# Fast cached dataset (reuses preprocessed 31GB cache)
try:
    from topocnv.data.dataset import CachedSOTADataset
    from topocnv.data.utils import DEFAULT_CACHE_DIR as _CACHE_DIR
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
AMP_DTYPE, USE_GRAD_SCALER = legacy_amp_settings()

# 对于小目标神经分割,需要特制训练策略:
# 1. 类别权重: 根据重要性设置不同权重
# 2. Focal Loss: 处理小目标和类别不平衡
# 3. 高前景采样率: 确保更多神经结构被采样




def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None) -> Tuple[float, float]:
    return train_supervised_epoch(model, loader, optimizer, loss_fn, device, scaler=scaler, amp_dtype=AMP_DTYPE, dice_fn=multiclass_dice_score)


def evaluate(model, loader, loss_fn, device) -> Tuple[float, float]:
    return evaluate_supervised_epoch(model, loader, loss_fn, device, amp_dtype=AMP_DTYPE, dice_fn=multiclass_dice_score)




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
            params = load_json(args.spacing_params)
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
        from topocnv.data.dataset import CachedSOTADataset
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
