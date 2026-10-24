"""
Unified SOTA Comparison Training — CNV protocol v0.3 / methods §14.

CNV protocol v0.3

External baselines (nnU-Net and peers) share the **B0 endpoint**:
  single-stage **binary Y** = 1∪2∪3∪4 (class5_bypass → background).

Primary: Dice_Y (+ HD95_Y offline via metrics_protocol).
FORBIDDEN Primary: 5-class mean Dice / soft-tree mean.

Models: nnunet | segresnet | attention_unet | unetr | swin_unetr | nnformer

Training, full-volume validation and OOF labels all live on the preprocess cache grid
(same cache, folds and Y labels as P0). With --select_by_sliding_window the best checkpoint
is the one with the highest full-volume Dice_Y (checked every --full_val_every epochs).
Outputs: results/sota_binary_y_<model>/fold<k>/ and OOF labels
results/sota_binary_y_<model>_label_predictions/<case>_stage2_label.nrrd (P0 format).

Usage (repo root; tools/sota_queue.py runs all models x folds):
  python trainer/npc_sota_train.py --model nnunet --fold 0 --select_by_sliding_window \
      --random_crop_prob 0.33
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
try:
    import nrrd as _nrrd
except ImportError:
    _nrrd = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "MIA" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from metrics_protocol import DEFAULT_SPACING_ZYX, _dice, _hd95, mask_Y
from npc_data import DATA_ROOT_DEFAULT, get_train_val_datasets, _get_all_cases, _create_folds
from npc_losses import WeightedBCEDiceLoss, multiclass_dice_score
import npc_preprocess as pp

# Reuse training loop and sliding-window validation from npc_baseline
from npc_baseline import (
    train_one_epoch,
    evaluate,
    validate_with_sliding_window,
    compute_dice_per_class,
    AMP_DTYPE,
    USE_GRAD_SCALER,
)

# Fast cached dataset (reuses preprocessed 31GB cache)
try:
    from npc_data_cached import CachedSOTADataset, DEFAULT_CACHE_DIR
    HAS_CACHED_DATASET = True
except ImportError:
    HAS_CACHED_DATASET = False
    DEFAULT_CACHE_DIR = None

# MONAI architectures
from monai.networks.nets import DynUNet, SegResNet, AttentionUnet, UNETR, SwinUNETR


# ==================== Config ====================
RESULTS_ROOT = os.environ.get("CNV_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
EXPERIMENT_PREFIX = "sota_binary_y"  # legacy 6-class runs: npc_sota_<model>
PATCH_SIZE: Tuple[int, int, int] = (128, 128, 128)
BATCH_SIZE = 2
NUM_CLASSES = 2  # BINARY_Y_ENDPOINT: bg + Y(1∪2∪3∪4); class5_bypass→bg
BINARY_Y_ENDPOINT = True  # protocol §14 / = B0
NUM_FOLDS = 5
EARLY_STOP_PATIENCE = 50  # Stop if no val Dice improvement for 50 epochs
VALIDATE_EVERY_N = 5  # Validate on patches every N epochs (was every epoch)
FULL_VAL_EVERY_N = 25  # Full sliding-window validation every N epochs
# With --select_by_sliding_window, no_improve only increments at full-val
# checks (every FULL_VAL_EVERY_N epochs), so patience is counted in checks:
# 6 checks = 150 epochs without full-val improvement.
SW_EARLY_STOP_CHECKS = 6
SPACING_PARAMS_PATH = str(PROJECT_ROOT / "spacing_params.json")

SUPPORTED_MODELS = [
    "nnunet",
    "segresnet",
    "attention_unet",
    "unetr",
    "swin_unetr",
    "nnformer",
]


# ==================== nnFormer Implementation ====================
class _ConvTransformerBlock(nn.Module):
    """Interleaved convolution + transformer block (nnFormer style)."""

    def __init__(self, in_ch: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.window_size = window_size

        # Convolution branch
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.GELU(),
        )

        # Local-window self-attention
        self.norm1 = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_ch, num_heads=num_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(in_ch)
        hidden = int(in_ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # Conv branch
        x = x + self.conv(x)

        # Pad to window boundaries
        wd, wh, ww = self.window_size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]

        # Window partition
        nD, nH, nW = Dp // wd, Hp // wh, Wp // ww
        # (B, C, nD*wd, nH*wh, nW*ww) -> (B*nD*nH*nW, C, wd, wh, ww)
        x = x.view(B, C, nD, wd, nH, wh, nW, ww)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(-1, C, wd * wh * ww)  # (num_windows*B, C, window_tokens)
        x = x.transpose(1, 2)  # (num_windows*B, window_tokens, C)

        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        # MLP
        x = x + self.mlp(self.norm2(x))

        # Reverse window partition
        x = x.transpose(1, 2)  # (num_windows*B, C, window_tokens)
        x = x.view(B, nD, nH, nW, C, wd, wh, ww)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, Dp, Hp, Wp)

        # Remove padding
        if pad_d or pad_h or pad_w:
            x = x[:, :, :D, :H, :W]

        return x


class _nnFormerEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_heads: int = 4):
        super().__init__()
        self.conv_trans = _ConvTransformerBlock(out_ch, num_heads=num_heads)
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv_trans(x)
        return x


class _nnFormerDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_heads: int = 4):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if shapes don't match exactly
        if x.shape[2:] != skip.shape[2:]:
            diff = [s2 - s1 for s1, s2 in zip(x.shape[2:], skip.shape[2:])]
            x = F.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                          diff[1] // 2, diff[1] - diff[1] // 2,
                          diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class NNFormer3D(nn.Module):
    """Simplified nnFormer: interleaved Conv+Transformer encoder, U-Net decoder.

    Reference: Zhou et al., "nnFormer: Volumetric Medical Image Segmentation via
    a 3D Transformer", IEEE TIP 2023.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 32):
        super().__init__()
        c = base_channels
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.GELU(),
        )
        # Encoder
        self.enc1 = _nnFormerEncoderBlock(c, c * 2, num_heads=2)
        self.enc2 = _nnFormerEncoderBlock(c * 2, c * 4, num_heads=4)
        self.enc3 = _nnFormerEncoderBlock(c * 4, c * 8, num_heads=4)
        # Bottleneck
        self.bottleneck = _ConvTransformerBlock(c * 8, num_heads=4)
        # Decoder (skip from encoder at matching resolution)
        self.dec3 = _nnFormerDecoderBlock(c * 8, c * 4, c * 4, num_heads=4)
        self.dec2 = _nnFormerDecoderBlock(c * 4, c * 2, c * 2, num_heads=4)
        self.dec1 = _nnFormerDecoderBlock(c * 2, c, c, num_heads=2)
        # Output
        self.out_conv = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)       # full res, c channels
        x1 = self.enc1(x0)      # 1/2 res, c*2 channels
        x2 = self.enc2(x1)      # 1/4 res, c*4 channels
        x3 = self.enc3(x2)      # 1/8 res, c*8 channels
        b = self.bottleneck(x3)  # 1/8 res, c*8 channels
        d3 = self.dec3(b, x2)   # up: 1/8->1/4, skip: x2 at 1/4
        d2 = self.dec2(d3, x1)  # up: 1/4->1/2, skip: x1 at 1/2
        d1 = self.dec1(d2, x0)  # up: 1/2->1/1, skip: x0 at 1/1
        return self.out_conv(d1)


# ==================== Model Factory ====================

def convert_label_to_binary_Y(label):
    """Map original labels {0..5} → {0,1}; Y=1∪2∪3∪4; class5_bypass→0."""
    lab = np.asarray(label)
    return ((lab >= 1) & (lab <= 4)).astype(np.int64)


def _load_label_volume(label_path):
    """GT label in NIfTI (x, y, z) order; supports NIfTI and NRRD (4D one-hot -> ids)."""
    if str(label_path).endswith('.nrrd') and _nrrd is not None:
        data, _ = _nrrd.read(str(label_path))
        if data.ndim == 4:
            label = np.zeros(data.shape[1:], dtype=np.int16)
            for ci in range(data.shape[0]):
                label[data[ci] > 0] = ci + 1
            return np.clip(label, 0, 5)
        return data.astype(np.int16)
    return nib.load(str(label_path)).get_fdata().astype(np.int16)


def _save_label_nrrd(label: np.ndarray, save_path: str):
    """Integer label map as gzip NRRD (int16) — same format as the P0 OOF labels."""
    header = {'type': 'short', 'dimension': 3, 'sizes': list(label.shape),
              'kinds': ['domain', 'domain', 'domain'], 'encoding': 'gzip', 'endian': 'little'}
    _nrrd.write(str(save_path), label.astype(np.int16), header)


def validate_binary_y_cache_sw(model, val_ids, cache_dir, output_dir, patch_size, device,
                               fold, step_size=0.5, label_dir=None, with_hd95=True):
    """Full-volume sliding-window validation on the protocol Primary endpoint, in cache space.

    Runs on the preprocessed cache (`{case}_image.npy`, the grid and intensities the model
    was trained on and the grid P0 and eval_protocol_metrics.py score on). Gaussian-weighted
    windows with overlap = 1 - step_size (sigma = patch/8), as in the P0 Stage-2 driver;
    Y = softmax foreground > 0.5. Scores Dice_Y and HD95_Y (mm) against cache GT
    Y = 1∪2∪3∪4 (class5_bypass -> background). With label_dir, saves the OOF labels as
    `{case}_stage2_label.nrrd`. Writes per-case CSV and cv_results.json; returns mean Dice_Y.
    """
    from monai.inferers import sliding_window_inference
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if label_dir is not None:
        Path(label_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    t0 = time.time()
    for case_id in sorted(val_ids):
        image = np.load(Path(cache_dir) / f"{case_id}_image.npy")
        gt_y = mask_Y(np.load(Path(cache_dir) / f"{case_id}_label.npy"))
        x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(device)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE,
                                                 enabled=AMP_DTYPE is not None):
            logits = sliding_window_inference(
                x, roi_size=patch_size, sw_batch_size=2, predictor=model,
                overlap=1.0 - step_size, mode="gaussian", sigma_scale=0.125)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob_y = torch.softmax(logits.float(), dim=1)[0, 1]
        pred_y = (prob_y > 0.5).cpu().numpy()
        del x, logits, prob_y
        if label_dir is not None:
            _save_label_nrrd(pred_y.astype(np.int16), str(Path(label_dir) / f"{case_id}_stage2_label.nrrd"))
        row = {"patient_id": case_id, "Dice_Y": _dice(pred_y, gt_y),
               "HD95_Y": _hd95(pred_y, gt_y, DEFAULT_SPACING_ZYX) if with_hd95 else float("nan")}
        rows.append(row)
        print(f"  {case_id}: Dice_Y={row['Dice_Y']:.4f} HD95_Y={row['HD95_Y']:.2f} mm", flush=True)
    torch.cuda.empty_cache()

    if not rows:
        return None
    mean_dice = float(np.nanmean([r["Dice_Y"] for r in rows]))
    mean_hd95 = float(np.nanmean([r["HD95_Y"] for r in rows]))
    with open(out_dir / "validation_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "Dice_Y", "HD95_Y"])
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump({"fold": fold, "endpoint": "Y=1∪2∪3∪4", "space": "preprocess cache",
                   "num_cases": len(val_ids), "num_valid": len(rows), "mean_dice": mean_dice,
                   "Dice_Y": mean_dice, "HD95_Y": mean_hd95}, f, indent=2)
    print(f"[SOTA] fold {fold}: Dice_Y={mean_dice:.4f} HD95_Y={mean_hd95:.2f} mm (n={len(rows)}, "
          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    return mean_dice


def protocol_primary_dice(pred, gt):
    """Dice_Y only — never 5-class mean."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    p = pred.astype(bool) if pred.max() <= 1 else (pred > 0)
    g = gt.astype(bool) if gt.max() <= 1 else ((gt >= 1) & (gt <= 4))
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    return float(2.0 * inter / denom) if denom > 0 else float("nan")


def build_model(model_name: str, in_channels: int = 1,
                num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create a model instance by name."""
    if model_name == "nnunet":
        # DynUNet configured like nnU-Net's 3D full-resolution
        return DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=[[3, 3, 3]] * 6,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 5,
            filters=[32, 64, 128, 256, 320, 320],
            res_block=True,
        )
    elif model_name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.0,
        )
    elif model_name == "attention_unet":
        return AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
        )
    elif model_name == "unetr":
        return UNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.1,
            spatial_dims=3,
        )
    elif model_name == "swin_unetr":
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
            use_checkpoint=False,
            spatial_dims=3,
        )
    elif model_name == "nnformer":
        return NNFormer3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=32,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}"
        )


def get_experiment_name(model_name: str) -> str:
    if not BINARY_Y_ENDPOINT:
        return f"npc_sota_{model_name}"
    return f"{EXPERIMENT_PREFIX}_{model_name}"


# ==================== Main ====================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SOTA comparison training for NPC nerve segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--legacy_multiclass", type=int, default=0,
                        help="1=old 6-class (FORBIDDEN as Primary; appendix only)")
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS,
                        help="Model architecture to train")
    parser.add_argument("--fold", type=int, default=0,
                        help=f"Fold index for {NUM_FOLDS}-fold CV (0-{NUM_FOLDS-1})")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--spacing_params", type=str, default=SPACING_PARAMS_PATH,
                        help="Path to spacing parameters JSON")
    parser.add_argument("--step_size", type=float, default=0.5,
                        help="Sliding window step size ratio")
    parser.add_argument("--skip_sliding_window", action="store_true",
                        help="Skip sliding window validation at end")
    parser.add_argument("--random_crop_prob", type=float, default=0.0,
                        help="Probability of a fully random (possibly all-background) training crop; "
                             "nnU-Net-style mixed sampling, e.g. 0.33. Default 0.0 = always foreground-biased.")
    parser.add_argument("--full_val_every", type=int, default=FULL_VAL_EVERY_N,
                        help="Periodic full-volume sliding-window validation every N epochs; "
                             "0 disables it (patch-level validation only).")
    parser.add_argument("--select_by_sliding_window", action="store_true",
                        help="Select the best model by periodic full-volume sliding-window Dice "
                             "(evaluated on the CURRENT model every FULL_VAL_EVERY_N epochs) instead of "
                             "center-crop patch Dice. Recommended when patch Dice and full-volume "
                             "performance diverge (e.g. false positives outside the crop).")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device id (e.g. '0' or '1')")
    parser.add_argument("--val_only", action="store_true",
                        help="Only run sliding-window validation (skip training)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Checkpoint path for val_only mode")
    parser.add_argument("--use_cache", type=int, default=1,
                        help="Use preprocessed cache (1=yes, 0=no). Default: 1")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Path to preprocessed cache directory")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from fold*/latest_checkpoint.pth (written at every full-val check)")
    args = parser.parse_args()
    global NUM_CLASSES, BINARY_Y_ENDPOINT
    if int(getattr(args, 'legacy_multiclass', 0)):
        NUM_CLASSES = 6
        BINARY_Y_ENDPOINT = False
        print('[WARN] legacy_multiclass=1 — NOT protocol Primary')
    else:
        NUM_CLASSES = 2
        BINARY_Y_ENDPOINT = True
        print('[SOTA] binary Y endpoint (=B0); Primary=Dice_Y')

    if args.fold < 0 or args.fold >= NUM_FOLDS:
        raise ValueError(f"fold must be in [0, {NUM_FOLDS-1}], got {args.fold}")

    # Set GPU before torch import (already imported, but set env for subprocess consistency)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"GPU: {args.gpu}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load spacing parameters
    target_spacing = None
    clip_lower = -200.0
    clip_upper = 300.0
    if args.spacing_params and Path(args.spacing_params).exists():
        with open(args.spacing_params) as f:
            params = json.load(f)
        target_spacing = tuple(params["target_spacing"])
        clip_lower = params.get("clip_lower", clip_lower)
        clip_upper = params.get("clip_upper", clip_upper)
        print(f"Target spacing: {target_spacing}")
        print(f"Clip range: [{clip_lower}, {clip_upper}] HU")

    # Build model
    exp_name = get_experiment_name(args.model)
    print(f"\n{'='*70}")
    print(f"Model: {args.model}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")


    class _BinaryYWrap(torch.utils.data.Dataset):
        """Remap multiclass labels → binary Y for protocol Primary."""
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            item = self.base[i]
            if isinstance(item, dict):
                item = dict(item)
                for key in ("label", "label_meta"):
                    if key in item and item[key] is not None:
                        lab = item[key]
                        lab_np = lab.numpy() if torch.is_tensor(lab) else np.asarray(lab)
                        item[key] = torch.as_tensor(convert_label_to_binary_Y(lab_np))
                return item
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                img, lab = item[0], item[1]
                lab_np = lab.numpy() if torch.is_tensor(lab) else np.asarray(lab)
                lab_t = torch.as_tensor(convert_label_to_binary_Y(lab_np),
                                        dtype=lab.dtype if torch.is_tensor(lab) else torch.long)
                return (img, lab_t) + tuple(item[2:])
            return item

    model = build_model(args.model, in_channels=1, num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Data — use fast cached dataset if available, fall back to on-the-fly NPCDataset
    use_cache = bool(args.use_cache) and HAS_CACHED_DATASET and Path(args.cache_dir or "").exists()
    normalization = 'percentile' if use_cache else 'ct'  # Match training normalization
    if BINARY_Y_ENDPOINT and not use_cache:
        raise RuntimeError("binary Y endpoint trains and scores on the preprocess cache; "
                           f"cache not found: {args.cache_dir}")
    cases = _get_all_cases(DATA_ROOT_DEFAULT)
    # seed-42 folds == extras/splits.json (checked 2026-10-21)
    folds = _create_folds([c[0] for c in cases], num_folds=NUM_FOLDS, seed=42)
    val_ids = set(folds[args.fold])
    if use_cache:
        from npc_data_cached import CachedSOTADataset
        train_cases = [(p, i, l) for p, i, l in cases if p not in val_ids]
        val_cases = [(p, i, l) for p, i, l in cases if p in val_ids]
        train_dataset = CachedSOTADataset(train_cases, patch_size=PATCH_SIZE,
                                          cache_dir=args.cache_dir, training=True,
                                          random_crop_prob=args.random_crop_prob)
        val_dataset = CachedSOTADataset(val_cases, patch_size=PATCH_SIZE,
                                        cache_dir=args.cache_dir, training=False)
        print(f"Using CACHED dataset ({args.cache_dir}): train={len(train_dataset)}, val={len(val_dataset)}")
    else:
        train_dataset, val_dataset, _ = get_train_val_datasets(
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
    if BINARY_Y_ENDPOINT:
        # Must wrap before building the DataLoaders: DataLoader.dataset is read-only afterwards.
        train_dataset = _BinaryYWrap(train_dataset)
        val_dataset = _BinaryYWrap(val_dataset)
        print('[SOTA] wrapped train/val datasets → binary Y')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True,
    )
    print(f"Fold {args.fold}: train={len(train_dataset)}, val={len(val_dataset)}")

    # Output directory
    exp_dir = Path(RESULTS_ROOT) / exp_name / f"fold{args.fold}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = exp_dir / "best_model.pth"
    last_model_path = exp_dir / "last_model.pth"
    latest_ckpt_path = exp_dir / "latest_checkpoint.pth"
    label_dir = Path(RESULTS_ROOT) / f"{exp_name}_label_predictions"  # OOF labels, all folds
    print(f"Output: {exp_dir}")

    def run_sw(output_dir, save_labels=False):
        """Full-volume sliding-window validation of the current model on this fold.

        Periodic checks select by Dice_Y only and skip HD95_Y (the distance transform on
        early, noisy predictions dominates the check time); the final best-model pass
        scores both and writes the OOF labels.
        """
        if BINARY_Y_ENDPOINT:
            return validate_binary_y_cache_sw(
                model=model, val_ids=val_ids, cache_dir=args.cache_dir, output_dir=output_dir,
                patch_size=PATCH_SIZE, device=device, fold=args.fold, step_size=args.step_size,
                label_dir=label_dir if save_labels else None, with_hd95=save_labels)
        return validate_with_sliding_window(
            model=model, data_root=DATA_ROOT_DEFAULT, fold=args.fold, output_dir=output_dir,
            patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, device=str(device),
            spacing_params_path=args.spacing_params, step_size=args.step_size,
            num_folds=NUM_FOLDS, seed=42, normalization=normalization)

    # ---- Val-only mode ----
    if args.val_only:
        if args.model_path:
            ckpt_path = args.model_path
        else:
            ckpt_path = str(best_model_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state)
        run_sw(str(exp_dir / "val_predictions_best"), save_labels=True)
        print("Validation complete.")
        return

    # ---- Training ----

    loss_fn = WeightedBCEDiceLoss(
        weight_bce=0.3,
        weight_dice=0.7,
        include_background=False,
        class_weights=(torch.tensor([0.1, 10.0]) if BINARY_Y_ENDPOINT
                       else torch.tensor([0.1, 10.0, 8.0, 6.0, 3.0, 3.0])),
        focal_alpha=0.25,
        focal_gamma=3.0,
        use_focal=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # AMP scaler (only needed for float16; bfloat16 doesn't use it)
    scaler = torch.cuda.amp.GradScaler() if USE_GRAD_SCALER else None
    print(f"AMP: dtype={AMP_DTYPE}, grad_scaler={scaler is not None}")
    print(f"Validate every {VALIDATE_EVERY_N} epochs, full val every {FULL_VAL_EVERY_N} epochs")

    # Save config
    config = {
        "model": args.model, "fold": args.fold, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size,
        "patch_size": list(PATCH_SIZE), "num_classes": NUM_CLASSES,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "random_crop_prob": args.random_crop_prob, "full_val_every": args.full_val_every,
        "select_by_sliding_window": args.select_by_sliding_window,
        "cache_dir": args.cache_dir if use_cache else None, "val_ids": sorted(val_ids),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log_path = exp_dir / "train_log.csv"
    best_val_dice = 0.0
    best_epoch = 0
    no_improve = 0
    start_epoch = 1
    if args.resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        best_val_dice, best_epoch = ckpt["best_val_dice"], ckpt["best_epoch"]
        no_improve = ckpt["no_improve"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {latest_ckpt_path} at epoch {ckpt['epoch']} "
              f"(best {best_val_dice:.4f} @ ep{best_epoch}, no_improve={no_improve})")
    elif args.resume:
        print(f"[WARN] --resume but no {latest_ckpt_path}; starting from scratch")

    def save_latest(epoch):
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "best_val_dice": best_val_dice, "best_epoch": best_epoch,
                    "no_improve": no_improve}, str(latest_ckpt_path) + ".tmp")
        os.replace(str(latest_ckpt_path) + ".tmp", latest_ckpt_path)

    print(f"\n{'='*70}")
    print(f"Training {args.model} fold {args.fold} for {args.epochs} epochs")
    print(f"  AMP: {AMP_DTYPE}, cache: {'yes' if use_cache else 'no'}, val every {VALIDATE_EVERY_N}ep")
    print(f"{'='*70}")

    with open(log_path, "a" if start_epoch > 1 else "w", newline="") as f:
        writer = csv.writer(f)
        if start_epoch == 1:
            writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "lr"])

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_dice = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler=scaler,
            )

            # Patch-level validation — only every N epochs to save time
            do_val = (epoch % VALIDATE_EVERY_N == 0) or (epoch == 1)
            if do_val:
                val_loss, val_dice = evaluate(model, val_loader, loss_fn, device)
            else:
                # Use last known val_dice for early stopping continuity
                val_loss, val_dice = float('nan'), best_val_dice

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([epoch, train_loss, train_dice,
                           val_loss if not np.isnan(val_loss) else '',
                           val_dice, current_lr])
            f.flush()

            if do_val:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} | "
                    f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | "
                    f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f} | "
                    f"lr={current_lr:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} | "
                    f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | "
                    f"lr={current_lr:.6f}  (val skipped)"
                )

            if do_val and val_dice > best_val_dice and not args.select_by_sliding_window:
                best_val_dice = val_dice
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "val_dice": val_dice, "val_loss": val_loss},
                    best_model_path,
                )
            elif do_val and not args.select_by_sliding_window:
                no_improve += 1
            # Note: no_improve only increments on validation epochs

            # Early stopping. In sliding-window selection mode no_improve
            # counts full-val checks (every FULL_VAL_EVERY_N epochs);
            # otherwise it counts patch-val epochs (every VALIDATE_EVERY_N).
            if args.select_by_sliding_window:
                early_stop_hit = no_improve >= SW_EARLY_STOP_CHECKS
            else:
                early_stop_hit = no_improve >= EARLY_STOP_PATIENCE // VALIDATE_EVERY_N
            if early_stop_hit:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {no_improve} val checks)")
                break

            # Periodic full sliding-window validation
            if args.full_val_every > 0 and epoch % args.full_val_every == 0 and epoch > 0:
                if args.select_by_sliding_window:
                    # Evaluate the CURRENT model on the full volume and use that
                    # Dice as the best-model criterion. The model is left untouched
                    # (no best-model reload), so training continues undisturbed.
                    print(f"\n  [Full val @ epoch {epoch}] Sliding-window inference on current model...")
                    sw_dice = run_sw(str(exp_dir / f"val_predictions_ep{epoch:03d}"))
                    model.train()
                    if sw_dice is not None:
                        if sw_dice > best_val_dice:
                            best_val_dice = sw_dice
                            best_epoch = epoch
                            no_improve = 0
                            torch.save(
                                {"epoch": epoch, "model_state": model.state_dict(),
                                 "val_dice": sw_dice, "val_loss": val_loss},
                                best_model_path,
                            )
                            print(f"  [Full val @ epoch {epoch}] new best: {sw_dice:.4f}")
                        else:
                            no_improve += 1
                            print(f"  [Full val @ epoch {epoch}] sw_dice={sw_dice:.4f} "
                                  f"(best {best_val_dice:.4f} @ ep{best_epoch})")
                    save_latest(epoch)
                else:
                    print(f"\n  [Full val @ epoch {epoch}] Loading best model for sliding-window inference...")
                    if best_model_path.exists():
                        ckpt = torch.load(best_model_path, map_location=device)
                        model.load_state_dict(ckpt["model_state"])
                        run_sw(str(exp_dir / f"val_predictions_ep{epoch:03d}"))

    # Save last model
    torch.save(
        {"epoch": args.epochs, "model_state": model.state_dict()},
        last_model_path,
    )
    print(f"\nTraining complete! Best val Dice: {best_val_dice:.4f}")
    print(f"Best model: {best_model_path}")

    # Sliding-window validation
    if not args.skip_sliding_window and best_model_path.exists():
        print("\nLoading best model for sliding-window inference...")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (Epoch {ckpt['epoch']})")

        val_output_dir = str(exp_dir / "val_predictions_best")
        run_sw(val_output_dir, save_labels=True)

        # Copy cv_results.json to fold directory for easy aggregation
        import shutil
        cv_src = Path(val_output_dir) / "cv_results.json"
        cv_dst = exp_dir / "cv_results.json"
        if cv_src.exists():
            shutil.copy(cv_src, cv_dst)
            print(f"cv_results.json saved to: {cv_dst}")

    # Write COMPLETED marker — indicates training + validation finished successfully
    completed_marker = exp_dir / "COMPLETED"
    completed_marker.write_text(f"Training completed at {__import__('datetime').datetime.now().isoformat()}\n"
                                f"Best val_dice: {best_val_dice:.4f}\n"
                                f"Best epoch: {best_epoch}\n")
    print(f"COMPLETED marker written: {completed_marker}")


if __name__ == "__main__":
    main()
