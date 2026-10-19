"""
Two-stage CN V cascade with evidence-gated anatomical memory (protocol v0.3).

Stage 1  6-class segmentation (bg + V1/V2/V3/foramen/bypass), ViT-UNETR + MLA
         memory (``--backbone vit``).  Its class-4 (foramen) probability and a
         3 mm dilation of the hard class-4 mask form the cascade prior.
Stage 2  Binary CN V mask Y = 1∪2∪3∪4 (class 5 is not in the target) on
         [CT, P(c4), dilate(c4)] with a DynUNet + anatomical_v4 memory backbone
         (``--backbone nnunet_mla``).  An evidence map g(x) (bone window +
         foramen prior) gates local vs. memory features at every memory level:
         f = g*f_loc + (1-g)*f_mem; see ``cnv_protocol_s2.py``.

Usage (run from the repository root):

  # Stage 1, folds 0-4
  python trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py --stage 1 --fold 0 \
      --epochs 300 --exp_suffix _s1_6class_protocol_v03_20261016 --gpu 0
  # Out-of-fold Stage-1 softmax for every training case (after all 5 folds)
  python trainer/npc_vesselFM3D_cascade_MIA_v3_vit_memory.py --stage generate_cv \
      --exp_suffix _s1_6class_protocol_v03_20261016 --gpu 0
  # Stage 2: print the exact command for each P0 arm (Full/A_mem/A_gate/B0/C1/A2)
  python MIA/scripts/ablation_configs.py --commands 0

  # Rebuild the preprocessing cache
  ... --stage 1 --fold 0 --force_cache
"""

import argparse
import csv
import json
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Tuple, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import nrrd
except ImportError:
    nrrd = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from npc_data import DATA_ROOT_DEFAULT, _get_all_cases, _create_folds
# Protocol v0.3 Phase2/3 helpers
try:
    from cnv_protocol_s2 import (
        stage1_softmax_to_cascade_prior, convert_label_to_Y,
        class5_bypass_ignore_mask, BinaryYLoss, BinaryYStage2Model,
        EvidenceGatedYLoss,
        DEFAULT_SPACING_ZYX as _PROT_SPACING,
        scramble_cascade_prior,
        DEFAULT_GATE_TAU, DEFAULT_GATE_ALPHA_FIXED,
        DEFAULT_BONE_NORM_THR, DEFAULT_EVIDENCE_TEMP,
        DEFAULT_LAMBDA_L, DEFAULT_LAMBDA_MEM, DEFAULT_MEM_SUP_WEIGHT,
    )
    _HAS_PROTOCOL_S2 = True
except ImportError:
    _HAS_PROTOCOL_S2 = False
    print("[WARN] cnv_protocol_s2 not importable")

from npc_losses import WeightedBCEDiceLoss, multiclass_dice_score
import npc_preprocess as pp
from npc_vit_mla import ViTUNet3D, MLAMemoryBank, ContextAnchorMemoryBank, CNNDecoder3D, _DecoderBlock
from npc_anatomy_memory import AnatomySlotMemoryBank
from npc_anatomy_memory_v4 import AnatomySlotMemoryBankV4
from npc_anatomy_memory_v5 import AnatomySlotMemoryBankV5
from npc_anatomy_memory_v6 import AnatomySlotMemoryBankV6
from npc_anatomy_memory_v4_mgp import AnatomySlotMemoryBankV4MGP
from npc_nnformer_mla_v5 import NNFormerUNet3D
from npc_nnunet_mla import DynUNetMLA3D, MEMORY_FEATURE_DIMS as NNUNET_MLA_MEMORY_DIMS

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==================== 配置参数 ====================
RESULTS_ROOT = os.environ.get("CNV_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
PREPROCESS_CACHE_ROOT = str(Path(RESULTS_ROOT) / "preprocess_cache")
_SCRIPT_STEM = Path(__file__).stem
EXPERIMENT_NAME_STAGE1_TEMPLATE = f"{_SCRIPT_STEM}_stage1"
EXPERIMENT_NAME_STAGE2_TEMPLATE = f"{_SCRIPT_STEM}_stage2"
PATCH_SIZE: Tuple[int, int, int] = (40, 192, 192)
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8  # Effective batch=8 for stable ViT gradients
NUM_CLASSES_STAGE1 = 6  # bg+1..5; class5=class5_bypass/bypass_roi (NOT bony_*)
# CNV_PROTOCOL_V03_PHASE2
# Exclusive 6-class soft-tree output: {0:bg, 1:c1, 2:c2, 3:c3, 4:c4, 5:c5}.
# class4 (bony foramen / 骨性穿孔) is large, CT-conspicuous and anatomically the
# anchor of the V1 (class1) pathway, so it is grouped WITH nerve_body and split
# inside Branch A ({c1, c2, c3, c4}) rather than being an overlapping aux output.
NUM_CLASSES_STAGE2 = 6  # bg, class1, class2, class3, class4, class5
NUM_CLASSES_STAGE0 = 6  # Single-stage ViT: flat 6-class output
NUM_FOLDS = 5
# Ablation switch (set from --zero_stage1_prior): replace the two Stage-1 prior
# channels with constant zeros so Stage2 must predict from CT alone. Capacity,
# memory bank, decoder and training recipe stay identical, so the cascade prior
# is the only ablated component.
ZERO_STAGE1_PRIOR = False
# Per-sample prior-channel dropout probability during Stage-2 training
# (set from --prior_dropout_prob; 0 = disabled). Anti-shortcut-learning.
PRIOR_DROPOUT_PROB = 0.0
# Foreground-only prior mode (set from --prior_fg_only): the prior npy carries
# [bg, fg] channels so Stage2Dataset's [1:] slice yields a SINGLE foreground
# channel; the model is built with in_channels=2 (CT + fg) and the soft-tree
# confidence gate blends only the bg/fg evidence (c5 gets no prior share).
PRIOR_FG_ONLY = False
PRIOR_SCRAMBLE_MODE = '0'  # CNV protocol v0.3 A2
B0_SINGLE_STAGE = False
GATE_TAU = 0.5
GATE_ALPHA_FIXED = None
REFUSE_LEGACY_HEADS = True
SPACING_PARAMS_PATH = str(PROJECT_ROOT / "spacing_params.json")
VALIDATE_EVERY_N_EPOCHS = 5
EARLY_STOP_PATIENCE = 50
WARMUP_EPOCHS = 10  # Short warmup: ViT needs fast ramp-up

# ViT 架构参数 (Hybrid UNETR: Conv Stem + ViT + CNN Decoder)
# Scaled-up architecture: ~45M params (was 16.4M)
# Key changes: embed_dim 128→192, depths [2,4,2]→[3,6,4], heads 4→6, kv_latent 64→96
VIT_EMBED_DIM = 192
VIT_STEM_CHANNELS = (48, 96, 192)  # Conv stem channel progression (proportional to embed_dim)
VIT_TOKENIZE_KERNEL = (2, 4, 4)  # Tokenization after stem: effective total stride = 1*2*2*2=8D, 1*2*2*4=16H/W
VIT_NUM_HEADS = 6  # 6 heads × 32 head_dim = 192 = embed_dim (no info loss in projection)
VIT_HEAD_DIM = 32
VIT_KV_LATENT_DIM = 96  # Wider MLA bottleneck for richer attention
VIT_ENCODER_DEPTHS = [3, 6, 4]  # Deeper at cheap stages (stage1=108 tokens, stage2=18 tokens)
VIT_DECODER_DEPTHS = [2, 3]  # Slightly deeper decoder for better reconstruction
VIT_MLP_RATIO = 4.0
VIT_DROPOUT = 0.1
VIT_DROP_PATH_RATE = 0.2  # Increased regularization for larger model on small dataset

# Memory Bank 配置
NUM_PROTOTYPES_PER_CLASS = 32  # Balanced capacity vs utilization (was 64, most unused)
MEMORY_TOP_K = 8  # Retrieve top-k most relevant prototypes per query
MEMORY_MOMENTUM = 0.99
MEMORY_TEMPERATURE = 0.5  # Less aggressive temperature (was 0.1, caused sharp distributions)
MEMORY_PATCH_GRID = (2, 2, 2)
MEMORY_BANK_WARMUP_EPOCHS = 50  # Enable after model starts converging

# ViT encoder feature dims at each stage: [embed_dim, embed_dim*2, embed_dim*4]
VIT_FEATURE_DIMS = [VIT_EMBED_DIM, VIT_EMBED_DIM * 2, VIT_EMBED_DIM * 4]

# Hierarchical heads operate at token resolution on embed_dim features
TREE_DECODER_FEATURE_DIM = VIT_EMBED_DIM  # 128

# AMP configuration: bfloat16 for numerical stability (no GradScaler needed)
# bf16 has same exponent range as fp32, preventing attention score overflow
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
    USE_GRAD_SCALER = False  # bf16 doesn't need loss scaling
else:
    AMP_DTYPE = torch.float16
    USE_GRAD_SCALER = True


# ==================== 预处理函数 ====================
def normalize_percentile(image, lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True):
    a_min = np.percentile(image, lower)
    a_max = np.percentile(image, upper)
    if a_max - a_min < 1e-8:
        return np.full_like(image, (b_min + b_max) / 2, dtype=np.float32)
    image = (image - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
    if clip:
        image = np.clip(image, b_min, b_max)
    return image.astype(np.float32)


def _load_label_as_int3d(label_path) -> np.ndarray:
    """加载标签为3D int16数组, 支持NIfTI和NRRD格式"""
    label_path = Path(label_path)
    if str(label_path).endswith('.nrrd'):
        if nrrd is None:
            raise ImportError("pynrrd not installed. Run: pip install pynrrd")
        data, _ = nrrd.read(str(label_path))
        if data.ndim == 4:
            num_channels = data.shape[0]
            spatial_shape = data.shape[1:]
            label_3d = np.zeros(spatial_shape, dtype=np.int16)
            for ch_idx in range(num_channels):
                label_3d[data[ch_idx] > 0] = ch_idx + 1
            return np.clip(label_3d, 0, 5)
        elif data.ndim == 3:
            return np.clip(data.astype(np.int16), 0, 5)
        else:
            raise ValueError(f"Unexpected NRRD label ndim={data.ndim}")
    else:
        nii = nib.load(str(label_path))
        arr = np.asarray(nii.dataobj)
        if arr.ndim == 3:
            label_3d = arr.astype(np.int16)
        elif arr.ndim == 4:
            if arr.shape[-1] <= 10:
                bin_arr = (arr > 0)
                fg_any = bin_arr.any(axis=-1)
                fg_idx = bin_arr.argmax(axis=-1).astype(np.int16) + 1
                label_3d = np.where(fg_any, fg_idx, 0).astype(np.int16)
            else:
                num_channels = arr.shape[0]
                spatial_shape = arr.shape[1:]
                label_3d = np.zeros(spatial_shape, dtype=np.int16)
                for ch_idx in range(num_channels):
                    label_3d[arr[ch_idx] > 0] = ch_idx + 1
        else:
            raise ValueError(f"Unexpected label ndim={arr.ndim}")
        return np.clip(label_3d, 0, 5)


# ==================== 预处理缓存 ====================
def _get_cache_dir(target_spacing) -> Path:
    if target_spacing is None:
        sub = "native"
    else:
        sub = "sp_" + "_".join(f"{s:.3f}" for s in target_spacing)
    return Path(PREPROCESS_CACHE_ROOT) / sub


def prepare_preprocessed_cache(cases, target_spacing, force=False) -> Path:
    cache_dir = _get_cache_dir(target_spacing)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(cases)
    n_cached, n_new = 0, 0
    print(f"[Cache] dir={cache_dir}")
    for idx, (case_id, ct_path, label_path) in enumerate(cases):
        img_p = cache_dir / f"{case_id}_image.npy"
        lbl_p = cache_dir / f"{case_id}_label.npy"
        fg_p  = cache_dir / f"{case_id}_fg_indices.npz"
        if not force and img_p.exists() and lbl_p.exists() and fg_p.exists():
            n_cached += 1
            continue
        ct_nii = nib.load(ct_path)
        image = ct_nii.get_fdata().astype(np.float32)
        label_orig = _load_label_as_int3d(label_path)
        image = np.transpose(image, (2, 0, 1))
        label_orig = np.transpose(label_orig, (2, 0, 1))
        if target_spacing is not None:
            nii_spacing = ct_nii.header.get_zooms()[:3]
            orig_sp = (float(nii_spacing[2]), float(nii_spacing[1]), float(nii_spacing[0]))
            image = pp.resample_image_to_spacing(image, orig_sp, target_spacing, order=3)
            label_orig = pp.resample_image_to_spacing(
                label_orig.astype(np.float32), orig_sp, target_spacing, order=0
            ).astype(np.int16)
        image = normalize_percentile(image)
        label_clean = np.clip(label_orig, 0, 5).astype(np.uint8)
        np.save(str(img_p) + ".tmp", image.astype(np.float32))
        np.save(str(lbl_p) + ".tmp", label_clean)
        # Pre-compute foreground voxel indices as [shape_d, shape_h, shape_w] + raveled_indices
        fg_raveled = np.flatnonzero(label_clean > 0).astype(np.int64)
        d, h, w = label_clean.shape
        # Store shape + indices in a single .npz for atomic replace
        fg_tmp = str(fg_p) + ".tmp.npz"
        np.savez_compressed(fg_tmp, shape=np.array([d, h, w], dtype=np.int32), indices=fg_raveled)
        os.replace(str(img_p) + ".tmp.npy", img_p)
        os.replace(str(lbl_p) + ".tmp.npy", lbl_p)
        os.replace(fg_tmp, str(fg_p).replace(".npy", ".npz"))
        n_new += 1
        if n_new % 5 == 0 or (idx + 1) == n_total:
            print(f"  [{idx+1}/{n_total}] reused={n_cached} new={n_new} latest={case_id} shape={image.shape} fg={len(fg_raveled)}")
    print(f"[Cache] done. total={n_total} reused={n_cached} new={n_new}")
    return cache_dir


def _load_cached_case(case_id, cache_dir, mmap=True):
    mode = 'r' if mmap else None
    img = np.load(Path(cache_dir) / f"{case_id}_image.npy", mmap_mode=mode)
    lbl = np.load(Path(cache_dir) / f"{case_id}_label.npy", mmap_mode=mode)
    return img, lbl


def _load_cached_fg_indices(case_id, cache_dir):
    """Load pre-computed foreground voxel indices as (z, y, x) array [N_fg, 3]."""
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / f"{case_id}_fg_indices.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    shape = data['shape']  # [d, h, w]
    d, h, w = int(shape[0]), int(shape[1]), int(shape[2])
    raveled = data['indices']
    z = raveled // (h * w)
    y = (raveled % (h * w)) // w
    x = raveled % w
    return np.stack([z, y, x], axis=-1), (d, h, w)


# ==================== 标签转换 ====================
def convert_label_to_stage1(label: np.ndarray) -> np.ndarray:
    """Protocol S1: identity clip {0..5}. class5=class5_bypass; bony_roi=class4 only."""
    return np.clip(label, 0, 5).astype(np.int16, copy=False)


def convert_label_to_stage2(label: np.ndarray) -> np.ndarray:
    """Convert original label to the exclusive Stage2 target (6 classes 0-5).

    The raw annotation is an exclusive integer label (one class per voxel).
    Stage2 keeps the original class indices {0:bg, 1:c1, 2:c2, 3:c3, 4:c4, 5:c5}
    so that ``argmax`` over the hierarchical soft-tree output maps directly back
    to the original classes. The group / branch / class4-anchor sub-targets are
    derived from this label inside the loss (see ``SoftTreeLoss``), so no
    auxiliary label needs to be materialized here.
    """
    return np.clip(label, 0, 5).astype(np.int16, copy=False)


# ==================== 学习率调度 ====================
class PolyLRWithWarmup:
    def __init__(self, optimizer, max_epochs, warmup_epochs=20, power=0.9,
                 initial_lr=1e-3, warmup_start_lr=1e-5, min_lr=1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.power = power
        self.initial_lr = initial_lr
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            if 'lr_scale' in pg:
                pg['lr'] = lr * pg['lr_scale']
            else:
                pg['lr'] = lr

    def _compute_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            return max(self.initial_lr * (1 - progress) ** self.power, self.min_lr)

    def get_last_lr(self):
        return [self._compute_lr(self._step)]


# ==================== 数据集 ====================
class Stage1Dataset(Dataset):
    def __init__(self, cases, patch_size, cache_dir, training=True,
                 random_crop_prob=0.0):
        self.cases = cases
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir)
        self.training = training
        # nnU-Net-style mixed sampling: with p=random_crop_prob the training crop
        # is uniform-random instead of foreground-centered (reduces full-image FPs).
        self.random_crop_prob = float(random_crop_prob)
        # Pre-load foreground indices for all cases (cheap: just a few KB per case)
        self._fg_cache = {}
        if training:
            for case_id, _, _ in cases:
                fg_data = _load_cached_fg_indices(case_id, cache_dir)
                if fg_data is not None:
                    self._fg_cache[case_id] = fg_data

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id, _, _ = self.cases[idx]
        image_mm, _label_mm = _load_cached_case(case_id, self.cache_dir, mmap=True)
        d, h, w = image_mm.shape
        pd, ph, pw = self.patch_size
        # Use cached foreground indices for center sampling (fast path)
        need_pad = d < pd or h < ph or w < pw
        if self.training:
            fg_data = self._fg_cache.get(case_id)
            if self.random_crop_prob > 0 and np.random.rand() < self.random_crop_prob:
                # Uniform random crop (nnU-Net-style mixed sampling)
                z1 = np.random.randint(0, max(1, d - pd + 1))
                y1 = np.random.randint(0, max(1, h - ph + 1))
                x1 = np.random.randint(0, max(1, w - pw + 1))
            elif fg_data is not None and len(fg_data[0]) > 0:
                fg_coords, vol_shape = fg_data
                fi = np.random.randint(len(fg_coords))
                cz, cy, cx = fg_coords[fi]
                z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                x1 = int(np.clip(cx - pw // 2, 0, w - pw))
            else:
                # Fallback: load label to find foreground (shouldn't happen if cache is built)
                label_full = np.array(_label_mm, dtype=np.int16)
                if np.any(label_full > 0):
                    fg = np.argwhere(label_full > 0)
                    fi = np.random.randint(len(fg))
                    cz, cy, cx = fg[fi]
                    z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                    y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                    x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                else:
                    z1, y1, x1 = np.random.randint(0, max(1, d - pd + 1)), np.random.randint(0, max(1, h - ph + 1)), np.random.randint(0, max(1, w - pw + 1))
        else:
            z1, y1, x1 = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
        if need_pad:
            image_full = np.array(image_mm, dtype=np.float32)
            label_full = np.array(_label_mm, dtype=np.int16)
            nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
            image_full = pp._pad_to_shape_3d(image_full, (nd, nh, nw))
            label_full = pp._pad_to_shape_3d(label_full, (nd, nh, nw))
            image = image_full[z1:z2, y1:y2, x1:x2]
            label_orig = label_full[z1:z2, y1:y2, x1:x2]
        else:
            image = np.array(image_mm[z1:z2, y1:y2, x1:x2], dtype=np.float32)
            label_orig = np.array(_label_mm[z1:z2, y1:y2, x1:x2], dtype=np.int16)
        if self.training:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=ax)
                    label_orig = np.flip(label_orig, axis=ax)
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2))
                label_orig = np.rot90(label_orig, k=k, axes=(1, 2))
            image = pp.random_gamma(image, p=0.15, gamma_range=(0.7, 1.5))
            image = pp.random_gaussian_noise(image, p=0.1, sigma_range=(0.0, 0.05))
            image = np.clip(image, 0.0, 1.0)
        label = convert_label_to_stage1(label_orig)
        image_t = torch.from_numpy(np.ascontiguousarray(image)[None]).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
        return image_t, label_t


def _ensure_stage1_softmax_npy(stage1_pred_dir: Path, case_id: str) -> Path:
    """Convert Stage1 NRRD softmax to mmap-friendly .npy on first access; cache it."""
    npy_path = stage1_pred_dir / f"{case_id}_stage1_softmax.npy"
    if npy_path.exists():
        return npy_path
    nrrd_path = stage1_pred_dir / f"{case_id}_stage1_softmax.nrrd"
    if not nrrd_path.exists():
        raise FileNotFoundError(f"Stage1 prediction not found: {nrrd_path}")
    data = _load_softmax_nrrd(str(nrrd_path))  # [C, D, H, W]; C=6 protocol S1
    # Atomic write: concurrent stage-2 folds mmap these files while another
    # fold's process may still be building them -> write tmp + os.replace.
    tmp_path = str(npy_path) + f".tmp{os.getpid()}.npy"
    np.save(tmp_path, data.astype(np.float32))
    os.replace(tmp_path, npy_path)
    return npy_path


class Stage2Dataset(Dataset):
    def __init__(self, cases, stage1_pred_dir, patch_size, cache_dir, training=True,
                 random_crop_prob=0.0, return_binary_y=False):
        self.return_binary_y = bool(return_binary_y)
        self.cases = cases
        self.stage1_pred_dir = Path(stage1_pred_dir)
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir)
        self.training = training
        # nnU-Net-style mixed sampling: with p=random_crop_prob the training crop
        # is drawn uniformly from the volume (may be all background), so the model
        # learns to output background away from the anatomy instead of
        # hallucinating foreground in out-of-distribution regions.
        self.random_crop_prob = float(random_crop_prob)
        # Pre-load foreground indices
        self._fg_cache = {}
        if training:
            for case_id, _, _ in cases:
                fg_data = _load_cached_fg_indices(case_id, cache_dir)
                if fg_data is not None:
                    self._fg_cache[case_id] = fg_data

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id, _, _ = self.cases[idx]
        image_mm, _label_mm = _load_cached_case(case_id, self.cache_dir, mmap=True)
        # Use npy cache for stage1 softmax (avoids NRRD parse every time)
        s1_npy_path = _ensure_stage1_softmax_npy(self.stage1_pred_dir, case_id)
        stage1_full = np.load(s1_npy_path, mmap_mode='r')  # [C,D,H,W] C=6 protocol or 3 legacy
        # Cascade Full prior: P(c4)+dilate(bony_roi=class4). class5_bypass excluded.
        if _HAS_PROTOCOL_S2:
            stage1_mm = stage1_softmax_to_cascade_prior(
                np.asarray(stage1_full), spacing_zyx=_PROT_SPACING)
        else:
            stage1_mm = np.asarray(stage1_full)[1:]
        d, h, w = image_mm.shape
        pd, ph, pw = self.patch_size
        if stage1_mm.shape[1:] != (d, h, w):
            raise RuntimeError(
                f"Shape mismatch for {case_id}: image={image_mm.shape} vs stage1={stage1_mm.shape[1:]}.")
        need_pad = d < pd or h < ph or w < pw
        if self.training:
            fg_data = self._fg_cache.get(case_id)
            if self.random_crop_prob > 0 and np.random.rand() < self.random_crop_prob:
                z1 = np.random.randint(0, max(1, d - pd + 1))
                y1 = np.random.randint(0, max(1, h - ph + 1))
                x1 = np.random.randint(0, max(1, w - pw + 1))
            elif fg_data is not None and len(fg_data[0]) > 0:
                fg_coords, vol_shape = fg_data
                fi = np.random.randint(len(fg_coords))
                cz, cy, cx = fg_coords[fi]
                z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                x1 = int(np.clip(cx - pw // 2, 0, w - pw))
            else:
                label_full = np.array(_label_mm, dtype=np.int16)
                if np.any(label_full > 0):
                    fg = np.argwhere(label_full > 0)
                    fi = np.random.randint(len(fg))
                    cz, cy, cx = fg[fi]
                    z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                    y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                    x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                else:
                    z1, y1, x1 = np.random.randint(0, max(1, d - pd + 1)), np.random.randint(0, max(1, h - ph + 1)), np.random.randint(0, max(1, w - pw + 1))
        else:
            z1, y1, x1 = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
        if need_pad:
            image_full = np.array(image_mm, dtype=np.float32)
            label_full = np.array(_label_mm, dtype=np.int16)
            s1_full = np.array(stage1_mm, dtype=np.float32)
            nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
            image_full = pp._pad_to_shape_3d(image_full, (nd, nh, nw))
            label_full = pp._pad_to_shape_3d(label_full, (nd, nh, nw))
            s1_padded = np.zeros((s1_full.shape[0], nd, nh, nw), dtype=np.float32)
            s1_padded[:, :d, :h, :w] = s1_full
            s1_full = s1_padded
            d, h, w = image_full.shape
            image = image_full[z1:z2, y1:y2, x1:x2]
            stage1_softmax = s1_full[:, z1:z2, y1:y2, x1:x2]
            label_orig = label_full[z1:z2, y1:y2, x1:x2].astype(np.int16)
        else:
            image = np.array(image_mm[z1:z2, y1:y2, x1:x2], dtype=np.float32)
            stage1_softmax = np.array(stage1_mm[:, z1:z2, y1:y2, x1:x2], dtype=np.float32)
            label_orig = np.array(_label_mm[z1:z2, y1:y2, x1:x2], dtype=np.int16)
        label_mc = convert_label_to_stage2(label_orig)
        if _HAS_PROTOCOL_S2:
            label_y = convert_label_to_Y(label_orig)
            ignore_bypass = class5_bypass_ignore_mask(label_orig)
        else:
            label_y = ((label_mc >= 1) & (label_mc <= 4)).astype(np.int64)
            ignore_bypass = (label_mc == 5)
        label = label_mc
        flips = [0.0, 0.0, 0.0]
        if self.training:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=ax)
                    label = np.flip(label, axis=ax)
                    label_y = np.flip(label_y, axis=ax)
                    ignore_bypass = np.flip(ignore_bypass, axis=ax)
                    stage1_softmax = np.flip(stage1_softmax, axis=ax + 1)
                    flips[ax] = 1.0
        if PRIOR_SCRAMBLE_MODE not in ('0', 'none', 'off', '', None):
            try:
                seed = int(case_id) % (2 ** 31)
            except (TypeError, ValueError):
                seed = zlib.crc32(str(case_id).encode()) % (2 ** 31)
            stage1_softmax = scramble_cascade_prior(
                stage1_softmax, mode=PRIOR_SCRAMBLE_MODE,
                rng=np.random.RandomState(seed ^ 0xA2A2))
        if ZERO_STAGE1_PRIOR:
            stage1_softmax = np.zeros_like(stage1_softmax, dtype=np.float32)
        elif self.training and PRIOR_DROPOUT_PROB > 0 and np.random.rand() < PRIOR_DROPOUT_PROB:
            # Prior dropout: randomly zero the Stage-1 prior channels per-sample so
            # the network cannot shortcut-depend on the coarse prior and inherits
            # its boundary noise. Inference/validation always see the real prior.
            stage1_softmax = np.zeros_like(stage1_softmax, dtype=np.float32)
        image_t = torch.from_numpy(np.ascontiguousarray(image)[None]).float()
        s1_t = torch.from_numpy(np.ascontiguousarray(stage1_softmax)).float()
        if self.return_binary_y:
            y = label_y.astype(np.int64).copy()
            y[ignore_bypass] = -1  # class5_bypass ignored in BinaryYLoss
            label_t = torch.from_numpy(np.ascontiguousarray(y))
        else:
            label_t = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
        # Spatial metadata for SPM (V5) memory bank: crop origin / volume shape
        # / per-axis flip flags, all in the (possibly padded) volume frame.
        meta_t = torch.tensor([z1, y1, x1, d, h, w] + flips, dtype=torch.float32)
        return image_t, s1_t, label_t, meta_t


# ==================== 模型 (ViT + MLA Memory Bank) ====================
def _build_memory_bank(num_classes, num_prototypes_per_class, momentum, temperature, top_k,
                       retrieval_mode='cross_attention', memory_type='mla', feature_dims=None):
    """Create memory bank matching ViT encoder feature dims.

    feature_dims: per-level encoder dims; defaults to VIT_FEATURE_DIMS. Pass the
    DynUNet level dims ([128, 256, 320]) when backbone='nnunet_mla'.

    memory_type: 'mla' (class-organized), 'context_anchor' (context-organized),
    or 'anatomical' (AnatomySlotMemoryBank: per-token soft retrieval, per-class
    Sinkhorn-balanced slot assignment, neighbor-smooth anatomical territories).
    """
    fd = feature_dims or VIT_FEATURE_DIMS
    if memory_type == 'anatomical_v6':
        return AnatomySlotMemoryBankV6(
            feature_dims=fd,
            num_classes=num_classes,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            num_slots_per_class=num_prototypes_per_class,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            temperature=temperature,
            top_k=top_k,
            momentum=momentum,
            patch_grid=MEMORY_PATCH_GRID,
        )
    if memory_type == 'anatomical_v4_mgp':
        return AnatomySlotMemoryBankV4MGP(
            feature_dims=fd,
            num_classes=num_classes,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            num_slots_per_class=num_prototypes_per_class,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            temperature=temperature,
            top_k=top_k,
            momentum=momentum,
            patch_grid=MEMORY_PATCH_GRID,
        )
    if memory_type == 'anatomical_v5':
        return AnatomySlotMemoryBankV5(
            feature_dims=fd,
            num_classes=num_classes,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            num_slots_per_class=num_prototypes_per_class,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            temperature=temperature,
            top_k=top_k,
            momentum=momentum,
            patch_grid=MEMORY_PATCH_GRID,
        )
    if memory_type == 'anatomical_v4':
        return AnatomySlotMemoryBankV4(
            feature_dims=fd,
            num_classes=num_classes,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            num_slots_per_class=num_prototypes_per_class,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            temperature=temperature,
            top_k=top_k,
            momentum=momentum,
            patch_grid=MEMORY_PATCH_GRID,
        )
    if memory_type == 'anatomical':
        return AnatomySlotMemoryBank(
            feature_dims=fd,
            num_classes=num_classes,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            num_slots_per_class=num_prototypes_per_class,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            temperature=temperature,
            top_k=top_k,
            momentum=momentum,
            patch_grid=MEMORY_PATCH_GRID,
        )
    if memory_type == 'context_anchor':
        num_anchors = 8
        protos_per_anchor = max(1, num_prototypes_per_class // num_anchors)
        return ContextAnchorMemoryBank(
            feature_dims=fd,
            num_classes=num_classes,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            num_anchors=num_anchors,
            protos_per_anchor=protos_per_anchor,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            temperature=temperature,
            patch_grid=MEMORY_PATCH_GRID,
            top_k=top_k,
            momentum=momentum,
        )
    return MLAMemoryBank(
        feature_dims=fd,
        num_prototypes_per_class=num_prototypes_per_class,
        num_classes=num_classes,
        kv_latent_dim=VIT_KV_LATENT_DIM,
        num_heads=VIT_NUM_HEADS,
        head_dim=VIT_HEAD_DIM,
        momentum=momentum,
        temperature=temperature,
        patch_grid=MEMORY_PATCH_GRID,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )


def _build_vit(in_channels, num_classes, memory_bank=None, use_mla=True, backbone='vit',
               dual_path_stages=None, residual_conv_stages=None):
    """Create Hybrid backbone with standard config.

    backbone: 'vit' (ViTUNet3D, global MLA) or 'nnformer_v5' (NNFormerUNet3D,
    nnFormer-style interleaved conv + local/global dual-path MLA). Both share
    the identical token pyramid / feature dims / memory-bank contract, so the
    memory bank and cascade wiring are unchanged.
    """
    common = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=VIT_EMBED_DIM,
        stem_channels=VIT_STEM_CHANNELS,
        tokenize_kernel=VIT_TOKENIZE_KERNEL,
        num_heads=VIT_NUM_HEADS,
        head_dim=VIT_HEAD_DIM,
        kv_latent_dim=VIT_KV_LATENT_DIM,
        encoder_depths=VIT_ENCODER_DEPTHS,
        decoder_depths=VIT_DECODER_DEPTHS,
        mlp_ratio=VIT_MLP_RATIO,
        dropout=VIT_DROPOUT,
        drop_path_rate=VIT_DROP_PATH_RATE,
        memory_bank=memory_bank,
        use_mla=use_mla,
    )
    if backbone == 'nnunet_mla':
        return DynUNetMLA3D(
            in_channels=in_channels,
            num_classes=num_classes,
            num_heads=VIT_NUM_HEADS,
            head_dim=VIT_HEAD_DIM,
            kv_latent_dim=VIT_KV_LATENT_DIM,
            use_mla=use_mla,
            memory_bank=memory_bank,
        )
    if backbone == 'nnformer_v5':
        return NNFormerUNet3D(dual_path_stages=dual_path_stages,
                              residual_conv_stages=residual_conv_stages, **common)
    return ViTUNet3D(**common)


class Stage1Model(nn.Module):
    """Stage1: ViT-UNETR + Memory Bank, 1-ch CT -> 6-class (bg+1..5); class5=class5_bypass"""
    def __init__(self, use_memory_bank=True, num_prototypes_per_class=8,
                 memory_momentum=0.99, memory_temperature=0.1, memory_top_k=8,
                 use_mla=True, retrieval_mode='cross_attention', memory_type='mla', backbone='vit',
                 dual_path_stages=None, residual_conv_stages=None):
        super().__init__()
        self.use_memory_bank = use_memory_bank
        mb = None
        if use_memory_bank:
            mb = _build_memory_bank(NUM_CLASSES_STAGE1, num_prototypes_per_class,
                                    memory_momentum, memory_temperature, memory_top_k,
                                    retrieval_mode=retrieval_mode, memory_type=memory_type,
                                    feature_dims=NNUNET_MLA_MEMORY_DIMS if backbone == 'nnunet_mla' else None)
            print(f"✓ Stage1 Memory Bank ({memory_type}): prototypes={mb.num_prototypes}, "
                  f"top_k={memory_top_k}, kv_latent={VIT_KV_LATENT_DIM}, patch_grid={MEMORY_PATCH_GRID}")
        self.net = _build_vit(1, NUM_CLASSES_STAGE1, memory_bank=mb, use_mla=use_mla, backbone=backbone, dual_path_stages=dual_path_stages, residual_conv_stages=residual_conv_stages)

    def forward(self, x, labels=None):
        return self.net(x, labels)


class Stage2Model(nn.Module):
    """Legacy flat Stage2: 3-channel input -> flat 6-class output."""
    def __init__(self, use_memory_bank=True, num_prototypes_per_class=8,
                 memory_momentum=0.99, memory_temperature=0.1, memory_top_k=8,
                 use_mla=True, retrieval_mode='cross_attention', memory_type='mla', backbone='vit',
                 dual_path_stages=None, residual_conv_stages=None):
        super().__init__()
        self.use_memory_bank = use_memory_bank
        mb = None
        if use_memory_bank:
            mb = _build_memory_bank(NUM_CLASSES_STAGE2, num_prototypes_per_class,
                                    memory_momentum, memory_temperature, memory_top_k,
                                    retrieval_mode=retrieval_mode, memory_type=memory_type,
                                    feature_dims=NNUNET_MLA_MEMORY_DIMS if backbone == 'nnunet_mla' else None)
            print(f"✓ Stage2 Memory Bank ({memory_type}): prototypes={mb.num_prototypes}, top_k={memory_top_k}")
        self.net = _build_vit(3, NUM_CLASSES_STAGE2, memory_bank=mb, use_mla=use_mla, backbone=backbone, dual_path_stages=dual_path_stages, residual_conv_stages=residual_conv_stages)

    def forward(self, x, labels=None, coord_info=None):
        if coord_info is not None and isinstance(self.net, DynUNetMLA3D):
            return self.net(x, labels, coord_info)
        return self.net(x, labels)


# ==================== Hierarchical Soft-Tree Stage2 ====================
# Tree structure (exclusive 6-class output):
#   root (bg / fg)
#   ├── nerve_body (c1, c2, c3, c4)  ->  Branch A: class1 / class2 / class3 / class4
#   └── class5                       ->  leaf (bony foramen / landmark, no branching)
#
# class4 is the bony foramen (骨性穿孔): large, CT-conspicuous and the anatomical
# anchor of the V1 (class1) pathway. It is therefore grouped WITH nerve_body so
# the (c1∪c2∪c3∪c4) region is localized as one easy, well-anchored mass (the
# behaviour that makes "merge c4 into c1-3" score high), and Branch A then splits
# the easy/large c4 away from c1 — giving a cleaner c1 while still distinguishing c4.

class HierarchicalStage2Model(nn.Module):
    """Hierarchical Soft-Tree Stage2 + MLA Memory Bank (Hybrid UNETR).

    Uses ViTUNet3D.forward_features() to get decoder features + stem skips.
    The CNN decoder progressively upsamples to full resolution, then
    lightweight Conv3d heads produce the hierarchical decomposition.

    Exclusive output: 6 classes (bg, c1, c2, c3, c4, c5) via group x branch product.
    A lightweight class4 deep-supervision head (sigmoid) reinforces the easy bony
    anchor during training; it is NOT used at inference (argmax over final_probs).

    Output: (final_probs[6], group_logits[3], branch_a_logits[4], c4_logits[1], alpha[1])
    """
    def __init__(self, use_memory_bank=True, num_prototypes_per_class=8,
                 memory_momentum=0.99, memory_temperature=0.1, memory_top_k=8,
                 use_confidence_gate=True, use_mla=True, retrieval_mode='cross_attention',
                 memory_type='mla', backbone='vit', dual_path_stages=None, residual_conv_stages=None,
                 in_channels=3):
        super().__init__()
        self.use_memory_bank = use_memory_bank
        self.use_confidence_gate = use_confidence_gate
        mb = None
        if use_memory_bank:
            mb = _build_memory_bank(NUM_CLASSES_STAGE2, num_prototypes_per_class,
                                    memory_momentum, memory_temperature, memory_top_k,
                                    retrieval_mode=retrieval_mode, memory_type=memory_type,
                                    feature_dims=NNUNET_MLA_MEMORY_DIMS if backbone == 'nnunet_mla' else None)
            print(f"\u2713 Stage2-Hier Memory Bank ({memory_type}): prototypes={mb.num_prototypes}, top_k={memory_top_k}")
        # num_classes=1 placeholder — we use forward_features + custom CNN decoder
        self.net = _build_vit(in_channels, 1, memory_bank=mb, use_mla=use_mla, backbone=backbone, dual_path_stages=dual_path_stages, residual_conv_stages=residual_conv_stages)

        # Custom CNN decoder for hierarchical heads (shared stem skip path)
        # Produces full-resolution features from which heads branch
        stem_ch = VIT_STEM_CHANNELS  # (48, 96, 192)
        if getattr(self.net, 'provides_fullres_features', False):
            # nnunet_mla: the DynUNet decoder already yields full-res 32-ch
            # features; a light adapter maps them to feat_dim.
            self.hier_decoder = nn.Sequential(
                nn.Conv3d(32, stem_ch[0], kernel_size=3, padding=1),
                nn.InstanceNorm3d(stem_ch[0]),
                nn.GELU(),
            )
        else:
            # Reuse same progressive upsample but output stem_ch[0] channels
            self.hier_decoder = CNNDecoder3D(
                embed_dim=VIT_EMBED_DIM,
                stem_channels=VIT_STEM_CHANNELS,
                tokenize_kernel=VIT_TOKENIZE_KERNEL,
                num_classes=stem_ch[0],  # Output feature channels, not final classes
            )
            # Replace out_conv: we want features, not class logits
            self.hier_decoder.out_conv = nn.Sequential(
                nn.Conv3d(stem_ch[0], stem_ch[0], kernel_size=3, padding=1),
                nn.InstanceNorm3d(stem_ch[0]),
                nn.GELU(),
            )

        feat_dim = stem_ch[0]  # full-resolution feature dim
        # Group head: bg(0) / nerve_body(1) / class5(2)
        self.group_head = nn.Conv3d(feat_dim, 3, kernel_size=1)
        # Branch A: class1 / class2 / class3 / class4 (within nerve_body)
        self.branch_a_head = nn.Conv3d(feat_dim, 4, kernel_size=1)
        # class4 deep-supervision head: easy bony anchor (sigmoid), train-only
        self.c4_head = nn.Conv3d(feat_dim, 1, kernel_size=1)
        # Confidence gate for blending group prediction with Stage1 prior
        self.confidence_gate = nn.Sequential(
            nn.Conv3d(feat_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        print(f"\u2713 Hierarchical Soft-Tree heads (feat_dim={feat_dim}, full-res via CNN decoder)")
        print(f"  Group: bg/nerve_body/class5 | Branch A: c1/c2/c3/c4 | c4 deep-supervision (sigmoid)")

    def forward(self, x, labels=None, coord_info=None):
        """
        Args:
            x:      [B, 3, D, H, W]  (CT + 2 Stage1 fg softmax)
            labels: [B, D, H, W] optional, for Memory Bank EMA update
            coord_info: optional (origin, vshape, flips) for SPM/V5 memory bank

        Returns:
            final_probs:      [B, 6, D, H, W]  (distribution over {bg,c1,c2,c3,c4,c5})
            group_logits:     [B, 3, D, H, W]  (bg / nerve_body / class5)
            branch_a_logits:  [B, 4, D, H, W]  (c1 / c2 / c3 / c4 within nerve_body)
            c4_logits:        [B, 1, D, H, W]  (class4 deep-supervision logits, pre-sigmoid)
            alpha:            [B, 1, D, H, W]  (confidence gate in [0,1])
        """
        # Get ViT decoder features + stem skips
        if coord_info is not None and isinstance(self.net, DynUNetMLA3D):
            vit_features, stem_skips, target_shape = self.net.forward_features(x, labels, coord_info)
        else:
            vit_features, stem_skips, target_shape = self.net.forward_features(x, labels)

        # Progressive CNN upsample to full resolution (nnunet_mla: features are
        # already full-res, the adapter hier_decoder takes them directly)
        if getattr(self.net, 'provides_fullres_features', False):
            feat = self.hier_decoder(vit_features)
        else:
            feat = self.hier_decoder(vit_features, stem_skips, target_shape)

        # Heads at full resolution
        group_logits = self.group_head(feat)
        branch_a_logits = self.branch_a_head(feat)
        c4_logits = self.c4_head(feat)
        if self.use_confidence_gate:
            alpha = self.confidence_gate(feat)
        else:
            alpha = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4],
                               device=x.device, dtype=x.dtype)

        # Blend with cascade prior: [P(c4), dilate(bony_roi)]. class5_bypass gets 0 prior.
        # Legacy SoftTree path; primary S2 = binary Y (--s2_head binary_y).
        if x.shape[1] == 2:
            stage1_fg = x[:, 1:2].clamp(0.0, 1.0)
            stage1_group = torch.cat([1.0 - stage1_fg, stage1_fg,
                                      torch.zeros_like(stage1_fg)], dim=1)
        else:
            stage1_pc4 = x[:, 1:2].clamp(0.0, 1.0)  # P(bony_roi)
            stage1_dil = x[:, 2:3].clamp(0.0, 1.0)  # dilate(bony_roi)
            stage1_nerve_prior = (stage1_pc4 + stage1_dil).clamp(0.0, 1.0)
            stage1_c5_prior = torch.zeros_like(stage1_pc4)  # class5_bypass
            stage1_bg = (1.0 - stage1_nerve_prior).clamp_(0.0, 1.0)
            stage1_group = torch.cat([stage1_bg, stage1_nerve_prior, stage1_c5_prior], dim=1)
        stage1_group = stage1_group / stage1_group.sum(dim=1, keepdim=True).clamp(min=1e-8)

        p_group = alpha * torch.softmax(group_logits, dim=1) + (1.0 - alpha) * stage1_group
        p_group = p_group / p_group.sum(dim=1, keepdim=True).clamp(min=1e-8)

        p_branch_a = torch.softmax(branch_a_logits, dim=1)  # [B, 4, D, H, W] -> c1,c2,c3,c4

        # Final 6-class probability: [bg, c1, c2, c3, c4, c5]
        final_probs = torch.cat([
            p_group[:, 0:1],                    # bg
            p_group[:, 1:2] * p_branch_a,       # nerve_body * (c1, c2, c3, c4)
            p_group[:, 2:3],                    # class5
        ], dim=1)  # [B, 6, D, H, W]

        return final_probs, group_logits, branch_a_logits, c4_logits, alpha


# ==================== Hierarchical Soft-Tree Loss (self-contained) ====================
def _soft_dice_from_probs(probs: torch.Tensor, target_onehot: torch.Tensor,
                          weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted soft Dice loss from probabilities.

    Args:
        probs:         [B, C, ...] probabilities (sum over C == 1)
        target_onehot: [B, C, ...] one-hot ground truth
        weights:       [C] per-class weights
    """
    dims = tuple(range(2, probs.ndim))
    inter = (probs * target_onehot).sum(dims)
    denom = probs.sum(dims) + target_onehot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)  # [B, C]
    loss_per_class = 1.0 - dice
    w = weights.to(probs.dtype).to(probs.device)
    return (loss_per_class * w).sum(dim=1).mean() / w.sum().clamp(min=eps)


class SoftTreeLoss(nn.Module):
    """Self-contained hierarchical soft-tree loss for the exclusive 6-class Stage2.

    Targets are all derived from a single exclusive integer label (0-5):
      * final  : 6-class NLL + soft Dice on ``final_probs`` (the tree product).
      * group  : 3-class CE + Dice (bg / nerve_body[c1-4] / c5) on ``group_logits``.
      * branch : 4-class CE + Dice (c1/c2/c3/c4) on ``branch_a_logits`` but only
                 inside the nerve_body region (masked), so it learns to split the
                 easy/large c4 anchor from c1.
      * c4 aux : binary BCE + Dice on ``c4_logits`` for the conspicuous bony anchor.

    This replaces the external ``HierarchicalTreeLoss`` because the tree shape
    changed (Branch A now has 4 children and class5 is a group leaf); keeping the
    loss local makes the whole Stage2 path explicit and verifiable.
    """

    def __init__(self, final_weights, group_weights, branch_weights,
                 lambda_group=0.3, lambda_branch=0.3, lambda_c4=0.5,
                 weight_dice=0.6, weight_ce=0.4):
        super().__init__()
        self.register_buffer('final_weights', torch.as_tensor(final_weights, dtype=torch.float32))
        self.register_buffer('group_weights', torch.as_tensor(group_weights, dtype=torch.float32))
        self.register_buffer('branch_weights', torch.as_tensor(branch_weights, dtype=torch.float32))
        self.lambda_group = lambda_group
        self.lambda_branch = lambda_branch
        self.lambda_c4 = lambda_c4
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    @staticmethod
    def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        oh = F.one_hot(labels.clamp(0, num_classes - 1), num_classes)
        return oh.movedim(-1, 1).to(torch.float32)

    def forward(self, final_probs, group_logits, branch_a_logits, c4_logits, labels):
        labels = labels.long()

        # --- Final: NLL + Dice over 6 exclusive classes ---
        log_probs = torch.log(final_probs.clamp(min=1e-7))
        final_nll = F.nll_loss(log_probs, labels, weight=self.final_weights)
        final_oh = self._onehot(labels, final_probs.shape[1])
        final_dice = _soft_dice_from_probs(final_probs, final_oh, self.final_weights)
        loss_final = self.weight_ce * final_nll + self.weight_dice * final_dice

        # --- Group: bg / nerve_body(c1-4) / c5 ---
        group_labels = torch.zeros_like(labels)
        group_labels[(labels >= 1) & (labels <= 4)] = 1
        group_labels[labels == 5] = 2
        group_ce = F.cross_entropy(group_logits, group_labels, weight=self.group_weights)
        group_probs = torch.softmax(group_logits, dim=1)
        group_oh = self._onehot(group_labels, 3)
        group_dice = _soft_dice_from_probs(group_probs, group_oh, self.group_weights)
        loss_group = self.weight_ce * group_ce + self.weight_dice * group_dice

        # --- Branch A: c1/c2/c3/c4, supervised only inside nerve_body ---
        nerve_mask = (labels >= 1) & (labels <= 4)
        if nerve_mask.any():
            branch_labels = torch.full_like(labels, -100)  # ignore_index outside nerve_body
            branch_labels[nerve_mask] = (labels[nerve_mask] - 1)  # c1->0 .. c4->3
            branch_ce = F.cross_entropy(branch_a_logits, branch_labels,
                                        weight=self.branch_weights, ignore_index=-100)
            branch_probs = torch.softmax(branch_a_logits, dim=1)
            m = nerve_mask.unsqueeze(1).to(branch_probs.dtype)
            branch_oh = self._onehot(branch_labels.clamp(min=0), 4) * m
            branch_dice = _soft_dice_from_probs(branch_probs * m, branch_oh, self.branch_weights)
            loss_branch = self.weight_ce * branch_ce + self.weight_dice * branch_dice
        else:
            # No nerve_body voxels in this patch: keep a graph-connected zero
            loss_branch = branch_a_logits.sum() * 0.0

        # --- class4 deep supervision (binary) ---
        c4_target = (labels == 4).to(final_probs.dtype).unsqueeze(1)
        c4_prob = torch.sigmoid(c4_logits)
        c4_bce = F.binary_cross_entropy(c4_prob.clamp(1e-6, 1 - 1e-6), c4_target)
        c4_inter = (c4_prob * c4_target).sum()
        c4_dice = 1.0 - (2.0 * c4_inter + 1e-6) / (c4_prob.sum() + c4_target.sum() + 1e-6)
        loss_c4 = self.weight_ce * c4_bce + self.weight_dice * c4_dice

        total = (loss_final
                 + self.lambda_group * loss_group
                 + self.lambda_branch * loss_branch
                 + self.lambda_c4 * loss_c4)
        return {
            'total': total,
            'final': loss_final.detach(),
            'group': loss_group.detach(),
            'branch': loss_branch.detach(),
            'c4': loss_c4.detach(),
        }


# ==================== 滑窗推理 ====================
@torch.no_grad()
def sliding_window_inference(image, model, patch_size, num_classes, device, step_size=0.5,
                             coord_aware=False):
    """Sliding window inference returning an aggregated [num_classes, D, H, W] softmax.

    For HierarchicalStage2Model the model already returns ``final_probs`` (an
    exclusive 6-class distribution) as ``out[0]``; auxiliary heads are ignored at
    inference so the result is obtained directly by ``argmax`` over the output.
    """
    model.eval()
    orig_shape = image.shape[-3:]
    d, h, w = orig_shape if image.ndim == 3 else image.shape[1:]
    pd, ph, pw = patch_size
    if image.ndim == 3:
        image_work = image[None]
    else:
        image_work = image
    C_in = image_work.shape[0]
    nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
    if nd != d or nh != h or nw != w:
        padded = np.zeros((C_in, nd, nh, nw), dtype=np.float32)
        padded[:, :d, :h, :w] = image_work
        image_work = padded
        d, h, w = nd, nh, nw
    agg_results = np.zeros((num_classes, d, h, w), dtype=np.float32)
    agg_weights = np.zeros((d, h, w), dtype=np.float32)
    if HAS_SCIPY:
        sigmas = [s / 8.0 for s in patch_size]
        tmp = np.zeros(patch_size, dtype=np.float32)
        center = tuple(s // 2 for s in patch_size)
        tmp[center] = 1.0
        gw = gaussian_filter(tmp, sigma=sigmas, mode='constant', cval=0.0)
        gw = (gw / gw.max()).astype(np.float32)
    else:
        gw = np.ones(patch_size, dtype=np.float32)
    import itertools
    steps = []
    for dim_size, p_size in zip([d, h, w], [pd, ph, pw]):
        if dim_size <= p_size:
            steps.append([0])
        else:
            step = max(int(p_size * step_size), 1)
            pos = list(range(0, dim_size - p_size + 1, step))
            if pos[-1] + p_size < dim_size:
                pos.append(dim_size - p_size)
            steps.append(pos)
    positions = list(itertools.product(*steps))
    for z1, y1, x1 in positions:
        patch = image_work[:, z1:z1+pd, y1:y1+ph, x1:x1+pw]
        patch_t = torch.from_numpy(patch[None]).float().to(device)
        if coord_aware:
            # Volume-normalized window coords for SPM (V5) memory bank
            meta = torch.tensor([[z1, y1, x1, d, h, w, 0., 0., 0.]],
                                dtype=torch.float32, device=device)
            out = model(patch_t, None, (meta[:, 0:3], meta[:, 3:6], meta[:, 6:9]))
        else:
            out = model(patch_t)
        if isinstance(out, (list, tuple)):
            # HierarchicalStage2Model: out[0]=softmax; BinaryY: out[0]=y_prob [B,1,...]
            prob = out[0].detach().float().cpu().numpy()[0]
        else:
            prob = torch.softmax(out, dim=1).detach().float().cpu().numpy()[0]
        if prob.ndim == 3:
            prob = prob[None]
        n_out = min(num_classes, prob.shape[0])
        for c in range(n_out):
            agg_results[c, z1:z1+pd, y1:y1+ph, x1:x1+pw] += prob[c] * gw
        agg_weights[z1:z1+pd, y1:y1+ph, x1:x1+pw] += gw
    agg_weights = np.maximum(agg_weights, 1e-8)
    for c in range(num_classes):
        agg_results[c] /= agg_weights
    main_result = agg_results[:, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
    return main_result


def get_curriculum_proto_weight(epoch: int, weight_start: float, weight_end: float,
                                anneal_epochs: int) -> float:
    """Compute curriculum proto_loss_weight with linear annealing.

    Args:
        epoch: Current epoch (0-indexed)
        weight_start: Starting weight (high, for strong clustering signal)
        weight_end: Ending weight (lower, for fine-tuning)
        anneal_epochs: Epochs over which to anneal

    Returns:
        Current proto_loss_weight
    """
    if epoch >= anneal_epochs:
        return weight_end
    # Linear annealing from start to end
    alpha = epoch / anneal_epochs
    return weight_start + (weight_end - weight_start) * alpha


# ==================== 训练函数 ====================
def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_accum_steps=1, scaler=None,
                    args_proto_weight=0.0):
    model.train()
    total_loss, total_dice, n = 0., 0., 0
    optimizer.zero_grad()
    use_scaler = scaler is not None
    for step_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
            logits = model(images, labels)
            logits = logits.float()  # cast to fp32 for stable loss computation
        loss = loss_fn(logits, labels)
        # Add prototype contrastive loss (stored on ViTUNet3D during forward)
        proto_loss = getattr(model.net, '_proto_loss', None)
        if proto_loss is not None:
            loss = loss + args_proto_weight * proto_loss
        loss = loss / grad_accum_steps
        # NaN/Inf detection: skip step if loss is corrupted
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] NaN/Inf loss at step {step_idx}, skipping")
            optimizer.zero_grad()
            n += 1
            continue
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
            if use_scaler:
                scaler.unscale_(optimizer)
            # Check for NaN/Inf gradients before clipping
            skip_step = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    skip_step = True
                    break
            if skip_step:
                print(f"  [WARN] NaN/Inf gradients, skipping optimizer step")
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            dice = multiclass_dice_score(logits, labels, include_background=False)
        total_loss += loss.item() * grad_accum_steps
        total_dice += dice
        n += 1
    return total_loss / max(n, 1), total_dice / max(n, 1)


def train_stage2_one_epoch(model, loader, optimizer, loss_fn, device, grad_accum_steps=1, scaler=None,
                           args_proto_weight=0.0):
    """Train one epoch for Stage2. Supports flat Stage2Model and HierarchicalStage2Model."""
    model.train()
    total_loss, total_dice, n = 0., 0., 0
    optimizer.zero_grad()
    is_hierarchical = isinstance(model, HierarchicalStage2Model)
    use_scaler = scaler is not None

    for step_idx, batch in enumerate(loader):
        ct_img, s1_soft, labels, meta = batch
        images = torch.cat([ct_img, s1_soft], dim=1).to(device)
        labels = labels.to(device)
        meta = meta.to(device)
        coord_info = (meta[:, 0:3], meta[:, 3:6], meta[:, 6:9])

        with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
            output = model(images, labels, coord_info)
        # Cast to fp32 for stable loss computation
        if is_hierarchical:
            final_probs = output[0].float()
            group_logits = output[1].float()
            branch_a_logits = output[2].float()
            c4_logits = output[3].float()
            loss_dict = loss_fn(final_probs, group_logits, branch_a_logits,
                                c4_logits, labels)
            loss = loss_dict['total']
        else:
            logits = output.float()
            loss = loss_fn(logits, labels)

        # Add prototype contrastive loss (stored on ViTUNet3D during forward)
        proto_loss = getattr(model.net, '_proto_loss', None)
        if proto_loss is not None:
            loss = loss + args_proto_weight * proto_loss
        loss = loss / grad_accum_steps

        # NaN/Inf detection: skip step if loss is corrupted
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] NaN/Inf loss at step {step_idx}, skipping")
            optimizer.zero_grad()
            n += 1
            continue
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
            if use_scaler:
                scaler.unscale_(optimizer)
            # Check for NaN/Inf gradients before clipping
            skip_step = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    skip_step = True
                    break
            if skip_step:
                print(f"  [WARN] NaN/Inf gradients, skipping optimizer step")
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            if is_hierarchical:
                final_logits = torch.log(final_probs.clamp(1e-7, 1.0))
                dice = multiclass_dice_score(final_logits, labels, include_background=False)
            else:
                dice = multiclass_dice_score(logits, labels, include_background=False)
        total_loss += loss.item() * grad_accum_steps
        total_dice += dice
        n += 1
    return total_loss / max(n, 1), total_dice / max(n, 1)


# ==================== 验证函数 ====================
@torch.no_grad()
def validate_stage1_sliding(model, cases, device, cache_dir, patch_size):
    model.eval()
    class_dice_sums = [0., 0., 0., 0., 0.]
    class_counts = [0, 0, 0, 0, 0]
    for case_id, _, _ in cases:
        image_mm, label_mm = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        label_orig = np.array(label_mm, dtype=np.int16)
        label = convert_label_to_stage1(label_orig)
        softmax = sliding_window_inference(image, model, patch_size, NUM_CLASSES_STAGE1, device)
        pred = np.argmax(softmax, axis=0)
        for cls_idx in range(1, 6):
            pred_m = (pred == cls_idx)
            gt_m = (label == cls_idx)
            tp = (pred_m & gt_m).sum()
            union = pred_m.sum() + gt_m.sum()
            if union > 0:
                d = 2.0 * tp / union
                class_dice_sums[cls_idx - 1] += d
                class_counts[cls_idx - 1] += 1
    avg_class_dice = [s / max(c, 1) for s, c in zip(class_dice_sums, class_counts)]
    return np.mean(avg_class_dice), avg_class_dice


@torch.no_grad()
def validate_stage1_patch(model, cases, device, cache_dir, patch_size,
                          patches_per_case=4, random_crop_prob=0.33):
    """Patch-based validation for Stage1 (1-channel input, 6-class labels bg+1..5).

    Same deterministic patch sampling as validate_stage2_patch: every val case
    contributes `patches_per_case` fixed crops, so all 31 cases are covered and
    the metric is comparable across epochs.
    """
    model.eval()
    class_dice_sums = [0., 0., 0., 0., 0.]
    class_counts = [0, 0, 0, 0, 0]
    pd, ph, pw = patch_size
    for case_id, _, _ in cases:
        image_mm, label_mm = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        label = convert_label_to_stage1(np.array(label_mm, dtype=np.int16))
        d, h, w = image.shape
        nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
        if (nd, nh, nw) != (d, h, w):
            pad = np.zeros((nd, nh, nw), dtype=np.float32)
            pad[:d, :h, :w] = image
            image = pad
            padl = np.zeros((nd, nh, nw), dtype=np.int16)
            padl[:d, :h, :w] = label
            label = padl
            d, h, w = nd, nh, nw
        try:
            seed = int(case_id) % (2 ** 31)
        except (TypeError, ValueError):
            seed = zlib.crc32(str(case_id).encode()) % (2 ** 31)
        rng = np.random.RandomState(seed)
        fg = np.argwhere(label > 0)
        positions = []
        for _ in range(patches_per_case):
            if len(fg) > 0 and rng.rand() >= random_crop_prob:
                cz, cy, cx = fg[rng.randint(len(fg))]
                z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                x1 = int(np.clip(cx - pw // 2, 0, w - pw))
            else:
                z1 = int(rng.randint(0, max(1, d - pd + 1)))
                y1 = int(rng.randint(0, max(1, h - ph + 1)))
                x1 = int(rng.randint(0, max(1, w - pw + 1)))
            positions.append((z1, y1, x1))
        preds = []
        for i in range(0, len(positions), 2):
            chunk = positions[i:i + 2]
            patch_np = np.stack([image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]
                                 for z1, y1, x1 in chunk])
            patch_t = torch.from_numpy(patch_np[:, None]).float().to(device)
            out = model(patch_t)
            if isinstance(out, (list, tuple)):
                prob = out[0]
            else:
                prob = torch.softmax(out, dim=1)
            preds.append(torch.argmax(prob, dim=1).cpu().numpy().astype(np.int16))
        pred_all = np.concatenate(preds, axis=0)
        for i, (z1, y1, x1) in enumerate(positions):
            lab_i = label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]
            for cls_idx in range(1, 6):
                pred_m = (pred_all[i] == cls_idx)
                gt_m = (lab_i == cls_idx)
                tp = (pred_m & gt_m).sum()
                union = pred_m.sum() + gt_m.sum()
                if union > 0:
                    class_dice_sums[cls_idx - 1] += 2.0 * tp / union
                    class_counts[cls_idx - 1] += 1
    avg_class_dice = [s / max(c, 1) for s, c in zip(class_dice_sums, class_counts)]
    return np.mean(avg_class_dice), avg_class_dice


@torch.no_grad()
def validate_stage2_sliding(model, cases, device, cache_dir, stage1_pred_dir, patch_size):
    """Validate Stage2 with sliding window.

    Reports per-class Dice for c1..c5 plus a nerve_body (c1∪c2∪c3∪c4) union Dice
    as a direct reference against the "merge c4 into c1-3" baseline. The mean Dice
    used for checkpoint selection averages the five foreground classes.
    """
    model.eval()
    # 6 quantities to report: c1, c2, c3, c4, c5, nerve_body(c1-4 union)
    class_dice_sums = [0., 0., 0., 0., 0., 0.]
    class_counts = [0, 0, 0, 0, 0, 0]
    stage1_pred_dir = Path(stage1_pred_dir)
    for case_id, _, _ in cases:
        image_mm, label_mm = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        label_orig = np.array(label_mm, dtype=np.int16)  # original labels 0-5
        s1_path = stage1_pred_dir / f"{case_id}_stage1_softmax.nrrd"
        s1_npy = stage1_pred_dir / f"{case_id}_stage1_softmax.npy"
        if s1_path.exists():
            stage1_soft = _load_softmax_nrrd(str(s1_path))
        elif s1_npy.exists():
            stage1_soft = np.load(s1_npy, mmap_mode='r')
        else:
            continue
        if _HAS_PROTOCOL_S2:
            stage1_soft = stage1_softmax_to_cascade_prior(
                np.asarray(stage1_soft), spacing_zyx=_PROT_SPACING)
        else:
            stage1_soft = stage1_soft[1:]
        if ZERO_STAGE1_PRIOR:
            stage1_soft = np.zeros_like(stage1_soft, dtype=np.float32)
        combined = np.concatenate([image[None], stage1_soft], axis=0)
        main_softmax = sliding_window_inference(combined, model, patch_size, NUM_CLASSES_STAGE2, device,
                                                coord_aware=True)

        # Exclusive 6-class output: argmax index == original class id {0..5}
        final_pred = np.argmax(main_softmax, axis=0).astype(np.int16)

        # Per-class Dice for original classes 1-5
        for cls_idx in range(1, 6):
            pred_m = (final_pred == cls_idx)
            gt_m = (label_orig == cls_idx)
            tp = (pred_m & gt_m).sum()
            union = pred_m.sum() + gt_m.sum()
            if union > 0:
                d = 2.0 * tp / union
                class_dice_sums[cls_idx - 1] += d
                class_counts[cls_idx - 1] += 1

        # nerve_body union Dice (c1∪c2∪c3∪c4) vs merged baseline
        pred_nb = (final_pred >= 1) & (final_pred <= 4)
        gt_nb = (label_orig >= 1) & (label_orig <= 4)
        union_nb = pred_nb.sum() + gt_nb.sum()
        if union_nb > 0:
            class_dice_sums[5] += 2.0 * (pred_nb & gt_nb).sum() / union_nb
            class_counts[5] += 1

    avg_class_dice = [s / max(c, 1) for s, c in zip(class_dice_sums, class_counts)]
    # Checkpoint-selection mean over the five foreground classes only
    mean_fg = np.mean(avg_class_dice[:5])
    return mean_fg, avg_class_dice


@torch.no_grad()
def validate_stage2_patch(model, cases, device, cache_dir, stage1_pred_dir, patch_size,
                          patches_per_case=4, random_crop_prob=0.33):
    """Patch-based validation over ALL val cases (deterministic, no sliding window).

    For every val case, draws `patches_per_case` crops with the same
    foreground-centered / uniform-random mixture as training, but from a per-case
    fixed seed so the exact same patches are used at every validation point and the
    metric is comparable across epochs. One forward pass per patch.
    Per-class Dice and checkpoint-selection mean follow validate_stage2_sliding.
    """
    model.eval()
    class_dice_sums = [0., 0., 0., 0., 0., 0.]
    class_counts = [0, 0, 0, 0, 0, 0]
    stage1_pred_dir = Path(stage1_pred_dir)
    pd, ph, pw = patch_size
    for case_id, _, _ in cases:
        image_mm, label_mm = _load_cached_case(case_id, cache_dir, mmap=True)
        s1_path = stage1_pred_dir / f"{case_id}_stage1_softmax.nrrd"
        s1_npy = stage1_pred_dir / f"{case_id}_stage1_softmax.npy"
        if s1_path.exists():
            stage1_soft = _load_softmax_nrrd(str(s1_path))
        elif s1_npy.exists():
            stage1_soft = np.asarray(np.load(s1_npy, mmap_mode='r'))
        else:
            continue
        if _HAS_PROTOCOL_S2:
            stage1_soft = stage1_softmax_to_cascade_prior(
                np.asarray(stage1_soft), spacing_zyx=_PROT_SPACING)
        else:
            stage1_soft = stage1_soft[1:]
        image = np.array(image_mm, dtype=np.float32)
        label_orig = np.array(label_mm, dtype=np.int16)
        if ZERO_STAGE1_PRIOR:
            stage1_soft = np.zeros_like(stage1_soft, dtype=np.float32)
        d, h, w = image.shape
        nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
        if (nd, nh, nw) != (d, h, w):
            pad_img = np.zeros((nd, nh, nw), dtype=np.float32)
            pad_img[:d, :h, :w] = image
            image = pad_img
            pad_s1 = np.zeros((stage1_soft.shape[0], nd, nh, nw), dtype=np.float32)
            pad_s1[:, :d, :h, :w] = stage1_soft
            stage1_soft = pad_s1
            pad_lab = np.zeros((nd, nh, nw), dtype=np.int16)
            pad_lab[:d, :h, :w] = label_orig
            label_orig = pad_lab
            d, h, w = nd, nh, nw
        combined = np.concatenate([image[None], stage1_soft], axis=0)
        # Deterministic per-case patch positions (same patches every epoch)
        try:
            seed = int(case_id) % (2 ** 31)
        except (TypeError, ValueError):
            seed = zlib.crc32(str(case_id).encode()) % (2 ** 31)
        rng = np.random.RandomState(seed)
        fg = np.argwhere(label_orig > 0)
        positions = []
        for _ in range(patches_per_case):
            if len(fg) > 0 and rng.rand() >= random_crop_prob:
                cz, cy, cx = fg[rng.randint(len(fg))]
                z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                x1 = int(np.clip(cx - pw // 2, 0, w - pw))
            else:
                z1 = int(rng.randint(0, max(1, d - pd + 1)))
                y1 = int(rng.randint(0, max(1, h - ph + 1)))
                x1 = int(rng.randint(0, max(1, w - pw + 1)))
            positions.append((z1, y1, x1))
        # Forward in chunks of 2 (matches training batch memory footprint)
        preds = []
        for i in range(0, len(positions), 2):
            chunk = positions[i:i + 2]
            patch_np = np.stack([combined[:, z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]
                                 for z1, y1, x1 in chunk])
            patch_t = torch.from_numpy(patch_np).float().to(device)
            meta = torch.tensor([[z1, y1, x1, d, h, w, 0., 0., 0.] for z1, y1, x1 in chunk],
                                dtype=torch.float32, device=device)
            out = model(patch_t, None, (meta[:, 0:3], meta[:, 3:6], meta[:, 6:9]))
            if isinstance(out, (list, tuple)):
                prob = out[0]  # HierarchicalStage2Model: final_probs already softmax
            else:
                prob = torch.softmax(out, dim=1)
            preds.append(torch.argmax(prob, dim=1).cpu().numpy().astype(np.int16))
        pred_all = np.concatenate(preds, axis=0)
        for i, (z1, y1, x1) in enumerate(positions):
            lab_i = label_orig[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]
            for cls_idx in range(1, 6):
                pred_m = (pred_all[i] == cls_idx)
                gt_m = (lab_i == cls_idx)
                tp = (pred_m & gt_m).sum()
                union = pred_m.sum() + gt_m.sum()
                if union > 0:
                    class_dice_sums[cls_idx - 1] += 2.0 * tp / union
                    class_counts[cls_idx - 1] += 1
            pred_nb = (pred_all[i] >= 1) & (pred_all[i] <= 4)
            gt_nb = (lab_i >= 1) & (lab_i <= 4)
            union_nb = pred_nb.sum() + gt_nb.sum()
            if union_nb > 0:
                class_dice_sums[5] += 2.0 * (pred_nb & gt_nb).sum() / union_nb
                class_counts[5] += 1
    avg_class_dice = [s / max(c, 1) for s, c in zip(class_dice_sums, class_counts)]
    # Checkpoint-selection mean over the five foreground classes only
    mean_fg = np.mean(avg_class_dice[:5])
    return mean_fg, avg_class_dice


# ==================== NRRD I/O + Stage1 CV Predictions ====================
def _save_softmax_nrrd(softmax: np.ndarray, save_path: str):
    if nrrd is None:
        raise ImportError("pynrrd not installed. Run: pip install pynrrd")
    header = {
        'type': 'float', 'dimension': 4, 'sizes': list(softmax.shape),
        'kinds': ['list', 'domain', 'domain', 'domain'],
        'encoding': 'gzip', 'endian': 'little',
    }
    nrrd.write(str(save_path), softmax.astype(np.float32), header)


def _save_label_nrrd_with_space(label: np.ndarray, save_path: str, affine: np.ndarray):
    """Save integer label map as NRRD (3D, int16) with RAS->LPS spatial metadata.

    Args:
        label:      (H, W, D) int16 array in NIfTI orientation
        save_path:  output .nrrd path
        affine:     4x4 NIfTI affine (RAS+ convention)
    """
    if nrrd is None:
        raise ImportError("pynrrd not installed. Run: pip install pynrrd")
    # RAS+ (NIfTI) -> LPS (NRRD/3D Slicer): negate X and Y rows
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps_affine = ras_to_lps @ affine
    # space directions: each row = one voxel axis direction in LPS world
    space_directions = lps_affine[:3, :3].T  # (3, 3)
    space_origin = lps_affine[:3, 3]
    header = {
        'type': 'short', 'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': list(label.shape),
        'space directions': space_directions.tolist(),
        'space origin': space_origin.tolist(),
        'kinds': ['domain', 'domain', 'domain'],
        'encoding': 'gzip', 'endian': 'little',
    }
    nrrd.write(str(save_path), label.astype(np.int16), header)


def _save_label_nrrd(label: np.ndarray, save_path: str):
    """Save integer label map as NRRD (3D, int16)."""
    if nrrd is None:
        raise ImportError("pynrrd not installed. Run: pip install pynrrd")
    header = {
        'type': 'short', 'dimension': 3,
        'sizes': list(label.shape),
        'kinds': ['domain', 'domain', 'domain'],
        'encoding': 'gzip', 'endian': 'little',
    }
    nrrd.write(str(save_path), label.astype(np.int16), header)


def _load_softmax_nrrd(nrrd_path: str) -> np.ndarray:
    if nrrd is None:
        raise ImportError("pynrrd not installed. Run: pip install pynrrd")
    data, _ = nrrd.read(str(nrrd_path))
    return data.astype(np.float32)


@torch.no_grad()
def generate_stage1_predictions(model, cases, save_dir, patch_size, cache_dir, device):
    """Generate Stage1 softmax AND argmax label predictions as NRRD."""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for case_id, _, _ in cases:
        image_mm, _ = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        softmax = sliding_window_inference(image, model, patch_size, NUM_CLASSES_STAGE1, device)
        _save_softmax_nrrd(softmax, str(save_dir / f"{case_id}_stage1_softmax.nrrd"))
        # Also save argmax label prediction as NRRD
        label_pred = np.argmax(softmax, axis=0).astype(np.int16)
        _save_label_nrrd(label_pred, str(save_dir / f"{case_id}_stage1_label.nrrd"))
        print(f"    Saved: {case_id} softmax + label (NRRD, shape={softmax.shape})")


@torch.no_grad()
def generate_stage2_label_predictions(model, cases, stage1_pred_dir, save_dir, patch_size, cache_dir, device):
    """Generate and save Stage2 label predictions as NRRD.

    The hierarchical soft-tree produces an exclusive 6-class distribution whose
    argmax index already equals the original class id {0:bg, 1-5}, so the label
    map is obtained by a single argmax (no auxiliary thresholding / priority).
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    binary_y = isinstance(model, BinaryYStage2Model) if _HAS_PROTOCOL_S2 else False
    for case_id, _, _ in cases:
        image_mm, _ = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        combined = _stage2_input(case_id, image, stage1_pred_dir)
        if combined is None:
            print(f"    [SKIP] No stage1 prediction for {case_id}")
            continue
        if binary_y:
            # Binary Y head: label 1 = merged CNV Y (1∪2∪3∪4); class5_bypass is not predicted.
            y_map = sliding_window_inference(combined, model, patch_size, 1, device, coord_aware=True)
            label_pred = (y_map[0] > 0.5).astype(np.int16)
        else:
            main_softmax = sliding_window_inference(combined, model, patch_size, NUM_CLASSES_STAGE2, device)
            # Exclusive 6-class output: argmax index == original class id {0..5}
            label_pred = np.argmax(main_softmax, axis=0).astype(np.int16)

        _save_label_nrrd(label_pred, str(save_dir / f"{case_id}_stage2_label.nrrd"))
        print(f"    Saved: {case_id} stage2 label (NRRD, shape={label_pred.shape})")


def _stage2_input(case_id, image, stage1_pred_dir):
    """Stage-2 input volume for one case: CT only (B0) or [CT, cascade prior].

    Applies the same prior conversion, A2 scramble and zero-prior switches as
    Stage2Dataset so evaluation sees exactly the training-time input. Returns
    None when the Stage-1 prediction is missing.
    """
    if B0_SINGLE_STAGE or stage1_pred_dir is None:
        return image[None]
    stage1_pred_dir = Path(stage1_pred_dir)
    s1_path = stage1_pred_dir / f"{case_id}_stage1_softmax.nrrd"
    s1_npy = stage1_pred_dir / f"{case_id}_stage1_softmax.npy"
    if s1_path.exists():
        stage1_soft = _load_softmax_nrrd(str(s1_path))
    elif s1_npy.exists():
        stage1_soft = np.asarray(np.load(s1_npy, mmap_mode='r'))
    else:
        return None
    if _HAS_PROTOCOL_S2:
        stage1_soft = stage1_softmax_to_cascade_prior(
            np.asarray(stage1_soft), spacing_zyx=_PROT_SPACING)
    else:
        stage1_soft = stage1_soft[1:]
    if PRIOR_SCRAMBLE_MODE not in ('0', 'none', 'off', '', None):
        try:
            seed = int(case_id) % (2 ** 31)
        except (TypeError, ValueError):
            seed = zlib.crc32(str(case_id).encode()) % (2 ** 31)
        stage1_soft = scramble_cascade_prior(
            stage1_soft, mode=PRIOR_SCRAMBLE_MODE,
            rng=np.random.RandomState(seed ^ 0xA2A2))
    if ZERO_STAGE1_PRIOR:
        stage1_soft = np.zeros_like(stage1_soft, dtype=np.float32)
    return np.concatenate([image[None], stage1_soft], axis=0)


# ==================== Layer-wise LR Decay ====================
def _build_param_groups(model, base_lr, layer_decay=0.75):
    """Build parameter groups with layer-wise LR decay for ViT.

    Deeper (earlier) encoder stages get lower LR.
    Decoder and heads use full LR.
    """
    # Identify parameter groups
    encoder_stage0_params = []  # deepest encoder
    encoder_stage1_params = []
    other_params = []  # encoder_stage2, decoder, heads, memory bank

    net = model.net if hasattr(model, 'net') else model

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'net.encoder_stages.0' in name or 'net.patch_embed' in name:
            encoder_stage0_params.append(param)
        elif 'net.encoder_stages.1' in name or 'net.patch_merges.0' in name:
            encoder_stage1_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': encoder_stage0_params, 'lr_scale': layer_decay ** 2},
        {'params': encoder_stage1_params, 'lr_scale': layer_decay},
        {'params': other_params, 'lr_scale': 1.0},
    ]
    # Filter out empty groups
    param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
    return param_groups




# ==================== Protocol binary-Y Stage2 (primary) ====================

@torch.no_grad()
def validate_stage2_binary_y_patch(model, cases, device, cache_dir, stage1_pred_dir, patch_size,
                                   patches_per_case=4, random_crop_prob=0.33):
    """Protocol Primary val: Dice_Y on binary Y (class5_bypass excluded). §13 ②."""
    model.eval()
    dice_sums, n_cases = 0.0, 0
    stage1_pred_dir = Path(stage1_pred_dir) if stage1_pred_dir else None
    pd, ph, pw = patch_size
    for case_id, _, _ in cases:
        image_mm, label_mm = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        label_orig = np.array(label_mm, dtype=np.int16)
        gt_y = ((label_orig >= 1) & (label_orig <= 4))
        combined = _stage2_input(case_id, image, stage1_pred_dir)
        if combined is None:
            continue
        d, h, w = image.shape
        try:
            seed = int(case_id) % (2 ** 31)
        except (TypeError, ValueError):
            seed = zlib.crc32(str(case_id).encode()) % (2 ** 31)
        rng = np.random.RandomState(seed)
        fg = np.argwhere(gt_y)
        case_dices = []
        for _ in range(patches_per_case):
            if len(fg) > 0 and rng.rand() >= random_crop_prob:
                cz, cy, cx = fg[rng.randint(len(fg))]
                z1 = int(np.clip(cz - pd // 2, 0, max(0, d - pd)))
                y1 = int(np.clip(cy - ph // 2, 0, max(0, h - ph)))
                x1 = int(np.clip(cx - pw // 2, 0, max(0, w - pw)))
            else:
                z1 = rng.randint(0, max(1, d - pd + 1))
                y1 = rng.randint(0, max(1, h - ph + 1))
                x1 = rng.randint(0, max(1, w - pw + 1))
            z2, y2, x2 = min(z1 + pd, d), min(y1 + ph, h), min(x1 + pw, w)
            patch = combined[:, z1:z2, y1:y2, x1:x2]
            if patch.shape[1:] != (pd, ph, pw):
                pad = np.zeros((patch.shape[0], pd, ph, pw), dtype=np.float32)
                pad[:, :patch.shape[1], :patch.shape[2], :patch.shape[3]] = patch
                patch = pad
            gt_p = gt_y[z1:z2, y1:y2, x1:x2]
            if gt_p.shape != (pd, ph, pw):
                gpad = np.zeros((pd, ph, pw), dtype=bool)
                gpad[:gt_p.shape[0], :gt_p.shape[1], :gt_p.shape[2]] = gt_p
                gt_p = gpad
            t = torch.from_numpy(patch[None]).float().to(device)
            out = model(t)
            y_prob = out[0] if isinstance(out, (list, tuple)) else torch.sigmoid(out)
            pred = (y_prob[0, 0].detach().float().cpu().numpy() > 0.5)
            inter = (pred & gt_p).sum()
            denom = pred.sum() + gt_p.sum()
            if denom > 0:
                case_dices.append(2.0 * inter / denom)
        if case_dices:
            dice_sums += float(np.mean(case_dices))
            n_cases += 1
    mean_y = dice_sums / max(n_cases, 1)
    return mean_y, [mean_y, float('nan'), float('nan'), float('nan'), float('nan'), mean_y]


@torch.no_grad()
def validate_stage2_binary_y_sliding(model, cases, device, cache_dir, stage1_pred_dir, patch_size):
    """Full-volume binary-Y sliding-window Dice_Y (Primary). §13 ②."""
    model.eval()
    dice_sums, n_cases = 0.0, 0
    stage1_pred_dir = Path(stage1_pred_dir) if stage1_pred_dir else None
    for case_id, _, _ in cases:
        image_mm, label_mm = _load_cached_case(case_id, cache_dir, mmap=True)
        image = np.array(image_mm, dtype=np.float32)
        label_orig = np.array(label_mm, dtype=np.int16)
        gt_y = ((label_orig >= 1) & (label_orig <= 4))
        combined = _stage2_input(case_id, image, stage1_pred_dir)
        if combined is None:
            continue
        y_map = sliding_window_inference(combined, model, patch_size, 1, device, coord_aware=True)
        pred = y_map[0] > 0.5
        inter = (pred & gt_y).sum()
        denom = pred.sum() + gt_y.sum()
        if denom > 0:
            dice_sums += 2.0 * inter / denom
            n_cases += 1
    mean_y = dice_sums / max(n_cases, 1)
    return mean_y, [mean_y, float('nan'), float('nan'), float('nan'), float('nan'), mean_y]


def build_binary_y_stage2_model(args, use_mb, use_gate=True, in_channels=3):
    """Construct BinaryYStage2Model (Y=1U2U3U4; class5_bypass excluded)."""
    if not _HAS_PROTOCOL_S2:
        raise RuntimeError("cnv_protocol_s2 required for --s2_head binary_y")
    mb = None
    if use_mb:
        mb = _build_memory_bank(
            2, args.num_prototypes_per_class,
            args.memory_momentum, args.memory_temperature, args.memory_top_k,
            retrieval_mode=args.retrieval_mode, memory_type=args.memory_type,
            feature_dims=NNUNET_MLA_MEMORY_DIMS if args.backbone == "nnunet_mla" else None,
        )
        print(f"OK BinaryY Memory Bank ({args.memory_type})")
    net = _build_vit(in_channels, 1, memory_bank=mb, use_mla=bool(args.use_mla),
                     backbone=args.backbone, dual_path_stages=args.dual_path_stages,
                     residual_conv_stages=args.residual_conv_stages)
    stem_ch = VIT_STEM_CHANNELS
    if getattr(net, "provides_fullres_features", False):
        hier_decoder = nn.Sequential(
            nn.Conv3d(32, stem_ch[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(stem_ch[0]),
            nn.GELU(),
        )
    else:
        hier_decoder = CNNDecoder3D(
            embed_dim=VIT_EMBED_DIM,
            stem_channels=VIT_STEM_CHANNELS,
            tokenize_kernel=VIT_TOKENIZE_KERNEL,
            num_classes=stem_ch[0],
        )
        hier_decoder.out_conv = nn.Sequential(
            nn.Conv3d(stem_ch[0], stem_ch[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(stem_ch[0]),
            nn.GELU(),
        )
    mem_dim = NNUNET_MLA_MEMORY_DIMS[0] if args.backbone == "nnunet_mla" else None
    m = BinaryYStage2Model(net, hier_decoder, feat_dim=stem_ch[0],
                           use_confidence_gate=use_gate, in_channels=in_channels,
                           gate_alpha_fixed=GATE_ALPHA_FIXED,
                           bone_thr=float(getattr(args, 'bone_norm_thr', DEFAULT_BONE_NORM_THR)),
                           evidence_temp=float(getattr(args, 'evidence_temp', DEFAULT_EVIDENCE_TEMP)),
                           mem_dim=mem_dim)
    print(f'[GATE] mode={m.gate_mode} tau={GATE_TAU} alpha_fixed={GATE_ALPHA_FIXED}')
    return m


def train_binary_y_one_epoch(model, loader, optimizer, loss_fn, device,
                             grad_accum_steps=1, scaler=None, args_proto_weight=0.0):
    """Binary-Y train epoch with the evidence-gated loss (protocol §2.3)."""
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        image, s1, label, meta = batch
        image = image.to(device, non_blocking=True)
        s1 = s1.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        if B0_SINGLE_STAGE or (hasattr(model, 'in_channels') and model.in_channels == 1):
            x = image
        else:
            x = torch.cat([image, s1], dim=1)
        ignore = (label < 0)
        tgt = label.clamp(min=0)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            _y_prob, y_logits, g, y_mem_logits = model(
                x, tgt if args_proto_weight > 0 else None,
                (meta[:, 0:3].to(device), meta[:, 3:6].to(device), meta[:, 6:9].to(device)),
            )
        losses = loss_fn(y_logits, tgt.float(), ignore_mask=ignore, g=g,
                         y_mem_logits=y_mem_logits)
        loss = losses["total"]
        proto_loss = getattr(model.net, '_proto_loss', None)
        if proto_loss is not None and args_proto_weight > 0:
            loss = loss + args_proto_weight * proto_loss.float()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(losses["total"].detach())
        n += 1
    return total_loss / max(n, 1), 0.0


# ==================== External Inference ====================
def _preprocess_ct_for_inference(ct_nii, target_spacing):
    """Preprocess a raw CT NIfTI for inference (matches training pipeline).

    Steps:
      1. Transpose (H, W, D) -> (D, H, W)
      2. Resample to target_spacing
      3. Percentile normalization

    Returns:
        image_norm:       preprocessed image (D, H, W) float32
        original_spacing: (z, y, x) in mm
        original_shape:   (D, H, W) before resampling
    """
    image = ct_nii.get_fdata().astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # (H,W,D) -> (D,H,W)
    original_shape = image.shape

    nii_zooms = ct_nii.header.get_zooms()[:3]  # (x, y, z)
    original_spacing = (float(nii_zooms[2]), float(nii_zooms[1]), float(nii_zooms[0]))  # (z, y, x)

    if target_spacing is not None:
        image = pp.resample_image_to_spacing(image, original_spacing, target_spacing, order=3)

    image = normalize_percentile(image)
    return image, original_spacing, original_shape


def _resample_pred_back(pred, original_spacing, target_spacing, original_shape_dhw):
    """Resample prediction back to original spacing/shape then transpose for NIfTI."""
    if target_spacing is not None:
        pred_orig = pp.resample_image_to_spacing(
            pred.astype(np.float32), target_spacing, original_spacing, order=0
        ).astype(np.int16)
    else:
        pred_orig = pred.astype(np.int16)

    # Crop or pad to exactly original_shape_dhw
    result = np.zeros(original_shape_dhw, dtype=np.int16)
    slices_src, slices_dst = [], []
    for i in range(3):
        s = min(pred_orig.shape[i], original_shape_dhw[i])
        slices_src.append(slice(0, s))
        slices_dst.append(slice(0, s))
    result[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        pred_orig[slices_src[0], slices_src[1], slices_src[2]]

    # Transpose (D, H, W) -> (H, W, D) for NIfTI saving
    return np.transpose(result, (1, 2, 0))


@torch.no_grad()
def run_inference_external(args, device, target_spacing):
    """Run two-stage cascade inference on external CT NIfTI files.

    Loads all folds (ensemble) or a single fold, performs Stage1 + Stage2
    inference with sliding window, resamples back to original space, and
    saves NIfTI label predictions.
    """
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if input_dir is None or output_dir is None:
        raise ValueError("--input_dir and --output_dir are required for inference stage")

    output_dir.mkdir(parents=True, exist_ok=True)
    use_mb = bool(args.use_memory_bank)
    use_hier = bool(args.use_hierarchical)
    # PROTOCOL: default --s2_head binary_y (SoftTree = legacy soft_tree)
    s2_head = getattr(args, 's2_head', 'binary_y')

    # Discover CT files
    ct_files = sorted(input_dir.glob("*.nii.gz")) + sorted(input_dir.glob("*.nii"))
    if not ct_files:
        print(f"No .nii.gz or .nii files found in {input_dir}")
        return
    print(f"Found {len(ct_files)} CT files in {input_dir}")

    # Load models (ensemble or single fold)
    stage1_models = []
    stage2_models = []
    folds_to_load = list(range(NUM_FOLDS)) if args.ensemble else [args.fold]

    s1_suffix = args.stage1_exp_suffix if args.stage1_exp_suffix is not None else args.exp_suffix
    s1_exp = f"{EXPERIMENT_NAME_STAGE1_TEMPLATE}{s1_suffix}"
    s2_exp = f"{EXPERIMENT_NAME_STAGE2_TEMPLATE}{args.exp_suffix}"
    use_binary_y = (s2_head == 'binary_y')
    print(f"\nLoading models ({'ensemble' if args.ensemble else f'fold {args.fold}'})...")
    print(f"  Stage1 exp: {s1_exp}\n  Stage2 exp: {s2_exp} (s2_head={s2_head})")
    for fold_idx in folds_to_load:
        # Stage1
        s1_path = Path(RESULTS_ROOT) / s1_exp / f"fold{fold_idx}" / "best_model.pth"
        if B0_SINGLE_STAGE:
            pass
        elif s1_path.exists():
            model_s1 = Stage1Model(
                use_memory_bank=use_mb,
                num_prototypes_per_class=args.num_prototypes_per_class,
                memory_momentum=args.memory_momentum,
                memory_temperature=args.memory_temperature,
                memory_top_k=args.memory_top_k,
                use_mla=bool(args.use_mla),
                retrieval_mode=args.retrieval_mode,
                memory_type=args.memory_type,
                backbone=args.backbone,
                dual_path_stages=args.dual_path_stages,
                residual_conv_stages=args.residual_conv_stages,
            ).to(device)
            ckpt = torch.load(s1_path, map_location=device, weights_only=False)
            model_s1.load_state_dict(ckpt.get('model_state_dict', ckpt))
            model_s1.eval()
            stage1_models.append(model_s1)
            print(f"  Stage1 fold{fold_idx}: loaded")
        else:
            print(f"  Stage1 fold{fold_idx}: NOT FOUND ({s1_path}), skip")

        # Stage2
        s2_path = Path(RESULTS_ROOT) / s2_exp / f"fold{fold_idx}" / "best_model.pth"
        if s2_path.exists():
            if use_binary_y:
                model_s2 = build_binary_y_stage2_model(
                    args, use_mb, use_gate=bool(args.use_confidence_gate),
                    in_channels=1 if B0_SINGLE_STAGE else (2 if PRIOR_FG_ONLY else 3),
                ).to(device)
            elif use_hier:
                model_s2 = HierarchicalStage2Model(
                    use_memory_bank=use_mb,
                    num_prototypes_per_class=args.num_prototypes_per_class,
                    memory_momentum=args.memory_momentum,
                    memory_temperature=args.memory_temperature,
                    memory_top_k=args.memory_top_k,
                    use_mla=bool(args.use_mla),
                    retrieval_mode=args.retrieval_mode,
                    memory_type=args.memory_type,
                    backbone=args.backbone,
                    dual_path_stages=args.dual_path_stages,
                    residual_conv_stages=args.residual_conv_stages,
                    in_channels=(2 if PRIOR_FG_ONLY else 3),
                ).to(device)
            else:
                model_s2 = Stage2Model(
                    use_memory_bank=use_mb,
                    num_prototypes_per_class=args.num_prototypes_per_class,
                    memory_momentum=args.memory_momentum,
                    memory_temperature=args.memory_temperature,
                    memory_top_k=args.memory_top_k,
                    use_mla=bool(args.use_mla),
                    retrieval_mode=args.retrieval_mode,
                    memory_type=args.memory_type,
                    backbone=args.backbone,
                    dual_path_stages=args.dual_path_stages,
                    residual_conv_stages=args.residual_conv_stages,
                ).to(device)
            ckpt = torch.load(s2_path, map_location=device, weights_only=False)
            model_s2.load_state_dict(ckpt.get('model_state_dict', ckpt))
            model_s2.eval()
            stage2_models.append(model_s2)
            print(f"  Stage2 fold{fold_idx}: loaded")
        else:
            print(f"  Stage2 fold{fold_idx}: NOT FOUND ({s2_path}), skip")

    if not stage1_models and not B0_SINGLE_STAGE:
        raise FileNotFoundError("No Stage1 models found. Cannot run inference.")
    if not stage2_models:
        raise FileNotFoundError("No Stage2 models found. Cannot run inference.")
    print(f"\nStage1 models: {len(stage1_models)}, Stage2 models: {len(stage2_models)}")

    # Run inference on each CT
    print(f"\n{'='*60}")
    print(f"Cascade Inference (ViT+MLA+MemoryBank)")
    print(f"{'='*60}")

    CLASS_NAMES = (['bg', 'CNV_Y(1-4)'] if use_binary_y else
                   ['bg', 'c1(V1_CN)', 'c2(V2_CN)', 'c3(V3_CN)', 'c4(foramen)', 'c5(bypass)'])

    for ct_path in ct_files:
        patient_id = ct_path.stem.replace('.nii', '')
        print(f"\n[{patient_id}] {ct_path.name}")

        try:
            ct_nii = nib.load(str(ct_path))
            affine = ct_nii.affine
            print(f"  Original shape: {ct_nii.shape}, spacing: {ct_nii.header.get_zooms()[:3]}")

            # Preprocess
            image_norm, original_spacing, original_shape_dhw = \
                _preprocess_ct_for_inference(ct_nii, target_spacing)
            print(f"  After preprocess: shape={image_norm.shape}")

            if B0_SINGLE_STAGE:
                combined_input = image_norm[None]
            else:
                # Stage1: ensemble softmax (bg + classes 1..5)
                print(f"  Stage1 inference ({len(stage1_models)} models)...")
                s1_probs_list = []
                for m in stage1_models:
                    s1_soft = sliding_window_inference(
                        image_norm, m, PATCH_SIZE, NUM_CLASSES_STAGE1, device)
                    s1_probs_list.append(s1_soft)
                s1_prob = np.mean(s1_probs_list, axis=0)
                if _HAS_PROTOCOL_S2:
                    prior = stage1_softmax_to_cascade_prior(s1_prob, spacing_zyx=_PROT_SPACING)
                else:
                    prior = s1_prob[1:]
                combined_input = np.concatenate([image_norm[None], prior], axis=0)

            print(f"  Stage2 inference ({len(stage2_models)} models)...")
            n_out = 1 if use_binary_y else NUM_CLASSES_STAGE2
            s2_probs_list = []
            for m in stage2_models:
                s2_soft = sliding_window_inference(
                    combined_input, m, PATCH_SIZE, n_out, device, coord_aware=use_binary_y)
                s2_probs_list.append(s2_soft)
            s2_prob = np.mean(s2_probs_list, axis=0)

            if use_binary_y:
                # Label 1 = merged CNV Y (1∪2∪3∪4)
                s2_pred = (s2_prob[0] > 0.5).astype(np.int16)
            else:
                s2_pred = np.argmax(s2_prob, axis=0).astype(np.int16)

            # Resample back to original space
            label_final = _resample_pred_back(
                s2_pred, original_spacing, target_spacing, original_shape_dhw)

            # Save as NRRD (with RAS->LPS spatial metadata)
            out_path = output_dir / f"{patient_id}.nrrd"
            _save_label_nrrd_with_space(label_final, str(out_path), affine)

            # Print class distribution
            uniq, cnts = np.unique(label_final, return_counts=True)
            dist_str = ', '.join(f"{CLASS_NAMES[int(u)]}:{int(c)}" for u, c in zip(uniq, cnts) if int(u) < len(CLASS_NAMES))
            print(f"  Saved: {out_path.name} | {dist_str}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"{'='*60}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="MIA v3 ViT + MLA Memory Bank")
    parser.add_argument("--stage", type=str, required=True, choices=['0', '1', '2', 'generate_cv', 'inference'])
    parser.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--stage1_pred_dir", type=str, default=None)
    parser.add_argument("--force_cache", action="store_true")
    parser.add_argument("--use_memory_bank", type=int, default=1, help="Enable Memory Bank (1=on, 0=off)")
    parser.add_argument("--num_prototypes_per_class", type=int, default=NUM_PROTOTYPES_PER_CLASS)
    parser.add_argument("--memory_top_k", type=int, default=MEMORY_TOP_K, help="Number of top prototypes to retrieve")
    parser.add_argument("--memory_momentum", type=float, default=MEMORY_MOMENTUM)
    parser.add_argument("--memory_temperature", type=float, default=MEMORY_TEMPERATURE)
    parser.add_argument("--exp_suffix", type=str, default="", help="Suffix for experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Specific checkpoint path to resume from (overrides default latest_checkpoint.pth)")
    parser.add_argument("--s2_head", type=str, default="binary_y",
                        choices=["binary_y", "soft_tree"],
                        help="Primary S2 head: binary_y or legacy soft_tree")
    parser.add_argument("--prior_scramble", type=str, default="0",
                        choices=["0", "none", "off", "spatial_roll", "channel_permute", "drop"],
                        help="A2: scramble cascade prior (bony_roi class4)")
    parser.add_argument("--b0_single_stage", type=int, default=0,
                        help="B0: 1=CT-only in_channels=1 binary Y")
    parser.add_argument("--gate_tau", type=float, default=0.5,
                        help="Evidence/gate threshold tau (frozen)")
    parser.add_argument("--gate_alpha_fixed", type=float, default=None,
                        help="A_gate: freeze fusion alpha constant")
    parser.add_argument("--refuse_legacy_heads", type=int, default=1,
                        help="1=refuse soft_tree/3-class ckpt into binary_y")
    parser.add_argument("--lambda_L", type=float, default=DEFAULT_LAMBDA_L,
                        help="Weight of the low-evidence (g<tau) segmentation term; 0 for C1/B0/A2")
    parser.add_argument("--lambda_mem", type=float, default=DEFAULT_LAMBDA_MEM,
                        help="Weight of aligning Y_hat to the memory decoder Y_mem")
    parser.add_argument("--mem_sup_weight", type=float, default=DEFAULT_MEM_SUP_WEIGHT,
                        help="Weight of supervising the memory decoder Y_mem with Y")
    parser.add_argument("--mem_align", type=str, default="low", choices=["low", "global", "off"],
                        help="Memory alignment region: low=L only (Full), global (A_gate), off")
    parser.add_argument("--bone_norm_thr", type=float, default=DEFAULT_BONE_NORM_THR,
                        help="Normalized-CT bone-window threshold for the G1 evidence map")
    parser.add_argument("--evidence_temp", type=float, default=DEFAULT_EVIDENCE_TEMP,
                        help="Sigmoid temperature of the G1 bone-window proxy")
    parser.add_argument("--stage1_exp_suffix", type=str, default=None,
                        help="Inference: Stage-1 experiment suffix if it differs from --exp_suffix")
    parser.add_argument("--use_hierarchical", type=int, default=1, help="1=Hierarchical (default), 0=flat")
    parser.add_argument("--lambda_group", type=float, default=0.3)
    parser.add_argument("--lambda_branch", type=float, default=0.2)
    parser.add_argument("--lambda_aux", type=float, default=0.5, help="Weight for class4 deep-supervision (bony anchor) loss")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory for inference (containing .nii.gz CT files)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for inference results")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble all folds for inference (recommended)")
    parser.add_argument("--use_confidence_gate", type=int, default=1,
                        help="binary_y: 1=evidence-gated memory fusion (Full), 0=plain memory; "
                             "legacy heads: confidence gate blending")
    parser.add_argument("--use_mla", type=int, default=1, help="1=MLA attention (default), 0=standard MHA for ablation E")
    parser.add_argument("--memory_warmup_epochs", type=int, default=MEMORY_BANK_WARMUP_EPOCHS, help="Epochs before enabling memory bank (ablation I)")
    parser.add_argument("--retrieval_mode", type=str, default='cross_attention', choices=['cross_attention', 'cosine'],
                        help="Memory bank retrieval mode: cross_attention (default) or cosine (ablation J)")
    parser.add_argument("--memory_type", type=str, default='mla', choices=['mla', 'context_anchor', 'anatomical', 'anatomical_v4', 'anatomical_v4_mgp', 'anatomical_v5', 'anatomical_v6'],
                        help="Memory bank type: mla (class-organized), context_anchor (context-organized), anatomical (Sinkhorn-balanced), anatomical_v4 (factorized), anatomical_v4_mgp (minimal gated prototypes), v5 (SPM), v6 (SCM)")
    parser.add_argument("--backbone", type=str, default='vit', choices=['vit', 'nnformer_v5', 'nnunet_mla'],
                        help="Backbone: vit (ViTUNet3D global MLA), nnformer_v5 (nnFormer-style interleaved conv + local/global dual-path MLA), or nnunet_mla (DynUNet/nnU-Net topology + local/global MLA injection)")
    parser.add_argument("--dual_path_stages", type=str, default='0,1',
                        help="nnformer_v5 only: comma-separated encoder-stage indices that use the local+global dual path (default '0,1' = shallow/mid stages; stage2 stays global-only). Use '0,1,2' for full dual-path, 'none' for global-only.")
    parser.add_argument("--residual_conv_stages", type=str, default='0',
                        help="nnformer_v5 only: comma-separated encoder-stage indices whose interleaved conv branch becomes a bottleneck residual conv block (default '0' = shallowest stage only, ~+0.1M params). Use 'none' to disable.")
    parser.add_argument("--random_crop_prob", type=float, default=0.0,
                        help="Stage-2 training: probability of a fully random (possibly all-background) crop; "
                             "nnU-Net-style mixed sampling, e.g. 0.33. Default 0.0 = always foreground-biased.")
    parser.add_argument("--zero_stage1_prior", type=int, default=0,
                        help="Ablation: 1=replace Stage-1 prior channels with constant zeros (Stage2 predicts from CT alone); 0=normal cascade")
    parser.add_argument("--prior_dropout_prob", type=float, default=0.0,
                        help="Stage-2 training: per-sample probability of zeroing the Stage-1 prior "
                             "channels (anti-shortcut-learning; e.g. 0.3). Validation/inference unaffected.")
    parser.add_argument("--prior_fg_only", type=int, default=0,
                        help="1=prior npy carries [bg, fg] only -> single foreground prior channel "
                             "(model in_channels=2); 0=legacy 2-channel group prior (in_channels=3)")
    parser.add_argument("--nnunet_loss", type=int, default=0,
                        help="1=flat branch uses the exact nnU-Net baseline loss "
                             "(WeightedBCEDiceLoss 0.3/0.7, weights [0.1,10,8,6,3,3], "
                             "focal a=0.25 g=3.0) for apples-to-apples architecture comparison")
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE,
                        help="Epochs without val-dice improvement before early stop")
    parser.add_argument("--proto_loss_weight", type=float, default=0.1,
                        help="Weight for prototype contrastive loss (0 to disable)")
    parser.add_argument("--proto_init_from_features", action="store_true",
                        help="Initialize prototypes from class-wise feature statistics")
    parser.add_argument("--proto_weight_start", type=float, default=None,
                        help="Starting proto_loss_weight for curriculum annealing (overrides --proto_loss_weight)")
    parser.add_argument("--proto_weight_end", type=float, default=None,
                        help="Ending proto_loss_weight for curriculum annealing")
    parser.add_argument("--proto_anneal_epochs", type=int, default=100,
                        help="Epochs over which to anneal proto_loss_weight")
    args = parser.parse_args()

    # Parse comma-separated dual-path stage indices (nnformer_v5 backbone only).
    # 'none'/'off'/'' all mean "no dual path" (global attention only) — accepting a
    # keyword avoids passing an empty string through shell scripts.
    _dps = str(args.dual_path_stages).strip()
    if _dps.lower() in ('', 'none', 'off'):
        args.dual_path_stages = []
    else:
        args.dual_path_stages = [int(s) for s in _dps.split(',') if s.strip() != '']

    # Same parsing for the residual conv branch stages.
    _rcs = str(args.residual_conv_stages).strip()
    if _rcs.lower() in ('', 'none', 'off'):
        args.residual_conv_stages = []
    else:
        args.residual_conv_stages = [int(s) for s in _rcs.split(',') if s.strip() != '']

    # Ablation: zero out the Stage-1 prior channels (module-level switch)
    global ZERO_STAGE1_PRIOR
    ZERO_STAGE1_PRIOR = bool(args.zero_stage1_prior)
    global PRIOR_DROPOUT_PROB
    PRIOR_DROPOUT_PROB = float(args.prior_dropout_prob)
    global PRIOR_FG_ONLY
    PRIOR_FG_ONLY = bool(args.prior_fg_only)
    global PRIOR_SCRAMBLE_MODE, B0_SINGLE_STAGE, GATE_TAU, GATE_ALPHA_FIXED, REFUSE_LEGACY_HEADS
    PRIOR_SCRAMBLE_MODE = str(getattr(args, 'prior_scramble', '0'))
    B0_SINGLE_STAGE = bool(getattr(args, 'b0_single_stage', 0))
    GATE_TAU = float(getattr(args, 'gate_tau', 0.5))
    _gaf = getattr(args, 'gate_alpha_fixed', None)
    GATE_ALPHA_FIXED = float(_gaf) if _gaf is not None else None
    REFUSE_LEGACY_HEADS = bool(getattr(args, 'refuse_legacy_heads', 1))
    if PRIOR_SCRAMBLE_MODE not in ('0', 'none', 'off', ''):
        print(f'[A2] prior_scramble={PRIOR_SCRAMBLE_MODE}')
    if B0_SINGLE_STAGE:
        print('[B0] single-stage binary Y (in_channels=1)')
    print(f'[GATE] tau={GATE_TAU} alpha_fixed={GATE_ALPHA_FIXED}')
    if PRIOR_FG_ONLY:
        print("[PRIOR] Foreground-only prior mode: single fg channel (in_channels=2).")
    if PRIOR_DROPOUT_PROB > 0:
        print(f"[ABLATION] Stage-1 prior dropout enabled: p={PRIOR_DROPOUT_PROB}")
    if ZERO_STAGE1_PRIOR:
        print("[ABLATION] Stage-1 prior channels are zeroed (no cascade prior).")

    # CUDA_VISIBLE_DEVICES is set by the watchdog — always use cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_spacing = None
    if Path(SPACING_PARAMS_PATH).exists():
        with open(SPACING_PARAMS_PATH) as f:
            target_spacing = tuple(json.load(f)['target_spacing'])
        print(f"Target spacing: {target_spacing}")

    # ==================== Inference (external data) ====================
    if args.stage == 'inference':
        run_inference_external(args, device, target_spacing)
        return

    data_root = Path(args.data_root)
    cases = _get_all_cases(data_root)
    patient_ids = sorted(set(pid for pid, _, _ in cases))
    folds = _create_folds(patient_ids, NUM_FOLDS, seed=42)
    train_ids = set()
    for i in range(NUM_FOLDS):
        if i != args.fold:
            train_ids.update(folds[i])
    val_ids = set(folds[args.fold])
    train_cases = [(p, i, l) for p, i, l in cases if p in train_ids]
    val_cases = [(p, i, l) for p, i, l in cases if p in val_ids]
    print(f"Fold {args.fold}: train={len(train_cases)}, val={len(val_cases)}")

    cache_dir = prepare_preprocessed_cache(cases, target_spacing, force=args.force_cache)

    suffix = args.exp_suffix
    EXPERIMENT_NAME_STAGE1 = f"{EXPERIMENT_NAME_STAGE1_TEMPLATE}{suffix}"
    EXPERIMENT_NAME_STAGE2 = f"{EXPERIMENT_NAME_STAGE2_TEMPLATE}{suffix}"
    print(f"Experiment names: {EXPERIMENT_NAME_STAGE1}, {EXPERIMENT_NAME_STAGE2}")

    use_mb = bool(args.use_memory_bank)

    # ==================== Generate CV Predictions ====================
    if args.stage == 'generate_cv':
        cv_dir = Path(RESULTS_ROOT) / f"{EXPERIMENT_NAME_STAGE1}_cv_predictions"
        for fold_idx in range(NUM_FOLDS):
            model_path = Path(RESULTS_ROOT) / EXPERIMENT_NAME_STAGE1 / f"fold{fold_idx}" / "best_model.pth"
            if not model_path.exists():
                print(f"Model not found for fold {fold_idx}, skip")
                continue
            model = Stage1Model(
                use_memory_bank=use_mb,
                num_prototypes_per_class=args.num_prototypes_per_class,
                memory_momentum=args.memory_momentum,
                memory_temperature=args.memory_temperature,
                memory_top_k=args.memory_top_k,
                use_mla=bool(args.use_mla),
                retrieval_mode=args.retrieval_mode,
                memory_type=args.memory_type,
                backbone=args.backbone,
                dual_path_stages=args.dual_path_stages,
                residual_conv_stages=args.residual_conv_stages,
            ).to(device)
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            other_ids = set(folds[fold_idx])
            other_cases = [(p, i, l) for p, i, l in cases if p in other_ids]
            print(f"Fold {fold_idx}: predicting {len(other_cases)} cases")
            generate_stage1_predictions(model, other_cases, cv_dir, PATCH_SIZE, cache_dir, device)
        print(f"Done. CV predictions at: {cv_dir}")
        return

    # ==================== Stage 0 (Single-Stage ViT Ablation) ====================
    if args.stage == '0':
        print(f"\n{'='*60}\nSingle-Stage ViT Ablation (No Cascade) - Fold {args.fold}\n{'='*60}")

        # Dataset: reuses Stage1Dataset's sampling logic but keeps original 6-class labels
        class _Stage0Dataset(Stage1Dataset):
            def __getitem__(self, idx):
                case_id, _, _ = self.cases[idx]
                image_mm, _label_mm = _load_cached_case(case_id, self.cache_dir, mmap=True)
                d, h, w = image_mm.shape
                pd, ph, pw = self.patch_size
                need_pad = d < pd or h < ph or w < pw
                if self.training:
                    fg_data = self._fg_cache.get(case_id)
                    if fg_data is not None and len(fg_data[0]) > 0:
                        fg_coords, vol_shape = fg_data
                        fi = np.random.randint(len(fg_coords))
                        cz, cy, cx = fg_coords[fi]
                        z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                        y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                        x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                    else:
                        label_full = np.array(_label_mm, dtype=np.int16)
                        if np.any(label_full > 0):
                            fg = np.argwhere(label_full > 0)
                            fi = np.random.randint(len(fg))
                            cz, cy, cx = fg[fi]
                            z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                            y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                            x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                        else:
                            z1, y1, x1 = np.random.randint(0, max(1, d - pd + 1)), np.random.randint(0, max(1, h - ph + 1)), np.random.randint(0, max(1, w - pw + 1))
                else:
                    z1, y1, x1 = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
                z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
                if need_pad:
                    image_full = np.array(image_mm, dtype=np.float32)
                    label_full = np.array(_label_mm, dtype=np.int16)
                    nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
                    image_full = pp._pad_to_shape_3d(image_full, (nd, nh, nw))
                    label_full = pp._pad_to_shape_3d(label_full, (nd, nh, nw))
                    image = image_full[z1:z2, y1:y2, x1:x2]
                    label_orig = label_full[z1:z2, y1:y2, x1:x2]
                else:
                    image = np.array(image_mm[z1:z2, y1:y2, x1:x2], dtype=np.float32)
                    label_orig = np.array(_label_mm[z1:z2, y1:y2, x1:x2], dtype=np.int16)
                if self.training:
                    for ax in range(3):
                        if np.random.rand() < 0.5:
                            image = np.flip(image, axis=ax)
                            label_orig = np.flip(label_orig, axis=ax)
                label = np.clip(label_orig.astype(np.int64), 0, 5)
                image_t = torch.from_numpy(np.ascontiguousarray(image)[None]).float()
                label_t = torch.from_numpy(np.ascontiguousarray(label)).long()
                return image_t, label_t

        train_ds = _Stage0Dataset(train_cases, PATCH_SIZE, cache_dir, training=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)

        # Model: flat ViT-UNETR (same backbone as full model, no memory bank)
        _raw_vit = _build_vit(in_channels=1, num_classes=NUM_CLASSES_STAGE0, memory_bank=None, use_mla=True)
        # Wrap to accept (x, labels) signature expected by train_one_epoch
        class _Stage0Wrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x, labels=None):
                return self.net(x)
        model = _Stage0Wrapper(_raw_vit).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M (ViT flat-6class, no memory bank)")

        loss_fn = WeightedBCEDiceLoss(
            weight_bce=0.4, weight_dice=0.6, include_background=False,
            class_weights=torch.tensor([0.1, 10.0, 8.0, 6.0, 3.0, 3.0]),
            use_focal=True, focal_gamma=2.0, focal_alpha=1.0,
        )
        param_groups = _build_param_groups(model, args.lr, layer_decay=0.75)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.05)
        scheduler = PolyLRWithWarmup(
            optimizer, max_epochs=args.epochs, warmup_epochs=WARMUP_EPOCHS,
            power=0.9, initial_lr=args.lr, warmup_start_lr=1e-5, min_lr=1e-6
        )
        scaler = torch.amp.GradScaler() if USE_GRAD_SCALER else None
        exp_dir = Path(RESULTS_ROOT) / f"{EXPERIMENT_NAME_STAGE1}_stage0_ablation" / f"fold{args.fold}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        best_dice, best_epoch, no_improve = 0., 0, 0

        for epoch in range(args.epochs):
            train_loss, train_dice = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device,
                grad_accum_steps=GRAD_ACCUM_STEPS, scaler=scaler)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            do_val = ((epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0) or (epoch == 0)
            if do_val:
                # Sliding-window validation: 6-class output
                model.eval()
                val_dice_sum = 0.0
                val_cls_sums = np.zeros(NUM_CLASSES_STAGE0)
                val_n = 0
                with torch.no_grad():
                    for case_id, _, _ in val_cases:
                        image_mm, _ = _load_cached_case(case_id, cache_dir, mmap=True)
                        label_mm_raw = np.array(_load_cached_case(case_id, cache_dir, mmap=False)[1], dtype=np.int16)
                        d, h, w = image_mm.shape
                        pd, ph, pw = PATCH_SIZE
                        # Pad to patch size
                        nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
                        img = np.array(image_mm, dtype=np.float32)
                        img = pp._pad_to_shape_3d(img, (nd, nh, nw))
                        lbl = pp._pad_to_shape_3d(label_mm_raw, (nd, nh, nw))
                        img_t = img[None]  # numpy [1, D, H, W] for sliding_window_inference
                        with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                            logits = sliding_window_inference(img_t, model, PATCH_SIZE, NUM_CLASSES_STAGE0, device)
                        # logits: numpy [C, D_pad, H_pad, W_pad]
                        logits_t = torch.from_numpy(logits).float().to(device)
                        probs = torch.softmax(logits_t, dim=0)  # [C, D, H, W]
                        pred = torch.argmax(probs, dim=0).cpu().numpy()  # [D, H, W]
                        pred = pred[:d, :h, :w]
                        lbl = lbl[:d, :h, :w]
                        # Per-class Dice and aggregate
                        cls_dices = []
                        for c in range(NUM_CLASSES_STAGE0):
                            pb = (pred == c)
                            gb = (lbl == c)
                            inter = (pb & gb).sum()
                            denom = pb.sum() + gb.sum()
                            cd = 2 * inter / denom if denom > 0 else 0.0
                            val_cls_sums[c] += cd
                            cls_dices.append(cd)
                        # Mean over foreground classes (1-5)
                        val_dice_sum += np.mean(cls_dices[1:]) if len(cls_dices) > 1 else 0.0
                        val_n += 1

                val_dice = val_dice_sum / val_n
                val_cls = val_cls_sums / val_n
                print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} tr_dice={train_dice:.4f} "
                      f"val_dice={val_dice:.4f} c1={val_cls[1]:.3f} c2={val_cls[2]:.3f} "
                      f"c3={val_cls[3]:.3f} c4={val_cls[4]:.3f} c5={val_cls[5]:.3f} lr={lr:.6f}")

                if val_dice > best_dice:
                    best_dice, best_epoch, no_improve = val_dice, epoch + 1, 0
                    torch.save({'model_state_dict': model.state_dict(), 'best_dice': best_dice,
                                'best_epoch': best_epoch, 'epoch': epoch + 1},
                               exp_dir / 'best_model.pth')
                    print(f"  -> Best model saved (dice={val_dice:.4f})")
                else:
                    no_improve += VALIDATE_EVERY_N_EPOCHS
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} dice={train_dice:.4f} lr={lr:.6f}")

            if do_val and no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Single-stage ViT done. Best val dice={best_dice:.4f} at epoch {best_epoch}")
        from datetime import datetime as dt
        (exp_dir / "COMPLETED").write_text(
            f"Training completed at {dt.now().isoformat()}\nBest val_dice: {best_dice:.4f}\nBest epoch: {best_epoch}\n")
        print(f"COMPLETED marker written: {exp_dir / 'COMPLETED'}")
        return

    # ==================== Stage 1 ====================
    if args.stage == '1':
        print(f"\n{'='*60}\nStage1 Training (ViT+MLA+MemoryBank={use_mb}) - Fold {args.fold}\n{'='*60}")
        model = Stage1Model(
            use_memory_bank=use_mb,
            num_prototypes_per_class=args.num_prototypes_per_class,
            memory_momentum=args.memory_momentum,
            memory_temperature=args.memory_temperature,
            memory_top_k=args.memory_top_k,
            use_mla=bool(args.use_mla),
            retrieval_mode=args.retrieval_mode,
            memory_type=args.memory_type,
            backbone=args.backbone,
            dual_path_stages=args.dual_path_stages,
            residual_conv_stages=args.residual_conv_stages,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params/1e6:.1f}M (ViT+MLA+MemoryBank)")

        train_ds = Stage1Dataset(train_cases, PATCH_SIZE, cache_dir, training=True,
                                 random_crop_prob=args.random_crop_prob)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
        # Stage1: 6-class [bg,c1,c2,c3,c4=bony_roi,c5=class5_bypass]; c5 lowest weight
        loss_fn = WeightedBCEDiceLoss(
            weight_bce=0.4, weight_dice=0.6, include_background=False,
            class_weights=torch.tensor([0.1, 2.0, 2.0, 2.0, 1.5, 0.5]),
            use_focal=True, focal_gamma=2.0, focal_alpha=1.0,
        )
        # Layer-wise LR decay: earlier encoder stages get lower LR
        param_groups = _build_param_groups(model, args.lr, layer_decay=0.75)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.05)
        scheduler = PolyLRWithWarmup(
            optimizer, max_epochs=args.epochs, warmup_epochs=WARMUP_EPOCHS,
            power=0.9, initial_lr=args.lr, warmup_start_lr=1e-5, min_lr=1e-6
        )
        scaler = torch.amp.GradScaler() if USE_GRAD_SCALER else None
        exp_dir = Path(RESULTS_ROOT) / EXPERIMENT_NAME_STAGE1 / f"fold{args.fold}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        best_dice, best_epoch, no_improve = 0., 0, 0
        history = []
        start_epoch = 0

        # === Resume checkpoint loading (Stage1) ===
        if args.resume:
            resume_path = args.resume_checkpoint or str(exp_dir / 'latest_checkpoint.pth')
            if not Path(resume_path).exists():
                print(f"[WARN] Resume checkpoint not found: {resume_path}, starting from scratch")
            else:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                sd = ckpt.get('model_state_dict', ckpt)
                if globals().get('use_binary_y', False) or str(getattr(args, 's2_head', '')) == 'binary_y':
                    if REFUSE_LEGACY_HEADS:
                        bad = [k for k in sd if ('group_head' in k or 'soft_tree' in k
                                                 or 'hier_decoder.class_heads' in k)]
                        if bad and 'y_head.weight' not in sd:
                            raise RuntimeError(
                                '[refuse_legacy_heads] soft-tree/legacy ckpt; will not load into binary_y')
                        if 'y_head.weight' in sd and hasattr(model, 'y_head'):
                            if tuple(sd['y_head.weight'].shape) != tuple(model.y_head.weight.shape):
                                raise RuntimeError('[refuse_legacy_heads] y_head shape mismatch; no reuse')
                model.load_state_dict(sd)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler._step = ckpt['scheduler_step']
                if scaler and ckpt.get('scaler_state_dict'):
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_epoch = ckpt['epoch']
                best_dice = ckpt.get('best_dice', 0.)
                best_epoch = ckpt.get('best_epoch', 0)
                no_improve = ckpt.get('no_improve', 0)
                history = ckpt.get('history', [])
                print(f"[Resume] Loaded checkpoint epoch={start_epoch}, best_dice={best_dice:.4f}, no_improve={no_improve}")
                # If already past memory bank warmup, enable now
                if use_mb and start_epoch > args.memory_warmup_epochs:
                    mb = model.net.memory_bank
                    if mb is not None:
                        mb.enabled.fill_(True)
                        print(f"  [Resume] Memory Bank past warmup ({start_epoch}>{args.memory_warmup_epochs}), ENABLED")

        last_val_dice = 0.0  # track across epochs for checkpoint
        for epoch in range(start_epoch, args.epochs):
            # Memory Bank warmup: enable after args.memory_warmup_epochs
            if epoch == args.memory_warmup_epochs and use_mb:
                mb = model.net.memory_bank
                if mb is not None:
                    mb.enabled.fill_(True)
                    print(f"  [Epoch {epoch+1}] Memory Bank ENABLED")
            train_loss, train_dice = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device,
                grad_accum_steps=GRAD_ACCUM_STEPS, scaler=scaler,
                args_proto_weight=args.proto_loss_weight)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            do_val = ((epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0) or (epoch == start_epoch and start_epoch == 0)
            if do_val:
                val_dice, val_cls = validate_stage1_patch(
                    model, val_cases, device, cache_dir, PATCH_SIZE,
                    patches_per_case=4, random_crop_prob=args.random_crop_prob)
                last_val_dice = val_dice
                print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} tr_dice={train_dice:.4f} "
                      f"val_dice={val_dice:.4f} c1-5={[round(x,3) for x in val_cls]} lr={lr:.6f}")
                if val_dice > best_dice:
                    best_dice, best_epoch, no_improve = val_dice, epoch + 1, 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_step': scheduler._step,
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'epoch': epoch + 1,
                        'val_dice': val_dice,
                        'best_dice': best_dice,
                        'best_epoch': best_epoch,
                        'no_improve': no_improve,
                        'history': history,
                    }, exp_dir / 'best_model.pth')
                    print(f"  -> Best model saved (dice={val_dice:.4f})")
                else:
                    no_improve += VALIDATE_EVERY_N_EPOCHS
                history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_dice': train_dice,
                                'val_dice': val_dice, 'nerve_body': val_cls[0], 'combined': val_cls[1]})
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} dice={train_dice:.4f} lr={lr:.6f}")

            # Save latest checkpoint EVERY epoch for resume safety
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step': scheduler._step,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'epoch': epoch + 1,
                'val_dice': last_val_dice,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'no_improve': no_improve,
                'history': history,
            }, exp_dir / 'latest_checkpoint.pth')

            if do_val and no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if history:
            with open(exp_dir / 'history.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
        # Generate final predictions: softmax + label NRRD
        cv_dir = Path(RESULTS_ROOT) / f"{EXPERIMENT_NAME_STAGE1}_cv_predictions"
        best_ckpt = torch.load(exp_dir / 'best_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        generate_stage1_predictions(model, val_cases, cv_dir, PATCH_SIZE, cache_dir, device)
        print(f"Stage1 done. Best dice={best_dice:.4f} at epoch {best_epoch}")
        print(f"  Predictions (softmax + label NRRD) saved to: {cv_dir}")
        # Write COMPLETED marker for watchdog
        from datetime import datetime
        (exp_dir / "COMPLETED").write_text(
            f"Training completed at {datetime.now().isoformat()}\n"
            f"Best val_dice: {best_dice:.4f}\n"
            f"Best epoch: {best_epoch}\n"
        )
        print(f"COMPLETED marker written: {exp_dir / 'COMPLETED'}")
        return

    # ==================== Stage 2 ====================
    if args.stage == '2':
        use_hier = bool(args.use_hierarchical)
        s2_head = getattr(args, 's2_head', 'binary_y')
        use_binary_y = (s2_head == 'binary_y')
        if use_binary_y:
            model_type_str = "Binary-Y (protocol primary; class5_bypass excluded)"
        else:
            model_type_str = "Hierarchical Soft-Tree" if use_hier else "Legacy Flat"
        print(f"\n{'='*60}\nStage2 Training ({model_type_str} + ViT+MLA+MemoryBank={use_mb}) - Fold {args.fold}\n{'='*60}")

        stage1_pred_dir = args.stage1_pred_dir or str(Path(RESULTS_ROOT) / f"{EXPERIMENT_NAME_STAGE1}_cv_predictions")
        if not Path(stage1_pred_dir).exists():
            raise FileNotFoundError(f"Stage1 predictions not found: {stage1_pred_dir}")

        if use_binary_y:
            if B0_SINGLE_STAGE:
                _in_ch = 1
            elif PRIOR_FG_ONLY:
                _in_ch = 2
            else:
                _in_ch = 3
            model = build_binary_y_stage2_model(
                args, use_mb,
                use_gate=bool(args.use_confidence_gate),
                in_channels=_in_ch,
            ).to(device)
        elif use_hier:
            model = HierarchicalStage2Model(
                use_memory_bank=use_mb,
                num_prototypes_per_class=args.num_prototypes_per_class,
                memory_momentum=args.memory_momentum,
                memory_temperature=args.memory_temperature,
                memory_top_k=args.memory_top_k,
                use_confidence_gate=bool(args.use_confidence_gate),
                use_mla=bool(args.use_mla),
                retrieval_mode=args.retrieval_mode,
                memory_type=args.memory_type,
                backbone=args.backbone,
                dual_path_stages=args.dual_path_stages,
                residual_conv_stages=args.residual_conv_stages,
                in_channels=(2 if PRIOR_FG_ONLY else 3),
            ).to(device)
        else:
            model = Stage2Model(
                use_memory_bank=use_mb,
                num_prototypes_per_class=args.num_prototypes_per_class,
                memory_momentum=args.memory_momentum,
                memory_temperature=args.memory_temperature,
                memory_top_k=args.memory_top_k,
                use_mla=bool(args.use_mla),
                retrieval_mode=args.retrieval_mode,
                memory_type=args.memory_type,
                backbone=args.backbone,
                dual_path_stages=args.dual_path_stages,
                residual_conv_stages=args.residual_conv_stages,
            ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Stage2 parameters: {total_params/1e6:.1f}M")

        train_ds = Stage2Dataset(train_cases, stage1_pred_dir, PATCH_SIZE, cache_dir, training=True,
                                 random_crop_prob=args.random_crop_prob,
                                 return_binary_y=use_binary_y)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)

        if use_binary_y:
            if not _HAS_PROTOCOL_S2:
                raise RuntimeError('cnv_protocol_s2 required for binary_y')
            loss_fn = EvidenceGatedYLoss(
                lambda_L=args.lambda_L, lambda_mem=args.lambda_mem,
                mem_sup_weight=args.mem_sup_weight, tau=GATE_TAU,
                mem_align=args.mem_align, weight_bce=0.4, weight_dice=0.6)
            print(f'[LOSS] EvidenceGatedYLoss lambda_L={args.lambda_L} lambda_mem={args.lambda_mem} '
                  f'mem_sup={args.mem_sup_weight} mem_align={args.mem_align} tau={GATE_TAU}')
        elif use_hier:
            # Self-contained soft-tree loss for the exclusive 6-class output.
            # Weights calibrated to sqrt-inverse-frequency of corrected 154-case dataset
            # (all cases have foreground; c5=45.7%, c2=20.1%, c3=14.5%, c4=11.1%, c1=8.6% of FG).
            # final: [bg, c1, c2, c3, c4, c5] — balanced by data freq + clinical priority.
            # group: [bg, nerve_body(c1-4)=54%, c5=46%] — near-equal to reflect true ratio.
            # branch: [c1, c2, c3, c4] — inverse rarity (c1 rarest → highest).
            loss_fn = SoftTreeLoss(
                final_weights=[0.1, 16.0, 12.0, 14.0, 14.0, 7.0],
                group_weights=[0.1, 8.0, 6.0],
                branch_weights=[3.5, 2.0, 2.5, 1.8],
                lambda_group=args.lambda_group,
                lambda_branch=args.lambda_branch,
                lambda_c4=args.lambda_aux,
            ).to(device)
        else:
            if args.nnunet_loss:
                # Exact loss of the fixed nnU-Net baseline (npc_sota_train.py):
                # isolates architecture (MLA + memory bank) as the only difference.
                loss_fn = WeightedBCEDiceLoss(
                    weight_bce=0.3, weight_dice=0.7, include_background=False,
                    class_weights=torch.tensor([0.1, 10.0, 8.0, 6.0, 3.0, 3.0]),
                    use_focal=True, focal_gamma=3.0, focal_alpha=0.25,
                )
            else:
                loss_fn = WeightedBCEDiceLoss(
                    weight_bce=0.4, weight_dice=0.6, include_background=False,
                    class_weights=torch.tensor([0.1, 16.0, 12.0, 14.0, 14.0, 7.0]),  # bg,c1,c2,c3,c4,c5
                    use_focal=True, focal_gamma=2.0, focal_alpha=1.0,
                )

        optimizer = torch.optim.AdamW(
            _build_param_groups(model, args.lr, layer_decay=0.75),
            lr=args.lr, weight_decay=0.05
        )
        scheduler = PolyLRWithWarmup(
            optimizer, max_epochs=args.epochs, warmup_epochs=WARMUP_EPOCHS,
            power=0.9, initial_lr=args.lr, warmup_start_lr=1e-5, min_lr=1e-6
        )
        scaler = torch.amp.GradScaler() if USE_GRAD_SCALER else None
        exp_dir = Path(RESULTS_ROOT) / EXPERIMENT_NAME_STAGE2 / f"fold{args.fold}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        best_dice, best_epoch, no_improve = 0., 0, 0
        history = []
        start_epoch = 0

        # === Resume checkpoint loading (Stage2) ===
        if args.resume:
            resume_path = args.resume_checkpoint or str(exp_dir / 'latest_checkpoint.pth')
            if not Path(resume_path).exists():
                print(f"[WARN] Resume checkpoint not found: {resume_path}, starting from scratch")
            else:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                sd = ckpt.get('model_state_dict', ckpt)
                if globals().get('use_binary_y', False) or str(getattr(args, 's2_head', '')) == 'binary_y':
                    if REFUSE_LEGACY_HEADS:
                        bad = [k for k in sd if ('group_head' in k or 'soft_tree' in k
                                                 or 'hier_decoder.class_heads' in k)]
                        if bad and 'y_head.weight' not in sd:
                            raise RuntimeError(
                                '[refuse_legacy_heads] soft-tree/legacy ckpt; will not load into binary_y')
                        if 'y_head.weight' in sd and hasattr(model, 'y_head'):
                            if tuple(sd['y_head.weight'].shape) != tuple(model.y_head.weight.shape):
                                raise RuntimeError('[refuse_legacy_heads] y_head shape mismatch; no reuse')
                model.load_state_dict(sd)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler._step = ckpt['scheduler_step']
                if scaler and ckpt.get('scaler_state_dict'):
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_epoch = ckpt['epoch']
                best_dice = ckpt.get('best_dice', 0.)
                best_epoch = ckpt.get('best_epoch', 0)
                no_improve = ckpt.get('no_improve', 0)
                history = ckpt.get('history', [])
                print(f"[Resume] Loaded checkpoint epoch={start_epoch}, best_dice={best_dice:.4f}, no_improve={no_improve}")
                # If already past memory bank warmup, enable now
                if use_mb and start_epoch > args.memory_warmup_epochs:
                    mb = model.net.memory_bank if hasattr(model, 'net') else None
                    if mb is not None:
                        mb.enabled.fill_(True)
                        print(f"  [Resume] Memory Bank past warmup ({start_epoch}>{args.memory_warmup_epochs}), ENABLED")

        # === Feature-based prototype initialization ===
        if args.proto_init_from_features and start_epoch == 0 and use_mb:
            mb = model.net.memory_bank if hasattr(model, 'net') else None
            if mb is not None:
                print("  [Proto Init] Initializing prototypes from feature statistics...")
                model.eval()
                with torch.no_grad():
                    # Initialize using a single forward pass
                    print("  [Proto Init] Using single-batch feature extraction...")
                    batch = next(iter(train_loader))
                    ct_img, s1_soft, labels, _meta = batch
                    images = torch.cat([ct_img, s1_soft], dim=1).to(device)
                    labels = labels.to(device)

                    # Capture encoder features and grid shapes via patched forward
                    captured_features = []
                    grid_shapes_captured = []

                    # Temporarily replace forward_features to capture intermediates
                    orig_forward_features = model.net.forward_features
                    def capturing_forward(x, labels_arg=None):
                        target_shape = x.shape[2:]
                        tokens, grid0, stem_skips = model.net.conv_stem(x)
                        s0 = model.net.encoder_stages[0](tokens)
                        s1_tokens, grid1 = model.net.patch_merges[0](s0, grid0)
                        s1 = model.net.encoder_stages[1](s1_tokens)
                        s2_tokens, grid2 = model.net.patch_merges[1](s1, grid1)
                        s2 = model.net.encoder_stages[2](s2_tokens)
                        # Store captured data
                        captured_features.clear()
                        captured_features.extend([s0, s1, s2])
                        grid_shapes_captured.clear()
                        grid_shapes_captured.extend([grid0, grid1, grid2])
                        # Continue with original forward (will re-run encoder, but that's ok)
                        return orig_forward_features(x, labels_arg)

                    model.net.forward_features = capturing_forward
                    _ = model(images, labels)
                    model.net.forward_features = orig_forward_features

                    # Initialize prototypes
                    if len(captured_features) == 3 and len(grid_shapes_captured) == 3:
                        mb.initialize_from_features(
                            captured_features, labels, grid_shapes_captured)
                        print("  [Proto Init] Prototypes initialized from feature statistics")
                    else:
                        print(f"  [Proto Init] WARN: Could not capture features "
                              f"(got {len(captured_features)} features, {len(grid_shapes_captured)} grids)")

                model.train()

        # === Curriculum proto_loss_weight setup ===
        use_curriculum = args.proto_weight_start is not None and args.proto_weight_end is not None
        if use_curriculum:
            print(f"  [Curriculum] proto_loss_weight: {args.proto_weight_start} -> {args.proto_weight_end} "
                  f"over {args.proto_anneal_epochs} epochs")

        last_val_dice = 0.0  # track across epochs for checkpoint
        for epoch in range(start_epoch, args.epochs):
            # Memory Bank warmup: enable after args.memory_warmup_epochs
            if epoch == args.memory_warmup_epochs and use_mb:
                mb = model.net.memory_bank if hasattr(model, 'net') else None
                if mb is not None:
                    mb.enabled.fill_(True)
                    print(f"  [Epoch {epoch+1}] Memory Bank ENABLED")

            # SPM (V5): anneal content-vs-position alpha 1.0 -> alpha_final
            # over 150 epochs once the memory bank is enabled.
            if use_mb and epoch > args.memory_warmup_epochs:
                mb = model.net.memory_bank if hasattr(model, 'net') else None
                if mb is not None and hasattr(mb, 'set_alpha'):
                    t = min(1.0, (epoch - args.memory_warmup_epochs) / 150.0)
                    a = 1.0 - (1.0 - mb.alpha_final) * 0.5 * (1 - math.cos(math.pi * t))
                    mb.set_alpha(a)
                    if (epoch + 1) % 25 == 0:
                        print(f"  [Epoch {epoch+1}] SPM alpha={a:.3f}")
            # SCM (V6): anneal semantic-anchor beta 0 -> beta_final over 150 epochs.
            if use_mb and epoch > args.memory_warmup_epochs:
                mb = model.net.memory_bank if hasattr(model, 'net') else None
                if mb is not None and hasattr(mb, 'set_beta'):
                    t = min(1.0, (epoch - args.memory_warmup_epochs) / 150.0)
                    b = mb.beta_final * 0.5 * (1 - math.cos(math.pi * t))
                    mb.set_beta(b)
                    if (epoch + 1) % 25 == 0:
                        print(f"  [Epoch {epoch+1}] SCM beta={b:.3f}")

            # Compute curriculum proto_loss_weight
            if use_curriculum:
                current_proto_weight = get_curriculum_proto_weight(
                    epoch, args.proto_weight_start, args.proto_weight_end,
                    args.proto_anneal_epochs)
            else:
                current_proto_weight = args.proto_loss_weight

            if use_binary_y:
                train_loss, train_dice = train_binary_y_one_epoch(
                    model, train_loader, optimizer, loss_fn, device,
                    grad_accum_steps=GRAD_ACCUM_STEPS, scaler=scaler,
                    args_proto_weight=current_proto_weight)
            else:
                train_loss, train_dice = train_stage2_one_epoch(
                    model, train_loader, optimizer, loss_fn, device,
                    grad_accum_steps=GRAD_ACCUM_STEPS, scaler=scaler,
                    args_proto_weight=current_proto_weight)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            do_val = ((epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0) or (epoch == start_epoch and start_epoch == 0)
            if do_val:
                if use_binary_y:
                    val_dice, val_cls = validate_stage2_binary_y_patch(
                        model, val_cases, device, cache_dir, stage1_pred_dir, PATCH_SIZE,
                        patches_per_case=4, random_crop_prob=args.random_crop_prob)
                    last_val_dice = val_dice
                    print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} tr_dice={train_dice:.4f} "
                          f"val_Dice_Y={val_dice:.4f} (Primary Y; no 5-class mean) lr={lr:.6f}")
                else:
                    val_dice, val_cls = validate_stage2_patch(
                        model, val_cases, device, cache_dir, stage1_pred_dir, PATCH_SIZE,
                        patches_per_case=4, random_crop_prob=args.random_crop_prob)
                    last_val_dice = val_dice
                    print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} tr_dice={train_dice:.4f} "
                          f"val_dice={val_dice:.4f} c1={val_cls[0]:.3f} c2={val_cls[1]:.3f} "
                          f"c3={val_cls[2]:.3f} c4={val_cls[3]:.3f} c5={val_cls[4]:.3f} "
                          f"nerve_body={val_cls[5]:.3f} lr={lr:.6f}")
                if val_dice > best_dice:
                    best_dice, best_epoch, no_improve = val_dice, epoch + 1, 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_step': scheduler._step,
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'epoch': epoch + 1,
                        'val_dice': val_dice,
                        'best_dice': best_dice,
                        'best_epoch': best_epoch,
                        'no_improve': no_improve,
                        'history': history,
                    }, exp_dir / 'best_model.pth')
                    print(f"  -> Best model saved (dice={val_dice:.4f})")
                else:
                    no_improve += VALIDATE_EVERY_N_EPOCHS
                history.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_dice': train_dice,
                                'val_dice': val_dice, 'cls1': val_cls[0], 'cls2': val_cls[1],
                                'cls3': val_cls[2], 'cls4': val_cls[3], 'cls5': val_cls[4],
                                'nerve_body': val_cls[5]})
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"E[{epoch+1}/{args.epochs}] loss={train_loss:.4f} dice={train_dice:.4f} lr={lr:.6f}")

            # Save latest checkpoint EVERY epoch for resume safety
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step': scheduler._step,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'epoch': epoch + 1,
                'val_dice': last_val_dice,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'no_improve': no_improve,
                'history': history,
            }, exp_dir / 'latest_checkpoint.pth')

            if do_val and no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if history:
            with open(exp_dir / 'history.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
        # Generate final Stage2 label predictions as NRRD
        stage2_label_dir = Path(RESULTS_ROOT) / f"{EXPERIMENT_NAME_STAGE2}_label_predictions"
        best_ckpt = torch.load(exp_dir / 'best_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        generate_stage2_label_predictions(model, val_cases, stage1_pred_dir, stage2_label_dir, PATCH_SIZE, cache_dir, device)
        print(f"Stage2 done. Best val dice={best_dice:.4f} at epoch {best_epoch}")
        print(f"  Label predictions (NRRD) saved to: {stage2_label_dir}")
        # Write COMPLETED marker for watchdog
        from datetime import datetime
        (exp_dir / "COMPLETED").write_text(
            f"Training completed at {datetime.now().isoformat()}\n"
            f"Best val_dice: {best_dice:.4f}\n"
            f"Best epoch: {best_epoch}\n"
        )
        print(f"COMPLETED marker written: {exp_dir / 'COMPLETED'}")


if __name__ == '__main__':
    main()