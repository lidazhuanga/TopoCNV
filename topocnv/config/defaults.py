"""Canonical defaults extracted from the TopoCNV training entry points."""

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DEFAULT = os.environ.get(
    "CNV_DATA_ROOT", str(PROJECT_ROOT / "npc_data" / "train")
)
RESULTS_ROOT = os.environ.get("CNV_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
PREPROCESS_CACHE_ROOT = str(Path(RESULTS_ROOT) / "preprocess_cache")
SPACING_PARAMS_PATH = str(PROJECT_ROOT / "spacing_params.json")

# Shared data and protocol defaults.
TARGET_SPACING: Tuple[float, float, float] = (
    2.5,
    1.1328120231628418,
    1.1328120231628418,
)
CLIP_LOWER = -250.0
CLIP_UPPER = 400.0
NORMALIZATION_PERCENTILE_LOWER = 1.0
NORMALIZATION_PERCENTILE_UPPER = 99.0
NUM_FOLDS = 5
SEED = 42

# Stage-1 and Stage-2 ViT/MLA defaults.
PATCH_SIZE: Tuple[int, int, int] = (40, 192, 192)
BATCH_SIZE = 1
NUM_WORKERS = 8
GRAD_ACCUM_STEPS = 8
NUM_CLASSES_STAGE0 = 6
NUM_CLASSES_STAGE1 = 6
NUM_CLASSES_STAGE2 = 6
EPOCHS = 500
LEARNING_RATE = 1e-3
OPTIMIZER = "adamw"
WEIGHT_DECAY = 0.05
LAYER_DECAY = 0.75
SCHEDULER = "poly_warmup"
WARMUP_EPOCHS = 10
WARMUP_START_LR = 1e-5
MIN_LR = 1e-6
POLY_POWER = 0.9
VALIDATE_EVERY_N_EPOCHS = 5
EARLY_STOP_PATIENCE = 50
GRAD_CLIP_NORM = 1.0

VIT_EMBED_DIM = 192
VIT_STEM_CHANNELS: Tuple[int, int, int] = (48, 96, 192)
VIT_TOKENIZE_KERNEL: Tuple[int, int, int] = (2, 4, 4)
VIT_NUM_HEADS = 6
VIT_HEAD_DIM = 32
VIT_KV_LATENT_DIM = 96
VIT_ENCODER_DEPTHS: Tuple[int, int, int] = (3, 6, 4)
VIT_DECODER_DEPTHS: Tuple[int, int] = (2, 3)
VIT_MLP_RATIO = 4.0
VIT_DROPOUT = 0.1
VIT_DROP_PATH_RATE = 0.2
TREE_DECODER_FEATURE_DIM = VIT_EMBED_DIM

NUM_PROTOTYPES_PER_CLASS = 32
MEMORY_TOP_K = 8
MEMORY_MOMENTUM = 0.99
MEMORY_TEMPERATURE = 0.5
MEMORY_PATCH_GRID: Tuple[int, int, int] = (2, 2, 2)
MEMORY_BANK_WARMUP_EPOCHS = 50

ZERO_STAGE1_PRIOR = False
PRIOR_DROPOUT_PROB = 0.0
PRIOR_FG_ONLY = False
PRIOR_SCRAMBLE_MODE = "0"
B0_SINGLE_STAGE = False
GATE_TAU = 0.5
GATE_ALPHA_FIXED = None
REFUSE_LEGACY_HEADS = True
DEFAULT_BONE_NORM_THR = 0.97
DEFAULT_EVIDENCE_TEMP = 0.01
DEFAULT_LAMBDA_L = 1.0
DEFAULT_LAMBDA_MEM = 0.5
DEFAULT_MEM_SUP_WEIGHT = 0.3
LAMBDA_GROUP = 0.3
LAMBDA_BRANCH = 0.2
LAMBDA_AUX = 0.5
PROTO_LOSS_WEIGHT = 0.1
PROTO_ANNEAL_EPOCHS = 100
RANDOM_CROP_PROB = 0.0

# SOTA baseline defaults.
SOTA_PATCH_SIZE: Tuple[int, int, int] = (128, 128, 128)
SOTA_BATCH_SIZE = 2
SOTA_NUM_WORKERS = 4
SOTA_NUM_CLASSES = 2
SOTA_BINARY_Y_ENDPOINT = True
SOTA_EPOCHS = 500
SOTA_LEARNING_RATE = 5e-4
SOTA_OPTIMIZER = "adam"
SOTA_SCHEDULER = "cosine"
SOTA_ETA_MIN_FACTOR = 0.01
SOTA_VALIDATE_EVERY_N = 5
SOTA_FULL_VAL_EVERY_N = 25
SOTA_EARLY_STOP_PATIENCE = 50
SOTA_SW_EARLY_STOP_CHECKS = 6
SOTA_STEP_SIZE = 0.5
SOTA_SW_BATCH_SIZE = 2
SOTA_SIGMA_SCALE = 0.125
SUPPORTED_SOTA_MODELS: Tuple[str, ...] = (
    "nnunet",
    "segresnet",
    "attention_unet",
    "unetr",
    "swin_unetr",
    "nnformer",
)

COMMON_DEFAULTS: Dict[str, Any] = {
    "data_root": DATA_ROOT_DEFAULT,
    "results_root": RESULTS_ROOT,
    "fold": 0,
    "gpu": "0",
    "num_folds": NUM_FOLDS,
    "seed": SEED,
    "data": {
        "patch_size": PATCH_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "spacing": TARGET_SPACING,
        "spacing_params": SPACING_PARAMS_PATH,
        "clip_lower": CLIP_LOWER,
        "clip_upper": CLIP_UPPER,
        "normalization": "percentile",
        "normalization_lower": NORMALIZATION_PERCENTILE_LOWER,
        "normalization_upper": NORMALIZATION_PERCENTILE_UPPER,
        "cache_dir": PREPROCESS_CACHE_ROOT,
        "random_crop_prob": RANDOM_CROP_PROB,
    },
    "model": {
        "model": "vit_unetr_mla",
        "num_classes": NUM_CLASSES_STAGE1,
        "embed_dim": VIT_EMBED_DIM,
        "stem_channels": VIT_STEM_CHANNELS,
        "tokenize_kernel": VIT_TOKENIZE_KERNEL,
        "heads": VIT_NUM_HEADS,
        "head_dim": VIT_HEAD_DIM,
        "kv_latent_dim": VIT_KV_LATENT_DIM,
        "encoder_depths": VIT_ENCODER_DEPTHS,
        "decoder_depths": VIT_DECODER_DEPTHS,
        "mlp_ratio": VIT_MLP_RATIO,
        "dropout": VIT_DROPOUT,
        "drop_path_rate": VIT_DROP_PATH_RATE,
        "use_memory_bank": True,
        "num_prototypes_per_class": NUM_PROTOTYPES_PER_CLASS,
        "memory_top_k": MEMORY_TOP_K,
        "memory_momentum": MEMORY_MOMENTUM,
        "memory_temperature": MEMORY_TEMPERATURE,
        "memory_patch_grid": MEMORY_PATCH_GRID,
        "memory_warmup_epochs": MEMORY_BANK_WARMUP_EPOCHS,
        "gate_tau": GATE_TAU,
        "gate_alpha_fixed": GATE_ALPHA_FIXED,
    },
    "train": {
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "optimizer": OPTIMIZER,
        "weight_decay": WEIGHT_DECAY,
        "layer_decay": LAYER_DECAY,
        "scheduler": SCHEDULER,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_start_lr": WARMUP_START_LR,
        "min_lr": MIN_LR,
        "scheduler_power": POLY_POWER,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "gradient_accumulation": GRAD_ACCUM_STEPS,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "validate_every": VALIDATE_EVERY_N_EPOCHS,
    },
    "infer": {
        "patch_size": PATCH_SIZE,
        "step_size": 0.5,
        "overlap": 0.5,
        "sliding_window_batch_size": 1,
    },
}

STAGE1_DEFAULTS: Dict[str, Any] = {
    "stage": "1",
    "exp_name": "npc_vesselFM3D_cascade_MIA_v3_vit_memory_stage1",
    "model": {"num_classes": NUM_CLASSES_STAGE1},
}

STAGE2_DEFAULTS: Dict[str, Any] = {
    "stage": "2",
    "exp_name": "npc_vesselFM3D_cascade_MIA_v3_vit_memory_stage2",
    "model": {
        "num_classes": NUM_CLASSES_STAGE2,
        "s2_head": "binary_y",
        "use_confidence_gate": True,
        "use_hierarchical": True,
        "lambda_l": DEFAULT_LAMBDA_L,
        "lambda_mem": DEFAULT_LAMBDA_MEM,
        "mem_sup_weight": DEFAULT_MEM_SUP_WEIGHT,
        "bone_norm_thr": DEFAULT_BONE_NORM_THR,
        "evidence_temp": DEFAULT_EVIDENCE_TEMP,
    },
    "train": {
        "prior_dropout_prob": PRIOR_DROPOUT_PROB,
        "lambda_group": LAMBDA_GROUP,
        "lambda_branch": LAMBDA_BRANCH,
        "lambda_aux": LAMBDA_AUX,
        "proto_loss_weight": PROTO_LOSS_WEIGHT,
        "proto_anneal_epochs": PROTO_ANNEAL_EPOCHS,
    },
}

SOTA_DEFAULTS: Dict[str, Any] = {
    "stage": "sota",
    "exp_name": "sota_binary_y",
    "data": {
        "patch_size": SOTA_PATCH_SIZE,
        "batch_size": SOTA_BATCH_SIZE,
        "num_workers": SOTA_NUM_WORKERS,
    },
    "model": {
        "model": "nnunet",
        "num_classes": SOTA_NUM_CLASSES,
        "binary_y_endpoint": SOTA_BINARY_Y_ENDPOINT,
        "use_memory_bank": False,
    },
    "train": {
        "epochs": SOTA_EPOCHS,
        "lr": SOTA_LEARNING_RATE,
        "optimizer": SOTA_OPTIMIZER,
        "weight_decay": 0.0,
        "scheduler": SOTA_SCHEDULER,
        "min_lr": SOTA_LEARNING_RATE * SOTA_ETA_MIN_FACTOR,
        "early_stop_patience": SOTA_EARLY_STOP_PATIENCE,
        "gradient_accumulation": 1,
        "validate_every": SOTA_VALIDATE_EVERY_N,
        "full_val_every": SOTA_FULL_VAL_EVERY_N,
        "sw_early_stop_checks": SOTA_SW_EARLY_STOP_CHECKS,
        "loss_bce_weight": 0.3,
        "loss_dice_weight": 0.7,
        "focal_alpha": 0.25,
        "focal_gamma": 3.0,
    },
    "infer": {
        "patch_size": SOTA_PATCH_SIZE,
        "step_size": SOTA_STEP_SIZE,
        "overlap": 1.0 - SOTA_STEP_SIZE,
        "sliding_window_batch_size": SOTA_SW_BATCH_SIZE,
        "sigma_scale": SOTA_SIGMA_SCALE,
    },
}


def get_stage_defaults(stage: str) -> Dict[str, Any]:
    """Return an independent nested defaults dictionary for a training stage."""
    normalized = str(stage).lower().replace("stage", "").replace("-", "")
    stage_defaults = STAGE1_DEFAULTS
    if normalized in {"2", "s2"}:
        stage_defaults = STAGE2_DEFAULTS
    elif normalized in {"sota", "baseline"}:
        stage_defaults = SOTA_DEFAULTS
    elif normalized in {"0", "s0"}:
        stage_defaults = {"stage": "0", "model": {"num_classes": NUM_CLASSES_STAGE0}}

    result = deepcopy(COMMON_DEFAULTS)

    def update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                update(base[key], value)
            else:
                base[key] = deepcopy(value)

    update(result, stage_defaults)
    return result
