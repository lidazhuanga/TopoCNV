"""Typed configuration schema for TopoCNV experiments."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

from . import defaults

_ConfigT = TypeVar("_ConfigT")


def _filtered_values(
    config_type: Type[_ConfigT], values: Mapping[str, Any]
) -> Dict[str, Any]:
    """Keep only declared dataclass fields from an input mapping."""
    names = {item.name for item in fields(config_type)}
    return {key: value for key, value in values.items() if key in names}


def _tuple_value(value: Any) -> Any:
    """Convert YAML/JSON sequences to immutable tuples when present."""
    return tuple(value) if isinstance(value, (list, tuple)) else value


@dataclass
class DataConfig:
    """Input data, preprocessing, sampling, and loader configuration."""

    patch_size: Tuple[int, int, int] = defaults.PATCH_SIZE
    batch_size: int = defaults.BATCH_SIZE
    num_workers: int = defaults.NUM_WORKERS
    spacing: Optional[Tuple[float, float, float]] = defaults.TARGET_SPACING
    spacing_params: Optional[str] = defaults.SPACING_PARAMS_PATH
    clip_lower: float = defaults.CLIP_LOWER
    clip_upper: float = defaults.CLIP_UPPER
    normalization: str = "percentile"
    normalization_lower: float = defaults.NORMALIZATION_PERCENTILE_LOWER
    normalization_upper: float = defaults.NORMALIZATION_PERCENTILE_UPPER
    normalization_min: float = 0.0
    normalization_max: float = 1.0
    ensure_foreground: bool = True
    random_crop_prob: float = defaults.RANDOM_CROP_PROB
    force_cache: bool = False
    use_cache: bool = True
    cache_dir: Optional[str] = defaults.PREPROCESS_CACHE_ROOT

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DataConfig":
        """Build a data configuration from a mapping."""
        data = _filtered_values(cls, values)
        for name in ("patch_size", "spacing"):
            if name in data and data[name] is not None:
                data[name] = _tuple_value(data[name])
        return cls(**data)


@dataclass
class ModelConfig:
    """Backbone, MLA, memory-bank, and evidence-gate configuration."""

    model: str = "vit_unetr_mla"
    backbone: str = "vit"
    in_channels: int = 1
    num_classes: int = defaults.NUM_CLASSES_STAGE1
    embed_dim: int = defaults.VIT_EMBED_DIM
    stem_channels: Tuple[int, int, int] = defaults.VIT_STEM_CHANNELS
    tokenize_kernel: Tuple[int, int, int] = defaults.VIT_TOKENIZE_KERNEL
    heads: int = defaults.VIT_NUM_HEADS
    head_dim: int = defaults.VIT_HEAD_DIM
    kv_latent_dim: int = defaults.VIT_KV_LATENT_DIM
    encoder_depths: Tuple[int, ...] = defaults.VIT_ENCODER_DEPTHS
    decoder_depths: Tuple[int, ...] = defaults.VIT_DECODER_DEPTHS
    mlp_ratio: float = defaults.VIT_MLP_RATIO
    dropout: float = defaults.VIT_DROPOUT
    drop_path_rate: float = defaults.VIT_DROP_PATH_RATE
    use_mla: bool = True
    use_memory_bank: bool = True
    num_prototypes_per_class: int = defaults.NUM_PROTOTYPES_PER_CLASS
    memory_top_k: int = defaults.MEMORY_TOP_K
    memory_momentum: float = defaults.MEMORY_MOMENTUM
    memory_temperature: float = defaults.MEMORY_TEMPERATURE
    memory_patch_grid: Tuple[int, int, int] = defaults.MEMORY_PATCH_GRID
    memory_warmup_epochs: int = defaults.MEMORY_BANK_WARMUP_EPOCHS
    memory_type: str = "mla"
    retrieval_mode: str = "cross_attention"
    s2_head: str = "binary_y"
    binary_y_endpoint: bool = False
    use_confidence_gate: bool = True
    gate_tau: float = defaults.GATE_TAU
    gate_alpha_fixed: Optional[float] = defaults.GATE_ALPHA_FIXED
    refuse_legacy_heads: bool = defaults.REFUSE_LEGACY_HEADS
    use_hierarchical: bool = True
    b0_single_stage: bool = defaults.B0_SINGLE_STAGE
    zero_stage1_prior: bool = defaults.ZERO_STAGE1_PRIOR
    prior_fg_only: bool = defaults.PRIOR_FG_ONLY
    prior_scramble: str = defaults.PRIOR_SCRAMBLE_MODE
    dual_path_stages: str = "0,1"
    residual_conv_stages: str = "0"
    lambda_l: float = defaults.DEFAULT_LAMBDA_L
    lambda_mem: float = defaults.DEFAULT_LAMBDA_MEM
    mem_sup_weight: float = defaults.DEFAULT_MEM_SUP_WEIGHT
    mem_align: str = "low"
    bone_norm_thr: float = defaults.DEFAULT_BONE_NORM_THR
    evidence_temp: float = defaults.DEFAULT_EVIDENCE_TEMP

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Build a model configuration from a mapping."""
        data = _filtered_values(cls, values)
        tuple_fields = (
            "stem_channels",
            "tokenize_kernel",
            "encoder_depths",
            "decoder_depths",
            "memory_patch_grid",
        )
        for name in tuple_fields:
            if name in data:
                data[name] = _tuple_value(data[name])
        return cls(**data)


@dataclass
class TrainConfig:
    """Optimization, scheduling, validation, and checkpoint configuration."""

    epochs: int = defaults.EPOCHS
    lr: float = defaults.LEARNING_RATE
    optimizer: str = defaults.OPTIMIZER
    weight_decay: float = defaults.WEIGHT_DECAY
    layer_decay: float = defaults.LAYER_DECAY
    scheduler: str = defaults.SCHEDULER
    scheduler_power: float = defaults.POLY_POWER
    warmup_epochs: int = defaults.WARMUP_EPOCHS
    warmup_start_lr: float = defaults.WARMUP_START_LR
    min_lr: float = defaults.MIN_LR
    early_stopping: bool = True
    early_stop_patience: int = defaults.EARLY_STOP_PATIENCE
    gradient_accumulation: int = defaults.GRAD_ACCUM_STEPS
    gradient_clip_norm: float = defaults.GRAD_CLIP_NORM
    validate_every: int = defaults.VALIDATE_EVERY_N_EPOCHS
    full_val_every: int = 0
    sw_early_stop_checks: int = defaults.SOTA_SW_EARLY_STOP_CHECKS
    resume: bool = False
    resume_checkpoint: Optional[str] = None
    select_by_sliding_window: bool = False
    prior_dropout_prob: float = defaults.PRIOR_DROPOUT_PROB
    lambda_group: float = defaults.LAMBDA_GROUP
    lambda_branch: float = defaults.LAMBDA_BRANCH
    lambda_aux: float = defaults.LAMBDA_AUX
    proto_loss_weight: float = defaults.PROTO_LOSS_WEIGHT
    proto_init_from_features: bool = False
    proto_weight_start: Optional[float] = None
    proto_weight_end: Optional[float] = None
    proto_anneal_epochs: int = defaults.PROTO_ANNEAL_EPOCHS
    loss_bce_weight: float = 0.4
    loss_dice_weight: float = 0.6
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    amp: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainConfig":
        """Build a training configuration from a mapping."""
        return cls(**_filtered_values(cls, values))


@dataclass
class InferConfig:
    """Sliding-window, TTA, ensemble, and inference I/O configuration."""

    patch_size: Tuple[int, int, int] = defaults.PATCH_SIZE
    step_size: float = 0.5
    overlap: float = 0.5
    sliding_window_batch_size: int = 1
    sigma_scale: float = defaults.SOTA_SIGMA_SCALE
    tta: bool = False
    tta_flips: Tuple[Tuple[int, ...], ...] = ()
    ensemble: bool = False
    skip_sliding_window: bool = False
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    model_path: Optional[str] = None
    val_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "InferConfig":
        """Build an inference configuration from a mapping."""
        data = _filtered_values(cls, values)
        if "patch_size" in data:
            data["patch_size"] = _tuple_value(data["patch_size"])
        if "tta_flips" in data:
            data["tta_flips"] = tuple(tuple(item) for item in data["tta_flips"])
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Top-level configuration for one TopoCNV experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    stage: str = "1"
    fold: int = 0
    gpu: str = "0"
    exp_name: str = "topocnv"
    exp_suffix: str = ""
    data_root: str = defaults.DATA_ROOT_DEFAULT
    results_root: str = defaults.RESULTS_ROOT
    num_folds: int = defaults.NUM_FOLDS
    seed: int = defaults.SEED
    stage1_pred_dir: Optional[str] = None
    stage1_exp_suffix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete nested configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a complete configuration while ignoring unknown future keys."""
        data = dict(values)
        data_config = DataConfig.from_dict(data.pop("data", {}) or {})
        model_config = ModelConfig.from_dict(data.pop("model", {}) or {})
        train_config = TrainConfig.from_dict(data.pop("train", {}) or {})
        infer_config = InferConfig.from_dict(data.pop("infer", {}) or {})
        top_level = _filtered_values(cls, data)
        return cls(
            data=data_config,
            model=model_config,
            train=train_config,
            infer=infer_config,
            **top_level,
        )
