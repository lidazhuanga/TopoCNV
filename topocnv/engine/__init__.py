"""Public training and evaluation infrastructure for TopoCNV."""

from .amp import AMPConfig, AMPController, autocast_context, create_grad_scaler, resolve_amp_config
from .checkpoint import CheckpointManager, EarlyStopping, load_checkpoint, save_checkpoint, save_ckpt
from .evaluator import (
    Predictor,
    SegmentationMetrics,
    Window,
    as_probabilities,
    ensemble_predict,
    evaluate_segmentation,
    sliding_window_inference,
    tta_predict,
)
from .scheduler import PolyLRWithWarmup
from .segmentation import (
    compute_dice_per_class, evaluate_supervised_epoch,
    train_supervised_epoch, validate_with_sliding_window,
)
from .trainer import (
    BatchMover,
    BatchStep,
    EpochResult,
    StepResult,
    evaluate_one_epoch,
    gradients_are_finite,
    move_batch_to_device,
    should_validate,
    train_one_epoch,
    train_epoch,
    make_optim,
)

from topocnv.engine.v4_inference import (
    AMP_DTYPE, inference_region, predict_stage1, predict_stage2,
    save_label_nrrd, sliding_window_logits, stage2_input,
)

__all__ = [
    "compute_dice_per_class", "evaluate_supervised_epoch",
    "train_supervised_epoch", "validate_with_sliding_window",
    "make_optim", "train_epoch", "save_ckpt", "AMP_DTYPE", "inference_region",
    "predict_stage1", "predict_stage2", "save_label_nrrd", "sliding_window_logits", "stage2_input",
    "AMPConfig",
    "AMPController",
    "BatchMover",
    "BatchStep",
    "CheckpointManager",
    "EarlyStopping",
    "EpochResult",
    "PolyLRWithWarmup",
    "Predictor",
    "SegmentationMetrics",
    "StepResult",
    "Window",
    "as_probabilities",
    "autocast_context",
    "create_grad_scaler",
    "ensemble_predict",
    "evaluate_one_epoch",
    "evaluate_segmentation",
    "gradients_are_finite",
    "load_checkpoint",
    "move_batch_to_device",
    "resolve_amp_config",
    "save_checkpoint",
    "should_validate",
    "sliding_window_inference",
    "train_one_epoch",
    "tta_predict",
]
