"""TopoCNV 公共评估接口。

硬多分类 Dice 复用 engine，训练批次的 soft Dice 复用 losses；
数组层二值指标和冻结的 CNV 协议由 protocol 提供。
"""

from ..engine.evaluator import SegmentationMetrics, evaluate_segmentation
from ..losses.seg_losses import multiclass_dice_score
from .protocol import (
    BONE_HU_HIGH,
    BONE_HU_LOW,
    BONY_ROI_LABEL,
    BYPASS_ROI_LABEL,
    DEFAULT_DATA_ROOT,
    DEFAULT_SPACING_ZYX,
    L1_DILATE_R_MM,
    SURF_TOL_MM,
    TIER_LABELS,
    V04_KEYS,
    VIS_DILATE_R_MM,
    VIS_LABELS,
    Y_LABELS,
    aggregate_metrics,
    assd,
    bone_window_high_contrast,
    build_H_bone,
    build_L1,
    dice_score,
    dilate_mm,
    hausdorff_distance,
    hd95,
    iou_score,
    load_raw_hu_dhw,
    mask_123,
    mask_Y,
    mask_bypass,
    mask_c4,
    partition_Y_by_posterior,
    precision_score,
    pred_to_Y,
    recall_gt3,
    recall_score,
    score_case_from_paths,
    score_protocol,
    score_v04,
    surface_dice,
    weighted_dice_y,
)

__all__ = [
    "SegmentationMetrics", "evaluate_segmentation", "multiclass_dice_score",
    "BONE_HU_HIGH", "BONE_HU_LOW", "BONY_ROI_LABEL", "BYPASS_ROI_LABEL",
    "DEFAULT_DATA_ROOT", "DEFAULT_SPACING_ZYX", "L1_DILATE_R_MM", "SURF_TOL_MM", "TIER_LABELS", "V04_KEYS",
    "VIS_DILATE_R_MM", "VIS_LABELS", "Y_LABELS", "aggregate_metrics", "assd",
    "bone_window_high_contrast", "build_H_bone", "build_L1", "dice_score", "dilate_mm",
    "hausdorff_distance", "hd95", "iou_score", "load_raw_hu_dhw", "mask_123", "mask_Y", "mask_bypass", "mask_c4",
    "partition_Y_by_posterior", "precision_score", "pred_to_Y", "recall_gt3", "recall_score",
    "score_case_from_paths", "score_protocol", "score_v04", "surface_dice", "weighted_dice_y",
]
