"""Segmentation, topology-aware, and evidence-gated TopoCNV losses."""

from .evidence_losses import (
    BinaryYLoss,
    DEFAULT_BONE_NORM_THR,
    DEFAULT_EVIDENCE_SMOOTH,
    DEFAULT_EVIDENCE_TEMP,
    DEFAULT_GATE_ALPHA_FIXED,
    DEFAULT_GATE_TAU,
    DEFAULT_LAMBDA_L,
    DEFAULT_LAMBDA_MEM,
    DEFAULT_MEM_SUP_WEIGHT,
    EvidenceGatedYLoss,
)
from .seg_losses import (
    BCEDiceLoss,
    HierarchicalTreeLoss,
    MaskedWeightedBCEDice,
    WeightedBCEDiceLoss,
    multiclass_dice_score,
)
from .topo_losses import (
    CombinedTopologyLoss,
    SoftClDiceLoss,
    TemplateGraphTopologyLoss,
)

from topocnv.losses.visibility_loss import VisibilityLoss, visibility_target
from topocnv.losses.skeleton_losses import skeleton_recall_loss

__all__ = [
    "MaskedWeightedBCEDice",
    "VisibilityLoss",
    "visibility_target",
    "skeleton_recall_loss",
    "BCEDiceLoss",
    "WeightedBCEDiceLoss",
    "multiclass_dice_score",
    "HierarchicalTreeLoss",
    "SoftClDiceLoss",
    "TemplateGraphTopologyLoss",
    "CombinedTopologyLoss",
    "BinaryYLoss",
    "EvidenceGatedYLoss",
    "DEFAULT_GATE_TAU",
    "DEFAULT_GATE_ALPHA_FIXED",
    "DEFAULT_BONE_NORM_THR",
    "DEFAULT_EVIDENCE_TEMP",
    "DEFAULT_EVIDENCE_SMOOTH",
    "DEFAULT_LAMBDA_L",
    "DEFAULT_LAMBDA_MEM",
    "DEFAULT_MEM_SUP_WEIGHT",
]
