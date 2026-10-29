"""Backward-compatible imports for the TopoCNV loss package."""

from topocnv.losses.seg_losses import (
    BCEDiceLoss,
    HierarchicalTreeLoss,
    WeightedBCEDiceLoss,
    multiclass_dice_score,
)
from topocnv.losses.topo_losses import (
    CombinedTopologyLoss,
    SoftClDiceLoss,
    TemplateGraphTopologyLoss,
    _soft_skel3d,
)

__all__ = [
    "BCEDiceLoss",
    "WeightedBCEDiceLoss",
    "multiclass_dice_score",
    "HierarchicalTreeLoss",
    "SoftClDiceLoss",
    "TemplateGraphTopologyLoss",
    "CombinedTopologyLoss",
]
