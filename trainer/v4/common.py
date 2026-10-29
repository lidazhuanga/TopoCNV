"""已迁移至 topocnv 的 v4 路径、配置与数组工具兼容导出。"""

from topocnv.data.utils import (
    CACHE_DIR, LR_AXIS, MARGINS_MM, PRIOR_BOX_MARGIN, PRIOR_NAMES, PRIORS_DIR,
    REPO, RESULTS, RUNS_DIR, SKEL_DIR, SPACING, STAGE1_DIR, load_image, load_label,
)
from topocnv.config.loader import load_config, save_config
from topocnv.utils.box_ops import (
    box_shape, box_slices, boxes_mask, case_shift, expand_box,
    mask_bbox, mask_Y, region_from_box, union_box,
)

__all__ = [
    "CACHE_DIR", "LR_AXIS", "MARGINS_MM", "PRIOR_BOX_MARGIN", "PRIOR_NAMES", "PRIORS_DIR",
    "REPO", "RESULTS", "RUNS_DIR", "SKEL_DIR", "SPACING", "STAGE1_DIR", "load_image", "load_label",
    "load_config", "save_config", "box_shape", "box_slices", "boxes_mask", "case_shift", "expand_box",
    "mask_bbox", "mask_Y", "region_from_box", "union_box",
]
