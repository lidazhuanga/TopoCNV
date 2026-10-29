"""已迁移至 topocnv 的 v4 ICP、Atlas 及先验构建兼容导出。"""

from topocnv.priors.atlas import (
    ATLAS_ROI_THR, TEMPLATE_MARGIN_MM, Atlas, choose_template, side_c4_centroids,
)
from topocnv.priors.builders import (
    DIST_TRUNC_MM, atlas_roi_record, build_case_priors, dist_prior, prior_box_from_rec, to_uint8,
)
from topocnv.priors.icp import (
    MAX_ICP_POINTS, MIN_ICP_POINTS, SCALE_BOUNDS, TARGET_SCALE_BOUNDS,
    apply_sim, bilateral_c4_distance_mm, icp_similarity, mask_points_mm, side_points, umeyama,
)

__all__ = [
    "ATLAS_ROI_THR", "TEMPLATE_MARGIN_MM", "Atlas", "choose_template", "side_c4_centroids",
    "DIST_TRUNC_MM", "atlas_roi_record", "build_case_priors", "dist_prior", "prior_box_from_rec", "to_uint8",
    "MAX_ICP_POINTS", "MIN_ICP_POINTS", "SCALE_BOUNDS", "TARGET_SCALE_BOUNDS",
    "apply_sim", "bilateral_c4_distance_mm", "icp_similarity", "mask_points_mm", "side_points", "umeyama",
]
