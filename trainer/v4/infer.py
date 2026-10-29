"""已迁移至 topocnv 的 v4 全体积、ROI 推理及 NRRD 导出兼容入口。"""

from topocnv.engine.v4_inference import (
    AMP_DTYPE, inference_region, predict_stage1, predict_stage2,
    save_label_nrrd, sliding_window_logits, stage2_input,
)

__all__ = [
    "AMP_DTYPE", "inference_region", "predict_stage1", "predict_stage2",
    "save_label_nrrd", "sliding_window_logits", "stage2_input",
]
