"""已迁移至 topocnv 的 v4 报告折、内部验证及 Stage-1 双层划分兼容导出。"""

from topocnv.data.v4_splits import (
    IVAL_FRAC, N_INNER, SEED, SPLITS_PATH, all_cases, inner_folds, internal_split,
    report_cases, report_folds, stage1_pool, stage1_source, stage1_train_cases, train_cases,
)

__all__ = [
    "IVAL_FRAC", "N_INNER", "SEED", "SPLITS_PATH", "all_cases", "inner_folds", "internal_split",
    "report_cases", "report_folds", "stage1_pool", "stage1_source", "stage1_train_cases", "train_cases",
]
