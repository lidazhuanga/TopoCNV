"""v4 报告折、内部验证集与 Stage-1 双层嵌套划分，使用固定种子。"""

import json
import numpy as np

from topocnv.data.utils import REPO

SPLITS_PATH = REPO / "extras" / "splits.json"
SEED = 20261022
IVAL_FRAC = 0.15
N_INNER = 3


def report_folds():
    with open(SPLITS_PATH) as f:
        return [sorted(f_) for f_ in json.load(f)["folds"]]


def all_cases():
    return sorted(c for f in report_folds() for c in f)


def report_cases(k):
    return report_folds()[k]


def train_cases(k):
    """T_k：报告折 k 之外的全部病例。"""
    held = set(report_folds()[k])
    return [c for c in all_cases() if c not in held]


def internal_split(k):
    """返回 T_k - V_k 和 V_k；内部验证集为 T_k 的固定 15%。"""
    T = train_cases(k)
    perm = np.random.default_rng(SEED + k).permutation(len(T))
    n_val = int(round(IVAL_FRAC * len(T)))
    ival = sorted(T[i] for i in perm[:n_val])
    ival_set = set(ival)
    return [c for c in T if c not in ival_set], ival


def stage1_pool(k, tier):
    """返回该层的 Stage-1 内层模型允许训练的病例池。"""
    if tier == "report":
        return train_cases(k)
    if tier == "dev":
        return internal_split(k)[0]
    raise ValueError(f"tier must be 'report' or 'dev', got {tier!r}")


def inner_folds(k, n_inner=N_INNER, tier="report"):
    """划分该层病例池；每个内层模型训练时排除其对应的折。"""
    T = list(stage1_pool(k, tier))
    off = 0 if tier == "report" else 5000
    perm = np.random.default_rng(SEED + 100 * k + off).permutation(len(T))
    return [sorted(T[i] for i in perm[j::n_inner]) for j in range(n_inner)]


def stage1_train_cases(k, j, tier="report"):
    held = set(inner_folds(k, tier=tier)[j])
    return [c for c in stage1_pool(k, tier) if c not in held]


def stage1_source(k, case_id, tier="report"):
    """训练池内病例采用 inner-OOF，其余病例采用三模型集成。"""
    for j, f in enumerate(inner_folds(k, tier=tier)):
        if case_id in f:
            return "inner", j
    return "ensemble", None


__all__ = [
    "SPLITS_PATH", "SEED", "IVAL_FRAC", "N_INNER", "report_folds", "all_cases", "report_cases",
    "train_cases", "internal_split", "stage1_pool", "inner_folds", "stage1_train_cases", "stage1_source",
]
