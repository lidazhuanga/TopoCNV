"""Report folds, internal validation sets and Stage-1 inner folds (protocol v0.4-draft §3, V4_PLAN §2).

Report folds are extras/splits.json (= _create_folds(seed=42), as P0 and SOTA). Everything else is
derived from them with fixed seeds, so every run of every arm sees the same sets.

Stage-1 inner folds have two tiers (nested selection, V4_PLAN §5 as revised 2026-10-24):
  tier='report'  over T_k:          the report phase runs no selection, so V_k is ordinary training
                                    data there and single inner-OOF predictions are fine.
  tier='dev'     over T_k - V_k:    V_k is held out of every dev Stage-1 model and is predicted by
                                    the 3-model ensemble, exactly like report-fold cases.
"""
import json

import numpy as np

from .common import REPO

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
    """T_k: every case outside report fold k."""
    held = set(report_folds()[k])
    return [c for c in all_cases() if c not in held]


def internal_split(k):
    """(T_k - V_k, V_k). V_k = 15% of T_k, fixed seed per fold; Stage 2 selects checkpoints on V_k only."""
    T = train_cases(k)
    perm = np.random.default_rng(SEED + k).permutation(len(T))
    n_val = int(round(IVAL_FRAC * len(T)))
    ival = sorted(T[i] for i in perm[:n_val])
    ival_set = set(ival)
    return [c for c in T if c not in ival_set], ival


def stage1_pool(k, tier):
    """Cases the tier's Stage-1 inner models may train on."""
    if tier == "report":
        return train_cases(k)
    if tier == "dev":
        return internal_split(k)[0]
    raise ValueError(f"tier must be 'report' or 'dev', got {tier!r}")


def inner_folds(k, n_inner=N_INNER, tier="report"):
    """Stage-1 inner folds partitioning the tier's pool. Inner model j trains on the pool minus fold j."""
    T = list(stage1_pool(k, tier))
    off = 0 if tier == "report" else 5000
    perm = np.random.default_rng(SEED + 100 * k + off).permutation(len(T))
    return [sorted(T[i] for i in perm[j::n_inner]) for j in range(n_inner)]


def stage1_train_cases(k, j, tier="report"):
    held = set(inner_folds(k, tier=tier)[j])
    return [c for c in stage1_pool(k, tier) if c not in held]


def stage1_source(k, case_id, tier="report"):
    """Which Stage-1 model predicts case_id in this tier: ('inner', j) inside the pool,
    ('ensemble', None) for cases no inner model trained on (report fold k; V_k in the dev tier)."""
    for j, f in enumerate(inner_folds(k, tier=tier)):
        if case_id in f:
            return "inner", j
    return "ensemble", None
