"""Backward-compatible imports for the relocated cached dataset."""

from topocnv.data.dataset import (
    DEFAULT_CACHE_DIR,
    CachedSOTADataset,
    _RESULTS_ROOT,
    _load_fg_indices,
    _load_from_cache,
)

__all__ = [
    "_RESULTS_ROOT",
    "DEFAULT_CACHE_DIR",
    "CachedSOTADataset",
    "_load_from_cache",
    "_load_fg_indices",
]
