"""Reusable data loading, preprocessing, caching, and augmentation APIs."""

from .augmentation import (
    augment_aligned_arrays,
    augment_cached_patch,
    random_affine_3d,
    random_crop_3d,
    random_flip_3d,
    random_gamma,
    random_gaussian_noise,
    random_rotate90_3d,
)
from .dataset import (
    CachedSOTADataset,
    NPCDataset,
    _create_folds,
    _get_all_cases,
    _load_fg_indices,
    _load_from_cache,
    create_folds,
    get_all_cases,
    get_train_val_datasets,
    load_fg_indices,
    load_from_cache,
)
from .preprocess import (
    _pad_or_crop_to_shape_3d,
    _pad_to_shape_3d,
    center_crop_3d,
    compute_gaussian_weights,
    compute_median_spacing,
    compute_steps_for_sliding_window,
    load_normalization_params,
    normalize_ct,
    preprocess_training,
    preprocess_validation,
    resample_image_to_spacing,
    save_normalization_params,
    sliding_window_inference,
)
from .labels import BinaryYDataset, convert_label_to_binary_Y, load_label_volume
from .utils import (
    DATA_ROOT_DEFAULT, DEFAULT_CACHE_DIR, _RESULTS_ROOT, normalize_percentile,
    build_preprocessed_cache, sample_validation_patch_positions,
)
from topocnv.data.utils import (
    CACHE_DIR, LR_AXIS, MARGINS_MM, PRIOR_BOX_MARGIN, PRIOR_NAMES, PRIORS_DIR,
    REPO, RESULTS, RUNS_DIR, SKEL_DIR, SPACING, STAGE1_DIR, load_cached_array, load_image, load_label,
)
from topocnv.data.harmonize import PLANE4, SIGMA, THIN_THR, T_GRID, harmonise, run_length, skeleton, thickness
from topocnv.data.v4_splits import (
    IVAL_FRAC, N_INNER, SEED, SPLITS_PATH,
    all_cases, inner_folds, internal_split, report_cases, report_folds,
    stage1_pool, stage1_source, stage1_train_cases, train_cases,
)


def __getattr__(name):
    if name in {"V4PatchDataset", "load_fg", "prior_dir", "load_roi_record", "load_prior", "load_skeleton", "stage1_target"}:
        from topocnv.data import v4_dataset
        return getattr(v4_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BinaryYDataset", "convert_label_to_binary_Y", "load_label_volume",
    "build_preprocessed_cache", "sample_validation_patch_positions",
    "CACHE_DIR", "LR_AXIS", "MARGINS_MM", "PRIOR_BOX_MARGIN", "PRIOR_NAMES", "PRIORS_DIR",
    "REPO", "RESULTS", "RUNS_DIR", "SKEL_DIR", "SPACING", "STAGE1_DIR", "load_cached_array", "load_image", "load_label",
    "PLANE4", "SIGMA", "THIN_THR", "T_GRID", "IVAL_FRAC", "N_INNER", "SEED", "SPLITS_PATH",
    "harmonise", "run_length", "skeleton", "thickness", "all_cases", "inner_folds", "internal_split", "report_cases",
    "report_folds", "stage1_pool", "stage1_source", "stage1_train_cases", "train_cases",
    "V4PatchDataset", "load_fg", "prior_dir", "load_roi_record", "load_prior", "load_skeleton", "stage1_target",
    "DATA_ROOT_DEFAULT",
    "_RESULTS_ROOT",
    "DEFAULT_CACHE_DIR",
    "NPCDataset",
    "CachedSOTADataset",
    "get_train_val_datasets",
    "get_all_cases",
    "create_folds",
    "_get_all_cases",
    "_create_folds",
    "load_from_cache",
    "load_fg_indices",
    "_load_from_cache",
    "_load_fg_indices",
    "normalize_ct",
    "normalize_percentile",
    "compute_median_spacing",
    "resample_image_to_spacing",
    "save_normalization_params",
    "load_normalization_params",
    "_pad_to_shape_3d",
    "_pad_or_crop_to_shape_3d",
    "random_crop_3d",
    "center_crop_3d",
    "random_flip_3d",
    "random_rotate90_3d",
    "random_affine_3d",
    "random_gamma",
    "random_gaussian_noise",
    "augment_cached_patch",
    "augment_aligned_arrays",
    "preprocess_training",
    "preprocess_validation",
    "compute_gaussian_weights",
    "compute_steps_for_sliding_window",
    "sliding_window_inference",
]
