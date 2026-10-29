"""Shared data-pipeline utilities and canonical normalization functions."""

import os
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DEFAULT = os.environ.get(
    "CNV_DATA_ROOT", str(_PROJECT_ROOT / "npc_data" / "train")
)
_RESULTS_ROOT = os.environ.get("CNV_RESULTS_ROOT", str(_PROJECT_ROOT / "results"))
DEFAULT_CACHE_DIR = str(
    Path(_RESULTS_ROOT) / "preprocess_cache" / "sp_2.500_1.133_1.133"
)


REPO = _PROJECT_ROOT
RESULTS = Path(_RESULTS_ROOT)
CACHE_DIR = Path(DEFAULT_CACHE_DIR)
SPACING = (2.5, 1.1328120231628418, 1.1328120231628418)
LR_AXIS = 1
MARGINS_MM = (15, 25, 35)
PRIOR_BOX_MARGIN = 35
PRIOR_NAMES = ("a", "b", "c")
STAGE1_DIR = RESULTS / "v4_stage1"
PRIORS_DIR = Path(os.environ.get("V4_PRIORS_DIR", str(RESULTS / "v4_priors")))
SKEL_DIR = RESULTS / "v4_skel"
RUNS_DIR = RESULTS / "v4_runs"


def load_cached_array(case_id, kind, mmap=True, cache_dir=None):
    """读取单个缓存字段，兼容旧版仅加载图像或标签的调用。"""
    root = CACHE_DIR if cache_dir is None else Path(cache_dir)
    return np.load(root / f"{case_id}_{kind}.npy", mmap_mode="r" if mmap else None)


def load_image(case_id, mmap=True):
    return load_cached_array(case_id, "image", mmap)


def load_label(case_id, mmap=True):
    return load_cached_array(case_id, "label", mmap)


def normalize_percentile(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """Map percentile bounds to ``[b_min, b_max]`` and return float32.

    This is the canonical implementation used by cache building, preprocessing,
    and all legacy compatibility imports. Its constant-input and clipping
    behaviour intentionally matches the historical implementation.
    """
    a_min = np.percentile(image, lower)
    a_max = np.percentile(image, upper)
    if a_max - a_min < 1e-8:
        return np.full_like(image, (b_min + b_max) / 2, dtype=np.float32)
    normalized = (image - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
    if clip:
        normalized = np.clip(normalized, b_min, b_max)
    return normalized.astype(np.float32)


def get_all_cases(data_root: str) -> List[Tuple[str, Path, Path]]:
    """Discover image/label pairs, preferring NRRD labels over NIfTI."""
    data_root_path = Path(data_root)
    image_dir = data_root_path / "image"
    label_dir = data_root_path / "label"

    cases: List[Tuple[str, Path, Path]] = []
    for image_path in sorted(image_dir.glob("*.nii.gz")):
        patient_id = image_path.stem
        if patient_id.endswith(".nii"):
            patient_id = patient_id[:-4]

        nrrd_path = label_dir / f"{patient_id}.nrrd"
        nifti_path = label_dir / f"{patient_id}.nii.gz"
        if nrrd_path.exists():
            label_path = nrrd_path
        elif nifti_path.exists():
            label_path = nifti_path
        else:
            raise FileNotFoundError(
                f"Label file not found for {patient_id}: checked {nrrd_path} and {nifti_path}"
            )
        cases.append((patient_id, image_path, label_path))
    return cases


def create_folds(
    patient_ids: List[str], num_folds: int = 5, seed: int = 42
) -> List[List[str]]:
    """Create deterministic shuffled folds with the historical RandomState API."""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(patient_ids))
    indices = np.arange(len(ids))
    rng.shuffle(indices)
    return [ids[split].tolist() for split in np.array_split(indices, num_folds)]


def sample_validation_patch_positions(label, case_id, patch_size,
                                      patches_per_case=4, random_crop_prob=0.33):
    """按病例固定种子抽取裁剪位置，保持历史 RandomState 调用顺序。"""
    d, h, w = label.shape
    pd, ph, pw = patch_size
    try:
        seed = int(case_id) % (2 ** 31)
    except (TypeError, ValueError):
        seed = zlib.crc32(str(case_id).encode()) % (2 ** 31)
    rng = np.random.RandomState(seed)
    fg = np.argwhere(label > 0)
    positions = []
    for _ in range(patches_per_case):
        if len(fg) > 0 and rng.rand() >= random_crop_prob:
            cz, cy, cx = fg[rng.randint(len(fg))]
            z1 = int(np.clip(cz - pd // 2, 0, d - pd))
            y1 = int(np.clip(cy - ph // 2, 0, h - ph))
            x1 = int(np.clip(cx - pw // 2, 0, w - pw))
        else:
            z1 = int(rng.randint(0, max(1, d - pd + 1)))
            y1 = int(rng.randint(0, max(1, h - ph + 1)))
            x1 = int(rng.randint(0, max(1, w - pw + 1)))
        positions.append((z1, y1, x1))
    return positions


def build_preprocessed_cache(cases, target_spacing, cache_dir, force=False, *,
                             load_label_fn, resample_fn,
                             normalize_fn=normalize_percentile):
    """构建同源图像、标签和前景索引缓存；调用者提供协议相关预处理策略。

    保留历史轴顺序、插值阶数、dtype、临时文件名和原子替换顺序。
    """
    import nibabel as nib

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(cases)
    n_cached, n_new = 0, 0
    print(f"[Cache] dir={cache_dir}")
    for idx, (case_id, ct_path, label_path) in enumerate(cases):
        img_p = cache_dir / f"{case_id}_image.npy"
        lbl_p = cache_dir / f"{case_id}_label.npy"
        fg_p = cache_dir / f"{case_id}_fg_indices.npz"
        if not force and img_p.exists() and lbl_p.exists() and fg_p.exists():
            n_cached += 1
            continue
        ct_nii = nib.load(ct_path)
        image = ct_nii.get_fdata().astype(np.float32)
        label_orig = load_label_fn(label_path)
        image = np.transpose(image, (2, 0, 1))
        label_orig = np.transpose(label_orig, (2, 0, 1))
        if target_spacing is not None:
            nii_spacing = ct_nii.header.get_zooms()[:3]
            orig_sp = (float(nii_spacing[2]), float(nii_spacing[1]), float(nii_spacing[0]))
            image = resample_fn(image, orig_sp, target_spacing, order=3)
            label_orig = resample_fn(
                label_orig.astype(np.float32), orig_sp, target_spacing, order=0
            ).astype(np.int16)
        image = normalize_fn(image)
        label_clean = np.clip(label_orig, 0, 5).astype(np.uint8)
        np.save(str(img_p) + ".tmp", image.astype(np.float32))
        np.save(str(lbl_p) + ".tmp", label_clean)
        fg_raveled = np.flatnonzero(label_clean > 0).astype(np.int64)
        d, h, w = label_clean.shape
        fg_tmp = str(fg_p) + ".tmp.npz"
        np.savez_compressed(fg_tmp, shape=np.array([d, h, w], dtype=np.int32), indices=fg_raveled)
        os.replace(str(img_p) + ".tmp.npy", img_p)
        os.replace(str(lbl_p) + ".tmp.npy", lbl_p)
        os.replace(fg_tmp, str(fg_p).replace(".npy", ".npz"))
        n_new += 1
        if n_new % 5 == 0 or (idx + 1) == n_total:
            print(f"  [{idx+1}/{n_total}] reused={n_cached} new={n_new} latest={case_id} shape={image.shape} fg={len(fg_raveled)}")
    print(f"[Cache] done. total={n_total} reused={n_cached} new={n_new}")
    return cache_dir


# Historical private names are deliberately retained as supported aliases.
_get_all_cases = get_all_cases
_create_folds = create_folds

__all__ = [
    "REPO", "RESULTS", "CACHE_DIR", "SPACING", "LR_AXIS", "MARGINS_MM",
    "PRIOR_BOX_MARGIN", "PRIOR_NAMES", "STAGE1_DIR", "PRIORS_DIR", "SKEL_DIR", "RUNS_DIR",
    "load_cached_array", "load_image", "load_label",
    "sample_validation_patch_positions", "build_preprocessed_cache",
    "DATA_ROOT_DEFAULT",
    "_RESULTS_ROOT",
    "DEFAULT_CACHE_DIR",
    "normalize_percentile",
    "get_all_cases",
    "create_folds",
    "_get_all_cases",
    "_create_folds",
]
