"""On-the-fly and cached dataset interfaces for TopoCNV."""

from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nrrd as _nrrd
except ImportError:  # pragma: no cover
    _nrrd = None

from . import preprocess as pp
from .augmentation import augment_cached_patch
from .utils import (
    DATA_ROOT_DEFAULT,
    DEFAULT_CACHE_DIR,
    _RESULTS_ROOT,
    _create_folds,
    _get_all_cases,
    create_folds,
    get_all_cases,
)


class NPCDataset(Dataset):
    """Legacy on-the-fly NIfTI/NRRD dataset with unchanged preprocessing."""

    def __init__(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
        patch_size: Tuple[int, int, int],
        training: bool = True,
        target_spacing: Tuple[float, float, float] = None,
        ensure_foreground: bool = True,
        clip_lower: float = -200.0,
        clip_upper: float = 300.0,
    ) -> None:
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.training = training
        self.target_spacing = target_spacing
        self.ensure_foreground = ensure_foreground
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image_nii = nib.load(str(image_path))
        image = image_nii.get_fdata()

        if str(label_path).endswith(".nrrd") and _nrrd is not None:
            label, _ = _nrrd.read(str(label_path))
            if label.ndim == 4:
                label = label[0]
        else:
            label = nib.load(str(label_path)).get_fdata()

        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int16)
        if image.ndim != 3:
            raise ValueError(
                f"Expect 3D image, got shape {image.shape} from {image_path}"
            )
        if label.shape != image.shape:
            raise ValueError(
                f"Image and label shape mismatch: {image.shape} vs {label.shape} for {image_path}"
            )

        original_spacing = None
        if self.target_spacing is not None:
            spacing = image_nii.header.get_zooms()[:3]
            original_spacing = (
                float(spacing[2]),
                float(spacing[1]),
                float(spacing[0]),
            )
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        if self.training:
            return pp.preprocess_training(
                image,
                label,
                self.patch_size,
                original_spacing=original_spacing,
                target_spacing=self.target_spacing,
                ensure_foreground=self.ensure_foreground,
                clip_lower=self.clip_lower,
                clip_upper=self.clip_upper,
            )
        return pp.preprocess_validation(
            image,
            label,
            self.patch_size,
            original_spacing=original_spacing,
            target_spacing=self.target_spacing,
            clip_lower=self.clip_lower,
            clip_upper=self.clip_upper,
        )


def get_train_val_datasets(
    data_root: str,
    patch_size: Tuple[int, int, int],
    fold: int,
    num_folds: int = 5,
    seed: int = 42,
    target_spacing: Tuple[float, float, float] = None,
    ensure_foreground: bool = True,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
) -> Tuple[NPCDataset, NPCDataset, List[List[str]]]:
    """Build the historical train/validation pair and return all folds."""
    cases = _get_all_cases(data_root)
    folds = _create_folds(
        [case[0] for case in cases], num_folds=num_folds, seed=seed
    )
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}], got {fold}")
    validation_ids = set(folds[fold])
    train_images, train_labels, validation_images, validation_labels = [], [], [], []
    for patient_id, image_path, label_path in cases:
        if patient_id in validation_ids:
            validation_images.append(image_path)
            validation_labels.append(label_path)
        else:
            train_images.append(image_path)
            train_labels.append(label_path)

    train_dataset = NPCDataset(
        train_images,
        train_labels,
        patch_size,
        training=True,
        target_spacing=target_spacing,
        ensure_foreground=ensure_foreground,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    validation_dataset = NPCDataset(
        validation_images,
        validation_labels,
        patch_size,
        training=False,
        target_spacing=target_spacing,
        ensure_foreground=False,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    return train_dataset, validation_dataset, folds


def _load_from_cache(case_id: str, cache_dir: str):
    """Memory-map one cached image/label pair."""
    from topocnv.data.utils import load_cached_array

    return (
        load_cached_array(case_id, "image", cache_dir=cache_dir),
        load_cached_array(case_id, "label", cache_dir=cache_dir),
    )


def _load_fg_indices(case_id: str, cache_dir: str):
    """Load foreground coordinates and cached D-H-W shape."""
    path = Path(cache_dir) / f"{case_id}_fg_indices.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    d, h, w = (int(value) for value in data["shape"])
    indices = data["indices"]
    z = indices // (h * w)
    y = (indices % (h * w)) // w
    x = indices % w
    return np.stack([z, y, x], axis=-1), (d, h, w)


load_from_cache = _load_from_cache
load_fg_indices = _load_fg_indices


class CachedSOTADataset(Dataset):
    """Fast patch dataset backed by resampled percentile-normalized ``.npy`` files."""

    def __init__(
        self,
        cases: List[Tuple[str, str, str]],
        patch_size: Tuple[int, int, int],
        cache_dir: str = DEFAULT_CACHE_DIR,
        training: bool = True,
        random_crop_prob: float = 0.0,
    ):
        self.cases = cases
        self.patch_size = patch_size
        self.cache_dir = cache_dir
        self.training = training
        self.random_crop_prob = float(random_crop_prob)
        self._fg_cache: dict = {}
        if training:
            for case_id, _, _ in cases:
                foreground = _load_fg_indices(case_id, cache_dir)
                if foreground[0] is not None:
                    self._fg_cache[case_id] = foreground

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id, _image_path, _label_path = self.cases[idx]
        image_mm, label_mm = _load_from_cache(case_id, self.cache_dir)
        d, h, w = image_mm.shape
        pd, ph, pw = self.patch_size
        need_pad = d < pd or h < ph or w < pw

        if self.training:
            if self.random_crop_prob > 0 and np.random.rand() < self.random_crop_prob:
                z1 = np.random.randint(0, max(1, d - pd + 1))
                y1 = np.random.randint(0, max(1, h - ph + 1))
                x1 = np.random.randint(0, max(1, w - pw + 1))
            else:
                foreground = self._fg_cache.get(case_id)
                if foreground is not None and len(foreground[0]) > 0:
                    coordinates, _volume_shape = foreground
                    cz, cy, cx = coordinates[np.random.randint(len(coordinates))]
                    z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                    y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                    x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                else:
                    label_full = np.array(label_mm, dtype=np.int16)
                    if np.any(label_full > 0):
                        coordinates = np.argwhere(label_full > 0)
                        cz, cy, cx = coordinates[
                            np.random.randint(len(coordinates))
                        ]
                        z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                        y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                        x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                    else:
                        z1 = np.random.randint(0, max(1, d - pd + 1))
                        y1 = np.random.randint(0, max(1, h - ph + 1))
                        x1 = np.random.randint(0, max(1, w - pw + 1))
        else:
            z1, y1, x1 = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2

        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
        if need_pad:
            image_full = np.array(image_mm, dtype=np.float32)
            label_full = np.array(label_mm, dtype=np.int16)
            target = (max(d, pd), max(h, ph), max(w, pw))
            image_full = pp._pad_to_shape_3d(image_full, target)
            label_full = pp._pad_to_shape_3d(label_full, target)
            image = image_full[z1:z2, y1:y2, x1:x2]
            label = label_full[z1:z2, y1:y2, x1:x2]
        else:
            image = np.array(
                image_mm[z1:z2, y1:y2, x1:x2], dtype=np.float32
            )
            label = np.array(
                label_mm[z1:z2, y1:y2, x1:x2], dtype=np.int16
            )

        if self.training:
            image, label = augment_cached_patch(image, label)
        return (
            torch.from_numpy(np.ascontiguousarray(image)[None]).float(),
            torch.from_numpy(np.ascontiguousarray(label).astype(np.int64)),
        )


__all__ = [
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
]
