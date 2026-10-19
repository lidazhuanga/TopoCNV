"""
Fast cached dataset for SOTA model training — reuses the preprocessed cache.

The cache at ``results/preprocess_cache/sp_2.500_1.133_1.133/`` contains
resampled + percentile-normalised images saved as memmap-friendly .npy files.
This module provides a drop-in replacement for ``NPCDataset`` that loads from
the cache instead of doing on-the-fly resampling, yielding a 3-5× speedup on
data loading.

Usage::

    from npc_data_cached import CachedSOTADataset
    ds = CachedSOTADataset(cases, patch_size=(128,128,128), training=True)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import npc_preprocess as pp

try:
    from scipy.ndimage import gaussian_filter as _scipy_gf
except ImportError:
    _scipy_gf = None

# Default cache location (same as the proposed model's preprocessed data)
_RESULTS_ROOT = os.environ.get(
    "CNV_RESULTS_ROOT", str(Path(__file__).resolve().parent / "results"))
DEFAULT_CACHE_DIR = str(Path(_RESULTS_ROOT) / "preprocess_cache" / "sp_2.500_1.133_1.133")


def _load_from_cache(case_id: str, cache_dir: str):
    """Memory-map the cached image and label arrays."""
    cp = Path(cache_dir)
    img = np.load(cp / f"{case_id}_image.npy", mmap_mode="r")
    lbl = np.load(cp / f"{case_id}_label.npy", mmap_mode="r")
    return img, lbl


def _load_fg_indices(case_id: str, cache_dir: str):
    """Load pre-computed foreground voxel indices as (z, y, x) array [N, 3]."""
    npz_path = Path(cache_dir) / f"{case_id}_fg_indices.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    shape = data["shape"]  # [d, h, w]
    d, h, w = int(shape[0]), int(shape[1]), int(shape[2])
    raveled = data["indices"]
    z = raveled // (h * w)
    y = (raveled % (h * w)) // w
    x = raveled % w
    return np.stack([z, y, x], axis=-1), (d, h, w)


class CachedSOTADataset(Dataset):
    """Fast 3D patch dataset loading from a preprocessed .npy cache.

    The cache is assumed to have been built by
    ``npc_vesselFM3D_cascade_MIA_v3_vit_memory.prepare_preprocessed_cache``,
    i.e. images are percentile-normalised to [0, 1] and already resampled
    to the target spacing.  This class skips resampling and CT HU
    normalisation entirely — only cropping and augmentation are applied.

    Parameters
    ----------
    cases: List[Tuple[str, str, str]]
        Each element is ``(case_id, image_path, label_path)``.  Only
        ``case_id`` is used for cache lookup; the paths are ignored.
    patch_size: Tuple[int, int, int]
        (D, H, W) patch size.
    cache_dir: str
        Path to the prepared cache.
    training: bool
        If True, random foreground-biased cropping + augmentation.
        If False, center crop only.
    """

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
        # Probability of a fully random (possibly background-only) crop during
        # training. nnU-Net-style mixed sampling: with p=random_crop_prob the
        # patch is drawn uniformly from the volume, so the model also learns to
        # predict all-background; the rest stays foreground-biased.
        self.random_crop_prob = float(random_crop_prob)

        # Pre-load foreground indices (a few KB per case) for fast sampling
        self._fg_cache: dict = {}
        if training:
            for case_id, _, _ in cases:
                fg_data = _load_fg_indices(case_id, cache_dir)
                if fg_data[0] is not None:
                    self._fg_cache[case_id] = fg_data

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id, _img_path, _lbl_path = self.cases[idx]

        # Memory-map the cached arrays
        image_mm, label_mm = _load_from_cache(case_id, self.cache_dir)
        d, h, w = image_mm.shape
        pd, ph, pw = self.patch_size
        need_pad = d < pd or h < ph or w < pw

        if self.training:
            if self.random_crop_prob > 0 and np.random.rand() < self.random_crop_prob:
                # nnU-Net-style mixed sampling: uniform random crop (may be all background)
                z1 = np.random.randint(0, max(1, d - pd + 1))
                y1 = np.random.randint(0, max(1, h - ph + 1))
                x1 = np.random.randint(0, max(1, w - pw + 1))
            # Foreground-biased random crop
            else:
                fg_data = self._fg_cache.get(case_id)
                if fg_data is not None and len(fg_data[0]) > 0:
                    fg_coords, _vol_shape = fg_data
                    fi = np.random.randint(len(fg_coords))
                    cz, cy, cx = fg_coords[fi]
                    z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                    y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                    x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                else:
                    # Fallback: load label to find foreground (should rarely happen)
                    label_full = np.array(label_mm, dtype=np.int16)
                    if np.any(label_full > 0):
                        fg = np.argwhere(label_full > 0)
                        fi = np.random.randint(len(fg))
                        cz, cy, cx = fg[fi]
                        z1 = int(np.clip(cz - pd // 2, 0, d - pd))
                        y1 = int(np.clip(cy - ph // 2, 0, h - ph))
                        x1 = int(np.clip(cx - pw // 2, 0, w - pw))
                    else:
                        z1 = np.random.randint(0, max(1, d - pd + 1))
                        y1 = np.random.randint(0, max(1, h - ph + 1))
                        x1 = np.random.randint(0, max(1, w - pw + 1))
        else:
            # Center crop for validation
            z1 = (d - pd) // 2
            y1 = (h - ph) // 2
            x1 = (w - pw) // 2

        z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw

        if need_pad:
            # Rare case: volume smaller than patch — pad first then crop
            image_full = np.array(image_mm, dtype=np.float32)
            label_full = np.array(label_mm, dtype=np.int16)
            nd, nh, nw = max(d, pd), max(h, ph), max(w, pw)
            image_full = pp._pad_to_shape_3d(image_full, (nd, nh, nw))
            label_full = pp._pad_to_shape_3d(label_full, (nd, nh, nw))
            image = image_full[z1:z2, y1:y2, x1:x2]
            label = label_full[z1:z2, y1:y2, x1:x2]
        else:
            image = np.array(image_mm[z1:z2, y1:y2, x1:x2], dtype=np.float32)
            label = np.array(label_mm[z1:z2, y1:y2, x1:x2], dtype=np.int16)

        # --- Augmentation (same as proposed model's Stage1Dataset) ---
        if self.training:
            # Random flip (per axis, 50% probability)
            for ax in range(3):
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=ax)
                    label = np.flip(label, axis=ax)
            # Random 90-degree rotation in H-W plane
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2))
                label = np.rot90(label, k=k, axes=(1, 2))
            # Intensity augmentations
            image = pp.random_gamma(image, p=0.15, gamma_range=(0.7, 1.5))
            image = pp.random_gaussian_noise(image, p=0.1, sigma_range=(0.0, 0.05))
            image = np.clip(image, 0.0, 1.0)

        # To tensor: add channel dim for image
        image_t = torch.from_numpy(np.ascontiguousarray(image)[None]).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
        return image_t, label_t
