"""Image/label preprocessing migrated from the legacy ``npc_preprocess`` module."""

import itertools
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter, zoom

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    gaussian_filter = None
    zoom = None
    HAS_SCIPY = False

from .augmentation import (
    _rotation_matrix_3d,
    random_affine_3d,
    random_crop_3d,
    random_flip_3d,
    random_gamma,
    random_gaussian_noise,
    random_rotate90_3d,
)
from .utils import normalize_percentile


def normalize_ct(
    image: np.ndarray, clip_lower: float = -200.0, clip_upper: float = 300.0
) -> np.ndarray:
    """Clip CT Hounsfield units and apply whole-volume z-score normalization."""
    image = np.clip(image, clip_lower, clip_upper)
    mean = image.mean()
    std = image.std()
    if std < 1e-6:
        std = 1.0
    return ((image - mean) / std).astype(np.float32)


def compute_median_spacing(
    spacing_list: List[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Return the per-axis median spacing in z-y-x order."""
    if not spacing_list:
        raise ValueError("spacing_list 不能为空")
    return tuple(np.median(np.array(spacing_list), axis=0).tolist())


def resample_image_to_spacing(
    image: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 3,
    cval: float = 0.0,
) -> np.ndarray:
    """Resample a D-H-W array, or a D-H-W-C array channel by channel."""
    if not HAS_SCIPY:
        raise ImportError("需要安装 scipy 才能进行 spacing 重采样")
    zoom_factors = [
        original_spacing[index] / target_spacing[index] for index in range(3)
    ]
    if image.ndim == 4:
        resampled = np.stack(
            [
                zoom(
                    image[..., channel],
                    zoom_factors,
                    order=order,
                    mode="constant",
                    cval=cval,
                )
                for channel in range(image.shape[3])
            ],
            axis=-1,
        )
    elif image.ndim == 3:
        resampled = zoom(
            image, zoom_factors, order=order, mode="constant", cval=cval
        )
    else:
        raise ValueError(f"Expected image with 3 or 4 dimensions, got {image.ndim}D")
    return resampled.astype(image.dtype)


def save_normalization_params(params: dict, save_path: str) -> None:
    """Save preprocessing parameters as UTF-8 JSON."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(params, file, indent=2, ensure_ascii=False)
    print(f"归一化参数已保存到: {save_path}")


def load_normalization_params(load_path: str) -> dict:
    """Load preprocessing parameters from JSON."""
    if not Path(load_path).exists():
        raise FileNotFoundError(f"归一化参数文件不存在: {load_path}")
    with open(load_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _pad_to_shape_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    assert arr.ndim == 3
    pads = []
    for current, target in zip(arr.shape, target_shape):
        total = max(target - current, 0)
        before = total // 2
        pads.append((before, total - before))
    return np.pad(arr, pads, mode="constant", constant_values=0)


def _pad_or_crop_to_shape_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    """Center crop and/or center pad a 3D array to an exact shape."""
    assert arr.ndim == 3
    result = arr
    for axis in range(3):
        current = result.shape[axis]
        target = target_shape[axis]
        if current > target:
            start = (current - target) // 2
            slices = [slice(None)] * 3
            slices[axis] = slice(start, start + target)
            result = result[tuple(slices)]
        elif current < target:
            difference = target - current
            pads = [(0, 0)] * 3
            pads[axis] = (difference // 2, difference - difference // 2)
            result = np.pad(result, pads, mode="constant", constant_values=0)
    return result


def center_crop_3d(
    image: np.ndarray, label: np.ndarray, patch_size
) -> Tuple[np.ndarray, np.ndarray]:
    """Center crop a pair, symmetrically padding smaller dimensions first."""
    assert image.shape == label.shape
    d, h, w = image.shape
    pd, ph, pw = patch_size
    if d < pd or h < ph or w < pw:
        target = (max(d, pd), max(h, ph), max(w, pw))
        image = _pad_to_shape_3d(image, target)
        label = _pad_to_shape_3d(label, target)
        d, h, w = image.shape
    z1, y1, x1 = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
    return (
        image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
        label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
    )


def preprocess_training(
    image: np.ndarray,
    label: np.ndarray,
    patch_size,
    original_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    ensure_foreground: bool = True,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the legacy on-the-fly training preprocessing and augmentation."""
    if original_spacing is not None and target_spacing is not None:
        image = resample_image_to_spacing(
            image, original_spacing, target_spacing, order=3
        )
        label = resample_image_to_spacing(
            label, original_spacing, target_spacing, order=0
        )
    image = normalize_ct(image, clip_lower=clip_lower, clip_upper=clip_upper)
    image, label = random_crop_3d(
        image, label, patch_size, ensure_foreground=ensure_foreground
    )
    image, label = random_flip_3d(image, label)
    image, label = random_rotate90_3d(image, label)
    image, label = random_affine_3d(image, label)
    image = random_gamma(image)
    image = random_gaussian_noise(image)
    image = np.ascontiguousarray(image)
    label = np.ascontiguousarray(label)
    return (
        torch.from_numpy(image[None, ...]).float(),
        torch.from_numpy(label.astype(np.int64)),
    )


def preprocess_validation(
    image: np.ndarray,
    label: np.ndarray,
    patch_size,
    original_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run deterministic validation preprocessing."""
    if original_spacing is not None and target_spacing is not None:
        image = resample_image_to_spacing(
            image, original_spacing, target_spacing, order=3
        )
        label = resample_image_to_spacing(
            label, original_spacing, target_spacing, order=0
        )
    image = normalize_ct(image, clip_lower=clip_lower, clip_upper=clip_upper)
    image, label = center_crop_3d(image, label, patch_size)
    image = np.ascontiguousarray(image)
    label = np.ascontiguousarray(label)
    return (
        torch.from_numpy(image[None, ...]).float(),
        torch.from_numpy(label.astype(np.int64)),
    )


def compute_gaussian_weights(
    patch_size: Tuple[int, int, int], sigma_scale: float = 1.0 / 8
) -> np.ndarray:
    """Build the normalized Gaussian blending map used by sliding windows."""
    if not HAS_SCIPY:
        return np.ones(patch_size, dtype=np.float32)
    sigmas = [size * sigma_scale for size in patch_size]
    impulse = np.zeros(patch_size, dtype=np.float32)
    impulse[tuple(size // 2 for size in patch_size)] = 1.0
    weights = gaussian_filter(impulse, sigma=sigmas, mode="constant", cval=0.0)
    return (weights / weights.max()).astype(np.float32)


def compute_steps_for_sliding_window(
    image_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    step_size: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """Return all D-H-W starting coordinates for complete window coverage."""
    axes = []
    for image_size, patch_dimension in zip(image_shape, patch_size):
        if image_size <= patch_dimension:
            positions = [0]
        else:
            step = max(int(patch_dimension * step_size), 1)
            positions = list(range(0, image_size - patch_dimension + 1, step))
            if positions[-1] + patch_dimension < image_size:
                positions.append(image_size - patch_dimension)
        axes.append(positions)
    return list(itertools.product(axes[0], axes[1], axes[2]))


def sliding_window_inference(
    image: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    step_size: float = 0.5,
    device: str = "cuda",
    use_gaussian: bool = True,
    original_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    clip_lower: float = -200.0,
    clip_upper: float = 300.0,
    normalization: str = "ct",
) -> np.ndarray:
    """Run 3D sliding-window inference and return uint8 class indices."""
    model.eval()
    original_shape = image.shape
    if original_spacing is not None and target_spacing is not None:
        image = resample_image_to_spacing(
            image, original_spacing, target_spacing, order=3
        )
    resampled_shape = image.shape
    if normalization == "percentile":
        image = normalize_percentile(
            image, lower=1.0, upper=99.0, b_min=0.0, b_max=1.0
        )
    else:
        image = normalize_ct(image, clip_lower=clip_lower, clip_upper=clip_upper)

    d, h, w = image.shape
    pd, ph, pw = patch_size
    pad_d, pad_h, pad_w = max(pd - d, 0), max(ph - h, 0), max(pw - w, 0)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = _pad_to_shape_3d(image, (max(d, pd), max(h, ph), max(w, pw)))
        d, h, w = image.shape

    results = np.zeros((num_classes, d, h, w), dtype=np.float32)
    accumulated_weights = np.zeros((d, h, w), dtype=np.float32)
    weights = (
        compute_gaussian_weights(patch_size)
        if use_gaussian
        else np.ones(patch_size, dtype=np.float32)
    )
    steps = compute_steps_for_sliding_window((d, h, w), patch_size, step_size)
    print(
        f"  滑窗推理: 图像shape {(d, h, w)}, patch_size {patch_size}, 共 {len(steps)} 个窗口"
    )
    with torch.no_grad():
        for index, (z1, y1, x1) in enumerate(steps):
            z2, y2, x2 = z1 + pd, y1 + ph, x1 + pw
            patch = image[z1:z2, y1:y2, x1:x2]
            output = model(torch.from_numpy(patch[None, None, ...]).float().to(device))
            if isinstance(output, (list, tuple)):
                output = output[0]
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            for class_index in range(num_classes):
                results[class_index, z1:z2, y1:y2, x1:x2] += (
                    probabilities[class_index] * weights
                )
            accumulated_weights[z1:z2, y1:y2, x1:x2] += weights
            if (index + 1) % 10 == 0 or (index + 1) == len(steps):
                print(f"    进度: {index + 1}/{len(steps)}")

    accumulated_weights = np.maximum(accumulated_weights, 1e-8)
    for class_index in range(num_classes):
        results[class_index] /= accumulated_weights
    prediction = np.argmax(results, axis=0).astype(np.uint8)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        prediction = prediction[
            :resampled_shape[0], :resampled_shape[1], :resampled_shape[2]
        ]
    if original_spacing is not None and target_spacing is not None:
        prediction = resample_image_to_spacing(
            prediction.astype(np.float32),
            target_spacing,
            original_spacing,
            order=0,
        ).astype(np.uint8)
        if prediction.shape != original_shape:
            print(
                f"    警告: 重采样后形状 {prediction.shape} 与原始形状 {original_shape} 不一致,进行裁剪/填充"
            )
            prediction = _pad_or_crop_to_shape_3d(prediction, original_shape)
    return prediction


__all__ = [
    "HAS_SCIPY",
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
    "_rotation_matrix_3d",
    "random_affine_3d",
    "random_gamma",
    "random_gaussian_noise",
    "preprocess_training",
    "preprocess_validation",
    "compute_gaussian_weights",
    "compute_steps_for_sliding_window",
    "sliding_window_inference",
]
