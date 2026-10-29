"""Reusable spatial and intensity augmentations for 3D TopoCNV data."""

from typing import List, Sequence, Tuple

import numpy as np

try:
    import scipy.ndimage as ndi

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    ndi = None
    HAS_SCIPY = False


def _pad_to_shape_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    assert arr.ndim == 3
    pads = []
    for current, target in zip(arr.shape, target_shape):
        total = max(target - current, 0)
        before = total // 2
        pads.append((before, total - before))
    return np.pad(arr, pads, mode="constant", constant_values=0)


def random_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    patch_size,
    ensure_foreground: bool = True,
    foreground_ratio: float = 0.001,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crop a 3D pair using the historical foreground policy."""
    assert image.shape == label.shape
    d, h, w = image.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        target = (max(d, pd), max(h, ph), max(w, pw))
        image = _pad_to_shape_3d(image, target)
        label = _pad_to_shape_3d(label, target)
        d, h, w = image.shape

    if not ensure_foreground:
        z1 = np.random.randint(0, d - pd + 1)
        y1 = np.random.randint(0, h - ph + 1)
        x1 = np.random.randint(0, w - pw + 1)
        return (
            image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
            label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
        )

    foreground_coords = np.argwhere(label > 0)
    if len(foreground_coords) == 0:
        z1 = np.random.randint(0, d - pd + 1)
        y1 = np.random.randint(0, h - ph + 1)
        x1 = np.random.randint(0, w - pw + 1)
        return (
            image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
            label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
        )

    minimum = max(1, int(pd * ph * pw * foreground_ratio))
    for _attempt in range(max_attempts):
        fg_z, fg_y, fg_x = foreground_coords[np.random.randint(0, len(foreground_coords))]
        z1_min, z1_max = max(0, fg_z - pd + 1), min(d - pd, fg_z)
        y1_min, y1_max = max(0, fg_y - ph + 1), min(h - ph, fg_y)
        x1_min, x1_max = max(0, fg_x - pw + 1), min(w - pw, fg_x)
        if z1_max < z1_min:
            z1_min = z1_max = max(0, min(d - pd, fg_z))
        if y1_max < y1_min:
            y1_min = y1_max = max(0, min(h - ph, fg_y))
        if x1_max < x1_min:
            x1_min = x1_max = max(0, min(w - pw, fg_x))
        z1 = np.random.randint(z1_min, z1_max + 1) if z1_max > z1_min else z1_min
        y1 = np.random.randint(y1_min, y1_max + 1) if y1_max > y1_min else y1_min
        x1 = np.random.randint(x1_min, x1_max + 1) if x1_max > x1_min else x1_min
        label_patch = label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]
        if np.sum(label_patch > 0) >= minimum:
            return image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw], label_patch

    return (
        image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
        label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw],
    )


def random_flip_3d(
    image: np.ndarray, label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip a 3D pair independently on each axis with probability 0.5."""
    assert image.shape == label.shape
    for axis in range(3):
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    return image, label


def random_rotate90_3d(
    image: np.ndarray, label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a 3D pair by 0/90/180/270 degrees in the H-W plane."""
    assert image.shape == label.shape
    k = np.random.randint(0, 4)
    if k > 0:
        image = np.rot90(image, k=k, axes=(1, 2))
        label = np.rot90(label, k=k, axes=(1, 2))
    return image, label


def _rotation_matrix_3d(angles_rad: Tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = angles_rad
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]],
        dtype=np.float32,
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]],
        dtype=np.float32,
    )
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return rz @ ry @ rx


def random_affine_3d(
    image: np.ndarray,
    label: np.ndarray,
    p: float = 0.3,
    rotation_deg: float = 15.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the historical approximate nnU-Net 3D affine transform."""
    if not HAS_SCIPY or np.random.rand() > p:
        return image, label
    assert image.shape == label.shape
    angles = np.deg2rad(
        np.random.uniform(-rotation_deg, rotation_deg, size=3).astype(np.float32)
    )
    matrix = _rotation_matrix_3d(tuple(angles)) * np.random.uniform(*scale_range)
    center = np.array([size / 2.0 for size in image.shape], dtype=np.float32)
    offset = center - matrix @ center
    return (
        ndi.affine_transform(image, matrix, offset=offset, order=1, mode="constant", cval=0.0),
        ndi.affine_transform(label, matrix, offset=offset, order=0, mode="constant", cval=0.0),
    )


def random_gamma(
    image: np.ndarray,
    p: float = 0.3,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    """Apply gamma after min-max normalization, preserving the input range."""
    if np.random.rand() > p:
        return image
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max - image_min < 1e-6:
        return image
    normalized = (image - image_min) / (image_max - image_min)
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    return (np.power(normalized, gamma) * (image_max - image_min) + image_min).astype(
        np.float32
    )


def random_gaussian_noise(
    image: np.ndarray,
    p: float = 0.3,
    sigma_range: Tuple[float, float] = (0.0, 0.1),
) -> np.ndarray:
    """Add float32 Gaussian noise with a uniformly sampled sigma."""
    if np.random.rand() > p:
        return image
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if sigma <= 0.0:
        return image
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return (image + noise).astype(np.float32)


def augment_cached_patch(
    image: np.ndarray, label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the cached SOTA recipe with the original RNG call order."""
    image, label = random_flip_3d(image, label)
    image, label = random_rotate90_3d(image, label)
    image = random_gamma(image, p=0.15, gamma_range=(0.7, 1.5))
    image = random_gaussian_noise(image, p=0.1, sigma_range=(0.0, 0.05))
    return np.clip(image, 0.0, 1.0), label


def augment_aligned_arrays(
    arrays: Sequence[np.ndarray], image_idx: int = 0
) -> List[np.ndarray]:
    """Apply SOTA spatial transforms jointly to arrays ending in D,H,W.

    The selected image is expected to be channel-first; only channel zero gets
    gamma/noise, matching the online v4 training augmentation.
    """
    augmented = list(arrays)
    for axis in range(3):
        if np.random.rand() < 0.5:
            augmented = [
                np.flip(array, axis=array.ndim - 3 + axis) for array in augmented
            ]
    k = np.random.randint(0, 4)
    if k > 0:
        augmented = [
            np.rot90(array, k=k, axes=(array.ndim - 2, array.ndim - 1))
            for array in augmented
        ]
    image = np.array(augmented[image_idx], dtype=np.float32)
    ct = random_gamma(np.ascontiguousarray(image[0]), p=0.15, gamma_range=(0.7, 1.5))
    ct = random_gaussian_noise(ct, p=0.1, sigma_range=(0.0, 0.05))
    image[0] = np.clip(ct, 0.0, 1.0)
    augmented[image_idx] = image
    return augmented


__all__ = [
    "HAS_SCIPY",
    "random_crop_3d",
    "random_flip_3d",
    "random_rotate90_3d",
    "random_affine_3d",
    "random_gamma",
    "random_gaussian_noise",
    "augment_cached_patch",
    "augment_aligned_arrays",
]
