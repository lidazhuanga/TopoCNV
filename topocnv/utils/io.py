"""Filesystem and medical-image I/O with lazy optional dependencies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

PathLike = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class MedicalImage:
    """Array data plus format-specific spatial metadata."""

    data: np.ndarray
    affine: Optional[np.ndarray] = None
    header: Any = None


def read_json(path: PathLike) -> Any:
    """Read a UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: PathLike, value: Any, *, indent: int = 2) -> Path:
    """Atomically write a UTF-8 JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=indent)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_numpy(path: PathLike, *, mmap_mode: Optional[str] = None) -> np.ndarray:
    """Load an NPY array, optionally memory mapped."""
    return np.load(Path(path), mmap_mode=mmap_mode, allow_pickle=False)


def save_numpy(path: PathLike, array: np.ndarray) -> Path:
    """Atomically save an NPY array without pickle support."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(array), allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_nifti(path: PathLike, *, dtype: Optional[np.dtype[Any]] = np.float32) -> MedicalImage:
    """Load NIfTI data; ``nibabel`` is imported only when this function is called."""
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError("NIfTI I/O requires the optional 'nibabel' package") from error
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    return MedicalImage(data=data, affine=np.asarray(image.affine), header=image.header.copy())


def save_nifti(
    path: PathLike,
    data: np.ndarray,
    affine: Optional[np.ndarray] = None,
    *,
    header: Any = None,
    dtype: Optional[np.dtype[Any]] = None,
) -> Path:
    """Save a NIfTI image while importing ``nibabel`` lazily."""
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError("NIfTI I/O requires the optional 'nibabel' package") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=dtype)
    transform = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine)
    nib.save(nib.Nifti1Image(array, transform, header=header), str(destination))
    return destination


def load_nrrd(path: PathLike, *, dtype: Optional[np.dtype[Any]] = None) -> MedicalImage:
    """Load NRRD data; ``pynrrd`` is imported only when this function is called."""
    try:
        import nrrd
    except ImportError as error:
        raise ImportError("NRRD I/O requires the optional 'pynrrd' package") from error
    data, header = nrrd.read(str(path))
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    return MedicalImage(data=data, header=dict(header))


def save_nrrd(
    path: PathLike,
    data: np.ndarray,
    *,
    header: Optional[Mapping[str, Any]] = None,
    dtype: Optional[np.dtype[Any]] = None,
) -> Path:
    """Save NRRD data while importing ``pynrrd`` lazily."""
    try:
        import nrrd
    except ImportError as error:
        raise ImportError("NRRD I/O requires the optional 'pynrrd' package") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(destination), np.asarray(data, dtype=dtype), header=dict(header or {}))
    return destination


def load_medical_image(path: PathLike, *, dtype: Optional[np.dtype[Any]] = np.float32) -> MedicalImage:
    """Dispatch loading by ``.nii``, ``.nii.gz``, or ``.nrrd`` suffix."""
    source = Path(path)
    name = source.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return load_nifti(source, dtype=dtype)
    if name.endswith(".nrrd"):
        return load_nrrd(source, dtype=dtype)
    raise ValueError(f"Unsupported medical-image format: {source}")


def save_medical_image(path: PathLike, image: MedicalImage, *, dtype: Optional[np.dtype[Any]] = None) -> Path:
    """Dispatch saving a :class:`MedicalImage` by destination suffix."""
    destination = Path(path)
    name = destination.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return save_nifti(
            destination,
            image.data,
            image.affine,
            header=image.header,
            dtype=dtype,
        )
    if name.endswith(".nrrd"):
        header: Optional[Dict[str, Any]]
        header = dict(image.header) if isinstance(image.header, Mapping) else None
        return save_nrrd(destination, image.data, header=header, dtype=dtype)
    raise ValueError(f"Unsupported medical-image format: {destination}")


__all__ = [
    "MedicalImage",
    "load_medical_image",
    "load_nifti",
    "load_nrrd",
    "load_numpy",
    "read_json",
    "save_medical_image",
    "save_nifti",
    "save_nrrd",
    "save_numpy",
    "write_json",
]
