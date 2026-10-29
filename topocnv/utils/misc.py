"""Determinism, device selection, and small runtime helpers."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

DeviceLike = Union[str, torch.device]


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch and return the normalized seed."""
    normalized = int(seed) % (2**32)
    os.environ["PYTHONHASHSEED"] = str(normalized)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return normalized


def seed_worker(worker_id: int) -> None:
    """Initialize DataLoader worker RNGs from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: Optional[DeviceLike] = None) -> torch.device:
    """Resolve an explicit device or select CUDA, MPS, then CPU."""
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        if device.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError("MPS device requested but unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(value: Any, device: DeviceLike, non_blocking: Optional[bool] = None) -> Any:
    """Recursively move tensors inside mappings and sequences to a device."""
    resolved = torch.device(device)
    async_copy = resolved.type == "cuda" if non_blocking is None else non_blocking
    if isinstance(value, torch.Tensor):
        return value.to(resolved, non_blocking=async_copy)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, resolved, async_copy) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, resolved, async_copy) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, resolved, async_copy) for item in value]
    return value


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count all parameters or only parameters requiring gradients."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def ensure_tuple3(value: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    """Normalize an integer or three-element sequence to a spatial tuple."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("spatial values must be positive")
        return value, value, value
    normalized = tuple(int(item) for item in value)
    if len(normalized) != 3 or any(item <= 0 for item in normalized):
        raise ValueError("value must contain exactly three positive integers")
    return normalized


@contextmanager
def evaluating(model: nn.Module) -> Iterator[nn.Module]:
    """Temporarily switch a model to evaluation mode and restore its state."""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


__all__ = [
    "count_parameters",
    "ensure_tuple3",
    "evaluating",
    "move_to_device",
    "resolve_device",
    "seed_everything",
    "seed_worker",
]
