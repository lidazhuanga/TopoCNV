"""Checkpoint persistence, resume, and early-stopping utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer

PathLike = Union[str, os.PathLike[str]]


def _state_dict(component: Optional[Any]) -> Optional[Dict[str, Any]]:
    if component is None:
        return None
    state = component.state_dict()
    return None if state is None else dict(state)


def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    *,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    best_metric: Optional[float] = None,
    best_epoch: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save a complete, resume-capable training checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        scheduler_state = _state_dict(scheduler)
        payload["scheduler_state_dict"] = scheduler_state
        if hasattr(scheduler, "_step"):
            payload["scheduler_step"] = int(scheduler._step)
    if scaler is not None:
        payload["scaler_state_dict"] = _state_dict(scaler)
    if best_metric is not None:
        payload["best_metric"] = float(best_metric)
        payload.setdefault("best_dice", float(best_metric))
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    if extra:
        payload.update(dict(extra))

    _save_payload(destination, payload, destination.with_name(f".{destination.name}.tmp"))
    return destination


def _save_payload(destination, payload, temporary):
    """复用原子保存，允许兼容接口保持既有 payload 和临时文件名。"""
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_ckpt(path, **state):
    """保持 v4 原有 checkpoint 字段，不添加统一格式的额外字段。"""
    destination = Path(path)
    _save_payload(destination, state, Path(str(destination) + ".tmp"))


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    *,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Any = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model state and any supplied training components.

    Both complete dictionaries and legacy model-only state dictionaries are
    accepted. The returned dictionary is the original checkpoint payload.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    try:
        loaded = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 2.1 compatibility
        loaded = torch.load(source, map_location=map_location)
    if not isinstance(loaded, MutableMapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(loaded).__name__}")

    payload = dict(loaded)
    model_state = payload.get("model_state_dict", payload)
    model.load_state_dict(model_state, strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        if payload.get("scheduler_state_dict") is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        elif "scheduler_step" in payload and hasattr(scheduler, "_step"):
            scheduler._step = int(payload["scheduler_step"])
            if hasattr(scheduler, "step"):
                scheduler.step(scheduler._step)
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


@dataclass
class EarlyStopping:
    """Track validation improvements and signal when patience is exhausted."""

    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best: Optional[float] = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

    def is_improvement(self, value: float) -> bool:
        """Return whether a monitored value improves over the current best."""
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float, epoch: int, increment: int = 1) -> bool:
        """Update state and return ``True`` when the value improved."""
        if increment <= 0:
            raise ValueError("increment must be positive")
        if self.is_improvement(float(value)):
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += int(increment)
        return False

    @property
    def should_stop(self) -> bool:
        """Whether the configured patience has been exhausted."""
        return self.best is not None and self.bad_epochs > 0 and self.bad_epochs >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        """Serialize early-stopping progress."""
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore early-stopping progress."""
        self.patience = int(state_dict.get("patience", self.patience))
        self.mode = str(state_dict.get("mode", self.mode))
        self.min_delta = float(state_dict.get("min_delta", self.min_delta))
        best = state_dict.get("best")
        self.best = None if best is None else float(best)
        self.best_epoch = int(state_dict.get("best_epoch", 0))
        self.bad_epochs = int(state_dict.get("bad_epochs", 0))
        self.__post_init__()


class CheckpointManager:
    """Manage conventional ``best_model`` and ``latest_checkpoint`` files."""

    def __init__(
        self,
        directory: PathLike,
        *,
        monitor: str = "val_loss",
        mode: str = "min",
        best_filename: str = "best_model.pth",
        latest_filename: str = "latest_checkpoint.pth",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_path = self.directory / best_filename
        self.latest_path = self.directory / latest_filename
        self.best_metric: Optional[float] = None
        self.best_epoch = 0

    def is_improvement(self, value: float) -> bool:
        """Return whether ``value`` improves the manager's best metric."""
        if self.best_metric is None:
            return True
        return value > self.best_metric if self.mode == "max" else value < self.best_metric

    def save(
        self,
        model: nn.Module,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Save latest state, update best state, and return improvement status."""
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' is missing")
        value = float(metrics[self.monitor])
        improved = self.is_improvement(value)
        if improved:
            self.best_metric = value
            self.best_epoch = int(epoch)

        common = dict(
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            metrics=metrics,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            extra=extra,
        )
        save_checkpoint(self.latest_path, model, **common)
        if improved:
            save_checkpoint(self.best_path, model, **common)
        return improved

    def load_latest(self, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
        """Restore the latest checkpoint and synchronize manager best state."""
        payload = load_checkpoint(self.latest_path, model, **kwargs)
        best = payload.get("best_metric", payload.get("best_dice"))
        self.best_metric = None if best is None else float(best)
        self.best_epoch = int(payload.get("best_epoch", 0))
        return payload


__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "load_checkpoint",
    "save_checkpoint",
    "save_ckpt",
]
