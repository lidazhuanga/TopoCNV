"""Device-aware automatic mixed-precision helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch.optim import Optimizer

DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class AMPConfig:
    """Resolved mixed-precision policy for one runtime device."""

    enabled: bool
    device_type: str
    dtype: torch.dtype
    use_grad_scaler: bool


def resolve_amp_config(
    device: DeviceLike,
    enabled: bool = True,
    prefer_bfloat16: bool = True,
) -> AMPConfig:
    """Resolve a safe AMP policy, disabling AMP on non-CUDA devices.

    CUDA devices use bfloat16 when supported and requested; otherwise float16
    with gradient scaling. CPU and other backends deliberately fall back to
    float32 so callers can use the same code path in smoke tests.
    """
    resolved = torch.device(device)
    cuda_enabled = bool(enabled and resolved.type == "cuda" and torch.cuda.is_available())
    if not cuda_enabled:
        return AMPConfig(False, resolved.type, torch.float32, False)

    use_bfloat16 = bool(prefer_bfloat16 and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    return AMPConfig(True, "cuda", dtype, not use_bfloat16)


def create_grad_scaler(config: AMPConfig) -> Optional[Any]:
    """Create a gradient scaler when required by the resolved AMP policy."""
    if not config.enabled or not config.use_grad_scaler:
        return None
    try:
        return torch.amp.GradScaler(config.device_type, enabled=True)
    except TypeError:  # PyTorch 2.1 compatibility
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(config: AMPConfig) -> AbstractContextManager[Any]:
    """Return an autocast context, or a no-op context for CPU fallback."""
    if not config.enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=config.device_type, dtype=config.dtype, enabled=True)


class AMPController:
    """Small facade that keeps autocast and gradient scaling decisions together."""

    def __init__(self, config: AMPConfig) -> None:
        self.config = config
        self.scaler = create_grad_scaler(config)

    @classmethod
    def for_device(
        cls,
        device: DeviceLike,
        enabled: bool = True,
        prefer_bfloat16: bool = True,
    ) -> "AMPController":
        """Construct a controller from a device and user AMP preference."""
        return cls(resolve_amp_config(device, enabled, prefer_bfloat16))

    def autocast(self) -> AbstractContextManager[Any]:
        """Return the configured forward-pass context manager."""
        return autocast_context(self.config)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate a loss with scaling when active."""
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: Optimizer) -> None:
        """Unscale optimizer gradients before clipping or validation."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer) -> None:
        """Apply an optimizer step and update scaling state."""
        if self.scaler is None:
            optimizer.step()
        else:
            self.scaler.step(optimizer)
            self.scaler.update()

    def skip_step(self) -> None:
        """Advance scaler state after an intentionally skipped optimizer step."""
        if self.scaler is not None:
            self.scaler.update()

    def state_dict(self) -> Optional[Dict[str, Any]]:
        """Return scaler state, or ``None`` when scaling is inactive."""
        return None if self.scaler is None else self.scaler.state_dict()

    def load_state_dict(self, state_dict: Optional[Dict[str, Any]]) -> None:
        """Restore scaler state when this runtime uses gradient scaling."""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)


def legacy_amp_settings():
    """保持旧入口的 bf16/fp16 选择，包括 CPU 上的 fp16 默认值。"""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


__all__ = [
    "legacy_amp_settings",
    "AMPConfig",
    "AMPController",
    "autocast_context",
    "create_grad_scaler",
    "resolve_amp_config",
]
