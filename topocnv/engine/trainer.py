"""Model-agnostic epoch loops for TopoCNV training wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer

from .amp import AMPController

MetricValue = Union[float, int, torch.Tensor]
BatchStep = Callable[[nn.Module, Any], Union["StepResult", torch.Tensor, Tuple[torch.Tensor, Mapping[str, MetricValue]]]]
BatchMover = Callable[[Any, torch.device], Any]


@dataclass
class StepResult:
    """Loss and metrics produced by a model-specific batch adapter."""

    loss: torch.Tensor
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(frozen=True)
class EpochResult:
    """Aggregated output from one train or evaluation epoch."""

    loss: float
    metrics: Mapping[str, float]
    num_batches: int
    skipped_batches: int = 0
    optimizer_steps: int = 0

    def as_dict(self) -> Dict[str, float]:
        """Return loss and named metrics in a flat dictionary."""
        return {"loss": self.loss, **dict(self.metrics)}


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in common batch containers to a device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=device.type == "cuda")
    if isinstance(batch, Mapping):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def gradients_are_finite(model: nn.Module) -> bool:
    """Return ``False`` when any materialized gradient contains NaN or Inf."""
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _as_float(value: MetricValue) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Metric tensors must contain exactly one element")
        return float(value.detach().item())
    return float(value)


def _normalize_step_result(value: Any) -> StepResult:
    if isinstance(value, StepResult):
        result = value
    elif isinstance(value, torch.Tensor):
        result = StepResult(value)
    elif isinstance(value, tuple) and len(value) == 2:
        result = StepResult(value[0], value[1])
    else:
        raise TypeError("batch_step must return StepResult, a loss tensor, or (loss, metrics)")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise ValueError("StepResult.loss must be a scalar tensor")
    if result.weight <= 0:
        raise ValueError("StepResult.weight must be positive")
    return result


def _optimizer_step(
    model: nn.Module,
    optimizer: Optimizer,
    amp: AMPController,
    max_grad_norm: Optional[float],
    skip_nonfinite_grad: bool = True,
) -> bool:
    amp.unscale_(optimizer)
    finite = not skip_nonfinite_grad or gradients_are_finite(model)
    if finite:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        amp.step(optimizer)
    else:
        amp.skip_step()
    optimizer.zero_grad(set_to_none=True)
    return finite


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    optimizer: Optimizer,
    batch_step: BatchStep,
    device: Union[str, torch.device],
    *,
    grad_accum_steps: int = 1,
    max_grad_norm: Optional[float] = 1.0,
    amp: Optional[AMPController] = None,
    batch_mover: BatchMover = move_batch_to_device,
    on_nonfinite: Optional[Callable[[int, str], None]] = None,
    skip_nonfinite_grad: bool = True,
) -> EpochResult:
    """Train for one epoch using a model-specific batch adapter.

    ``batch_step(model, batch)`` owns model invocation and loss construction,
    which keeps this loop independent of cascade stages, heads, and losses.
    The loop provides device transfer, AMP, accumulation, finite checks,
    clipping, and weighted metric aggregation.
    """
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    resolved_device = torch.device(device)
    controller = amp or AMPController.for_device(resolved_device, enabled=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_weight = 0.0
    metric_sums: Dict[str, float] = {}
    batches = skipped = optimizer_steps = pending = 0

    for batch_index, raw_batch in enumerate(loader):
        batch = batch_mover(raw_batch, resolved_device)
        with controller.autocast():
            result = _normalize_step_result(batch_step(model, batch))
        batches += 1
        if not bool(torch.isfinite(result.loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            skipped += 1
            if on_nonfinite is not None:
                on_nonfinite(batch_index, "loss")
            continue

        controller.backward(result.loss / grad_accum_steps)
        pending += 1
        total_loss += float(result.loss.detach().item()) * result.weight
        total_weight += result.weight
        for name, value in result.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + _as_float(value) * result.weight

        if pending == grad_accum_steps:
            finite = _optimizer_step(model, optimizer, controller, max_grad_norm, skip_nonfinite_grad)
            optimizer_steps += int(finite)
            if not finite:
                skipped += 1
                if on_nonfinite is not None:
                    on_nonfinite(batch_index, "gradient")
            pending = 0

    if pending:
        finite = _optimizer_step(model, optimizer, controller, max_grad_norm, skip_nonfinite_grad)
        optimizer_steps += int(finite)
        if not finite:
            skipped += 1
            if on_nonfinite is not None:
                on_nonfinite(max(batches - 1, 0), "gradient")

    denominator = max(total_weight, 1.0)
    metrics = {name: value / denominator for name, value in metric_sums.items()}
    return EpochResult(total_loss / denominator, metrics, batches, skipped, optimizer_steps)


def make_optim(model, lr, epochs):
    """v4 的 Adam 与余弦退火工厂，最低学习率固定为初始值的 1%。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    return opt, sched


def train_epoch(model, loader, optimizer, step_loss, device):
    """适配旧版 v4 的调用签名、bf16 策略及指标字典返回值。"""
    from topocnv.engine.v4_inference import AMP_DTYPE

    controller = AMPController.for_device(device, enabled=AMP_DTYPE is not None)
    result = train_one_epoch(
        model, loader, optimizer, step_loss, device,
        amp=controller, batch_mover=lambda batch, _device: batch,
        skip_nonfinite_grad=False,
        on_nonfinite=lambda _index, _kind: print("  [WARN] non-finite loss, batch skipped", flush=True),
    )
    return dict(result.metrics)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    batch_step: BatchStep,
    device: Union[str, torch.device],
    *,
    amp: Optional[AMPController] = None,
    batch_mover: BatchMover = move_batch_to_device,
) -> EpochResult:
    """Evaluate one epoch with the same adapter contract as training."""
    resolved_device = torch.device(device)
    controller = amp or AMPController.for_device(resolved_device, enabled=False)
    model.eval()
    total_loss = total_weight = 0.0
    metric_sums: Dict[str, float] = {}
    batches = skipped = 0

    for raw_batch in loader:
        batch = batch_mover(raw_batch, resolved_device)
        with controller.autocast():
            result = _normalize_step_result(batch_step(model, batch))
        batches += 1
        if not bool(torch.isfinite(result.loss.detach())):
            skipped += 1
            continue
        total_loss += float(result.loss.detach().item()) * result.weight
        total_weight += result.weight
        for name, value in result.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + _as_float(value) * result.weight

    denominator = max(total_weight, 1.0)
    metrics = {name: value / denominator for name, value in metric_sums.items()}
    return EpochResult(total_loss / denominator, metrics, batches, skipped, 0)


def should_validate(epoch: int, interval: int, start_epoch: int = 0) -> bool:
    """Match the legacy policy: validate first epoch and every N epochs."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    return epoch == start_epoch or (epoch + 1) % interval == 0


__all__ = [
    "BatchMover",
    "BatchStep",
    "EpochResult",
    "StepResult",
    "evaluate_one_epoch",
    "gradients_are_finite",
    "move_batch_to_device",
    "should_validate",
    "train_one_epoch",
    "make_optim",
    "train_epoch",
]
