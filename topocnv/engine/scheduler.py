"""Learning-rate schedulers used by TopoCNV training wrappers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch.optim import Optimizer


class PolyLRWithWarmup:
    """Linear warmup followed by polynomial learning-rate decay.

    The implementation preserves the epoch-based behavior of the original
    training script. Parameter groups may define ``lr_scale`` to retain
    discriminative learning-rate ratios.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 20,
        power: float = 0.9,
        initial_lr: float = 1e-3,
        warmup_start_lr: float = 1e-5,
        min_lr: float = 1e-6,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if power <= 0:
            raise ValueError("power must be positive")
        if min(initial_lr, warmup_start_lr, min_lr) < 0:
            raise ValueError("learning rates must be non-negative")

        self.optimizer = optimizer
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.power = float(power)
        self.initial_lr = float(initial_lr)
        self.warmup_start_lr = float(warmup_start_lr)
        self.min_lr = float(min_lr)
        self._step = 0

    def _compute_lr(self, epoch: int) -> float:
        """Return the unscaled base learning rate for an epoch step."""
        epoch = max(int(epoch), 0)
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            ratio = epoch / self.warmup_epochs
            return self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * ratio

        decay_epochs = max(self.max_epochs - self.warmup_epochs, 1)
        progress = min(max((epoch - self.warmup_epochs) / decay_epochs, 0.0), 1.0)
        return max(self.initial_lr * (1.0 - progress) ** self.power, self.min_lr)

    def step(self, epoch: Optional[int] = None) -> None:
        """Advance one epoch, or set an explicit epoch, and update the optimizer."""
        self._step = self._step + 1 if epoch is None else max(int(epoch), 0)
        base_lr = self._compute_lr(self._step)
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * float(group.get("lr_scale", 1.0))

    def get_last_lr(self) -> List[float]:
        """Return current learning rates for all parameter groups."""
        base_lr = self._compute_lr(self._step)
        return [base_lr * float(group.get("lr_scale", 1.0)) for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scheduler progress and hyperparameters."""
        return {
            "max_epochs": self.max_epochs,
            "warmup_epochs": self.warmup_epochs,
            "power": self.power,
            "initial_lr": self.initial_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "min_lr": self.min_lr,
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore scheduler state and synchronize optimizer learning rates."""
        self.max_epochs = int(state_dict.get("max_epochs", self.max_epochs))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        self.power = float(state_dict.get("power", self.power))
        self.initial_lr = float(state_dict.get("initial_lr", self.initial_lr))
        self.warmup_start_lr = float(state_dict.get("warmup_start_lr", self.warmup_start_lr))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self._step = int(state_dict.get("step", state_dict.get("_step", 0)))
        self.step(self._step)


class LegacyPolyLRWithWarmup:
    """历史入口调度器；保留单元素 base LR 与直接恢复 _step 的契约。"""

    def __init__(self, optimizer, max_epochs, warmup_epochs=20, power=0.9,
                 initial_lr=1e-3, warmup_start_lr=1e-5, min_lr=1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.power = power
        self.initial_lr = initial_lr
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            if 'lr_scale' in pg:
                pg['lr'] = lr * pg['lr_scale']
            else:
                pg['lr'] = lr

    def _compute_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            return max(self.initial_lr * (1 - progress) ** self.power, self.min_lr)

    def get_last_lr(self):
        return [self._compute_lr(self._step)]


__all__ = ["PolyLRWithWarmup", "LegacyPolyLRWithWarmup"]
