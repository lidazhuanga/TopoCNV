"""Training-loop pieces shared by the v4 Stage-1 and Stage-2 trainers (the SOTA DynUNet recipe:
Adam, cosine annealing to 1% of the lr, bf16 autocast, gradient clipping at 1.0, NaN batches skipped)."""
import os
from pathlib import Path

import numpy as np
import torch

from .infer import AMP_DTYPE


def make_optim(model, lr, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    return opt, sched


def train_epoch(model, loader, optimizer, step_loss, device):
    """step_loss(model, batch) -> (loss tensor, dict of float parts). Returns mean parts."""
    model.train()
    sums, n = {}, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE or torch.float16,
                                enabled=AMP_DTYPE is not None and device.type == "cuda"):
            loss, parts = step_loss(model, batch)
        if not torch.isfinite(loss):
            print("  [WARN] non-finite loss, batch skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def save_ckpt(path, **state):
    path = Path(path)
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
