"""TopoCNV 记忆库实现与统一导出。"""
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .memory_bank_v1 import AnatomySlotMemoryBank
from .vit_unetr_mla import ContextAnchorMemoryBank, MLAMemoryBank, _MLACrossFusion


class AnatomySlotMemoryBankV4(AnatomySlotMemoryBank):
    """坐标无关、读出与塑形共享因子化分配的 anatomy-slot memory。"""

    def __init__(self, *args, usage_momentum: float = 0.99,
                 revive_every: int = 200, revive_thresh: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_momentum = usage_momentum
        self.revive_every = revive_every
        self.revive_thresh = revive_thresh
        for level in range(self.num_levels):
            self.register_buffer(
                f"usage_level{level}",
                torch.ones(self.num_prototypes) / self.num_prototypes)
        self.register_buffer("_revive_step", torch.zeros((), dtype=torch.long))

    def compute_assignments(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        latents = self._token_latents(tokens, level)
        sim = self._block_logits(latents, level)
        tau = self.temperature
        block_logit = torch.logsumexp(sim / tau, dim=3)
        p_class = F.softmax(block_logit, dim=-1)
        p_slot = F.softmax(sim / tau, dim=3)
        assignments = p_class.unsqueeze(-1) * p_slot
        return assignments.reshape(assignments.shape[0], assignments.shape[1],
                                   self.num_prototypes)

    @torch.no_grad()
    def _revive_dead_slots(self, level: int) -> int:
        usage = getattr(self, f"usage_level{level}")
        slots = self.get_slots(level)
        nb, K, _ = slots.shape
        usage_bk = usage.view(nb, K)
        revived = 0
        mean_usage = usage.mean()
        for block in range(nb):
            dead = torch.nonzero(usage_bk[block] < self.revive_thresh,
                                 as_tuple=False).flatten()
            if dead.numel() == 0 or dead.numel() == K:
                continue
            donor = int(usage_bk[block].argmax())
            for slot in dead.tolist():
                new_slot = slots.data[block, donor] + 0.1 * torch.randn_like(
                    slots.data[block, donor])
                slots.data[block, slot] = F.normalize(new_slot, dim=-1)
                usage_bk[block, slot] = mean_usage
                revived += 1
        return revived

    def compute_prototype_loss(
        self,
        features_list: List[torch.Tensor],
        labels: torch.Tensor,
        grid_shapes: List[Tuple[int, int, int]],
        cls_weight: float = 1.0,
        ot_weight: float = 0.5,
        smooth_weight: float = 0.2,
        div_weight: float = 0.1,
        div_margin: float = 0.5,
    ) -> torch.Tensor:
        K = self.num_slots_per_class
        nb = self.num_blocks
        tau = self.temperature
        cls_loss = features_list[0].new_zeros((), dtype=torch.float32)
        ot_loss = features_list[0].new_zeros((), dtype=torch.float32)
        smooth_loss = features_list[0].new_zeros((), dtype=torch.float32)
        div_loss = features_list[0].new_zeros((), dtype=torch.float32)
        n_ot = 0
        n_smooth = 0
        for level, features in enumerate(features_list):
            B, N, _ = features.shape
            gd, gh, gw = grid_shapes[level]
            latents = self._token_latents(features, level)
            sim = self._block_logits(latents, level)
            dom = self._labels_to_token_grid(labels, grid_shapes[level])
            block_logits = torch.logsumexp(sim / tau, dim=3)
            per_token_ce = F.cross_entropy(
                block_logits.reshape(-1, nb), dom.reshape(-1), reduction="none")
            per_token_ce = per_token_ce.view(B, N)
            ce_per_class = []
            for cls in range(nb):
                mask = dom == cls
                if mask.any():
                    ce_per_class.append(per_token_ce[mask].mean())
            if ce_per_class:
                cls_loss = cls_loss + torch.stack(ce_per_class).mean()
            block_logp = F.log_softmax(sim / tau, dim=3)
            for batch in range(B):
                for cls in range(nb):
                    mask = dom[batch] == cls
                    n_cls = int(mask.sum())
                    if n_cls < 2:
                        continue
                    scores = sim[batch][mask][:, cls, :]
                    target = self._sinkhorn(
                        scores, eps=self.sinkhorn_eps, iters=self.sinkhorn_iters)
                    logp = block_logp[batch][mask][:, cls, :]
                    ot_loss = ot_loss + (-(target * logp).sum(dim=1)).mean()
                    n_ot += 1
            tok_logp = torch.gather(
                block_logp, 2,
                dom.view(B, N, 1, 1).expand(-1, -1, 1, K)).squeeze(2)
            tok_p = tok_logp.exp()
            tok_logp_g = tok_logp.view(B, gd, gh, gw, K)
            tok_p_g = tok_p.view(B, gd, gh, gw, K)
            dom_g = dom.view(B, gd, gh, gw)
            for axis in (1, 2, 3):
                logp_i = tok_logp_g.narrow(axis, 0, tok_logp_g.shape[axis] - 1)
                logp_j = tok_logp_g.narrow(axis, 1, tok_logp_g.shape[axis] - 1)
                p_i = tok_p_g.narrow(axis, 0, tok_p_g.shape[axis] - 1)
                p_j = tok_p_g.narrow(axis, 1, tok_p_g.shape[axis] - 1)
                same = (dom_g.narrow(axis, 0, dom_g.shape[axis] - 1) ==
                        dom_g.narrow(axis, 1, dom_g.shape[axis] - 1))
                if same.any():
                    kl_ij = (p_i * (logp_i - logp_j)).sum(-1)
                    kl_ji = (p_j * (logp_j - logp_i)).sum(-1)
                    smooth_loss = smooth_loss + 0.5 * (kl_ij + kl_ji)[same].mean()
                    n_smooth += 1
            flat = F.normalize(self.get_prototypes(level), dim=-1).float()
            gram = flat @ flat.t()
            P = gram.shape[0]
            off = gram - torch.diag(torch.diagonal(gram))
            div_loss = div_loss + F.relu(off - div_margin).sum() / (P * (P - 1))
            with torch.no_grad():
                assignments = self.compute_assignments(features, level)
                mass = assignments.mean(dim=(0, 1)).float()
                usage = getattr(self, f"usage_level{level}")
                usage.mul_(self.usage_momentum).add_(
                    mass, alpha=1 - self.usage_momentum)
        if self.revive_every > 0:
            with torch.no_grad():
                self._revive_step += 1
                if int(self._revive_step) % self.revive_every == 0:
                    for level in range(self.num_levels):
                        self._revive_dead_slots(level)
        n_levels = len(features_list)
        ot_loss = ot_loss / max(n_ot, 1)
        smooth_loss = smooth_loss / max(n_smooth, 1)
        return (cls_weight * cls_loss / n_levels + ot_weight * ot_loss +
                smooth_weight * smooth_loss + div_weight * div_loss / n_levels)


__all__ = [
    "_MLACrossFusion", "MLAMemoryBank", "ContextAnchorMemoryBank",
    "AnatomySlotMemoryBank", "AnatomySlotMemoryBankV4",
]


def _self_test():
    torch.manual_seed(0)
    dims = [48, 96, 192]
    bank = AnatomySlotMemoryBankV4(
        feature_dims=dims, num_classes=6, kv_latent_dim=24,
        num_slots_per_class=8, num_heads=4, head_dim=16, revive_every=2)
    grids = [(4, 6, 6), (2, 3, 3), (1, 2, 2)]
    feats = [torch.randn(2, d * h * w, c)
             for (d, h, w), c in zip(grids, dims)]
    labels = torch.randint(0, 6, (2, 32, 48, 48))
    assignments = bank.compute_assignments(feats[0], 0)
    assert torch.allclose(assignments.sum(-1),
                          torch.ones_like(assignments.sum(-1)), atol=1e-4)
    loss = bank.compute_prototype_loss(feats, labels, grids)
    loss.backward()
    assert torch.isfinite(loss)
    print("AnatomySlotMemoryBankV4 smoke test OK")


if __name__ == "__main__":
    _self_test()
