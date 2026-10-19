"""AnatomySlotMemoryBankV6 — Semantic-Conditioned Memory (SCM).

Replaces SPM/V5's conditioning source (unregistered physical coordinates —
failed: -0.021 Dice on fold1, p<1e-4) with the Stage-1 prior softmax, a
semantically aligned space shared across patients. Two minimal changes on top
of the proven V4 machinery (slot layout, Sinkhorn, dead-slot revival and the
three shaping losses are all inherited unchanged):

1. Semantic condition injection into token latents
       latents = normalize(kv_down(tokens) + sem_proj(sem_tokens))
   where sem_tokens [B,N,2] are the Stage-1 prior channels (nerve_body, c5)
   average-pooled to the token grid. The prior already travels with the model
   input (flips, crops and sliding windows included), so there is no extra
   data pipeline and no train/test mismatch.

2. Per-block semantic anchor on the class posterior
       block_logit_b = logsumexp(cos(feat, slot_b)/tau) + beta * cos(sem, anchor_b)
   anchor_b is a learnable 2-dim vector per class block; beta anneals
   0 -> beta_final (cosine, driven by the trainer after memory warmup).
   beta starts at 0 so the worst case degrades gracefully to V4.

Expected mechanism (honest scope): the 2-channel prior carries NO c1-specific
probability (c1-c4 share the nerve_body channel), so the semantic anchor can
only provide coarse localization (nerve region / c5 foramen / background) —
its realistic benefit is background-FP suppression and c5 anchoring, NOT c1
recall. Anything called without ``sem`` behaves exactly like V4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from npc_anatomy_memory_v4 import AnatomySlotMemoryBankV4


class AnatomySlotMemoryBankV6(AnatomySlotMemoryBankV4):
    """V4 + Stage-1-softmax-conditioned prototypes. See module docstring."""

    def __init__(self, *args, sem_dim: int = 2, beta_final: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sem_dim = int(sem_dim)
        self.beta_final = float(beta_final)
        # Mixing weight for the semantic anchor term (annealed externally).
        self.register_buffer('beta', torch.tensor(0.0))
        # Semantic projection into the latent space (one per level).
        self.sem_projs = nn.ModuleList([
            nn.Linear(self.sem_dim, self.kv_latent_dim, bias=False)
            for _ in range(self.num_levels)])
        # Learnable semantic anchors [num_blocks, sem_dim] per level.
        for level in range(self.num_levels):
            anchors = torch.randn(self.num_blocks, self.sem_dim) * 0.1
            self.register_parameter(f'sem_anchors_level{level}', nn.Parameter(anchors))

    # ------------------------------------------------------------------ utils

    def get_sem_anchors(self, level: int) -> torch.Tensor:
        return getattr(self, f'sem_anchors_level{level}')

    @torch.no_grad()
    def set_beta(self, b: float) -> None:
        self.beta.fill_(float(b))

    def _token_latents_sem(self, tokens: torch.Tensor, level: int,
                           sem: torch.Tensor) -> torch.Tensor:
        lat = self.kv_downs[level](tokens) + self.sem_projs[level](sem.to(tokens.dtype))
        return F.normalize(lat, dim=-1)

    # ------------------------------------------------------------------ assignment

    def compute_assignments(self, tokens: torch.Tensor, level: int,
                            sem: torch.Tensor = None) -> torch.Tensor:
        """Factorized soft assignment [B,N,P]; semantic-conditioned when sem given."""
        if sem is None:
            return super().compute_assignments(tokens, level)
        latents = self._token_latents_sem(tokens, level, sem)
        sim = self._block_logits(latents, level)              # [B,N,nb,K]
        tau = self.temperature
        block_logit = torch.logsumexp(sim / tau, dim=3)       # [B,N,nb]
        beta = float(self.beta.item())
        if beta > 0:
            anchors = F.normalize(self.get_sem_anchors(level), dim=-1)   # [nb,2]
            sem_n = F.normalize(sem, dim=-1)                             # [B,N,2]
            block_logit = block_logit + beta * torch.einsum('bnc,oc->bno', sem_n, anchors)
        p_class = F.softmax(block_logit, dim=-1)
        p_slot = F.softmax(sim / tau, dim=3)
        A = p_class.unsqueeze(-1) * p_slot
        return A.reshape(A.shape[0], A.shape[1], self.num_prototypes)

    # ------------------------------------------------------------------ read-out

    def enhance_features(self, tokens: torch.Tensor, level: int, grid_shape,
                         sem: torch.Tensor = None) -> torch.Tensor:
        if not self.enabled:
            return tokens
        if sem is None:
            return super().enhance_features(tokens, level, grid_shape)
        B, N, C = tokens.shape
        w = self.compute_assignments(tokens, level, sem).to(tokens.dtype)
        slots = F.normalize(self.get_prototypes(level), dim=-1)
        v = self.v_ups[level](slots)
        mem = torch.matmul(w, v)
        mem = self.out_projs[level](mem)
        return self.memory_fusions[level](tokens, mem)


if __name__ == '__main__':
    # Smoke test: shapes, backward, beta annealing, sem-free fallback.
    torch.manual_seed(0)
    dims = [128, 256, 320]
    grids = [(2, 6, 6), (1, 3, 3), (1, 2, 2)]
    bank = AnatomySlotMemoryBankV6(feature_dims=dims, num_classes=6, kv_latent_dim=24,
                                   num_slots_per_class=8, num_heads=4, head_dim=8)
    bank.enabled.fill_(True)
    level = 0
    B, N, C = 2, grids[level][0] * grids[level][1] * grids[level][2], dims[level]
    tokens = torch.randn(B, N, C, requires_grad=True)
    sem = torch.rand(B, N, 2)
    bank.set_beta(0.1)  # anchor term active only when beta > 0
    out = bank.enhance_features(tokens, level, grids[level], sem)
    assert out.shape == tokens.shape
    out.sum().backward()
    assert bank.get_sem_anchors(level).grad is not None, "anchors must receive gradient"
    assert bank.sem_projs[level].weight.grad is not None, "sem_proj must receive gradient"
    A = bank.compute_assignments(tokens.detach(), level, sem)
    assert A.shape == (B, N, bank.num_prototypes)
    A_ns = bank.compute_assignments(tokens.detach(), level)  # V4 fallback
    assert A_ns.shape == (B, N, bank.num_prototypes)
    # bg token (sem ~ 0) must not produce NaN through F.normalize
    sem_bg = torch.zeros(1, 5, 2)
    t_bg = torch.randn(1, 5, C)
    A_bg = bank.compute_assignments(t_bg, level, sem_bg)
    assert torch.isfinite(A_bg).all(), "bg tokens must stay finite"
    print("V6 smoke test OK:",
          f"assign sum={A.sum(-1).mean().item():.3f}",
          f"beta={float(bank.beta):.2f}")
