"""AnatomySlotMemoryBankV4MGP — Minimal Gated Prototypes.

Strips V4 down to its core hypothesis: can prototypes learn useful representations
purely from the main segmentation loss gradient (no auxiliary losses) with
top-k gated readout (no full-softmax noise dilution)?

Three changes on top of V4:
1. ``compute_prototype_loss`` returns zero — all four auxiliary terms (L_cls,
   L_ot, L_smooth, L_div) are removed. Prototypes learn ONLY through the
   read-out gradient flowing back from the main Dice/BCE loss.
2. ``compute_assignments`` — factorized softmax (p_class × p_slot) followed by
   top-k gating: only the top ``top_k_gate`` (default 8) out of ``P`` prototypes
   contribute to each token's read-out; the rest are masked to -inf before
   re-normalization. This reduces noise dilution by P/k = 192/8 = 24×.
3. Warmup is extended to 100 epochs (set via trainer CLI arg, NOT in this class)
   — prototypes need more main-loss gradient iterations before being trusted for
   read-out.

Dead-slot revival from V4 is retained as a safety net (no-grad, no effect on
loss graph) but with top-k gating, unused slots naturally receive zero gradient
and the revival becomes a softer necessity.

Design rationale:
- Full softmax over 192 prototypes: every prototype contributes to every token,
  including cosine_sim ≈ 0.05 noise → noise overwhelms signal.
- Sinkhorn OT forces equal slot usage → destroys natural anatomical clustering.
- Smoothness loss trades boundary precision for neighbor consensus (bad for
  fine nerve structures).
- By KEEPING only what's necessary (factorized class×slot structure) and
  REMOVING everything else, we test the minimal viable memory bank hypothesis.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

from npc_anatomy_memory_v4 import AnatomySlotMemoryBankV4


class AnatomySlotMemoryBankV4MGP(AnatomySlotMemoryBankV4):
    """V4 stripped to Minimal Gated Prototypes. See module docstring."""

    def __init__(self, *args, top_k_gate: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_gate = int(top_k_gate)

    # ------------------------------------------------------------------ assignment

    def compute_assignments(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        """Factorized soft assignment + top-k gating: [B, N, num_prototypes].

        Same factorized decomposition as V4 (p_class × p_slot), followed by
        masking all but the top-k prototypes to -inf and re-normalizing via
        softmax. Each token retrieves from at most ``top_k_gate`` prototypes
        instead of all P.
        """
        latents = self._token_latents(tokens, level)            # [B,N,kv]
        sim = self._block_logits(latents, level)                # [B,N,nb,K] f32
        tau = self.temperature

        # Factorized: p_class (which class block?) × p_slot (which slot within block?)
        block_logit = torch.logsumexp(sim / tau, dim=3)         # [B,N,nb]
        p_class = F.softmax(block_logit, dim=-1)                # [B,N,nb]
        p_slot = F.softmax(sim / tau, dim=3)                    # [B,N,nb,K]
        A = p_class.unsqueeze(-1) * p_slot                      # [B,N,nb,K]
        A_flat = A.reshape(A.shape[0], A.shape[1], self.num_prototypes)  # [B,N,P]

        # Top-k gating: keep only the strongest k assignments per token,
        # mask the rest to -inf, re-normalize.
        k = min(self.top_k_gate, self.num_prototypes)
        if k >= self.num_prototypes:
            return A_flat  # no gating needed

        topk_vals, _ = A_flat.topk(k, dim=-1)                   # [B,N,k]
        threshold = topk_vals.min(dim=-1, keepdim=True).values  # [B,N,1]
        mask = A_flat >= threshold
        A_gated = A_flat.masked_fill(~mask, float('-inf'))
        A_gated = F.softmax(A_gated, dim=-1)                    # re-normalize within top-k
        return A_gated

    # ------------------------------------------------------------------ losses

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
        """Minimal: all auxiliary losses removed.

        Returns a zero scalar loss so the trainer's loss aggregation is
        unchanged, but no auxiliary gradient competes with the main task.

        Dead-slot revival (V4 safety net) is still run under no_grad.  It
        becomes less critical with top-k gating (unused slots naturally
        receive ~zero gradient) but serves as a hygiene checkpoint.
        """
        # Still refresh usage EMA and revive dead slots (no-grad safety net).
        with torch.no_grad():
            for level, features in enumerate(features_list):
                A = self.compute_assignments(features, level)    # [B,N,P]
                mass = A.mean(dim=(0, 1)).float()                # [P]
                usage = getattr(self, f'usage_level{level}')
                usage.mul_(self.usage_momentum).add_(
                    mass, alpha=1 - self.usage_momentum)

            if self.revive_every > 0:
                self._revive_step += 1
                if int(self._revive_step) % self.revive_every == 0:
                    for level in range(self.num_levels):
                        self._revive_dead_slots(level)

        # Zero loss — prototypes learn purely from main loss gradient.
        return features_list[0].new_zeros((), dtype=torch.float32)


# ------------------------------------------------------------------ smoke test

if __name__ == '__main__':
    torch.manual_seed(0)
    dims = [128, 256, 320]
    grids = [(2, 6, 6), (1, 3, 3), (1, 2, 2)]
    bank = AnatomySlotMemoryBankV4MGP(
        feature_dims=dims, num_classes=6, kv_latent_dim=24,
        num_slots_per_class=8, num_heads=4, head_dim=8,
        top_k_gate=3)   # small for smoke test
    bank.enabled.fill_(True)

    # Test top-k gating: only up to top_k_gate non-zero entries per token row.
    tokens = torch.randn(2, 36, dims[0])  # B=2, N=2*6*6=72 (grid says 36? minor)
    # Use actual grid shape
    level = 0
    B, N, C = 2, grids[level][0] * grids[level][1] * grids[level][2], dims[level]
    tokens = torch.randn(B, N, C)
    A = bank.compute_assignments(tokens, level)
    assert A.shape == (B, N, bank.num_prototypes), \
        f"expected {(B, N, bank.num_prototypes)}, got {A.shape}"
    # Each row must have at most top_k_gate non-zero entries
    nonzeros = (A > 0).sum(dim=-1)
    assert (nonzeros <= bank.top_k_gate).all(), \
        f"top-k gating violated: max nonzero={nonzeros.max()}, expected <= {bank.top_k_gate}"
    assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-4), \
        "assignment rows must sum to 1"

    # Test zero proto loss
    labels = torch.randint(0, 6, (B, 32, 48, 48))
    feats = [tokens]
    loss = bank.compute_prototype_loss(feats, labels, grids[:1])
    assert loss.item() == 0.0, f"proto loss must be zero, got {loss.item()}"

    # Test readout path
    out = bank.enhance_features(tokens, level, grids[level])
    assert out.shape == tokens.shape
    out.sum().backward()
    g = bank.get_slots(level).grad
    assert g is not None and torch.isfinite(g).all(), "slots must receive gradient"

    print(f"MGP smoke test OK: top_k_gate={bank.top_k_gate}, "
          f"max nonzero={nonzeros.max().item()}, "
          f"slot grad norm={float(g.norm()):.4f}")
