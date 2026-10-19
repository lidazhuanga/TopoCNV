"""AnatomySlotMemoryBankV5 — Spatial-anchored Prototype Memory (SPM-lite).

Extends V4 with two minimal, single-variable changes (class-topology graph
regularization from the original SPM proposal is deliberately DROPPED: the
c1-c5 classes are nested expansion rings around CN V, not sequential segments
along the nerve, so a chain-manifold prior is factually wrong for this label
schema):

1. Anatomical position encoding fused into token latents
   latents = normalize(kv_down(tokens) + pos_proj(sinusoidal_3d(coords)))
   Coordinates are patient-volume-normalized [0,1]^3 — computable at BOTH train
   and inference time (unlike foreground-bbox normalization, which needs labels
   and would create a train/test mismatch).

2. Learnable spatial center per prototype slot ([0,1]^3). Assignment similarity
   becomes an annealed mix of content cosine and a spatial Gaussian:
       sim = alpha * cos_sim + (1 - alpha) * (2*exp(-d^2/2sigma^2) - 1)
   The spatial term is rescaled to (-1,1] to match the cosine range before
   mixing (avoids tau-scale domination). alpha anneals 1.0 -> 0.7 with a cosine
   schedule driven by the trainer (memory warmup ends -> anneal over 150 epochs).
   Centers are plain nn.Parameters trained by gradient through the read-out
   path (consistent with V4 slot training; no second EMA chain).

Anything called without ``coords`` behaves exactly like V4 (content-only), so
prototype-init, reports and legacy eval paths are unaffected.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from npc_anatomy_memory_v4 import AnatomySlotMemoryBankV4


class AnatomySlotMemoryBankV5(AnatomySlotMemoryBankV4):
    """V4 + spatial-anchored prototypes. See module docstring."""

    def __init__(self, *args, spatial_sigma: float = 0.15,
                 alpha_final: float = 0.7, pos_freqs: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_sigma = float(spatial_sigma)
        self.alpha_final = float(alpha_final)
        self.pos_freqs = int(pos_freqs)
        # Mixing weight for content vs position (annealed externally).
        self.register_buffer('alpha', torch.tensor(1.0))
        pe_dim = 3 * 2 * self.pos_freqs
        # Position projection into the latent space (one per level).
        self.pos_projs = nn.ModuleList([
            nn.Linear(pe_dim, self.kv_latent_dim, bias=False)
            for _ in range(self.num_levels)])
        # Learnable spatial centers [num_blocks, K, 3], init in [0.2, 0.8]^3.
        for level in range(self.num_levels):
            centers = torch.rand(self.num_blocks, self.num_slots_per_class, 3) * 0.6 + 0.2
            self.register_parameter(f'centers_level{level}', nn.Parameter(centers))

    # ------------------------------------------------------------------ utils

    def get_centers(self, level: int) -> torch.Tensor:
        """[num_blocks, K, 3] spatial centers for one level."""
        return getattr(self, f'centers_level{level}')

    @torch.no_grad()
    def set_alpha(self, a: float) -> None:
        self.alpha.fill_(float(a))

    def _pos_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Sinusoidal 3D PE: [B,N,3] -> [B,N, 3*2*pos_freqs]."""
        feats = []
        for f in range(self.pos_freqs):
            w = (2.0 ** f) * math.pi
            feats.append(torch.sin(coords * w))
            feats.append(torch.cos(coords * w))
        return torch.cat(feats, dim=-1)

    def _token_latents_pos(self, tokens: torch.Tensor, level: int,
                           coords: Optional[torch.Tensor]) -> torch.Tensor:
        lat = self.kv_downs[level](tokens)
        if coords is not None:
            lat = lat + self.pos_projs[level](
                self._pos_encoding(coords).to(lat.dtype))
        return F.normalize(lat, dim=-1)

    # ------------------------------------------------------------------ assignment

    def compute_assignments(self, tokens: torch.Tensor, level: int,
                            coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Factorized soft assignment [B,N,P]; content+position when coords given."""
        if coords is None:
            return super().compute_assignments(tokens, level)
        latents = self._token_latents_pos(tokens, level, coords)
        sim = self._block_logits(latents, level)              # [B,N,nb,K] cosine
        alpha = float(self.alpha.item())
        if alpha < 1.0:
            centers = self.get_centers(level)                 # [nb,K,3]
            d2 = ((coords.unsqueeze(2).unsqueeze(3) - centers) ** 2).sum(-1)
            spatial = torch.exp(-d2 / (2 * self.spatial_sigma ** 2))
            spatial = spatial * 2.0 - 1.0                     # -> (-1, 1], match cosine
            sim = alpha * sim + (1 - alpha) * spatial.to(sim.dtype)
        tau = self.temperature
        block_logit = torch.logsumexp(sim / tau, dim=3)       # [B,N,nb]
        p_class = F.softmax(block_logit, dim=-1)
        p_slot = F.softmax(sim / tau, dim=3)                  # [B,N,nb,K]
        A = p_class.unsqueeze(-1) * p_slot
        return A.reshape(A.shape[0], A.shape[1], self.num_prototypes)

    # ------------------------------------------------------------------ read-out

    def enhance_features(self, tokens: torch.Tensor, level: int,
                         grid_shape: Tuple[int, int, int],
                         coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return tokens
        if coords is None:
            return super().enhance_features(tokens, level, grid_shape)
        B, N, C = tokens.shape
        w = self.compute_assignments(tokens, level, coords).to(tokens.dtype)
        slots = F.normalize(self.get_prototypes(level), dim=-1)
        v = self.v_ups[level](slots)                          # [P, H*d]
        mem = torch.matmul(w, v)                              # [B,N,H*d]
        mem = self.out_projs[level](mem)                      # [B,N,C]
        return self.memory_fusions[level](tokens, mem)


if __name__ == '__main__':
    # Smoke test: shapes, backward, alpha annealing, coords-free fallback.
    torch.manual_seed(0)
    dims = [128, 256, 320]
    grids = [(2, 6, 6), (1, 3, 3), (1, 2, 2)]
    bank = AnatomySlotMemoryBankV5(feature_dims=dims, num_classes=6, kv_latent_dim=24,
                                   num_slots_per_class=8, num_heads=4, head_dim=8)
    bank.enabled.fill_(True)
    level = 0
    B, N, C = 2, grids[level][0] * grids[level][1] * grids[level][2], dims[level]
    tokens = torch.randn(B, N, C, requires_grad=True)
    coords = torch.rand(B, N, 3)
    bank.set_alpha(0.85)  # spatial branch active only when alpha < 1
    out = bank.enhance_features(tokens, level, grids[level], coords)
    assert out.shape == tokens.shape
    out.sum().backward()
    assert bank.get_centers(level).grad is not None, "centers must receive gradient"
    bank.set_alpha(0.7)
    A = bank.compute_assignments(tokens.detach(), level, coords)
    assert A.shape == (B, N, bank.num_prototypes)
    A_nc = bank.compute_assignments(tokens.detach(), level)  # V4 fallback
    assert A_nc.shape == (B, N, bank.num_prototypes)
    print("V5 smoke test OK:",
          f"assign sum={A.sum(-1).mean().item():.3f}",
          f"alpha={float(bank.alpha):.2f}")
