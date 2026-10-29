"""
Anatomy Slot Memory Bank (解剖槽位记忆库) for 3D medical image segmentation.

实现迁移自顶层 ``npc_anatomy_memory``；模块名变化不影响参数层级与 state_dict key。

Drop-in replacement for MLAMemoryBank / ContextAnchorMemoryBank with the same
interface (compute_prototype_loss / enhance_features / forward / get_prototypes /
num_prototypes / enabled buffer), designed to fix two empirically diagnosed
failure modes of the previous banks on this project:

  1. Utilization collapse (only 8-13 of 320 prototypes used). Root cause: the
     old enhance_features averaged ALL patch latents into ONE query per volume
     (query_avg) and selected top-k prototypes per SAMPLE, so every patch in a
     volume shared the same handful of prototypes.
  2. Functionally random prototypes (1-NN class purity ~= chance). Root cause:
     winner-take-all updates + no balanced-assignment mechanism; the
     context-anchor variant additionally collapsed to uniform assignment
     because feature-space anchors have no grounding signal on this small
     dataset.

Design
------
Slot layout per level: (C_fg + 1) blocks x K slots
  - one block per foreground class + one background block (background also has
    anatomical context: skull, air, soft tissue ...)
  - slots are learnable latent vectors, normalized on use.

Training signal (compute_prototype_loss), per level, at NATIVE token resolution:
  1. L_cls : class-balanced cross-entropy over block logits (max within block)
             -> decides WHICH anatomical class a slot block serves.
  2. L_ot  : per-class Sinkhorn optimal-transport assignment. Tokens of class c
             are matched to the K slots of block c with (approximately) EQUAL
             mass per slot (SwAV-style). Balanced utilization is enforced BY
             CONSTRUCTION, not by a weak KL-to-uniform proxy.
  3. L_smooth: spatial smoothness on the token grid. Adjacent tokens with the
             same dominant class must share slot-assignment distributions
             (symmetric KL). This is AUGMENTATION-IMMUNE: it uses only local
             neighborhood relations, no absolute coordinates, so it survives
             the random fg-centred crops + random flips/rot90 of the training
             pipeline. It yields spatially contiguous "anatomical territories",
             each owned by a slot.
  4. L_div : pairwise cosine hinge between all slots -> slots stay distinct.

Read-out (enhance_features), per level:
  - PER-TOKEN soft assignment over ALL slots (cosine in latent space, no
    winner-take-all, no per-volume averaging) -> every slot receives gradient
    through the segmentation loss every step.
  - value read-out + output projection + MLA cross-attention fusion
    (_MLACrossFusion imported from npc_vit_mla).

Verification API
----------------
anatomy_report(tokens, level, grid_shape, labels=None, coords=None) returns a
JSON-serializable dict with:
  - utilization: slot mass entropy, effective slot count exp(H), active count
  - class purity: per-slot dominant-class fraction (needs labels)
  - spatial localization: per-slot assignment-weighted centroid and spread in
    caller-provided normalized coordinates (e.g. foreground-bbox frame at
    full-volume inference), so cross-patient anatomical consistency can be
    measured externally (see MIA/scripts/verify_anatomy_slots.py).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from topocnv.models.vit_unetr_mla import _MLACrossFusion


class AnatomySlotMemoryBank(nn.Module):
    """Anatomy-slot memory bank. See module docstring for the full rationale.

    Args:
        feature_dims:         feature dimensions per encoder level, e.g. [192, 384, 768]
        num_classes:          total classes INCLUDING background (stage1: 3, stage2: 6)
        kv_latent_dim:        latent dimension of slots (default 96)
        num_slots_per_class:  K slots per (foreground or background) class block
        num_heads / head_dim: attention config of the MLA cross-fusion block
        temperature:          softmax temperature for assignment / read-out
        top_k:                kept for interface compatibility (soft read-out uses
                              all slots; stored but not used for sparsification)
        momentum:             kept for interface compatibility (slots are plain
                              gradient-updated parameters, no EMA)
        sinkhorn_eps:         entropic regularization of the OT assignment
        sinkhorn_iters:       Sinkhorn normalization iterations
        patch_grid:           kept for interface compatibility (unused; the bank
                              operates at native token resolution)
    """

    def __init__(
        self,
        feature_dims: List[int] = (192, 384, 768),
        num_classes: int = 6,
        kv_latent_dim: int = 96,
        num_slots_per_class: int = 32,
        num_heads: int = 6,
        head_dim: int = 32,
        temperature: float = 0.5,
        top_k: int = 8,
        momentum: float = 0.99,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 10,
        patch_grid: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.feature_dims = list(feature_dims)
        self.num_levels = len(feature_dims)
        self.num_classes = num_classes
        self.num_fg_classes = num_classes - 1
        self.num_blocks = num_classes          # bg block + one per fg class
        self.num_slots_per_class = num_slots_per_class
        self.num_prototypes = self.num_blocks * num_slots_per_class
        self.kv_latent_dim = kv_latent_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temperature = temperature
        self.top_k = top_k
        self.momentum = momentum
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.patch_grid = patch_grid

        # Warmup control (same protocol as MLAMemoryBank): enhancement bypassed
        # until the trainer flips this buffer; shaping losses run from epoch 0.
        self.register_buffer('enabled', torch.tensor(False, dtype=torch.bool))

        # ---- slots: [num_blocks, K, kv_dim] per level, learnable ----
        for level in range(self.num_levels):
            slots = torch.randn(self.num_blocks, num_slots_per_class, kv_latent_dim)
            slots = F.normalize(slots, dim=-1)
            self.register_parameter(f'slots_level{level}', nn.Parameter(slots))

        # ---- per-level projections ----
        self.kv_downs = nn.ModuleList([
            nn.Linear(dim, kv_latent_dim, bias=False) for dim in feature_dims])
        self.v_ups = nn.ModuleList([
            nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)
            for _ in feature_dims])
        self.out_projs = nn.ModuleList([
            nn.Linear(num_heads * head_dim, dim, bias=False) for dim in feature_dims])

        # ---- per-level MLA cross-attention fusion (tokens <-> memory read-out) ----
        self.memory_fusions = nn.ModuleList([
            _MLACrossFusion(
                dim=dim, num_heads=num_heads, head_dim=head_dim,
                kv_latent_dim=kv_latent_dim, dropout=0.1)
            for dim in feature_dims
        ])

        # flat slot index -> block (class) index, for verification / reports
        self.register_buffer(
            'slot_block_ids',
            torch.arange(self.num_blocks).unsqueeze(1)
                 .expand(-1, num_slots_per_class).flatten())

    # ------------------------------------------------------------------ utils

    def get_slots(self, level: int) -> torch.Tensor:
        """[num_blocks, K, kv_dim] raw slot parameters for one level."""
        return getattr(self, f'slots_level{level}')

    def get_prototypes(self, level: int) -> torch.Tensor:
        """Flat [num_prototypes, kv_dim] view (interface compatibility)."""
        return self.get_slots(level).view(self.num_prototypes, self.kv_latent_dim)

    def _token_latents(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        """Project encoder tokens to the normalized latent space: [B, N, kv]."""
        return F.normalize(self.kv_downs[level](tokens), dim=-1)

    def _block_logits(self, latents: torch.Tensor, level: int) -> torch.Tensor:
        """Cosine similarity to every slot, grouped by block.

        Args:
            latents: [B, N, kv] normalized token latents
        Returns:
            [B, N, num_blocks, K] similarities (float32)
        """
        slots = F.normalize(self.get_slots(level), dim=-1)  # [nb, K, kv]
        flat = slots.view(-1, self.kv_latent_dim).float()   # [nb*K, kv]
        sim = latents.float() @ flat.t()                    # [B, N, nb*K]
        return sim.view(latents.shape[0], latents.shape[1],
                        self.num_blocks, self.num_slots_per_class)

    def _labels_to_token_grid(
        self,
        labels: torch.Tensor,
        grid_shape: Tuple[int, int, int],
        fg_thresh: float = 0.10,
    ) -> torch.Tensor:
        """Dominant class per token at the native token grid (hard labels).

        Same rule as MLAMemoryBank: a token takes the foreground class with the
        largest occupancy fraction; it stays background only if no foreground
        class exceeds ``fg_thresh``.  Small nerves are therefore not washed out.

        Args:
            labels:     [B, D, H, W] integer GT labels
            grid_shape: (gd, gh, gw) token grid of this level
        Returns:
            [B, gd*gh*gw] long tensor with values in 0..num_classes-1
        """
        gd, gh, gw = grid_shape
        oh = F.one_hot(
            labels.long().clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 4, 1, 2, 3).float()                 # [B, C, D, H, W]
        pooled = F.adaptive_avg_pool3d(oh, (gd, gh, gw))  # class occupancy per token
        fg = pooled[:, 1:]                                # [B, C-1, gd, gh, gw]
        fg_max, fg_arg = fg.max(dim=1)
        dom = torch.where(
            fg_max > fg_thresh, fg_arg + 1, torch.zeros_like(fg_arg))
        return dom.flatten(1)                             # [B, gd*gh*gw]

    @staticmethod
    @torch.no_grad()
    def _sinkhorn(logits: torch.Tensor, eps: float = 0.1, iters: int = 3) -> torch.Tensor:
        """Balanced soft assignment of N tokens to K slots (SwAV-style).

        Args:
            logits: [N, K] assignment scores (float32, detached internally)
        Returns:
            [N, K] assignment matrix: rows sum to 1 (each token distributes a
            unit mass), column sums are approximately N / K (equal slot mass).
        """
        N, K = logits.shape
        Q = torch.exp(logits / eps).t()                   # [K, N]
        Q /= Q.sum().clamp_min(1e-8)
        for _ in range(iters):
            Q /= Q.sum(dim=1, keepdim=True).clamp_min(1e-8)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-8)
            Q /= N
        Q *= N                                            # columns sum to 1
        return Q.t()                                      # [N, K]

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
        """Anatomy-slot shaping loss, averaged over levels.

        Terms (see module docstring): L_cls (block CE), L_ot (per-class
        Sinkhorn assignment), L_smooth (same-class neighbor consensus),
        L_div (slot diversity hinge).  All computed in float32 for stability
        under bf16 autocast.
        """
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

            latents = self._token_latents(features, level)          # [B,N,kv]
            sim = self._block_logits(latents, level)                # [B,N,nb,K] f32
            dom = self._labels_to_token_grid(labels, grid_shapes[level])  # [B,N]

            # ---- 1. block classification CE (class-balanced) ----
            block_logits = sim.max(dim=3).values / tau              # [B,N,nb]
            per_token_ce = F.cross_entropy(
                block_logits.reshape(-1, nb), dom.reshape(-1), reduction='none')
            per_token_ce = per_token_ce.view(B, N)
            ce_per_class = []
            for c in range(nb):
                m = dom == c
                if m.any():
                    ce_per_class.append(per_token_ce[m].mean())
            if ce_per_class:
                cls_loss = cls_loss + torch.stack(ce_per_class).mean()

            # log-softmax over slots within each block (reused by OT/smooth)
            block_logp = F.log_softmax(sim / tau, dim=3)            # [B,N,nb,K]

            # ---- 2. per-class Sinkhorn OT assignment ----
            for b in range(B):
                for c in range(nb):
                    m = dom[b] == c
                    n_c = int(m.sum())
                    if n_c < 2:
                        continue
                    scores = sim[b][m][:, c, :]                     # [N_c, K]
                    target = self._sinkhorn(
                        scores, eps=self.sinkhorn_eps, iters=self.sinkhorn_iters)
                    logp = block_logp[b][m][:, c, :]                # [N_c, K]
                    ot_loss = ot_loss + (-(target * logp).sum(dim=1)).mean()
                    n_ot += 1

            # ---- 3. same-class neighbor smoothness (augmentation-immune) ----
            # per-token assignment distribution over its OWN block's slots
            tok_logp = torch.gather(
                block_logp, 2,
                dom.view(B, N, 1, 1).expand(-1, -1, 1, K)).squeeze(2)  # [B,N,K]
            tok_p = tok_logp.exp()
            tok_logp_g = tok_logp.view(B, gd, gh, gw, K)
            tok_p_g = tok_p.view(B, gd, gh, gw, K)
            dom_g = dom.view(B, gd, gh, gw)
            for axis in (1, 2, 3):  # d, h, w neighbors
                logp_i = tok_logp_g.narrow(axis, 0, tok_logp_g.shape[axis] - 1)
                logp_j = tok_logp_g.narrow(axis, 1, tok_logp_g.shape[axis] - 1)
                p_i = tok_p_g.narrow(axis, 0, tok_p_g.shape[axis] - 1)
                p_j = tok_p_g.narrow(axis, 1, tok_p_g.shape[axis] - 1)
                same = (dom_g.narrow(axis, 0, dom_g.shape[axis] - 1)
                        == dom_g.narrow(axis, 1, dom_g.shape[axis] - 1))
                if same.any():
                    kl_ij = (p_i * (logp_i - logp_j)).sum(-1)
                    kl_ji = (p_j * (logp_j - logp_i)).sum(-1)
                    smooth_loss = smooth_loss + 0.5 * (kl_ij + kl_ji)[same].mean()
                    n_smooth += 1

            # ---- 4. slot diversity hinge (flat, across all blocks) ----
            flat = F.normalize(self.get_prototypes(level), dim=-1).float()  # [P,kv]
            gram = flat @ flat.t()
            P = gram.shape[0]
            off = gram - torch.diag(torch.diagonal(gram))
            div_loss = div_loss + F.relu(off - div_margin).sum() / (P * (P - 1))

        n_levels = len(features_list)
        ot_loss = ot_loss / max(n_ot, 1)
        smooth_loss = smooth_loss / max(n_smooth, 1)

        return (cls_weight * cls_loss / n_levels
                + ot_weight * ot_loss
                + smooth_weight * smooth_loss
                + div_weight * div_loss / n_levels)

    # ------------------------------------------------------------------ read-out

    def compute_assignments(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        """Per-token soft assignment over ALL slots: [B, N, num_prototypes].

        Single consistent assignment definition shared by the read-out path,
        the shaping losses and the verification API.
        """
        latents = self._token_latents(tokens, level)
        slots = F.normalize(self.get_prototypes(level), dim=-1)     # [P, kv]
        logits = latents.float() @ slots.float().t() / self.temperature
        return F.softmax(logits, dim=-1)                            # [B, N, P]

    def enhance_features(
        self,
        tokens: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Per-token soft prototype read-out + MLA cross-attention fusion.

        Every token retrieves from EVERY slot (softmax-weighted value sum) --
        no per-volume query averaging, no winner-take-all -- so slot usage is
        driven by per-location anatomy and all slots receive gradient.
        """
        if not self.enabled:
            return tokens
        B, N, C = tokens.shape

        w = self.compute_assignments(tokens, level).to(tokens.dtype)  # [B,N,P]
        slots = F.normalize(self.get_prototypes(level), dim=-1)
        v = self.v_ups[level](slots)                                # [P, H*d]
        mem = torch.matmul(w, v)                                    # [B,N,H*d]
        mem = self.out_projs[level](mem)                            # [B,N,C]

        enhanced = self.memory_fusions[level](tokens, mem)
        return enhanced

    def forward(
        self,
        features_list: List[torch.Tensor],
        grid_shapes: List[Tuple[int, int, int]],
    ) -> List[torch.Tensor]:
        """Multi-scale feature enhancement (same contract as MLAMemoryBank)."""
        return [
            self.enhance_features(feat, level, grid_shapes[level])
            for level, feat in enumerate(features_list)
        ]

    # ------------------------------------------------------------------ verification

    @torch.no_grad()
    def anatomy_report(
        self,
        tokens: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int, int],
        labels: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Anatomical-validity report for one forward pass (JSON-serializable).

        Args:
            tokens:  [B, N, C] encoder tokens at this level
            level:   encoder level index
            grid_shape: (gd, gh, gw) token grid
            labels:  optional [B, D, H, W] GT labels -> class-purity metrics
            coords:  optional [B, N, 3] per-token normalized anatomical
                     coordinates (e.g. foreground-bbox frame, full-volume
                     inference) -> per-slot spatial centroid / spread

        Returns:
            dict with keys: level, num_slots, mass, utilization
            {entropy, effective, active_1pct}, per_block_mass,
            (optional) purity {per_slot, mean, mean_fg},
            (optional) spatial {centroid, spread, mean_spread}.
        """
        w = self.compute_assignments(tokens, level).float()         # [B,N,P]
        B, N, P = w.shape
        mass = w.mean(dim=(0, 1))                                   # [P]
        eps = 1e-12
        entropy = float(-(mass * (mass + eps).log()).sum())
        report = {
            'level': int(level),
            'num_slots': int(P),
            'mass': [float(x) for x in mass],
            'utilization': {
                'entropy': entropy,
                'effective': float(math.exp(entropy)),
                'active_1pct': int((mass > mass.max() * 0.01).sum()),
            },
            'per_block_mass': [
                float(mass[self.slot_block_ids == b].sum()) for b in range(self.num_blocks)],
        }

        if labels is not None:
            dom = self._labels_to_token_grid(labels, grid_shape)    # [B,N]
            onehot = F.one_hot(dom, self.num_classes).float()       # [B,N,C]
            # per-slot class distribution: M[p, c] = sum_i w_ip * 1[dom_i=c] / sum_i w_ip
            numer = torch.einsum('bnp,bnc->pc', w, onehot)          # [P,C]
            denom = numer.sum(dim=1, keepdim=True).clamp_min(eps)
            cls_dist = numer / denom
            purity = cls_dist.max(dim=1).values                     # [P]
            fg_mask = self.slot_block_ids > 0
            report['purity'] = {
                'per_slot': [float(x) for x in purity],
                'mean': float((purity * mass).sum() / mass.sum().clamp_min(eps)),
                'mean_fg': float((purity[fg_mask] * mass[fg_mask]).sum()
                                 / mass[fg_mask].sum().clamp_min(eps)),
            }

        if coords is not None:
            c = coords.float().view(B, N, 3)
            numer = torch.einsum('bnp,bnk->pk', w, c)               # [P,3]
            denom = w.sum(dim=(0, 1)).unsqueeze(1).clamp_min(eps)   # [P,1]
            centroid = numer / denom                                # [P,3]
            diff = c.unsqueeze(2) - centroid.view(1, 1, P, 3)       # [B,N,P,3]
            var = (w.unsqueeze(-1) * diff.pow(2)).sum(dim=(0, 1)) / denom
            spread = var.sum(dim=1).sqrt()                          # [P]
            report['spatial'] = {
                'centroid': centroid.tolist(),
                'spread': [float(x) for x in spread],
                'mean_spread': float((spread * mass).sum() / mass.sum().clamp_min(eps)),
            }

        return report


# ------------------------------------------------------------------ smoke test


def _self_test():
    torch.manual_seed(0)
    dims = [48, 96, 192]
    bank = AnatomySlotMemoryBank(
        feature_dims=dims, num_classes=6, kv_latent_dim=24,
        num_slots_per_class=8, num_heads=4, head_dim=16)
    grids = [(4, 6, 6), (2, 3, 3), (1, 2, 2)]
    feats = [torch.randn(2, g[0] * g[1] * g[2], d) for g, d in zip(grids, dims)]
    labels = torch.randint(0, 6, (2, 32, 48, 48))

    loss = bank.compute_prototype_loss(feats, labels, grids)
    loss.backward()
    g = bank.get_slots(0).grad
    print(f'loss={float(loss.detach()):.4f}  slot grad norm={float(g.norm()):.4f}')
    assert torch.isfinite(loss) and g is not None and torch.isfinite(g.norm())

    # enhancement path (disabled -> bypass; enabled -> same shape)
    out0 = bank.enhance_features(feats[0], 0, grids[0])
    assert out0.shape == feats[0].shape and out0 is feats[0]
    bank.enabled.fill_(True)
    out1 = bank.enhance_features(feats[0], 0, grids[0])
    assert out1.shape == feats[0].shape and not torch.equal(out1, feats[0])

    # sinkhorn balance: column masses approximately equal (CV < 5%)
    # (bounded logits, matching the cosine-similarity scores used in the model)
    T = AnatomySlotMemoryBank._sinkhorn(torch.tanh(torch.randn(64, 8)), eps=0.1, iters=10)
    col = T.sum(0)
    cv = float(col.std() / col.mean())
    assert cv < 0.05, f'sinkhorn column-mass CV={cv:.3f}: {col}'

    # report API
    rep = bank.anatomy_report(feats[0], 0, grids[0], labels=labels,
                              coords=torch.rand(2, grids[0][0] * grids[0][1] * grids[0][2], 3))
    print('utilization:', rep['utilization'])
    print('mean purity: %.3f  mean_fg: %.3f  mean_spread: %.3f'
          % (rep['purity']['mean'], rep['purity']['mean_fg'], rep['spatial']['mean_spread']))
    print('AnatomySlotMemoryBank smoke test OK')


__all__ = [
    "AnatomySlotMemoryBank", "_MLACrossFusion",
    "math", "Dict", "List", "Optional", "Tuple", "torch", "nn", "F",
]


if __name__ == "__main__":
    _self_test()
