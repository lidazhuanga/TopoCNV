"""
Coordinate-free Anatomy Slot Memory Bank v4 (坐标-free 解剖槽位记忆库).

Drop-in subclass of ``AnatomySlotMemoryBank`` that makes the anatomical
"position sense" EMERGE from content clustering instead of relying on any
explicit 3D coordinate input.  Same public interface, same slot layout
((C_fg + 1) blocks x K slots per level), so it plugs into ViTUNet3D and the
trainer with no other changes.

Why v4 (root cause of the v3 collapse)
--------------------------------------
The v3 bank shaped slots with a WITHIN-CLASS Sinkhorn OT loss, but its
read-out (``compute_assignments``) used a GLOBAL softmax over ALL P slots.
The two objectives were misaligned: the balanced within-class structure the
loss built was never the thing the read-out actually used, so read-out
degenerated to winner-take-all and utilization collapsed.

v4 fix -- factorized assignment shared by read-out AND shaping
--------------------------------------------------------------
    A(token -> slot k in block c) = P(class=c | token) * P(slot=k | class=c, token)
      - P(class=c | token) : block posterior, logsumexp-pooled over the K slots
                             of each block (soft-max pool, NOT hard max).
      - P(slot=k | class=c, token) : within-block softmax over the K slots.

The within-class Sinkhorn balance now acts on exactly the same
P(slot | class, token) distribution that drives the read-out, so balanced
utilization is enforced BY CONSTRUCTION on the object the network really uses.
No coordinates enter training; position consistency is only *measured*
post-hoc (see ``anatomy_report`` / MIA/scripts/verify_anatomy_slots.py).

Extra machinery
---------------
  - per-level EMA usage buffers + periodic dead-slot revival (VQ-VAE style)
    as a safety net.  The factorized within-block softmax already gives every
    slot gradient every step, so revival is secondary.
  - all shaping / assignment math runs in float32 for stability under bf16
    autocast (inherited convention: ``_block_logits`` returns float32).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from npc_anatomy_memory import AnatomySlotMemoryBank


class AnatomySlotMemoryBankV4(AnatomySlotMemoryBank):
    """Coordinate-free anatomy-slot bank. Position emerges from content clustering.

    Key change vs v3: read-out and shaping share ONE factorized assignment
    A[token, c, k] = P(class=c|token) * P(slot=k|class=c,token), so the
    within-class Sinkhorn balance actually drives the read-out (fixes the
    global-softmax winner-take-all collapse).

    Extra args (all defaulted, so ``_build_memory_bank`` keeps calling with the
    same v3 keyword arguments):
        usage_momentum: EMA momentum for per-slot usage tracking.
        revive_every:   revive dead slots every N ``compute_prototype_loss`` calls.
        revive_thresh:  a slot is "dead" when its EMA usage falls below this.
    """

    def __init__(self, *args, usage_momentum: float = 0.99,
                 revive_every: int = 200, revive_thresh: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_momentum = usage_momentum
        self.revive_every = revive_every
        self.revive_thresh = revive_thresh
        # per-level EMA usage mass over flat slots [num_prototypes]
        for level in range(self.num_levels):
            self.register_buffer(
                f'usage_level{level}',
                torch.ones(self.num_prototypes) / self.num_prototypes)
        self.register_buffer('_revive_step', torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ read-out

    def compute_assignments(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        """Factorized per-token soft assignment: [B, N, num_prototypes].

        A[token, block c, slot k] = P(class=c | token) * P(slot=k | class=c, token)
        Read-out, shaping losses and the verification API all consume THIS
        object, so the within-class Sinkhorn balance drives the actual read-out.
        """
        latents = self._token_latents(tokens, level)            # [B,N,kv]
        sim = self._block_logits(latents, level)                # [B,N,nb,K] f32
        tau = self.temperature
        # P(class=c | token): block posterior via logsumexp over K (soft-max pool)
        block_logit = torch.logsumexp(sim / tau, dim=3)         # [B,N,nb]
        p_class = F.softmax(block_logit, dim=-1)                # [B,N,nb]
        # P(slot=k | class=c, token): within-block softmax over K
        p_slot = F.softmax(sim / tau, dim=3)                    # [B,N,nb,K]
        A = p_class.unsqueeze(-1) * p_slot                      # [B,N,nb,K]
        return A.reshape(A.shape[0], A.shape[1], self.num_prototypes)  # [B,N,P]

    # ------------------------------------------------------------------ dead-slot revival

    @torch.no_grad()
    def _revive_dead_slots(self, level: int) -> int:
        """Re-init zero-mass slots from a high-usage sibling in the SAME block.

        In-place edit of ``slots_levelX.data`` only (no graph); accepts minor
        Adam-momentum staleness (small perturbation, Adam re-adapts fast).
        Returns the number of slots revived (for tests / logging).
        """
        usage = getattr(self, f'usage_level{level}')            # [P]
        slots = self.get_slots(level)                           # nn.Parameter [nb,K,kv]
        nb, K, kv = slots.shape
        usage_bk = usage.view(nb, K)
        revived = 0
        mean_usage = usage.mean()
        for b in range(nb):
            dead = torch.nonzero(usage_bk[b] < self.revive_thresh, as_tuple=False).flatten()
            if dead.numel() == 0 or dead.numel() == K:
                continue  # nothing dead, or whole block dead (nothing to clone from)
            donor = int(usage_bk[b].argmax())
            for k in dead.tolist():
                new_slot = slots.data[b, donor] + 0.1 * torch.randn_like(slots.data[b, donor])
                slots.data[b, k] = F.normalize(new_slot, dim=-1)
                usage_bk[b, k] = mean_usage
                revived += 1
        return revived

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
        """Anatomy-slot shaping loss (coordinate-free), averaged over levels.

        Terms:
          L_cls    : block CE using LOGSUMEXP-pooled block logits (soft-max pool,
                     reinforces utilization vs. v3's hard max).
          L_ot     : per-class Sinkhorn OT on the SAME within-block softmax that
                     drives the read-out (detached target -> stop-gradient).
          L_smooth : same-class neighbor consensus (augmentation-immune).
          L_div    : slot diversity hinge.
        All computed in float32 for stability under bf16 autocast.
        Also updates per-level EMA usage buffers and periodically revives dead
        slots (both under no_grad, no effect on the returned loss graph).
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

            # ---- 1. block classification CE (class-balanced), logsumexp-pooled ----
            block_logits = torch.logsumexp(sim / tau, dim=3)        # [B,N,nb]
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

            # within-block log-softmax over slots -- the SAME distribution the
            # factorized read-out uses (P(slot|class,token)); reused by OT/smooth.
            block_logp = F.log_softmax(sim / tau, dim=3)            # [B,N,nb,K]

            # ---- 2. per-class Sinkhorn OT assignment (within-class balance) ----
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

            # ---- EMA usage update + periodic dead-slot revival (no grad) ----
            with torch.no_grad():
                A = self.compute_assignments(features, level)       # [B,N,P]
                mass = A.mean(dim=(0, 1)).float()                   # [P]
                usage = getattr(self, f'usage_level{level}')
                usage.mul_(self.usage_momentum).add_(
                    mass, alpha=1 - self.usage_momentum)

        # revive once per call (after all levels' usage refreshed)
        if self.revive_every > 0:
            with torch.no_grad():
                self._revive_step += 1
                if int(self._revive_step) % self.revive_every == 0:
                    for level in range(self.num_levels):
                        self._revive_dead_slots(level)

        n_levels = len(features_list)
        ot_loss = ot_loss / max(n_ot, 1)
        smooth_loss = smooth_loss / max(n_smooth, 1)

        return (cls_weight * cls_loss / n_levels
                + ot_weight * ot_loss
                + smooth_weight * smooth_loss
                + div_weight * div_loss / n_levels)


# ------------------------------------------------------------------ smoke test

if __name__ == '__main__':
    torch.manual_seed(0)
    dims = [48, 96, 192]
    bank = AnatomySlotMemoryBankV4(
        feature_dims=dims, num_classes=6, kv_latent_dim=24,
        num_slots_per_class=8, num_heads=4, head_dim=16,
        revive_every=2)  # small so the revival path runs in the smoke test
    grids = [(4, 6, 6), (2, 3, 3), (1, 2, 2)]
    feats = [torch.randn(2, g[0] * g[1] * g[2], d) for g, d in zip(grids, dims)]
    labels = torch.randint(0, 6, (2, 32, 48, 48))

    # factorized assignment: rows sum to 1, shape [B,N,P]
    A = bank.compute_assignments(feats[0], 0)
    assert A.shape == (2, grids[0][0] * grids[0][1] * grids[0][2], bank.num_prototypes)
    assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-4), \
        'factorized assignment rows must sum to 1'

    loss = bank.compute_prototype_loss(feats, labels, grids)
    loss.backward()
    g = bank.get_slots(0).grad
    print(f'loss={float(loss.detach()):.4f}  slot grad norm={float(g.norm()):.4f}')
    assert torch.isfinite(loss) and g is not None and torch.isfinite(g.norm())

    # usage buffers updated (moved away from uniform init)
    u0 = bank.usage_level0
    assert torch.isfinite(u0).all() and float(u0.sum()) > 0

    # second call triggers revival (revive_every=2); must stay finite
    loss2 = bank.compute_prototype_loss(feats, labels, grids)
    assert torch.isfinite(loss2)
    assert torch.isfinite(bank.get_slots(0)).all(), 'revival produced non-finite slots'

    # enhancement path (disabled -> bypass; enabled -> same shape)
    out0 = bank.enhance_features(feats[0], 0, grids[0])
    assert out0.shape == feats[0].shape and out0 is feats[0]
    bank.enabled.fill_(True)
    out1 = bank.enhance_features(feats[0], 0, grids[0])
    assert out1.shape == feats[0].shape and not torch.equal(out1, feats[0])

    rep = bank.anatomy_report(feats[0], 0, grids[0], labels=labels,
                              coords=torch.rand(2, grids[0][0] * grids[0][1] * grids[0][2], 3))
    print('utilization:', rep['utilization'])
    print('mean purity: %.3f  mean_fg: %.3f  mean_spread: %.3f'
          % (rep['purity']['mean'], rep['purity']['mean_fg'], rep['spatial']['mean_spread']))
    print('AnatomySlotMemoryBankV4 smoke test OK')
