"""
DynUNet-MLA backbone (nnunet_mla): nnU-Net topology + local/global MLA injection
+ anatomy-prototype memory-bank hooks.

Rationale: the ViT/nnFormer-style token backbones underperform a plain DynUNet
on this 154-case task (fixed nnU-Net reaches ~0.62 sliding-window Dice). This
module keeps the MONAI DynUNet EXACTLY as configured for our nnU-Net baseline
(same filters/strides/res_block, so the backbone comparison in Table 1 is a
true backbone reference) and injects the framework's attention design into it:

* local-window MLA at mid/high-resolution encoder stages (fine boundary detail,
  linear cost in tokens),
* global MLA at the deep/bottleneck stages (long-range skull-base context,
  trivial cost at these token counts),
* anatomy-prototype memory bank (anatomical_v4) at three encoder levels, with
  the same enhance/loss contract as the ViT backbone.

Contracts mirror NNFormerUNet3D so the two-stage cascade, the hierarchical
soft-tree heads and sliding-window inference work unchanged:
  forward(x, labels=None)          -> [B, num_classes, D, H, W] logits
  forward_features(x, labels=None) -> (full_res_features [B,32,D,H,W], None, target_shape)
  provides_fullres_features = True  (decoder features are already full-resolution)

Note on patch size: MONAI DynUNet requires every spatial dim divisible by 32
(5 stride-2 stages). Inputs are zero-padded to the next multiple of 32 and the
outputs cropped back, so the trainer's 40x192x192 patches work unchanged.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet

from npc_vit_mla import MultiHeadLatentAttention, StandardMHAWrapper

# DynUNet topology shared with the npc_sota_nnunet baseline.
FILTERS = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

# Encoder level indices (0 = input_block output ... 5 = bottleneck output):
#   L0 32ch  (D,  H,  W)     L1 64ch  (D/2,  H/2,  W/2)
#   L2 128ch (D/4,  H/4,  W/4)   L3 256ch (D/8,  H/8,  W/8)
#   L4 320ch (D/16, H/16, W/16)  L5 320ch (D/32, H/32, W/32)
LOCAL_MLA_LEVELS = [2, 3]       # local-window MLA (fine detail, linear cost)
GLOBAL_MLA_LEVELS = [4, 5]      # global MLA (long-range context, cheap)
MEMORY_LEVELS = {2: 0, 3: 1, 4: 2}   # encoder level -> memory-bank level
MEMORY_FEATURE_DIMS = [FILTERS[2], FILTERS[3], FILTERS[4]]   # [128, 256, 320]


def windowed_attention(attn: nn.Module, h: torch.Tensor,
                       grid_shape: Tuple[int, int, int],
                       window_size: Tuple[int, int, int]) -> torch.Tensor:
    """Windowed attention on token grid (same partition logic as DualPathMLABlock).

    h: [B, N, C] tokens over grid_shape; attends within each window only.
    """
    B, N, C = h.shape
    D, H, W = grid_shape
    wd = max(1, min(window_size[0], D))
    wh = max(1, min(window_size[1], H))
    ww = max(1, min(window_size[2], W))

    x = h.view(B, D, H, W, C)
    pad_d = (wd - D % wd) % wd
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
    nd, nh, nw = Dp // wd, Hp // wh, Wp // ww

    x = x.view(B, nd, wd, nh, wh, nw, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(B * nd * nh * nw, wd * wh * ww, C)

    windows = attn(windows)

    x = windows.view(B, nd, nh, nw, wd, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, Dp, Hp, Wp, C)
    if pad_d or pad_h or pad_w:
        x = x[:, :D, :H, :W, :].contiguous()
    return x.view(B, N, C)


class MLAInject(nn.Module):
    """Residual attention injection on a 5D feature map.

    window_size=None -> global attention; otherwise windowed attention.
    """

    def __init__(self, dim: int, num_heads: int = 6, head_dim: int = 32,
                 kv_latent_dim: int = 96, window_size: Optional[Tuple[int, int, int]] = None,
                 use_mla: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        Attn = MultiHeadLatentAttention if use_mla else StandardMHAWrapper
        self.attn = Attn(dim=dim, num_heads=num_heads, head_dim=head_dim,
                         kv_latent_dim=kv_latent_dim, dropout=dropout)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B,C,D,H,W]
        B, C, D, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                  # [B,N,C]
        h = self.norm(t)
        if self.window_size is None:
            h = self.attn(h)
        else:
            h = windowed_attention(self.attn, h, (D, H, W), self.window_size)
        t = t + h
        return t.transpose(1, 2).view(B, C, D, H, W)


class DynUNetMLA3D(nn.Module):
    """nnU-Net (DynUNet) backbone + MLA/memory injection, NNFormerUNet3D contract."""

    provides_fullres_features = True   # forward_features yields full-res decoder feats

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        filters: Optional[List[int]] = None,
        num_heads: int = 6,
        head_dim: int = 32,
        kv_latent_dim: int = 96,
        window_size: Tuple[int, int, int] = (2, 4, 4),
        use_mla: bool = True,
        dropout: float = 0.0,
        memory_bank: Optional[nn.Module] = None,
        local_mla_levels: Optional[List[int]] = None,
        global_mla_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.memory_bank = memory_bank
        filters = filters or FILTERS
        local_mla_levels = LOCAL_MLA_LEVELS if local_mla_levels is None else local_mla_levels
        global_mla_levels = GLOBAL_MLA_LEVELS if global_mla_levels is None else global_mla_levels

        self.dynunet = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=[[3, 3, 3]] * len(filters),
            strides=STRIDES[:len(filters)],
            upsample_kernel_size=[[2, 2, 2]] * (len(filters) - 1),
            filters=filters,
            res_block=True,
        )

        # --- MLA injection modules, keyed by encoder level ---
        self.inject = nn.ModuleDict()
        for lv in local_mla_levels:
            self.inject[str(lv)] = MLAInject(
                filters[lv], num_heads, head_dim, kv_latent_dim,
                window_size=window_size, use_mla=use_mla, dropout=dropout)
        for lv in global_mla_levels:
            self.inject[str(lv)] = MLAInject(
                filters[lv], num_heads, head_dim, kv_latent_dim,
                window_size=None, use_mla=use_mla, dropout=dropout)

        self._mem_feats = {}          # level -> (tokens [B,N,C], grid) stashed per forward
        self._dec_feats = None        # full-res decoder features stashed per forward
        self._mem_tok0 = None         # memory-path tokens at memory level 0 (pre-gate)
        # Evidence gate set by the caller before each forward:
        #   None   -> plain memory enhancement (f = f_mem)
        #   float  -> constant fusion f = a*f_loc + (1-a)*f_mem   (A_gate arm)
        #   tensor -> [B,1,D,H,W] evidence map g, f = g*f_loc + (1-g)*f_mem
        self.evidence_gate = None
        self._gate_src = None
        self._register_hooks()

    # ---- hook machinery -------------------------------------------------
    def _encoder_modules(self) -> List[nn.Module]:
        """Encoder level -> module: L0 input_block, L1..L5 downsamples+bottleneck."""
        d = self.dynunet
        return [d.input_block] + list(d.downsamples) + [d.bottleneck]

    def _register_hooks(self):
        enc = self._encoder_modules()

        def make_hook(lv):
            def hook(_module, _inp, out):
                key = str(lv)
                if key in self.inject:
                    out = self.inject[key](out)
                if self.memory_bank is not None and lv in MEMORY_LEVELS:
                    B, C, D, H, W = out.shape
                    t_loc = out.flatten(2).transpose(1, 2)
                    coords = self._level_coords((D, H, W), out.device)
                    t_mem = self._enhance_with_memory(t_loc, MEMORY_LEVELS[lv], (D, H, W), coords)
                    t = self._gate_fuse(t_loc, t_mem, (D, H, W))
                    if MEMORY_LEVELS[lv] == 0:
                        self._mem_tok0 = (t_mem, (D, H, W))
                    self._mem_feats[MEMORY_LEVELS[lv]] = (t, (D, H, W))
                    out = t.transpose(1, 2).reshape(B, C, D, H, W)
                return out
            return hook

        for lv, mod in enumerate(enc):
            if str(lv) in self.inject or (self.memory_bank is not None and lv in MEMORY_LEVELS):
                mod.register_forward_hook(make_hook(lv))

        # Stash full-res decoder features (last upsample output, 32 channels).
        self.dynunet.upsamples[-1].register_forward_hook(
            lambda _m, _i, out: self._stash_dec(out))

    def _stash_dec(self, out):
        self._dec_feats = out
        return out

    def _gate_fuse(self, t_loc, t_mem, grid):
        """Evidence-gated fusion of local and memory-retrieved tokens."""
        gate = self._gate_src
        if gate is None:
            return t_mem
        if isinstance(gate, float):
            return gate * t_loc + (1.0 - gate) * t_mem
        g = F.adaptive_avg_pool3d(gate.float(), grid)          # [B,1,gd,gh,gw]
        g = g.flatten(2).transpose(1, 2).to(t_loc.dtype)        # [B,N,1]
        return g * t_loc + (1.0 - g) * t_mem

    # ---- spatial coords for SPM (V5) memory bank --------------------------
    # ---- semantic/spatial conditioning for V6/V5 memory banks --------------
    def _enhance_with_memory(self, tokens, mem_level, grid, coords):
        """Dispatch to enhance_features with the richest supported condition.

        Priority: V6 semantic (Stage-1 prior pooled to the token grid) >
        V5 spatial coords > plain V4.
        """
        mb = self.memory_bank
        if hasattr(mb, 'get_sem_anchors'):
            sem = self._sem_tokens(grid, tokens.device)
            if sem is not None:
                return mb.enhance_features(tokens, mem_level, grid, sem)
        if coords is not None and hasattr(mb, 'set_alpha'):
            return mb.enhance_features(tokens, mem_level, grid, coords)
        return mb.enhance_features(tokens, mem_level, grid)

    def _sem_tokens(self, grid, device):
        """Stage-1 prior channels pooled to the token grid: [B, N, 2].

        The prior travels with the model input (crops, flips and sliding
        windows included), so no extra pipeline and no train/test mismatch.
        """
        src = getattr(self, '_sem_source', None)      # [B,2,Dp,Hp,Wp] padded
        if src is None:
            return None
        pooled = F.adaptive_avg_pool3d(src, grid)     # [B,2,gd,gh,gw]
        return pooled.flatten(2).transpose(1, 2)      # [B,N,2]

    def _level_coords(self, grid, device):
        """Patient-volume-normalized [0,1]^3 coords for every token of ``grid``.

        Uses ``self._coord_info`` stashed by forward: (origin, vshape, flips)
        as [B,3] tensors plus the unpadded/padded patch sizes. Flip-consistent:
        a token whose content was mirrored by augmentation gets the mirrored
        anatomical coordinate. Returns [B, N, 3] or None when coords unknown.
        """
        info = getattr(self, '_coord_info', None)
        if info is None:
            return None
        origin, vshape, flips = info                     # [B,3] float tensors
        dorig = self._patch_orig                         # unpadded patch (D,H,W)
        ppad = self._patch_padded                        # padded patch (D',H',W')
        axes = []
        for a, g in enumerate(grid):
            pix = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) \
                * (ppad[a] / g)
            axes.append(pix.clamp(0, dorig[a] - 1))
        gz, gy, gx = torch.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        q = torch.stack([gz, gy, gx], dim=-1).view(-1, 3)          # [N,3]
        B = origin.shape[0]
        dorig_t = torch.tensor(dorig, device=device, dtype=torch.float32)
        qb = q.unsqueeze(0).expand(B, -1, -1)
        adj = torch.where(flips.unsqueeze(1) > 0.5,
                          (dorig_t - 1 - q).unsqueeze(0).expand(B, -1, -1), qb)
        coords = (origin.float().unsqueeze(1) + adj) / \
                 (vshape.float().unsqueeze(1) - 1).clamp(min=1)
        return coords.clamp(0, 1)                                   # [B,N,3]


    # ---- padding to multiples of 32 (DynUNet requirement) ---------------
    @staticmethod
    def _pad32(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        D, H, W = x.shape[2:]
        pd = (32 - D % 32) % 32
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (D, H, W)

    @staticmethod
    def _crop(x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        D, H, W = shape
        return x[:, :, :D, :H, :W]

    # ---- memory loss -----------------------------------------------------
    def _maybe_proto_loss(self, labels: Optional[torch.Tensor]):
        if self.memory_bank is None or not self.training or labels is None:
            self._proto_loss = None
            return
        levels = sorted(self._mem_feats)
        feats = [self._mem_feats[l][0] for l in levels]
        grids = [self._mem_feats[l][1] for l in levels]
        self._proto_loss = self.memory_bank.compute_prototype_loss(
            feats, labels, grid_shapes=grids)

    # ---- public contract --------------------------------------------------
    def _stash_coord_info(self, x, xp, coord_info):
        """Stash patch geometry for _level_coords (SPM/V5 memory bank)."""
        self._patch_orig = tuple(x.shape[2:])
        self._patch_padded = tuple(xp.shape[2:])
        self._coord_info = coord_info   # (origin, vshape, flips) [B,3] or None
        # SCM/V6: Stage-1 prior channels (padded) for per-level pooling.
        self._sem_source = xp[:, 1:3] if xp.shape[1] >= 3 else None
        gate = self.evidence_gate
        if torch.is_tensor(gate):
            gate, _ = self._pad32(gate)
        elif gate is not None:
            gate = float(gate)
        self._gate_src = gate

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None,
                coord_info=None) -> torch.Tensor:
        self._mem_feats, self._dec_feats, self._mem_tok0 = {}, None, None
        xp, orig = self._pad32(x)
        self._stash_coord_info(x, xp, coord_info)
        logits = self.dynunet(xp)
        self._maybe_proto_loss(labels)
        return self._crop(logits, orig)

    def forward_features(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None,
                         coord_info=None):
        """Return full-res decoder features for the hierarchical soft-tree heads."""
        self._mem_feats, self._dec_feats, self._mem_tok0 = {}, None, None
        xp, orig = self._pad32(x)
        self._stash_coord_info(x, xp, coord_info)
        _ = self.dynunet(xp)
        self._maybe_proto_loss(labels)
        feats = self._crop(self._dec_feats, orig)   # [B, 32, D, H, W]
        return feats, None, orig


# ==================== Self-test ====================

if __name__ == "__main__":
    torch.manual_seed(0)
    model = DynUNetMLA3D(in_channels=3, num_classes=6)
    n = sum(p.numel() for p in model.parameters())
    n_base = sum(p.numel() for p in model.dynunet.parameters())
    print(f"params: total={n/1e6:.2f}M  dynunet={n_base/1e6:.2f}M  inject+={(n-n_base)/1e6:.2f}M")

    model.eval()
    x = torch.randn(1, 3, 40, 192, 192)
    with torch.no_grad():
        y = model(x)
    print(f"forward: in={tuple(x.shape)} out={tuple(y.shape)}")
    assert y.shape == (1, 6, 40, 192, 192), y.shape
    assert torch.isfinite(y).all()

    with torch.no_grad():
        feats, skips, ts = model.forward_features(x)
    print(f"forward_features: feats={tuple(feats.shape)} skips={skips} target={ts}")
    assert feats.shape == (1, 32, 40, 192, 192), feats.shape
    assert torch.isfinite(feats).all()
    print("DynUNetMLA3D smoke test OK")
