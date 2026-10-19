"""
v5 backbone: nnFormer-style interleaved-conv + local/global dual-path MLA.

This module provides a drop-in replacement for `ViTUNet3D` (npc_vit_mla.py)
that injects the two core ideas of nnFormer into our hierarchical token
pyramid:

1. Interleaved convolution: a depthwise Conv3d branch inside every transformer
   block, restoring local inductive bias (nnFormer's Conv+Transformer stage).
2. Local-window + global dual-path attention: each block runs a windowed MLA
   (fine detail / boundaries, nnFormer's LV-MSA) in parallel with a global MLA
   (long-range anatomical context, nnFormer's GV-MSA), then fuses the two.
   Option A (default): the dual path is only enabled at shallow / high-resolution
   stages (`dual_path_stages=[0]`); deeper stages fall back to a single global
   MLA, where the clamped local window already spans (approximately) the whole
   grid so the two paths are redundant. This trims the second MLA + fusion layer
   from the expensive deep stages (~12M -> ~1M extra), keeping the high-res
   long-range benefit at near-baseline parameter count.

The token pyramid (ConvStem -> 2x PatchMerging), feature dims
[embed_dim, embed_dim*2, embed_dim*4], grid resolutions, memory-bank hook,
ViT decoder + CNN decoder, and forward / forward_features contracts are kept
IDENTICAL to ViTUNet3D so that the two-stage cascade, the v4 coordinate-free
memory bank, and sliding-window inference all work unchanged.

Components:
- DualPathMLABlock: interleaved conv + (local-window MLA || global MLA) + FFN
- NNFormerUNet3D:    drop-in ViTUNet3D backbone using DualPathMLABlock
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse all tested building blocks from the ViT-MLA backbone.
from npc_vit_mla import (
    ConvStem3D,
    PatchMerging3D,
    PatchExpanding3D,
    CNNDecoder3D,
    MultiHeadLatentAttention,
    StandardMHAWrapper,
    DropPath,
)


# ==================== Dual-Path (local-window + global) MLA Block ====================

class DualPathMLABlock(nn.Module):
    """nnFormer-style block: interleaved conv + dual-path (local/global) MLA + FFN.

    forward(x, grid_shape):
        x = x + conv(x)                                   # interleaved conv branch
        h = norm1(x)
        g = global_attn(h)                                # full-sequence MLA
        l = local_windowed_attn(h, grid_shape)            # windowed MLA (own weights)
        x = x + drop_path(fuse([g, l]))                   # concat + linear fusion
        x = x + drop_path(ffn(norm2(x)))

    The local path partitions the token grid into windows (clamped per-dim to
    the grid so small deep-stage grids degrade gracefully to a single/global
    window) and runs MLA within each window.

    use_mla=False falls back to StandardMHAWrapper on both paths for a fair
    MLA-vs-MHA ablation, mirroring TransformerEncoderBlock.

    residual_conv=True upgrades the interleaved conv branch from a single
    depthwise Conv3d to a bottleneck residual conv block
    (1x1 reduce -> depthwise 3x3 -> 1x1 expand), restoring channel mixing in the
    local branch at a fraction of a full 3x3 conv's parameters. Intended for the
    shallow, high-resolution stages where local detail matters most.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        head_dim: int = 32,
        kv_latent_dim: int = 96,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        window_size: Tuple[int, int, int] = (2, 4, 4),
        use_mla: bool = True,
        dual_path: bool = True,
        residual_conv: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.dual_path = dual_path

        # --- Interleaved convolution branch (nnFormer style) ---
        if residual_conv:
            # Bottleneck residual block: 1x1 reduce -> depthwise 3x3 -> 1x1 expand.
            # ~dim^2/2 params vs dim*27 for the plain depthwise branch, but with
            # channel mixing; the residual add is the `x = x + conv(x)` in forward.
            r = max(16, dim // 4)
            self.conv = nn.Sequential(
                nn.Conv3d(dim, r, kernel_size=1, bias=False),
                nn.InstanceNorm3d(r, affine=True),
                nn.GELU(),
                nn.Conv3d(r, r, kernel_size=3, padding=1, groups=r, bias=False),
                nn.InstanceNorm3d(r, affine=True),
                nn.GELU(),
                nn.Conv3d(r, dim, kernel_size=1, bias=False),
                nn.InstanceNorm3d(dim, affine=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.InstanceNorm3d(dim, affine=True),
                nn.GELU(),
            )

        # --- Attention: global path always; local path + fusion only when
        #     dual_path is enabled (Option A: shallow/high-res stages only). ---
        Attn = MultiHeadLatentAttention if use_mla else StandardMHAWrapper
        self.norm1 = nn.LayerNorm(dim)
        self.global_attn = Attn(
            dim=dim, num_heads=num_heads, head_dim=head_dim,
            kv_latent_dim=kv_latent_dim, dropout=dropout,
        )
        if dual_path:
            self.local_attn = Attn(
                dim=dim, num_heads=num_heads, head_dim=head_dim,
                kv_latent_dim=kv_latent_dim, dropout=dropout,
            )
            # Fuse the two paths: concat -> linear projection back to dim.
            self.fuse = nn.Linear(2 * dim, dim)
        else:
            self.local_attn = None
            self.fuse = None

        # --- FFN (same structure as TransformerEncoderBlock) ---
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def _clamp_window(self, grid_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        D, H, W = grid_shape
        wd, wh, ww = self.window_size
        return max(1, min(wd, D)), max(1, min(wh, H)), max(1, min(ww, W))

    def _local_attn(self, h: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Windowed MLA: partition token grid into windows, attend within each."""
        B, N, C = h.shape
        D, H, W = grid_shape
        wd, wh, ww = self._clamp_window(grid_shape)

        # tokens -> spatial grid
        x = h.view(B, D, H, W, C)

        # pad to window multiples
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
        nd, nh, nw = Dp // wd, Hp // wh, Wp // ww

        # partition: [B, nd, wd, nh, wh, nw, ww, C] -> [B*nWin, wd*wh*ww, C]
        x = x.view(B, nd, wd, nh, wh, nw, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = x.view(B * nd * nh * nw, wd * wh * ww, C)

        # attention within windows
        windows = self.local_attn(windows)

        # reverse partition -> [B, Dp, Hp, Wp, C]
        x = windows.view(B, nd, nh, nw, wd, wh, ww, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, Dp, Hp, Wp, C)

        # crop padding
        if pad_d or pad_h or pad_w:
            x = x[:, :D, :H, :W, :].contiguous()

        return x.view(B, N, C)

    def forward(self, x: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        D, H, W = grid_shape

        # --- Interleaved conv branch ---
        xc = x.transpose(1, 2).view(B, C, D, H, W)
        xc = self.conv(xc).flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + xc

        # --- Attention: global always, fused local-window path when dual_path ---
        h = self.norm1(x)
        g = self.global_attn(h)                  # global MLA
        if self.dual_path:
            l = self._local_attn(h, grid_shape)  # local-window MLA
            attn = self.fuse(torch.cat([g, l], dim=-1))
        else:
            attn = g
        x = x + self.drop_path(attn)

        # --- FFN ---
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ==================== nnFormer-MLA UNETR 3D (drop-in ViTUNet3D) ====================

class NNFormerUNet3D(nn.Module):
    """Hybrid UNETR with nnFormer-style dual-path MLA blocks.

    Drop-in replacement for ViTUNet3D: identical __init__ keywords,
    forward / forward_features contracts, 3-level token pyramid, and memory
    bank hook. Only the transformer blocks differ (DualPathMLABlock instead of
    TransformerEncoderBlock), and stages are stored as ModuleList so the
    spatial grid can be passed explicitly to each block (required by the
    local-window attention path).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        embed_dim: int = 128,
        stem_channels: Tuple[int, int, int] = (32, 64, 128),
        tokenize_kernel: Tuple[int, int, int] = (2, 4, 4),
        num_heads: int = 4,
        head_dim: int = 32,
        kv_latent_dim: int = 64,
        encoder_depths: List[int] = [2, 4, 2],
        decoder_depths: List[int] = [2, 2],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        memory_bank: Optional[nn.Module] = None,
        use_mla: bool = True,
        window_size: Tuple[int, int, int] = (2, 4, 4),
        dual_path_stages: Optional[List[int]] = None,
        residual_conv_stages: Optional[List[int]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.memory_bank = memory_bank
        self.stem_channels = stem_channels
        self.tokenize_kernel = tokenize_kernel
        self.use_mla = use_mla

        # Stage dimensions (doubling at each merge) -- matches VIT_FEATURE_DIMS
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]

        # Option A: enable the local+global dual path only at the listed
        # (shallow, high-resolution) encoder-stage dims; every other stage uses
        # a single global MLA. Decoder stages reuse dims[0]/dims[1], so a decoder
        # stage sitting at a shallow dim inherits the dual path automatically.
        if dual_path_stages is None:
            dual_path_stages = [0]
        self.dual_path_stages = list(dual_path_stages)
        self._dual_path_dims = {dims[i] for i in self.dual_path_stages}

        # Residual conv branch: bottleneck residual block instead of a single
        # depthwise conv, enabled only at the listed (shallow) encoder-stage dims
        # to keep the parameter cost negligible. Decoder stages reuse dims[0]/
        # dims[1] and inherit the setting automatically, as with the dual path.
        self.residual_conv_stages = list(residual_conv_stages or [])
        self._residual_conv_dims = {dims[i] for i in self.residual_conv_stages}

        # --- Convolutional Stem (same as ViTUNet3D) ---
        self.conv_stem = ConvStem3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            stem_channels=stem_channels,
            tokenize_kernel=tokenize_kernel,
        )

        # --- Per-block drop path rates (linearly increasing) ---
        total_blocks = sum(encoder_depths) + sum(decoder_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        def _make_stage(dim: int, depth: int, start_idx: int) -> nn.ModuleList:
            dual_path = dim in self._dual_path_dims
            residual_conv = dim in self._residual_conv_dims
            return nn.ModuleList([
                DualPathMLABlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_latent_dim=kv_latent_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path_rate=dpr[start_idx + j],
                    window_size=window_size,
                    use_mla=use_mla,
                    dual_path=dual_path,
                    residual_conv=residual_conv,
                ) for j in range(depth)
            ])

        # --- Encoder stages ---
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i in range(3):
            self.encoder_stages.append(_make_stage(dims[i], encoder_depths[i], block_idx))
            block_idx += encoder_depths[i]

        self.patch_merges = nn.ModuleList([
            PatchMerging3D(dims[0], dims[1]),
            PatchMerging3D(dims[1], dims[2]),
        ])

        # --- ViT Decoder stages (token-level) ---
        self.patch_expands = nn.ModuleList([
            PatchExpanding3D(dims[2], dims[1]),
            PatchExpanding3D(dims[1], dims[0]),
        ])

        self.decoder_stages = nn.ModuleList()
        for i in range(2):
            self.decoder_stages.append(_make_stage(dims[1 - i], decoder_depths[i], block_idx))
            block_idx += decoder_depths[i]

        # --- ViT output norm ---
        self.final_norm = nn.LayerNorm(dims[0])

        # --- CNN Decoder (progressive upsample with stem skips) ---
        self.cnn_decoder = CNNDecoder3D(
            embed_dim=embed_dim,
            stem_channels=stem_channels,
            tokenize_kernel=tokenize_kernel,
            num_classes=num_classes,
        )

    @staticmethod
    def _match_resolution(
        skip: torch.Tensor,
        skip_grid: Tuple[int, int, int],
        target_grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Interpolate skip features to match target grid if sizes differ."""
        D_s, H_s, W_s = skip_grid
        D_t, H_t, W_t = target_grid
        if D_s == D_t and H_s == H_t and W_s == W_t:
            return skip
        B, _, C = skip.shape
        skip = skip.transpose(1, 2).view(B, C, D_s, H_s, W_s)
        skip = F.interpolate(skip.float(), size=(D_t, H_t, W_t), mode='trilinear',
                             align_corners=False).to(skip.dtype)
        return skip.flatten(2).transpose(1, 2)

    @staticmethod
    def _run_stage(stage: nn.ModuleList, x: torch.Tensor,
                   grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        for block in stage:
            x = block(x, grid_shape)
        return x

    def _encode(self, x: torch.Tensor):
        """Shared encoder + memory-bank enhancement path."""
        target_shape = x.shape[2:]

        # --- Conv Stem ---
        tokens, grid0, stem_skips = self.conv_stem(x)

        # --- Encoder (grid passed explicitly for windowed attention) ---
        s0 = self._run_stage(self.encoder_stages[0], tokens, grid0)
        s1_tokens, grid1 = self.patch_merges[0](s0, grid0)
        s1 = self._run_stage(self.encoder_stages[1], s1_tokens, grid1)
        s2_tokens, grid2 = self.patch_merges[1](s1, grid1)
        s2 = self._run_stage(self.encoder_stages[2], s2_tokens, grid2)

        return tokens, stem_skips, target_shape, (s0, s1, s2), (grid0, grid1, grid2)

    def _decode(self, s0, s1, s2, grid0, grid1, grid2):
        """Shared ViT decoder path (token resolution)."""
        d1_tokens, grid_d1 = self.patch_expands[0](s2, grid2)
        s1_matched = self._match_resolution(s1, grid1, grid_d1)
        d1_tokens = d1_tokens + s1_matched
        d1 = self._run_stage(self.decoder_stages[0], d1_tokens, grid_d1)

        d2_tokens, grid_d2 = self.patch_expands[1](d1, grid_d1)
        s0_matched = self._match_resolution(s0, grid0, grid_d2)
        d2_tokens = d2_tokens + s0_matched
        d2 = self._run_stage(self.decoder_stages[1], d2_tokens, grid_d2)

        d2 = self.final_norm(d2)
        B, N, C = d2.shape
        vit_features = d2.transpose(1, 2).view(B, C, *grid_d2)
        return vit_features

    def _apply_memory(self, s0, s1, s2, grids, labels):
        grid0, grid1, grid2 = grids
        if self.training and labels is not None:
            self._proto_loss = self.memory_bank.compute_prototype_loss(
                [s0, s1, s2], labels,
                grid_shapes=[grid0, grid1, grid2],
            )
        else:
            self._proto_loss = None
        s0 = self.memory_bank.enhance_features(s0, level=0, grid_shape=grid0)
        s1 = self.memory_bank.enhance_features(s1, level=1, grid_shape=grid1)
        s2 = self.memory_bank.enhance_features(s2, level=2, grid_shape=grid2)
        return s0, s1, s2

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B,C,D,H,W] -> [B,num_classes,D,H,W] full-resolution logits."""
        _, stem_skips, target_shape, (s0, s1, s2), grids = self._encode(x)

        if self.memory_bank is not None:
            s0, s1, s2 = self._apply_memory(s0, s1, s2, grids, labels)

        grid0, grid1, grid2 = grids
        vit_features = self._decode(s0, s1, s2, grid0, grid1, grid2)
        out = self.cnn_decoder(vit_features, stem_skips, target_shape)
        return out

    def forward_features(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int, int]]:
        """Return ViT decoder features + stem skips for custom heads.

        Mirrors ViTUNet3D.forward_features (used by HierarchicalStage2Model).
        """
        _, stem_skips, target_shape, (s0, s1, s2), grids = self._encode(x)

        if self.memory_bank is not None:
            s0, s1, s2 = self._apply_memory(s0, s1, s2, grids, labels)

        grid0, grid1, grid2 = grids
        vit_features = self._decode(s0, s1, s2, grid0, grid1, grid2)
        return vit_features, stem_skips, target_shape


# ==================== Self-test ====================

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = dict(
        in_channels=1, num_classes=6, embed_dim=192,
        stem_channels=(48, 96, 192), tokenize_kernel=(2, 4, 4),
        num_heads=6, head_dim=32, kv_latent_dim=96,
        encoder_depths=[3, 6, 4], decoder_depths=[2, 3],
        mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.2,
    )
    x = torch.randn(1, 1, 40, 192, 192)

    # Option A (default): dual path only at shallow stage0.
    model = NNFormerUNet3D(**cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Option A] dual_path_stages={model.dual_path_stages} "
          f"params={n_params/1e6:.2f}M")
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"  in={tuple(x.shape)}  out={tuple(out.shape)}")
    assert out.shape == (1, 6, 40, 192, 192), out.shape
    assert torch.isfinite(out).all()

    feats, skips, ts = model.forward_features(x)
    print(f"  forward_features: feats={tuple(feats.shape)} "
          f"n_skips={len(skips)} target={ts}")

    # Full dual-path (all stages) for parameter comparison / ablation.
    full = NNFormerUNet3D(dual_path_stages=[0, 1, 2], **cfg)
    n_full = sum(p.numel() for p in full.parameters())
    print(f"[full]     dual_path_stages={full.dual_path_stages} "
          f"params={n_full/1e6:.2f}M  (saved {(n_full-n_params)/1e6:.2f}M)")

    # Residual conv branch at shallow stage0: bottleneck residual conv block.
    res = NNFormerUNet3D(residual_conv_stages=[0], **cfg)
    n_res = sum(p.numel() for p in res.parameters())
    print(f"[resconv]  residual_conv_stages={res.residual_conv_stages} "
          f"params={n_res/1e6:.2f}M  (+{(n_res-n_params)/1e6:.2f}M vs Option A)")
    res.eval()
    with torch.no_grad():
        out_r = res(x)
    assert out_r.shape == (1, 6, 40, 192, 192), out_r.shape
    assert torch.isfinite(out_r).all()
    print("NNFormerUNet3D smoke test OK")
