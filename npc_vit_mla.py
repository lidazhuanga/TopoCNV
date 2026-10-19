"""
ViT + MLA (Multi-head Latent Attention) module for 3D medical image segmentation.

This module provides the core architectural components for replacing the DynUNet
CNN backbone with a Vision Transformer (ViT) that integrates DeepSeek's MLA
mechanism and bidirectional (full) attention.

Components:
- MultiHeadLatentAttention: DeepSeek-style MLA with optional 3D RoPE
- TransformerEncoderBlock: Pre-norm transformer block with MLA + FFN
- PatchEmbedding3D: Conv3d-based 3D volume to token sequence
- PatchMerging3D: Merge 2x2x2 neighboring tokens (downsample)
- PatchExpanding3D: Expand each token to 2x2x2 (upsample)
- ViTUNet3D: Full U-shaped 3D ViT with MLA for segmentation
- MLAMemoryBank: Memory Bank with MLA-based cross-attention retrieval

Architecture is optimized for thin tubular nerve structure segmentation:
- Small patch size (8^3) preserves fine nerve details
- Hierarchical encoder captures both local and global context
- MLA low-rank bottleneck provides implicit regularization for small datasets
- Bidirectional attention links distant nerve segments
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== DropPath (Stochastic Depth) ====================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Standard regularization technique for Vision Transformers.
    During training, randomly drops entire residual branches.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob
        return x / keep_prob * random_tensor


# ==================== Multi-Head Latent Attention (DeepSeek MLA) ====================

class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-style Multi-head Latent Attention for 3D ViT.

    Compresses K,V into a shared latent vector c_KV via down-projection,
    then reconstructs K and V via separate up-projections.  This low-rank
    bottleneck reduces parameters and provides implicit regularization
    beneficial for small medical datasets.

    Uses full bidirectional attention (no causal mask) -- standard ViT
    setting where every token attends to every other token.

    Args:
        dim:           Input embedding dimension
        num_heads:     Number of attention heads
        head_dim:      Per-head dimension
        kv_latent_dim: Compressed KV latent dimension (d_c)
        dropout:       Attention dropout rate
        use_rope:      Whether to use decoupled 3D RoPE (default: False,
                       learned positional embedding used instead)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        kv_latent_dim: int = 128,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.use_rope = use_rope

        inner_dim = num_heads * head_dim

        # Query projection (standard, not compressed)
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)

        # KV down-projection: compress to latent space
        self.kv_down = nn.Linear(dim, kv_latent_dim, bias=False)

        # KV up-projections: reconstruct K and V from latent
        self.k_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Scale factor
        self.scale = head_dim ** -0.5

        # QK LayerNorm: prevents attention score overflow regardless of LR
        # (used by ViT-22B, StableDiffusion XL, etc.)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

        # Optional decoupled RoPE
        if use_rope:
            self.rope_dim = head_dim // 2
            self.nope_dim = head_dim - self.rope_dim
            # Separate projection for RoPE part of key (not from latent)
            self.k_rope_proj = nn.Linear(dim, num_heads * self.rope_dim, bias=False)
        else:
            self.rope_dim = 0
            self.nope_dim = head_dim

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, N, C] input tokens
            rope_cos: [N, rope_dim//2] cosine table for RoPE (optional)
            rope_sin: [N, rope_dim//2] sine table for RoPE (optional)
            attn_bias: [B, H, N, N] or broadcastable float additive bias

        Returns:
            [B, N, C] output tokens
        """
        B, N, C = x.shape

        # --- Query ---
        q = self.q_proj(x)  # [B, N, inner_dim]
        q = q.view(B, N, self.num_heads, self.head_dim)

        # --- KV via latent compression ---
        c_kv = self.kv_down(x)               # [B, N, kv_latent_dim]
        k_nope = self.k_up(c_kv)             # [B, N, inner_dim]
        v = self.v_up(c_kv)                  # [B, N, inner_dim]

        if self.use_rope:
            k_nope = k_nope.view(B, N, self.num_heads, self.nope_dim)
            # Separate key projection for RoPE part
            k_rope = self.k_rope_proj(x)  # [B, N, num_heads * rope_dim]
            k_rope = k_rope.view(B, N, self.num_heads, self.rope_dim)

            # Split query
            q_nope = q[..., :self.nope_dim]
            q_rope = q[..., self.nope_dim:]

            # Apply RoPE
            if rope_cos is not None and rope_sin is not None:
                q_rope = self._apply_rope(q_rope, rope_cos, rope_sin)
                k_rope = self._apply_rope(k_rope, rope_cos, rope_sin)

            q_full = torch.cat([q_nope, q_rope], dim=-1)
            k_full = torch.cat([k_nope, k_rope], dim=-1)
        else:
            q_full = q
            k_full = k_nope.view(B, N, self.num_heads, self.head_dim)

        v = v.view(B, N, self.num_heads, self.head_dim)

        # Transpose for attention: [B, num_heads, N, head_dim]
        q_full = q_full.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply QK normalization to prevent attention overflow
        q_full = self.q_norm(q_full)
        k_full = self.k_norm(k_full)

        # Bidirectional (full) attention via SDPA (avoids materializing [B,H,N,N])
        out = F.scaled_dot_product_attention(
            q_full, k_full, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    @staticmethod
    def _apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embedding.

        Args:
            x:   [B, N, num_heads, rope_dim]
            cos: [N, rope_dim//2]
            sin: [N, rope_dim//2]
        """
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]

        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, N, 1, d//2]
        sin = sin.unsqueeze(0).unsqueeze(2)

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ==================== Standard MHA (Ablation E) ====================

class StandardMHAWrapper(nn.Module):
    """Standard Multi-Head Attention for ablation E (MLA vs standard MHA).

    Drop-in replacement for MultiHeadLatentAttention that uses full-rank
    K, V projections instead of low-rank KV latent compression.
    Maintains QK LayerNorm for fair comparison.
    Ignores rope_cos, rope_sin (standard MHA does not use RoPE).
    """

    def __init__(self, dim, num_heads=8, head_dim=32, kv_latent_dim=128,
                 dropout=0.0, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        # Full-rank Q, K, V projections (no low-rank compression)
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        # QK LayerNorm (same as MLA for fair comparison)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ==================== Transformer Encoder Block ====================

class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block with MLA and FFN.

    Architecture: LN -> MLA -> DropPath -> residual -> LN -> FFN -> DropPath -> residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        kv_latent_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_rope: bool = False,
        use_mla: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if use_mla:
            self.attn = MultiHeadLatentAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_latent_dim=kv_latent_dim,
                dropout=dropout,
                use_rope=use_rope,
            )
        else:
            self.attn = StandardMHAWrapper(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_latent_dim=kv_latent_dim,
                dropout=dropout,
                use_rope=use_rope,
            )
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

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rope_cos, rope_sin))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ==================== Conv Stem 3D (Hybrid UNETR) ====================

class ConvStem3D(nn.Module):
    """Convolutional stem for Hybrid UNETR.

    Extracts multi-scale CNN features BEFORE tokenization, preserving
    high-frequency detail (edges, 1-2 voxel structures) that would be lost
    by direct large-stride patchification.

    Architecture:
        stem_0: Conv3d(in_ch -> stem_channels[0], k3, s1) -> full-res features
        stem_1: Conv3d(stem_channels[0] -> stem_channels[1], k3, s2) -> 1/2 res
        stem_2: Conv3d(stem_channels[1] -> stem_channels[2], k3, s2) -> 1/4 res
        tokenize: Conv3d(stem_channels[2] -> embed_dim, k=tok_kernel, s=tok_kernel) -> tokens

    Returns:
        tokens: [B, N, embed_dim]
        grid_shape: (gd, gh, gw)
        stem_skips: [skip_0, skip_1, skip_2] at full, 1/2, 1/4 resolution
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        stem_channels: Tuple[int, int, int] = (32, 64, 128),
        tokenize_kernel: Tuple[int, int, int] = (2, 4, 4),
        max_grid_size: Tuple[int, int, int] = (16, 16, 16),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_grid = max_grid_size
        self.stem_channels = stem_channels

        # Stem: progressive downsampling with local feature extraction
        self.stem_0 = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(stem_channels[0]),
            nn.GELU(),
        )
        self.stem_1 = nn.Sequential(
            nn.Conv3d(stem_channels[0], stem_channels[1], kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(stem_channels[1]),
            nn.GELU(),
        )
        self.stem_2 = nn.Sequential(
            nn.Conv3d(stem_channels[1], stem_channels[2], kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(stem_channels[2]),
            nn.GELU(),
        )

        # Tokenization: map stem output to embed_dim tokens
        self.tokenize = nn.Conv3d(
            stem_channels[2], embed_dim,
            kernel_size=tokenize_kernel,
            stride=tokenize_kernel,
        )

        # Learned 3D positional embedding
        max_tokens = max_grid_size[0] * max_grid_size[1] * max_grid_size[2]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], List[torch.Tensor]]:
        # Stem feature extraction
        s0 = self.stem_0(x)   # [B, 32,  D, H, W]      full-res
        s1 = self.stem_1(s0)  # [B, 64,  D/2, H/2, W/2]
        s2 = self.stem_2(s1)  # [B, 128, D/4, H/4, W/4]

        # Tokenize
        tok = self.tokenize(s2)  # [B, embed_dim, gd, gh, gw]
        gd, gh, gw = tok.shape[2], tok.shape[3], tok.shape[4]
        N = gd * gh * gw

        tokens = tok.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Add positional embedding
        if N != self.pos_embed.shape[1]:
            pe = self.pos_embed.transpose(1, 2).view(
                1, self.embed_dim, *self.max_grid
            )
            pe = F.interpolate(
                pe.float(), size=(gd, gh, gw),
                mode='trilinear', align_corners=False,
            ).to(tokens.dtype)
            pe = pe.flatten(2).transpose(1, 2)
            tokens = tokens + pe
        else:
            tokens = tokens + self.pos_embed[:, :N]

        return tokens, (gd, gh, gw), [s0, s1, s2]


# ==================== Patch Embedding 3D (Legacy) ====================

class PatchEmbedding3D(nn.Module):
    """Convert 3D volume to token sequence via Conv3d patchification.

    Input:  [B, C_in, D, H, W]
    Output: (tokens [B, N, embed_dim], grid_shape (D//ps, H//ps, W//ps))

    Includes learned 3D positional embedding with interpolation for
    variable input sizes.
    NOTE: This is the legacy version. New code should use ConvStem3D.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        max_grid_size: Tuple[int, int, int] = (16, 16, 16),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_grid = max_grid_size

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        max_tokens = max_grid_size[0] * max_grid_size[1] * max_grid_size[2]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, C, D, H, W = x.shape
        ps = self.patch_size

        x = self.proj(x)  # [B, embed_dim, D//ps, H//ps, W//ps]
        gd, gh, gw = x.shape[2], x.shape[3], x.shape[4]
        N = gd * gh * gw

        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Interpolate positional embedding if grid size differs
        if N != self.pos_embed.shape[1]:
            pe = self.pos_embed.transpose(1, 2).view(
                1, self.embed_dim, *self.max_grid
            )
            pe = F.interpolate(
                pe, size=(gd, gh, gw),
                mode='trilinear', align_corners=False,
            )
            pe = pe.flatten(2).transpose(1, 2)
            x = x + pe
        else:
            x = x + self.pos_embed[:, :N]

        return x, (gd, gh, gw)


# ==================== Patch Merging 3D (Downsample) ====================

class PatchMerging3D(nn.Module):
    """Merge 2x2x2 neighboring tokens (downsample spatial resolution by 2).

    Input:  tokens [B, N, C], grid_shape (D, H, W)
    Output: tokens [B, N//8, out_dim], grid_shape (D//2, H//2, W//2)

    Handles odd dimensions by padding to even before merging.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Linear(8 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(8 * in_dim)

    def forward(
        self, x: torch.Tensor, grid_shape: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, N, C = x.shape
        D, H, W = grid_shape

        # Pad odd dimensions to even (zero-pad)
        D_pad = D % 2
        H_pad = H % 2
        W_pad = W % 2
        if D_pad or H_pad or W_pad:
            x = x.view(B, D, H, W, C)
            x = F.pad(x, (0, 0, 0, W_pad, 0, H_pad, 0, D_pad), mode='constant', value=0)
            D, H, W = D + D_pad, H + H_pad, W + W_pad
            x = x.view(B, -1, C)

        x = x.view(B, D, H, W, C)

        # Take 2x2x2 groups
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = x.view(B, -1, 8 * C)  # [B, N//8, 8C]
        x = self.norm(x)
        x = self.reduction(x)     # [B, N//8, out_dim]

        return x, (D // 2, H // 2, W // 2)


# ==================== Patch Expanding 3D (Upsample) ====================

class PatchExpanding3D(nn.Module):
    """Expand each token to 2x2x2 (upsample spatial resolution by 2).

    Input:  tokens [B, N, C], grid_shape (D, H, W)
    Output: tokens [B, 8*N, out_dim], grid_shape (2D, 2H, 2W)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.expand = nn.Linear(in_dim, 8 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, grid_shape: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, N, C = x.shape
        D, H, W = grid_shape

        x = self.expand(x)  # [B, N, 8*out_dim]
        x = x.view(B, D, H, W, 2, 2, 2, self.out_dim)
        # Rearrange: [B, D, 2, H, 2, W, 2, out_dim] -> view [B, 2D, 2H, 2W, out_dim]
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, -1, self.out_dim)
        x = self.norm(x)

        return x, (2 * D, 2 * H, 2 * W)


# ==================== CNN Decoder 3D (Hybrid UNETR) ====================

class _DecoderBlock(nn.Module):
    """Single decoder upsample block: ConvTranspose3d + skip concat + conv refine."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel_size: Tuple[int, int, int] = (2, 2, 2),
                 stride: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        # After concat: out_ch + skip_ch -> out_ch
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Crop/pad to match skip spatial size (handle rounding differences)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x.float(), size=skip.shape[2:], mode='trilinear',
                              align_corners=False).to(x.dtype)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class CNNDecoder3D(nn.Module):
    """Progressive CNN decoder with stem skip connections for Hybrid UNETR.

    Takes ViT decoder output at token resolution and progressively upsamples
    through 3 stages with skip connections from the convolutional stem,
    restoring full spatial resolution with high-frequency detail.

    Architecture:
        up_2: ConvT(embed_dim -> stem_ch[2], tok_kernel) + skip_2 -> [B, stem_ch[2], D/4, H/4, W/4]
        up_1: ConvT(stem_ch[2] -> stem_ch[1], k2s2) + skip_1 -> [B, stem_ch[1], D/2, H/2, W/2]
        up_0: ConvT(stem_ch[1] -> stem_ch[0], k2s2) + skip_0 -> [B, stem_ch[0], D, H, W]
        out:  Conv3d(stem_ch[0], num_classes, k=1)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        stem_channels: Tuple[int, int, int] = (32, 64, 128),
        tokenize_kernel: Tuple[int, int, int] = (2, 4, 4),
        num_classes: int = 3,
    ):
        super().__init__()
        # up_2: token grid -> 1/4 res (invert tokenization)
        self.up_2 = _DecoderBlock(
            in_ch=embed_dim, skip_ch=stem_channels[2], out_ch=stem_channels[2],
            kernel_size=tokenize_kernel, stride=tokenize_kernel,
        )
        # up_1: 1/4 res -> 1/2 res
        self.up_1 = _DecoderBlock(
            in_ch=stem_channels[2], skip_ch=stem_channels[1], out_ch=stem_channels[1],
            kernel_size=(2, 2, 2), stride=(2, 2, 2),
        )
        # up_0: 1/2 res -> full res
        self.up_0 = _DecoderBlock(
            in_ch=stem_channels[1], skip_ch=stem_channels[0], out_ch=stem_channels[0],
            kernel_size=(2, 2, 2), stride=(2, 2, 2),
        )
        # Final segmentation head
        self.out_conv = nn.Conv3d(stem_channels[0], num_classes, kernel_size=1)

    def forward(
        self,
        vit_features: torch.Tensor,
        stem_skips: List[torch.Tensor],
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Progressive upsample with skip connections.

        Args:
            vit_features: [B, embed_dim, gd, gh, gw] from ViT decoder
            stem_skips:   [skip_0, skip_1, skip_2] from ConvStem3D
            target_shape: (D, H, W) original input spatial dims for final crop

        Returns:
            [B, num_classes, D, H, W] full-resolution segmentation logits
        """
        skip_0, skip_1, skip_2 = stem_skips

        x = self.up_2(vit_features, skip_2)  # -> 1/4 res
        x = self.up_1(x, skip_1)              # -> 1/2 res
        x = self.up_0(x, skip_0)              # -> full res

        out = self.out_conv(x)
        # Crop to match original input (in case of padding artifacts)
        D, H, W = target_shape
        out = out[:, :, :D, :H, :W]
        return out


# ==================== ViT-UNETR 3D ====================

class ViTUNet3D(nn.Module):
    """Hybrid UNETR: Conv Stem + ViT Encoder/Decoder + CNN Decoder with full-res skips.

    Architecture:
      Conv Stem (preserves high-frequency detail):
        stem_0: Conv3d(s=1) -> full-res features (skip_0)
        stem_1: Conv3d(s=2) -> 1/2 res (skip_1)
        stem_2: Conv3d(s=2) -> 1/4 res (skip_2)
        tokenize: Conv3d(s=tok_kernel) -> tokens

      ViT Encoder:
        Stage 0: encoder_depths[0] blocks @ grid0  -> skip_vit_0
        PatchMerging (grid0 -> grid1)
        Stage 1: encoder_depths[1] blocks @ grid1  -> skip_vit_1
        PatchMerging (grid1 -> grid2)
        Stage 2: encoder_depths[2] blocks @ grid2  (bottleneck)

      ViT Decoder:
        PatchExpanding (grid2 -> grid1) + skip_vit_1
        PatchExpanding (grid1 -> grid0) + skip_vit_0

      CNN Decoder (progressive upsample with stem skips):
        up_2: token_grid -> 1/4 res + skip_2
        up_1: 1/4 res -> 1/2 res + skip_1
        up_0: 1/2 res -> full res + skip_0
        Conv3d(1x1) -> segmentation output

    This ensures 1-2 voxel structures (thin nerves) survive to the output
    via full-resolution CNN skip connections.
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.memory_bank = memory_bank
        self.stem_channels = stem_channels
        self.tokenize_kernel = tokenize_kernel
        self.use_mla = use_mla

        # Stage dimensions (doubling at each merge)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]

        # --- Convolutional Stem (replaces PatchEmbedding3D) ---
        self.conv_stem = ConvStem3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            stem_channels=stem_channels,
            tokenize_kernel=tokenize_kernel,
        )

        # --- Compute per-block drop path rates (linearly increasing) ---
        total_encoder_blocks = sum(encoder_depths)
        total_decoder_blocks = sum(decoder_depths)
        total_blocks = total_encoder_blocks + total_decoder_blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # --- Encoder stages ---
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i in range(3):
            blocks = nn.Sequential(*[
                TransformerEncoderBlock(
                    dim=dims[i],
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_latent_dim=kv_latent_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path_rate=dpr[block_idx + j],
                    use_mla=use_mla,
                ) for j in range(encoder_depths[i])
            ])
            block_idx += encoder_depths[i]
            self.encoder_stages.append(blocks)

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
            blocks = nn.Sequential(*[
                TransformerEncoderBlock(
                    dim=dims[1 - i],
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_latent_dim=kv_latent_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path_rate=dpr[block_idx + j],
                    use_mla=use_mla,
                ) for j in range(decoder_depths[i])
            ])
            block_idx += decoder_depths[i]
            self.decoder_stages.append(blocks)

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
        """Interpolate skip features to match target grid if sizes differ.

        Handles cases where PatchMerging padding causes grid size mismatches.
        """
        D_s, H_s, W_s = skip_grid
        D_t, H_t, W_t = target_grid
        if D_s == D_t and H_s == H_t and W_s == W_t:
            return skip
        B, _, C = skip.shape
        skip = skip.transpose(1, 2).view(B, C, D_s, H_s, W_s)
        skip = F.interpolate(skip.float(), size=(D_t, H_t, W_t), mode='trilinear', align_corners=False).to(skip.dtype)
        return skip.flatten(2).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      [B, C, D, H, W] input volume
            labels: [B, D, H, W] optional, for Memory Bank EMA update

        Returns:
            [B, num_classes, D, H, W] segmentation logits (full resolution)
        """
        target_shape = x.shape[2:]  # (D, H, W)

        # --- Conv Stem ---
        tokens, grid0, stem_skips = self.conv_stem(x)  # [B, N0, embed_dim], skips=[s0,s1,s2]

        # --- ViT Encoder ---
        s0 = self.encoder_stages[0](tokens)                  # [B, N0, dims[0]]

        s1_tokens, grid1 = self.patch_merges[0](s0, grid0)   # [B, N1, dims[1]]
        s1 = self.encoder_stages[1](s1_tokens)                # [B, N1, dims[1]]

        s2_tokens, grid2 = self.patch_merges[1](s1, grid1)   # [B, N2, dims[2]]
        s2 = self.encoder_stages[2](s2_tokens)                # [B, N2, dims[2]] (bottleneck)

        # --- Memory Bank Enhancement ---
        if self.memory_bank is not None:
            # Compute prototype contrastive loss during training
            if self.training and labels is not None:
                self._proto_loss = self.memory_bank.compute_prototype_loss(
                    [s0, s1, s2], labels,
                    grid_shapes=[grid0, grid1, grid2],
                )
            else:
                self._proto_loss = None
            # Enhance skip features via MLA cross-attention (all 3 levels)
            s0 = self.memory_bank.enhance_features(s0, level=0, grid_shape=grid0)
            s1 = self.memory_bank.enhance_features(s1, level=1, grid_shape=grid1)
            s2 = self.memory_bank.enhance_features(s2, level=2, grid_shape=grid2)

        # --- ViT Decoder ---
        d1_tokens, grid_d1 = self.patch_expands[0](s2, grid2)  # [B, N1, dims[1]]
        s1_matched = self._match_resolution(s1, grid1, grid_d1)
        d1_tokens = d1_tokens + s1_matched                      # additive skip
        d1 = self.decoder_stages[0](d1_tokens)                   # [B, N1, dims[1]]

        d2_tokens, grid_d2 = self.patch_expands[1](d1, grid_d1) # [B, N0, dims[0]]
        s0_matched = self._match_resolution(s0, grid0, grid_d2)
        d2_tokens = d2_tokens + s0_matched                      # additive skip
        d2 = self.decoder_stages[1](d2_tokens)                   # [B, N0, dims[0]]

        # --- Reshape ViT output to spatial ---
        d2 = self.final_norm(d2)
        B, N, C = d2.shape
        vit_features = d2.transpose(1, 2).view(B, C, *grid_d2)  # [B, embed_dim, gd, gh, gw]

        # --- CNN Decoder (progressive upsample with stem skip connections) ---
        out = self.cnn_decoder(vit_features, stem_skips, target_shape)
        return out

    def forward_features(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int, int]]:
        """Run encoder-decoder and return features for custom heads.

        Used by HierarchicalStage2Model to apply custom heads.
        Returns ViT decoder features at token resolution AND stem skips for
        the CNN decoder to produce full-resolution outputs.

        Args:
            x:      [B, C, D, H, W] input volume
            labels: [B, D, H, W] optional, for Memory Bank EMA update

        Returns:
            vit_features: [B, embed_dim, gd, gh, gw] decoder features at token resolution
            stem_skips:   [skip_0, skip_1, skip_2] from conv stem
            target_shape: (D, H, W) original spatial dimensions
        """
        target_shape = x.shape[2:]

        # --- Conv Stem ---
        tokens, grid0, stem_skips = self.conv_stem(x)

        # --- ViT Encoder ---
        s0 = self.encoder_stages[0](tokens)
        s1_tokens, grid1 = self.patch_merges[0](s0, grid0)
        s1 = self.encoder_stages[1](s1_tokens)
        s2_tokens, grid2 = self.patch_merges[1](s1, grid1)
        s2 = self.encoder_stages[2](s2_tokens)

        # --- Memory Bank Enhancement ---
        if self.memory_bank is not None:
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

        # --- ViT Decoder ---
        d1_tokens, grid_d1 = self.patch_expands[0](s2, grid2)
        s1_matched = self._match_resolution(s1, grid1, grid_d1)
        d1_tokens = d1_tokens + s1_matched
        d1 = self.decoder_stages[0](d1_tokens)

        d2_tokens, grid_d2 = self.patch_expands[1](d1, grid_d1)
        s0_matched = self._match_resolution(s0, grid0, grid_d2)
        d2_tokens = d2_tokens + s0_matched
        d2 = self.decoder_stages[1](d2_tokens)

        # --- Return features for custom heads ---
        d2 = self.final_norm(d2)
        B, N, C = d2.shape
        vit_features = d2.transpose(1, 2).view(B, C, *grid_d2)  # [B, embed_dim, gd, gh, gw]
        return vit_features, stem_skips, target_shape


# ==================== MLA Cross-Attention Fusion ====================

class _MLACrossFusion(nn.Module):
    """MLA-based cross-attention fusion block.

    Replaces simple Linear([tokens; memory]) concatenation with
    DeepSeek-style MLA cross-attention:
    - Q from original tokens (query side)
    - KV from memory output (context side, compressed via latent)
    - Residual + FFN (like a Transformer decoder cross-attention block)

    This allows each token to selectively attend to memory-enhanced
    features at all spatial positions, rather than a fixed linear blend.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 kv_latent_dim: int, dropout: float = 0.1):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-norm for Q and KV branches
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Q projection from tokens
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        # MLA latent bottleneck for KV
        self.kv_down = nn.Linear(dim, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Post-attention FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, memory_out: torch.Tensor) -> torch.Tensor:
        """Cross-attention fusion.

        Args:
            tokens:     [B, N, C] original encoder features
            memory_out: [B, N, C] memory-enhanced features from prototype retrieval

        Returns:
            [B, N, C] fused features
        """
        B, N, C = tokens.shape

        # Q from tokens
        q = self.q_proj(self.norm_q(tokens))
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]

        # KV from memory output (MLA latent compression → decompression)
        mem = self.norm_kv(memory_out)
        latent = self.kv_down(mem)  # [B, N, kv_latent_dim]
        k = self.k_up(latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product cross-attention via SDPA (avoids materializing [B,H,N,N])
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.head_dim ** -0.5,
        )  # [B, H, N, d]

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)  # [B, N, C]

        # Residual + FFN (decoder-style block)
        enhanced = tokens + out
        enhanced = enhanced + self.ffn(enhanced)
        return enhanced


# ==================== MLA Memory Bank ====================

class MLAMemoryBank(nn.Module):
    """Memory Bank with MLA-based cross-attention retrieval.

    Upgrades the existing MultiScaleMemoryBank from cosine-similarity to
    MLA cross-attention:
    - Prototypes stored as learnable latent vectors [num_prototypes, kv_latent_dim]
    - Query: current feature tokens (projected via q_proj)
    - KV: memory bank latent prototypes (reconstructed via k_up, v_up)
    - Cross-attention retrieves relevant anatomical patterns

    Prototypes are nn.Parameter updated by gradient descent via:
      1. Prototype contrastive loss (class-discriminative structure)
      2. Segmentation loss (task utility, via cross-attention read-out)

    Maintains patch-level spatial information (2x2x2 grid).

    Args:
        feature_dims:    Feature dimensions per level [256, 512, 1024]
        num_prototypes_per_class: Prototypes per foreground class (default: 8)
        num_classes:     Total classes including background (default: 3)
        kv_latent_dim:   MLA latent dimension (default: 128)
        num_heads:       Attention heads (default: 8)
        head_dim:        Per-head dimension (default: 32)
        temperature:     Attention temperature (default: 1.0)
        patch_grid:      Spatial pooling grid (default: (2, 2, 2))
    """

    def __init__(
        self,
        feature_dims: List[int] = [256, 512, 1024],
        num_prototypes_per_class: int = 8,
        num_classes: int = 3,
        kv_latent_dim: int = 128,
        num_heads: int = 8,
        head_dim: int = 32,
        momentum: float = 0.99,
        temperature: float = 1.0,
        patch_grid: Tuple[int, int, int] = (2, 2, 2),
        top_k: int = 8,
        retrieval_mode: str = 'cross_attention',
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_levels = len(feature_dims)
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_classes = num_classes
        self.num_foreground_classes = num_classes - 1
        self.num_prototypes = num_prototypes_per_class * self.num_foreground_classes
        self.kv_latent_dim = kv_latent_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temperature = temperature
        self.patch_grid = patch_grid
        self.top_k = min(top_k, self.num_prototypes)  # Retrieve top-k most relevant prototypes
        self.retrieval_mode = retrieval_mode  # 'cross_attention' or 'cosine'
        # Warmup control: disabled initially to avoid noise injection into
        # feature enhancement. Prototype contrastive loss runs from epoch 0.
        # Saved as persistent buffer so checkpoints preserve the flag.
        self.register_buffer('enabled', torch.tensor(False, dtype=torch.bool))

        inner_dim = num_heads * head_dim

        # Prototypes as learnable parameters (gradient-updated via
        # contrastive loss + segmentation loss through cross-attention)
        for level in range(self.num_levels):
            prototypes = torch.randn(self.num_prototypes, kv_latent_dim)
            prototypes = F.normalize(prototypes, dim=1)
            self.register_parameter(
                f'prototypes_level{level}', nn.Parameter(prototypes))

        # Per-level MLA cross-attention projections (used in cross_attention mode)
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, inner_dim, bias=False) for dim in feature_dims
        ])
        self.k_ups = nn.ModuleList([
            nn.Linear(kv_latent_dim, inner_dim, bias=False) for _ in feature_dims
        ])
        self.v_ups = nn.ModuleList([
            nn.Linear(kv_latent_dim, inner_dim, bias=False) for _ in feature_dims
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(inner_dim, dim, bias=False) for dim in feature_dims
        ])

        # Per-level KV down-projection (for EMA update, also used in cosine mode)
        self.kv_downs = nn.ModuleList([
            nn.Linear(dim, kv_latent_dim, bias=False) for dim in feature_dims
        ])

        if self.retrieval_mode == 'cross_attention':
            # Per-level MLA cross-attention fusion: tokens attend to memory output
            self.memory_fusions = nn.ModuleList([
                _MLACrossFusion(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_latent_dim=kv_latent_dim,
                    dropout=0.1,
                )
                for dim in feature_dims
            ])
        else:
            # Cosine-similarity mode: simple linear blend instead of cross-attention fusion
            self.cosine_fuse_projs = nn.ModuleList([
                nn.Linear(dim, dim) for dim in feature_dims
            ])
            self.cosine_fuse_norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in feature_dims
            ])
            self.memory_fusions = None

    def get_prototypes(self, level: int) -> torch.Tensor:
        return getattr(self, f'prototypes_level{level}')

    def set_prototypes(self, level: int, prototypes: torch.Tensor):
        param = getattr(self, f'prototypes_level{level}')
        param.data.copy_(prototypes)

    @torch.no_grad()
    def initialize_from_features(
        self,
        features_list: List[torch.Tensor],
        labels: torch.Tensor,
        grid_shapes: List[Tuple[int, int, int]],
    ):
        """Initialize prototypes from class-wise feature statistics.

        Runs a forward pass on training data and computes class-wise feature
        means in the latent space. Each class's prototypes are initialized
        around its feature mean with small random perturbation.

        Args:
            features_list: [tokens_level0, tokens_level1, tokens_level2]
            labels:        [B, D_full, H_full, W_full] ground truth labels
            grid_shapes:   [(D0,H0,W0), (D1,H1,W1), (D2,H2,W2)] token grids
        """
        for level, features in enumerate(features_list):
            # Pool features to grid
            patch_features = self._pool_to_grid(features, grid_shapes[level])
            B, N_patches, C = patch_features.shape

            # Down-project to latent space
            patch_latent = self.kv_downs[level](patch_features)
            patch_latent = F.normalize(patch_latent, dim=2)  # [B, N_patches, kv_latent_dim]

            # Pool labels to same grid
            labels_f = labels.unsqueeze(1).float()
            labels_grid = F.adaptive_avg_pool3d(labels_f, self.patch_grid)
            labels_grid_flat = labels_grid.flatten(2).squeeze(1)  # [B, N_patches]

            # Compute class-wise feature means
            prototypes = self.get_prototypes(level).clone()
            for cls_idx in range(1, self.num_classes):
                mask = (
                    (labels_grid_flat > cls_idx - 0.5) &
                    (labels_grid_flat < cls_idx + 0.5)
                )  # [B, N_patches]
                if mask.any():
                    cls_features = patch_latent[mask]  # [M, kv_latent_dim]
                    cls_mean = cls_features.mean(dim=0)  # [kv_latent_dim]
                    cls_mean = F.normalize(cls_mean.unsqueeze(0), dim=1).squeeze(0)

                    # Initialize this class's prototypes around the mean
                    proto_start = (cls_idx - 1) * self.num_prototypes_per_class
                    proto_end = cls_idx * self.num_prototypes_per_class
                    for i in range(proto_start, proto_end):
                        # Small random perturbation around class mean
                        noise = torch.randn_like(cls_mean) * 0.1
                        prototypes[i] = F.normalize((cls_mean + noise).unsqueeze(0), dim=1).squeeze(0)

            self.set_prototypes(level, prototypes)

    def _pool_to_grid(
        self, tokens: torch.Tensor, grid_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Pool tokens to patch_grid spatial resolution.

        Args:
            tokens: [B, N, C] where N = D*H*W
            grid_shape: (D, H, W) token spatial grid

        Returns:
            [B, gD*gH*gW, C] pooled tokens
        """
        B, N, C = tokens.shape
        D, H, W = grid_shape

        x = tokens.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        x = F.adaptive_avg_pool3d(x, self.patch_grid)           # [B, C, gD, gH, gW]
        x = x.flatten(2).transpose(1, 2)                         # [B, gD*gH*gW, C]
        return x

    def _labels_to_token_grid(
        self,
        labels: torch.Tensor,
        grid_shape: Tuple[int, int, int],
        fg_thresh: float = 0.10,
    ) -> torch.Tensor:
        """Downsample integer labels to the native token grid, keeping the
        dominant foreground class per token (hard label, not averaged).

        A token is assigned the foreground class with the largest occupancy
        fraction inside it; it stays background (0) only if no foreground
        class exceeds ``fg_thresh``.  This matches the resolution at which the
        prototypes are actually probed, unlike the coarse (2,2,2) averaging
        that washed out small nerve structures.

        Args:
            labels:     [B, D, H, W] integer GT labels
            grid_shape: (gd, gh, gw) token grid for this level
            fg_thresh:  minimum fg fraction for a token to be foreground

        Returns:
            [B, gd*gh*gw] long tensor with values in 0..num_classes-1
        """
        gd, gh, gw = grid_shape
        oh = F.one_hot(
            labels.long().clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 4, 1, 2, 3).float()                 # [B, num_classes, D, H, W]
        pooled = F.adaptive_avg_pool3d(oh, (gd, gh, gw))  # class occupancy per token
        fg = pooled[:, 1:]                                # [B, C-1, gd, gh, gw]
        fg_max, fg_arg = fg.max(dim=1)                    # dominant fg class fraction/idx
        dom = torch.where(
            fg_max > fg_thresh, fg_arg + 1, torch.zeros_like(fg_arg))
        return dom.flatten(1)                             # [B, gd*gh*gw]

    def compute_balance_loss(
        self,
        features_list: List[torch.Tensor],
        grid_shapes: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Utilization balance regularization via KL divergence.

        Prevents few prototypes from monopolizing retrieval by encouraging
        uniform usage distribution across all prototypes.

        L_balance = KL(Uniform || usage_dist)

        Args:
            features_list: [tokens_level0, tokens_level1, tokens_level2]
            grid_shapes:   [(D0,H0,W0), (D1,H1,W1), (D2,H2,W2)] token grids

        Returns:
            Scalar balance loss (averaged over levels).
        """
        total_loss = 0.0
        n_levels = 0

        for level, features in enumerate(features_list):
            # Pool features to grid
            patch_features = self._pool_to_grid(features, grid_shapes[level])
            B, N_patches, C = patch_features.shape

            # Down-project to latent space
            patch_latent = self.kv_downs[level](patch_features)
            patch_latent = F.normalize(patch_latent, dim=2)  # [B, N_patches, kv_latent_dim]

            # Get normalized prototypes
            prototypes = F.normalize(
                self.get_prototypes(level), dim=1)  # [num_prototypes, kv_latent_dim]

            # Compute similarity: [B, N_patches, num_prototypes]
            sim = torch.matmul(patch_latent, prototypes.t())

            # Soft assignment probabilities (softmax over prototypes)
            # Average over batch and patches to get usage distribution
            assignment = F.softmax(sim / self.temperature, dim=-1)  # [B, N_patches, num_prototypes]
            usage_dist = assignment.mean(dim=(0, 1))  # [num_prototypes]

            # Uniform target distribution
            uniform = torch.ones_like(usage_dist) / self.num_prototypes

            # KL(Uniform || usage_dist) = sum(uniform * log(uniform / usage_dist))
            # Add small epsilon for numerical stability
            eps = 1e-8
            kl_loss = F.kl_div(
                (usage_dist + eps).log(),
                uniform,
                reduction='sum',
                log_target=False
            )
            total_loss = total_loss + kl_loss
            n_levels += 1

        return total_loss / max(n_levels, 1)

    def compute_prototype_loss(
        self,
        features_list: List[torch.Tensor],
        labels: torch.Tensor,
        grid_shapes: List[Tuple[int, int, int]],
        balance_weight: float = 0.05,
        intra_weight: float = 0.3,
        ortho_weight: float = 0.5,
        ortho_margin: float = 0.3,
    ) -> torch.Tensor:
        """Prototype shaping loss operating at native token resolution.

        Fixes the collapse of the old coarse-grid formulation via four terms:

        1. Class discrimination (fine-res CE): every FOREGROUND token, at the
           real token grid, must be most similar to its own class' prototype
           block.  Labels are pooled with a dominant-class rule (see
           ``_labels_to_token_grid``) instead of averaging, so small nerves are
           not washed out.
        2. Intra-class balance: within a token's GT class block, usage is
           spread across all K prototypes (soft-assignment KL to uniform),
           preventing one prototype from monopolizing the class.
        3. Orthogonality hinge: pairwise prototype cosine similarity above
           ``ortho_margin`` is penalized, so prototypes stay distinct.
        4. Global utilization balance: keeps overall prototype usage even.

        Args:
            features_list: [tokens_level0, tokens_level1, tokens_level2]
            labels:        [B, D_full, H_full, W_full] ground truth labels
            grid_shapes:   [(D0,H0,W0), (D1,H1,W1), (D2,H2,W2)] token grids
            balance_weight: weight for global utilization balance
            intra_weight:   weight for intra-class prototype balance
            ortho_weight:   weight for orthogonality hinge
            ortho_margin:   allowed pairwise cosine before penalty

        Returns:
            Scalar prototype loss (averaged over levels).
        """
        ce_loss = 0.0
        intra_loss = 0.0
        ortho_loss = 0.0
        n_ce = 0
        n_intra = 0
        K = self.num_prototypes_per_class
        eps = 1e-8

        for level, features in enumerate(features_list):
            B, N, C = features.shape

            # Native-resolution token latents (no coarse pooling)
            token_latent = F.normalize(
                self.kv_downs[level](features), dim=-1)         # [B, N, kv]
            prototypes = F.normalize(
                self.get_prototypes(level), dim=1)              # [P, kv]

            sim = torch.matmul(token_latent, prototypes.t())    # [B, N, P]
            sim_blocks = sim.view(
                B, N, self.num_foreground_classes, K)           # [B, N, C_fg, K]

            # Dominant hard class per token at this grid
            dom = self._labels_to_token_grid(
                labels, grid_shapes[level])                     # [B, N]

            # --- 1. Class discrimination (max within block) ---
            class_logits = sim_blocks.max(dim=3).values / self.temperature
            fg_mask = dom > 0
            if fg_mask.any():
                logits_fg = class_logits[fg_mask]               # [M, C_fg]
                target = (dom[fg_mask] - 1).long()              # [M]
                ce_loss = ce_loss + F.cross_entropy(logits_fg, target)
                n_ce += 1

                # --- 2. Intra-class balance: spread over K prototypes ---
                uniform_k = torch.full(
                    (K,), 1.0 / K, device=sim.device)
                for cls_idx in range(1, self.num_classes):
                    m = dom == cls_idx
                    if m.sum() < 2:
                        continue
                    block = sim_blocks[..., cls_idx - 1, :][m]   # [Mc, K]
                    assign = F.softmax(block / self.temperature, dim=-1)
                    usage = assign.mean(dim=0)                   # [K]
                    intra_loss = intra_loss + F.kl_div(
                        (usage + eps).log(), uniform_k,
                        reduction='sum', log_target=False)
                    n_intra += 1

            # --- 3. Orthogonality hinge (per level) ---
            gram = torch.matmul(prototypes, prototypes.t())     # [P, P]
            P = gram.shape[0]
            off = gram - torch.diag(torch.diagonal(gram))
            ortho_loss = ortho_loss + F.relu(
                off.abs() - ortho_margin).sum() / (P * (P - 1))

        ce_loss = ce_loss / max(n_ce, 1)
        intra_loss = intra_loss / max(n_intra, 1)
        ortho_loss = ortho_loss / max(len(features_list), 1)

        # --- 4. Global utilization balance ---
        balance_loss = self.compute_balance_loss(features_list, grid_shapes)

        return (ce_loss
                + intra_weight * intra_loss
                + ortho_weight * ortho_loss
                + balance_weight * balance_loss)

    def enhance_features(
        self,
        tokens: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """MLA cross-attention feature enhancement.

        1. Pool tokens to GxGxG grid
        2. Cross-attend to prototypes (Q=features, KV=prototypes)
        3. Upsample enhanced features back to token resolution
        4. MLA cross-attention fusion (Q=tokens, KV=memory output) + FFN

        Args:
            tokens:     [B, N, C] feature tokens
            level:      encoder level index
            grid_shape: (D, H, W) token spatial grid

        Returns:
            [B, N, C] enhanced tokens
        """
        # Bypass during warmup period
        if not self.enabled:
            return tokens
        B, N, C = tokens.shape
        D, H, W = grid_shape
        gD, gH, gW = self.patch_grid
        N_patches = gD * gH * gW

        # Pool to grid
        patch_tokens = self._pool_to_grid(tokens, grid_shape)  # [B, N_patches, C]

        # Get prototypes (latent vectors)
        prototypes = self.get_prototypes(level)  # [num_prototypes, kv_latent_dim]

        # --- Top-k retrieval in latent space ---
        # Down-project patch tokens to latent space for similarity search
        patch_latent = self.kv_downs[level](patch_tokens)  # [B, N_patches, kv_latent_dim]
        patch_latent = F.normalize(patch_latent, dim=2)

        # Compute similarity: average patch query vs all prototypes
        query_avg = patch_latent.mean(dim=1)  # [B, kv_latent_dim]
        similarity = torch.matmul(query_avg, prototypes.t())  # [B, num_prototypes]

        # Select top-k most relevant prototypes per sample
        _, topk_indices = similarity.topk(self.top_k, dim=1)  # [B, top_k]

        # --- Retrieval: cross-attention or cosine-similarity ---
        if self.retrieval_mode == 'cross_attention':
            # MLA cross-attention with retrieved prototypes
            # Query from features
            q = self.q_projs[level](patch_tokens)  # [B, N_patches, inner_dim]
            q = q.view(B, N_patches, self.num_heads, self.head_dim)

            # KV from selected prototypes only (latent -> reconstructed)
            k = self.k_ups[level](prototypes)  # [num_prototypes, inner_dim]
            v = self.v_ups[level](prototypes)  # [num_prototypes, inner_dim]

            # Gather top-k prototypes per sample: [B, top_k, inner_dim]
            k_selected = k[topk_indices]  # [B, top_k, inner_dim]
            v_selected = v[topk_indices]  # [B, top_k, inner_dim]

            k_selected = k_selected.view(B, self.top_k, self.num_heads, self.head_dim)
            v_selected = v_selected.view(B, self.top_k, self.num_heads, self.head_dim)

            # Transpose: [B, num_heads, N, head_dim]
            q = q.transpose(1, 2)
            k_selected = k_selected.transpose(1, 2)
            v_selected = v_selected.transpose(1, 2)

            # Cross-attention with top-k prototypes
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k_selected.transpose(-2, -1)) * scale
            attn = F.softmax(attn / self.temperature, dim=-1)

            out = torch.matmul(attn, v_selected)  # [B, num_heads, N_patches, head_dim]
            out = out.transpose(1, 2).contiguous().view(B, N_patches, -1)
            out = self.out_projs[level](out)  # [B, N_patches, C]
        else:
            # Cosine-similarity weighted retrieval (ablation J)
            # Direct cosine similarity between patch latents and selected prototype latents
            proto_selected = prototypes[topk_indices]  # [B, top_k, kv_latent_dim]
            proto_selected = F.normalize(proto_selected, dim=2)

            # Cosine similarity: [B, N_patches, top_k]
            cos_sim = torch.matmul(patch_latent, proto_selected.transpose(1, 2))
            cos_sim = cos_sim / self.temperature
            weights = F.softmax(cos_sim, dim=-1)  # [B, N_patches, top_k]

            # Up-project selected prototypes to feature space
            v = self.v_ups[level](prototypes)  # [num_prototypes, inner_dim]
            v_selected = v[topk_indices]  # [B, top_k, inner_dim]

            # Weighted sum of prototype values
            out = torch.matmul(weights, v_selected)  # [B, N_patches, inner_dim]
            out = self.out_projs[level](out)  # [B, N_patches, C]

        # Upsample to original token resolution (cast to float32 for trilinear compatibility)
        out = out.view(B, gD, gH, gW, C).permute(0, 4, 1, 2, 3)  # [B, C, gD, gH, gW]
        orig_dtype = out.dtype
        out = F.interpolate(
            out.float(), size=(D, H, W),
            mode='trilinear', align_corners=False,
        ).to(orig_dtype)
        out = out.flatten(2).transpose(1, 2)  # [B, N, C]

        if self.retrieval_mode == 'cross_attention':
            # MLA cross-attention fusion: tokens attend to memory output
            enhanced = self.memory_fusions[level](tokens, out)  # [B, N, C]
        else:
            # Simple linear blend: residual + projected memory output
            enhanced = tokens + self.cosine_fuse_projs[level](self.cosine_fuse_norms[level](out))
        return enhanced

    def forward(
        self,
        features_list: List[torch.Tensor],
        grid_shapes: List[Tuple[int, int, int]],
    ) -> List[torch.Tensor]:
        """Multi-scale feature enhancement."""
        return [
            self.enhance_features(feat, level, grid_shapes[level])
            for level, feat in enumerate(features_list)
        ]


# ==================== Context Anchor Memory Bank ====================

class ContextAnchorMemoryBank(nn.Module):
    """Memory Bank with context-anchor indexed prototypes.

    Organizes prototypes by (anatomical_context, class) pairs instead of
    class alone.  Context anchors represent archetypal anatomical environments
    (e.g. bone-air interface, CSF-bone boundary) that are invariant across
    patients regardless of head size.

    Key idea: nerve identification relies on surrounding tissue context, not
    intrinsic nerve pixel features.  Different segments of the same nerve pass
    through different anatomical contexts.  Context anchors capture these
    recurring context patterns, enabling location-specific feature retrieval
    without any region annotation or spatial coordinate dependency.

    Prototype layout per level:
        [A, C-1, K, kv_latent_dim]
    where A = num_anchors, C-1 = foreground classes, K = protos_per_anchor.
    Total per class = A * K (e.g. 8*4 = 32, matching original design).

    Args:
        feature_dims:       Feature dimensions per encoder level
        num_classes:        Total classes including background
        kv_latent_dim:      MLA latent dimension
        num_anchors:        Number of context anchor vectors (A)
        protos_per_anchor:  Prototype variants per (anchor, class) pair (K)
        num_heads:          Attention heads
        head_dim:           Per-head dimension
        temperature:        Softmax temperature
        patch_grid:         Spatial pooling grid
        top_k:              Top-k prototypes retrieved per query
    """

    def __init__(
        self,
        feature_dims: List[int] = [192, 384, 768],
        num_classes: int = 3,
        kv_latent_dim: int = 96,
        num_anchors: int = 8,
        protos_per_anchor: int = 4,
        num_heads: int = 6,
        head_dim: int = 32,
        temperature: float = 0.5,
        patch_grid: Tuple[int, int, int] = (2, 2, 2),
        top_k: int = 8,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_levels = len(feature_dims)
        self.num_classes = num_classes
        self.num_fg_classes = num_classes - 1
        self.num_anchors = num_anchors
        self.protos_per_anchor = protos_per_anchor
        self.kv_latent_dim = kv_latent_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temperature = temperature
        self.patch_grid = patch_grid
        self.momentum = momentum

        self.num_prototypes = num_anchors * self.num_fg_classes * protos_per_anchor
        self.top_k = min(top_k, self.num_prototypes)

        # Warmup control (same protocol as MLAMemoryBank)
        self.register_buffer('enabled', torch.tensor(False, dtype=torch.bool))

        inner_dim = num_heads * head_dim

        # ---- Context anchors: learnable archetypal context signatures ----
        anchors = torch.randn(num_anchors, kv_latent_dim)
        anchors = F.normalize(anchors, dim=-1)
        self.context_anchors = nn.Parameter(anchors)

        # ---- Prototypes: [A, C-1, K, kv_dim] per level ----
        for level in range(self.num_levels):
            protos = torch.randn(
                num_anchors, self.num_fg_classes, protos_per_anchor, kv_latent_dim)
            protos = F.normalize(protos, dim=-1)
            self.register_parameter(
                f'prototypes_level{level}', nn.Parameter(protos))

        # ---- Per-level projections ----
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, inner_dim, bias=False) for dim in feature_dims])
        self.k_ups = nn.ModuleList([
            nn.Linear(kv_latent_dim, inner_dim, bias=False) for _ in feature_dims])
        self.v_ups = nn.ModuleList([
            nn.Linear(kv_latent_dim, inner_dim, bias=False) for _ in feature_dims])
        self.out_projs = nn.ModuleList([
            nn.Linear(inner_dim, dim, bias=False) for dim in feature_dims])
        self.kv_downs = nn.ModuleList([
            nn.Linear(dim, kv_latent_dim, bias=False) for dim in feature_dims])

        # ---- Per-level MLA cross-attention fusion ----
        self.memory_fusions = nn.ModuleList([
            _MLACrossFusion(
                dim=dim, num_heads=num_heads, head_dim=head_dim,
                kv_latent_dim=kv_latent_dim, dropout=0.1)
            for dim in feature_dims
        ])

        # Precompute which anchor each flat prototype belongs to
        proto_anchor_ids = torch.arange(num_anchors)
        proto_anchor_ids = proto_anchor_ids.unsqueeze(1).expand(
            -1, self.num_fg_classes * protos_per_anchor).flatten()
        self.register_buffer('proto_anchor_ids', proto_anchor_ids)

    # ------------------------------------------------------------------ utils

    def get_prototypes(self, level: int) -> torch.Tensor:
        return getattr(self, f'prototypes_level{level}')

    def _pool_to_grid(
        self, tokens: torch.Tensor, grid_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        B, N, C = tokens.shape
        D, H, W = grid_shape
        x = tokens.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x = F.adaptive_avg_pool3d(x, self.patch_grid)
        x = x.flatten(2).transpose(1, 2)
        return x

    # ------------------------------------------------------------- loss

    def compute_prototype_loss(
        self,
        features_list: List[torch.Tensor],
        labels: torch.Tensor,
        grid_shapes: List[Tuple[int, int, int]],
        balance_weight: float = 0.01,
    ) -> torch.Tensor:
        """Context-aware prototype loss.

        Three components:
        1. Margin contrastive: within each anchor, same-class prototypes
           should be closer to class features than other-class prototypes.
        2. Anchor diversity: different anchors should have orthogonal
           signatures (prevents collapse to a single context).
        3. Anchor utilization balance: encourage uniform usage across anchors.
        """
        total_loss = 0.0
        n_terms = 0

        for level, features in enumerate(features_list):
            patch_features = self._pool_to_grid(features, grid_shapes[level])
            patch_latent = self.kv_downs[level](patch_features)
            patch_latent = F.normalize(patch_latent, dim=-1)

            labels_f = labels.unsqueeze(1).float()
            labels_grid = F.adaptive_avg_pool3d(labels_f, self.patch_grid)
            labels_grid_flat = labels_grid.flatten(2).squeeze(1)

            anchors = F.normalize(self.context_anchors, dim=-1)
            prototypes = F.normalize(self.get_prototypes(level), dim=-1)

            # --- 1. Margin contrastive per class ---
            # Uses soft assignment over K variants (not hard max) so ALL
            # variants receive gradient signal — avoids winner-take-all
            # collapse where only 1/K variants learn and the rest stay random.
            for cls_idx in range(1, self.num_classes):
                mask = ((labels_grid_flat > cls_idx - 0.5) &
                        (labels_grid_flat < cls_idx + 0.5))
                if not mask.any():
                    continue
                cls_latent = patch_latent[mask]  # [M, kv_dim]

                # Positive: soft assignment over K variants
                pos_protos = prototypes[:, cls_idx - 1]  # [A, K, kv_dim]
                pos_raw = torch.einsum('md,akd->mak', cls_latent, pos_protos)
                pos_w = F.softmax(pos_raw / self.temperature, dim=-1)
                pos_sim = (pos_w * pos_raw).sum(dim=-1)  # [M, A]

                # Negative: soft assignment within class, max across classes
                neg_sims = []
                for oc in range(1, self.num_classes):
                    if oc == cls_idx:
                        continue
                    np_ = prototypes[:, oc - 1]
                    ns_raw = torch.einsum('md,akd->mak', cls_latent, np_)
                    ns_w = F.softmax(ns_raw / self.temperature, dim=-1)
                    neg_sims.append((ns_w * ns_raw).sum(dim=-1))
                if neg_sims:
                    neg_sim = torch.stack(neg_sims, dim=-1).max(dim=-1).values
                    loss = F.relu(neg_sim - pos_sim + 0.3).mean()
                    total_loss = total_loss + loss
                    n_terms += 1

            # --- 2. Anchor diversity (relaxed: 0.01 not 0.1) ---
            # Allows anchors to develop meaningful variation rather than
            # staying rigidly orthogonal.
            a_sim = torch.matmul(anchors, anchors.T)
            off = a_sim * (1.0 - torch.eye(self.num_anchors, device=anchors.device))
            total_loss = total_loss + 0.01 * off.pow(2).mean()
            n_terms += 1

            # --- 2b. Within-(anchor,class) variant diversity ---
            # Prevent mode collapse: K variants within each (anchor, class)
            # should be DIVERSE (capture different aspects), not identical.
            # Penalize pairwise cosine similarity > 0.5 between variants.
            K = self.protos_per_anchor
            eye_k = torch.eye(K, device=prototypes.device).unsqueeze(0)  # [1,K,K]
            for cls_idx in range(self.num_fg_classes):
                cls_protos = prototypes[:, cls_idx]  # [A, K, kv_dim]
                sim_kk = torch.matmul(cls_protos, cls_protos.transpose(-2, -1))
                off_k = sim_kk * (1.0 - eye_k)  # exclude self-similarity
                diversity_loss = F.relu(off_k - 0.5).mean()
                total_loss = total_loss + 0.3 * diversity_loss
                n_terms += 1

            # --- 3. Anchor utilization balance ---
            anchor_logits = torch.matmul(patch_latent, anchors.T) / self.temperature
            assign = F.softmax(anchor_logits, dim=-1)  # [B, N, A]
            usage = assign.mean(dim=(0, 1))  # [A]
            uniform = torch.ones_like(usage) / self.num_anchors
            eps = 1e-8
            kl = F.kl_div((usage + eps).log(), uniform,
                          reduction='sum', log_target=False)
            total_loss = total_loss + balance_weight * kl
            n_terms += 1

        return total_loss / max(n_terms, 1)

    # ------------------------------------------------------------- enhance

    def enhance_features(
        self,
        tokens: torch.Tensor,
        level: int,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Context-gated prototype retrieval + MLA cross-attention enhancement.

        1. Pool tokens to grid, project to latent
        2. Soft-assign tokens to context anchors (feature-based, not spatial)
        3. Gate prototype similarity by anchor assignment
        4. Top-k retrieval from gated prototypes
        5. MLA cross-attention + upsample + fusion
        """
        if not self.enabled:
            return tokens

        B, N, C = tokens.shape
        D, H, W = grid_shape
        gD, gH, gW = self.patch_grid
        N_patches = gD * gH * gW

        # Pool to grid
        patch_tokens = self._pool_to_grid(tokens, grid_shape)

        # Latent projection
        patch_latent = self.kv_downs[level](patch_tokens)
        patch_latent = F.normalize(patch_latent, dim=-1)

        # ---- Context anchor soft assignment ----
        anchors = F.normalize(self.context_anchors, dim=-1)  # [A, kv_dim]
        anchor_logits = torch.matmul(patch_latent, anchors.T) / self.temperature
        anchor_assign = F.softmax(anchor_logits, dim=-1)  # [B, N_patches, A]

        # ---- Flatten prototypes and gate by anchor assignment ----
        prototypes = F.normalize(self.get_prototypes(level), dim=-1)
        flat_protos = prototypes.view(-1, self.kv_latent_dim)  # [total, kv_dim]

        # Per-sample average assignment for gating
        query_avg = patch_latent.mean(dim=1)  # [B, kv_dim]
        avg_assign = anchor_assign.mean(dim=1)  # [B, A]

        sim = torch.matmul(query_avg, flat_protos.T)  # [B, total_protos]
        gate = avg_assign[:, self.proto_anchor_ids]  # [B, total_protos]
        gated_sim = sim * gate

        _, topk_indices = gated_sim.topk(self.top_k, dim=-1)  # [B, top_k]

        # ---- MLA cross-attention with top-k prototypes ----
        q = self.q_projs[level](patch_tokens)
        q = q.view(B, N_patches, self.num_heads, self.head_dim)

        k = self.k_ups[level](flat_protos)  # [total, inner_dim]
        v = self.v_ups[level](flat_protos)

        k_sel = k[topk_indices].view(B, self.top_k, self.num_heads, self.head_dim)
        v_sel = v[topk_indices].view(B, self.top_k, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k_sel = k_sel.transpose(1, 2)
        v_sel = v_sel.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_sel.transpose(-2, -1)) * scale
        attn = F.softmax(attn / self.temperature, dim=-1)

        out = torch.matmul(attn, v_sel)
        out = out.transpose(1, 2).contiguous().view(B, N_patches, -1)
        out = self.out_projs[level](out)

        # Upsample to token resolution
        out = out.view(B, gD, gH, gW, C).permute(0, 4, 1, 2, 3)
        orig_dtype = out.dtype
        out = F.interpolate(
            out.float(), size=(D, H, W),
            mode='trilinear', align_corners=False,
        ).to(orig_dtype)
        out = out.flatten(2).transpose(1, 2)

        # MLA cross-attention fusion
        enhanced = self.memory_fusions[level](tokens, out)
        return enhanced

    def forward(
        self,
        features_list: List[torch.Tensor],
        grid_shapes: List[Tuple[int, int, int]],
    ) -> List[torch.Tensor]:
        return [
            self.enhance_features(feat, level, grid_shapes[level])
            for level, feat in enumerate(features_list)
        ]
