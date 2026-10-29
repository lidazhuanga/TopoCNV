"""ViT-UNETR、MLA 与原型记忆模型。

实现迁移自顶层 ``npc_vit_mla``；模块名变化不影响参数层级与 state_dict key。
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
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


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 32,
                 kv_latent_dim: int = 128, dropout: float = 0.0,
                 use_rope: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.use_rope = use_rope
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.kv_down = nn.Linear(dim, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        if use_rope:
            self.rope_dim = head_dim // 2
            self.nope_dim = head_dim - self.rope_dim
            self.k_rope_proj = nn.Linear(dim, num_heads * self.rope_dim, bias=False)
        else:
            self.rope_dim = 0
            self.nope_dim = head_dim

    def forward(self, x: torch.Tensor, rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        c_kv = self.kv_down(x)
        k_nope = self.k_up(c_kv)
        v = self.v_up(c_kv)
        if self.use_rope:
            k_nope = k_nope.view(B, N, self.num_heads, self.nope_dim)
            k_rope = self.k_rope_proj(x).view(B, N, self.num_heads, self.rope_dim)
            q_nope, q_rope = q[..., :self.nope_dim], q[..., self.nope_dim:]
            if rope_cos is not None and rope_sin is not None:
                q_rope = self._apply_rope(q_rope, rope_cos, rope_sin)
                k_rope = self._apply_rope(k_rope, rope_cos, rope_sin)
            q_full = torch.cat([q_nope, q_rope], dim=-1)
            k_full = torch.cat([k_nope, k_rope], dim=-1)
        else:
            q_full = q
            k_full = k_nope.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)
        q_full = self.q_norm(q_full.transpose(1, 2))
        k_full = self.k_norm(k_full.transpose(1, 2))
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_full, k_full, v, attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.proj_drop(self.out_proj(out))

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class StandardMHAWrapper(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=32, kv_latent_dim=128,
                 dropout=0.0, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.proj_drop(self.out_proj(out))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 32,
                 kv_latent_dim: int = 128, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path_rate: float = 0.0,
                 use_rope: bool = False, use_mla: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        Attn = MultiHeadLatentAttention if use_mla else StandardMHAWrapper
        self.attn = Attn(dim=dim, num_heads=num_heads, head_dim=head_dim,
                         kv_latent_dim=kv_latent_dim, dropout=dropout,
                         use_rope=use_rope)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.drop_path(self.attn(self.norm1(x), rope_cos, rope_sin))
        return x + self.drop_path(self.ffn(self.norm2(x)))


class ConvStem3D(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128,
                 stem_channels: Tuple[int, int, int] = (32, 64, 128),
                 tokenize_kernel: Tuple[int, int, int] = (2, 4, 4),
                 max_grid_size: Tuple[int, int, int] = (16, 16, 16)):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_grid = max_grid_size
        self.stem_channels = stem_channels
        self.stem_0 = nn.Sequential(nn.Conv3d(in_channels, stem_channels[0], 3, 1, 1),
                                    nn.InstanceNorm3d(stem_channels[0]), nn.GELU())
        self.stem_1 = nn.Sequential(nn.Conv3d(stem_channels[0], stem_channels[1], 3, 2, 1),
                                    nn.InstanceNorm3d(stem_channels[1]), nn.GELU())
        self.stem_2 = nn.Sequential(nn.Conv3d(stem_channels[1], stem_channels[2], 3, 2, 1),
                                    nn.InstanceNorm3d(stem_channels[2]), nn.GELU())
        self.tokenize = nn.Conv3d(stem_channels[2], embed_dim,
                                  kernel_size=tokenize_kernel, stride=tokenize_kernel)
        max_tokens = math.prod(max_grid_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        s0 = self.stem_0(x)
        s1 = self.stem_1(s0)
        s2 = self.stem_2(s1)
        tok = self.tokenize(s2)
        grid = tuple(tok.shape[2:])
        N = math.prod(grid)
        tokens = tok.flatten(2).transpose(1, 2)
        if N != self.pos_embed.shape[1]:
            pe = self.pos_embed.transpose(1, 2).view(1, self.embed_dim, *self.max_grid)
            pe = F.interpolate(pe.float(), size=grid, mode="trilinear",
                               align_corners=False).to(tokens.dtype)
            tokens = tokens + pe.flatten(2).transpose(1, 2)
        else:
            tokens = tokens + self.pos_embed[:, :N]
        return tokens, grid, [s0, s1, s2]


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256,
                 patch_size: Tuple[int, int, int] = (8, 8, 8),
                 max_grid_size: Tuple[int, int, int] = (16, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_grid = max_grid_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, math.prod(max_grid_size), embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)
        grid = tuple(x.shape[2:])
        N = math.prod(grid)
        x = x.flatten(2).transpose(1, 2)
        if N != self.pos_embed.shape[1]:
            pe = self.pos_embed.transpose(1, 2).view(1, self.embed_dim, *self.max_grid)
            pe = F.interpolate(pe, size=grid, mode="trilinear", align_corners=False)
            x = x + pe.flatten(2).transpose(1, 2)
        else:
            x = x + self.pos_embed[:, :N]
        return x, grid


class PatchMerging3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Linear(8 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(8 * in_dim)

    def forward(self, x, grid_shape):
        B, _, C = x.shape
        D, H, W = grid_shape
        pd, ph, pw = D % 2, H % 2, W % 2
        if pd or ph or pw:
            x = x.view(B, D, H, W, C)
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd), mode="constant", value=0)
            D, H, W = D + pd, H + ph, W + pw
        else:
            x = x.view(B, D, H, W, C)
        parts = [x[:, dz::2, dy::2, dx::2, :]
                 for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
        # Preserve the historical concatenation order x0,x1,...,x7.
        parts = [x[:, 0::2, 0::2, 0::2, :], x[:, 1::2, 0::2, 0::2, :],
                 x[:, 0::2, 1::2, 0::2, :], x[:, 1::2, 1::2, 0::2, :],
                 x[:, 0::2, 0::2, 1::2, :], x[:, 1::2, 0::2, 1::2, :],
                 x[:, 0::2, 1::2, 1::2, :], x[:, 1::2, 1::2, 1::2, :]]
        x = torch.cat(parts, dim=-1).view(B, -1, 8 * C)
        return self.reduction(self.norm(x)), (D // 2, H // 2, W // 2)


class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.expand = nn.Linear(in_dim, 8 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, grid_shape):
        B, _, _ = x.shape
        D, H, W = grid_shape
        x = self.expand(x).view(B, D, H, W, 2, 2, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, -1, self.out_dim)
        return self.norm(x), (2 * D, 2 * H, 2 * W)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.conv = nn.Sequential(nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1),
                                  nn.InstanceNorm3d(out_ch), nn.GELU(),
                                  nn.Conv3d(out_ch, out_ch, 3, padding=1),
                                  nn.InstanceNorm3d(out_ch), nn.GELU())

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x.float(), size=skip.shape[2:], mode="trilinear",
                              align_corners=False).to(x.dtype)
        return self.conv(torch.cat([x, skip], dim=1))


class CNNDecoder3D(nn.Module):
    def __init__(self, embed_dim: int = 128,
                 stem_channels: Tuple[int, int, int] = (32, 64, 128),
                 tokenize_kernel: Tuple[int, int, int] = (2, 4, 4),
                 num_classes: int = 3):
        super().__init__()
        self.up_2 = _DecoderBlock(embed_dim, stem_channels[2], stem_channels[2],
                                  tokenize_kernel, tokenize_kernel)
        self.up_1 = _DecoderBlock(stem_channels[2], stem_channels[1], stem_channels[1])
        self.up_0 = _DecoderBlock(stem_channels[1], stem_channels[0], stem_channels[0])
        self.out_conv = nn.Conv3d(stem_channels[0], num_classes, 1)

    def forward(self, vit_features, stem_skips, target_shape):
        skip_0, skip_1, skip_2 = stem_skips
        x = self.up_2(vit_features, skip_2)
        x = self.up_1(x, skip_1)
        x = self.up_0(x, skip_0)
        D, H, W = target_shape
        return self.out_conv(x)[:, :, :D, :H, :W]


class ViTUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 embed_dim: int = 128,
                 stem_channels: Tuple[int, int, int] = (32, 64, 128),
                 tokenize_kernel: Tuple[int, int, int] = (2, 4, 4),
                 num_heads: int = 4, head_dim: int = 32,
                 kv_latent_dim: int = 64, encoder_depths: List[int] = [2, 4, 2],
                 decoder_depths: List[int] = [2, 2], mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path_rate: float = 0.0,
                 memory_bank: Optional[nn.Module] = None, use_mla: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.memory_bank = memory_bank
        self.stem_channels = stem_channels
        self.tokenize_kernel = tokenize_kernel
        self.use_mla = use_mla
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        self.conv_stem = ConvStem3D(in_channels, embed_dim, stem_channels, tokenize_kernel)
        total_blocks = sum(encoder_depths) + sum(decoder_depths)
        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, total_blocks)]
        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i in range(3):
            blocks = nn.Sequential(*[
                TransformerEncoderBlock(dims[i], num_heads, head_dim, kv_latent_dim,
                                        mlp_ratio, dropout, dpr[block_idx + j],
                                        use_mla=use_mla)
                for j in range(encoder_depths[i])])
            block_idx += encoder_depths[i]
            self.encoder_stages.append(blocks)
        self.patch_merges = nn.ModuleList([PatchMerging3D(dims[0], dims[1]),
                                            PatchMerging3D(dims[1], dims[2])])
        self.patch_expands = nn.ModuleList([PatchExpanding3D(dims[2], dims[1]),
                                             PatchExpanding3D(dims[1], dims[0])])
        self.decoder_stages = nn.ModuleList()
        for i in range(2):
            blocks = nn.Sequential(*[
                TransformerEncoderBlock(dims[1 - i], num_heads, head_dim, kv_latent_dim,
                                        mlp_ratio, dropout, dpr[block_idx + j],
                                        use_mla=use_mla)
                for j in range(decoder_depths[i])])
            block_idx += decoder_depths[i]
            self.decoder_stages.append(blocks)
        self.final_norm = nn.LayerNorm(dims[0])
        self.cnn_decoder = CNNDecoder3D(embed_dim, stem_channels, tokenize_kernel, num_classes)

    @staticmethod
    def _match_resolution(skip, skip_grid, target_grid):
        if skip_grid == target_grid:
            return skip
        B, _, C = skip.shape
        skip = skip.transpose(1, 2).view(B, C, *skip_grid)
        skip = F.interpolate(skip.float(), size=target_grid, mode="trilinear",
                             align_corners=False).to(skip.dtype)
        return skip.flatten(2).transpose(1, 2)

    def _features(self, x, labels=None):
        target_shape = x.shape[2:]
        tokens, grid0, stem_skips = self.conv_stem(x)
        s0 = self.encoder_stages[0](tokens)
        s1_tokens, grid1 = self.patch_merges[0](s0, grid0)
        s1 = self.encoder_stages[1](s1_tokens)
        s2_tokens, grid2 = self.patch_merges[1](s1, grid1)
        s2 = self.encoder_stages[2](s2_tokens)
        if self.memory_bank is not None:
            if self.training and labels is not None:
                self._proto_loss = self.memory_bank.compute_prototype_loss(
                    [s0, s1, s2], labels, grid_shapes=[grid0, grid1, grid2])
            else:
                self._proto_loss = None
            s0 = self.memory_bank.enhance_features(s0, 0, grid0)
            s1 = self.memory_bank.enhance_features(s1, 1, grid1)
            s2 = self.memory_bank.enhance_features(s2, 2, grid2)
        d1_tokens, grid_d1 = self.patch_expands[0](s2, grid2)
        d1 = self.decoder_stages[0](d1_tokens + self._match_resolution(s1, grid1, grid_d1))
        d2_tokens, grid_d2 = self.patch_expands[1](d1, grid_d1)
        d2 = self.decoder_stages[1](d2_tokens + self._match_resolution(s0, grid0, grid_d2))
        d2 = self.final_norm(d2)
        B, _, C = d2.shape
        return d2.transpose(1, 2).view(B, C, *grid_d2), stem_skips, target_shape

    def forward(self, x, labels=None):
        features, skips, shape = self._features(x, labels)
        return self.cnn_decoder(features, skips, shape)

    def forward_features(self, x, labels=None):
        return self._features(x, labels)


class _MLACrossFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 kv_latent_dim: int, dropout: float = 0.1):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.kv_down = nn.Linear(dim, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim),
                                 nn.Dropout(dropout))

    def forward(self, tokens, memory_out):
        B, N, _ = tokens.shape
        q = self.q_proj(self.norm_q(tokens)).view(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        latent = self.kv_down(self.norm_kv(memory_out))
        k = self.k_up(latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.head_dim ** -0.5)
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))
        enhanced = tokens + out
        return enhanced + self.ffn(enhanced)


class MLAMemoryBank(nn.Module):
    def __init__(self, feature_dims: List[int] = [256, 512, 1024],
                 num_prototypes_per_class: int = 8, num_classes: int = 3,
                 kv_latent_dim: int = 128, num_heads: int = 8,
                 head_dim: int = 32, momentum: float = 0.99,
                 temperature: float = 1.0,
                 patch_grid: Tuple[int, int, int] = (2, 2, 2),
                 top_k: int = 8, retrieval_mode: str = "cross_attention"):
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
        self.top_k = min(top_k, self.num_prototypes)
        self.retrieval_mode = retrieval_mode
        self.register_buffer("enabled", torch.tensor(False, dtype=torch.bool))
        inner_dim = num_heads * head_dim
        for level in range(self.num_levels):
            prototypes = F.normalize(torch.randn(self.num_prototypes, kv_latent_dim), dim=1)
            self.register_parameter(f"prototypes_level{level}", nn.Parameter(prototypes))
        self.q_projs = nn.ModuleList([nn.Linear(d, inner_dim, bias=False) for d in feature_dims])
        self.k_ups = nn.ModuleList([nn.Linear(kv_latent_dim, inner_dim, bias=False)
                                    for _ in feature_dims])
        self.v_ups = nn.ModuleList([nn.Linear(kv_latent_dim, inner_dim, bias=False)
                                    for _ in feature_dims])
        self.out_projs = nn.ModuleList([nn.Linear(inner_dim, d, bias=False) for d in feature_dims])
        self.kv_downs = nn.ModuleList([nn.Linear(d, kv_latent_dim, bias=False)
                                       for d in feature_dims])
        if retrieval_mode == "cross_attention":
            self.memory_fusions = nn.ModuleList([
                _MLACrossFusion(d, num_heads, head_dim, kv_latent_dim, 0.1)
                for d in feature_dims])
        else:
            self.cosine_fuse_projs = nn.ModuleList([nn.Linear(d, d) for d in feature_dims])
            self.cosine_fuse_norms = nn.ModuleList([nn.LayerNorm(d) for d in feature_dims])
            self.memory_fusions = None

    def get_prototypes(self, level):
        return getattr(self, f"prototypes_level{level}")

    def set_prototypes(self, level, prototypes):
        self.get_prototypes(level).data.copy_(prototypes)

    def _pool_to_grid(self, tokens, grid_shape):
        B, _, C = tokens.shape
        x = tokens.view(B, *grid_shape, C).permute(0, 4, 1, 2, 3)
        return F.adaptive_avg_pool3d(x, self.patch_grid).flatten(2).transpose(1, 2)

    def _labels_to_token_grid(self, labels, grid_shape, fg_thresh=0.10):
        oh = F.one_hot(labels.long().clamp(0, self.num_classes - 1), self.num_classes)
        oh = oh.permute(0, 4, 1, 2, 3).float()
        fg = F.adaptive_avg_pool3d(oh, grid_shape)[:, 1:]
        fg_max, fg_arg = fg.max(dim=1)
        return torch.where(fg_max > fg_thresh, fg_arg + 1,
                           torch.zeros_like(fg_arg)).flatten(1)

    @torch.no_grad()
    def initialize_from_features(self, features_list, labels, grid_shapes):
        for level, features in enumerate(features_list):
            patch = self._pool_to_grid(features, grid_shapes[level])
            latent = F.normalize(self.kv_downs[level](patch), dim=2)
            labels_grid = F.adaptive_avg_pool3d(labels.unsqueeze(1).float(), self.patch_grid)
            labels_flat = labels_grid.flatten(2).squeeze(1)
            prototypes = self.get_prototypes(level).clone()
            for cls_idx in range(1, self.num_classes):
                mask = (labels_flat > cls_idx - 0.5) & (labels_flat < cls_idx + 0.5)
                if mask.any():
                    mean = F.normalize(latent[mask].mean(0).unsqueeze(0), dim=1).squeeze(0)
                    start = (cls_idx - 1) * self.num_prototypes_per_class
                    end = cls_idx * self.num_prototypes_per_class
                    for i in range(start, end):
                        noise = torch.randn_like(mean) * 0.1
                        prototypes[i] = F.normalize((mean + noise).unsqueeze(0), dim=1).squeeze(0)
            self.set_prototypes(level, prototypes)

    def compute_balance_loss(self, features_list, grid_shapes):
        total_loss, n_levels = 0.0, 0
        for level, features in enumerate(features_list):
            patch = self._pool_to_grid(features, grid_shapes[level])
            latent = F.normalize(self.kv_downs[level](patch), dim=2)
            prototypes = F.normalize(self.get_prototypes(level), dim=1)
            assignment = F.softmax(torch.matmul(latent, prototypes.t()) / self.temperature, dim=-1)
            usage = assignment.mean(dim=(0, 1))
            uniform = torch.ones_like(usage) / self.num_prototypes
            total_loss = total_loss + F.kl_div((usage + 1e-8).log(), uniform,
                                                reduction="sum", log_target=False)
            n_levels += 1
        return total_loss / max(n_levels, 1)

    def compute_prototype_loss(self, features_list, labels, grid_shapes,
                               balance_weight=0.05, intra_weight=0.3,
                               ortho_weight=0.5, ortho_margin=0.3):
        ce_loss, intra_loss, ortho_loss = 0.0, 0.0, 0.0
        n_ce, n_intra = 0, 0
        K = self.num_prototypes_per_class
        for level, features in enumerate(features_list):
            B, N, _ = features.shape
            latent = F.normalize(self.kv_downs[level](features), dim=-1)
            prototypes = F.normalize(self.get_prototypes(level), dim=1)
            blocks = torch.matmul(latent, prototypes.t()).view(
                B, N, self.num_foreground_classes, K)
            dom = self._labels_to_token_grid(labels, grid_shapes[level])
            class_logits = blocks.max(dim=3).values / self.temperature
            fg_mask = dom > 0
            if fg_mask.any():
                ce_loss = ce_loss + F.cross_entropy(class_logits[fg_mask],
                                                     (dom[fg_mask] - 1).long())
                n_ce += 1
                uniform = torch.full((K,), 1.0 / K, device=blocks.device)
                for cls_idx in range(1, self.num_classes):
                    mask = dom == cls_idx
                    if mask.sum() < 2:
                        continue
                    usage = F.softmax(blocks[..., cls_idx - 1, :][mask] /
                                      self.temperature, dim=-1).mean(0)
                    intra_loss = intra_loss + F.kl_div((usage + 1e-8).log(), uniform,
                                                        reduction="sum", log_target=False)
                    n_intra += 1
            gram = prototypes @ prototypes.t()
            P = gram.shape[0]
            off = gram - torch.diag(torch.diagonal(gram))
            ortho_loss = ortho_loss + F.relu(off.abs() - ortho_margin).sum() / (P * (P - 1))
        ce_loss = ce_loss / max(n_ce, 1)
        intra_loss = intra_loss / max(n_intra, 1)
        ortho_loss = ortho_loss / max(len(features_list), 1)
        balance = self.compute_balance_loss(features_list, grid_shapes)
        return ce_loss + intra_weight * intra_loss + ortho_weight * ortho_loss + balance_weight * balance

    def enhance_features(self, tokens, level, grid_shape):
        if not self.enabled:
            return tokens
        B, _, C = tokens.shape
        D, H, W = grid_shape
        gD, gH, gW = self.patch_grid
        patch = self._pool_to_grid(tokens, grid_shape)
        Np = patch.shape[1]
        prototypes = self.get_prototypes(level)
        latent = F.normalize(self.kv_downs[level](patch), dim=2)
        similarity = torch.matmul(latent.mean(dim=1), prototypes.t())
        _, topk_indices = similarity.topk(self.top_k, dim=1)
        if self.retrieval_mode == "cross_attention":
            q = self.q_projs[level](patch).view(B, Np, self.num_heads,
                                                    self.head_dim).transpose(1, 2)
            k = self.k_ups[level](prototypes)[topk_indices].view(
                B, self.top_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_ups[level](prototypes)[topk_indices].view(
                B, self.top_k, self.num_heads, self.head_dim).transpose(1, 2)
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) *
                             self.head_dim ** -0.5 / self.temperature, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Np, -1)
            out = self.out_projs[level](out)
        else:
            selected = F.normalize(prototypes[topk_indices], dim=2)
            weights = F.softmax(torch.matmul(latent, selected.transpose(1, 2)) /
                                self.temperature, dim=-1)
            values = self.v_ups[level](prototypes)[topk_indices]
            out = self.out_projs[level](torch.matmul(weights, values))
        out = out.view(B, gD, gH, gW, C).permute(0, 4, 1, 2, 3)
        dtype = out.dtype
        out = F.interpolate(out.float(), size=(D, H, W), mode="trilinear",
                            align_corners=False).to(dtype).flatten(2).transpose(1, 2)
        if self.retrieval_mode == "cross_attention":
            return self.memory_fusions[level](tokens, out)
        return tokens + self.cosine_fuse_projs[level](self.cosine_fuse_norms[level](out))

    def forward(self, features_list, grid_shapes):
        return [self.enhance_features(f, i, grid_shapes[i])
                for i, f in enumerate(features_list)]


class ContextAnchorMemoryBank(nn.Module):
    def __init__(self, feature_dims: List[int] = [192, 384, 768],
                 num_classes: int = 3, kv_latent_dim: int = 96,
                 num_anchors: int = 8, protos_per_anchor: int = 4,
                 num_heads: int = 6, head_dim: int = 32,
                 temperature: float = 0.5,
                 patch_grid: Tuple[int, int, int] = (2, 2, 2),
                 top_k: int = 8, momentum: float = 0.99):
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
        self.register_buffer("enabled", torch.tensor(False, dtype=torch.bool))
        inner_dim = num_heads * head_dim
        self.context_anchors = nn.Parameter(F.normalize(torch.randn(num_anchors, kv_latent_dim), dim=-1))
        for level in range(self.num_levels):
            protos = F.normalize(torch.randn(num_anchors, self.num_fg_classes,
                                              protos_per_anchor, kv_latent_dim), dim=-1)
            self.register_parameter(f"prototypes_level{level}", nn.Parameter(protos))
        self.q_projs = nn.ModuleList([nn.Linear(d, inner_dim, bias=False) for d in feature_dims])
        self.k_ups = nn.ModuleList([nn.Linear(kv_latent_dim, inner_dim, bias=False)
                                    for _ in feature_dims])
        self.v_ups = nn.ModuleList([nn.Linear(kv_latent_dim, inner_dim, bias=False)
                                    for _ in feature_dims])
        self.out_projs = nn.ModuleList([nn.Linear(inner_dim, d, bias=False) for d in feature_dims])
        self.kv_downs = nn.ModuleList([nn.Linear(d, kv_latent_dim, bias=False)
                                       for d in feature_dims])
        self.memory_fusions = nn.ModuleList([
            _MLACrossFusion(d, num_heads, head_dim, kv_latent_dim, 0.1)
            for d in feature_dims])
        ids = torch.arange(num_anchors).unsqueeze(1).expand(
            -1, self.num_fg_classes * protos_per_anchor).flatten()
        self.register_buffer("proto_anchor_ids", ids)

    def get_prototypes(self, level):
        return getattr(self, f"prototypes_level{level}")

    def _pool_to_grid(self, tokens, grid_shape):
        B, _, C = tokens.shape
        x = tokens.view(B, *grid_shape, C).permute(0, 4, 1, 2, 3)
        return F.adaptive_avg_pool3d(x, self.patch_grid).flatten(2).transpose(1, 2)

    def compute_prototype_loss(self, features_list, labels, grid_shapes,
                               balance_weight=0.01):
        total_loss, n_terms = 0.0, 0
        for level, features in enumerate(features_list):
            patch = self._pool_to_grid(features, grid_shapes[level])
            latent = F.normalize(self.kv_downs[level](patch), dim=-1)
            labels_grid = F.adaptive_avg_pool3d(labels.unsqueeze(1).float(), self.patch_grid)
            labels_flat = labels_grid.flatten(2).squeeze(1)
            anchors = F.normalize(self.context_anchors, dim=-1)
            prototypes = F.normalize(self.get_prototypes(level), dim=-1)
            for cls_idx in range(1, self.num_classes):
                mask = (labels_flat > cls_idx - 0.5) & (labels_flat < cls_idx + 0.5)
                if not mask.any():
                    continue
                cls_latent = latent[mask]
                pos = prototypes[:, cls_idx - 1]
                raw = torch.einsum("md,akd->mak", cls_latent, pos)
                pos_sim = (F.softmax(raw / self.temperature, dim=-1) * raw).sum(-1)
                neg_sims = []
                for other in range(1, self.num_classes):
                    if other == cls_idx:
                        continue
                    neg = prototypes[:, other - 1]
                    raw = torch.einsum("md,akd->mak", cls_latent, neg)
                    neg_sims.append((F.softmax(raw / self.temperature, dim=-1) * raw).sum(-1))
                if neg_sims:
                    neg_sim = torch.stack(neg_sims, dim=-1).max(dim=-1).values
                    total_loss = total_loss + F.relu(neg_sim - pos_sim + 0.3).mean()
                    n_terms += 1
            a_sim = anchors @ anchors.T
            eye_a = torch.eye(self.num_anchors, device=anchors.device)
            total_loss = total_loss + 0.01 * (a_sim * (1.0 - eye_a)).pow(2).mean()
            n_terms += 1
            K = self.protos_per_anchor
            eye_k = torch.eye(K, device=prototypes.device).unsqueeze(0)
            for cls_idx in range(self.num_fg_classes):
                cp = prototypes[:, cls_idx]
                off = torch.matmul(cp, cp.transpose(-2, -1)) * (1.0 - eye_k)
                total_loss = total_loss + 0.3 * F.relu(off - 0.5).mean()
                n_terms += 1
            assign = F.softmax(torch.matmul(latent, anchors.T) / self.temperature, dim=-1)
            usage = assign.mean(dim=(0, 1))
            uniform = torch.ones_like(usage) / self.num_anchors
            kl = F.kl_div((usage + 1e-8).log(), uniform, reduction="sum", log_target=False)
            total_loss = total_loss + balance_weight * kl
            n_terms += 1
        return total_loss / max(n_terms, 1)

    def enhance_features(self, tokens, level, grid_shape):
        if not self.enabled:
            return tokens
        B, _, C = tokens.shape
        D, H, W = grid_shape
        gD, gH, gW = self.patch_grid
        patch = self._pool_to_grid(tokens, grid_shape)
        Np = patch.shape[1]
        latent = F.normalize(self.kv_downs[level](patch), dim=-1)
        anchors = F.normalize(self.context_anchors, dim=-1)
        assign = F.softmax(torch.matmul(latent, anchors.T) / self.temperature, dim=-1)
        flat = F.normalize(self.get_prototypes(level), dim=-1).view(-1, self.kv_latent_dim)
        sim = torch.matmul(latent.mean(dim=1), flat.T)
        gate = assign.mean(dim=1)[:, self.proto_anchor_ids]
        _, indices = (sim * gate).topk(self.top_k, dim=-1)
        q = self.q_projs[level](patch).view(B, Np, self.num_heads,
                                                self.head_dim).transpose(1, 2)
        k = self.k_ups[level](flat)[indices].view(
            B, self.top_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_ups[level](flat)[indices].view(
            B, self.top_k, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) *
                         self.head_dim ** -0.5 / self.temperature, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Np, -1)
        out = self.out_projs[level](out)
        out = out.view(B, gD, gH, gW, C).permute(0, 4, 1, 2, 3)
        dtype = out.dtype
        out = F.interpolate(out.float(), size=(D, H, W), mode="trilinear",
                            align_corners=False).to(dtype).flatten(2).transpose(1, 2)
        return self.memory_fusions[level](tokens, out)

    def forward(self, features_list, grid_shapes):
        return [self.enhance_features(f, i, grid_shapes[i])
                for i, f in enumerate(features_list)]


__all__ = [
    "DropPath", "MultiHeadLatentAttention", "StandardMHAWrapper",
    "TransformerEncoderBlock", "ConvStem3D", "PatchEmbedding3D",
    "PatchMerging3D", "PatchExpanding3D", "_DecoderBlock", "CNNDecoder3D",
    "ViTUNet3D", "_MLACrossFusion", "MLAMemoryBank", "ContextAnchorMemoryBank",
]
