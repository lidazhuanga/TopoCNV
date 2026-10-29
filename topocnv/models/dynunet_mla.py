"""DynUNet-MLA backbone，迁移自 ``npc_nnunet_mla``。"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet

from .vit_unetr_mla import MultiHeadLatentAttention, StandardMHAWrapper

FILTERS = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
LOCAL_MLA_LEVELS = [2, 3]
GLOBAL_MLA_LEVELS = [4, 5]
MEMORY_LEVELS = {2: 0, 3: 1, 4: 2}
MEMORY_FEATURE_DIMS = [FILTERS[2], FILTERS[3], FILTERS[4]]


def windowed_attention(attn: nn.Module, h: torch.Tensor,
                       grid_shape: Tuple[int, int, int],
                       window_size: Tuple[int, int, int]) -> torch.Tensor:
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
    def __init__(self, dim: int, num_heads: int = 6, head_dim: int = 32,
                 kv_latent_dim: int = 96,
                 window_size: Optional[Tuple[int, int, int]] = None,
                 use_mla: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        Attn = MultiHeadLatentAttention if use_mla else StandardMHAWrapper
        self.attn = Attn(dim=dim, num_heads=num_heads, head_dim=head_dim,
                         kv_latent_dim=kv_latent_dim, dropout=dropout)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        h = self.norm(t)
        if self.window_size is None:
            h = self.attn(h)
        else:
            h = windowed_attention(self.attn, h, (D, H, W), self.window_size)
        return (t + h).transpose(1, 2).view(B, C, D, H, W)


class DynUNetMLA3D(nn.Module):
    provides_fullres_features = True

    def __init__(self, in_channels: int = 3, num_classes: int = 6,
                 filters: Optional[List[int]] = None, num_heads: int = 6,
                 head_dim: int = 32, kv_latent_dim: int = 96,
                 window_size: Tuple[int, int, int] = (2, 4, 4),
                 use_mla: bool = True, dropout: float = 0.0,
                 memory_bank: Optional[nn.Module] = None,
                 local_mla_levels: Optional[List[int]] = None,
                 global_mla_levels: Optional[List[int]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.memory_bank = memory_bank
        filters = filters or FILTERS
        local_mla_levels = LOCAL_MLA_LEVELS if local_mla_levels is None else local_mla_levels
        global_mla_levels = GLOBAL_MLA_LEVELS if global_mla_levels is None else global_mla_levels
        self.dynunet = DynUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
            kernel_size=[[3, 3, 3]] * len(filters), strides=STRIDES[:len(filters)],
            upsample_kernel_size=[[2, 2, 2]] * (len(filters) - 1),
            filters=filters, res_block=True)
        self.inject = nn.ModuleDict()
        for lv in local_mla_levels:
            self.inject[str(lv)] = MLAInject(
                filters[lv], num_heads, head_dim, kv_latent_dim,
                window_size=window_size, use_mla=use_mla, dropout=dropout)
        for lv in global_mla_levels:
            self.inject[str(lv)] = MLAInject(
                filters[lv], num_heads, head_dim, kv_latent_dim,
                window_size=None, use_mla=use_mla, dropout=dropout)
        self._mem_feats = {}
        self._dec_feats = None
        self._mem_tok0 = None
        self.evidence_gate = None
        self._gate_src = None
        self._register_hooks()

    def _encoder_modules(self) -> List[nn.Module]:
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
                    t_mem = self._enhance_with_memory(
                        t_loc, MEMORY_LEVELS[lv], (D, H, W), coords)
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
        self.dynunet.upsamples[-1].register_forward_hook(
            lambda _m, _i, out: self._stash_dec(out))

    def _stash_dec(self, out):
        self._dec_feats = out
        return out

    def _gate_fuse(self, t_loc, t_mem, grid):
        gate = self._gate_src
        if gate is None:
            return t_mem
        if isinstance(gate, float):
            return gate * t_loc + (1.0 - gate) * t_mem
        g = F.adaptive_avg_pool3d(gate.float(), grid)
        g = g.flatten(2).transpose(1, 2).to(t_loc.dtype)
        return g * t_loc + (1.0 - g) * t_mem

    def _enhance_with_memory(self, tokens, mem_level, grid, coords):
        mb = self.memory_bank
        if hasattr(mb, "get_sem_anchors"):
            sem = self._sem_tokens(grid, tokens.device)
            if sem is not None:
                return mb.enhance_features(tokens, mem_level, grid, sem)
        if coords is not None and hasattr(mb, "set_alpha"):
            return mb.enhance_features(tokens, mem_level, grid, coords)
        return mb.enhance_features(tokens, mem_level, grid)

    def _sem_tokens(self, grid, device):
        src = getattr(self, "_sem_source", None)
        if src is None:
            return None
        return F.adaptive_avg_pool3d(src, grid).flatten(2).transpose(1, 2)

    def _level_coords(self, grid, device):
        info = getattr(self, "_coord_info", None)
        if info is None:
            return None
        origin, vshape, flips = info
        dorig = self._patch_orig
        ppad = self._patch_padded
        axes = []
        for a, g in enumerate(grid):
            pix = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) * (ppad[a] / g)
            axes.append(pix.clamp(0, dorig[a] - 1))
        gz, gy, gx = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        q = torch.stack([gz, gy, gx], dim=-1).view(-1, 3)
        B = origin.shape[0]
        dorig_t = torch.tensor(dorig, device=device, dtype=torch.float32)
        qb = q.unsqueeze(0).expand(B, -1, -1)
        adj = torch.where(flips.unsqueeze(1) > 0.5,
                          (dorig_t - 1 - q).unsqueeze(0).expand(B, -1, -1), qb)
        coords = (origin.float().unsqueeze(1) + adj) / \
                 (vshape.float().unsqueeze(1) - 1).clamp(min=1)
        return coords.clamp(0, 1)

    @staticmethod
    def _pad32(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        D, H, W = x.shape[2:]
        pd, ph, pw = (32 - D % 32) % 32, (32 - H % 32) % 32, (32 - W % 32) % 32
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (D, H, W)

    @staticmethod
    def _crop(x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        D, H, W = shape
        return x[:, :, :D, :H, :W]

    def _maybe_proto_loss(self, labels: Optional[torch.Tensor]):
        if self.memory_bank is None or not self.training or labels is None:
            self._proto_loss = None
            return
        levels = sorted(self._mem_feats)
        feats = [self._mem_feats[l][0] for l in levels]
        grids = [self._mem_feats[l][1] for l in levels]
        self._proto_loss = self.memory_bank.compute_prototype_loss(
            feats, labels, grid_shapes=grids)

    def _stash_coord_info(self, x, xp, coord_info):
        self._patch_orig = tuple(x.shape[2:])
        self._patch_padded = tuple(xp.shape[2:])
        self._coord_info = coord_info
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

    def forward_features(self, x: torch.Tensor,
                         labels: Optional[torch.Tensor] = None, coord_info=None):
        self._mem_feats, self._dec_feats, self._mem_tok0 = {}, None, None
        xp, orig = self._pad32(x)
        self._stash_coord_info(x, xp, coord_info)
        _ = self.dynunet(xp)
        self._maybe_proto_loss(labels)
        return self._crop(self._dec_feats, orig), None, orig


__all__ = [
    "FILTERS", "STRIDES", "LOCAL_MLA_LEVELS", "GLOBAL_MLA_LEVELS",
    "MEMORY_LEVELS", "MEMORY_FEATURE_DIMS", "windowed_attention", "MLAInject",
    "DynUNetMLA3D",
]


def _self_test():
    torch.manual_seed(0)
    model = DynUNetMLA3D(in_channels=3, num_classes=6).eval()
    x = torch.randn(1, 3, 40, 192, 192)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 6, 40, 192, 192) and torch.isfinite(y).all()
    print("DynUNetMLA3D smoke test OK")


if __name__ == "__main__":
    _self_test()
