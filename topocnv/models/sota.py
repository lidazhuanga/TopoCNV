"""SOTA comparison model definitions and registry。"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import AttentionUnet, DynUNet, SegResNet, SwinUNETR, UNETR

SUPPORTED_MODELS = [
    "nnunet", "segresnet", "attention_unet", "unetr", "swin_unetr", "nnformer",
]


class _ConvTransformerBlock(nn.Module):
    def __init__(self, in_ch: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch, affine=True), nn.GELU())
        self.norm1 = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(in_ch)
        hidden = int(in_ch * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(in_ch, hidden), nn.GELU(),
                                 nn.Linear(hidden, in_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x + self.conv(x)
        wd, wh, ww = self.window_size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x.shape[2:]
        nD, nH, nW = Dp // wd, Hp // wh, Wp // ww
        x = x.view(B, C, nD, wd, nH, wh, nW, ww)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(-1, C, wd * wh * ww).transpose(1, 2)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        x = x + self.mlp(self.norm2(x))
        x = x.transpose(1, 2).view(B, nD, nH, nW, C, wd, wh, ww)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, Dp, Hp, Wp)
        if pad_d or pad_h or pad_w:
            x = x[:, :, :D, :H, :W]
        return x


class _nnFormerEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_heads: int = 4):
        super().__init__()
        self.conv_trans = _ConvTransformerBlock(out_ch, num_heads=num_heads)
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_trans(self.down(x))


class _nnFormerDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_heads: int = 4):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.GELU())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            diff = [s2 - s1 for s1, s2 in zip(x.shape[2:], skip.shape[2:])]
            x = F.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                          diff[1] // 2, diff[1] - diff[1] // 2,
                          diff[0] // 2, diff[0] - diff[0] // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class NNFormer3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True), nn.GELU())
        self.enc1 = _nnFormerEncoderBlock(c, c * 2, num_heads=2)
        self.enc2 = _nnFormerEncoderBlock(c * 2, c * 4, num_heads=4)
        self.enc3 = _nnFormerEncoderBlock(c * 4, c * 8, num_heads=4)
        self.bottleneck = _ConvTransformerBlock(c * 8, num_heads=4)
        self.dec3 = _nnFormerDecoderBlock(c * 8, c * 4, c * 4, num_heads=4)
        self.dec2 = _nnFormerDecoderBlock(c * 4, c * 2, c * 2, num_heads=4)
        self.dec1 = _nnFormerDecoderBlock(c * 2, c, c, num_heads=2)
        self.out_conv = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        b = self.bottleneck(x3)
        return self.out_conv(self.dec1(self.dec2(self.dec3(b, x2), x1), x0))


def build_sota_model(model_name: str, in_channels: int = 1,
                     num_classes: int = 2) -> nn.Module:
    """按历史 SOTA 训练入口的固定参数构建模型。"""
    if model_name == "nnunet":
        return DynUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
            kernel_size=[[3, 3, 3]] * 6,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2],
                     [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 5,
            filters=[32, 64, 128, 256, 320, 320], res_block=True)
    if model_name == "segresnet":
        return SegResNet(spatial_dims=3, in_channels=in_channels,
                         out_channels=num_classes, init_filters=16,
                         blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                         dropout_prob=0.0)
    if model_name == "attention_unet":
        return AttentionUnet(spatial_dims=3, in_channels=in_channels,
                             out_channels=num_classes,
                             channels=(16, 32, 64, 128, 256),
                             strides=(2, 2, 2, 2), kernel_size=3,
                             up_kernel_size=3)
    if model_name == "unetr":
        return UNETR(in_channels=in_channels, out_channels=num_classes,
                     img_size=(128, 128, 128), feature_size=16,
                     hidden_size=768, mlp_dim=3072, num_heads=12,
                     proj_type="conv", norm_name="instance", res_block=True,
                     dropout_rate=0.1, spatial_dims=3)
    if model_name == "swin_unetr":
        return SwinUNETR(in_channels=in_channels, out_channels=num_classes,
                         feature_size=48, use_checkpoint=False, spatial_dims=3)
    if model_name == "nnformer":
        return NNFormer3D(in_channels=in_channels, num_classes=num_classes,
                          base_channels=32)
    raise ValueError(f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}")


build_model = build_sota_model

__all__ = [
    "SUPPORTED_MODELS", "_ConvTransformerBlock", "_nnFormerEncoderBlock",
    "_nnFormerDecoderBlock", "NNFormer3D", "build_sota_model", "build_model",
]
