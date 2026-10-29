"""基础 3D U-Net，迁移自 ``npc_model``。"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 32, deep_supervision: bool = True) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        self.out_conv_ds2 = nn.Conv3d(base_channels * 2, num_classes, kernel_size=1)
        self.out_conv_ds3 = nn.Conv3d(base_channels * 4, num_classes, kernel_size=1)

    def _pad_to_match(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        dz = ref.shape[2] - x.shape[2]
        dy = ref.shape[3] - x.shape[3]
        dx = ref.shape[4] - x.shape[4]
        pad = [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2,
               dz // 2, dz - dz // 2]
        return nn.functional.pad(x, pad)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x = self.bottleneck(self.pool3(x3))
        x = self.up3(x)
        x3_dec = self.dec3(torch.cat([x3, self._pad_to_match(x, x3)], dim=1))
        x = self.up2(x3_dec)
        x2_dec = self.dec2(torch.cat([x2, self._pad_to_match(x, x2)], dim=1))
        x = self.up1(x2_dec)
        x1_dec = self.dec1(torch.cat([x1, self._pad_to_match(x, x1)], dim=1))
        out_main = self.out_conv(x1_dec)
        if not self.deep_supervision:
            return out_main
        return [out_main, self.out_conv_ds2(x2_dec), self.out_conv_ds3(x3_dec)]


__all__ = ["ConvBlock3D", "UNet3D"]
