"""Segmentation model
"""

import torch
import torch.nn as nn
from models.vgg11 import _make_features, _VGG11_CFG


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    # Transpose-conv upsample -> concat skip -> double conv.

    def __init__(self, in_ch: int, skip_ch, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# Encoding to show intermediate skip tensors

class VGG11Encoder(nn.Module):
    def __init__(self, pretrained_vgg=None):
        super().__init__()
        if pretrained_vgg is not None:
            feats = list(pretrained_vgg.features.children())
        else:
            feats = list(_make_features(_VGG11_CFG, batch_norm=True).children())

        pool_indices = [i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d)]

        def _block(start, end):
            return nn.Sequential(*feats[start:end + 1])

        self.block1 = _block(0, pool_indices[0])
        self.block2 = _block(pool_indices[0] + 1, pool_indices[1])
        self.block3 = _block(pool_indices[1] + 1, pool_indices[2])
        self.block4 = _block(pool_indices[2] + 1, pool_indices[3])
        self.block5 = _block(pool_indices[3] + 1, pool_indices[4])

    def forward(self, x: torch.Tensor):
        e1 = self.block1(x)
        e2 = self.block2(e1)
        e3 = self.block3(e2)
        e4 = self.block4(e3)
        e5 = self.block5(e4)
        return e1, e2, e3, e4, e5


# Now its model for segmentation task
class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3,
                 pretrained_vgg=None, freeze_encoder: str = "none"):
        super().__init__()
        self.encoder = VGG11Encoder(pretrained_vgg)

        if freeze_encoder == "full":
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif freeze_encoder == "partial":
            for p in self.encoder.block1.parameters():
                p.requires_grad = False
            for p in self.encoder.block2.parameters():
                p.requires_grad = False
            for p in self.encoder.block3.parameters():
                p.requires_grad = False

        self.bottleneck = _double_conv(512, 1024)
        self.up5 = UpBlock(1024, 512, 512)
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        e1, e2, e3, e4, e5 = self.encoder(x)
        d = self.bottleneck(e5)
        d = self.up5(d, e5)
        d = self.up4(d, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        return self.seg_head(d)
