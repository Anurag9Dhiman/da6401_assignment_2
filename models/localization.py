"""Localization modules
"""

import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import _make_features, _VGG11_CFG


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, pretrained_vgg=None, freeze_encoder: bool = False):
        super().__init__()

        if pretrained_vgg is not None:
            self.encoder = pretrained_vgg.features
            self.avgpool = pretrained_vgg.avgpool
        else:
            self.encoder = _make_features(_VGG11_CFG, batch_norm=True)
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head will output (x_c, y_c, w, h)
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # restrict output in [0,1]
        )
        self._init_head()

    def _init_head(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        return self.regression_head(feat)
