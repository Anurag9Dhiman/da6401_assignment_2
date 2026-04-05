"""Unified multi-task model
"""

import torch
import torch.nn as nn
import gdown
from models.layers import CustomDropout
from models.segmentation import VGG11Encoder, UpBlock, _double_conv


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 dropout_p: float = 0.5, pretrained_vgg=None):
        super().__init__()

        classifier_path = "checkpoints/classifier.pth"
        localizer_path  = "checkpoints/localizer.pth"
        unet_path       = "checkpoints/unet.pth"
        gdown.download(id="1El_SmZBM-K-gr_pyIVL0mO5NTc7X-FTt", output=classifier_path, quiet=False)
        gdown.download(id="1d0-p1Ld7IkQkWkqp-n7lZuFT2tCWBPA_", output=localizer_path,  quiet=False)
        gdown.download(id="1_YX4arh42XNghcIVsu3jxZpyaXIE2nSY", output=unet_path,       quiet=False)

        # encoder being shared
        self.encoder = VGG11Encoder(pretrained_vgg)
        self.avgpool1 = nn.AdaptiveAvgPool2d((7, 7))

        # the utilized classification head
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_breeds),
        )

        # To do bounding-box for regression head
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # We are using following segmentation head (i.e., U-Net decoder)
        self.bottleneck = _double_conv(512, 1024)
        self.up5 = UpBlock(1024, 512, 512)
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        self._init_heads()

    def _init_heads(self):
        for m in list(self.cls_head.modules()) + list(self.bbox_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        e1, e2, e3, e4, e5 = self.encoder(x)

        pooled = self.avgpool1(e5)
        cls_logits = self.cls_head(pooled)
        bbox_coords = self.bbox_head(pooled)

        d = self.bottleneck(e5)
        d = self.up5(d, e5)
        d = self.up4(d, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        seg_logits = self.seg_head(d)

        if seg_logits.shape[2:] != x.shape[2:]:
            seg_logits = torch.nn.functional.interpolate(
                seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return {
            "classification": cls_logits,
            "localization": bbox_coords,
            "segmentation": seg_logits,
        }
