"""Classification components
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11, CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5,
                 batch_norm: bool = True):
        super().__init__()
        self.backbone = VGG11(
            num_classes=num_classes,
            dropout_p=dropout_p,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        return self.backbone(x)

    @property
    def features(self):
        """Exposing backbone conv for reuse purpose in task 2, 3, 4"""
        return self.backbone.features

    @property
    def avgpool(self):
        return self.backbone.avgpool
