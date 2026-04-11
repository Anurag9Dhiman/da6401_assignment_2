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
                 dropout_p: float = 0.5, pretrained_vgg=None, load_pretrained: bool = True):
        super().__init__()

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
        # Only load pretrained Drive checkpoints during inference/evaluation,
        # not during training (pretrained_vgg or imagenet init is used instead)
        if load_pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        import os
        os.makedirs("checkpoints", exist_ok=True)

        classifier_path = "checkpoints/classifier.pth"
        localizer_path  = "checkpoints/localizer.pth"
        unet_path       = "checkpoints/unet.pth"

        if not os.path.exists(classifier_path):
            gdown.download(id="1h6j6bOF288ZUAoSiU60jpsQJc86kyBK_", output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="1fZvPjjXXZ2WNjXMsx_iMkp914RoOTslu", output=localizer_path,  quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1SmNpZpswUc0M7AbEbU-LcrdI6gg64tdy", output=unet_path,        quiet=False)

        def _get_sd(path):
            ck = torch.load(path, map_location="cpu")
            return ck.get("state_dict", ck.get("model_state", ck))

        cls_sd  = _get_sd(classifier_path)
        loc_sd  = _get_sd(localizer_path)
        unet_sd = _get_sd(unet_path)

        new_sd = {}

        # backbone.features.N  →  encoder.blockB.M
        # VGG11-BN layout (with BatchNorm): Conv,BN,ReLU,Pool per block
        # Only Conv and BN have parameters (ReLU/Pool are skipped in state_dict)
        _feat_to_block = {
            0: ("block1", 0), 1: ("block1", 1),          # block1: Conv64, BN
            4: ("block2", 0), 5: ("block2", 1),          # block2: Conv128, BN
            8:  ("block3", 0), 9:  ("block3", 1),        # block3: Conv256, BN
            11: ("block3", 3), 12: ("block3", 4),        #         Conv256, BN
            15: ("block4", 0), 16: ("block4", 1),        # block4: Conv512, BN
            18: ("block4", 3), 19: ("block4", 4),        #         Conv512, BN
            22: ("block5", 0), 23: ("block5", 1),        # block5: Conv512, BN
            25: ("block5", 3), 26: ("block5", 4),        #         Conv512, BN
        }

        # 1. Encoder: remap backbone.features.* from classifier.pth
        #    This keeps the encoder consistent with the cls_head weights.
        for k, v in cls_sd.items():
            if k.startswith("backbone.features."):
                parts = k.split(".")  # [backbone, features, idx, param...]
                feat_idx = int(parts[2])
                if feat_idx in _feat_to_block:
                    block_name, layer_idx = _feat_to_block[feat_idx]
                    new_key = f"encoder.{block_name}.{layer_idx}.{'.'.join(parts[3:])}"
                    new_sd[new_key] = v

        # 2. Segmentation decoder: load from unet checkpoint
        for k, v in unet_sd.items():
            if k.startswith(("bottleneck.", "up", "seg_head.")):
                new_sd[k] = v

        # 3. cls_head: backbone.classifier.0/3/6 -> cls_head.1/4/7
        cls_remap = {"0": "1", "3": "4", "6": "7"}
        for k, v in cls_sd.items():
            if k.startswith("backbone.classifier."):
                parts = k.split(".")  # [backbone, classifier, idx, param]
                new_idx = cls_remap.get(parts[2], parts[2])
                new_sd[f"cls_head.{new_idx}.{'.'.join(parts[3:])}"] = v

        # 4. bbox_head: regression_head.* -> bbox_head.*
        for k, v in loc_sd.items():
            if k.startswith("regression_head."):
                new_sd[k.replace("regression_head.", "bbox_head.")] = v

        self.load_state_dict(new_sd, strict=False)

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
