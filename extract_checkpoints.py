"""Extract individual task checkpoints from a trained MultiTaskPerceptionModel.

After running:
    python train.py --task multi ...

Run this script to produce classifier.pth, localizer.pth, unet.pth
that each carry the SAME shared encoder — guaranteeing all three tasks
work correctly when assembled back into MultiTaskPerceptionModel.

Usage:
    python extract_checkpoints.py --multitask_ckpt checkpoints/multitask.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


# ── key maps ─────────────────────────────────────────────────────────────────

# MultiTask encoder.blockN.layer_idx → flat _make_features index
# (only layers with parameters: Conv2d and BatchNorm2d)
_ENCODER_TO_FLAT = {
    ("block1", 0): 0,   # Conv 64
    ("block1", 1): 1,   # BN
    ("block2", 0): 4,   # Conv 128
    ("block2", 1): 5,   # BN
    ("block3", 0): 8,   # Conv 256
    ("block3", 1): 9,   # BN
    ("block3", 3): 11,  # Conv 256
    ("block3", 4): 12,  # BN
    ("block4", 0): 15,  # Conv 512
    ("block4", 1): 16,  # BN
    ("block4", 3): 18,  # Conv 512
    ("block4", 4): 19,  # BN
    ("block5", 0): 22,  # Conv 512
    ("block5", 1): 23,  # BN
    ("block5", 3): 25,  # Conv 512
    ("block5", 4): 26,  # BN
}

# MultiTask encoder.blockN.layer_idx → backbone.features.flat_idx (for VGG11Classifier)
_ENCODER_TO_FEATURES = _ENCODER_TO_FLAT  # same mapping

# cls_head positions in MultiTask → backbone.classifier positions in VGG11Classifier
# MultiTask cls_head: Flatten(0), Linear(1), ReLU(2), Dropout(3), Linear(4), ReLU(5), Dropout(6), Linear(7)
# VGG11Classifier backbone.classifier: Linear(0), ReLU(1), Dropout(2), Linear(3), ReLU(4), Dropout(5), Linear(6)
_CLS_HEAD_TO_CLASSIFIER = {"1": "0", "4": "3", "7": "6"}


def _get_sd(path):
    ck = torch.load(path, map_location="cpu")
    return ck.get("state_dict", ck.get("model_state", ck)), ck


def extract_classifier(mt_sd: dict) -> dict:
    """Remap multitask state_dict → VGG11Classifier state_dict."""
    sd = {}

    # encoder.blockN.layer_idx.param → backbone.features.flat_idx.param
    for k, v in mt_sd.items():
        if not k.startswith("encoder."):
            continue
        parts = k.split(".")  # encoder, blockN, layer_idx, param...
        block = parts[1]
        layer = int(parts[2])
        rest  = ".".join(parts[3:])
        flat  = _ENCODER_TO_FEATURES.get((block, layer))
        if flat is not None:
            sd[f"backbone.features.{flat}.{rest}"] = v

    # cls_head.idx.param → backbone.classifier.idx.param
    for k, v in mt_sd.items():
        if not k.startswith("cls_head."):
            continue
        parts = k.split(".")  # cls_head, idx, param...
        old_idx = parts[1]
        new_idx = _CLS_HEAD_TO_CLASSIFIER.get(old_idx)
        if new_idx is not None:
            sd[f"backbone.classifier.{new_idx}.{'.'.join(parts[2:])}"] = v

    return sd


def extract_localizer(mt_sd: dict) -> dict:
    """Remap multitask state_dict → VGG11Localizer state_dict."""
    sd = {}

    # encoder.blockN.layer_idx.param → encoder.flat_idx.param
    for k, v in mt_sd.items():
        if not k.startswith("encoder."):
            continue
        parts = k.split(".")
        block = parts[1]
        layer = int(parts[2])
        rest  = ".".join(parts[3:])
        flat  = _ENCODER_TO_FLAT.get((block, layer))
        if flat is not None:
            sd[f"encoder.{flat}.{rest}"] = v

    # bbox_head.* → regression_head.*
    for k, v in mt_sd.items():
        if k.startswith("bbox_head."):
            sd[k.replace("bbox_head.", "regression_head.")] = v

    return sd


def extract_unet(mt_sd: dict) -> dict:
    """Remap multitask state_dict → VGG11UNet state_dict.
    VGG11UNet shares the same encoder.blockN.* structure — direct copy.
    """
    sd = {}
    for k, v in mt_sd.items():
        if k.startswith(("encoder.", "bottleneck.", "up", "seg_head.")):
            sd[k] = v
    return sd


def verify_load(model: nn.Module, sd: dict, name: str):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_params   = [k for k in missing   if "running" not in k and "num_batches" not in k]
    unexpected_keys  = [k for k in unexpected if "running" not in k and "num_batches" not in k]
    if missing_params:
        print(f"  [{name}] missing param keys: {missing_params[:5]}")
    if unexpected_keys:
        print(f"  [{name}] unexpected keys:    {unexpected_keys[:5]}")
    print(f"  [{name}] loaded {len(sd)} keys — OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multitask_ckpt", default="checkpoints/multitask.pth")
    p.add_argument("--out_dir",        default="checkpoints")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading multitask checkpoint: {args.multitask_ckpt}")
    mt_sd, mt_ck = _get_sd(args.multitask_ckpt)
    epoch = mt_ck.get("epoch", "?")
    print(f"  epoch={epoch}, keys={len(mt_sd)}")

    # ── classifier.pth ──────────────────────────────────────────────────────
    print("\nExtracting classifier.pth ...")
    cls_sd    = extract_classifier(mt_sd)
    cls_model = VGG11Classifier(num_classes=37)
    verify_load(cls_model, cls_sd, "VGG11Classifier")
    torch.save({"state_dict": cls_sd, "epoch": epoch}, out / "classifier.pth")

    # ── localizer.pth ───────────────────────────────────────────────────────
    print("\nExtracting localizer.pth ...")
    loc_sd    = extract_localizer(mt_sd)
    loc_model = VGG11Localizer()
    verify_load(loc_model, loc_sd, "VGG11Localizer")
    torch.save({"state_dict": loc_sd, "epoch": epoch}, out / "localizer.pth")

    # ── unet.pth ────────────────────────────────────────────────────────────
    print("\nExtracting unet.pth ...")
    unet_sd    = extract_unet(mt_sd)
    unet_model = VGG11UNet(num_classes=3)
    verify_load(unet_model, unet_sd, "VGG11UNet")
    torch.save({"state_dict": unet_sd, "epoch": epoch}, out / "unet.pth")

    print(f"\nDone. Files saved to {out}/")
    print("  classifier.pth")
    print("  localizer.pth")
    print("  unet.pth")


if __name__ == "__main__":
    main()
