"""
Diagnostic script to verify feature and prediction consistency between 
standalone VGG11Localizer and the MultiTaskPerceptionModel.
"""

import torch
import torch.nn as nn
from models.localization import VGG11Localizer
from models.multitask import MultiTaskPerceptionModel
from models.classification import VGG11Classifier

def verify_consistency():
    device = torch.device("cpu")
    print(f"Starting consistency check on {device}...")

    # 1. Create a dummy input
    img = torch.randn(1, 3, 224, 224).to(device)

    # 2. Setup VGG11Classifier (base for both)
    # We initialize with same weights to ensure baseline is identical
    cls_base = VGG11Classifier(num_classes=37).to(device)
    cls_base.eval()

    # 3. Initialize Localizer and MultiTask with the SAME cls_base encoder
    localizer = VGG11Localizer(pretrained_vgg=cls_base.backbone, freeze_encoder=True).to(device)
    # We turn off Drive loading to use our local weights
    mt_model = MultiTaskPerceptionModel(pretrained_vgg=cls_base.backbone, load_pretrained=False).to(device)
    
    # Force identical weights in regression heads (heads are normally randomized at init)
    mt_model.bbox_head.load_state_dict(localizer.regression_head.state_dict())
    
    localizer.eval()
    mt_model.eval()

    # 4. Forward pass
    with torch.no_grad():
        loc_out = localizer(img)
        mt_out = mt_model(img)["localization"]

    # 5. Check outputs
    diff = torch.abs(loc_out - mt_out).max().item()
    print(f"\nMax Prediction Difference: {diff:.8e}")

    if diff < 1e-6:
        print("✅ SUCCESS: Standalone Localizer and MultiTask predictions are identical!")
    else:
        print("❌ FAILURE: Predictions differ. Check the re-mapping in multitask.py or head architectures.")

    # 6. Check individual layer outputs if failure
    if diff >= 1e-6:
        print("\nTracing differences...")
        # Check encoder output
        feat_loc = localizer.encoder(img)
        # For MTPM, encoder returns blocks (e1...e5)
        _, _, _, _, feat_mt = mt_model.encoder(img)
        feat_diff = torch.abs(feat_loc - feat_mt).max().item()
        print(f"Encoder Feature Difference: {feat_diff:.8e}")

if __name__ == "__main__":
    verify_consistency()
