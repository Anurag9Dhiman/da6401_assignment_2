"""§2.4 Inside the Black Box: Feature Map Visualization.

Extracts and logs feature maps from the 1st Conv2d layer and the last Conv2d
layer (before the final MaxPool) of the trained VGG11 classification model.

Run:
    python analyze.py --checkpoint checkpoints/vgg11_cls.pth \
                      --image path/to/dog.jpg --wandb_project da6401-assignment2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.classification import VGG11Classifier


_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def _get_feature_maps(model: VGG11Classifier, img_tensor: torch.Tensor):
    """Return activations from 1st Conv2d and last Conv2d (before final MaxPool)."""
    conv_layers = [m for m in model.backbone.features.modules() if isinstance(m, nn.Conv2d)]
    first_conv = conv_layers[0]
    last_conv = conv_layers[-1]

    activations = {}

    def _make_hook(name):
        def _hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return _hook

    h1 = first_conv.register_forward_hook(_make_hook("first_conv"))
    h2 = last_conv.register_forward_hook(_make_hook("last_conv"))

    with torch.no_grad():
        model(img_tensor)

    h1.remove()
    h2.remove()
    return activations


def _fmaps_to_wandb_images(fmap: torch.Tensor, label: str, n_channels: int = 16):
    """Convert feature map tensor [1, C, H, W] to list of wandb.Image."""
    fmap = fmap[0]  # [C, H, W]
    n = min(n_channels, fmap.shape[0])
    images = []
    for c in range(n):
        ch = fmap[c].numpy()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ch_uint8 = (ch * 255).astype(np.uint8)
        images.append(wandb.Image(ch_uint8, caption=f"{label} ch{c}"))
    return images


def analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG11Classifier(num_classes=37)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    img_np = np.array(Image.open(args.image).convert("RGB"))
    img_tensor = _TRANSFORM(image=img_np)["image"].unsqueeze(0).to(device)

    activations = _get_feature_maps(model, img_tensor)

    wandb.init(project=args.wandb_project, name=f"feature_maps_{Path(args.image).stem}",
               job_type="analysis")

    # log the input image
    orig_disp = (img_np).astype(np.uint8)
    wandb.log({"feature_maps/input_image": wandb.Image(orig_disp, caption="Input Image")})

    # §2.4 — log first conv and last conv feature maps
    first_imgs = _fmaps_to_wandb_images(activations["first_conv"], label="first_conv")
    last_imgs = _fmaps_to_wandb_images(activations["last_conv"], label="last_conv")

    wandb.log({
        "feature_maps/first_conv_layer": first_imgs,
        "feature_maps/last_conv_layer": last_imgs,
    })

    print(f"Logged {len(first_imgs)} first-conv and {len(last_imgs)} last-conv feature maps.")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/vgg11_cls.pth")
    p.add_argument("--image", required=True, help="Path to a dog/cat image")
    p.add_argument("--wandb_project", default="da6401-assignment2")
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
