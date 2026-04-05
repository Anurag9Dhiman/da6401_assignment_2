"""Inference and evaluation
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel

# let's start with preprocessing step

_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

TRIMAP_COLORS = {
    0: (255, 100, 100),
    1: (100, 100, 255),
    2: (100, 255, 100),
}

def preprocess(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    img = np.array(Image.open(image_path).convert("RGB"))
    out = _TRANSFORM(image=img)["image"]   # This is (3, 224, 224)
    return out.unsqueeze(0), img   # (1, 3, 224, 224) raw dimensions

# Visualization helpers
def draw_bbox(img: Image.Image, bbox: list[float], color="red", width: int = 3) -> Image.Image:
    W, H = img.size
    xc, yc, bw, bh = bbox
    x1 = int((xc - bw / 2) * W)
    y1 = int((yc - bh / 2) * H)
    x2 = int((xc + bw / 2) * W)
    y2 = int((yc + bh / 2) * H)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img

def overlay_mask(img: Image.Image, mask: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """blending mask into original image"""
    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, colour in TRIMAP_COLORS.items():
        overlay[mask == cls_idx] = colour
    overlay_pil = Image.fromarray(overlay).resize(img.size, Image.NEAREST)
    return Image.blend(img, overlay_pil, alpha)

# Main function code

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Load class names
    class_names = None
    if args.class_names and Path(args.class_names).exists():
        with open(args.class_names) as f:
            class_names = [l.strip() for l in f.readlines()]

    # preprocess
    tensor, raw_img = preprocess(args.image)
    tensor = tensor.to(device)

    with torch.no_grad():
        out = model(tensor)
        cls_logits = out["classification"]
        bbox_pred = out["localization"]
        seg_logits = out["segmentation"]

    # Do classification
    probs = torch.softmax(cls_logits, dim=1)[0]
    top_k = probs.topk(5)
    top_labels = top_k.indices.cpu().tolist()
    top_probs = top_k.values.cpu().tolist()

    print("\n Classification---")
    for rank, (idx, p) in enumerate(zip(top_labels, top_probs), 1):
        name = class_names[idx] if class_names else f"class_{idx}"
        print(f" #{rank}: {name: <30s} {p*100:.1f}%")

    # Bounding box for images annotation
    bbox = bbox_pred[0].cpu().tolist()
    print(f"\n Bounding Box---")
    print(f" [x_c={bbox[0]:.3f}, y_c={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}]")

    # The segmentation part
    seg_mask = seg_logits[0].argmax(0).cpu().numpy()  # (H, W)

    # Visualization
    orig_pil = Image.fromarray(raw_img).resize((448, 448))

    # Draw bounding box
    vis_bbox = draw_bbox(orig_pil.copy(), bbox, color="red")

    # First overlay mask (resizing it to original size first)
    seg_mask_resized = np.array(
        Image.fromarray(seg_mask.astype(np.uint8)).resize((448, 448), Image.NEAREST)
    )
    vis_seg = overlay_mask(orig_pil.copy(), seg_mask_resized)

    # making code for side by side composite implementation
    composite = Image.new("RGB", (448 * 2, 448))
    composite.paste(vis_bbox, (0, 0))
    composite.paste(vis_seg, (448, 0))

    out_path = Path(args.image).stem + "_inference.jpg"
    composite.save(out_path)
    print(f"\n --saved visualization -> {out_path}")

    # §2.7 — log composite visualization to W&B
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=f"inference_{Path(args.image).stem}", job_type="inference")
        wandb.log({"inference/visualization": wandb.Image(str(out_path), caption=Path(args.image).name)})
        wandb.finish()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--checkpoint", default="checkpoints/multitask.pth")
    p.add_argument("--class_names", default=None, help="Path to class names .txt file")
    p.add_argument("--wandb_project", default=None, help="W&B project name for logging (§2.7)")
    return p.parse_args()

if __name__ == "__main__":
    run_inference(parse_args())
