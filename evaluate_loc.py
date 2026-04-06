"""§2.5 Object Detection: Confidence & IoU — W&B Table logging.

Run after training the localization model:
    python evaluate_loc.py --checkpoint checkpoints/localization.pth \
                           --data_root data/pets --wandb_project da6401-assignment2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import wandb

from data.pets_dataset import get_dataloader
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss


def _iou_single(pred: list[float], gt: list[float]) -> float:
    """Compute IoU for two (xc, yc, w, h) boxes in [0,1] coords."""
    def to_xyxy(b):
        xc, yc, w, h = b
        return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    px1, py1, px2, py2 = to_xyxy(pred)
    gx1, gy1, gx2, gy2 = to_xyxy(gt)
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter + 1e-6
    return inter / union


def _draw_boxes(img_tensor: torch.Tensor, pred_box: list, gt_box: list) -> Image.Image:
    """Return PIL image with GT (green) and pred (red) bounding boxes drawn."""
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    H, W = pil.size[1], pil.size[0]

    def _rect(box, colour):
        xc, yc, bw, bh = box
        x1, y1 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
        x2, y2 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

    _rect(gt_box, "green")
    _rect(pred_box, "red")
    return pil


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, _ = get_dataloader(args.data_root, batch_size=1, num_workers=2)
    model = VGG11Localizer().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt.get("model_state", ckpt)))
    model.eval()

    wandb.init(project=args.wandb_project, name="eval_loc_table", job_type="evaluation")

    columns = ["image", "gt_box", "pred_box", "iou", "confidence", "failure"]
    table = wandb.Table(columns=columns)

    n_logged = 0
    with torch.no_grad():
        for batch in loaders["test"]:
            if n_logged >= args.n_images:
                break
            img = batch["image"][0]       # (3, H, W)
            gt_box = batch["bbox"][0].tolist()

            pred_box_t = model(img.unsqueeze(0).to(device))[0]
            pred_box = pred_box_t.cpu().tolist()

            iou = _iou_single(pred_box, gt_box)
            # confidence = predicted box area (w*h); large confident boxes score high
            # boxes near [0.5,0.5,~1,~1] indicate model uncertainty (defaulting to centre)
            confidence = float(pred_box[2] * pred_box[3])  # predicted w * h
            is_failure = (confidence > 0.5 and iou < 0.3)

            pil = _draw_boxes(img, pred_box, gt_box)
            table.add_data(
                wandb.Image(pil, caption=f"IoU={iou:.3f}"),
                str([f"{v:.3f}" for v in gt_box]),
                str([f"{v:.3f}" for v in pred_box]),
                round(iou, 4),
                round(confidence, 4),
                is_failure,
            )
            n_logged += 1

    wandb.log({"detection/test_table": table})
    wandb.finish()
    print(f"Logged {n_logged} images to W&B detection table.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/localization.pth")
    p.add_argument("--data_root", default="data/pets")
    p.add_argument("--wandb_project", default="da6401-assignment2")
    p.add_argument("--n_images", type=int, default=15, help="Number of test images to log")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
