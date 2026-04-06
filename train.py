"""Training entrypoint
"""

# Here we will accomplish four task CLASSIFICATION, LOCALIZATION, SEGMENTATION, MULTI-TASKING

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score

from data.pets_dataset import get_dataloader, maybe_download
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel

# Now to implement helper functions

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_pretrained_vgg(path: str):
    vgg = VGG11Classifier(num_classes=37)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    vgg.load_state_dict(sd)
    return vgg

def save_checkpoint(model, path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **kwargs}, path)

# Implementation for loss helper functions

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        tgt_oh = torch.zeros_like(probs)
        tgt_oh.scatter_(1, targets.unsqueeze(1), 1)
        dims = (0, 2, 3)
        inter = (probs * tgt_oh).sum(dim=dims)
        denom = (probs + tgt_oh).sum(dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

class CombinedSegLoss(nn.Module):
    def __init__(self, weight_ce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice = DiceLoss()
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits, targets):
        return self.w_ce * self.ce(logits, targets) + self.w_dice * self.dice(logits, targets)

def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    smooth = 1.0
    scores = []
    for c in range(logits.shape[1]):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        scores.append(((2 * inter + smooth) / (denom + smooth)).item())
    return float(np.mean(scores))

def _resize_seg(logits, masks):
    if logits.shape[2:] != masks.shape[1:]:
        logits = nn.functional.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
    return logits

# Now we will code for 1st task

def train_cls(args):
    loaders, class_names = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device = get_device()
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p,
                            batch_norm=args.batch_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project=args.wandb_project, name=f"cls_{args.run_name}", config=vars(args))

    # §2.1 — register forward hook on the 3rd Conv2d; use a fixed probe image for fair comparison
    _activation_store: list[torch.Tensor] = []
    def _act_hook(module, inp, out):
        _activation_store.clear()
        _activation_store.append(out.detach().cpu())
    _conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    _hook_handle = _conv_layers[2].register_forward_hook(_act_hook)  # 3rd Conv2d (0-indexed: [2])
    # grab one fixed probe batch from val so both BN-on and BN-off runs use the same input
    _probe_batch = next(iter(loaders["val"]))
    _probe_img = _probe_batch["image"][:8].to(device)

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        n_train, n_val = len(loaders["train"]), len(loaders["val"])
        log_dict = {
            "epoch": epoch,
            "train/loss": total_loss / n_train,
            "val/loss": val_loss / n_val,
            "val/macro_f1": f1,
        }
        # §2.1 — run fixed probe image through model to get consistent activation histogram
        model.eval()
        with torch.no_grad():
            model(_probe_img)
        if _activation_store:
            log_dict["activations/conv3"] = wandb.Histogram(_activation_store[0].flatten().numpy())
        wandb.log(log_dict)
        print(f"[Cls] Epoch {epoch:3d} | train_loss={total_loss/n_train:.4f}"
              f" | val_loss={val_loss/n_val:.4f} | val_f1={f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            save_checkpoint(model, "checkpoints/vgg11_cls.pth", epoch=epoch, f1=f1)

    _hook_handle.remove()
    wandb.finish()
    return model

# Next task is to do Localization

def train_loc(args):
    loaders, _ = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device = get_device()

    pretrained_vgg = None
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        pretrained_vgg = _load_pretrained_vgg(args.pretrained_cls)

    model = VGG11Localizer(
        pretrained_vgg=pretrained_vgg,
        freeze_encoder=(args.freeze_encoder == "full"),
    ).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    iou_crit = IoULoss()

    wandb.init(project=args.wandb_project, name=f"loc_{args.run_name}", config=vars(args))

    best_val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            optimizer.zero_grad()
            loss = iou_crit(model(images), bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                val_iou += (1 - iou_crit(model(images), bboxes)).item()

        n_train, n_val = len(loaders["train"]), len(loaders["val"])
        wandb.log({
            "epoch": epoch,
            "train/loss": total_loss / n_train,
            "val/iou": val_iou / n_val,
        })
        print(f"[Loc] Epoch {epoch:3d} | train_loss={total_loss/n_train:.4f}"
              f" | val_iou={val_iou/n_val:.4f}")

        if val_iou / n_val > best_val_iou:
            best_val_iou = val_iou / n_val
            save_checkpoint(model, "checkpoints/localization.pth", epoch=epoch)
    wandb.finish()
    return model

# Third task is to do Segmentation

def train_seg(args):
    loaders, _ = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device = get_device()

    pretrained_vgg = None
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        pretrained_vgg = _load_pretrained_vgg(args.pretrained_cls)

    model = VGG11UNet(
        num_classes=3,
        pretrained_vgg=pretrained_vgg,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedSegLoss()

    wandb.init(project=args.wandb_project,
               name=f"seg_{args.freeze_encoder}_{args.run_name}", config=vars(args))

    # collect a fixed set of 5 val images for visual logging (§2.6)
    _seg_viz_batch = None

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = _resize_seg(model(images), masks)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        epoch_time = time.time() - t_start

        model.eval()
        val_loss = val_dice = val_px = 0.0
        with torch.no_grad():
            for i, batch in enumerate(loaders["val"]):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = _resize_seg(model(images), masks)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits, masks)
                val_px += (logits.argmax(1) == masks).float().mean().item()
                if i == 0 and _seg_viz_batch is None:
                    _seg_viz_batch = (images[:5].cpu(), masks[:5].cpu())

        n_train, n_val = len(loaders["train"]), len(loaders["val"])
        log_dict = {
            "epoch": epoch,
            "train/loss": total_loss / n_train,
            "val/loss": val_loss / n_val,
            "val/dice": val_dice / n_val,
            "val/pixel_acc": val_px / n_val,
            "epoch_time_s": epoch_time,
        }

        # §2.6 — log 5 sample segmentation images every 5 epochs
        if epoch % 5 == 0 and _seg_viz_batch is not None:
            viz_imgs, viz_masks = _seg_viz_batch
            with torch.no_grad():
                viz_logits = _resize_seg(model(viz_imgs.to(device)), viz_masks.to(device))
            pred_masks = viz_logits.argmax(1).cpu()
            # trimap colour map: 0=fg(red), 1=bg(blue), 2=boundary(green)
            _TRIMAP_RGB = torch.tensor([[255, 100, 100], [100, 100, 255], [100, 255, 100]], dtype=torch.uint8)
            seg_images = []
            for j in range(len(viz_imgs)):
                # de-normalise image for display
                img_np = viz_imgs[j].permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                gt_col = _TRIMAP_RGB[(viz_masks[j].clamp(0, 2))].numpy()
                pr_col = _TRIMAP_RGB[(pred_masks[j].clamp(0, 2))].numpy()
                row = np.concatenate([img_np, gt_col, pr_col], axis=1)
                seg_images.append(wandb.Image(row, caption=f"orig | gt_trimap | pred_trimap [{j}]"))
            log_dict["seg/val_samples"] = seg_images

        wandb.log(log_dict)
        print(f"[Seg] Epoch {epoch:3d} | train_loss={total_loss/n_train:.4f}"
              f" | val_loss={val_loss/n_val:.4f} | val_dice={val_dice/n_val:.4f}"
              f" | val_px_acc={val_px/n_val:.4f} | time={epoch_time:.1f}s")

        if val_dice / n_val > best_dice:
            best_dice = val_dice / n_val
            save_checkpoint(model, f"checkpoints/segmentation_{args.freeze_encoder}.pth",
                            epoch=epoch, dice=best_dice)

    wandb.finish()
    return model

# 4th task is Multi-Task

def train_multi(args):
    loaders, _ = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device = get_device()

    pretrained_vgg = None
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        pretrained_vgg = _load_pretrained_vgg(args.pretrained_cls)

    model = MultiTaskPerceptionModel(
        num_breeds=37, seg_classes=3,
        dropout_p=args.dropout_p,
        pretrained_vgg=pretrained_vgg,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cls_crit = nn.CrossEntropyLoss()
    iou_crit = IoULoss()
    seg_crit = CombinedSegLoss()
    W_CLS, W_LOC, W_SEG = 1.0, 1.0, 1.0

    wandb.init(project=args.wandb_project, name=f"multi_{args.run_name}", config=vars(args))

    best_metric = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = t_cls = t_loc = t_seg = 0.0
        for batch in loaders["train"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            out = model(images)
            seg_pred = _resize_seg(out["segmentation"], masks)
            l_cls = cls_crit(out["classification"], labels)
            l_loc = iou_crit(out["localization"], bboxes)
            l_seg = seg_crit(seg_pred, masks)
            loss = W_CLS * l_cls + W_LOC * l_loc + W_SEG * l_seg
            loss.backward()
            optimizer.step()
            t_loss += loss.item(); t_cls += l_cls.item()
            t_loc += l_loc.item(); t_seg += l_seg.item()
        scheduler.step()

        model.eval()
        all_preds, all_labels_v = [], []
        val_iou = val_dice = val_px = val_loss = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                masks = batch["mask"].to(device)
                out = model(images)
                seg_pred = _resize_seg(out["segmentation"], masks)
                l_cls = cls_crit(out["classification"], labels)
                l_loc = iou_crit(out["localization"], bboxes)
                l_seg = seg_crit(seg_pred, masks)
                val_loss += (W_CLS * l_cls + W_LOC * l_loc + W_SEG * l_seg).item()
                val_iou += (1 - l_loc).item()
                val_dice += dice_score(seg_pred, masks)
                val_px += (seg_pred.argmax(1) == masks).float().mean().item()
                all_preds.extend(out["classification"].argmax(1).cpu().tolist())
                all_labels_v.extend(labels.cpu().tolist())

        n_t, n_v = len(loaders["train"]), len(loaders["val"])
        f1 = f1_score(all_labels_v, all_preds, average="macro", zero_division=0)
        wandb.log({
            "epoch": epoch,
            "train/total_loss": t_loss / n_t,
            "train/cls_loss": t_cls / n_t,
            "train/loc_loss": t_loc / n_t,
            "train/seg_loss": t_seg / n_t,
            "val/total_loss": val_loss / n_v,
            "val/cls_f1": f1,
            "val/loc_iou": val_iou / n_v,
            "val/seg_dice": val_dice / n_v,
            "val/seg_px_acc": val_px / n_v,
        })
        print(f"[Multi] Epoch {epoch:3d} | loss={t_loss/n_t:.4f} | "
              f"f1={f1:.4f} | iou={val_iou/n_v:.4f} | dice={val_dice/n_v:.4f}")

        combined = f1 + val_iou / n_v + val_dice / n_v
        if combined > best_metric:
            best_metric = combined
            save_checkpoint(model, "checkpoints/multitask.pth", epoch=epoch)
    wandb.finish()
    return model


# The entry points for terminal

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cls", "loc", "seg", "multi"], default="cls")
    p.add_argument("--data_root", default="data/pets")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout_p", type=float, default=0.5)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--freeze_encoder", choices=["none", "partial", "full"], default="none")
    p.add_argument("--pretrained_cls", default=None)
    p.add_argument("--wandb_project", default="da6401-assignment2")
    p.add_argument("--run_name", default="run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--batch_norm", action="store_true", default=True,
                   help="Use BatchNorm in VGG11 (§2.1 comparison: pass --no-batch_norm to disable)")
    p.add_argument("--no-batch_norm", dest="batch_norm", action="store_false")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    if args.download:
        maybe_download(args.data_root)
    dispatch = {"cls": train_cls, "loc": train_loc, "seg": train_seg, "multi": train_multi}
    dispatch[args.task](args)
