"""Training entrypoint
"""

# Here we will accomplish four tasks: CLASSIFICATION, LOCALIZATION, SEGMENTATION, MULTI-TASKING

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


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def _ckpt_dir() -> Path:
    """Return checkpoint directory; use /kaggle/working on Kaggle."""
    base = Path("/kaggle/working/checkpoints") if _is_kaggle() else Path("checkpoints")
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_checkpoint(model: nn.Module, filename: str, **kwargs):
    path = _ckpt_dir() / filename
    torch.save({"state_dict": model.state_dict(), **kwargs}, path)
    return path


def _load_pretrained_vgg(path: str) -> VGG11Classifier:
    vgg = VGG11Classifier(num_classes=37)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    vgg.load_state_dict(sd)
    return vgg


def _load_imagenet_weights(features_module: nn.Module):
    """Load pretrained ImageNet VGG11_BN features into the given nn.Sequential."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        from torchvision.models import vgg11_bn, VGG11_BN_Weights
        tv_model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import vgg11_bn
        tv_model = vgg11_bn(pretrained=True)
    features_module.load_state_dict(tv_model.features.state_dict(), strict=True)
    print("[pretrained] Loaded ImageNet VGG11_BN features weights.")


def _init_wandb(args, name: str):
    """Initialise wandb. On Kaggle, pulls API key from Kaggle Secrets automatically."""
    if args.no_wandb:
        wandb.init(project=args.wandb_project, name=name, config=vars(args), mode="disabled")
        return

    # On Kaggle: try to pull the API key from Kaggle Secrets (add secret named WANDB_API_KEY)
    if _is_kaggle() and not os.environ.get("WANDB_API_KEY"):
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
            os.environ["WANDB_API_KEY"] = key
            wandb.login(key=key, relogin=True)
            print("[wandb] Logged in via Kaggle Secrets.")
        except Exception as e:
            print(f"[wandb] Kaggle Secrets login failed ({e}). Running wandb offline.")
            os.environ["WANDB_MODE"] = "offline"

    wandb.init(project=args.wandb_project, name=name, config=vars(args))


def _use_amp(device: torch.device) -> bool:
    """AMP is only safe on CUDA."""
    return device.type == "cuda"


def _autocast_dtype(device: torch.device) -> str:
    """Return a device_type string safe for torch.autocast ('cuda' or 'cpu')."""
    return "cuda" if device.type == "cuda" else "cpu"


# ──────────────────────────────────────────────
# Loss helpers
# ──────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        # Mask out ignored pixels (-1) before one-hot encoding
        valid = (targets >= 0)
        tgt_clamped = targets.clamp(min=0)
        tgt_oh = torch.zeros_like(probs)
        tgt_oh.scatter_(1, tgt_clamped.unsqueeze(1), 1)
        tgt_oh = tgt_oh * valid.unsqueeze(1).float()
        dims = (0, 2, 3)
        inter = (probs * tgt_oh).sum(dim=dims)
        denom = (probs + tgt_oh).sum(dim=dims)
        return 1 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, weight_ce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=-1)
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


def _resize_seg(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape[2:] != masks.shape[1:]:
        logits = nn.functional.interpolate(
            logits, size=masks.shape[1:], mode="bilinear", align_corners=False
        )
    return logits


# ──────────────────────────────────────────────
# Task 1 — Classification
# ──────────────────────────────────────────────

def train_cls(args):
    loaders, class_names = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device = get_device()
    use_amp = _use_amp(device)
    print(f"[device] {device} | AMP={use_amp}")

    model = VGG11Classifier(
        num_classes=37, dropout_p=args.dropout_p, batch_norm=args.batch_norm
    ).to(device)

    if args.pretrained_imagenet:
        _load_imagenet_weights(model.backbone.features)

    # Differential learning rates: pretrained backbone gets lr/10
    backbone_params = list(model.backbone.features.parameters())
    head_params     = list(model.backbone.classifier.parameters())
    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,     "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    _init_wandb(args, f"cls_{args.run_name}")

    # §2.1 — forward hook on 3rd Conv2d for activation histogram
    _activation_store: list[torch.Tensor] = []
    def _act_hook(module, inp, out):
        _activation_store.clear()
        _activation_store.append(out.detach().cpu())
    _conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    _hook_handle  = _conv_layers[2].register_forward_hook(_act_hook)
    _probe_batch  = next(iter(loaders["val"]))
    _probe_img    = _probe_batch["image"][:8].to(device)

    best_val_f1 = 0.0
    no_improve  = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                loss = criterion(model(images), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                    logits = model(images)
                val_loss += criterion(logits, labels).item()
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        f1      = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        n_train = len(loaders["train"])
        n_val   = len(loaders["val"])
        elapsed = time.time() - t0

        # activation histogram
        log_dict = {
            "epoch": epoch,
            "train/loss": total_loss / n_train,
            "val/loss":   val_loss   / n_val,
            "val/macro_f1": f1,
            "epoch_time_s": elapsed,
        }
        model.eval()
        with torch.no_grad():
            model(_probe_img)
        if _activation_store:
            log_dict["activations/conv3"] = wandb.Histogram(
                _activation_store[0].flatten().numpy()
            )
        wandb.log(log_dict)
        print(f"[Cls] Epoch {epoch:3d} | loss={total_loss/n_train:.4f}"
              f" | val_loss={val_loss/n_val:.4f} | val_f1={f1:.4f} | {elapsed:.1f}s")

        if f1 > best_val_f1:
            best_val_f1 = f1
            no_improve  = 0
            save_checkpoint(model, "vgg11_cls.pth", epoch=epoch, f1=f1)
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"[Cls] Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs)")
            break

    _hook_handle.remove()
    wandb.finish()
    return model


# ──────────────────────────────────────────────
# Task 2 — Localization
# ──────────────────────────────────────────────

def _iou_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample IoU for [B, 4] boxes in (cx, cy, w, h) format."""
    px1 = pred[:, 0] - pred[:, 2] / 2;  py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2;  py2 = pred[:, 1] + pred[:, 3] / 2
    tx1 = target[:, 0] - target[:, 2] / 2;  ty1 = target[:, 1] - target[:, 3] / 2
    tx2 = target[:, 0] + target[:, 2] / 2;  ty2 = target[:, 1] + target[:, 3] / 2
    ix1 = torch.max(px1, tx1);  iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2);  iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    return inter / (pa + ta - inter + eps)


def train_loc(args):
    loaders, _ = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device  = get_device()
    use_amp = _use_amp(device)
    print(f"[device] {device} | AMP={use_amp}")

    pretrained_vgg = None
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        pretrained_vgg = _load_pretrained_vgg(args.pretrained_cls)

    model = VGG11Localizer(
        pretrained_vgg=pretrained_vgg,
        freeze_encoder=(args.freeze_encoder == "full"),
    ).to(device)

    if args.pretrained_imagenet and pretrained_vgg is None:
        _load_imagenet_weights(model.encoder)

    # Differential LR: encoder lower, head higher
    encoder_params = list(model.encoder.parameters())
    head_params    = list(model.regression_head.parameters())
    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": head_params,    "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    iou_crit  = IoULoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    _init_wandb(args, f"loc_{args.run_name}")

    best_acc50 = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        # Keep frozen encoder's BN in eval mode so running stats don't drift
        # from the classifier.pth values used at MultiTask inference time
        if args.freeze_encoder == "full":
            for m in model.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
        total_loss = 0.0

        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            bboxes = batch["bbox"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                pred = model(images)
                # IoU loss + SmoothL1 auxiliary (stabilises early-training gradients)
                loss = iou_crit(pred, bboxes) + 0.1 * nn.functional.smooth_l1_loss(pred, bboxes)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # Validation: compute per-sample IoU → Acc@0.5 and Acc@0.75 (matches grader)
        model.eval()
        all_ious = []
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device, non_blocking=True)
                bboxes = batch["bbox"].to(device, non_blocking=True)
                with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                    pred = model(images)
                all_ious.append(_iou_per_sample(pred.float(), bboxes.float()).cpu())

        all_ious = torch.cat(all_ious)
        acc50    = (all_ious >= 0.50).float().mean().item()
        acc75    = (all_ious >= 0.75).float().mean().item()
        mean_iou = all_ious.mean().item()

        n_train = len(loaders["train"])
        elapsed = time.time() - t0

        wandb.log({
            "epoch":        epoch,
            "train/loss":   total_loss / n_train,
            "val/mean_iou": mean_iou,
            "val/acc@0.5":  acc50,
            "val/acc@0.75": acc75,
            "epoch_time_s": elapsed,
        })
        print(f"[Loc] Epoch {epoch:3d} | loss={total_loss/n_train:.4f}"
              f" | mIoU={mean_iou:.4f} | Acc@0.5={acc50:.4f} | Acc@0.75={acc75:.4f} | {elapsed:.1f}s")

        if acc50 > best_acc50:
            best_acc50 = acc50
            no_improve = 0
            save_checkpoint(model, "localization.pth", epoch=epoch, acc50=acc50, acc75=acc75)
            print(f"[Loc] ** best Acc@0.5={acc50:.4f} saved **")
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"[Loc] Early stopping at epoch {epoch}")
            break

    wandb.finish()
    return model


# ──────────────────────────────────────────────
# Task 3 — Segmentation
# ──────────────────────────────────────────────

def train_seg(args):
    loaders, _ = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device  = get_device()
    use_amp = _use_amp(device)
    print(f"[device] {device} | AMP={use_amp}")

    pretrained_vgg = None
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        pretrained_vgg = _load_pretrained_vgg(args.pretrained_cls)

    model = VGG11UNet(
        num_classes=3,
        pretrained_vgg=pretrained_vgg,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedSegLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    _init_wandb(args, f"seg_{args.freeze_encoder}_{args.run_name}")

    _seg_viz_batch = None
    best_dice  = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                logits = _resize_seg(model(images), masks)
                loss   = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0

        model.eval()
        val_loss = val_dice = val_px = 0.0
        with torch.no_grad():
            for i, batch in enumerate(loaders["val"]):
                images = batch["image"].to(device, non_blocking=True)
                masks  = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                    logits = _resize_seg(model(images), masks)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits, masks)
                val_px   += (logits.argmax(1) == masks).float().mean().item()
                if i == 0 and _seg_viz_batch is None:
                    _seg_viz_batch = (images[:5].cpu(), masks[:5].cpu())

        n_train = len(loaders["train"])
        n_val   = len(loaders["val"])
        log_dict = {
            "epoch":         epoch,
            "train/loss":    total_loss / n_train,
            "val/loss":      val_loss   / n_val,
            "val/dice":      val_dice   / n_val,
            "val/pixel_acc": val_px     / n_val,
            "epoch_time_s":  elapsed,
        }

        # §2.6 — log 5 sample segmentation images every 5 epochs
        if epoch % 5 == 0 and _seg_viz_batch is not None:
            viz_imgs, viz_masks = _seg_viz_batch
            with torch.no_grad():
                viz_logits = _resize_seg(model(viz_imgs.to(device)), viz_masks.to(device))
            pred_masks = viz_logits.argmax(1).cpu()
            _TRIMAP_RGB = torch.tensor(
                [[255, 100, 100], [100, 100, 255], [100, 255, 100]], dtype=torch.uint8
            )
            seg_images = []
            for j in range(len(viz_imgs)):
                img_np = viz_imgs[j].permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225])
                          + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                gt_col = _TRIMAP_RGB[viz_masks[j].clamp(0, 2)].numpy()
                pr_col = _TRIMAP_RGB[pred_masks[j].clamp(0, 2)].numpy()
                row = np.concatenate([img_np, gt_col, pr_col], axis=1)
                seg_images.append(wandb.Image(row, caption=f"orig|gt|pred [{j}]"))
            log_dict["seg/val_samples"] = seg_images

        wandb.log(log_dict)
        print(f"[Seg] Epoch {epoch:3d} | loss={total_loss/n_train:.4f}"
              f" | dice={val_dice/n_val:.4f} | px_acc={val_px/n_val:.4f} | {elapsed:.1f}s")

        if val_dice / n_val > best_dice:
            best_dice  = val_dice / n_val
            no_improve = 0
            save_checkpoint(
                model, f"segmentation_{args.freeze_encoder}.pth",
                epoch=epoch, dice=best_dice,
            )
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"[Seg] Early stopping at epoch {epoch}")
            break

    wandb.finish()
    return model


# ──────────────────────────────────────────────
# Task 4 — Multi-task
# ──────────────────────────────────────────────

def train_multi(args):
    loaders, _ = get_dataloader(
        args.data_root, args.batch_size, args.image_size, args.num_workers
    )
    device  = get_device()
    use_amp = _use_amp(device)
    print(f"[device] {device} | AMP={use_amp}")

    pretrained_vgg = None
    if args.pretrained_cls and os.path.exists(args.pretrained_cls):
        pretrained_vgg = _load_pretrained_vgg(args.pretrained_cls)

    model = MultiTaskPerceptionModel(
        num_breeds=37, seg_classes=3,
        dropout_p=args.dropout_p,
        pretrained_vgg=pretrained_vgg,
        load_pretrained=False,  # skip Drive download during training
    ).to(device)

    if args.pretrained_imagenet and pretrained_vgg is None:
        # Load ImageNet weights into shared encoder blocks
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            from torchvision.models import vgg11_bn, VGG11_BN_Weights
            tv = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import vgg11_bn
            tv = vgg11_bn(pretrained=True)
        # Rebuild flat features → encoder.blockN mapping
        _feat_to_block = {
            0: ("block1", 0), 1: ("block1", 1),
            4: ("block2", 0), 5: ("block2", 1),
            8:  ("block3", 0), 9:  ("block3", 1),
            11: ("block3", 3), 12: ("block3", 4),
            15: ("block4", 0), 16: ("block4", 1),
            18: ("block4", 3), 19: ("block4", 4),
            22: ("block5", 0), 23: ("block5", 1),
            25: ("block5", 3), 26: ("block5", 4),
        }
        enc_sd = {}
        for k, v in tv.features.state_dict().items():
            idx = int(k.split(".")[0])
            if idx in _feat_to_block:
                block, layer = _feat_to_block[idx]
                enc_sd[f"{block}.{layer}.{'.'.join(k.split('.')[1:])}"] = v
        model.encoder.load_state_dict(enc_sd, strict=False)
        print("[pretrained] Loaded ImageNet weights into shared encoder.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cls_crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    iou_crit  = IoULoss()
    seg_crit  = CombinedSegLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    W_CLS, W_LOC, W_SEG = 1.0, 1.0, 1.0

    _init_wandb(args, f"multi_{args.run_name}")

    best_metric = 0.0
    no_improve  = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        t_loss = t_cls = t_loc = t_seg = 0.0

        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            bboxes = batch["bbox"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                out      = model(images)
                seg_pred = _resize_seg(out["segmentation"], masks)
                l_cls    = cls_crit(out["classification"], labels)
                l_loc    = iou_crit(out["localization"],   bboxes)
                l_seg    = seg_crit(seg_pred,              masks)
                loss     = W_CLS * l_cls + W_LOC * l_loc + W_SEG * l_seg

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            t_cls  += l_cls.item()
            t_loc  += l_loc.item()
            t_seg  += l_seg.item()

        scheduler.step()
        elapsed = time.time() - t0

        model.eval()
        all_preds, all_labels_v = [], []
        val_iou = val_dice = val_px = val_loss = 0.0

        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                bboxes = batch["bbox"].to(device, non_blocking=True)
                masks  = batch["mask"].to(device, non_blocking=True)

                with torch.autocast(device_type=_autocast_dtype(device), enabled=use_amp):
                    out      = model(images)
                    seg_pred = _resize_seg(out["segmentation"], masks)
                    l_cls    = cls_crit(out["classification"], labels)
                    l_loc    = iou_crit(out["localization"],   bboxes)
                    l_seg    = seg_crit(seg_pred,              masks)

                val_loss += (W_CLS * l_cls + W_LOC * l_loc + W_SEG * l_seg).item()
                val_iou  += (1 - l_loc).item()
                val_dice += dice_score(seg_pred, masks)
                val_px   += (seg_pred.argmax(1) == masks).float().mean().item()
                all_preds.extend(out["classification"].argmax(1).cpu().tolist())
                all_labels_v.extend(labels.cpu().tolist())

        n_t = len(loaders["train"])
        n_v = len(loaders["val"])
        f1  = f1_score(all_labels_v, all_preds, average="macro", zero_division=0)

        wandb.log({
            "epoch":             epoch,
            "train/total_loss":  t_loss   / n_t,
            "train/cls_loss":    t_cls    / n_t,
            "train/loc_loss":    t_loc    / n_t,
            "train/seg_loss":    t_seg    / n_t,
            "val/total_loss":    val_loss / n_v,
            "val/cls_f1":        f1,
            "val/loc_iou":       val_iou  / n_v,
            "val/seg_dice":      val_dice / n_v,
            "val/seg_px_acc":    val_px   / n_v,
            "epoch_time_s":      elapsed,
        })
        print(f"[Multi] Epoch {epoch:3d} | loss={t_loss/n_t:.4f}"
              f" | f1={f1:.4f} | iou={val_iou/n_v:.4f} | dice={val_dice/n_v:.4f} | {elapsed:.1f}s")

        combined = f1 + val_iou / n_v + val_dice / n_v
        if combined > best_metric:
            best_metric = combined
            no_improve  = 0
            save_checkpoint(model, "multitask.pth", epoch=epoch)
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"[Multi] Early stopping at epoch {epoch}")
            break

    wandb.finish()
    return model


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       choices=["cls", "loc", "seg", "multi"], default="cls")
    p.add_argument("--data_root",  default="data/pets")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout_p",  type=float, default=0.5)
    p.add_argument("--image_size", type=int,   default=224)
    p.add_argument("--num_workers",type=int,   default=2)
    p.add_argument("--freeze_encoder", choices=["none", "partial", "full"], default="none")
    p.add_argument("--pretrained_cls", default=None)
    p.add_argument("--wandb_project",  default="da6401-assignment2")
    p.add_argument("--run_name",   default="run")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--download",   action="store_true")
    p.add_argument("--early_stop", type=int, default=15,
                   help="Stop if no val improvement for N epochs (0 = disabled)")
    p.add_argument("--batch_norm", action="store_true", default=True,
                   help="Use BatchNorm in VGG11 backbone")
    p.add_argument("--no-batch_norm", dest="batch_norm", action="store_false")
    p.add_argument("--pretrained_imagenet", action="store_true", default=False,
                   help="Init backbone features with pretrained ImageNet VGG11_BN weights")
    p.add_argument("--no_wandb", action="store_true", default=False,
                   help="Disable wandb logging (auto-disabled when WANDB_API_KEY is absent)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    if args.download:
        maybe_download(args.data_root)
    dispatch = {"cls": train_cls, "loc": train_loc, "seg": train_seg, "multi": train_multi}
    dispatch[args.task](args)
