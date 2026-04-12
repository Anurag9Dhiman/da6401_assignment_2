#Custom IoU loss


import torch
import torch.nn as nn

class IoULoss(nn.Module):
    # This is iou  loss use for bounding box regression

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none',got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        #Lets compute iou loss between prediction and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # convert (x_c, y_c, w, h) -> (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area + self.eps

        iou = inter_area / union_area
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
