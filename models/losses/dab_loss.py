import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.registry import MODELS


@MODELS.register_module()
class DABLoss(nn.Module):
    """EMASlideLoss converted to the MMDetection loss interface.

    The original YOLO implementation wraps BCEWithLogitsLoss and updates an
    EMA IoU threshold. Here `target` is the dense quality/classification target
    produced by EfficientDecoder.
    """

    def __init__(self,
                 decay=0.999,
                 tau=2000,
                 min_iou=0.2,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.decay_base = decay
        self.tau = tau
        self.min_iou = min_iou
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.updates = 0
        self.iou_mean = 1.0

    def _decay(self):
        return self.decay_base * (1 - math.exp(-self.updates / self.tau))

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                auto_iou=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if auto_iou is None:
            with torch.no_grad():
                pos = target[target > 0]
                auto_iou = pos.mean() if pos.numel() > 0 else target.new_tensor(-1.)

        if self.training and float(auto_iou.detach()) != -1.0:
            self.updates += 1
            d = self._decay()
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())

        slide_iou = max(self.iou_mean, self.min_iou)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        low = target <= slide_iou - 0.1
        mid = (target > slide_iou - 0.1) & (target < slide_iou)
        high = target >= slide_iou
        modulating_weight = (
            low.to(loss.dtype) +
            math.exp(1.0 - slide_iou) * mid.to(loss.dtype) +
            torch.exp(-(target - 1.0)) * high.to(loss.dtype))
        loss = loss * modulating_weight

        if weight is not None:
            weight = weight.to(loss.dtype)
            if weight.dim() == 1 and loss.dim() > 1:
                weight = weight[:, None]

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
