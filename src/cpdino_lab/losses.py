"""Cellpose task loss and output-level teacher matching."""
import math

import torch
import torch.nn.functional as F


def segmentation_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    teacher: torch.Tensor | None = None,
    teacher_weight: float = 1.0,
) -> torch.Tensor:
    """Labels are [foreground, flow_y, flow_x], after removing the instance-ID plane."""
    if prediction.ndim != 4 or prediction.shape[1] != 3 or prediction.shape != labels.shape:
        raise ValueError('Prediction and labels must have identical [N, 3, H, W] shapes')
    if not math.isfinite(teacher_weight) or teacher_weight < 0:
        raise ValueError('Teacher weight must be finite and nonnegative')
    if teacher is not None and teacher.shape != prediction.shape:
        raise ValueError('Teacher and student output shapes must match')
    prediction, labels = prediction.float(), labels.float()
    loss = F.mse_loss(prediction[:, :2], 5 * labels[:, -2:]) / 2
    loss = loss + F.binary_cross_entropy_with_logits(prediction[:, 2], (labels[:, 0] > .5).float())
    if teacher is not None and teacher_weight > 0:
        target = teacher.detach().float()
        loss = loss + teacher_weight * (
            F.mse_loss(prediction[:, :2], target[:, :2])
            + F.mse_loss(prediction[:, 2], target[:, 2])
        )
    return loss
