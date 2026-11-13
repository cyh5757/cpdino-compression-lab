"""Notebook-sized training and validation operations."""
import torch
from .losses import segmentation_loss


def train_step(student, optimizer, images, labels, teacher=None, teacher_weight=1.0) -> float:
    if teacher is student:
        raise ValueError('Teacher and student must be separate model instances')
    student.train()
    optimizer.zero_grad(set_to_none=True)
    target = None
    if teacher is not None:
        teacher.eval()
        with torch.no_grad():
            target = teacher(images)[0]
    loss = segmentation_loss(student(images)[0], labels, target, teacher_weight)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite loss; optimizer step was skipped')
    loss.backward()
    gradients = [p.grad for p in student.parameters() if p.grad is not None]
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('Non-finite gradient; optimizer step was skipped')
    optimizer.step()
    return float(loss.detach())


def evaluate(model, batches) -> float:
    """Sample-weighted supervised validation loss over an iterable of tensor batches."""
    training = model.training
    total, count = 0., 0
    model.eval()
    try:
        with torch.inference_mode():
            for images, labels in batches:
                loss = segmentation_loss(model(images)[0], labels)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite validation loss')
                total += float(loss) * len(images)
                count += len(images)
        if not count:
            raise ValueError('Validation batches are empty')
        return total / count
    finally:
        model.train(training)
