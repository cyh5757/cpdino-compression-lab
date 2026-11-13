import math

import pytest
import torch

from cpdino_lab.losses import segmentation_loss


def test_flow_scale_and_binary_foreground_loss():
    prediction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    labels = torch.ones(1, 3, 2, 2)
    loss = segmentation_loss(prediction, labels)
    assert loss.item() == pytest.approx(12.5 + math.log(2))
    loss.backward()
    assert prediction.grad[:, :2].mean() < 0


def test_distillation_matches_outputs_and_detaches_teacher():
    prediction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    teacher = torch.ones_like(prediction, requires_grad=True)
    loss = segmentation_loss(prediction, torch.zeros_like(prediction), teacher, 0.5)
    assert loss.item() == pytest.approx(math.log(2) + 1)
    loss.backward()
    assert teacher.grad is None
    assert prediction.grad is not None


def test_zero_distillation_weight_reproduces_supervised_loss():
    pred = torch.randn(2, 3, 4, 4)
    labels = torch.zeros_like(pred)
    assert torch.equal(segmentation_loss(pred, labels, torch.ones_like(pred), 0), segmentation_loss(pred, labels))


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf')])
def test_rejects_invalid_teacher_weight(weight):
    with pytest.raises(ValueError):
        segmentation_loss(torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2), teacher_weight=weight)


def test_rejects_broadcastable_incorrect_label_shape():
    with pytest.raises(ValueError):
        segmentation_loss(torch.zeros(2, 3, 4, 4), torch.zeros(1, 3, 4, 4))
