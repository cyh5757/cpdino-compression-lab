import copy

import pytest
import torch
from cpdino_lab.engine import evaluate, train_step
from cpdino_lab.models import build_model


def test_real_dino_forward_restores_rectangular_pixel_grid(small_model):
    pred, style = small_model.eval()(torch.rand(2, 3, 32, 48))
    assert pred.shape == (2, 3, 32, 48)
    assert style.shape == (2, 256)
    assert torch.isfinite(pred).all()


def test_single_channel_input_is_supported(small_model):
    assert small_model(torch.rand(1, 1, 32, 32))[0].shape == (1, 3, 32, 32)


def test_rejects_input_that_would_truncate_output(small_model):
    with pytest.raises(ValueError):
        small_model(torch.rand(1, 3, 33, 32))


def test_student_step_updates_head_and_keeps_teacher_and_patch_fixed(small_model):
    student = small_model
    teacher = copy.deepcopy(student)
    before = student.out.weight.detach().clone()
    teacher_before = teacher.out.weight.detach().clone()
    patch = student.encoder.patch_embed.proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
    loss = train_step(student, optimizer, torch.rand(1, 3, 32, 32), torch.zeros(1, 3, 32, 32), teacher)
    assert loss > 0
    assert not torch.equal(before, student.out.weight)
    assert torch.equal(teacher_before, teacher.out.weight)
    assert all(p.grad is None for p in teacher.parameters())
    assert torch.equal(patch, student.encoder.patch_embed.proj.weight)


def test_nonfinite_step_leaves_parameters_unchanged(small_model):
    before = small_model.out.weight.detach().clone()
    optimizer = torch.optim.AdamW(small_model.parameters())
    with pytest.raises(ValueError):
        train_step(small_model, optimizer, torch.full((1,3,32,32), float('nan')), torch.zeros(1,3,32,32))
    assert torch.equal(before, small_model.out.weight)


def test_validation_is_sample_weighted_and_restores_training_mode(small_model):
    images = torch.rand(3,3,32,32)
    labels = torch.zeros(3,3,32,32)
    whole = evaluate(small_model, [(images, labels)])
    split = evaluate(small_model, [(images[:2],labels[:2]),(images[2:],labels[2:])])
    assert whole == pytest.approx(split, rel=1e-5)
    assert small_model.training


def test_empty_validation_is_rejected(small_model):
    with pytest.raises(ValueError):
        evaluate(small_model, [])


def test_factory_selects_true_vits_backbone_without_download():
    with torch.device('meta'):
        model = build_model('vits')
    assert len(model.encoder.blocks) == 12
    assert model.out.weight.shape == (192, 384)
    assert model.encoder.blocks[0].attn.num_heads == 6
