import copy
import hashlib

import pytest
import torch

from cpdino_lab.training import EpochStore, prepare_models, train_epoch


def test_s_distillation_loads_complete_student_and_freezes_teacher(tmp_path, small_model, monkeypatch):
    import cpdino_lab.training as training
    with torch.no_grad():
        small_model.out.bias.fill_(3)
    student_file = tmp_path / 'student.pt'
    torch.save({'model': small_model.state_dict()}, student_file)
    with torch.no_grad():
        small_model.out.bias.fill_(7)
    teacher_file = tmp_path / 'teacher.pt'
    torch.save(small_model.state_dict(), teacher_file)
    paths = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
             for key, path in [('vits_model',student_file),('vitl_model',teacher_file)]}
    paths['vits_encoder'] = {'path': 'must-not-be-loaded', 'sha256': 'invalid'}
    monkeypatch.setattr(training, 'build_model', lambda name: copy.deepcopy(small_model))
    config = {'student':'vits', 'initialization_key':'vits_model', 'initialization_scope':'model',
              'teacher':'vitl', 'teacher_weight':1, 'learning_rate':1e-5, 'weight_decay':.1}
    student, teacher, optimizer, identity = prepare_models(config, paths, 'cpu')
    assert torch.all(student.out.bias == 3)
    assert torch.all(teacher.out.bias == 7)
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not optimized.intersection(id(p) for p in teacher.parameters())
    assert identity['student']['scope'] == 'model'


def test_best_checkpoint_selection_survives_resume(tmp_path, small_model):
    opt = torch.optim.AdamW(small_model.parameters())
    store = EpochStore(tmp_path, {'run':'kd'}, small_model, opt)
    with torch.no_grad():
        small_model.out.bias.fill_(1)
    store.commit(1, .3)
    with torch.no_grad():
        small_model.out.bias.fill_(2)
    store.commit(2, .4)
    best = torch.load(tmp_path/'best.pt', weights_only=True)
    assert best['epoch'] == 1
    assert torch.all(best['model']['out.bias'] == 1)
    clone = copy.deepcopy(small_model)
    new_opt = torch.optim.AdamW(clone.parameters())
    resumed = EpochStore(tmp_path, {'run':'kd'}, clone, new_opt)
    assert resumed.resume()
    assert (resumed.epoch,resumed.best_epoch,resumed.best_loss) == (2,1,.3)
    with torch.no_grad():
        clone.out.bias.fill_(3)
    resumed.commit(3,.2)
    best = torch.load(tmp_path/'best.pt',weights_only=True)
    assert best['epoch'] == 3
    assert torch.all(best['model']['out.bias'] == 3)


def test_nonfinite_validation_is_not_checkpointed(tmp_path, small_model):
    store = EpochStore(tmp_path, {}, small_model, torch.optim.AdamW(small_model.parameters()))
    with pytest.raises(ValueError, match='finite'):
        store.commit(1, float('nan'))
    assert not (tmp_path/'last.pt').exists()
    assert store.epoch == 0


def test_epoch_trains_all_samples_and_validates_without_teacher(tmp_path, small_model, monkeypatch):
    import cpdino_lab.training as training
    teacher = copy.deepcopy(small_model).eval().requires_grad_(False)
    calls = []
    def batch(samples, indices, seed, device, tile_size):
        calls.append((samples, list(indices)))
        return torch.full((len(indices),3,32,32),.2), torch.zeros(len(indices),3,32,32)
    monkeypatch.setattr(training, 'make_batch', batch)
    opt = torch.optim.AdamW(small_model.parameters(),lr=.001)
    store = EpochStore(tmp_path, {'run':'test'}, small_model,opt)
    config = {'epochs':2,'batch_size':2,'tile_size':32,'seed':19,'validation_interval':10,'teacher_weight':1}
    before = small_model.out.weight.detach().clone()
    row = train_epoch(small_model,opt,teacher,{'train':[0,1,2],'valid':[3,4]},config,store,'cpu')
    assert row['epoch'] == 1
    assert row['validation_loss'] is not None
    assert sorted(i for samples,ids in calls if samples==[0,1,2] for i in ids)==[0,1,2]
    assert not torch.equal(before,small_model.out.weight)
    assert all(p.grad is None for p in teacher.parameters())
    assert (tmp_path/'best.pt').is_file()
    assert (tmp_path/'progress.json').is_file()
    assert len((tmp_path/'history.jsonl').read_text().splitlines()) == 1
    config['epochs'] = 1
    with pytest.raises(ValueError, match='budget'):
        train_epoch(small_model,opt,teacher,{'train':[0],'valid':[1]},config,store,'cpu')
