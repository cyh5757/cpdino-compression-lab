import copy
import json

import numpy as np
import pytest
import torch
import tifffile
from cellpose import io
from cpdino_lab.data import discover_splits, make_batch, read_local_paths
from cpdino_lab.engine import train_step
from cpdino_lab.checkpoints import save_checkpoint, restore_checkpoint


def test_local_paths_are_relative_to_config_file_not_notebook_cwd(tmp_path, monkeypatch):
    notebook_dir = tmp_path / 'notebooks'
    notebook_dir.mkdir()
    absolute_weight = tmp_path / 'external.pth'
    path = tmp_path / 'local_paths.json'
    path.write_text(json.dumps({'dataset_root':'data/converted',
        'vits_encoder':{'path':'weights/encoder.pth','sha256':'a'},
        'vitl_model':{'path':str(absolute_weight),'sha256':'b'}}))
    monkeypatch.chdir(notebook_dir)
    config = read_local_paths(path)
    assert config['dataset_root'] == str(tmp_path / 'data/converted')
    assert config['vits_encoder']['path'] == str(tmp_path / 'weights/encoder.pth')
    assert config['vitl_model']['path'] == str(absolute_weight)


def write_dataset(root):
    for index, split in enumerate(['train', 'valid', 'test']):
        folder = root / split
        folder.mkdir()
        name = str(index)
        io.imsave(str(folder / (name + '.jpg')), np.zeros((48,48,3),dtype=np.uint8))
        tifffile.imwrite(folder / (name + '_flows.tif'), np.zeros((4,48,48),dtype=np.float32), photometric='minisblack')


def test_dataset_and_cellpose_augmentation_produce_expected_batch(tmp_path):
    write_dataset(tmp_path)
    splits = discover_splits(tmp_path)
    x, y = make_batch(splits['train'], [0], seed=42, tile_size=32)
    assert x.shape == (1,3,32,32)
    assert y.shape == (1,3,32,32)
    assert torch.isfinite(x).all()


def test_missing_flow_file_is_rejected(tmp_path):
    write_dataset(tmp_path)
    (tmp_path / 'valid/1_flows.tif').unlink()
    with pytest.raises(ValueError):
        discover_splits(tmp_path)


def test_split_overlap_is_rejected(tmp_path):
    write_dataset(tmp_path)
    (tmp_path / 'valid/1.jpg').rename(tmp_path / 'valid/0.jpg')
    (tmp_path / 'valid/1_flows.tif').rename(tmp_path / 'valid/0_flows.tif')
    with pytest.raises(ValueError):
        discover_splits(tmp_path)


def test_checkpoint_restores_optimizer_rng_and_next_update(tmp_path, small_model):
    model = small_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    x, labels = torch.rand(1,3,32,32), torch.zeros(1,3,32,32)
    train_step(model, optimizer, x, labels)
    save_checkpoint(tmp_path / 'last.pt', model, optimizer, 1, {'run':'unit'})
    expected_random = torch.rand(3)
    train_step(model, optimizer, x, labels)
    expected = model.out.weight.detach().clone()
    clone = copy.deepcopy(model)
    opt = torch.optim.AdamW(clone.parameters(), lr=.9)
    assert restore_checkpoint(tmp_path / 'last.pt', clone, opt, {'run':'unit'}) == 1
    assert torch.equal(torch.rand(3), expected_random)
    train_step(clone, opt, x, labels)
    assert torch.equal(clone.out.weight, expected)


def test_wrong_checkpoint_identity_fails_before_model_mutation(tmp_path, small_model):
    opt = torch.optim.AdamW(small_model.parameters())
    save_checkpoint(tmp_path / 'last.pt', small_model, opt, 1, {'run':'one'})
    before = small_model.out.weight.detach().clone()
    with pytest.raises(ValueError):
        restore_checkpoint(tmp_path / 'last.pt', small_model, opt, {'run':'two'})
    assert torch.equal(before, small_model.out.weight)
