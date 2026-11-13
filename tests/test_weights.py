import hashlib

import pytest
import torch
from cpdino_lab.weights import convert_huggingface, load_weights


def test_huggingface_qkv_conversion_order_and_zero_key_bias():
    source = {f'layer.0.attention.{key}_proj.weight': torch.full((2,2), value)
        for key, value in [('q',1.),('k',2.),('v',3.)]}
    source.update({'layer.0.attention.q_proj.bias':torch.ones(2), 'layer.0.attention.v_proj.bias':torch.full((2,),3.)})
    result = convert_huggingface(source, {'blocks.0.attn.qkv.weight':torch.empty(6,2), 'blocks.0.attn.qkv.bias':torch.empty(6)})
    assert result['blocks.0.attn.qkv.weight'][:,0].tolist() == [1,1,2,2,3,3]
    assert result['blocks.0.attn.qkv.bias'].tolist() == [1,1,0,0,3,3]


def test_huggingface_conversion_rejects_wrong_tensor_shape():
    with pytest.raises(ValueError):
        convert_huggingface({'embeddings.cls_token':torch.zeros(1,1,3)}, {'cls_token':torch.zeros(1,1,4)})


def test_checkpoint_hash_is_checked_before_loading(tmp_path, small_model):
    path = tmp_path / 'weights.pth'
    torch.save(small_model.encoder.state_dict(), path)
    with pytest.raises(ValueError):
        load_weights(small_model, path, '0'*64)


def test_loads_historical_wrapped_model_checkpoint(tmp_path, small_model):
    path = tmp_path / 'best.pt'
    before = small_model.out.weight.detach().clone()
    torch.save({'model':small_model.state_dict(),'identity':{'kind':'tiny'}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with torch.no_grad():
        small_model.out.weight.zero_()
    load_weights(small_model, path, digest, scope='model')
    assert torch.equal(small_model.out.weight, before)
