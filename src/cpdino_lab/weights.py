"""Verified loading of official DINOv3 and project CPDINO checkpoints."""
import hashlib
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def convert_huggingface(values: dict, template: dict) -> dict:
    """Map official Transformers ViT-S tensors to the original DINOv3 encoder."""
    normalized = {}
    for name, value in values.items():
        key = name.removeprefix('model.') if name.startswith('model.layer.') else name
        if key in normalized:
            raise ValueError(f'Duplicate tensor {key}')
        normalized[key] = value
    consumed, mapped = set(), {}

    def take(key):
        if key not in normalized:
            raise ValueError(f'Missing official tensor {key}')
        consumed.add(key)
        return normalized[key]

    substitutions = [
        (r'^cls_token$', 'embeddings.cls_token'),
        (r'^storage_tokens$', 'embeddings.register_tokens'),
        (r'^mask_token$', 'embeddings.mask_token'),
        (r'^patch_embed\.proj\.', 'embeddings.patch_embeddings.'),
        (r'^blocks\.(\d+)\.attn\.proj\.', r'layer.\1.attention.o_proj.'),
        (r'^blocks\.(\d+)\.ls(\d+)\.gamma$', r'layer.\1.layer_scale\2.lambda1'),
        (r'^blocks\.(\d+)\.mlp\.fc1\.', r'layer.\1.mlp.up_proj.'),
        (r'^blocks\.(\d+)\.mlp\.fc2\.', r'layer.\1.mlp.down_proj.'),
        (r'^blocks\.(\d+)\.norm', r'layer.\1.norm'),
    ]
    for key, default in template.items():
        if key == 'rope_embed.periods' or key.endswith('.attn.qkv.bias_mask'):
            mapped[key] = default
            continue
        match = re.fullmatch(r'blocks\.(\d+)\.attn\.qkv\.(weight|bias)', key)
        if match:
            block, kind = match.groups()
            prefix = f'layer.{block}.attention.'
            q = take(prefix + f'q_proj.{kind}')
            k = take(prefix + 'k_proj.weight') if kind == 'weight' else torch.zeros_like(q)
            v = take(prefix + f'v_proj.{kind}')
            value = torch.cat((q, k, v), dim=0)
        else:
            name = key
            for pattern, replacement in substitutions:
                name = re.sub(pattern, replacement, name)
            value = take(name)
            if key == 'mask_token':
                value = value.squeeze(1)
        if value.shape != default.shape:
            raise ValueError(f'Unexpected shape for {key}: {tuple(value.shape)}')
        mapped[key] = value
    if set(normalized) != consumed:
        raise ValueError(f'Unexpected tensors: {sorted(set(normalized) - consumed)}')
    return mapped


def load_weights(model, path: str | Path, expected_sha256: str, scope: str = 'encoder') -> dict:
    """Load official encoder or raw/wrapped project model weights after SHA-256 verification."""
    path = Path(path)
    if scope not in ('encoder', 'model'):
        raise ValueError('scope must be encoder or model')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError('Checkpoint SHA-256 differs')
    target = model.encoder if scope == 'encoder' else model
    if path.suffix == '.safetensors':
        values = load_file(str(path))
        if scope == 'encoder':
            values = convert_huggingface(values, target.state_dict())
    else:
        values = torch.load(path, map_location='cpu', weights_only=True)
        values = values.get('model', values)
    current = target.state_dict()
    if values.keys() != current.keys() or any(values[k].shape != current[k].shape for k in current):
        raise ValueError('Checkpoint keys or shapes do not match the target architecture')
    target.load_state_dict(values, strict=True)
    return {'sha256': digest, 'scope': scope, 'filename': path.name}
