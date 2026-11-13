"""Atomic snapshots for notebook training, with optimizer and RNG restoration."""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, epoch: int, identity: dict, training_state: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'epoch': epoch, 'identity': identity, 'torch_rng': torch.get_rng_state(),
        'python_rng': random.getstate(),
        'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    if training_state is not None:
        state['training_state'] = training_state
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_checkpoint(path, model, optimizer, identity: dict, *, return_training_state=False):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['identity'] != identity:
        raise ValueError('Checkpoint identity differs from this experiment')
    current = model.state_dict()
    incoming = state['model']
    if incoming.keys() != current.keys() or any(incoming[k].shape != current[k].shape for k in current):
        raise ValueError('Checkpoint architecture differs from this model')
    model.load_state_dict(incoming, strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    rng = state['numpy_rng']
    np.random.set_state((rng[0], np.array(rng[1], dtype=np.uint32), *rng[2:]))
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    if return_training_state:
        return state['epoch'], state.get('training_state', {})
    return state['epoch']
