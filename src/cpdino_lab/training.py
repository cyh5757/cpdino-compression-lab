"""Model initialization, one-epoch training and validation-based checkpoint selection."""
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import restore_checkpoint, save_checkpoint
from .data import make_batch
from .engine import evaluate, train_step
from .models import build_model
from .weights import load_weights


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def dataset_fingerprint(splits):
    manifest = []
    for split in ('train', 'valid'):
        for sample in splits[split]:
            for path in (sample.image, sample.flow):
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                manifest.append((split, path.name, digest))
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def prepare_models(config, paths, device='cpu'):
    """Load the configured student initialization and optional frozen teacher."""
    key = config.get('initialization_key')
    if key is None:
        suffix = 'encoder' if config['initialization_scope'] == 'encoder' else 'model'
        key = config['student'] + '_' + suffix
    student = build_model(config['student'])
    provenance = load_weights(student, paths[key]['path'], paths[key]['sha256'],
                              scope=config['initialization_scope'])
    student.to(device)
    teacher, teacher_provenance = None, None
    if config.get('teacher'):
        teacher_key = config['teacher'] + '_model'
        teacher = build_model(config['teacher'])
        teacher_provenance = load_weights(teacher, paths[teacher_key]['path'],
                                         paths[teacher_key]['sha256'], scope='model')
        teacher.to(device).eval().requires_grad_(False)
    elif config.get('teacher_weight', 0) != 0:
        raise ValueError('A positive teacher weight requires a teacher model')
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    return student, teacher, optimizer, {'student': provenance, 'teacher': teacher_provenance}


class EpochStore:
    """Keep latest state and the lowest supervised validation-loss checkpoint."""

    def __init__(self, directory, identity, model, optimizer):
        self.directory = Path(directory)
        self.identity, self.model, self.optimizer = identity, model, optimizer
        self.epoch, self.best_epoch, self.best_loss = 0, 0, float('inf')

    def resume(self):
        path = self.directory/'last.pt'
        if not path.exists():
            return False
        epoch, metadata = restore_checkpoint(path, self.model, self.optimizer, self.identity,
                                             return_training_state=True)
        if 'best_loss' not in metadata or 'best_epoch' not in metadata:
            raise ValueError('This checkpoint lacks validation-selection metadata')
        if metadata['best_epoch'] and not (self.directory/'best.pt').is_file():
            raise ValueError('The selected best checkpoint is missing')
        self.epoch, self.best_epoch, self.best_loss = epoch, metadata['best_epoch'], metadata['best_loss']
        return True

    def commit(self, epoch, validation_loss):
        if epoch != self.epoch + 1:
            raise ValueError('Epoch must advance by exactly one')
        if validation_loss is not None and not math.isfinite(validation_loss):
            raise ValueError('Validation loss must be finite')
        improved = validation_loss is not None and validation_loss < self.best_loss
        metadata = {'best_loss': validation_loss if improved else self.best_loss,
                    'best_epoch': epoch if improved else self.best_epoch}
        if improved:
            save_checkpoint(self.directory/'best.pt', self.model, self.optimizer, epoch,
                            self.identity, training_state=metadata)
        save_checkpoint(self.directory/'last.pt', self.model, self.optimizer, epoch,
                        self.identity, training_state=metadata)
        self.epoch, self.best_epoch, self.best_loss = epoch, metadata['best_epoch'], metadata['best_loss']
        return improved


def train_epoch(student, optimizer, teacher, splits, config, store, device='cpu', progress=None):
    """Execute one complete epoch, validate when scheduled, then save best/last states."""
    epoch = store.epoch + 1
    if epoch > config['epochs']:
        raise ValueError('The configured epoch budget has been reached')
    batch_size = config['batch_size']
    interval = config['validation_interval']
    if batch_size < 1 or interval < 1 or not splits['train'] or not splits['valid']:
        raise ValueError('Positive batch/validation intervals and nonempty splits are required')
    started = time.perf_counter()
    order = np.random.default_rng(config['seed'] + epoch).permutation(len(splits['train']))
    total = 0.0
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset+batch_size]
        images, labels = make_batch(splits['train'], indices, epoch*1000+offset,
                                    device, config['tile_size'])
        value = train_step(student, optimizer, images, labels, teacher, config['teacher_weight'])
        total += value * len(indices)
        if progress is not None:
            progress({'epoch':epoch,'completed_images':offset+len(indices),
                      'total_images':len(order),'batch_loss':value})
    validation = None
    if epoch == 1 or epoch % interval == 0 or epoch == config['epochs']:
        batches = (make_batch(splits['valid'], np.arange(offset,min(offset+batch_size,len(splits['valid']))),
                              42+offset,device,config['tile_size'])
                   for offset in range(0,len(splits['valid']),batch_size))
        validation = evaluate(student,batches)
    improved = store.commit(epoch,validation)
    row = {'epoch':epoch,'epochs':config['epochs'],'train_loss':total/len(order),
           'validation_loss':validation,'best_loss':store.best_loss,'best_epoch':store.best_epoch,
           'improved':improved,'seconds':time.perf_counter()-started}
    write_json(store.directory/'progress.json',row)
    with (store.directory/'history.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
    return row
