"""File pairing and the original Cellpose normalization/augmentation path."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from cellpose import models, train, transforms


def read_local_paths(path: str | Path) -> dict:
    """Resolve dataset and checkpoint paths against their config file, including in Jupyter."""
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding='utf8'))
    values['dataset_root'] = str(path.parent / values['dataset_root'])
    for key in ('vits_encoder', 'vits_model', 'vitb_model', 'vitl_model'):
        if key in values:
            values[key]['path'] = str(path.parent / values[key]['path'])
    return values


@dataclass(frozen=True)
class Sample:
    image: Path
    flow: Path
    mask: Path


def discover_splits(root: str | Path, training_split: str | None = None) -> dict[str, list[Sample]]:
    """Read train (or train_eligible), valid and test; reject overlap by image stem."""
    root = Path(root)
    result, seen = {}, set()
    if training_split not in (None, 'train', 'train_eligible'):
        raise ValueError('training_split must be train or train_eligible')
    if training_split is None and (root / 'train').exists() and (root / 'train_eligible').exists():
        raise ValueError('Use only one training directory: train or train_eligible')
    for split in ('train', 'valid', 'test'):
        folder = root / split
        if split == 'train':
            if training_split is not None:
                folder = root / training_split
            elif not folder.exists():
                folder = root / 'train_eligible'
        images = sorted(folder.glob('*.jpg'))
        if not images:
            raise ValueError(f'Empty split: {folder}')
        names = {image.stem.casefold() for image in images}
        if len(names) != len(images) or names & seen:
            raise ValueError(f'Duplicate image identity in split {split}')
        seen.update(names)
        samples = [Sample(p, p.with_name(p.stem + '_flows.tif'), p.with_name(p.stem + '_masks.tif')) for p in images]
        if split != 'test' and any(not sample.flow.is_file() for sample in samples):
            raise ValueError(f'Missing Cellpose flow labels in {split}')
        result[split] = samples
    return result


def make_batch(samples: list[Sample], indices, seed: int, device='cpu', tile_size: int = 384):
    """Return normalized/augmented NCHW tensors. Label TIFF has [IDs, foreground, dY, dX]."""
    if tile_size < 16 or tile_size % 8:
        raise ValueError('Tile size must be >=16 and divisible by 8')
    if len(indices) == 0:
        raise ValueError('Batch indices are empty')
    images, labels = train._get_batch(indices,
        files=[str(s.image) for s in samples], labels_files=[str(s.flow) for s in samples],
        normalize_params=models.normalize_default)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        x, y = transforms.random_rotate_and_resize(images, lbls=labels,
            rescale=np.ones(len(indices), dtype='float32'), bsize=tile_size,
            scale_range=.5, device=torch.device(device))[:2]
    finally:
        np.random.set_state(state)
    return x, y
