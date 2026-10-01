import torch
from torch.utils.data import TensorDataset

from . import boolean, mnist, npz


def load_dataset(config, split='train', path=None, dataset=None, download=False):
    dataset = dataset or config['data'].get('dataset', 'boolean' if config['model']['num_classes'] == 2 else 'mnist')
    if path:
        images, labels = npz.read_split(path, 'val' if split == 'test' else split)
    elif dataset == 'mnist':
        images, labels = mnist.read_split(config, split, download)
    elif dataset == 'boolean':
        images, labels = boolean.read_split(config, split)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not torch.isfinite(images).all():
        raise ValueError('Images must be finite')
    images = (images > config['data']['threshold']).float()
    if images.shape[1] != config['model']['input_dim'] or labels.shape != images.shape[:1] or not len(labels):
        raise ValueError('Invalid input or target dimensions')
    if labels.min() < 0 or labels.max() >= config['model']['num_classes']:
        raise ValueError('Class labels are outside the configured range')
    return TensorDataset(images, labels.long())
