"""Read Boolean, MNIST, and normalized archive partitions for wiring experiments."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def boolean_read_split(config, split):
    if config['model']['num_classes'] != 2 or config['model']['input_dim'] < 4:
        raise ValueError('Boolean task needs at least four inputs and two classes')
    generator = torch.Generator().manual_seed(1101 if split == 'train' else 1102)
    images = torch.randint(
        0, 2, (512 if split == 'train' else 128, config['model']['input_dim']), generator=generator
    ).float()
    labels = (
        (images[:, 0].bool() ^ images[:, 1].bool()) | (images[:, 2].bool() & images[:, 3].bool())
    ).long()
    return images, labels



def mnist_read_split(config, split, download=False):
    from torchvision.datasets import MNIST

    data = MNIST(config['data']['root'], train=(split != 'test'), download=download)
    images, labels = data.data.float().flatten(1) / 255.0, data.targets
    if split != 'test':
        generator = torch.Generator().manual_seed(config['train']['seed'])
        order = torch.randperm(len(labels), generator=generator)
        fraction = config['data'].get('validation_fraction', 0.1)
        if fraction == 0:
            if split == 'train':
                return images, labels
            raise ValueError(
                'No validation split configured; evaluate the official test split after training'
            )
        if not 0 < fraction < 1:
            raise ValueError('validation_fraction must lie strictly between zero and one')
        count = max(1, round(len(labels) * fraction))
        if count >= len(labels):
            raise ValueError('Validation split must leave training examples')
        indices = order[count:] if split == 'train' else order[:count]
        images, labels = images[indices], labels[indices]
    return images, labels



def npz_read_split(path, split):
    with np.load(path, allow_pickle=False) as arrays:
        images = torch.tensor(arrays[f'x_{split}'], dtype=torch.float32)
        labels = torch.tensor(arrays[f'y_{split}'], dtype=torch.long)
    return images.flatten(1), labels



def load_dataset(config, split='train', path=None, dataset=None, download=False):
    dataset = dataset or config['data'].get(
        'dataset', 'boolean' if config['model']['num_classes'] == 2 else 'mnist'
    )
    if path:
        images, labels = npz_read_split(path, 'val' if split == 'test' else split)
    elif dataset == 'mnist':
        images, labels = mnist_read_split(config, split, download)
    elif dataset == 'boolean':
        images, labels = boolean_read_split(config, split)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not torch.isfinite(images).all():
        raise ValueError('Images must be finite')
    images = (images > config['data']['threshold']).float()
    if (
        images.shape[1] != config['model']['input_dim']
        or labels.shape != images.shape[:1]
        or not len(labels)
    ):
        raise ValueError('Invalid input or target dimensions')
    if labels.min() < 0 or labels.max() >= config['model']['num_classes']:
        raise ValueError('Class labels are outside the configured range')
    return TensorDataset(images, labels.long())
