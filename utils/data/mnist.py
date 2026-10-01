import torch


def read_split(config, split, download=False):
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
            raise ValueError('No validation split configured; evaluate the official test split after training')
        if not 0 < fraction < 1:
            raise ValueError('validation_fraction must lie strictly between zero and one')
        count = max(1, round(len(labels) * fraction))
        if count >= len(labels):
            raise ValueError('Validation split must leave training examples')
        indices = order[count:] if split == 'train' else order[:count]
        images, labels = images[indices], labels[indices]
    return images, labels
