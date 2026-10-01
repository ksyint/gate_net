import torch


def read_split(config, split):
    if config['model']['num_classes'] != 2 or config['model']['input_dim'] < 4:
        raise ValueError('Boolean task needs at least four inputs and two classes')
    generator = torch.Generator().manual_seed(1101 if split == 'train' else 1102)
    images = torch.randint(0, 2, (512 if split == 'train' else 128, config['model']['input_dim']), generator=generator).float()
    labels = ((images[:, 0].bool() ^ images[:, 1].bool()) |
              (images[:, 2].bool() & images[:, 3].bool())).long()
    return images, labels
