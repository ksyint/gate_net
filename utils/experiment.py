"""Dataset/depth profiles and checkpoint reconstruction for the experiment scripts."""
from pathlib import Path

import torch

from .models import OSLGN
from .util import load_config, cuda_device


def experiment_config(args):
    dataset = 'boolean' if args.smoke else args.dataset
    depth = args.depth or (2 if dataset == 'boolean' else 4)
    path = args.config or f'configs/{dataset}/depth{depth}.yaml'
    config = load_config(path)
    if args.depth is not None:
        config['model']['depth'] = args.depth
    for key in ('seed', 'epochs'):
        value = getattr(args, key, None)
        if value is not None:
            config['train'][key] = value
    if args.output:
        config['train']['save_dir'] = args.output
    config['data']['dataset'] = dataset
    return config


def restore_model(checkpoint, device='cuda'):
    device = cuda_device(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = OSLGN(**saved['config']['model']).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved['config']
