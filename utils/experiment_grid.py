"""MNIST wiring studies consumed by the direct experiment script."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import math

import yaml


ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / 'configs' / 'mnist' / 'sweeps'
DEPTHS = (2, 3, 4, 5, 6, 8)
WIDTHS = (128, 256, 512)
INITIALIZATIONS = ((True, 0.5), (True, 1.0), (True, 2.0), (True, 4.0), (False, 2.0))
SEEDS = (42, 123, 456)


def initialization_name(model):
    if not model['local_init']:
        return 'random'
    return 'local_sigma_' + format(model['sigma'], 'g').replace('.', 'p')


def experiment_name(config):
    model, train = config['model'], config['train']
    return (Path(f"depth_{model['depth']:02d}") / f"width_{model['hidden_dim']:03d}" /
            initialization_name(model) / f"seed_{train['seed']}")


def validate_config(config):
    model, train, data = config['model'], config['train'], config['data']
    for field in ('input_dim', 'hidden_dim', 'num_classes', 'depth'):
        if not isinstance(model[field], int) or model[field] < 1:
            raise ValueError(f'Model field {field} must be a positive integer')
    if model['depth'] < 2:
        raise ValueError('Width studies require at least two logic layers')
    if data['dataset'] != 'mnist' or model['input_dim'] != 784 or model['num_classes'] != 10:
        raise ValueError('MNIST profiles require 784 inputs and 10 classes')
    if not math.isfinite(model['sigma']) or model['sigma'] <= 0:
        raise ValueError('Locality width must be positive and finite')
    if not isinstance(model['local_init'], bool) or not isinstance(model['detach_operands'], bool):
        raise ValueError('Wiring switches must be Boolean values')
    if not 0 < data['validation_fraction'] < 1 or not 0 <= data['threshold'] <= 1:
        raise ValueError('Invalid validation fraction or pixel threshold')
    if train['epochs'] < 1 or train['batch_size'] < 1 or train['learning_rate'] <= 0:
        raise ValueError('Invalid optimizer schedule')
    widths = [model['input_dim']] + [model['hidden_dim']] * (model['depth'] - 1) + [model['num_classes']]
    # Two operand selectors and sixteen gate scores per unit, plus the class head.
    parameters = sum(2 * before * after + 16 * after for before, after in zip(widths[:-1], widths[1:]))
    return parameters + model['num_classes'] ** 2 + model['num_classes']


def write_profiles():
    base = yaml.safe_load((ROOT / 'configs/mnist/depth2.yaml').read_text())
    written = []
    for depth, width, (local, sigma), seed in product(DEPTHS, WIDTHS, INITIALIZATIONS, SEEDS):
        config = deepcopy(base)
        config['model'].update(depth=depth, hidden_dim=width, local_init=local, sigma=sigma)
        config['train']['seed'] = seed
        key = experiment_name(config)
        config['train']['save_dir'] = str(Path('results/mnist/wiring') / key)
        validate_config(config)
        path = SWEEPS / key.with_suffix('.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# MNIST wiring study: depth, width, initialization, seed.\n' +
                        yaml.safe_dump(config, sort_keys=False))
        written.append(path)
    return written


def read_profiles():
    profiles = []
    for path in sorted(SWEEPS.rglob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        profiles.append((path, config, validate_config(config)))
    if not profiles:
        raise ValueError('No wiring experiment profiles were found')
    outputs = [config['train']['save_dir'] for _, config, _ in profiles]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Each wiring experiment needs a distinct output directory')
    return profiles


if __name__ == '__main__':
    print(f'Wrote {len(write_profiles())} complete MNIST experiment profiles')
