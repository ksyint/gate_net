"""Paper depth, operand-gradient, and locality studies with seed aggregation."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from train import main as train
from utils.util import load_config, write_json


def main(args):
    base = load_config('configs/mnist/depth4.yaml')
    if args.study == 'depth':
        settings = [(f'depth{depth}', {'depth': depth}) for depth in (2, 4, 8)]
    elif args.study == 'operands':
        settings = [('ste', {'depth': 1, 'detach_operands': False}),
                    ('detached', {'depth': 1, 'detach_operands': True})]
    else:
        settings = [('local', {'depth': 4, 'local_init': True}),
                    ('random', {'depth': 4, 'local_init': False})]
    rows, aggregate = [], []
    for name, changes in settings:
        values = []
        for seed in args.seeds:
            config = deepcopy(base)
            config['model'].update(changes)
            config['train'].update(seed=seed, epochs=args.epochs)
            output = Path(args.output) / args.study / name / f'seed{seed}'
            output.mkdir(parents=True, exist_ok=True)
            config['train']['save_dir'] = str(output)
            config['data']['root'] = args.data_root
            path = output / 'config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            report = train(argparse.Namespace(config=str(path), dataset='mnist', depth=None,
                           seed=None, epochs=None, output=None, data=None, device=args.device,
                           offline=args.offline, smoke=False, resume=None))
            accuracy = report['final_test']['accuracy']
            values.append(accuracy)
            rows.append({'variant': name, 'seed': seed, 'accuracy': accuracy, 'directory': str(output)})
        aggregate.append({'variant': name, 'seeds': args.seeds, 'mean_accuracy': float(np.mean(values)),
                          'std_accuracy': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0})
    write_json(Path(args.output) / args.study / 'summary.json', {'runs': rows, 'aggregate': aggregate})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['depth', 'operands', 'locality'], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/studies')
    args = parser.parse_args()
    if args.epochs < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error('Use positive epochs and unique seeds')
    main(args)
