"""Select a dataset/depth/wiring study and run the ordinary training loop."""
import argparse
from pathlib import Path

from utils.experiment_grid import ROOT, initialization_name, read_profiles


def selected(config, args):
    model, train = config['model'], config['train']
    values = ((args.depths, model['depth']), (args.widths, model['hidden_dim']),
              (args.initializations, initialization_name(model)), (args.seeds, train['seed']))
    return all(choices is None or value in choices for choices, value in values)


def main(args):
    available = read_profiles()
    profiles = [entry for entry in available if selected(entry[1], args)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        profiles = profiles[:args.limit]
    if not profiles:
        raise ValueError('No experiment matches the selected depth, width, initialization, and seed')
    print(f'Validated {len(available)} wiring profiles; selected {len(profiles)} experiments.')
    if args.dry_run:
        for path, config, parameters in profiles[:4]:
            print(f'{path.relative_to(ROOT)}: {parameters} trainable parameters')
        return
    if args.data and not Path(args.data).is_file():
        raise FileNotFoundError(args.data)
    if args.device != 'cuda' and not args.device.startswith('cuda:'):
        raise ValueError('Logic network experiments require CUDA')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('--epochs must be positive')
    from train import main as train
    for path, config, _ in profiles:
        train(argparse.Namespace(dataset='mnist', config=str(path), depth=None, seed=None,
                                 epochs=args.epochs, output=None, data=args.data,
                                 download=args.download, device=args.device, smoke=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depths', nargs='+', type=int)
    parser.add_argument('--widths', nargs='+', type=int)
    parser.add_argument('--initializations', nargs='+',
                        choices=['local_sigma_0p5', 'local_sigma_1', 'local_sigma_2', 'local_sigma_4', 'random'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--data', help='Optional NPZ with flattened MNIST-size examples')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
