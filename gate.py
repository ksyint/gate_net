"""Train and inspect Boolean wiring experiments."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from networks.gates import wiring_statistics, OSLGN, evaluate_circuit, symbolic_equations, minimize_features
from experiments.wiring import (
    load_dataset,
    experiment_config,
    cuda_device,
    evaluate,
    set_seed,
    write_json,
    restore_model,
    ROOT,
    initialization_name,
    read_profiles,
    load_config,
    write_config,
    catalog_cli,
)
from PIL import Image
from copy import deepcopy


def train_main(args):
    config = experiment_config(args)
    set_seed(config['train']['seed'])
    torch.set_num_threads(config['train'].get('num_threads', 1))
    device = cuda_device(args.device)
    download = not getattr(args, 'offline', False)
    print(f"Dataset: {config['data']['dataset']} | depth: {config['model']['depth']} | device: {device}")
    model = OSLGN(**config['model']).to(device)
    train_set = load_dataset(config, 'train', args.data, download=download)
    use_validation = (
        bool(args.data)
        or config['data']['dataset'] != 'mnist'
        or config['data']['validation_fraction'] > 0
    )
    val_set = load_dataset(config, 'val', args.data, download=download) if use_validation else None
    train_loader = DataLoader(
        train_set, batch_size=config['train']['batch_size'], shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=128, pin_memory=True) if val_set is not None else None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['train']['learning_rate'], betas=(0.9, 0.999)
    )
    output = Path(config['train']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch = [], float('inf'), 0
    if getattr(args, 'resume', None):
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        if saved['config']['model'] != config['model'] or saved['config']['data'] != config['data']:
            raise ValueError('Resume requires matching architecture and data configuration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = config['train']['learning_rate']
        start_epoch, best = saved['epoch'], saved.get('best_loss', float('inf'))
        history = saved.get('history', [])
        if 'torch_rng' in saved:
            torch.set_rng_state(saved['torch_rng'].cpu())
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        total_loss = correct = count = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            if config['train'].get('clip_max_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_max_norm'])
            optimizer.step()
            total_loss += loss.detach().item() * len(targets)
            correct += (logits.detach().argmax(-1) == targets).sum().item()
            count += len(targets)
        training = {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}
        scores = evaluate(model, val_loader, device) if val_loader is not None else training
        record = {'epoch': epoch + 1, 'train': training, 'wiring': wiring_statistics(model)}
        if val_loader is not None:
            record['validation'] = scores
        history.append(record)
        improved = scores['loss'] < best
        best = min(best, scores['loss'])
        saved = {
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'metrics': scores,
            'best_loss': best,
            'history': history,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(device),
        }
        torch.save(saved, output / 'last.pth')
        if improved:
            torch.save(saved, output / 'best.pth')
        print(
            f"Epoch {epoch + 1}: loss={scores['loss']:.4f}, accuracy={100 * scores['accuracy']:.2f}%",
            flush=True,
        )
    report = {
        'config': config,
        'task': 'npz' if args.data else config['data']['dataset'],
        'final_train': evaluate(model, train_loader, device),
        'history': history,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'wiring': wiring_statistics(model),
    }
    if config['data']['dataset'] == 'mnist' and not args.data:
        test_set = load_dataset(config, 'test', download=download)
        report['final_test'] = evaluate(model, DataLoader(test_set, batch_size=128), device)
    write_json(output / 'metrics.json', report)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'boolean'], default='mnist')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--config')
    parser.add_argument('--data', help='NPZ overrides the configured dataset loader')
    parser.add_argument('--data-root', help='Directory containing MNIST/raw')
    parser.add_argument(
        '--download', action='store_true', help='MNIST downloads automatically when online'
    )
    parser.add_argument(
        '--offline', action='store_true', help='Use existing MNIST files without downloads'
    )
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def train_cli():
    train_main(parse_args())


def eval_main(args):
    torch.set_num_threads(1)
    model, config = restore_model(args.checkpoint, args.device)
    if args.data_root:
        config['data']['root'] = args.data_root
    dataset = args.dataset or config['data'].get(
        'dataset', 'boolean' if config['model']['num_classes'] == 2 else 'mnist'
    )
    split = 'test' if dataset == 'mnist' and not args.data else 'val'
    loader = DataLoader(
        load_dataset(config, split, args.data, dataset, not args.offline), batch_size=128
    )
    metrics = evaluate(model, loader, args.device)
    circuit = model.export_circuit()
    maximum_error = 0.0
    with torch.no_grad():
        for images, _ in loader:
            features, logits = evaluate_circuit(circuit, images.numpy())
            np.testing.assert_array_equal(
                features, model.logic_features(images.to(args.device)).cpu().numpy()
            )
            expected = model(images.to(args.device)).cpu().numpy()
            np.testing.assert_allclose(logits, expected, atol=1e-5, rtol=1e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(logits - expected))))
    metrics.update(circuit_features_exact=True, circuit_logit_max_abs_error=maximum_error)
    write_json(args.output, metrics)
    print(metrics)


def eval_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['mnist', 'boolean'])
    parser.add_argument('--data')
    parser.add_argument(
        '--download', action='store_true', help='MNIST downloads automatically when online'
    )
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--data-root')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/eval/metrics.json')
    eval_main(parser.parse_args())


def predict_main(args):
    model, config = restore_model(args.checkpoint, args.device)
    if config['model']['input_dim'] != 784:
        raise ValueError('Image-file prediction requires an MNIST-size 784-input checkpoint')
    arrays = []
    for path in args.images:
        with Image.open(path) as image:
            array = np.asarray(image.convert('L').resize((28, 28)), dtype=np.float32) / 255.0
        arrays.append(array.reshape(-1))
    images = torch.as_tensor(np.stack(arrays), device=args.device)
    images = (images > config['data']['threshold']).float()
    with torch.no_grad():
        features = model.logic_features(images)
        logits = model.head(features)
        predictions = logits.argmax(-1).tolist()
    write_json(
        args.output,
        {
            'images': args.images,
            'predictions': predictions,
            'logic_features': features.tolist(),
            'logits': logits.tolist(),
        },
    )


def predict_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/predictions.json')
    predict_main(parser.parse_args())


def export_main(args):
    torch.set_num_threads(1)
    model, config = restore_model(args.checkpoint, args.device)
    circuit = model.export_circuit()
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randint(0, 2, (64, config['model']['input_dim']), generator=generator).float()
    features, logits = evaluate_circuit(circuit, inputs.numpy())
    with torch.no_grad():
        np.testing.assert_array_equal(
            features, model.logic_features(inputs.to(args.device)).cpu().numpy()
        )
        np.testing.assert_allclose(
            logits, model(inputs.to(args.device)).cpu().numpy(), atol=1e-5, rtol=1e-5
        )
    output = Path(args.output)
    write_json(output / 'circuit.json', circuit)
    (output / 'equations.txt').write_text(symbolic_equations(circuit), encoding='utf-8')
    print(f'Circuit and equations saved to {output}. 64 binary inputs agree exactly.')


def export_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/circuit')
    export_main(parser.parse_args())


def compress_main(args):
    circuit = json.loads(Path(args.circuit).read_text())
    write_json(
        args.output, {'features': minimize_features(circuit, args.max_support), 'head': circuit['head']}
    )


def compress_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--circuit', required=True)
    parser.add_argument('--max-support', type=int, default=16)
    parser.add_argument('--output', default='results/circuit/minimized.json')
    compress_main(parser.parse_args())


def prepare_data_main(args):
    from torchvision.datasets import MNIST

    for training in (True, False):
        dataset = MNIST(args.root, train=training, download=True)
        print(f'{"train" if training else "test"}: {len(dataset)} examples under {dataset.raw_folder}')


def prepare_data_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='datasets')
    prepare_data_main(parser.parse_args())


def selected(config, args):
    model, train = config['model'], config['train']
    values = (
        (args.depths, model['depth']),
        (args.widths, model['hidden_dim']),
        (args.initializations, initialization_name(model)),
        (args.seeds, train['seed']),
    )
    return all(choices is None or value in choices for choices, value in values)


def run_experiments_main(args):
    available = read_profiles()
    profiles = [entry for entry in available if selected(entry[1], args)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        profiles = profiles[: args.limit]
    if not profiles:
        raise ValueError('No experiment matches the selected depth, width, initialization, and seed')
    print(f'Validated {len(available)} wiring profiles. Selected {len(profiles)} experiments.')
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
    train = train_main

    for path, config, _ in profiles:
        train(
            argparse.Namespace(
                dataset='mnist',
                config=str(path),
                depth=None,
                seed=None,
                epochs=args.epochs,
                output=None,
                data=args.data,
                download=True,
                offline=args.offline,
                data_root=args.data_root,
                device=args.device,
                smoke=False,
            )
        )


def run_experiments_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depths', nargs='+', type=int)
    parser.add_argument('--widths', nargs='+', type=int)
    parser.add_argument(
        '--initializations',
        nargs='+',
        choices=['local_sigma_0p5', 'local_sigma_1', 'local_sigma_2', 'local_sigma_4', 'random'],
    )
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--data', help='Optional NPZ with flattened MNIST-size examples')
    parser.add_argument(
        '--download', action='store_true', help='MNIST downloads automatically when online'
    )
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    run_experiments_main(parser.parse_args())


def run_study_main(args):
    base = load_config('configs/mnist_depth4.yaml')
    if args.study == 'depth':
        settings = [(f'depth{depth}', {'depth': depth}) for depth in (2, 4, 8)]
    elif args.study == 'operands':
        settings = [
            ('ste', {'depth': 1, 'detach_operands': False}),
            ('detached', {'depth': 1, 'detach_operands': True}),
        ]
    else:
        settings = [
            ('local', {'depth': 4, 'local_init': True}),
            ('random', {'depth': 4, 'local_init': False}),
        ]
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
            write_config(path, config)
            report = train_main(
                argparse.Namespace(
                    config=str(path),
                    dataset='mnist',
                    depth=None,
                    seed=None,
                    epochs=None,
                    output=None,
                    data=None,
                    device=args.device,
                    offline=args.offline,
                    smoke=False,
                    resume=None,
                )
            )
            accuracy = report['final_test']['accuracy']
            values.append(accuracy)
            rows.append({'variant': name, 'seed': seed, 'accuracy': accuracy, 'directory': str(output)})
        aggregate.append(
            {
                'variant': name,
                'seeds': args.seeds,
                'mean_accuracy': float(np.mean(values)),
                'std_accuracy': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    write_json(Path(args.output) / args.study / 'summary.json', {'runs': rows, 'aggregate': aggregate})


def run_study_cli():
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
    run_study_main(args)


def main():
    import sys
    import importlib

    commands = {
        'train': train_cli,
        'evaluate': eval_cli,
        'predict': predict_cli,
        'export': export_cli,
        'minimize': compress_cli,
        'prepare': prepare_data_cli,
        'sweep': run_experiments_cli,
        'study': run_study_cli,
        'catalog': catalog_cli,
    }
    extensions = {
        'circuit': 'networks.evaluation.circuits.analysis',
        'verify': 'networks.evaluation.circuits.equivalence',
        'dataset': 'data.partitions',
        'predictions': 'networks.evaluation.predictions',
        'runs': 'experiments.runs',
        'artifacts': 'experiments.inspection',
    }
    parser = argparse.ArgumentParser(description='Train, evaluate, and inspect learned Boolean wiring.')
    parser.add_argument('operation', choices=tuple(commands)+tuple(extensions))
    operation = parser.parse_args(sys.argv[1:2]).operation
    del sys.argv[1]
    if operation in extensions:
        importlib.import_module(extensions[operation]).main()
    else:
        commands[operation]()


if __name__ == "__main__":
    main()
