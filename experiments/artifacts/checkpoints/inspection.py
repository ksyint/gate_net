"""Inspect saved wiring tensors and record reproducible run artifact inventories."""

import argparse
import json
from pathlib import Path

import torch

from data.preparation.arrays.partitions import digest_file
from experiments.runtime.wiring import cuda_device, write_json
from networks.logic.analysis.circuits import CircuitAnalysis, validate_circuit
from networks.logic.operators.gates import GATE_NAMES


INVENTORY = 'gate-artifacts.json'


def expected_shapes(config):
    model = config['model']
    for key in ('input_dim', 'hidden_dim', 'num_classes', 'depth'):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f'Invalid checkpoint architecture field: {key}')
    widths = [model['input_dim']] + [model['hidden_dim']]*(model['depth']-1) + [model['num_classes']]
    shapes = {}
    for depth, (before, after) in enumerate(zip(widths[:-1], widths[1:])):
        for operand in ('os1', 'os2'):
            shapes[f'layers.{depth}.{operand}.weight'] = (after, before)
        shapes[f'layers.{depth}.operator'] = (after, 16)
    shapes['head.weight'] = (model['num_classes'], model['num_classes'])
    shapes['head.bias'] = (model['num_classes'],)
    return widths, shapes


def read_saved(path, device='cuda'):
    device = cuda_device(device)
    saved = torch.load(path, map_location=device, weights_only=True)
    required = {'config', 'model', 'optimizer', 'epoch', 'metrics'}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError('Expected a complete gate_net training checkpoint')
    widths, shapes = expected_shapes(saved['config'])
    state = saved['model']
    if set(state) != set(shapes):
        raise ValueError('Checkpoint model keys differ from the configured logic network')
    for name, shape in shapes.items():
        tensor = state[name]
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f'Invalid shape for {name}, expected {shape}')
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name} must contain finite floating-point parameters')
    if type(saved['epoch']) is not int or saved['epoch'] < 0:
        raise ValueError('Checkpoint epoch must be a nonnegative integer')
    history = saved.get('history', [])
    epochs = [row['epoch'] for row in history]
    if epochs != sorted(set(epochs)) or (epochs and epochs[-1] != saved['epoch']):
        raise ValueError('Saved epoch and training history disagree')
    if not {'state', 'param_groups'} <= saved['optimizer'].keys():
        raise ValueError('Checkpoint optimizer is missing state or parameter groups')
    parameters = [key for group in saved['optimizer']['param_groups'] for key in group['params']]
    if len(parameters) != len(set(parameters)) or len(parameters) != len(shapes):
        raise ValueError('Optimizer parameter membership differs from the model')
    ordered_shapes = [tuple(tensor.shape) for tensor in state.values()]
    parameter_shapes = dict(zip(parameters, ordered_shapes))
    for parameter, statistics in saved['optimizer']['state'].items():
        if parameter not in parameter_shapes:
            raise ValueError('Optimizer state refers to an unknown parameter')
        for name, value in statistics.items():
            if not torch.is_tensor(value):
                raise ValueError(f'Expected a tensor in Adam state field {name}')
            if name == 'step':
                if value.numel() != 1 or float(value) < 0:
                    raise ValueError('Adam step counters must be nonnegative scalars')
            elif tuple(value.shape) != parameter_shapes[parameter]:
                raise ValueError(f'Adam tensor shape differs from its model parameter: {name}')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite Adam state field: {name}')
    return saved, widths


def circuit_from_saved(saved):
    layers = []
    for depth in range(saved['config']['model']['depth']):
        state = saved['model']
        layers.append(dict(left=state[f'layers.{depth}.os1.weight'].argmax(-1).tolist(),
                           right=state[f'layers.{depth}.os2.weight'].argmax(-1).tolist(),
                           gate=state[f'layers.{depth}.operator'].argmax(-1).tolist()))
    circuit = dict(format='oslgn-circuit-v1', input_dim=saved['config']['model']['input_dim'],
                   gate_names=list(GATE_NAMES), layers=layers,
                   head=dict(weight=saved['model']['head.weight'].tolist(), bias=saved['model']['head.bias'].tolist()))
    validate_circuit(circuit)
    return circuit


@torch.no_grad()
def inspect_checkpoint(path, device='cuda', extract=None):
    saved, widths = read_saved(path, device)
    rows = []
    for name, value in saved['model'].items():
        tensor = value.double()
        rows.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype), elements=value.numel(),
                         minimum=float(tensor.min()), maximum=float(tensor.max()),
                         mean=float(tensor.mean()), l2_norm=float(tensor.square().sum().sqrt())))
    circuit = circuit_from_saved(saved)
    if extract:
        if Path(extract).resolve() == Path(path).resolve():
            raise ValueError('Circuit extraction cannot overwrite a checkpoint')
        write_json(extract, circuit)
    margins = []
    for depth in range(len(widths)-1):
        selectors = {}
        for suffix in ('os1.weight', 'os2.weight', 'operator'):
            values = saved['model'][f'layers.{depth}.{suffix}']
            if values.shape[-1] > 1:
                top = values.topk(2, dim=-1).values
                gap = top[:, 0]-top[:, 1]
                selectors[suffix] = dict(minimum_gap=float(gap.min()), mean_gap=float(gap.mean()), ties=int((gap == 0).sum()))
            else:
                selectors[suffix] = dict(only_candidate=True)
        margins.append(dict(layer=depth, selectors=selectors))
    return dict(path=str(Path(path).resolve()), sha256=digest_file(path), epoch=saved['epoch'],
                config=saved['config'], metrics=saved['metrics'], history_entries=len(saved.get('history', [])),
                model_parameters=sum(row['elements'] for row in rows), tensors=rows, selection_margins=margins,
                optimizer_groups=len(saved['optimizer']['param_groups']),
                optimizer_state_entries=len(saved['optimizer']['state']),
                rng_fields=[key for key in ('torch_rng', 'cuda_rng') if key in saved],
                wiring=CircuitAnalysis(circuit).summary())


@torch.no_grad()
def compare_checkpoints(first, second, device='cuda'):
    left, _ = read_saved(first, device)
    right, _ = read_saved(second, device)
    if left['config']['model'] != right['config']['model']:
        raise ValueError('Checkpoint tensor comparison requires identical architecture settings')
    rows = []
    for name, value in left['model'].items():
        one, two = value.double(), right['model'][name].double()
        movement = two-one
        norm = float(one.square().sum().sqrt())
        change = float(movement.square().sum().sqrt())
        row = dict(name=name, maximum_absolute_change=float(movement.abs().max()),
                   l2_change=change, relative_l2_change=change/norm if norm else None)
        if name.startswith('layers.'):
            changed = one.argmax(-1) != two.argmax(-1)
            row.update(changed_selections=int(changed.sum()), units=changed.numel())
        rows.append(row)
    return dict(first=dict(path=str(first), epoch=left['epoch'], sha256=digest_file(first)),
                second=dict(path=str(second), epoch=right['epoch'], sha256=digest_file(second)), tensors=rows)


def inventory_files(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Artifact inventories require regular files and folders: {path}')
        if not path.is_file() or path.name == INVENTORY or path.name.endswith('.partial'):
            continue
        files.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest_file(path)))
    if not files:
        raise ValueError('The artifact directory is empty')
    return files


def write_inventory(directory):
    root = Path(directory).resolve()
    files = inventory_files(root)
    result = dict(format='gate-artifacts-v1', directory=str(root), files=files,
                  total_bytes=sum(row['bytes'] for row in files))
    metrics = root/'metrics.json'
    if metrics.is_file():
        run = json.loads(metrics.read_text())
        result['run'] = dict(config=run.get('config'), task=run.get('task'),
                             history_entries=len(run.get('history', [])))
    write_json(root/INVENTORY, result)
    return result


def verify_inventory(directory):
    root = Path(directory).resolve()
    saved = json.loads((root/INVENTORY).read_text())
    if saved.get('format') != 'gate-artifacts-v1':
        raise ValueError('Unknown artifact inventory format')
    expected = {row['path']: row for row in saved['files']}
    if len(expected) != len(saved['files']):
        raise ValueError('Inventory paths must be unique')
    actual = {row['path']: row for row in inventory_files(root)}
    changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    added, missing = sorted(actual.keys()-expected.keys()), sorted(expected.keys()-actual.keys())
    return dict(directory=str(root), files=len(actual), changed=changed, added=added, missing=missing,
                valid=not (changed or added or missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('inspect')
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--device', default='cuda')
    command.add_argument('--extract')
    command.add_argument('--output', required=True)
    command = commands.add_parser('compare')
    command.add_argument('--first', required=True)
    command.add_argument('--second', required=True)
    command.add_argument('--device', default='cuda')
    command.add_argument('--output', required=True)
    for name in ('inventory', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--directory', required=True)
    args = parser.parse_args()
    if args.operation in ('inspect', 'compare'):
        sources = [args.checkpoint] if args.operation == 'inspect' else [args.first, args.second]
        if Path(args.output).resolve() in {Path(path).resolve() for path in sources}:
            parser.error('The report must not overwrite a source checkpoint')
        if args.operation == 'inspect':
            if args.extract and Path(args.extract).resolve() == Path(args.output).resolve():
                parser.error('Use separate paths for the extracted circuit and report')
            result = inspect_checkpoint(args.checkpoint, args.device, args.extract)
        else:
            result = compare_checkpoints(args.first, args.second, args.device)
        write_json(args.output, result)
    else:
        result = write_inventory(args.directory) if args.operation == 'inventory' else verify_inventory(args.directory)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get('valid') is False:
        raise SystemExit('Artifact inventory differs from the saved files')
