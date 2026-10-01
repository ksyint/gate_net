"""Verify extracted circuits and enumerate feature truth tables on CUDA."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from networks.logic.circuits.analysis import CircuitAnalysis, read_circuit, validate_circuit
from experiments.runtime.wiring import cuda_device, restore_model, write_json


class CudaCircuit:
    def __init__(self, circuit, device='cuda'):
        self.widths = validate_circuit(circuit)
        self.device = cuda_device(device)
        self.layers = [{key: torch.tensor(value, dtype=torch.long, device=self.device)
                        for key, value in layer.items()} for layer in circuit['layers']]
        self.weight = torch.tensor(circuit['head']['weight'], dtype=torch.float32, device=self.device)
        self.bias = torch.tensor(circuit['head']['bias'], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def __call__(self, inputs):
        if inputs.device != self.weight.device:
            raise ValueError('Circuit inputs and tensors must use the same CUDA device')
        if inputs.ndim != 2 or inputs.shape[1] != self.widths[0]:
            raise ValueError('Circuit inputs require shape [batch, input_dim]')
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError('Circuit verification requires binary inputs')
        features = inputs.long()
        for layer in self.layers:
            left, right = features[:, layer['left']], features[:, layer['right']]
            features = (layer['gate'].unsqueeze(0) >> (3 - 2 * left - right)) & 1
        return features, features.float() @ self.weight.T + self.bias


def input_batches(input_dim, device, count=4096, batch_size=256, seed=42, data=None, split='val', threshold=.5):
    if count < 1 or batch_size < 1 or not 0 <= threshold <= 1:
        raise ValueError('Samples and batch size must be positive with threshold in [0,1]')
    device = cuda_device(device)
    if data:
        with np.load(data, allow_pickle=False) as archive:
            values = archive[f'x_{split}']
        values = values.reshape(len(values), -1)
        if values.shape[1] != input_dim or not len(values) or not np.isfinite(values).all():
            raise ValueError('NPZ inputs disagree with the circuit dimension or contain nonfinite values')
        count = min(count, len(values))
        for offset in range(0, count, batch_size):
            yield (torch.as_tensor(values[offset:min(offset+batch_size, count)], device=device) > threshold).float()
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        for offset in range(0, count, batch_size):
            size = min(batch_size, count-offset)
            yield torch.randint(0, 2, (size, input_dim), generator=generator, device=device).float()


@torch.no_grad()
def verify_checkpoint(checkpoint, circuit=None, device='cuda', **inputs):
    model, config = restore_model(checkpoint, device)
    source = read_circuit(circuit) if circuit else model.export_circuit()
    evaluator = CudaCircuit(source, device)
    if evaluator.widths[0] != config['model']['input_dim']:
        raise ValueError('The extracted circuit and checkpoint have different input widths')
    if evaluator.weight.shape[0] != config['model']['num_classes']:
        raise ValueError('The extracted circuit and checkpoint have different class counts')
    count = disagreements = feature_errors = 0
    maximum, total_error = 0., 0.
    exact_width = evaluator.widths[-1] == config['model']['num_classes']
    for images in input_batches(evaluator.widths[0], device, **inputs):
        features, logits = evaluator(images)
        expected_features = model.logic_features(images)
        expected = model.head(expected_features)
        if exact_width:
            feature_errors += int((features != expected_features).sum())
        difference = (logits-expected).abs()
        maximum = max(maximum, float(difference.max()))
        total_error += float(difference.sum())
        disagreements += int((logits.argmax(-1) != expected.argmax(-1)).sum())
        count += len(images)
    return dict(samples=count, feature_comparison=exact_width,
                feature_mismatches=feature_errors if exact_width else None,
                class_disagreements=disagreements, maximum_logit_error=maximum,
                mean_logit_error=total_error/(count*evaluator.weight.shape[0]),
                logits_close=maximum <= 1e-5, checkpoint=str(Path(checkpoint).resolve()),
                input_source=str(Path(inputs['data']).resolve()) if inputs.get('data') else 'seeded_binary',
                seed=inputs.get('seed', 42))


@torch.no_grad()
def compare_circuits(first, second, device='cuda', **inputs):
    one, two = CudaCircuit(read_circuit(first), device), CudaCircuit(read_circuit(second), device)
    if one.widths[0] != two.widths[0] or one.weight.shape[0] != two.weight.shape[0]:
        raise ValueError('Circuit comparison requires equal input and class dimensions')
    equal_features = one.widths[-1] == two.widths[-1]
    samples = disagreements = differences = 0
    maximum = 0.
    for images in input_batches(one.widths[0], device, **inputs):
        left, left_logits = one(images)
        right, right_logits = two(images)
        if equal_features:
            differences += int((left != right).sum())
        disagreements += int((left_logits.argmax(-1) != right_logits.argmax(-1)).sum())
        maximum = max(maximum, float((left_logits-right_logits).abs().max()))
        samples += len(images)
    return dict(first=str(first), second=str(second), samples=samples,
                feature_mismatches=differences if equal_features else None,
                class_disagreements=disagreements, maximum_logit_error=maximum,
                logits_close=maximum <= 1e-5)


@torch.no_grad()
def truth_table(circuit, feature, destination, device='cuda', max_support=16, batch_size=512):
    source = read_circuit(circuit)
    analysis = CircuitAnalysis(source)
    cone = analysis.cone(feature)
    support = cone['input_support']
    if not 0 <= max_support <= 24 or len(support) > max_support:
        raise ValueError(f'Feature needs {len(support)} inputs, increase --max-support up to 24 if appropriate')
    if batch_size < 1:
        raise ValueError('Truth-table batch size must be positive')
    destination = Path(destination).resolve()
    if destination.exists() or Path(str(destination)+'.json').exists() or destination == Path(circuit).resolve():
        raise FileExistsError('Choose a new NPZ destination for the truth table')
    evaluator = CudaCircuit(source, device)
    device = evaluator.weight.device
    positions = torch.arange(len(support), device=device)
    coordinates = torch.tensor(support, dtype=torch.long, device=device)
    assignments, outputs = [], []
    count = 2 ** len(support)
    for offset in range(0, count, batch_size):
        integers = torch.arange(offset, min(offset+batch_size, count), device=device)
        bits = ((integers[:, None] >> positions) & 1).float()
        inputs = torch.zeros((len(integers), source['input_dim']), device=device)
        inputs[:, coordinates] = bits
        features, _ = evaluator(inputs)
        assignments.append(bits.byte().cpu().numpy())
        outputs.append(features[:, feature].byte().cpu().numpy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        np.savez_compressed(stream, input_coordinates=np.array(support, dtype=np.int64),
                            assignments=np.concatenate(assignments), outputs=np.concatenate(outputs))
    report = dict(circuit=str(Path(circuit).resolve()), feature=feature, support=support,
                  assignments=count, output=str(destination), little_endian_assignments=True,
                  ones=int(sum(array.sum() for array in outputs)))
    write_json(str(destination)+'.json', report)
    return report


def common_arguments(parser):
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--output', required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('checkpoint')
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--circuit')
    common_arguments(command)
    command = commands.add_parser('compare')
    command.add_argument('--first', required=True)
    command.add_argument('--second', required=True)
    common_arguments(command)
    command = commands.add_parser('truth-table')
    command.add_argument('--circuit', required=True)
    command.add_argument('--feature', type=int, required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--device', default='cuda')
    command.add_argument('--max-support', type=int, default=16)
    command.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    if args.operation == 'truth-table':
        report = truth_table(args.circuit, args.feature, args.output, args.device, args.max_support, args.batch_size)
    else:
        protected = [getattr(args, name, None) for name in ('checkpoint', 'circuit', 'first', 'second', 'data')]
        if Path(args.output).resolve() in {Path(path).resolve() for path in protected if path}:
            raise ValueError('Verification reports must not overwrite their input artifacts')
        options = dict(count=args.samples, batch_size=args.batch_size, seed=args.seed,
                       data=args.data, split=args.split, threshold=args.threshold)
        if args.operation == 'checkpoint':
            report = verify_checkpoint(args.checkpoint, args.circuit, args.device, **options)
        else:
            report = compare_circuits(args.first, args.second, args.device, **options)
        write_json(args.output, report)
        if report['class_disagreements'] or report.get('feature_mismatches') or not report['logits_close']:
            raise SystemExit('Circuit comparison found differences, inspect the saved report')
    print(json.dumps(report, indent=2))
