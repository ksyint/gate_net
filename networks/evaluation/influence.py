"""Exact single-bit prediction effects on held-out binary inputs."""

import argparse

import torch
import torch.nn.functional as F

from data.binary import add_data_arguments, batches, load_analysis_data
from experiments.wiring import write_json


def select_inputs(expression, input_dim):
    if expression == 'all':
        return list(range(input_dim))
    values = []
    for item in expression.split(','):
        if ':' in item:
            start, stop = map(int, item.split(':'))
            values.extend(range(start, stop))
        else:
            values.append(int(item))
    if not values or len(set(values)) != len(values):
        raise ValueError('Input selection must be nonempty and contain no duplicates')
    if min(values) < 0 or max(values) >= input_dim:
        raise ValueError('Input index is outside the circuit')
    return values


@torch.no_grad()
def single_bit_effects(model, loader, device, indices, chunk=16, limit=None):
    if chunk < 1:
        raise ValueError('Intervention chunk size must be positive')
    size = len(indices)
    changed = torch.zeros(size, dtype=torch.long, device=device)
    harmed = torch.zeros_like(changed)
    helped = torch.zeros_like(changed)
    magnitude = torch.zeros(size, dtype=torch.float64, device=device)
    loss_delta = torch.zeros_like(magnitude)
    class_harmed = torch.zeros(size, model.num_classes, dtype=torch.long, device=device)
    class_samples = torch.zeros(model.num_classes, dtype=torch.long, device=device)
    clean_correct = samples = 0
    for values, labels in batches(loader, device, limit):
        logits = model(values)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Baseline logits are nonfinite')
        baseline = logits.argmax(-1)
        baseline_correct = baseline == labels
        original_loss = F.cross_entropy(logits, labels, reduction='none')
        samples += len(labels)
        clean_correct += int(baseline_correct.sum())
        class_samples.add_(torch.bincount(labels, minlength=model.num_classes))
        for start in range(0, size, chunk):
            selected = indices[start:start + chunk]
            replicas = values[None].expand(len(selected), -1, -1).clone()
            for index, feature in enumerate(selected):
                replicas[index, :, feature] = 1 - replicas[index, :, feature]
            output = model(replicas.flatten(0, 1)).reshape(len(selected), len(values), -1)
            if not torch.isfinite(output).all():
                raise FloatingPointError('Single-bit logits are nonfinite')
            predictions = output.argmax(-1)
            correct = predictions == labels[None]
            affected = baseline_correct[None] & ~correct
            recovered = ~baseline_correct[None] & correct
            stop = start + len(selected)
            changed[start:stop].add_((predictions != baseline[None]).sum(-1))
            harmed[start:stop].add_(affected.sum(-1))
            helped[start:stop].add_(recovered.sum(-1))
            magnitude[start:stop].add_((output - logits[None]).abs().mean(-1).double().sum(-1))
            losses = F.cross_entropy(
                output.flatten(0, 1), labels.repeat(len(selected)), reduction='none',
            ).reshape(len(selected), len(values))
            loss_delta[start:stop].add_((losses - original_loss[None]).double().sum(-1))
            for offset in range(len(selected)):
                class_harmed[start + offset].add_(torch.bincount(
                    labels[affected[offset]], minlength=model.num_classes,
                ))
    if not samples:
        raise ValueError('No examples were available for input influence')
    rows = []
    for position, feature in enumerate(indices):
        rows.append(dict(
            input=feature,
            changed_predictions=int(changed[position]),
            prediction_change_rate=float(changed[position]) / samples,
            harmed=int(harmed[position]),
            helped=int(helped[position]),
            accuracy_delta=float(helped[position] - harmed[position]) / samples,
            mean_absolute_logit_change=float(magnitude[position]) / samples,
            mean_loss_change=float(loss_delta[position]) / samples,
            class_harmed=class_harmed[position].tolist(),
        ))
    return dict(
        samples=samples,
        clean_accuracy=clean_correct / samples,
        class_samples=class_samples.tolist(),
        input_count=size,
        effects=rows,
        ranked_inputs=[row['input'] for row in sorted(rows, key=lambda row: (-row['harmed'], row['input']))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--inputs', default='all', help='all, comma-separated indices, or start:stop ranges')
    parser.add_argument('--chunk-size', type=int, default=16)
    args = parser.parse_args()
    model, loader, device, context = load_analysis_data(args)
    indices = select_inputs(args.inputs, model.input_dim)
    report = single_bit_effects(model, loader, device, indices, args.chunk_size, args.max_batches)
    write_json(args.output, dict(context=context, **report))
