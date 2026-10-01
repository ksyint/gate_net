"""Inspect STE operand and operator gradients on a CUDA task minibatch."""

import argparse

import torch
import torch.nn.functional as F

from data.binary import add_data_arguments, batches, load_analysis_data
from experiments.wiring import write_json
from networks.monitoring import selector_items, summarize_values


def selector_gradient(weight):
    gradient = weight.grad
    if gradient is None:
        return dict(present=False, parameters=weight.numel())
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('STE gradient contains nonfinite values')
    selected = weight.detach().argmax(-1)
    winner = gradient.gather(1, selected[:, None])[:, 0]
    norm = gradient.float().norm(dim=-1)
    projected = gradient.detach().clone()
    projected.scatter_(1, selected[:, None], 0.)
    return dict(
        present=True,
        parameters=weight.numel(),
        nonzero=int(torch.count_nonzero(gradient)),
        norm=float(gradient.float().norm()),
        selected_gradient=summarize_values(winner),
        alternative_gradient=summarize_values(projected),
        row_gradient_norm=summarize_values(norm),
    )


def gradient_probe(model, values, labels, detach):
    original = [layer.detach_operands for layer in model.layers]
    previous_mode = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        for layer in model.layers:
            layer.detach_operands = detach
        logits = model(values)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Gradient probe loss is nonfinite')
        loss.backward()
        rows = []
        for depth, kind, weight in selector_items(model):
            rows.append(dict(layer=depth, selector=kind, **selector_gradient(weight)))
        result = dict(
            detached_operands=detach,
            loss=float(loss.detach()),
            accuracy=float((logits.detach().argmax(-1) == labels).float().mean()),
            selectors=rows,
            head_weight=selector_gradient(model.head.weight),
        )
    finally:
        for layer, setting in zip(model.layers, original):
            layer.detach_operands = setting
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
    return result


def compare_probes(first, second):
    mapping = {(row['layer'], row['selector']): row for row in second['selectors']}
    rows = []
    for row in first['selectors']:
        other = mapping[(row['layer'], row['selector'])]
        norm = row.get('norm', 0.)
        other_norm = other.get('norm', 0.)
        rows.append(dict(
            layer=row['layer'],
            selector=row['selector'],
            enabled_norm=norm,
            detached_norm=other_norm,
            detached_to_enabled_ratio=other_norm / norm if norm else None,
            enabled_present=row['present'],
            detached_present=other['present'],
        ))
    return dict(
        loss_difference=second['loss'] - first['loss'],
        accuracy_difference=second['accuracy'] - first['accuracy'],
        selectors=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--step-sizes', nargs='+', type=float)
    parser.add_argument('--class-alignment', action='store_true')
    args = parser.parse_args()
    model, loader, device, context = load_analysis_data(args)
    reports = []
    limit = args.max_batches or 1
    for index, (values, labels) in enumerate(batches(loader, device, limit)):
        enabled = gradient_probe(model, values, labels, False)
        detached = gradient_probe(model, values, labels, True)
        reports.append(dict(
            batch=index,
            samples=len(labels),
            enabled=enabled,
            detached=detached,
            comparison=compare_probes(enabled, detached),
            counterfactual=counterfactual_steps(model, values, labels, args.step_sizes) if args.step_sizes else None,
            class_alignment=per_class_gradient_alignment(model, values, labels) if args.class_alignment else None,
        ))
    if not reports:
        raise ValueError('The gradient probe received no minibatches')
    write_json(args.output, dict(context=context, batches=reports))


def counterfactual_steps(model, values, labels, rates, detach=False):
    if not rates or min(rates) <= 0 or len(set(rates)) != len(rates):
        raise ValueError('Gradient step sizes must be distinct positive values')
    old_mode = model.training
    old_detach = [layer.detach_operands for layer in model.layers]
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        for layer in model.layers:
            layer.detach_operands = detach
        baseline_logits = model(values)
        baseline_loss = F.cross_entropy(baseline_logits, labels)
        baseline_loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        selectors = {(depth, kind): weight.detach().argmax(-1)
                     for depth, kind, weight in selector_items(model)}
        gradient_norm_squared = sum(float(gradient.float().square().sum()) for gradient in gradients.values())
        rows = []
        with torch.no_grad():
            for rate in rates:
                for name, parameter in model.named_parameters():
                    parameter.copy_(originals[name])
                    if name in gradients:
                        parameter.add_(gradients[name], alpha=-rate)
                logits = model(values)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('A counterfactual gradient step produced a nonfinite loss')
                changes = []
                for depth, kind, weight in selector_items(model):
                    before = selectors[(depth, kind)]
                    after = weight.argmax(-1)
                    changes.append(dict(layer=depth, selector=kind, changed=int((before != after).sum()), units=len(before)))
                actual = float(loss - baseline_loss.detach())
                predicted = -rate * gradient_norm_squared
                rows.append(dict(
                    learning_rate=rate,
                    baseline_loss=float(baseline_loss.detach()),
                    updated_loss=float(loss),
                    actual_loss_change=actual,
                    linearized_loss_change=predicted,
                    realized_fraction=actual / predicted if predicted else None,
                    accuracy=float((logits.argmax(-1) == labels).float().mean()),
                    prediction_changes=int((logits.argmax(-1) != baseline_logits.detach().argmax(-1)).sum()),
                    selector_changes=changes,
                ))
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(originals[name])
        for layer, value in zip(model.layers, old_detach):
            layer.detach_operands = value
        model.zero_grad(set_to_none=True)
        model.train(old_mode)
    return dict(detached_operands=detach, gradient_norm_squared=gradient_norm_squared, steps=rows)


def per_class_gradient_alignment(model, values, labels):
    old_mode = model.training
    model.train()
    selected = [(depth, kind, weights) for depth, kind, weights in selector_items(model)]
    accumulators = []
    classes = labels.unique(sorted=True).tolist()
    try:
        for category in classes:
            model.zero_grad(set_to_none=True)
            chosen = labels == category
            loss = F.cross_entropy(model(values[chosen]), labels[chosen])
            loss.backward()
            rows = []
            for depth, kind, weights in selected:
                gradient = weights.grad
                if gradient is None:
                    rows.append(None)
                    continue
                flat = gradient.detach().float().flatten()
                rows.append(flat.clone())
            accumulators.append(rows)
        result = []
        for selector_index, (depth, kind, _) in enumerate(selected):
            present = [index for index in range(len(classes)) if accumulators[index][selector_index] is not None]
            if not present:
                result.append(dict(layer=depth, selector=kind, classes=[], cosine=[]))
                continue
            matrix = torch.stack([accumulators[index][selector_index] for index in present])
            normalized = F.normalize(matrix, dim=-1)
            result.append(dict(
                layer=depth,
                selector=kind,
                classes=[classes[index] for index in present],
                norms=matrix.norm(dim=-1).tolist(),
                cosine=(normalized @ normalized.T).tolist(),
            ))
    finally:
        model.zero_grad(set_to_none=True)
        model.train(old_mode)
    return result
