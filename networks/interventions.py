"""Measure the held-out effect of clamping selected intermediate Boolean units."""

import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from data.binary import add_data_arguments, batches, load_analysis_data
from experiments.wiring import write_json


def parse_interventions(expressions, model):
    specifications = []
    known = set()
    for expression in expressions:
        parts = expression.split(':')
        if len(parts) != 3:
            raise ValueError('An intervention has layer:unit:value form')
        layer, unit, value = map(int, parts)
        if not 0 <= layer < len(model.layers):
            raise ValueError('Intervention layer is outside the network')
        if not 0 <= unit < model.layers[layer].operator.shape[0]:
            raise ValueError('Intervention unit is outside the layer')
        if value not in (0, 1):
            raise ValueError('Clamped values must be zero or one')
        if (layer, unit) in known:
            raise ValueError('A unit can be clamped only once in an intervention')
        known.add((layer, unit))
        specifications.append(dict(layer=layer, unit=unit, value=value))
    if not specifications:
        raise ValueError('Choose at least one logic unit to clamp')
    return specifications


@contextmanager
def clamped_units(model, specifications):
    groups = {}
    for item in specifications:
        groups.setdefault(item['layer'], []).append(item)
    handles = []
    def hook_for(items):
        def replace(module, inputs, outputs):
            result = outputs.clone()
            for item in items:
                result[:, item['unit']] = item['value']
            return result
        return replace
    try:
        for layer, items in groups.items():
            handles.append(model.layers[layer].register_forward_hook(hook_for(items)))
        yield
    finally:
        for handle in handles:
            handle.remove()


class InterventionResult:
    def __init__(self, classes, device):
        self.samples = 0
        self.clean_correct = self.changed_correct = 0
        self.harmed = self.helped = self.changed = 0
        self.loss_before = self.loss_after = 0.
        self.margin_delta = 0.
        self.class_count = torch.zeros(classes, device=device, dtype=torch.long)
        self.class_harmed = torch.zeros_like(self.class_count)
        self.class_helped = torch.zeros_like(self.class_count)
        self.transitions = torch.zeros(classes, classes, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, clean, changed, labels):
        if clean.shape != changed.shape or not torch.isfinite(changed).all():
            raise ValueError('Intervention logits are nonfinite or have inconsistent shapes')
        before = clean.argmax(-1)
        after = changed.argmax(-1)
        old_correct = before == labels
        new_correct = after == labels
        harmed = old_correct & ~new_correct
        helped = ~old_correct & new_correct
        self.samples += len(labels)
        self.clean_correct += int(old_correct.sum())
        self.changed_correct += int(new_correct.sum())
        self.harmed += int(harmed.sum())
        self.helped += int(helped.sum())
        self.changed += int((before != after).sum())
        self.loss_before += float(F.cross_entropy(clean, labels, reduction='sum'))
        self.loss_after += float(F.cross_entropy(changed, labels, reduction='sum'))
        index = torch.arange(len(labels), device=labels.device)
        self.margin_delta += float((changed[index, labels] - clean[index, labels]).sum())
        classes = len(self.class_count)
        self.class_count.add_(torch.bincount(labels, minlength=classes))
        self.class_harmed.add_(torch.bincount(labels[harmed], minlength=classes))
        self.class_helped.add_(torch.bincount(labels[helped], minlength=classes))
        self.transitions.add_(torch.bincount(before * classes + after, minlength=classes ** 2).reshape(classes, classes))

    def report(self):
        if not self.samples:
            raise ValueError('Intervention evaluation received no examples')
        return dict(
            samples=self.samples,
            baseline_accuracy=self.clean_correct / self.samples,
            intervention_accuracy=self.changed_correct / self.samples,
            harmed=self.harmed,
            helped=self.helped,
            changed_predictions=self.changed,
            baseline_loss=self.loss_before / self.samples,
            intervention_loss=self.loss_after / self.samples,
            mean_target_logit_change=self.margin_delta / self.samples,
            class_count=self.class_count.tolist(),
            class_harmed=self.class_harmed.tolist(),
            class_helped=self.class_helped.tolist(),
            prediction_transitions=self.transitions.tolist(),
            transition_axes=['baseline_prediction', 'intervention_prediction'],
        )


@torch.no_grad()
def measure(model, loader, device, specifications, limit=None):
    meter = InterventionResult(model.num_classes, device)
    for values, labels in batches(loader, device, limit):
        clean = model(values)
        with clamped_units(model, specifications):
            changed = model(values)
        meter.update(clean, changed, labels)
    return dict(interventions=specifications, **meter.report())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--clamp', nargs='+', required=True, help='Layer:unit:value entries')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--pairs', action='store_true')
    args = parser.parse_args()
    model, loader, device, context = load_analysis_data(args)
    specifications = parse_interventions(args.clamp, model)
    groups = [[item] for item in specifications] if args.separate else [specifications]
    reports = [measure(model, loader, device, group, args.max_batches) for group in groups]
    interactions = pair_interactions(model, loader, device, specifications, args.max_batches) if args.pairs else None
    write_json(args.output, dict(context=context, measurements=reports, pair_interactions=interactions))


@torch.no_grad()
def pair_interactions(model, loader, device, specifications, limit=None):
    if len(specifications) < 2:
        raise ValueError('Pair interactions require at least two different units')
    pairs = [(a, b) for a in range(len(specifications)) for b in range(a + 1, len(specifications))]
    totals = {pair: dict(samples=0, excess_loss=0., joint_harm=0, individual_harm=0,
                        joint_rescue=0, maximum_logit_interaction=0.) for pair in pairs}
    for values, labels in batches(loader, device, limit):
        baseline = model(values)
        baseline_loss = F.cross_entropy(baseline, labels, reduction='none')
        correct = baseline.argmax(-1) == labels
        single = []
        for item in specifications:
            with clamped_units(model, [item]):
                output = model(values)
            single.append(output)
        for first, second in pairs:
            with clamped_units(model, [specifications[first], specifications[second]]):
                joint = model(values)
            one, two = single[first], single[second]
            losses = F.cross_entropy(joint, labels, reduction='none')
            one_loss = F.cross_entropy(one, labels, reduction='none')
            two_loss = F.cross_entropy(two, labels, reduction='none')
            interaction = joint - one - two + baseline
            first_correct = one.argmax(-1) == labels
            second_correct = two.argmax(-1) == labels
            both_correct = joint.argmax(-1) == labels
            total = totals[(first, second)]
            total['samples'] += len(labels)
            total['excess_loss'] += float((losses - one_loss - two_loss + baseline_loss).sum())
            total['joint_harm'] += int((correct & first_correct & second_correct & ~both_correct).sum())
            total['individual_harm'] += int((correct & (~first_correct | ~second_correct)).sum())
            total['joint_rescue'] += int((~first_correct & ~second_correct & both_correct).sum())
            total['maximum_logit_interaction'] = max(total['maximum_logit_interaction'], float(interaction.abs().max()))
    results = []
    for (first, second), values in totals.items():
        if not values['samples']:
            raise ValueError('No examples were evaluated for intervention interactions')
        values['mean_excess_loss'] = values.pop('excess_loss') / values['samples']
        results.append(dict(first=specifications[first], second=specifications[second], **values))
    return results
