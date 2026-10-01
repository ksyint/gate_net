"""Per-gate activation and class-conditional redundancy in CUDA evaluation."""

import argparse

import torch

from data.binary import add_data_arguments, batches, load_analysis_data
from experiments.wiring import write_json


class ActivationCollector:
    def __init__(self, model, pair_limit=128):
        if pair_limit < 0:
            raise ValueError('Pair analysis width must be nonnegative')
        self.model = model
        self.pair_limit = pair_limit
        self.handles = []
        self.current_labels = None
        self.state = {}
        self.expected_batch = 0

    def __enter__(self):
        if self.handles:
            raise RuntimeError('Activation hooks are already installed')
        for index, layer in enumerate(self.model.layers):
            self.handles.append(layer.register_forward_hook(self.capture(index)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.current_labels = None

    def capture(self, index):
        def record(module, inputs, outputs):
            if self.current_labels is None:
                raise RuntimeError('Set the current targets before evaluating activations')
            values = outputs.detach().double()
            if values.ndim != 2 or len(values) != self.expected_batch:
                raise ValueError('Logic activation dimensions do not match the batch')
            if not ((values == 0) | (values == 1)).all():
                raise ValueError('A hard Boolean layer produced nonbinary activations')
            if index not in self.state:
                width = values.shape[1]
                selected = min(width, self.pair_limit)
                self.state[index] = dict(
                    samples=0,
                    ones=values.new_zeros(width),
                    class_ones=values.new_zeros(self.model.num_classes, width),
                    class_counts=torch.zeros(self.model.num_classes, dtype=torch.long, device=values.device),
                    active=torch.zeros(width + 1, dtype=torch.long, device=values.device),
                    pairs=values.new_zeros(selected, selected),
                )
            state = self.state[index]
            state['samples'] += len(values)
            state['ones'].add_(values.sum(0))
            state['class_ones'].index_add_(0, self.current_labels, values)
            state['class_counts'].add_(torch.bincount(self.current_labels, minlength=self.model.num_classes))
            state['active'].add_(torch.bincount(values.sum(-1).long(), minlength=values.shape[1] + 1))
            selected = state['pairs'].shape[0]
            state['pairs'].add_(values[:, :selected].T @ values[:, :selected])
        return record

    @torch.no_grad()
    def evaluate(self, values, labels):
        self.current_labels = labels
        self.expected_batch = len(labels)
        try:
            return self.model(values)
        finally:
            self.current_labels = None

    def report(self):
        layers = []
        for index, state in sorted(self.state.items()):
            count = state['samples']
            means = state['ones'] / count
            conditional = state['class_ones'] / state['class_counts'][:, None].clamp_min(1)
            present = state['class_counts'] > 0
            contrast = conditional[present].max(0).values - conditional[present].min(0).values
            selected = state['pairs'].shape[0]
            first = state['ones'][:selected]
            disagreements = first[:, None] + first[None] - 2 * state['pairs']
            left, right = torch.triu_indices(selected, selected, 1, device=means.device)
            same = disagreements[left, right] == 0
            opposite = disagreements[left, right] == count
            layers.append(dict(
                layer=index,
                samples=count,
                width=len(means),
                positive_rates=means.tolist(),
                constant_zero=int((state['ones'] == 0).sum()),
                constant_one=int((state['ones'] == count).sum()),
                class_positive_rates=conditional.tolist(),
                class_contrast=contrast.tolist(),
                active_count_histogram=state['active'].tolist(),
                pair_prefix_width=selected,
                identical_pairs=torch.stack((left[same], right[same]), -1).tolist(),
                complementary_pairs=torch.stack((left[opposite], right[opposite]), -1).tolist(),
            ))
        if not layers:
            raise ValueError('No activation batches were collected')
        return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--pair-width', type=int, default=128)
    parser.add_argument('--head-contributions', action='store_true')
    parser.add_argument('--feature-groups', action='store_true')
    args = parser.parse_args()
    model, loader, device, context = load_analysis_data(args)
    correct = count = 0
    with ActivationCollector(model, args.pair_width) as collector:
        for values, labels in batches(loader, device, args.max_batches):
            logits = collector.evaluate(values, labels)
            correct += int((logits.argmax(-1) == labels).sum())
            count += len(labels)
        result = collector.report()
    report = dict(context=context, accuracy=correct / count, layers=result)
    if args.head_contributions:
        report['head_contributions'] = class_head_contributions(model, loader, device, args.max_batches)
    if args.feature_groups:
        report['feature_groups'] = duplicate_feature_classes(model, loader, device, args.max_batches)
    write_json(args.output, report)


@torch.no_grad()
def duplicate_feature_classes(model, loader, device, limit=None):
    width = model.layers[-1].operator.shape[0]
    equal = torch.ones(width, width, dtype=torch.bool, device=device)
    complementary = torch.ones_like(equal)
    count = 0
    ones = torch.zeros(width, dtype=torch.long, device=device)
    for values, labels in batches(loader, device, limit):
        features = model.logic_features(values).bool()
        count += len(labels)
        ones.add_(features.sum(0))
        for start in range(0, len(features), 32):
            current = features[start:start + 32]
            agreement = current[:, :, None] == current[:, None, :]
            equal &= agreement.all(0)
            complementary &= (~agreement).all(0)
    if not count:
        raise ValueError('Feature grouping received no examples')
    assigned = set()
    groups = []
    for feature in range(width):
        if feature in assigned:
            continue
        identical = [index for index in torch.where(equal[feature])[0].tolist() if index not in assigned]
        inverse = [index for index in torch.where(complementary[feature])[0].tolist() if index not in assigned]
        assigned.update(identical + inverse)
        positive_weight = model.head.weight[:, identical].sum(-1)
        inverse_weight = model.head.weight[:, inverse].sum(-1) if inverse else torch.zeros_like(positive_weight)
        groups.append(dict(
            representative=feature,
            identical=identical,
            complementary=inverse,
            effective_weight=(positive_weight - inverse_weight).tolist(),
            bias_contribution=inverse_weight.tolist(),
            positive_rate=float(ones[feature]) / count,
            constant=bool(ones[feature] == 0 or ones[feature] == count),
        ))
    return dict(
        samples=count,
        original_features=width,
        observed_equivalence_classes=len(groups),
        groups=groups,
        domain='evaluated examples',
    )


@torch.no_grad()
def class_head_contributions(model, loader, device, limit=None):
    width = model.layers[-1].operator.shape[0]
    classes = model.num_classes
    total = torch.zeros(classes, classes, width, device=device, dtype=torch.float64)
    support = torch.zeros(classes, device=device, dtype=torch.long)
    negative = torch.zeros_like(total)
    positive = torch.zeros_like(total)
    margins = torch.zeros(classes, device=device, dtype=torch.float64)
    for values, labels in batches(loader, device, limit):
        features = model.logic_features(values).float()
        contributions = features[:, None, :] * model.head.weight[None]
        logits = contributions.sum(-1) + model.head.bias
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Class-head contributions are nonfinite')
        total.index_add_(0, labels, contributions.double())
        positive.index_add_(0, labels, contributions.clamp_min(0).double())
        negative.index_add_(0, labels, contributions.clamp_max(0).double())
        support.add_(torch.bincount(labels, minlength=classes))
        target_score = logits.gather(1, labels[:, None])[:, 0]
        competing = logits.clone()
        competing.scatter_(1, labels[:, None], -torch.inf)
        margins.index_add_(0, labels, (target_score - competing.max(-1).values).double())
    if not support.sum():
        raise ValueError('Class-head contribution analysis received no samples')
    reports = []
    for reference in range(classes):
        count = int(support[reference])
        if not count:
            continue
        average = total[reference] / count
        target = average[reference]
        order = torch.argsort(target.abs(), descending=True, stable=True)
        reports.append(dict(
            reference=reference,
            samples=count,
            mean_margin=float(margins[reference]) / count,
            mean_feature_contributions=average.tolist(),
            mean_positive_contributions=(positive[reference] / count).tolist(),
            mean_negative_contributions=(negative[reference] / count).tolist(),
            ranked_target_features=order.tolist(),
            target_class_bias=float(model.head.bias[reference]),
        ))
    return reports
