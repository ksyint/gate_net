"""CUDA dataset summaries for the binary inputs consumed by a saved circuit."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.partitions import digest_file
from experiments.wiring import cuda_device, load_dataset, restore_model, write_json


def add_data_arguments(parser):
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data')
    parser.add_argument('--data-root')
    parser.add_argument('--split', choices=('train', 'val', 'test'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)


def load_analysis_data(args):
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('The batch limit must be positive')
    device = cuda_device(args.device)
    model, config = restore_model(args.checkpoint, device)
    if args.data_root:
        config['data']['root'] = args.data_root
    split = args.split or ('test' if config['data']['dataset'] == 'mnist' and not args.data else 'val')
    if args.data and split == 'test':
        raise ValueError('NPZ archives expose their held-out partition as val')
    dataset = load_dataset(config, split, args.data, download=not args.offline)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    context = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=digest_file(args.checkpoint),
        data_sha256=digest_file(args.data) if args.data else None,
        split=split,
        data=config['data'],
        model=config['model'],
        training=config['train'],
        batch_limit=args.max_batches,
    )
    return model, loader, device, context


def batches(loader, device, limit=None):
    for index, (images, targets) in enumerate(loader):
        if limit is not None and index >= limit:
            break
        values = images.flatten(1).to(device, non_blocking=True)
        labels = targets.to(device, non_blocking=True)
        if not torch.isfinite(values).all() or not ((values == 0) | (values == 1)).all():
            raise ValueError('Circuit analysis requires finite binary inputs')
        yield values, labels


class BinarySummary:
    def __init__(self, input_dim, classes, device):
        self.count = 0
        self.ones = torch.zeros(input_dim, dtype=torch.float64, device=device)
        self.by_class = torch.zeros(classes, input_dim, dtype=torch.float64, device=device)
        self.class_counts = torch.zeros(classes, dtype=torch.long, device=device)
        self.active_histogram = torch.zeros(input_dim + 1, dtype=torch.long, device=device)
        self.positive_pairs = None

    @torch.no_grad()
    def update(self, values, labels, pairs=False):
        if values.shape[1] != len(self.ones):
            raise ValueError('Binary input dimension changed between batches')
        if labels.min() < 0 or labels.max() >= len(self.class_counts):
            raise ValueError('Class IDs are outside the configured range')
        self.ones.add_(values.double().sum(0))
        self.by_class.index_add_(0, labels, values.double())
        self.class_counts.add_(torch.bincount(labels, minlength=len(self.class_counts)))
        self.active_histogram.add_(torch.bincount(values.sum(-1).long(), minlength=len(self.ones) + 1))
        if pairs:
            if self.positive_pairs is None:
                self.positive_pairs = torch.zeros(
                    len(self.ones), len(self.ones), dtype=torch.float64, device=values.device,
                )
            self.positive_pairs.add_(values.double().T @ values.double())
        self.count += len(labels)

    def report(self, top=30):
        if self.count == 0 or top < 0:
            raise ValueError('Summary requires observations and a nonnegative ranking size')
        probabilities = self.ones / self.count
        epsilon = torch.finfo(torch.float64).eps
        entropy = -(probabilities * probabilities.clamp_min(epsilon).log2())
        entropy -= (1 - probabilities) * (1 - probabilities).clamp_min(epsilon).log2()
        order = torch.argsort(entropy, descending=True, stable=True)[:top]
        output = dict(
            samples=self.count,
            input_dim=len(self.ones),
            class_counts=self.class_counts.tolist(),
            class_positive_rates=(self.by_class / self.class_counts[:, None].clamp_min(1)).tolist(),
            positive_rates=probabilities.tolist(),
            entropy_bits=entropy.tolist(),
            constant_zero=int((self.ones == 0).sum()),
            constant_one=int((self.ones == self.count).sum()),
            active_count_histogram=self.active_histogram.tolist(),
            highest_entropy_inputs=order.tolist(),
        )
        if self.positive_pairs is not None:
            variance = probabilities * (1 - probabilities)
            covariance = self.positive_pairs / self.count - probabilities[:, None] * probabilities[None]
            denominator = torch.sqrt(variance[:, None] * variance[None])
            correlation = torch.where(denominator > 0, covariance / denominator.clamp_min(epsilon), 0.)
            row, col = torch.triu_indices(len(self.ones), len(self.ones), 1, device=self.ones.device)
            selected = torch.argsort(correlation[row, col].abs(), descending=True, stable=True)[:top]
            output['correlated_inputs'] = [
                dict(first=int(row[i]), second=int(col[i]), correlation=float(correlation[row[i], col[i]]))
                for i in selected.tolist()
            ]
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--thresholds', nargs='+', type=float)
    args = parser.parse_args()
    model, loader, device, context = load_analysis_data(args)
    summary = BinarySummary(model.input_dim, model.num_classes, device)
    for values, labels in batches(loader, device, args.max_batches):
        summary.update(values, labels, args.pairs)
    report = dict(context=context, binary=summary.report(args.top), information=mutual_information(summary))
    if args.thresholds:
        config = dict(model=context['model'], data=context['data'], train=context['training'])
        report['thresholds'] = threshold_curve(config, context['split'], args.thresholds, args.data, device, args.offline)
    write_json(args.output, report)


@torch.no_grad()
def mutual_information(summary):
    if not summary.count:
        raise ValueError('Input-label information needs observed binary examples')
    positive = summary.by_class / summary.count
    negative = (summary.class_counts[:, None] - summary.by_class) / summary.count
    joint = torch.stack((negative, positive), -1)
    p_class = summary.class_counts.double() / summary.count
    p_bit = torch.stack((1 - summary.ones / summary.count, summary.ones / summary.count), -1)
    independent = p_class[:, None, None] * p_bit[None]
    valid = joint > 0
    logarithm = torch.zeros_like(joint)
    logarithm[valid] = (joint[valid] / independent[valid].clamp_min(1e-30)).log2()
    information = (joint * logarithm).sum((0, 2))
    class_entropy = -(p_class[p_class > 0] * p_class[p_class > 0].log2()).sum()
    ordering = torch.argsort(information, descending=True, stable=True)
    return dict(
        class_entropy_bits=float(class_entropy),
        input_information_bits=information.tolist(),
        ranked_inputs=ordering.tolist(),
        normalized_information=(information / class_entropy.clamp_min(1e-30)).tolist(),
    )


@torch.no_grad()
def threshold_curve(config, split, thresholds, data, device, offline=False):
    from data.loaders import mnist_read_split, boolean_read_split, npz_read_split
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError('Input thresholds must lie in [0,1]')
    if data:
        images, labels = npz_read_split(data, split)
    elif config['data']['dataset'] == 'mnist':
        images, labels = mnist_read_split(config, split, download=not offline)
    else:
        images, labels = boolean_read_split(config, split)
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Threshold curves require finite normalized source pixels')
    rows = []
    for threshold in sorted(set(thresholds)):
        summary = BinarySummary(config['model']['input_dim'], config['model']['num_classes'], device)
        for start in range(0, len(images), 256):
            values = (images[start:start + 256].to(device) > threshold).float()
            summary.update(values, labels[start:start + 256].to(device))
        information = mutual_information(summary)
        rows.append(dict(
            threshold=threshold,
            mean_positive_fraction=float(summary.ones.sum()) / (summary.count * len(summary.ones)),
            constant_inputs=int(((summary.ones == 0) | (summary.ones == summary.count)).sum()),
            information=information,
        ))
    return rows
