"""Evaluate discrete bit-flip sensitivity without changing the training distribution."""

import argparse
import math

import torch
import torch.nn.functional as F

from data.binary import add_data_arguments, batches, load_analysis_data
from experiments.wiring import write_json


def perturb(values, amount, generator, mode='bernoulli'):
    if not 0 <= amount <= 1 or not math.isfinite(amount):
        raise ValueError('The perturbation fraction must lie in [0,1]')
    if mode == 'bernoulli':
        mask = torch.rand(values.shape, device=values.device, generator=generator) < amount
    elif mode == 'fixed':
        size = round(values.shape[1] * amount)
        random = torch.rand(values.shape, device=values.device, generator=generator)
        positions = random.argsort(-1)[:, :size]
        mask = torch.zeros_like(values, dtype=torch.bool)
        mask.scatter_(1, positions, True)
    elif mode == 'erase':
        mask = (torch.rand(values.shape, device=values.device, generator=generator) < amount) & values.bool()
    elif mode == 'insert':
        mask = (torch.rand(values.shape, device=values.device, generator=generator) < amount) & ~values.bool()
    else:
        raise ValueError('Unknown binary perturbation mode')
    return torch.where(mask, 1 - values, values), mask.sum(-1)


class RobustnessMeter:
    def __init__(self, classes, device):
        self.classes = classes
        self.samples = 0
        self.correct = 0
        self.clean_correct = 0
        self.changed = 0
        self.changed_correct = 0
        self.loss = 0.
        self.flips = 0
        self.maximum_flips = 0
        self.probability_drift = 0.
        self.confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, clean, corrupt, labels, flips):
        if not torch.isfinite(clean).all() or not torch.isfinite(corrupt).all():
            raise FloatingPointError('Perturbation evaluation produced nonfinite logits')
        reference = clean.argmax(-1)
        prediction = corrupt.argmax(-1)
        correct_clean = reference == labels
        correct_corrupt = prediction == labels
        self.samples += len(labels)
        self.clean_correct += int(correct_clean.sum())
        self.correct += int(correct_corrupt.sum())
        self.changed += int((reference != prediction).sum())
        self.changed_correct += int((correct_clean & ~correct_corrupt).sum())
        self.loss += float(F.cross_entropy(corrupt, labels, reduction='sum'))
        self.flips += int(flips.sum())
        self.maximum_flips = max(self.maximum_flips, int(flips.max()))
        self.probability_drift += float((clean.softmax(-1) - corrupt.softmax(-1)).abs().sum())
        self.confusion.add_(torch.bincount(
            labels * self.classes + prediction,
            minlength=self.classes ** 2,
        ).reshape(self.classes, self.classes))

    def result(self):
        if not self.samples:
            raise ValueError('No examples were evaluated')
        return dict(
            samples=self.samples,
            accuracy=self.correct / self.samples,
            clean_accuracy=self.clean_correct / self.samples,
            accuracy_drop=(self.clean_correct - self.correct) / self.samples,
            prediction_change_rate=self.changed / self.samples,
            clean_correct_failure_rate=self.changed_correct / self.clean_correct if self.clean_correct else None,
            loss=self.loss / self.samples,
            mean_flips=self.flips / self.samples,
            maximum_flips=self.maximum_flips,
            mean_probability_l1=self.probability_drift / self.samples,
            confusion=self.confusion.tolist(),
        )


@torch.no_grad()
def evaluate_perturbations(model, loader, device, fractions, repeats, seed, mode, limit=None):
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError('Supply distinct perturbation fractions')
    if repeats < 1:
        raise ValueError('At least one perturbation repeat is required')
    rows = []
    for fraction in sorted(fractions):
        for repeat in range(repeats):
            generator = torch.Generator(device=device).manual_seed(seed + repeat)
            meter = RobustnessMeter(model.num_classes, device)
            for values, labels in batches(loader, device, limit):
                clean = model(values)
                changed, flips = perturb(values, fraction, generator, mode)
                corrupt = model(changed)
                meter.update(clean, corrupt, labels, flips)
            rows.append(dict(fraction=fraction, repeat=repeat, seed=seed + repeat, **meter.result()))
    summary = []
    for fraction in sorted(fractions):
        records = [row for row in rows if row['fraction'] == fraction]
        accuracies = [row['accuracy'] for row in records]
        average = sum(accuracies) / len(accuracies)
        deviation = math.sqrt(sum((value - average) ** 2 for value in accuracies) / len(accuracies))
        summary.append(dict(
            fraction=fraction,
            accuracy_mean=average,
            accuracy_std=deviation,
            repeat_count=len(records),
            mean_flips=sum(row['mean_flips'] for row in records) / len(records),
        ))
    return dict(mode=mode, repeats=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--fractions', nargs='+', type=float, default=[0., .01, .05, .1])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spatial', nargs='+', choices=('left', 'right', 'up', 'down', 'dilate', 'erode'))
    parser.add_argument('--magnitudes', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--image-shape', nargs=2, type=int, default=[28, 28])
    parser.add_argument('--radius-one', action='store_true')
    parser.add_argument('--mode', choices=('bernoulli', 'fixed', 'erase', 'insert'), default='bernoulli')
    args = parser.parse_args()
    model, loader, device, context = load_analysis_data(args)
    report = evaluate_perturbations(
        model, loader, device, args.fractions, args.repeats, args.seed, args.mode, args.max_batches,
    )
    if args.spatial:
        report['spatial'] = evaluate_spatial(model, loader, device, args.spatial, args.magnitudes, args.image_shape, args.max_batches)
    if args.radius_one:
        report['radius_one'] = exhaustive_radius_one(model, loader, device, limit=args.max_batches)
    write_json(args.output, dict(context=context, **report))


@torch.no_grad()
def spatial_perturb(values, width, height, operation, magnitude):
    if values.shape[1] != width * height or min(width, height) < 1:
        raise ValueError('Spatial perturbations require a matching binary image grid')
    if magnitude < 0 or int(magnitude) != magnitude:
        raise ValueError('Spatial perturbation magnitude must be a nonnegative integer')
    images = values.reshape(-1, 1, height, width)
    distance = int(magnitude)
    if operation in ('left', 'right', 'up', 'down'):
        output = torch.zeros_like(images)
        if distance == 0:
            output.copy_(images)
        elif operation == 'left' and distance < width:
            output[..., :width - distance] = images[..., distance:]
        elif operation == 'right' and distance < width:
            output[..., distance:] = images[..., :width - distance]
        elif operation == 'up' and distance < height:
            output[..., :height - distance, :] = images[..., distance:, :]
        elif operation == 'down' and distance < height:
            output[..., distance:, :] = images[..., :height - distance, :]
    elif operation == 'dilate':
        size = 2 * distance + 1
        output = F.max_pool2d(images, size, stride=1, padding=distance)
    elif operation == 'erode':
        size = 2 * distance + 1
        complement = F.pad(1 - images, (distance,) * 4, value=1)
        output = 1 - F.max_pool2d(complement, size, stride=1)
    else:
        raise ValueError('Unknown spatial binary perturbation')
    result = output.flatten(1)
    return result, (result != values).sum(-1)


@torch.no_grad()
def evaluate_spatial(model, loader, device, operations, magnitudes, shape, limit=None):
    rows = []
    for operation in operations:
        for magnitude in magnitudes:
            meter = RobustnessMeter(model.num_classes, device)
            for values, labels in batches(loader, device, limit):
                baseline = model(values)
                perturbed, flips = spatial_perturb(values, shape[1], shape[0], operation, magnitude)
                meter.update(baseline, model(perturbed), labels, flips)
            rows.append(dict(operation=operation, magnitude=magnitude, **meter.result()))
    return rows


@torch.no_grad()
def exhaustive_radius_one(model, loader, device, chunk_size=16, limit=None):
    if chunk_size < 1:
        raise ValueError('Radius-one evaluation chunk must be positive')
    total = clean_correct = robust_correct = 0
    minimum_flips = []
    for values, labels in batches(loader, device, limit):
        baseline = model(values).argmax(-1)
        correct = baseline == labels
        robust = correct.clone()
        adversarial = torch.full((len(values),), -1, device=device, dtype=torch.long)
        for start in range(0, model.input_dim, chunk_size):
            positions = list(range(start, min(start + chunk_size, model.input_dim)))
            replicas = values[None].expand(len(positions), -1, -1).clone()
            for offset, feature in enumerate(positions):
                replicas[offset, :, feature] = 1 - replicas[offset, :, feature]
            prediction = model(replicas.flatten(0, 1)).reshape(len(positions), len(values), -1).argmax(-1)
            failures = prediction != labels[None]
            affected = failures.any(0)
            first = failures.long().argmax(0) + start
            new = affected & (adversarial < 0)
            adversarial[new] = first[new]
            robust &= ~affected
        total += len(labels)
        clean_correct += int(correct.sum())
        robust_correct += int(robust.sum())
        minimum_flips.extend(adversarial.tolist())
    if not total:
        raise ValueError('Radius-one evaluation received no examples')
    return dict(
        samples=total,
        clean_accuracy=clean_correct / total,
        exact_radius_one_accuracy=robust_correct / total,
        robust_correct=robust_correct,
        first_failing_input=minimum_flips,
        evaluated_single_bit_neighbors=model.input_dim,
    )
