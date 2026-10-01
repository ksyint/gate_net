"""Prepare normalized NPZ partitions and inspect local MNIST IDX archives."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from experiments.wiring import write_json


SPLITS = ('train', 'val', 'test')
IDX_FILES = {
    'train_images': 'train-images-idx3-ubyte',
    'train_labels': 'train-labels-idx1-ubyte',
    'test_images': 't10k-images-idx3-ubyte',
    'test_labels': 't10k-labels-idx1-ubyte',
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_images(values, scale):
    values = np.asarray(values)
    if values.ndim < 2 or not len(values) or values.dtype.kind not in 'uif':
        raise ValueError('Images must be a nonempty numeric array with batch as the first axis')
    if scale not in ('unit', 'uint8'):
        raise ValueError('Choose unit values or uint8 pixel scaling')
    upper = 1 if scale == 'unit' else 255
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > upper:
        raise ValueError(f'Image values must be finite and within [0,{upper}]')
    result = values.astype(np.float32)
    if scale == 'uint8':
        result /= 255.0
    return result.reshape(len(result), -1)


def validate_labels(labels, count, classes):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != count or labels.dtype.kind not in 'iu':
        raise ValueError('Labels must be a one-dimensional integer array matching the image count')
    if classes < 2 or labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Class IDs must lie within the configured class count')
    return labels.astype(np.int64)


def binary_identities(images, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('The pixel threshold must be in [0,1]')
    packed = np.packbits(images > threshold, axis=1)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in packed]


def split_statistics(images, labels, classes, threshold):
    identities = binary_identities(images, threshold)
    frequencies = Counter(identities)
    labels_by_image = defaultdict(set)
    for identity, label in zip(identities, labels):
        labels_by_image[identity].add(int(label))
    return dict(samples=len(labels), input_dim=images.shape[1], minimum=float(images.min()),
                maximum=float(images.max()), classes={str(index): int((labels == index).sum()) for index in range(classes)},
                ones_fraction=float((images > threshold).mean()), unique_binary_inputs=len(frequencies),
                duplicate_examples=sum(value-1 for value in frequencies.values()),
                conflicting_binary_inputs=sum(len(values) > 1 for values in labels_by_image.values()))


def inspect_archive(path, input_dim=784, classes=10, threshold=.5):
    path = Path(path).resolve()
    before = digest_file(path)
    report, identities = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        expected = {f'{prefix}_{split}' for split in SPLITS for prefix in ('x', 'y')}
        if not {'x_train', 'y_train', 'x_val', 'y_val'} <= set(archive.files):
            raise ValueError('Training archives require x_train, y_train, x_val, and y_val')
        if set(archive.files)-expected:
            raise ValueError(f'Unknown NPZ fields: {sorted(set(archive.files)-expected)}')
        for split in SPLITS:
            keys = {f'x_{split}', f'y_{split}'}
            present = keys & set(archive.files)
            if not present:
                continue
            if present != keys:
                raise ValueError(f'Both image and label arrays are required for {split}')
            images = normalize_images(archive[f'x_{split}'], 'unit')
            labels = validate_labels(archive[f'y_{split}'], len(images), classes)
            if images.shape[1] != input_dim:
                raise ValueError(f'{split}: expected {input_dim} flattened input values')
            report[split] = split_statistics(images, labels, classes, threshold)
            identities[split] = set(binary_identities(images, threshold))
    overlap = []
    for index, first in enumerate(identities):
        for second in list(identities)[index+1:]:
            shared = identities[first] & identities[second]
            overlap.append(dict(first=first, second=second, shared_binary_inputs=len(shared)))
    if digest_file(path) != before:
        raise RuntimeError('The archive changed during inspection')
    return dict(path=str(path), sha256=before, threshold=threshold, classes=classes,
                input_dim=input_dim, splits=report, overlap=overlap)


def grouped_partition(images, labels, fraction, seed, threshold):
    if not 0 < fraction < 1 or not 0 <= seed < 2 ** 32:
        raise ValueError('Validation fraction must be in (0,1) and seed in [0,2**32)')
    groups = defaultdict(list)
    for index, identity in enumerate(binary_identities(images, threshold)):
        groups[identity].append(index)
    by_class = defaultdict(list)
    for identity, indices in groups.items():
        counts = Counter(int(labels[index]) for index in indices)
        majority = min(counts, key=lambda label: (-counts[label], label))
        by_class[majority].append((identity, indices))
    train, validation = [], []
    for label, entries in sorted(by_class.items()):
        if len(entries) < 2:
            raise ValueError(f'Class {label} needs at least two distinct binary input groups')
        entries.sort(key=lambda item: hashlib.sha256(f'{seed}:{item[0]}'.encode()).digest())
        target = max(1, round(sum(len(indices) for _, indices in entries) * fraction))
        selected = 0
        for position, (_, indices) in enumerate(entries):
            if position < len(entries)-1 and (selected < target or not selected):
                validation.extend(indices)
                selected += len(indices)
            else:
                train.extend(indices)
    return np.array(sorted(train)), np.array(sorted(validation))


def write_partition(destination, images, labels, fraction, seed, classes, threshold,
                    test_images=None, test_labels=None, sources=None):
    destination = Path(destination).resolve()
    if destination.exists() or Path(str(destination)+'.json').exists():
        raise FileExistsError('Choose a new output archive and metadata path')
    training, validation = grouped_partition(images, labels, fraction, seed, threshold)
    arrays = dict(x_train=images[training], y_train=labels[training],
                  x_val=images[validation], y_val=labels[validation])
    if test_images is not None:
        if test_images.shape[1] != images.shape[1]:
            raise ValueError('Training and test images have different flattened dimensions')
        arrays.update(x_test=test_images, y_test=test_labels)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = inspect_archive(destination, images.shape[1], classes, threshold)
    report.update(seed=seed, requested_validation_fraction=fraction,
                  grouping='identical inputs after the configured pixel threshold', sources=sources or [])
    write_json(str(destination)+'.json', report)
    return report


def read_idx(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError(f'Truncated IDX header: {path}')
        zero, kind, dimensions = struct.unpack('>HBB', header)
        if zero != 0 or kind != 8 or dimensions not in (1, 3):
            raise ValueError(f'Expected an unsigned-byte MNIST IDX file: {path}')
        raw_shape = stream.read(4*dimensions)
        if len(raw_shape) != 4*dimensions:
            raise ValueError(f'Truncated IDX dimensions: {path}')
        shape = struct.unpack('>'+'I'*dimensions, raw_shape)
        if any(value < 1 for value in shape):
            raise ValueError('IDX dimensions must be positive')
        payload = stream.read()
    if len(payload) != int(np.prod(shape)):
        raise ValueError(f'IDX payload size differs from its declared shape: {path}')
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


def mnist_idx(root):
    root = Path(root).resolve()
    if (root/'MNIST'/'raw').is_dir():
        root = root/'MNIST'/'raw'
    arrays, sources = {}, []
    for key, name in IDX_FILES.items():
        path = root/name
        if not path.is_file():
            path = root/(name+'.gz')
        if not path.is_file():
            raise FileNotFoundError(f'Place {name} or {name}.gz under {root}')
        arrays[key] = read_idx(path)
        sources.append(dict(path=str(path), sha256=digest_file(path)))
    for split in ('train', 'test'):
        if arrays[f'{split}_images'].shape[1:] != (28, 28):
            raise ValueError('MNIST IDX images must have shape [N,28,28]')
        arrays[f'{split}_images'] = normalize_images(arrays[f'{split}_images'], 'uint8')
        arrays[f'{split}_labels'] = validate_labels(arrays[f'{split}_labels'], len(arrays[f'{split}_images']), 10)
    return arrays, sources


def partition_arguments(parser):
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('inspect')
    command.add_argument('--data', required=True)
    command.add_argument('--input-dim', type=int, default=784)
    command.add_argument('--classes', type=int, default=10)
    command.add_argument('--threshold', type=float, default=.5)
    command.add_argument('--output', required=True)
    command = commands.add_parser('pack')
    command.add_argument('--images', required=True)
    command.add_argument('--labels', required=True)
    command.add_argument('--scale', choices=['unit', 'uint8'], required=True)
    command.add_argument('--classes', type=int, default=10)
    partition_arguments(command)
    command = commands.add_parser('idx')
    command.add_argument('--root', required=True)
    partition_arguments(command)
    args = parser.parse_args()
    if args.operation == 'inspect':
        if Path(args.output).resolve() == Path(args.data).resolve():
            parser.error('The report cannot overwrite the input archive')
        result = inspect_archive(args.data, args.input_dim, args.classes, args.threshold)
        write_json(args.output, result)
    elif args.operation == 'pack':
        images = normalize_images(np.load(args.images, allow_pickle=False), args.scale)
        labels = validate_labels(np.load(args.labels, allow_pickle=False), len(images), args.classes)
        sources = [dict(path=str(Path(path).resolve()), sha256=digest_file(path)) for path in (args.images, args.labels)]
        result = write_partition(args.output, images, labels, args.validation_fraction,
                                 args.seed, args.classes, args.threshold, sources=sources)
    else:
        arrays, sources = mnist_idx(args.root)
        result = write_partition(args.output, arrays['train_images'], arrays['train_labels'],
                                 args.validation_fraction, args.seed, 10, args.threshold,
                                 arrays['test_images'], arrays['test_labels'], sources)
    print(json.dumps(result, indent=2))
