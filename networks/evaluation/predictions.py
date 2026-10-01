"""Collect CUDA predictions and summarize classification errors across saved outputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.partitions import digest_file
from experiments.wiring import cuda_device, load_dataset, restore_model, write_json


def read_predictions(path):
    path = Path(path).resolve()
    rows, identifiers = [], set()
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {'id', 'target', 'prediction', 'logits', 'input_sha256'}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f'{path}:{number}: missing prediction fields')
            identifier = str(row['id'])
            if identifier in identifiers:
                raise ValueError(f'Duplicate prediction ID: {identifier}')
            identifiers.add(identifier)
            logits = row['logits']
            if not isinstance(logits, list) or len(logits) < 2:
                raise ValueError('Prediction logits require at least two classes')
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in logits):
                raise ValueError('Prediction logits must be finite numbers')
            for field in ('target', 'prediction'):
                if type(row[field]) is not int or not 0 <= row[field] < len(logits):
                    raise ValueError(f'Invalid {field} class ID')
            expected = max(range(len(logits)), key=logits.__getitem__)
            if row['prediction'] != expected:
                raise ValueError('The prediction ID differs from the saved logit argmax')
            if not isinstance(row['input_sha256'], str) or len(row['input_sha256']) != 64:
                raise ValueError('Each prediction needs a binary input digest')
            rows.append(row)
    if not rows or len({len(row['logits']) for row in rows}) != 1:
        raise ValueError('Prediction records must be nonempty and share a class count')
    validate_metadata(path, rows)
    return rows


def validate_metadata(path, rows):
    metadata = Path(str(path)+'.json')
    if not metadata.is_file():
        return None
    saved = json.loads(metadata.read_text())
    if saved.get('predictions_sha256') != digest_file(path):
        raise ValueError('Prediction content differs from its saved metadata digest')
    if saved.get('samples') != len(rows):
        raise ValueError('Prediction count differs from its saved metadata')
    if any(row.get('split') != saved.get('split') for row in rows):
        raise ValueError('Prediction partition differs from its saved metadata')
    classes = saved.get('config', {}).get('model', {}).get('num_classes')
    if classes != len(rows[0]['logits']):
        raise ValueError('Prediction class count differs from the saved configuration')
    measured = sum(row['target'] == row['prediction'] for row in rows)/len(rows)
    if not math.isclose(measured, saved.get('accuracy', -1), abs_tol=1e-12):
        raise ValueError('Prediction accuracy differs from its saved metadata')
    return saved


@torch.no_grad()
def collect_predictions(checkpoint, destination, device='cuda', data=None,
                        data_root=None, split=None, offline=False, batch_size=128):
    if batch_size < 1:
        raise ValueError('Prediction batch size must be positive')
    destination = Path(destination).resolve()
    metadata = Path(str(destination)+'.json')
    if destination.exists() or metadata.exists():
        raise FileExistsError('Choose a new prediction file and sidecar metadata path')
    model, config = restore_model(checkpoint, device)
    if data_root:
        config['data']['root'] = data_root
    split = split or ('val' if data else 'test' if config['data']['dataset'] == 'mnist' else 'val')
    if data and split == 'test':
        raise ValueError('The training NPZ interface uses val for held-out prediction')
    dataset = load_dataset(config, split, data, download=not offline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    count = correct = 0
    try:
        with temporary.open('x') as stream:
            for images, labels in loader:
                values = images.to(device, non_blocking=True)
                logits = model(values)
                predictions = logits.argmax(-1)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Prediction logits contain nonfinite values')
                correct += int((predictions == labels.to(device)).sum())
                for binary, target, prediction, scores in zip(images, labels.tolist(), predictions.tolist(), logits.tolist()):
                    identity = hashlib.sha256(binary.byte().numpy().tobytes()).hexdigest()
                    row = dict(id=count, target=target, prediction=prediction,
                               logits=scores, input_sha256=identity, split=split)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    count += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    report = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest_file(checkpoint),
                  predictions=str(destination), predictions_sha256=digest_file(destination),
                  dataset='npz' if data else config['data']['dataset'], split=split,
                  samples=count, accuracy=correct/count, config=config,
                  data_sha256=digest_file(data) if data else None)
    write_json(metadata, report)
    return report


@torch.no_grad()
def score_predictions(path, device='cuda', bins=15, top_errors=30):
    if bins < 1 or top_errors < 0:
        raise ValueError('Calibration bins must be positive and error count nonnegative')
    rows = read_predictions(path)
    device = cuda_device(device)
    logits = torch.tensor([row['logits'] for row in rows], dtype=torch.float64, device=device)
    labels = torch.tensor([row['target'] for row in rows], dtype=torch.long, device=device)
    predictions = logits.argmax(-1)
    classes = logits.shape[1]
    confusion = torch.bincount(labels*classes+predictions, minlength=classes**2).reshape(classes, classes)
    support, predicted = confusion.sum(1), confusion.sum(0)
    correct = confusion.diag()
    precision = correct / predicted.clamp_min(1)
    recall = correct / support.clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(torch.finfo(torch.float64).eps)
    probabilities = logits.softmax(-1)
    confidence = probabilities.max(-1).values
    success = predictions == labels
    assigned = (confidence*bins).long().clamp_max(bins-1)
    calibration, ece = [], 0.
    for index in range(bins):
        selected = assigned == index
        count = int(selected.sum())
        accuracy = float(success[selected].double().mean()) if count else None
        certainty = float(confidence[selected].mean()) if count else None
        if count:
            ece += count/len(rows)*abs(accuracy-certainty)
        calibration.append(dict(bin=index, lower=index/bins, upper=(index+1)/bins,
                                samples=count, accuracy=accuracy, confidence=certainty))
    target = torch.nn.functional.one_hot(labels, classes)
    brier = float((probabilities-target).square().sum(-1).mean())
    nll = float(torch.nn.functional.cross_entropy(logits, labels))
    present = support > 0
    per_class = [dict(class_id=index, support=int(support[index]), predicted=int(predicted[index]),
                      precision=float(precision[index]), recall=float(recall[index]), f1=float(f1[index]))
                 for index in range(classes)]
    errors = torch.where(~success)[0]
    errors = errors[torch.argsort(confidence[errors], descending=True, stable=True)[:top_errors]].tolist()
    top = [dict(id=rows[index]['id'], target=rows[index]['target'], prediction=rows[index]['prediction'],
                input_sha256=rows[index]['input_sha256'], confidence=float(confidence[index])) for index in errors]
    return dict(path=str(Path(path).resolve()), sha256=digest_file(path), samples=len(rows), classes=classes,
                accuracy=float(success.double().mean()), macro_f1=float(f1[present].mean()),
                balanced_accuracy=float(recall[present].mean()), weighted_f1=float((f1*support).sum()/len(rows)),
                macro_classes='classes present among reference labels', negative_log_likelihood=nll,
                brier_score=brier, expected_calibration_error=ece,
                confusion=confusion.tolist(), confusion_axes=['reference', 'prediction'],
                per_class=per_class, calibration=calibration, confident_errors=top)


def exact_discordance_probability(first_only, second_only):
    trials = first_only+second_only
    if not trials:
        return 1.0
    tail = min(first_only, second_only)
    logarithms = [math.lgamma(trials+1)-math.lgamma(index+1)-math.lgamma(trials-index+1)
                  -trials*math.log(2) for index in range(tail+1)]
    maximum = max(logarithms)
    return min(1.0, 2*math.exp(maximum)*sum(math.exp(value-maximum) for value in logarithms))


def compare_predictions(first, second):
    left = {str(row['id']): row for row in read_predictions(first)}
    right = {str(row['id']): row for row in read_predictions(second)}
    if left.keys() != right.keys():
        raise ValueError('Paired prediction comparison requires identical example IDs')
    both = neither = first_only = second_only = changed = 0
    for key, one in left.items():
        two = right[key]
        if one['target'] != two['target'] or one['input_sha256'] != two['input_sha256']:
            raise ValueError(f'Paired input or reference differs for example {key}')
        if len(one['logits']) != len(two['logits']):
            raise ValueError('Paired predictions have different class counts')
        a, b = one['target'] == one['prediction'], two['target'] == two['prediction']
        both += a and b
        neither += not a and not b
        first_only += a and not b
        second_only += b and not a
        changed += one['prediction'] != two['prediction']
    count = len(left)
    return dict(first=str(first), second=str(second), samples=count, both_correct=both,
                both_wrong=neither, first_only_correct=first_only, second_only_correct=second_only,
                changed_predictions=changed, first_accuracy=(both+first_only)/count,
                second_accuracy=(both+second_only)/count,
                paired_accuracy_change=(second_only-first_only)/count,
                mcnemar_exact_two_sided=exact_discordance_probability(first_only, second_only))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('collect')
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--data')
    command.add_argument('--data-root')
    command.add_argument('--split', choices=['train', 'val', 'test'])
    command.add_argument('--offline', action='store_true')
    command.add_argument('--batch-size', type=int, default=128)
    command.add_argument('--device', default='cuda')
    command.add_argument('--output', required=True)
    command = commands.add_parser('score')
    command.add_argument('--predictions', required=True)
    command.add_argument('--device', default='cuda')
    command.add_argument('--bins', type=int, default=15)
    command.add_argument('--top-errors', type=int, default=30)
    command.add_argument('--output', required=True)
    command = commands.add_parser('compare')
    command.add_argument('--first', required=True)
    command.add_argument('--second', required=True)
    command.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.operation == 'collect':
        result = collect_predictions(args.checkpoint, args.output, args.device, args.data,
                                     args.data_root, args.split, args.offline, args.batch_size)
    else:
        sources = [args.predictions] if args.operation == 'score' else [args.first, args.second]
        protected = {Path(path).resolve() for path in sources}
        protected.update(Path(str(path)+'.json').resolve() for path in sources)
        if Path(args.output).resolve() in protected:
            parser.error('Reports must not overwrite their source predictions')
        result = score_predictions(args.predictions, args.device, args.bins, args.top_errors) if args.operation == 'score' else compare_predictions(args.first, args.second)
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
