"""Summarize optimization and discrete selection trajectories from saved run files."""

import argparse
import csv
import json
import math
from pathlib import Path

from experiments.wiring import write_json


def read_history(directory):
    root = Path(directory).resolve()
    metrics = json.loads((root / 'metrics.json').read_text())
    history = metrics.get('history', [])
    if not history:
        raise ValueError(f'No completed epoch history in {root}')
    epochs = [int(row['epoch']) for row in history]
    if epochs != sorted(set(epochs)):
        raise ValueError('Epoch history must be ordered and unique')
    for row in history:
        for partition in ('train', 'validation'):
            if partition not in row:
                continue
            for name in ('loss', 'accuracy'):
                value = row[partition][name]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'Invalid {partition} {name} at epoch {row["epoch"]}')
            if not 0 <= row[partition]['accuracy'] <= 1:
                raise ValueError('Accuracy must lie between zero and one')
    wiring_path = root / 'wiring-history.json'
    wiring = json.loads(wiring_path.read_text()) if wiring_path.exists() else None
    return root, metrics, history, wiring


def convergence(history, partition, fraction=.95):
    if not 0 < fraction <= 1:
        raise ValueError('Convergence fraction must lie in (0,1]')
    rows = [row for row in history if partition in row]
    if not rows:
        return None
    maximum = max(row[partition]['accuracy'] for row in rows)
    threshold = maximum * fraction
    first = next(row['epoch'] for row in rows if row[partition]['accuracy'] >= threshold)
    persistent = None
    for index, row in enumerate(rows):
        if all(later[partition]['accuracy'] >= threshold for later in rows[index:]):
            persistent = row['epoch']
            break
    best = min(rows, key=lambda row: (row[partition]['loss'], row['epoch']))
    return dict(
        partition=partition,
        maximum_accuracy=maximum,
        fraction=fraction,
        threshold=threshold,
        first_epoch=first,
        persistent_epoch=persistent,
        minimum_loss_epoch=best['epoch'],
        minimum_loss=best[partition]['loss'],
        final_accuracy=rows[-1][partition]['accuracy'],
    )


def selection_summary(wiring):
    if wiring is None:
        return None
    grouped = {}
    for record in wiring.get('histories', []):
        for item in record['selectors']:
            key = (item['layer'], item['selector'])
            grouped.setdefault(key, []).append(dict(step=record['step'], **item))
    rows = []
    for (layer, selector), observations in sorted(grouped.items()):
        total = sum(row['changed'] for row in observations)
        last_change = max((row['step'] for row in observations if row['changed']), default=None)
        final = observations[-1]
        rows.append(dict(
            layer=layer,
            selector=selector,
            observations=len(observations),
            units=final['units'],
            observed_selection_changes=total,
            last_observed_change_step=last_change,
            final_used_candidates=final['used_candidates'],
            final_margin=final['margin'],
            different_from_initial=final['different_from_initial'],
        ))
    return dict(interval=wiring['interval'], selectors=rows)


def summarize_run(directory, fraction):
    root, metrics, history, wiring = read_history(directory)
    final = history[-1]
    train = convergence(history, 'train', fraction)
    validation = convergence(history, 'validation', fraction)
    gap = None
    if 'validation' in final:
        gap = final['train']['accuracy'] - final['validation']['accuracy']
    return dict(
        directory=str(root),
        seed=metrics['config']['train']['seed'],
        model=metrics['config']['model'],
        epochs=len(history),
        train=train,
        validation=validation,
        final_generalization_gap=gap,
        selection=selection_summary(wiring),
        final_test=metrics.get('final_test'),
    )


def export_epochs(directory, destination):
    _, _, history, _ = read_history(directory)
    fields = ['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'step']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(dict(
                epoch=row['epoch'],
                train_loss=row['train']['loss'],
                train_accuracy=row['train']['accuracy'],
                validation_loss=row.get('validation', {}).get('loss'),
                validation_accuracy=row.get('validation', {}).get('accuracy'),
                step=row.get('step'),
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--fraction', type=float, default=.95)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epoch-csv')
    args = parser.parse_args()
    if args.epoch_csv and len(args.runs) != 1:
        parser.error('Epoch CSV export requires exactly one run')
    reports = [summarize_run(path, args.fraction) for path in args.runs]
    write_json(args.output, dict(runs=reports))
    if args.epoch_csv:
        export_epochs(args.runs[0], args.epoch_csv)
