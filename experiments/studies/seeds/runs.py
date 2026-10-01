"""Plan wiring sweeps, resume unfinished runs, and aggregate measured seed results."""

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

import yaml

from data.preparation.arrays.partitions import digest_file
from experiments.runtime.wiring import ROOT, initialization_name, read_profiles, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def configuration_identity(config):
    value = deepcopy(config)
    value['train'].pop('save_dir', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def chosen(config, depths=None, widths=None, initializations=None, seeds=None):
    values = ((depths, config['model']['depth']), (widths, config['model']['hidden_dim']),
              (initializations, initialization_name(config['model'])), (seeds, config['train']['seed']))
    return all(options is None or value in options for options, value in values)


def plan_runs(destination, depths=None, widths=None, initializations=None, seeds=None,
              epochs=None, data_root=None, data=None, offline=False, device='cuda', limit=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new study plan directory')
    if device != 'cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('Wiring study execution requires a CUDA device')
    if epochs is not None and epochs < 1:
        raise ValueError('Epochs must be positive')
    if limit is not None and limit < 1:
        raise ValueError('The study limit must be positive')
    source_data = str(Path(data).resolve()) if data else None
    data_hash = digest_file(source_data) if source_data else None
    profiles = [(path, value, count) for path, value, count in read_profiles()
                if chosen(value, depths, widths, initializations, seeds)]
    profiles = profiles[:limit] if limit else profiles
    if not profiles:
        raise ValueError('The study filters selected no wiring profiles')
    configurations = destination/'configurations'
    configurations.mkdir(parents=True)
    jobs = []
    for path, original, parameters in profiles:
        config = deepcopy(original)
        if epochs is not None:
            config['train']['epochs'] = epochs
        config['data']['root'] = str(Path(data_root or config['data']['root']).resolve())
        identifier = configuration_identity(config)
        output = destination/'runs'/identifier[:16]
        config['train']['save_dir'] = str(output)
        target = configurations/(identifier+'.yaml')
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        job = dict(id=identifier, configuration=str(target), configuration_sha256=digest_file(target),
                   source_profile=str(path.relative_to(ROOT)), output=str(output), parameters=parameters,
                   seed=config['train']['seed'], epochs=config['train']['epochs'], device=device,
                   data=source_data, data_sha256=data_hash, offline=offline)
        jobs.append(job)
    if len({row['id'] for row in jobs}) != len(jobs):
        raise ValueError('Study jobs must have distinct configurations')
    manifest = destination/'jobs.jsonl'
    manifest.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in jobs))
    write_json(destination/'plan.json', dict(format='gate-study-v1', created=timestamp(), jobs=len(jobs),
                                            jobs_sha256=digest_file(manifest)))
    return jobs


class RunQueue:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        path = self.directory/'jobs.jsonl'
        metadata = json.loads((self.directory/'plan.json').read_text())
        if metadata.get('format') != 'gate-study-v1' or metadata['jobs_sha256'] != digest_file(path):
            raise ValueError('Study job manifest differs from its saved plan')
        self.jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(self.jobs) != metadata['jobs'] or len({row['id'] for row in self.jobs}) != len(self.jobs):
            raise ValueError('The planned job membership is invalid')
        self.journal = self.directory/'events.jsonl'

    def statuses(self):
        state = {row['id']: dict(status='pending') for row in self.jobs}
        if self.journal.is_file():
            for line in self.journal.read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event['id'] not in state or event['status'] not in ('running', 'complete', 'failed'):
                    raise ValueError('Unexpected study journal entry')
                state[event['id']] = event
        return state

    def event(self, job, status, **details):
        with self.journal.open('a') as stream:
            stream.write(json.dumps(dict(id=job['id'], status=status, time=timestamp(), **details))+'\n')

    def command(self, job, resume=False):
        if digest_file(job['configuration']) != job['configuration_sha256']:
            raise ValueError('Planned configuration has changed')
        if job['data'] and digest_file(job['data']) != job['data_sha256']:
            raise ValueError('Planned NPZ input has changed')
        command = [sys.executable, str(ROOT/'gate.py'), 'train', '--config', job['configuration'],
                   '--dataset', 'mnist', '--output', job['output'], '--device', job['device']]
        if job['data']:
            command.extend(('--data', job['data']))
        if job['offline']:
            command.append('--offline')
        checkpoint = Path(job['output'])/'last.pth'
        if resume and checkpoint.is_file():
            command.extend(('--resume', str(checkpoint)))
        return command

    def completed(self, job):
        path = Path(job['output'])/'metrics.json'
        if not path.is_file() or not (path.parent/'last.pth').is_file():
            return False
        report = json.loads(path.read_text())
        config = yaml.safe_load(Path(job['configuration']).read_text())
        if report.get('config') != config or not report.get('history'):
            return False
        return report['history'][-1]['epoch'] == job['epochs']

    def run(self, resume=False, dry_run=False, keep_going=False):
        if dry_run:
            return [dict(id=job['id'], command=shlex.join(self.command(job, resume))) for job in self.jobs]
        lock = self.directory/'.runner.lock'
        with lock.open('x') as stream:
            stream.write(timestamp()+'\n')
        try:
            for job in self.jobs:
                if self.completed(job):
                    self.event(job, 'complete', recovered=True)
                    continue
                output = Path(job['output'])
                if output.exists() and any(output.iterdir()) and not resume:
                    raise FileExistsError(f'Use --resume for an existing run directory: {output}')
                if output.exists() and any(output.iterdir()) and not (output/'last.pth').is_file():
                    raise ValueError(f'Existing run has no last.pth to resume: {output}')
                command = self.command(job, resume)
                output.mkdir(parents=True, exist_ok=True)
                write_json(output/'planned-run.json', job)
                self.event(job, 'running', command=command)
                result = subprocess.run(command, cwd=ROOT, check=False)
                success = result.returncode == 0 and self.completed(job)
                self.event(job, 'complete' if success else 'failed', returncode=result.returncode)
                if not success and not keep_going:
                    raise RuntimeError(f'Study run failed: {job["id"]}')
        finally:
            lock.unlink(missing_ok=True)
        return self.statuses()


def distribution(values):
    return dict(count=len(values), mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def summarize_runs(roots, seeds=None):
    paths = sorted({path.resolve() for root in roots for path in Path(root).rglob('metrics.json')})
    groups = defaultdict(list)
    for path in paths:
        report = json.loads(path.read_text())
        if not {'config', 'history', 'final_train'} <= report.keys():
            continue
        config = deepcopy(report['config'])
        seed = config['train'].pop('seed')
        config['train'].pop('save_dir', None)
        planned = path.parent/'planned-run.json'
        provenance = json.loads(planned.read_text()) if planned.is_file() else {}
        data_identity = provenance.get('data_sha256')
        if report.get('task') == 'npz' and not data_identity:
            raise ValueError(f'NPZ seed aggregation needs planned-run.json data identity: {path}')
        if not report['history'] or report['history'][-1]['epoch'] != config['train']['epochs']:
            raise ValueError(f'The run has not reached its configured final epoch: {path}')
        identity = json.dumps(dict(config=config, task=report.get('task'), data_sha256=data_identity), sort_keys=True)
        history = report['history']
        row = dict(path=str(path), seed=seed, final_train=report['final_train'])
        if 'final_test' in report:
            row['final_test'] = report['final_test']
        if all('validation' in epoch for epoch in history):
            row['final_validation'] = history[-1]['validation']
            row['peak_validation_accuracy'] = max(epoch['validation']['accuracy'] for epoch in history)
            best = min(history, key=lambda epoch: epoch['validation']['loss'])
            row['best_validation_epoch'] = best['epoch']
        groups[identity].append(row)
    if not groups:
        raise ValueError('No complete training metric files were found')
    output = []
    for key, rows in sorted(groups.items()):
        observed = [row['seed'] for row in rows]
        if len(observed) != len(set(observed)):
            raise ValueError('A configuration group contains duplicate seeds')
        if seeds is not None and set(observed) != set(seeds):
            raise ValueError(f'Seed membership differs: expected {sorted(seeds)}, observed {sorted(observed)}')
        common = set.intersection(*(set(row) for row in rows)) - {'path', 'seed'}
        summaries = {}
        for name in sorted(common):
            values = [row[name] for row in rows]
            if isinstance(values[0], dict):
                summaries[name] = {metric: distribution([value[metric] for value in values])
                                   for metric in ('accuracy', 'loss', 'samples')}
            else:
                summaries[name] = distribution(values)
        output.append(dict(setting=json.loads(key), seeds=sorted(observed), metrics=summaries,
                           runs=sorted(rows, key=lambda row: row['seed'])))
    return dict(groups=output, group_count=len(output), runs=sum(len(rows) for rows in groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('plan')
    command.add_argument('--output', required=True)
    command.add_argument('--depths', type=int, nargs='+')
    command.add_argument('--widths', type=int, nargs='+')
    command.add_argument('--initializations', nargs='+')
    command.add_argument('--seeds', type=int, nargs='+')
    command.add_argument('--epochs', type=int)
    command.add_argument('--data-root')
    command.add_argument('--data')
    command.add_argument('--offline', action='store_true')
    command.add_argument('--device', default='cuda')
    command.add_argument('--limit', type=int)
    for name in ('status', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--plan', required=True)
        if name == 'run':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--dry-run', action='store_true')
            command.add_argument('--keep-going', action='store_true')
    command = commands.add_parser('aggregate')
    command.add_argument('--roots', nargs='+', required=True)
    command.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    command.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.operation == 'plan':
        result = plan_runs(args.output, args.depths, args.widths, args.initializations, args.seeds,
                           args.epochs, args.data_root, args.data, args.offline, args.device, args.limit)
    elif args.operation == 'aggregate':
        if Path(args.output).name == 'metrics.json':
            parser.error('Use a distinct filename for aggregated metrics')
        result = summarize_runs(args.roots, args.seeds)
        write_json(args.output, result)
    else:
        queue = RunQueue(args.plan)
        result = queue.statuses() if args.operation == 'status' else queue.run(args.resume, args.dry_run, args.keep_going)
    print(json.dumps(result, indent=2, allow_nan=False))
