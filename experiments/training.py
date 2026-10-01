"""Epoch optimization with selector trajectories and separate Adam parameter groups."""

from copy import deepcopy
import hashlib
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.wiring import (
    cuda_device, evaluate, experiment_config, load_dataset, set_seed, write_json,
)
from networks.gates import OSLGN, wiring_statistics
from networks.monitoring import WiringMonitor
from networks.monitoring import StepSchedule, configure_optimizer, gradient_statistics


def train_epoch(model, loader, optimizer, schedule, monitor, device, settings):
    model.train()
    loss_sum = correct = samples = 0
    gradient_rows = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Training loss is nonfinite')
        loss.backward()
        observe = (schedule.step_number + 1) % monitor.interval == 0
        if observe:
            gradient_rows.append(dict(
                step=schedule.step_number + 1,
                groups=gradient_statistics(optimizer),
            ))
        clip = float(settings.get('clip_max_norm', 0))
        if clip > 0:
            parameters = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        monitor.observe(model, schedule.step_number)
        loss_sum += float(loss.detach()) * len(labels)
        correct += int((logits.detach().argmax(-1) == labels).sum())
        samples += len(labels)
    if not samples:
        raise ValueError('The training partition is empty')
    monitor.observe(model, schedule.step_number, force=True)
    return dict(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
        gradient_observations=gradient_rows,
    )


def train_main(args):
    config = experiment_config(args)
    settings = config['train']
    set_seed(settings['seed'])
    torch.set_num_threads(settings.get('num_threads', 1))
    device = cuda_device(args.device)
    download = not getattr(args, 'offline', False)
    model = OSLGN(**config['model']).to(device)
    train_set = load_dataset(config, 'train', args.data, download=download)
    has_validation = (
        bool(args.data)
        or config['data']['dataset'] != 'mnist'
        or config['data']['validation_fraction'] > 0
    )
    validation = load_dataset(config, 'val', args.data, download=download) if has_validation else None
    generator = torch.Generator().manual_seed(settings['seed'])
    loader = DataLoader(
        train_set,
        batch_size=settings['batch_size'],
        shuffle=True,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(validation, batch_size=128, pin_memory=True) if validation is not None else None
    optimizer = configure_optimizer(model, settings)
    schedule = StepSchedule(optimizer, settings['epochs'] * len(loader), settings)
    monitor = WiringMonitor(model, settings.get('monitor_interval', 100))
    output = Path(settings['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    history, best, start = [], float('inf'), 0
    fingerprint = data_signature(args.data)
    if getattr(args, 'resume', None):
        start, best, history = restore_snapshot(
            args.resume, model, optimizer, schedule, monitor, config,
            device, generator, fingerprint,
        )
    print(f'Dataset: {config["data"]["dataset"]} | depth: {config["model"]["depth"]} | device: {device}')
    for epoch in range(start, settings['epochs']):
        training = train_epoch(model, loader, optimizer, schedule, monitor, device, settings)
        scores = evaluate(model, val_loader, device) if val_loader is not None else training
        record = dict(
            epoch=epoch + 1,
            train=training,
            wiring=wiring_statistics(model),
            step=schedule.step_number,
            learning_rates={g['name']: g['lr'] for g in optimizer.param_groups},
        )
        if val_loader is not None:
            record['validation'] = scores
        improved = scores['loss'] < best
        best = min(best, scores['loss'])
        history.append(record)
        if improved:
            save_snapshot(
                output / 'best.pth', model, optimizer, schedule, monitor, config,
                epoch + 1, best, history, device, generator, fingerprint,
            )
        save_snapshot(
            output / 'last.pth', model, optimizer, schedule, monitor, config,
            epoch + 1, best, history, device, generator, fingerprint,
        )
        write_json(output / 'wiring-history.json', monitor.summary())
        write_json(output / 'metrics.json', dict(config=config, history=history))
        print(f'Epoch {epoch + 1}: loss={scores["loss"]:.4f}, accuracy={100 * scores["accuracy"]:.2f}%', flush=True)
    if not (output / 'last.pth').is_file():
        save_snapshot(
            output / 'last.pth', model, optimizer, schedule, monitor, config,
            settings['epochs'], best, history, device, generator, fingerprint,
        )
    report = dict(
        config=config,
        task='npz' if args.data else config['data']['dataset'],
        final_train=evaluate(model, DataLoader(train_set, batch_size=128), device),
        history=history,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        wiring=wiring_statistics(model),
        data_sha256=fingerprint,
        complete=True,
    )
    if config['data']['dataset'] == 'mnist' and not args.data:
        test_set = load_dataset(config, 'test', download=download)
        report['final_test'] = evaluate(model, DataLoader(test_set, batch_size=128), device)
    write_json(output / 'metrics.json', report)
    return report



FORMAT = 'gate-training-v2'


def data_signature(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rng_state(device, generator):
    numpy_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy_name=numpy_state[0],
        numpy_keys=numpy_state[1].tolist(),
        numpy_position=numpy_state[2],
        numpy_gaussian=numpy_state[3],
        numpy_cached=numpy_state[4],
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device),
        loader=generator.get_state(),
    )


def restore_rng(state, device, generator):
    random.setstate(state['python'])
    np.random.set_state((
        state['numpy_name'],
        np.asarray(state['numpy_keys'], dtype=np.uint32),
        state['numpy_position'],
        state['numpy_gaussian'],
        state['numpy_cached'],
    ))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    generator.set_state(state['loader'].cpu())


def training_contract(config):
    result = deepcopy(config)
    result['train'].pop('save_dir', None)
    if result['train'].get('schedule', 'constant') == 'constant':
        result['train'].pop('epochs', None)
    result['data'].pop('root', None)
    return result


def save_snapshot(path, model, optimizer, schedule, monitor, config,
                  epoch, best, history, device, generator, data_hash):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = dict(
        format=FORMAT,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        optimizer_parameter_names=optimizer_parameter_names(model, optimizer),
        metrics=history[-1].get('validation', history[-1]['train']) if history else {},
        schedule=schedule.state_dict(),
        monitor=monitor.state_dict(),
        config=config,
        epoch=int(epoch),
        best_loss=float(best),
        history=history,
        data_sha256=data_hash,
        rng=rng_state(device, generator),
    )
    temporary = destination.with_suffix(destination.suffix + '.partial')
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_snapshot(path, model, optimizer, schedule, monitor, config,
                     device, generator, data_hash):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get('format') != FORMAT:
        result = restore_legacy(state, model, optimizer, schedule, config, device)
        monitor.__init__(model, monitor.interval)
        monitor.last_step = schedule.step_number
        return result
    if training_contract(state['config']) != training_contract(config):
        raise ValueError('Resume requires the same data, architecture and optimization settings')
    if state['data_sha256'] != data_hash:
        raise ValueError('The NPZ input changed after the snapshot was written')
    if not 0 <= state['epoch'] <= config['train']['epochs']:
        raise ValueError('The snapshot cannot exceed the requested final epoch')
    if state.get('optimizer_parameter_names') != optimizer_parameter_names(model, optimizer):
        raise ValueError('Snapshot optimizer parameter names differ from this trainable layout')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['schedule'])
    monitor.load_state_dict(state['monitor'])
    restore_rng(state['rng'], device, generator)
    return state['epoch'], state['best_loss'], state['history']


def restore_legacy(state, model, optimizer, schedule, config, device):
    if state['config']['model'] != config['model']:
        raise ValueError('Legacy snapshot architecture differs')
    if state['config']['data'] != config['data']:
        raise ValueError('Legacy snapshot data configuration differs')
    settings = config['train']
    if any(settings.get(key) for key in ('freeze', 'learning_rate_scales', 'warmup_steps')):
        raise ValueError('Legacy resume requires the original ungrouped Adam settings')
    if settings.get('schedule', 'constant') != 'constant':
        raise ValueError('A legacy snapshot does not contain a learning-rate schedule')
    if not 0 <= state['epoch'] <= settings['epochs']:
        raise ValueError('The snapshot cannot exceed the requested final epoch')
    model.load_state_dict(state['model'], strict=True)
    legacy = state['optimizer']
    old_ids = legacy['param_groups'][0]['params']
    if len(old_ids) != len(list(model.parameters())):
        raise ValueError('Legacy optimizer does not cover all model parameters')
    by_parameter = dict(zip((id(value) for value in model.parameters()), old_ids))
    current = optimizer.state_dict()
    for live, saved_group in zip(optimizer.param_groups, current['param_groups']):
        for parameter, new_id in zip(live['params'], saved_group['params']):
            old_id = by_parameter[id(parameter)]
            if old_id in legacy['state']:
                current['state'][new_id] = legacy['state'][old_id]
    optimizer.load_state_dict(current)
    schedule.step_number = state['epoch'] * schedule.total_steps // settings['epochs']
    schedule.apply()
    if 'torch_rng' in state:
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    return state['epoch'], state.get('best_loss', float('inf')), state.get('history', [])


def optimizer_parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        selected = []
        for parameter in group['params']:
            if id(parameter) not in names:
                raise ValueError('An optimized parameter is absent from the logic network')
            selected.append(names[id(parameter)])
        groups.append(selected)
    return groups
