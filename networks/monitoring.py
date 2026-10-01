"""Track discrete operand changes alongside their continuous selection margins."""

import math

import torch

from networks.gates import GATE_NAMES


def selector_items(model):
    for depth, layer in enumerate(model.layers):
        yield depth, 'left', layer.os1.weight
        yield depth, 'right', layer.os2.weight
        yield depth, 'gate', layer.operator


def summarize_values(values):
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return dict(count=0, mean=None, minimum=None, maximum=None)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Selection diagnostics contain nonfinite values')
    quantiles = torch.quantile(values, values.new_tensor([.1, .5, .9]))
    return dict(
        count=values.numel(),
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        p10=float(quantiles[0]),
        median=float(quantiles[1]),
        p90=float(quantiles[2]),
    )


@torch.no_grad()
def selection_state(model):
    return {
        f'{depth}.{kind}': weights.argmax(-1).detach().clone()
        for depth, kind, weights in selector_items(model)
    }


@torch.no_grad()
def selector_report(weights):
    if weights.ndim != 2 or not torch.isfinite(weights).all():
        raise ValueError('Selector scores must be finite matrices')
    top = weights.float().topk(min(2, weights.shape[1]), dim=-1).values
    margins = top[:, 0] - top[:, 1] if top.shape[1] == 2 else top[:, 0].abs()
    selected = weights.argmax(-1)
    counts = torch.bincount(selected, minlength=weights.shape[1])
    used = counts > 0
    probabilities = counts[used].double() / len(selected)
    return dict(
        units=weights.shape[0],
        candidates=weights.shape[1],
        used_candidates=int(used.sum()),
        largest_fanout=int(counts.max()),
        selection_entropy=float(-(probabilities * probabilities.log2()).sum()),
        ties=int((weights == weights.max(-1, keepdim=True).values).sum(-1).gt(1).sum()),
        margin=summarize_values(margins),
        score_norm=summarize_values(weights.float().norm(dim=-1)),
    )


class WiringMonitor:
    def __init__(self, model, interval=100):
        if interval < 1:
            raise ValueError('Wiring observation interval must be positive')
        self.interval = int(interval)
        self.previous = selection_state(model)
        self.initial = {name: value.clone() for name, value in self.previous.items()}
        self.changes = {name: torch.zeros_like(value) for name, value in self.previous.items()}
        self.observations = 0
        self.last_step = 0
        self.records = []

    @torch.no_grad()
    def observe(self, model, step, force=False):
        if step < self.last_step:
            raise ValueError('Wiring steps must be monotone')
        if step == self.last_step or (step % self.interval and not force):
            return None
        current = selection_state(model)
        if current.keys() != self.previous.keys():
            raise ValueError('The monitored architecture changed during training')
        layers = []
        for depth, kind, weights in selector_items(model):
            name = f'{depth}.{kind}'
            selected = current[name]
            changed = selected != self.previous[name]
            self.changes[name].add_(changed.long())
            report = selector_report(weights)
            report.update(
                layer=depth,
                selector=kind,
                changed=int(changed.sum()),
                changed_fraction=float(changed.float().mean()),
                different_from_initial=int((selected != self.initial[name]).sum()),
                ever_changed=int((self.changes[name] > 0).sum()),
            )
            if kind == 'gate':
                counts = torch.bincount(selected, minlength=16)
                report['gate_counts'] = dict(zip(GATE_NAMES, counts.tolist()))
            layers.append(report)
        record = dict(step=int(step), since_step=self.last_step, selectors=layers)
        self.previous = current
        self.last_step = int(step)
        self.observations += 1
        self.records.append(record)
        return record

    def state_dict(self):
        return dict(
            interval=self.interval,
            previous={key: value.cpu() for key, value in self.previous.items()},
            initial={key: value.cpu() for key, value in self.initial.items()},
            changes={key: value.cpu() for key, value in self.changes.items()},
            observations=self.observations,
            last_step=self.last_step,
            records=self.records,
        )

    def load_state_dict(self, state):
        if state['interval'] != self.interval:
            raise ValueError('Resume must retain the wiring observation interval')
        for field in ('previous', 'initial', 'changes'):
            target = getattr(self, field)
            if state[field].keys() != target.keys():
                raise ValueError('Saved wiring monitor has different selector names')
            for key, current in target.items():
                saved = state[field][key]
                if saved.shape != current.shape:
                    raise ValueError('Saved wiring monitor has different unit counts')
                target[key] = saved.to(current.device)
        self.observations = int(state['observations'])
        self.last_step = int(state['last_step'])
        self.records = list(state['records'])

    def summary(self):
        return dict(
            interval=self.interval,
            observations=self.observations,
            last_step=self.last_step,
            total_changes={key: int(value.sum()) for key, value in self.changes.items()},
            observed_units={key: value.numel() for key, value in self.changes.items()},
            histories=self.records,
        )



def parameter_groups(model, options):
    rate = float(options['learning_rate'])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('The learning rate must be finite and positive')
    scales = options.get('learning_rate_scales', {})
    groups = {'operands': [], 'operators': [], 'head': []}
    for layer in model.layers:
        groups['operands'].extend((layer.os1.weight, layer.os2.weight))
        groups['operators'].append(layer.operator)
    groups['head'].extend(model.head.parameters())
    requested = set(options.get('freeze', []))
    if requested - groups.keys():
        raise ValueError('Frozen parameter groups are operands, operators or head')
    results = []
    identities = set()
    for name, parameters in groups.items():
        scale = float(scales.get(name, 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Learning-rate scales must be finite and positive')
        for parameter in parameters:
            parameter.requires_grad_(name not in requested)
            if id(parameter) in identities:
                raise ValueError('Optimizer groups overlap')
            identities.add(id(parameter))
        active = [parameter for parameter in parameters if parameter.requires_grad]
        if active:
            results.append(dict(params=active, lr=rate * scale, name=name))
    if not results:
        raise ValueError('At least one parameter group must remain trainable')
    if identities != {id(value) for value in model.parameters()}:
        raise ValueError('A model parameter is not covered by optimizer grouping')
    return results


def configure_optimizer(model, options):
    betas = tuple(float(value) for value in options.get('betas', (.9, .999)))
    if len(betas) != 2 or any(not 0 <= value < 1 for value in betas):
        raise ValueError('Adam betas must lie in [0,1)')
    decay = float(options.get('weight_decay', 0.))
    if not math.isfinite(decay) or decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    return torch.optim.Adam(
        parameter_groups(model, options),
        lr=options['learning_rate'],
        betas=betas,
        weight_decay=decay,
    )


class StepSchedule:
    def __init__(self, optimizer, total_steps, options):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.kind = options.get('schedule', 'constant')
        self.warmup = int(options.get('warmup_steps', 0))
        self.minimum = float(options.get('minimum_lr_ratio', 0.))
        self.step_number = 0
        self.base_rates = [group['lr'] for group in optimizer.param_groups]
        if self.kind not in ('constant', 'cosine'):
            raise ValueError('Schedule must be constant or cosine')
        if self.total_steps < 1 or not 0 <= self.warmup < self.total_steps:
            raise ValueError('Warmup must leave at least one ordinary optimization step')
        if not 0 <= self.minimum <= 1:
            raise ValueError('Minimum learning-rate ratio must lie in [0,1]')
        self.apply()

    def factor(self):
        if self.step_number < self.warmup:
            return (self.step_number + 1) / self.warmup
        if self.kind == 'constant':
            return 1.
        progress = (self.step_number - self.warmup) / max(1, self.total_steps - self.warmup - 1)
        progress = min(1., max(0., progress))
        return self.minimum + (1 - self.minimum) * .5 * (1 + math.cos(math.pi * progress))

    def apply(self):
        factor = self.factor()
        for group, base in zip(self.optimizer.param_groups, self.base_rates):
            group['lr'] = base * factor

    def step(self):
        self.step_number += 1
        self.apply()

    def state_dict(self):
        return dict(
            total_steps=self.total_steps,
            kind=self.kind,
            warmup=self.warmup,
            minimum=self.minimum,
            base_rates=self.base_rates,
            step_number=self.step_number,
        )

    def load_state_dict(self, state):
        for key in ('kind', 'warmup', 'minimum', 'base_rates'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Resume schedule differs in {key}')
        if state['total_steps'] != self.total_steps:
            if self.kind != 'constant' or self.total_steps < state['total_steps']:
                raise ValueError('Only a constant schedule can extend its original optimization horizon')
        self.step_number = int(state['step_number'])
        if not 0 <= self.step_number <= self.total_steps:
            raise ValueError('Saved optimizer step is outside the schedule')
        self.apply()


@torch.no_grad()
def gradient_statistics(optimizer):
    rows = []
    for group in optimizer.param_groups:
        squared = maximum = 0.
        elements = nonzero = missing = 0
        for parameter in group['params']:
            if parameter.grad is None:
                missing += parameter.numel()
                continue
            gradient = parameter.grad.detach().float()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Nonfinite gradient in {group["name"]}')
            squared += float(gradient.square().sum())
            maximum = max(maximum, float(gradient.abs().max()))
            elements += gradient.numel()
            nonzero += int(torch.count_nonzero(gradient))
        rows.append(dict(
            group=group['name'],
            learning_rate=group['lr'],
            norm=math.sqrt(squared),
            maximum=maximum,
            elements=elements,
            nonzero=nonzero,
            missing=missing,
        ))
    return rows
