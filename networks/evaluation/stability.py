"""Compare selected wiring and score margins across compatible training snapshots."""

import argparse
from pathlib import Path

import torch

from experiments.wiring import cuda_device, restore_model, write_json
from networks.monitoring import selector_items, summarize_values


@torch.no_grad()
def compare_models(first, second):
    first_items = list(selector_items(first))
    second_items = list(selector_items(second))
    if len(first_items) != len(second_items):
        raise ValueError('Compared models have different numbers of selectors')
    rows = []
    total = changed = 0
    for (depth, kind, left), (other_depth, other_kind, right) in zip(first_items, second_items):
        if (depth, kind, left.shape) != (other_depth, other_kind, right.shape):
            raise ValueError('Compared selector layouts differ')
        chosen_left = left.argmax(-1)
        chosen_right = right.argmax(-1)
        different = chosen_left != chosen_right
        delta = right.float() - left.float()
        top = right.topk(min(2, right.shape[1]), -1).values
        margins = top[:, 0] - top[:, 1] if top.shape[1] == 2 else top[:, 0].abs()
        rows.append(dict(
            layer=depth,
            selector=kind,
            units=len(chosen_left),
            changed=int(different.sum()),
            changed_fraction=float(different.float().mean()),
            changed_units=torch.where(different)[0].tolist(),
            changed_from=chosen_left[different].tolist(),
            changed_to=chosen_right[different].tolist(),
            score_delta_norm=float(delta.norm()),
            row_score_delta=summarize_values(delta.norm(dim=-1)),
            new_margin=summarize_values(margins),
        ))
        changed += int(different.sum())
        total += len(chosen_left)
    head = {}
    for name in ('weight', 'bias'):
        one = getattr(first.head, name)
        two = getattr(second.head, name)
        if one.shape != two.shape:
            raise ValueError('Compared classifiers have different dimensions')
        head[name] = summarize_values(two - one)
    return dict(
        selectors=rows,
        total_units=total,
        total_changed=changed,
        changed_fraction=changed / total,
        head_changes=head,
        truth_table_changes=truth_table_changes(first, second),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference', choices=('first', 'previous'), default='previous')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        parser.error('At least two checkpoints are needed')
    paths = [Path(value).resolve() for value in args.checkpoints]
    if len(set(paths)) != len(paths):
        parser.error('Checkpoint paths must be distinct')
    device = cuda_device(args.device)
    first, config = restore_model(paths[0], device)
    previous_path = paths[0]
    records = []
    for path in paths[1:]:
        other, other_config = restore_model(path, device)
        if config['model'] != other_config['model']:
            raise ValueError('Checkpoint model configurations differ')
        records.append(dict(
            reference=str(previous_path),
            checkpoint=str(path),
            **compare_models(first, other),
        ))
        if args.reference == 'previous':
            first = other
            previous_path = path
    write_json(args.output, dict(reference_mode=args.reference, comparisons=records))


@torch.no_grad()
def truth_table_changes(first, second):
    shifts = torch.arange(3, -1, -1, device=first.head.weight.device)
    records = []
    for depth, (left, right) in enumerate(zip(first.layers, second.layers)):
        left_gate, right_gate = left.operator.argmax(-1), right.operator.argmax(-1)
        left_table = (left_gate[:, None] >> shifts[None]) & 1
        right_table = (right_gate[:, None] >> shifts[None]) & 1
        changes = (left_table != right_table).sum(-1)
        operands_unchanged = ((left.os1.weight.argmax(-1) == right.os1.weight.argmax(-1))
                              & (left.os2.weight.argmax(-1) == right.os2.weight.argmax(-1)))
        counts = torch.bincount(changes, minlength=5)
        restricted = torch.bincount(changes[operands_unchanged], minlength=5)
        records.append(dict(
            layer=depth,
            changed_truth_rows_histogram=counts.tolist(),
            complement_gate_count=int((changes == 4).sum()),
            mean_changed_truth_rows=float(changes.float().mean()),
            unchanged_operand_units=int(operands_unchanged.sum()),
            unchanged_operands_changed_truth_rows_histogram=restricted.tolist(),
            identical_local_functions=int(((changes == 0) & operands_unchanged).sum()),
            truth_table_domain=['00', '01', '10', '11'],
        ))
    return records
