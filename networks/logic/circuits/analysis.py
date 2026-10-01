"""Inspect discrete wiring, recover feature cones, and remove unused internal nodes."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from networks.logic.circuits.gates import GATE_NAMES
from experiments.runtime.wiring import ROOT, write_json


CATALOG = ROOT / 'assets' / 'logic' / 'truth_tables'


def gate_catalog():
    entries = {}
    for path in sorted(CATALOG.glob('*.json')):
        row = json.loads(path.read_text())
        gate = row['id']
        expected = [(gate >> (3 - index)) & 1 for index in range(4)]
        if gate in entries or not 0 <= gate < 16:
            raise ValueError(f'Invalid gate identifier in {path}')
        if row['name'] != GATE_NAMES[gate] or row['outputs'] != expected:
            raise ValueError(f'Truth table differs from the circuit encoding: {path}')
        if row['inputs'] != [[0, 0], [0, 1], [1, 0], [1, 1]]:
            raise ValueError(f'Unexpected truth-table input ordering: {path}')
        entries[gate] = row
    if set(entries) != set(range(16)):
        raise ValueError('The gate catalog must contain every two-input Boolean function')
    return entries


def validate_circuit(circuit):
    if circuit.get('format') != 'oslgn-circuit-v1':
        raise ValueError('Expected an oslgn-circuit-v1 circuit')
    width = circuit.get('input_dim')
    if type(width) is not int or width < 1:
        raise ValueError('The circuit input dimension must be positive')
    if circuit.get('gate_names') != list(GATE_NAMES):
        raise ValueError('The saved gate ordering differs from the operator definitions')
    layers = circuit.get('layers')
    if not isinstance(layers, list) or not layers:
        raise ValueError('A circuit requires at least one logic layer')
    widths = [width]
    for depth, layer in enumerate(layers):
        if set(layer) != {'left', 'right', 'gate'}:
            raise ValueError(f'Unexpected layer fields at depth {depth}')
        if not all(isinstance(layer[key], list) for key in layer):
            raise ValueError('Circuit layer entries must be arrays')
        size = len(layer['gate'])
        if not size or len(layer['left']) != size or len(layer['right']) != size:
            raise ValueError(f'Inconsistent operand and gate counts at depth {depth}')
        for key, upper in (('left', width), ('right', width), ('gate', 16)):
            if any(type(value) is not int or not 0 <= value < upper for value in layer[key]):
                raise ValueError(f'Out-of-range {key} coordinate at depth {depth}')
        width = size
        widths.append(size)
    head = circuit.get('head', {})
    if set(head) != {'weight', 'bias'} or not head['weight']:
        raise ValueError('The circuit requires a nonempty linear class head')
    if len(head['weight']) != len(head['bias']):
        raise ValueError('Head weights and biases disagree on the class count')
    for values in head['weight']:
        if len(values) != width:
            raise ValueError('Head columns must correspond to the final logic features')
    numbers = head['bias'] + [value for row in head['weight'] for value in row]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
        raise ValueError('Head coefficients must be finite numbers')
    return widths


def read_circuit(path):
    circuit = json.loads(Path(path).read_text())
    validate_circuit(circuit)
    return circuit


def operand_dependencies(gate, left, right):
    table = [(gate >> (3 - index)) & 1 for index in range(4)]
    if left == right:
        return (left,) if table[0] != table[3] else ()
    operands = []
    if table[0] != table[2] or table[1] != table[3]:
        operands.append(left)
    if table[0] != table[1] or table[2] != table[3]:
        operands.append(right)
    return tuple(operands)


class CircuitAnalysis:
    def __init__(self, circuit):
        self.circuit = circuit
        self.widths = validate_circuit(circuit)
        self.catalog = gate_catalog()
        self.supports = []
        previous = [frozenset([index]) for index in range(self.widths[0])]
        for layer in circuit['layers']:
            current = []
            for gate, left, right in zip(layer['gate'], layer['left'], layer['right']):
                dependencies = operand_dependencies(gate, left, right)
                current.append(frozenset().union(*(previous[index] for index in dependencies)))
            self.supports.append(current)
            previous = current

    def cone(self, feature):
        final = len(self.circuit['layers']) - 1
        if not 0 <= feature < self.widths[-1]:
            raise ValueError('The selected feature is outside the final layer')
        required = {feature}
        nodes = []
        for depth in range(final, -1, -1):
            layer = self.circuit['layers'][depth]
            preceding = set()
            for index in sorted(required):
                gate = layer['gate'][index]
                left, right = layer['left'][index], layer['right'][index]
                dependencies = operand_dependencies(gate, left, right)
                nodes.append(dict(layer=depth, index=index, gate=gate, name=GATE_NAMES[gate],
                                  left=left, right=right, operands=list(dependencies)))
                preceding.update(dependencies)
            required = preceding
        return dict(feature=feature, input_support=sorted(required),
                    nodes=sorted(nodes, key=lambda row: (row['layer'], row['index'])),
                    head_weights=[row[feature] for row in self.circuit['head']['weight']])

    def active_nodes(self, outputs=None, semantic=True):
        required = set(range(self.widths[-1])) if outputs is None else set(outputs)
        selected = []
        for layer in reversed(self.circuit['layers']):
            selected.append(required)
            preceding = set()
            for index in required:
                left, right = layer['left'][index], layer['right'][index]
                operands = operand_dependencies(layer['gate'][index], left, right) if semantic else (left, right)
                preceding.update(operands)
            required = preceding
        return list(reversed(selected)), required

    def summary(self, distance=100):
        if distance < 0:
            raise ValueError('The long-range distance must be nonnegative')
        active, inputs = self.active_nodes()
        rows = []
        for depth, layer in enumerate(self.circuit['layers']):
            count = len(layer['gate'])
            fanout = Counter(layer['left'] + layer['right'])
            distances = [abs(source - index % self.widths[depth])
                         for side in ('left', 'right') for index, source in enumerate(layer[side])]
            gates = Counter(layer['gate'])
            rows.append(dict(layer=depth, width=count, previous_width=self.widths[depth],
                             semantic_active=len(active[depth]), inactive=count-len(active[depth]),
                             same_operand_rate=sum(a == b for a, b in zip(layer['left'], layer['right'])) / count,
                             unique_operands=len(fanout), maximum_fanout=max(fanout.values()),
                             mean_distance=sum(distances)/len(distances),
                             long_range_rate=sum(value > distance for value in distances)/len(distances),
                             mean_input_support=sum(map(len, self.supports[depth]))/count,
                             maximum_input_support=max(map(len, self.supports[depth])),
                             gates={self.catalog[key]['name']: gates[key] for key in range(16)}))
        head = self.circuit['head']['weight']
        used_features = [index for index in range(self.widths[-1]) if any(row[index] != 0 for row in head)]
        head_active, head_inputs = self.active_nodes(used_features)
        return dict(format=self.circuit['format'], input_dim=self.widths[0], depth=len(rows),
                    logic_units=sum(self.widths[1:]), layers=rows, feature_input_support=len(inputs),
                    head_used_features=used_features, head_input_support=sorted(head_inputs),
                    head_active_units=sum(map(len, head_active)),
                    support_bound='Boolean operand dependency propagation',
                    gate_catalog=[self.catalog[index] for index in range(16)])

    def compact(self):
        selected, _ = self.active_nodes(semantic=False)
        result = deepcopy(self.circuit)
        previous = {index: index for index in range(self.widths[0])}
        mappings = []
        for depth, (layer, indices) in enumerate(zip(self.circuit['layers'], selected)):
            ordered = sorted(indices)
            current = {index: position for position, index in enumerate(ordered)}
            result['layers'][depth] = dict(left=[previous[layer['left'][index]] for index in ordered],
                                           right=[previous[layer['right'][index]] for index in ordered],
                                           gate=[layer['gate'][index] for index in ordered])
            mappings.append(dict(layer=depth, original_indices=ordered))
            previous = current
        validate_circuit(result)
        return result, mappings


def circuit_graph(analysis, features=None):
    circuit = analysis.circuit
    features = list(range(analysis.widths[-1])) if features is None else list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError('Select one or more distinct final features')
    if any(index < 0 or index >= analysis.widths[-1] for index in features):
        raise ValueError('A graph feature lies outside the final layer')
    selected, inputs = analysis.active_nodes(features, semantic=False)
    lines = ['digraph gate_net {', 'rankdir=LR', 'node [shape=box]']
    lines.extend(f'x{index} [label="x[{index}]", shape=circle]' for index in sorted(inputs))
    for depth, indices in enumerate(selected):
        layer = circuit['layers'][depth]
        lines.extend([f'subgraph cluster_{depth} {{', f'label="Logic layer {depth}"'])
        for index in sorted(indices):
            name = GATE_NAMES[layer['gate'][index]]
            lines.append(f'n{depth}_{index} [label="{index}: {name}"]')
        lines.append('}')
        for index in sorted(indices):
            dependencies = operand_dependencies(layer['gate'][index], layer['left'][index], layer['right'][index])
            for side, label in (('left', 'A'), ('right', 'B')):
                source = layer[side][index]
                node = f'x{source}' if depth == 0 else f'n{depth-1}_{source}'
                style = 'solid' if source in dependencies else 'dashed'
                lines.append(f'{node} -> n{depth}_{index} [label="{label}", style={style}]')
    final = len(circuit['layers'])-1
    for target, (weights, bias) in enumerate(zip(circuit['head']['weight'], circuit['head']['bias'])):
        lines.append(f'c{target} [label="class {target}, bias={bias:.5g}", shape=ellipse]')
        for feature in features:
            if weights[feature] != 0:
                lines.append(f'n{final}_{feature} -> c{target} [label="{weights[feature]:.5g}"]')
    lines.extend(['}', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    command = commands.add_parser('inspect')
    command.add_argument('--circuit', required=True)
    command.add_argument('--distance', type=int, default=100)
    command.add_argument('--output', required=True)
    command = commands.add_parser('cone')
    command.add_argument('--circuit', required=True)
    command.add_argument('--feature', type=int, required=True)
    command.add_argument('--output', required=True)
    command = commands.add_parser('compact')
    command.add_argument('--circuit', required=True)
    command.add_argument('--output', required=True)
    command = commands.add_parser('graph')
    command.add_argument('--circuit', required=True)
    command.add_argument('--feature', type=int, action='append')
    command.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.circuit).resolve(), Path(args.output).resolve()
    if source == output:
        raise ValueError('Circuit inspection requires a separate output path')
    analysis = CircuitAnalysis(read_circuit(source))
    if args.operation == 'graph':
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(circuit_graph(analysis, args.feature))
        print(f'Saved circuit graph to {output}')
        return
    if args.operation == 'inspect':
        result = analysis.summary(args.distance)
    elif args.operation == 'cone':
        result = analysis.cone(args.feature)
    else:
        result, mappings = analysis.compact()
        write_json(str(output) + '.mapping.json', dict(source=str(source), layers=mappings))
    write_json(output, result)
    print(f'Saved {args.operation} output to {output}')
