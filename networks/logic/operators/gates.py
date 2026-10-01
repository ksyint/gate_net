"""Straight-through logic layers and exact circuit representations."""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from functools import lru_cache

GATE_NAMES = (
    "FALSE",
    "AND",
    "A_AND_NOT_B",
    "A",
    "NOT_A_AND_B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT_B",
    "A_OR_NOT_B",
    "NOT_A",
    "NOT_A_OR_B",
    "NAND",
    "TRUE",
)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def hard_selection(weights):
    """One-hot argmax in forward, identity derivative in backward (paper Eq.)."""
    mask = torch.zeros_like(weights).scatter_(-1, weights.argmax(-1, keepdim=True), 1.0)
    return mask + (weights - weights.detach())


def boolean_gates(a, b):
    """All 16 Table-I multilinear surrogates, last axis is gate ID."""
    ab = a * b
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack(
        (
            zero,
            ab,
            a - ab,
            a,
            b - ab,
            b,
            a + b - 2 * ab,
            a + b - ab,
            one - a - b + ab,
            one - a - b + 2 * ab,
            one - b,
            one - b + ab,
            one - a,
            one - a + ab,
            one - ab,
            one,
        ),
        dim=-1,
    )


class LogicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=2.0, local_init=True, detach_operands=False):
        super().__init__()
        self.os1 = OperandSelector(input_dim, output_dim, sigma, shift=0, local_init=local_init)
        self.os2 = OperandSelector(input_dim, output_dim, sigma, shift=1, local_init=local_init)
        self.operator = nn.Parameter(torch.empty(output_dim, 16))
        nn.init.normal_(self.operator, std=0.1)
        self.detach_operands = detach_operands

    def forward(self, x):
        a, b = self.os1(x), self.os2(x)
        if self.detach_operands:
            a, b = a.detach(), b.detach()
        gates = boolean_gates(a, b)
        return RoundSTE.apply((gates * hard_selection(self.operator)).sum(-1))

    @torch.no_grad()
    def export(self):
        return {
            "left": self.os1.weight.argmax(-1).tolist(),
            "right": self.os2.weight.argmax(-1).tolist(),
            "gate": self.operator.argmax(-1).tolist(),
        }


class OperandSelector(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=2.0, shift=0, local_init=True):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or sigma <= 0:
            raise ValueError("dimensions and locality sigma must be positive")
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        if local_init:
            centers = (torch.arange(output_dim) + shift) % input_dim
            positions = torch.arange(input_dim)
            with torch.no_grad():
                self.weight.copy_(
                    torch.exp(-(positions[None, :] - centers[:, None]).float().square() / (2 * sigma**2))
                )
        else:
            nn.init.normal_(self.weight, std=0.1)

    def forward(self, x):
        return F.linear(x, hard_selection(self.weight))


class OSLGN(nn.Module):
    """Hard logic features followed by the paper's trainable linear class head."""

    def __init__(
        self,
        input_dim=784,
        hidden_dim=512,
        num_classes=10,
        depth=4,
        sigma=2.0,
        local_init=True,
        detach_operands=False,
    ):
        super().__init__()
        if depth < 1 or min(input_dim, hidden_dim, num_classes) < 1:
            raise ValueError("dimensions and depth must be positive")
        self.input_dim, self.num_classes = input_dim, num_classes
        widths = [input_dim] + [hidden_dim] * (depth - 1) + [num_classes]
        self.layers = nn.ModuleList(
            [
                LogicLayer(widths[i], widths[i + 1], sigma, local_init, detach_operands)
                for i in range(depth)
            ]
        )
        self.head = nn.Linear(num_classes, num_classes)

    def logic_features(self, x):
        x = x.reshape(x.shape[0], -1)
        if x.shape[-1] != self.input_dim:
            raise ValueError("input dimension does not match the circuit")
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return self.head(self.logic_features(x))

    @torch.no_grad()
    def export_circuit(self):
        return {
            "format": "oslgn-circuit-v1",
            "input_dim": self.input_dim,
            "gate_names": list(GATE_NAMES),
            "layers": [layer.export() for layer in self.layers],
            "head": {"weight": self.head.weight.tolist(), "bias": self.head.bias.tolist()},
        }


def evaluate_circuit(circuit, inputs):
    """Return binary features and float class logits, with no PyTorch dependency."""
    if circuit.get("format") != "oslgn-circuit-v1":
        raise ValueError("unknown circuit format")
    x = np.asarray(inputs)
    if x.ndim != 2 or x.shape[1] != circuit["input_dim"] or not np.isin(x, [0, 1]).all():
        raise ValueError("inputs must be a binary [N, input_dim] array")
    x = x.astype(np.int64)
    for layer in circuit["layers"]:
        a, b = x[:, layer["left"]], x[:, layer["right"]]
        gate = np.asarray(layer["gate"], dtype=np.int64)
        # IDs enumerate truth tables with 00 as most significant and 11 as least.
        x = (gate[None, :] >> (3 - (2 * a + b))) & 1
    weight = np.asarray(circuit["head"]["weight"], dtype=np.float32)
    bias = np.asarray(circuit["head"]["bias"], dtype=np.float32)
    return x, x.astype(np.float32) @ weight.T + bias


def symbolic_equations(circuit):
    """Name every DAG node once, avoiding exponential expression expansion."""
    lines = ["# Binary inputs x[0], ..., x[input_dim - 1]."]
    previous = [f"x[{j}]" for j in range(circuit["input_dim"])]
    for depth, layer in enumerate(circuit["layers"]):
        current = []
        for j, (left, right, gate) in enumerate(zip(layer["left"], layer["right"], layer["gate"])):
            a, b = previous[left], previous[right]
            expressions = (
                "False",
                f"({a} and {b})",
                f"({a} and not {b})",
                a,
                f"(not {a} and {b})",
                b,
                f"({a} != {b})",
                f"({a} or {b})",
                f"not ({a} or {b})",
                f"({a} == {b})",
                f"not {b}",
                f"({a} or not {b})",
                f"not {a}",
                f"(not {a} or {b})",
                f"not ({a} and {b})",
                "True",
            )
            name = f"layer{depth}_{j}"
            lines.append(f"{name} = {expressions[gate]}")
            current.append(name)
        previous = current
    lines.append("# Final logic features: " + ", ".join(previous))
    lines.append("# Class logits use the linear head saved in circuit.json.")
    return "\n".join(lines) + "\n"


def minimize_features(circuit, max_support=16):
    from pyeda.inter import expr
    from pyeda.inter import exprvar
    from pyeda.inter import espresso_exprs

    if circuit.get('format') != 'oslgn-circuit-v1':
        raise ValueError('Unsupported circuit format')
    if max_support < 1:
        raise ValueError('max_support must be positive')

    @lru_cache(None)
    def node(depth, index):
        if depth < 0:
            return exprvar('x', index)
        layer = circuit['layers'][depth]
        gate = layer['gate'][index]
        if gate in (0, 15):
            return expr(int(gate == 15))
        a, b = node(depth - 1, layer['left'][index]), node(depth - 1, layer['right'][index])
        result = expr(0)
        for av, bv in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if (gate >> (3 - (2 * av + bv))) & 1:
                result |= (a if av else ~a) & (b if bv else ~b)
        return result.simplify()

    outputs = []
    last = len(circuit['layers']) - 1
    for index in range(len(circuit['layers'][last]['gate'])):
        expression = node(last, index)
        support = len(expression.support)
        eligible = support <= max_support and not expression.is_zero() and not expression.is_one()
        minimized = espresso_exprs(expression.to_dnf())[0] if eligible else expression
        outputs.append(
            {
                'feature': index,
                'support_variables': support,
                'espresso_applied': eligible,
                'expression': str(expression),
                'result': str(minimized),
            }
        )
    return outputs


@torch.no_grad()
def wiring_statistics(model):
    rows = []
    for index, layer in enumerate(model.layers):
        left, right = layer.os1.weight.argmax(-1), layer.os2.weight.argmax(-1)
        centers = torch.arange(len(left), device=left.device) % layer.os1.weight.shape[1]
        distances = torch.cat(((left - centers).abs(), (right - centers).abs()))
        gates = layer.operator.argmax(-1)
        gradients = [
            parameter.grad.detach().float().square().sum()
            for parameter in layer.parameters()
            if parameter.grad is not None
        ]
        rows.append(
            {
                'layer': index,
                'logic_units': len(left),
                'same_operand_rate': (left == right).float().mean().item(),
                'unique_operands': torch.cat((left, right)).unique().numel(),
                'long_range_rate': (distances > 100).float().mean().item(),
                'last_batch_gradient_norm': (
                    torch.stack(gradients).sum().sqrt().item() if gradients else 0.0
                ),
                'gates': {name: int((gates == gate).sum()) for gate, name in enumerate(GATE_NAMES)},
            }
        )
    return rows
