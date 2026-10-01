"""Operand Selective Logic Gated Networks with exact hard circuit extraction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

GATE_NAMES = ("FALSE", "AND", "A_AND_NOT_B", "A", "NOT_A_AND_B", "B", "XOR", "OR",
              "NOR", "XNOR", "NOT_B", "A_OR_NOT_B", "NOT_A", "NOT_A_OR_B", "NAND", "TRUE")


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
    return torch.stack((zero, ab, a - ab, a, b - ab, b, a + b - 2 * ab,
                        a + b - ab, one - a - b + ab, one - a - b + 2 * ab,
                        one - b, one - b + ab, one - a, one - a + ab,
                        one - ab, one), dim=-1)


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
                self.weight.copy_(torch.exp(-(positions[None, :] - centers[:, None]).float().square() / (2 * sigma ** 2)))
        else:
            nn.init.normal_(self.weight, std=0.1)

    def forward(self, x):
        return F.linear(x, hard_selection(self.weight))


class LogicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=2.0, local_init=True,
                 detach_operands=False):
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
        return {"left": self.os1.weight.argmax(-1).tolist(),
                "right": self.os2.weight.argmax(-1).tolist(),
                "gate": self.operator.argmax(-1).tolist()}


class OSLGN(nn.Module):
    """Hard logic features followed by the paper's trainable linear class head."""
    def __init__(self, input_dim=784, hidden_dim=512, num_classes=10, depth=4,
                 sigma=2.0, local_init=True, detach_operands=False):
        super().__init__()
        if depth < 1 or min(input_dim, hidden_dim, num_classes) < 1:
            raise ValueError("dimensions and depth must be positive")
        self.input_dim, self.num_classes = input_dim, num_classes
        widths = [input_dim] + [hidden_dim] * (depth - 1) + [num_classes]
        self.layers = nn.ModuleList([LogicLayer(widths[i], widths[i + 1], sigma,
                                               local_init, detach_operands) for i in range(depth)])
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
        return {"format": "oslgn-circuit-v1", "input_dim": self.input_dim,
                "gate_names": list(GATE_NAMES), "layers": [layer.export() for layer in self.layers],
                "head": {"weight": self.head.weight.tolist(), "bias": self.head.bias.tolist()}}
