import torch

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
