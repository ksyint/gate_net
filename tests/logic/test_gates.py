import itertools
import numpy as np
import torch
from torch import nn
from utils.models import OSLGN
from utils.logic import LogicLayer, OperandSelector, boolean_gates, hard_selection, RoundSTE
from utils.symbolic import evaluate_circuit


def test_all_sixteen_gate_truth_tables():
    ab = torch.tensor(list(itertools.product([0., 1.], repeat=2)))
    actual = boolean_gates(ab[:, 0], ab[:, 1])
    expected = torch.tensor([[(gate >> (3 - row)) & 1 for gate in range(16)] for row in range(4)]).float()
    torch.testing.assert_close(actual, expected)


def test_selector_and_round_ste_gradients():
    w = torch.tensor([[0.1, 0.5, -0.2]], requires_grad=True)
    selected = hard_selection(w)
    torch.testing.assert_close(selected, torch.tensor([[0., 1., 0.]]))
    (selected * torch.tensor([[1., 3., 7.]])).sum().backward()
    torch.testing.assert_close(w.grad, torch.tensor([[1., 3., 7.]]))
    x = torch.tensor([0.2, 0.7], requires_grad=True)
    RoundSTE.apply(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
