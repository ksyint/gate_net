import itertools
import numpy as np
import torch
from torch import nn
from utils.models import OSLGN
from utils.logic import LogicLayer, OperandSelector, boolean_gates, hard_selection, RoundSTE
from utils.symbolic import evaluate_circuit


def test_local_initialization_selects_shifted_neighbors():
    first = OperandSelector(5, 7, sigma=2, shift=0)
    second = OperandSelector(5, 7, sigma=2, shift=1)
    assert first.weight.argmax(-1).tolist() == [0, 1, 2, 3, 4, 0, 1]
    assert second.weight.argmax(-1).tolist() == [1, 2, 3, 4, 0, 1, 2]


def test_operand_learning_and_detachment_ablation():
    layer = LogicLayer(3, 1)
    with torch.no_grad():
        layer.operator.fill_(-1)
        layer.operator[0, 1] = 2  # AND
    layer(torch.tensor([[1., 1., 0.]])).sum().backward()
    assert layer.os1.weight.grad.abs().sum() > 0
    assert layer.os2.weight.grad.abs().sum() > 0
    layer.zero_grad(set_to_none=True)
    layer.detach_operands = True
    layer(torch.tensor([[1., 1., 0.]])).sum().backward()
    assert layer.os1.weight.grad is None
    assert layer.os2.weight.grad is None
    assert layer.operator.grad is not None
