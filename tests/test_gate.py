"""Logic selection, exact circuit extraction, and split invariants."""

import sys
import types
import torch
import itertools
import numpy as np
from experiment import load_dataset, load_config
from torch import nn
from logic import (
    OSLGN,
    LogicLayer,
    OperandSelector,
    boolean_gates,
    hard_selection,
    RoundSTE,
    evaluate_circuit,
)


def test_mnist_validation_is_disjoint_from_training(monkeypatch):
    class FakeMNIST:
        def __init__(self, root, train, download):
            size = 10 if train else 4
            numbers = torch.arange(size)
            bits = ((numbers[:, None] >> torch.arange(4)) & 1).byte() * 255
            self.data = bits.reshape(size, 2, 2)
            self.targets = numbers % 2

    package, datasets = types.ModuleType('torchvision'), types.ModuleType('torchvision.datasets')
    datasets.MNIST = FakeMNIST
    monkeypatch.setitem(sys.modules, 'torchvision', package)
    monkeypatch.setitem(sys.modules, 'torchvision.datasets', datasets)
    cfg = {
        'model': {'input_dim': 4, 'num_classes': 2},
        'train': {'seed': 42},
        'data': {'root': 'unused', 'threshold': 0.5, 'validation_fraction': 0.2},
    }
    train = load_dataset(cfg, 'train', dataset='mnist')
    val = load_dataset(cfg, 'val', dataset='mnist')
    test = load_dataset(cfg, 'test', dataset='mnist')
    train_rows = {tuple(row.tolist()) for row in train.tensors[0]}
    val_rows = {tuple(row.tolist()) for row in val.tensors[0]}
    assert len(train) == 8 and len(val) == 2 and len(test) == 4
    assert train_rows.isdisjoint(val_rows)


def test_boolean_task_uses_disjoint_seeded_splits():
    cfg = load_config('configs/boolean/depth2.yaml')
    train = load_dataset(cfg, 'train', dataset='boolean')
    val = load_dataset(cfg, 'val', dataset='boolean')
    assert len(train) == 512 and len(val) == 128
    assert not torch.equal(train.tensors[0][:128], val.tensors[0])


def test_all_sixteen_gate_truth_tables():
    ab = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    actual = boolean_gates(ab[:, 0], ab[:, 1])
    expected = torch.tensor(
        [[(gate >> (3 - row)) & 1 for gate in range(16)] for row in range(4)]
    ).float()
    torch.testing.assert_close(actual, expected)


def test_selector_and_round_ste_gradients():
    w = torch.tensor([[0.1, 0.5, -0.2]], requires_grad=True)
    selected = hard_selection(w)
    torch.testing.assert_close(selected, torch.tensor([[0.0, 1.0, 0.0]]))
    (selected * torch.tensor([[1.0, 3.0, 7.0]])).sum().backward()
    torch.testing.assert_close(w.grad, torch.tensor([[1.0, 3.0, 7.0]]))
    x = torch.tensor([0.2, 0.7], requires_grad=True)
    RoundSTE.apply(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


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
    layer(torch.tensor([[1.0, 1.0, 0.0]])).sum().backward()
    assert layer.os1.weight.grad.abs().sum() > 0
    assert layer.os2.weight.grad.abs().sum() > 0
    layer.zero_grad(set_to_none=True)
    layer.detach_operands = True
    layer(torch.tensor([[1.0, 1.0, 0.0]])).sum().backward()
    assert layer.os1.weight.grad is None
    assert layer.os2.weight.grad is None
    assert layer.operator.grad is not None


def test_export_is_exact_exhaustively_and_head_is_preserved():
    torch.manual_seed(6)
    model = OSLGN(input_dim=4, hidden_dim=9, num_classes=3, depth=3)
    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)))
    circuit = model.export_circuit()
    features, logits = evaluate_circuit(circuit, x.numpy())
    assert np.array_equal(features, model.logic_features(x).detach().numpy())
    np.testing.assert_allclose(logits, model(x).detach().numpy(), atol=1e-6)
    assert set(np.unique(features)).issubset({0, 1})
