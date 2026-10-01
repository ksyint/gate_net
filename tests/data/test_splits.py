import sys
import types

import torch
from utils.data import load_dataset
from utils.util import load_config


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
    cfg = {'model': {'input_dim': 4, 'num_classes': 2}, 'train': {'seed': 42},
           'data': {'root': 'unused', 'threshold': 0.5, 'validation_fraction': 0.2}}
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
