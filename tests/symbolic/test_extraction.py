import itertools
import numpy as np
import torch
from torch import nn
from utils.models import OSLGN
from utils.logic import LogicLayer, OperandSelector, boolean_gates, hard_selection, RoundSTE
from utils.symbolic import evaluate_circuit


def test_export_is_exact_exhaustively_and_head_is_preserved():
    torch.manual_seed(6)
    model = OSLGN(input_dim=4, hidden_dim=9, num_classes=3, depth=3)
    x = torch.tensor(list(itertools.product([0., 1.], repeat=4)))
    circuit = model.export_circuit()
    features, logits = evaluate_circuit(circuit, x.numpy())
    assert np.array_equal(features, model.logic_features(x).detach().numpy())
    np.testing.assert_allclose(logits, model(x).detach().numpy(), atol=1e-6)
    assert set(np.unique(features)).issubset({0, 1})
