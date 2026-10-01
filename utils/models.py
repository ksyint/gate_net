import torch
from torch import nn

from .logic import GATE_NAMES, LogicLayer


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
