import torch
from torch import nn

from .gates import hard_selection, boolean_gates, RoundSTE
from .selectors import OperandSelector


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
