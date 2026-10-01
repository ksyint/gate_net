import torch
from torch import nn
from torch.nn import functional as F

from .gates import hard_selection


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
