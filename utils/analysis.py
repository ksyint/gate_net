"""Per-layer wiring and gradient diagnostics for learned logic networks."""
import torch

from .logic import GATE_NAMES


@torch.no_grad()
def wiring_statistics(model):
    rows = []
    for index, layer in enumerate(model.layers):
        left, right = layer.os1.weight.argmax(-1), layer.os2.weight.argmax(-1)
        centers = torch.arange(len(left), device=left.device) % layer.os1.weight.shape[1]
        distances = torch.cat(((left - centers).abs(), (right - centers).abs()))
        gates = layer.operator.argmax(-1)
        gradients = [parameter.grad.detach().float().square().sum() for parameter in layer.parameters()
                     if parameter.grad is not None]
        rows.append({'layer': index, 'logic_units': len(left),
                     'same_operand_rate': (left == right).float().mean().item(),
                     'unique_operands': torch.cat((left, right)).unique().numel(),
                     'long_range_rate': (distances > 100).float().mean().item(),
                     'last_batch_gradient_norm': torch.stack(gradients).sum().sqrt().item() if gradients else 0.0,
                     'gates': {name: int((gates == gate).sum()) for gate, name in enumerate(GATE_NAMES)}})
    return rows
