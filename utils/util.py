import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = count = 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += len(y)
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cuda_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('OSLGN model training and evaluation require CUDA')
    return device
