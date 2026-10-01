"""Binarized data, deterministic experiments, and metrics."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset(cfg, split="train", path=None, mnist=False, download=False):
    if path and mnist:
        raise ValueError("choose NPZ or MNIST, not both")
    if mnist:
        from torchvision.datasets import MNIST
        data = MNIST(cfg["data"]["root"], train=(split != "test"), download=download)
        x = data.data.float().flatten(1) / 255.0
        y = data.targets
        if split != "test":
            generator = torch.Generator().manual_seed(cfg["train"]["seed"])
            order = torch.randperm(len(y), generator=generator)
            val_count = max(1, round(len(y) * cfg["data"].get("validation_fraction", 0.1)))
            if val_count >= len(y):
                raise ValueError("validation_fraction must leave training samples")
            indices = order[val_count:] if split == "train" else order[:val_count]
            x, y = x[indices], y[indices]
    elif path:
        with np.load(path, allow_pickle=False) as blob:
            x = torch.tensor(blob[f"x_{split}"], dtype=torch.float32)
            y = torch.tensor(blob[f"y_{split}"], dtype=torch.long)
        x = x.flatten(1)
    else:
        gen = torch.Generator().manual_seed(1101 if split == "train" else 1102)
        x = torch.randint(0, 2, (512 if split == "train" else 128, cfg["model"]["input_dim"]), generator=gen).float()
        # A binary symbolic target with nontrivial AND/XOR structure.
        y = ((x[:, 0].bool() ^ x[:, 1].bool()) | (x[:, 2].bool() & x[:, 3].bool())).long()
        if cfg["model"]["num_classes"] != 2:
            raise ValueError("synthetic Boolean task requires num_classes=2; use configs/smoke.yaml")
    if not torch.isfinite(x).all():
        raise ValueError("inputs must be finite")
    x = (x > cfg["data"]["threshold"]).float()
    if x.shape[1] != cfg["model"]["input_dim"] or y.shape != x.shape[:1] or not len(y):
        raise ValueError("invalid data dimensions")
    if y.min() < 0 or y.max() >= cfg["model"]["num_classes"]:
        raise ValueError("class labels out of range")
    return TensorDataset(x, y.long())


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
