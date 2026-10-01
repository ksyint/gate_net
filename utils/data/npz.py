import numpy as np
import torch


def read_split(path, split):
    with np.load(path, allow_pickle=False) as arrays:
        images = torch.tensor(arrays[f'x_{split}'], dtype=torch.float32)
        labels = torch.tensor(arrays[f'y_{split}'], dtype=torch.long)
    return images.flatten(1), labels
