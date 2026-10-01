import numpy as np


def evaluate_circuit(circuit, inputs):
    """Return binary features and float class logits, with no PyTorch dependency."""
    if circuit.get("format") != "oslgn-circuit-v1":
        raise ValueError("unknown circuit format")
    x = np.asarray(inputs)
    if x.ndim != 2 or x.shape[1] != circuit["input_dim"] or not np.isin(x, [0, 1]).all():
        raise ValueError("inputs must be a binary [N, input_dim] array")
    x = x.astype(np.int64)
    for layer in circuit["layers"]:
        a, b = x[:, layer["left"]], x[:, layer["right"]]
        gate = np.asarray(layer["gate"], dtype=np.int64)
        # IDs enumerate truth tables with 00 as most significant and 11 as least.
        x = (gate[None, :] >> (3 - (2 * a + b))) & 1
    weight = np.asarray(circuit["head"]["weight"], dtype=np.float32)
    bias = np.asarray(circuit["head"]["bias"], dtype=np.float32)
    return x, x.astype(np.float32) @ weight.T + bias
