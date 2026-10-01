"""NumPy-only execution and compact symbolic export of extracted OSLGN circuits."""
from __future__ import annotations

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


def symbolic_equations(circuit):
    """Name every DAG node once, avoiding exponential expression expansion."""
    lines = ["# Binary inputs x[0], ..., x[input_dim - 1]."]
    previous = [f"x[{j}]" for j in range(circuit["input_dim"])]
    for depth, layer in enumerate(circuit["layers"]):
        current = []
        for j, (left, right, gate) in enumerate(zip(layer["left"], layer["right"], layer["gate"])):
            a, b = previous[left], previous[right]
            expressions = ("False", f"({a} and {b})", f"({a} and not {b})", a,
                           f"(not {a} and {b})", b, f"({a} != {b})", f"({a} or {b})",
                           f"not ({a} or {b})", f"({a} == {b})", f"not {b}",
                           f"({a} or not {b})", f"not {a}", f"(not {a} or {b})",
                           f"not ({a} and {b})", "True")
            name = f"layer{depth}_{j}"
            lines.append(f"{name} = {expressions[gate]}")
            current.append(name)
        previous = current
    lines.append("# Final logic features: " + ", ".join(previous))
    lines.append("# Class logits use the linear head saved in circuit.json.")
    return "\n".join(lines) + "\n"
