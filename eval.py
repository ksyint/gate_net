"""Evaluate a checkpoint and independently verify its extracted circuit."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from circuit import evaluate_circuit, symbolic_equations
from model import OSLGN
from utils import evaluate, load_dataset, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/eval")
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = state["config"]
    model = OSLGN(**cfg["model"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(load_dataset(cfg, "test" if args.mnist else "val", args.data, args.mnist, args.download), batch_size=128)
    metrics = evaluate(model, loader, args.device)
    circuit = model.export_circuit()
    max_error, identical = 0.0, True
    with torch.no_grad():
        for x, _ in loader:
            features, logits = evaluate_circuit(circuit, x.numpy())
            expected_features = model.logic_features(x.to(args.device)).cpu().numpy()
            expected_logits = model(x.to(args.device)).cpu().numpy()
            identical &= np.array_equal(features, expected_features)
            max_error = max(max_error, float(np.max(np.abs(logits - expected_logits))))
            np.testing.assert_allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)
    if not identical:
        raise RuntimeError("extracted Boolean features differ from the model")
    metrics.update(circuit_features_exact=identical, circuit_logit_max_abs_error=max_error)
    output = Path(args.output_dir)
    write_json(output / "metrics.json", metrics)
    write_json(output / "circuit.json", circuit)
    (output / "equations.txt").write_text(symbolic_equations(circuit), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
