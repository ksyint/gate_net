"""Extract a trained hard-wired circuit without requiring its training dataset."""
import argparse
from pathlib import Path

import torch
import numpy as np

from utils.experiment import restore_model
from utils.symbolic import evaluate_circuit, symbolic_equations
from utils.util import write_json


def main(args):
    torch.set_num_threads(1)
    model, config = restore_model(args.checkpoint, args.device)
    circuit = model.export_circuit()
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randint(0, 2, (64, config['model']['input_dim']), generator=generator).float()
    features, logits = evaluate_circuit(circuit, inputs.numpy())
    with torch.no_grad():
        np.testing.assert_array_equal(features, model.logic_features(inputs.to(args.device)).cpu().numpy())
        np.testing.assert_allclose(logits, model(inputs.to(args.device)).cpu().numpy(), atol=1e-5, rtol=1e-5)
    output = Path(args.output)
    write_json(output / 'circuit.json', circuit)
    (output / 'equations.txt').write_text(symbolic_equations(circuit), encoding='utf-8')
    print(f'Circuit and equations saved to {output}; 64 binary inputs agree exactly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/circuit')
    main(parser.parse_args())
