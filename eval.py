import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data import load_dataset
from utils.experiment import restore_model
from utils.symbolic import evaluate_circuit
from utils.util import evaluate, write_json


def main(args):
    torch.set_num_threads(1)
    model, config = restore_model(args.checkpoint, args.device)
    dataset = args.dataset or config['data'].get('dataset', 'boolean' if config['model']['num_classes'] == 2 else 'mnist')
    split = 'test' if dataset == 'mnist' and not args.data else 'val'
    loader = DataLoader(load_dataset(config, split, args.data, dataset, args.download), batch_size=128)
    metrics = evaluate(model, loader, args.device)
    circuit = model.export_circuit()
    maximum_error = 0.0
    with torch.no_grad():
        for images, _ in loader:
            features, logits = evaluate_circuit(circuit, images.numpy())
            np.testing.assert_array_equal(features, model.logic_features(images.to(args.device)).cpu().numpy())
            expected = model(images.to(args.device)).cpu().numpy()
            np.testing.assert_allclose(logits, expected, atol=1e-5, rtol=1e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(logits - expected))))
    metrics.update(circuit_features_exact=True, circuit_logit_max_abs_error=maximum_error)
    write_json(args.output, metrics)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['mnist', 'boolean'])
    parser.add_argument('--data')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/eval/metrics.json')
    main(parser.parse_args())
