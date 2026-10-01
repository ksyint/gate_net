"""Minimize exported Boolean features while preserving the saved linear class head."""
import argparse
import json
from pathlib import Path

from utils.symbolic.minimize import minimize_features
from utils.util import write_json


def main(args):
    circuit = json.loads(Path(args.circuit).read_text())
    write_json(args.output, {'features': minimize_features(circuit, args.max_support), 'head': circuit['head']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--circuit', required=True)
    parser.add_argument('--max-support', type=int, default=16)
    parser.add_argument('--output', default='results/circuit/minimized.json')
    main(parser.parse_args())
