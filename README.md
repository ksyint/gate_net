# Learning to Wire: End-to-End Operand Selection for Symbolic Logic Networks

Independent PyTorch implementation of Operand Selective Logic Gated Networks (OSLGN), based on the supplied manuscript by Soon Ho Choi and Soo Yong Kim.

## Summary

Each logic unit learns two operand selectors and one of sixteen Boolean gates. Hard argmax selection uses a straight-through estimator; a Gaussian proximity prior initializes local wiring. Arithmetic gate surrogates and straight-through rounding keep every logic-layer output binary. A linear classification head maps the final logic features to class logits.

The extracted JSON circuit runs independently with NumPy. It contains the exact discrete wiring, gates, and final linear head. Symbolic equations name each intermediate node once, preserving the computation graph without expanding exponentially large expressions.

## Environmental Set-up

```bash
conda create -n oslgn python=3.10
conda activate oslgn
pip install -r requirements.txt
```

## Quick Start

```bash
python train.py --smoke
python eval.py --checkpoint results/smoke/best.pth
python -m pytest -q
```

The synthetic Boolean task checks real optimization without downloading data. Evaluation writes `results/eval/circuit.json`, `equations.txt`, and actual metrics. It verifies every held-out sample against the independent circuit evaluator.

## MNIST

Install a torchvision version compatible with your PyTorch installation, then run:

```bash
pip install torchvision
python train.py --config configs/oslgn.yaml --mnist --download
python eval.py --checkpoint results/oslgn/best.pth --mnist
```

The default configuration follows the depth-4 setup: 784 binary inputs, width 512, 10 final logic features, a linear classification head, sigma 2.0, Adam at 0.001, batch size 64, and 50 epochs. Images are thresholded at 0.5. A seeded 10% split of the original training set selects checkpoints; evaluation uses the separate MNIST test set. Set depth to 2 or 8 for the depth study; `local_init` and `detach_operands` expose the two ablations.

For an existing NPZ, use keys `x_train`, `y_train`, `x_val`, `y_val` and configure the dimensions and class count. Inputs are normalized to [0,1] by the data producer; this loader thresholds them. Labels are integer class IDs.

```bash
python train.py --config configs/oslgn.yaml --data binary_images.npz
python eval.py --checkpoint results/oslgn/best.pth --data binary_images.npz
```

## Circuit inference

```python
import json
from circuit import evaluate_circuit

with open("results/eval/circuit.json") as handle:
    circuit = json.load(handle)
features, logits = evaluate_circuit(circuit, binary_input_array)
predictions = logits.argmax(axis=-1)
```

Boolean equivalence applies exactly to the logic features. The final class decision additionally uses the saved real-valued linear head. The exporter retains that head and does not turn it into a Boolean gate. Inputs to the extracted circuit must already be binary.

## Validation and scope

Tests exhaust all four inputs for all sixteen gates, verify straight-through derivatives and the operand-detachment ablation, and exhaustively compare an extracted multilayer circuit on all sixteen four-bit inputs. The synthetic run checks optimization and serialization. Full MNIST training and manuscript accuracy numbers have not been reproduced here. No trained weights or benchmark scores are bundled.

## Source

*Learning to Wire: End-to-End Operand Selection for Symbolic Logic Networks*, Soon Ho Choi and Soo Yong Kim, supplied manuscript. The core follows Section III and Listings 1–2; MNIST settings follow Section IV-A.
