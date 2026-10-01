# Learning to Wire: End-to-End Operand Selection for Symbolic Logic Networks

PyTorch implementation of OSLGN with learned operand selection and exact circuit export.

## Summary

Each unit selects two operands and one of sixteen Boolean gates. Argmax selectors and rounded gate outputs use straight-through gradients. Gaussian initialization starts the wiring near neighboring inputs. A final linear classifier reads the binary logic features.

The experiment scripts select a dataset and depth. Logic layers are in `utils/logic`, datasets in `utils/data`, and exact circuit execution/export in `utils/symbolic`.

## Environmental Set-up

Use PyTorch with CUDA. Experiment scripts and circuit-export verification use CUDA, selected with `--device cuda` or `--device cuda:N`. NumPy handles the exported circuit representation.

```bash
conda create -n oslgn python=3.10
conda activate oslgn
pip install -r requirements.txt
```

## Training

Install torchvision compatible with your PyTorch build, then select depth 2, 4, or 8:

```bash
pip install torchvision
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mnist --depth 4 --seed 42 --download
```

Profiles live at `configs/<dataset>/depth<N>.yaml`. `--config` selects a custom file, `--epochs` sets the training duration, and `--output` changes the checkpoint directory. The direct Adam loop trains hard logic gates and records measured metrics.

MNIST profiles use 784 inputs, width 512, 10 final logic features, sigma 2.0, Adam at 0.001, batch size 64, and 50 epochs. Pixel values are thresholded at 0.5. A seeded 10% split of the training data selects checkpoints; evaluation uses the separate test set. `local_init` and `detach_operands` enable the wiring ablations.

For external arrays, pass `--data images.npz` and a matching configuration. The archive supplies `x_train`, `y_train`, `x_val`, `y_val`. Images are flattened and thresholded; normalize them to [0,1] beforehand. Labels are integer class IDs.

## Depth, width, and wiring experiments

The 270 configurations in `configs/mnist/sweeps` combine six depths (2/3/4/5/6/8), three hidden widths (128/256/512), five operand initialization choices, and three seeds (42/123/456). Initializations use Gaussian locality widths 0.5/1/2/4 or random operand scores. Every depth includes hidden logic units, so each width setting changes the network.

Select experiments with the direct runner:

```bash
python run_experiments.py --depths 2 4 8 --widths 512 \
  --initializations local_sigma_2 random --seeds 42 123 456 --download --device cuda
```

Use `--data images.npz` for prepared 784-input, 10-class arrays, or cached MNIST under `datasets`. Each profile preserves the 50-epoch Adam schedule and its own result directory under `results/mnist/wiring`; `--epochs` changes the total duration. `python run_experiments.py --dry-run` reads all profiles and displays selected parameter counts without model execution. `utils/experiment_grid.py` defines and validates the grid, while each YAML file is also accepted by `train.py --config`.

## Evaluation and export

```bash
python eval.py --checkpoint results/mnist/depth4/best.pth --dataset mnist
python export.py --checkpoint results/mnist/depth4/best.pth --output results/mnist/circuit
```

Evaluation compares held-out examples with an independent NumPy circuit evaluator. `circuit.json` stores wiring, gates, and the linear head. `equations.txt` names every intermediate node once, keeping the graph compact. Extraction does not need the training dataset.

## Using the exported circuit

```python
import json
from utils.symbolic import evaluate_circuit

with open("results/mnist/circuit/circuit.json") as handle:
    circuit = json.load(handle)
features, logits = evaluate_circuit(circuit, binary_input_array)
predictions = logits.argmax(axis=-1)
```

The Boolean features match the trained network exactly. Class logits apply the exported real-valued head to those features. Circuit inputs contain zeros and ones.
