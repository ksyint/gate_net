# gate_net

[Paper](https://linklings.s3.amazonaws.com/organizations/WCCI/wcci2026/submissions/stype114/YihWd-ijcnn_pap3087s2.pdf)

gate_net learns Boolean wiring and gate choices directly from binarized MNIST. Each logic unit selects two operands and one of sixteen gates with straight-through gradients. Gaussian initialization favors neighboring inputs, and a learned linear classifier reads the final ten Boolean features.

## Environmental Set-up

Use Python 3.10+ with a matched CUDA build of PyTorch and torchvision:

```bash
conda create -n gate_net python=3.10
conda activate gate_net
pip install -r requirements.txt
```

Training, checkpoint evaluation, image prediction, and model export run on CUDA. Select a card with `CUDA_VISIBLE_DEVICES` or `--device cuda:N`. Circuit files are evaluated with NumPy, and Boolean expression minimization uses PyEDA.

## MNIST and training

The logic network is initialized and trained from scratch. MNIST downloads automatically on the first online run through [torchvision's verified MNIST loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html). Cached archives and extracted IDX files live in `datasets/MNIST/raw`. To prepare data explicitly or choose a shared directory:

```bash
python gate.py prepare --root datasets
CUDA_VISIBLE_DEVICES=0 python gate.py train --dataset mnist --depth 4 --seed 42 --data-root datasets
```

The default experiment uses all 60,000 training images for 50 epochs, then measures the final checkpoint on the 10,000-image test partition. It uses 784 binary inputs, hidden width 512, ten final logic features, a 10→10 linear head, locality sigma 2, Adam at 0.001 with betas (0.9, 0.999), and batch size 64. Pixels greater than 0.5 become one. No augmentation is applied.

`configs/mnist_depth2.yaml`, `mnist_depth4.yaml`, and `mnist_depth8.yaml` select the depth studies. `--epochs`, `--seed`, `--config`, and `--output` select the run. `--offline` uses the existing MNIST files without network access:

```bash
python gate.py train --dataset mnist --depth 8 --seed 123 --data-root /data/mnist --offline
python gate.py train --dataset mnist --depth 4 --resume results/mnist/depth4/last.pth --epochs 100
```

For manual placement, torchvision's archive mirror is [ossci-datasets.s3.amazonaws.com/mnist](https://ossci-datasets.s3.amazonaws.com/mnist/). The four files are `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, and `t10k-labels-idx1-ubyte.gz`. `python gate.py prepare --root <directory>` downloads, checks, and extracts these into `<directory>/MNIST/raw`.

The archive filenames map to these direct download URLs. The preparation command places each extracted file beside its archive under `<root>/MNIST/raw`:

| Archive | Download |
| --- | --- |
| `train-images-idx3-ubyte.gz` | [Training images](https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz) |
| `train-labels-idx1-ubyte.gz` | [Training labels](https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz) |
| `t10k-images-idx3-ubyte.gz` | [Test images](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz) |
| `t10k-labels-idx1-ubyte.gz` | [Test labels](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz) |

Runtime preprocessing divides the uint8 grayscale pixels by 255, flattens each 28×28 image to 784 values, then applies `pixel > 0.5`. Labels remain integer digits 0–9. Dataset preparation preserves the standard split. It does not mix test images into training.

`last.pth` stores the fixed-epoch model, optimizer, RNG state, and configuration. `metrics.json` records final train/test scores, epoch histories, per-layer gate counts, operand overlap, long-range selections, and last-batch gradient norms. `data.validation_fraction` optionally reserves training examples for model selection. Its default of zero trains on the complete training split. NPZ input through `--data images.npz` supplies `x_train`, `y_train`, `x_val`, and `y_val` with normalized input arrays and integer class IDs.

For an existing normalized image array, the NPZ schema is:

```python
import numpy as np

np.savez_compressed("digits.npz", x_train=train_images.astype("float32"),
                    y_train=train_labels.astype("int64"),
                    x_val=validation_images.astype("float32"),
                    y_val=validation_labels.astype("int64"))
```

`train_images` and `validation_images` have shape `[N,28,28]` or `[N,784]` and values in [0,1]. Labels have shape `[N]`. Supply separately prepared train/validation partitions, then run `python gate.py train --dataset mnist --depth 4 --data digits.npz`. The loader applies the same threshold. `gate.py evaluate --data digits.npz` evaluates its `x_val`/`y_val` partition.

## Paper experiments

Run the depth, operand-gradient, and locality studies across seeds 42, 123, and 456:

```bash
python gate.py study --study depth --device cuda
python gate.py study --study operands --device cuda
python gate.py study --study locality --device cuda
```

The depth study uses 2/4/8 layers. The operand study compares trainable and detached operand paths at depth 1. The locality study compares Gaussian and random initialization at depth 4 with width 512. Each writes per-run checkpoints and a `summary.json` with measured mean and sample standard deviation over the selected seeds.

The broader 270-profile catalog in `configs/sweeps` groups files by depth and width, with two profiles for each depth beside the depth folders and two more beside its width folders. Each width keeps its random initialization profiles beside a `gaussian` folder containing the locality settings. Filenames record initialization and seed as `<initialization>__seed_<seed>.yaml` or `.py`, prefixed with depth and width when those values are not already in the directory path. It combines depths 2/3/4/5/6/8, widths 128/256/512, Gaussian sigma 0.5/1/2/4 or random initialization, and three seeds. Every axis changes the actual logic model:

```bash
python gate.py sweep --depths 2 4 8 --widths 512 \
  --initializations local_sigma_2 random --seeds 42 123 456 --device cuda
```

`--data-root`, `--offline`, and `--epochs` configure these runs. Every YAML or Python profile is directly accepted by `gate.py train --config`. Python profiles contain one literal dictionary assigned to `config` and are read with `ast.literal_eval`. Sweep selection and queued runs use the same loader. `python gate.py sweep --dry-run` reads configurations and reports their parameter counts without model execution.

## Evaluation, prediction, and circuit extraction

```bash
python gate.py evaluate --checkpoint results/mnist/depth4/last.pth --dataset mnist
python gate.py predict --checkpoint results/mnist/depth4/last.pth --images digit.png
python gate.py export --checkpoint results/mnist/depth4/last.pth --output results/mnist/circuit
```

Evaluation measures the held-out test set and compares network features with an independent exact NumPy circuit evaluator. Prediction resizes grayscale images to 28×28 and applies the saved threshold. `circuit.json` stores operand indices, gates, and the learned class head. `equations.txt` names each intermediate Boolean node once.

```python
import json
from networks.gates import evaluate_circuit

with open("results/mnist/circuit/circuit.json") as handle:
    circuit = json.load(handle)
features, logits = evaluate_circuit(circuit, binary_input_array)
predictions = logits.argmax(axis=-1)
```

For two-level Boolean minimization of the extracted features:

```bash
pip install pyeda
python gate.py minimize --circuit results/mnist/circuit/circuit.json --max-support 16
```

The compression command records each original expression and its Espresso result for features within the selected support budget, and preserves the saved linear head. Features above that budget retain their exact symbolic expression. `networks/gates.py` contains differentiable gates and exact circuit operations. `data/loaders.py` owns dataset splits. `experiments/wiring.py` owns saved configurations and the wiring catalog. The `gate.py` subcommands use those same definitions for training, studies, prediction, and export.

## Working with prepared data and saved runs

The source tree separates differentiable operators from circuit analysis, CUDA verification, array preparation, classification reports, study queues, and artifact inspection:

```text
networks/
  gates.py
  analysis.py
  test_gate.py
  evaluation/
    predictions.py
    equivalence.py
data/
  loaders.py
  partitions.py
experiments/
  wiring.py
  inspection.py
  runs.py
assets/
  00_false.json
  15_true.json
  logic/
configs/
  mnist_depth2.yaml
  mnist_depth4.yaml
  mnist_depth8.yaml
  boolean_depth1.yaml
  boolean_depth2.yaml
  sweeps/
    <depth>__<width>__<initialization>__seed_<seed>.{yaml,py}
    <depth>/
      <width>__<initialization>__seed_<seed>.{yaml,py}
      <width>/
        <initialization>__seed_<seed>.{yaml,py}
```

- [Prepare and inspect NPZ or local IDX data](docs/arrays.md)
- [Analyze circuits, extract feature cones, and verify on CUDA](docs/workflows/analysis.md)
- [Collect predictions and summarize classification errors](docs/workflows/predictions.md)
- [Plan resumable studies and aggregate measured seeds](docs/queues.md)
- [Inspect checkpoint tensors and verify saved artifacts](docs/workflows/checkpoints.md)

The existing training commands and checkpoint fields remain the same. Structural reports and file preparation read the supplied artifacts. Model inference, circuit verification, truth-table enumeration, checkpoint tensor analysis, and classification tensor metrics use CUDA.
