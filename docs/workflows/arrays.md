# Prepared arrays and dataset inspection

The training loader accepts `x_train`, `y_train`, `x_val`, and `y_val` in one NPZ archive. Images contain finite normalized values in [0,1]. Labels are integer class IDs. The configured threshold is applied when the loader constructs binary features.

Inspect an existing archive before scheduling a study:

```bash
python gate.py dataset inspect --data digits.npz --input-dim 784 --classes 10 \
  --threshold 0.5 --output reports/data/digits.json
```

The report contains the file SHA256, shapes, label counts, binary density, duplicate thresholded inputs, contradictory labels on identical inputs, and overlap between partitions. Overlap counts describe identical binary feature vectors, including naturally repeated images. Inspection retains the supplied partitions.

## From image and label arrays

Save your image array and label array separately as ordinary NumPy `.npy` files. Use `[N,28,28]` or `[N,784]` for digit images and `[N]` for labels. The explicit scaling choice controls normalization:

```bash
python gate.py dataset pack --images raw/images.npy --labels raw/labels.npy \
  --scale uint8 --classes 10 --validation-fraction 0.1 --seed 42 \
  --threshold 0.5 --output prepared/digits.npz
```

Choose `--scale unit` for images already normalized into [0,1]. Preparation groups identical thresholded inputs together and assigns each group wholly to training or validation. Assignment is deterministic for the supplied seed and aims for the requested validation fraction within each class. Groups containing several labels use their most frequent label for stratification while retaining every original label. The inspection sidecar reports these groups.

`prepared/digits.npz.json` records source file paths and hashes, the split seed, class counts, and the resulting archive hash. Use the same threshold in the training configuration that you used while preparing the partition.

```bash
python gate.py train --dataset mnist --depth 4 --data prepared/digits.npz \
  --output results/custom/depth4 --device cuda
```

## From local MNIST files

The original `python gate.py prepare --root datasets` command handles automatic MNIST download. The IDX conversion command reads either extracted IDX files or their `.gz` archives from `<root>/MNIST/raw` or a raw directory:

```bash
python gate.py dataset idx --root datasets --validation-fraction 0.1 \
  --seed 42 --output prepared/mnist-validation.npz
```

It checks the IDX headers, dimensions, payload length, image size, label range, and normalization. It creates a grouped validation partition from the official training images and retains the official test images as `x_test` and `y_test` in the same archive.

The existing `train --data` and `evaluate --data` interfaces use `x_train` and `x_val`. Their held-out NPZ partition is `val`. The saved `x_test` and `y_test` remain available for explicit array processing and `verify --split test`. For the paper's full 60,000-image training protocol and official test evaluation, continue using the normal MNIST loader without `--data` and keep `data.validation_fraction: 0`.
