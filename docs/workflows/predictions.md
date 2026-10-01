# Saved predictions and error analysis

Collect predictions with stable example indices and input hashes:

```bash
python gate.py predictions collect --checkpoint results/mnist/depth4/last.pth \
  --data-root datasets --split test --device cuda \
  --output reports/predictions/depth4.jsonl
```

The loader applies the saved normalization and binary threshold. CUDA inference writes the reference class, predicted class, logits, split, and binary input SHA256 for each example. A JSON sidecar records checkpoint and prediction file hashes, the configuration, and dataset selection. `--offline` uses only cached MNIST files. For a prepared NPZ use `--data prepared/digits.npz --split val`.

```bash
python gate.py predictions score --predictions reports/predictions/depth4.jsonl \
  --bins 15 --top-errors 30 --device cuda --output reports/predictions/depth4-metrics.json
```

When the collection sidecar is present, the reader verifies the prediction hash, sample count, partition, class count, and recorded accuracy before scoring or pairing. The scorer checks every saved class ID against the logit argmax, then computes the confusion matrix, per-class precision/recall/F1, accuracy, balanced accuracy, weighted F1, negative log likelihood, Brier score, and calibration bins on CUDA. Confusion rows are reference classes and columns are predicted classes. Macro metrics include the classes present among the references. Classes with no predicted examples receive zero precision.

The report also lists confidently wrong examples by their original ID and input hash. These IDs follow the deterministic dataset order used while collecting predictions.

## Compare two runs on identical examples

```bash
python gate.py predictions compare --first reports/predictions/depth4.jsonl \
  --second reports/predictions/depth8.jsonl --output reports/predictions/depth4-vs-depth8.json
```

Pairing requires identical example IDs, input hashes, reference labels, and class counts. The result includes examples correct only in either run, shared correct and incorrect counts, changed predictions, the paired accuracy difference, and the exact two-sided McNemar probability. It reads saved predictions and does not construct another model.
