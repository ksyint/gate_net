# Checkpoint inspection and artifact inventories

Inspect a checkpoint directly on CUDA and optionally extract its selected circuit:

```bash
python gate.py artifacts inspect --checkpoint results/mnist/depth4/last.pth \
  --device cuda --extract reports/artifacts/depth4-circuit.json \
  --output reports/artifacts/depth4.json
```

Inspection validates model tensor names and shapes against the saved architecture, finite parameter values, optimizer membership, and epoch/history consistency. It reports parameter norms, selection margins and ties, RNG fields, gate structure, and the checkpoint SHA256. The extracted circuit uses the same format and hard selections as model export. This command reads the saved tensors without constructing or training another network.

```bash
python gate.py artifacts compare --first results/mnist/depth4/best.pth \
  --second results/mnist/depth4/last.pth --device cuda \
  --output reports/artifacts/best-vs-last.json
```

Comparison requires matching architecture settings. It reports tensor movement and changes to selected operands and operators. It keeps the existing checkpoint fields and names unchanged.

## Inventory a completed run

```bash
python gate.py artifacts inventory --directory results/mnist/depth4
python gate.py artifacts verify --directory results/mnist/depth4
```

The inventory records relative filenames, byte counts, and SHA256 hashes in `gate-artifacts.json`. Verification reports missing, added, or changed files and returns a failing status when membership or content differs. Save inspection reports outside the inventoried directory unless you want to include them before generating its inventory.

Record the inventory after finishing training, export, and report generation. Refresh it intentionally after adding new artifacts or continuing training. File hashing and inventory comparison operate on the saved files without running the model.

The training runner groups Adam parameters into `operands`, `operators` and `head`. The default rates and Adam options keep the ordinary training recipe. Optional `train.learning_rate_scales` assigns a multiplier to each group. `train.freeze` lists groups excluded from optimization. Set `train.monitor_interval` to control how often selector margins, discrete wiring changes and gradient norms are recorded in `wiring-history.json`.

A full training snapshot records optimizer parameter names, schedule position, observed wiring, shuffled-loader state and Python, NumPy, PyTorch and CUDA random states. The artifact inspector uses the saved parameter names to associate Adam tensors with their model tensors, including runs with frozen groups.

```bash
python gate.py train --config configs/mnist_depth4.yaml --resume results/mnist/depth4/last.pth --epochs 100
python gate.py history --runs results/mnist/depth4/metrics.json --output reports/training-history.json
```

The default constant learning rate permits extending the final epoch count while retaining its recorded optimizer position. A cosine schedule retains its original total epoch count on resume. Resuming at an already completed final epoch runs final evaluation and writes the completed metric report. Study queues count a run as complete after the final training and applicable test scores are present.

`best.pth` is written before the corresponding `last.pth` update. Both files use a temporary file followed by an atomic replacement. If only `best.pth` was written before interruption, a resumed study queue uses that complete training snapshot.
