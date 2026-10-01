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
