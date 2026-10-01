# Circuit structure and CUDA verification

Extract the discrete circuit from a saved model with the existing command:

```bash
python gate.py export --checkpoint results/mnist/depth4/last.pth \
  --device cuda --output results/mnist/circuit
```

Every layer stores its left operand, right operand, and gate ID. Gate IDs encode the output bits for inputs `00`, `01`, `10`, `11`, from most significant to least significant. The sixteen entries in `assets/logic/truth_tables` are read and checked by the circuit analyzer.

```bash
python gate.py circuit inspect --circuit results/mnist/circuit/circuit.json \
  --distance 100 --output reports/circuits/depth4.json
python gate.py circuit cone --circuit results/mnist/circuit/circuit.json \
  --feature 3 --output reports/circuits/feature3.json
```

Inspection validates every operand coordinate, gate identifier, layer width, head dimension, and coefficient. It reports gate counts, same-operand frequency, input distances, fanout, inactive units, and input support. The feature cone lists the Boolean nodes that feed one final feature and its weights in the class head.

Support propagation accounts for constant gates, unary gates, and repeated operands. It gives a conservative input support bound when expressions share dependent intermediate values. No expanded exponential expression is required for the structural report.

## Draw selected feature paths

```bash
python gate.py circuit graph --circuit results/mnist/circuit/circuit.json \
  --feature 3 --output reports/circuits/feature3.dot
```

The DOT file labels intermediate gates, operand edges, and the selected features' class-head weights. Dashed edges identify encoded operands that the chosen Boolean gate does not depend on. Repeat `--feature` to include more paths or omit it for the complete circuit. Render the file with a Graphviz installation when a diagram is needed. With selected features, head edges describe those features' contributions only.

## Remove unused internal units

```bash
python gate.py circuit compact --circuit results/mnist/circuit/circuit.json \
  --output results/mnist/circuit/compact.json
python gate.py verify compare --first results/mnist/circuit/circuit.json \
  --second results/mnist/circuit/compact.json --samples 4096 --device cuda \
  --output reports/circuits/compact-comparison.json
```

Compaction follows both saved operand edges from all final features, drops unreferenced internal units, and remaps their coordinates. It preserves final feature ordering and the learned class head. The adjacent `.mapping.json` records original unit indices for each retained layer. Constant and unary gates retain their encoded operand coordinates for format compatibility.

## Check the checkpoint and circuit together

```bash
python gate.py verify checkpoint --checkpoint results/mnist/depth4/last.pth \
  --circuit results/mnist/circuit/circuit.json --samples 4096 --seed 42 \
  --device cuda --output reports/circuits/checkpoint.json
```

The independent verifier uses integer truth-table operations on CUDA and compares its features, logits, and predicted classes with the restored network. It reports maximum and mean absolute logit error. A mismatch gives a failing exit status after writing the report. These comparisons use the selected inputs. Choose `--data prepared/digits.npz --split val --threshold 0.5` to use prepared images instead of seeded binary vectors.

To enumerate every assignment in one feature's propagated input support:

```bash
python gate.py verify truth-table --circuit results/mnist/circuit/circuit.json \
  --feature 3 --max-support 16 --batch-size 512 --device cuda \
  --output reports/circuits/feature3.npz
```

The NPZ contains original input coordinates, their binary assignments, and the selected feature output. Assignment column zero is the least significant bit in the row index. Other input coordinates are zero because they lie outside the selected feature's dependency cone. Enumeration runs in CUDA batches and writes the resulting arrays with metadata beside the archive.
