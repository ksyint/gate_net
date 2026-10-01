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

## Observed wiring and input behavior

The analysis commands below restore a checkpoint and use its binary input threshold. MNIST is downloaded automatically when required. Supply `--offline` to use prepared local data. `--data prepared/digits.npz` selects an NPZ partition and `--split val` selects its held-out rows. Every numerical model analysis uses CUDA.

```bash
python gate.py binary --checkpoint results/mnist/depth4/last.pth --pairs --output reports/input-bits.json
python gate.py activations --checkpoint results/mnist/depth4/last.pth --head-contributions --feature-groups --output reports/activations.json
python gate.py stability --checkpoints results/mnist/depth4/best.pth results/mnist/depth4/last.pth --output reports/wiring-movement.json
```

`binary` records class-conditional bit rates, entropy and bit-label mutual information. `--pairs` adds input correlations. For MNIST or continuous NPZ source arrays, `--thresholds 0.25 0.5 0.75` compares binary input density under different preprocessing thresholds.

`activations` records the binary units reached by the supplied examples. `--pair-width` controls the prefix used for pairwise activation counts. `--feature-groups` groups final features with identical or complementary values across those examples and combines their head coefficients. `--head-contributions` measures signed contributions and correct-class margins separately for each target class.

`stability` compares selected operands, gate IDs, selection-score margins and class-head movement. Gate changes also include the number of truth-table rows changed per unit. Its default reference is the preceding checkpoint. `--reference first` compares every checkpoint against the first supplied one.

## Perturbations and gradients

Use the same checkpoint, input partition and batch limit for paired circuit measurements.

| Command | Measurement |
| --- | --- |
| `robustness` | Binary corruption, image shifts and optional single-bit neighborhoods |
| `intervene` | Effects of forcing selected internal units to zero or one |
| `influence` | Prediction and accuracy changes from individual input-bit flips |
| `gradients` | Operand gradient propagation and temporary parameter steps |
| `benchmark` | CUDA event timings for model, layer and optional packed-circuit execution |

Each command exposes its own analysis settings through `--help`. `--max-batches` bounds the evaluated partition. Temporary interventions and gradient steps restore the original model tensors before returning.
