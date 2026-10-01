# Planned wiring studies

The existing `study` command runs the paper's depth, operand-gradient, and locality variants. `sweep` uses the 270 complete wiring profiles. The additional `runs` commands materialize a selected sweep as a persistent queue:

```bash
python gate.py runs plan --depths 2 4 8 --widths 512 \
  --initializations local_sigma_2 --seeds 42 123 456 --data-root datasets \
  --device cuda --output studies/depth
python gate.py runs run --plan studies/depth --dry-run
python gate.py runs run --plan studies/depth
python gate.py runs status --plan studies/depth
```

A plan contains complete configurations in each source profile's YAML or Python format, a hashed JSONL job manifest, a plan manifest, and distinct result directories. Data roots become absolute paths. `--data` records the prepared NPZ path and its SHA256. `--offline` is preserved in every job command.

The queue verifies configuration and NPZ hashes before invoking `gate.py train`. It records start and completion events, checks that the configured final epoch and `last.pth` exist, and skips completed jobs. Each result directory receives `planned-run.json` with the planned configuration identity and data hash.

```bash
python gate.py runs run --plan studies/depth --resume
```

Resume restores `last.pth` through the original training command. `--keep-going` allows the queue to continue after a failed subprocess. A `.runner.lock` prevents two queue processes from writing the same study. If the runner was terminated outside normal shutdown, inspect the process state and its event journal before removing that lock.

## Aggregate measured seeds

```bash
python gate.py runs aggregate --roots studies/depth/runs \
  --seeds 42 123 456 --output reports/studies/depth.json
```

Aggregation groups identical architecture, data, and training settings while separating seeds. It requires the requested seed set in every group and rejects duplicate seeds or incomplete runs. It reports measured means and sample standard deviations for final train/test metrics. When validation is configured, it also summarizes final validation metrics, peak validation accuracy, and the epoch with lowest validation loss.

The test results remain descriptive summaries. The validation-loss epoch identifies the same criterion used to save `best.pth` when a validation partition is configured. Original runs without validation retain their existing training-loss checkpoint rule.

NPZ runs require the queue's `planned-run.json` so aggregation can confirm their data identity. Standard MNIST run directories can also be supplied to `--roots` directly when they contain the existing `metrics.json` reports.
