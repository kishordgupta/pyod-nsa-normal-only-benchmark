# AutoEncoder/VAE training-batch sensitivity

This is a separate, predeclared sensitivity to the primary 56-configuration
normal-only ODDS benchmark. It does not replace or edit any primary JSONL row,
model factory, or PyOD source file.

PyOD's `BaseDeepLearningDetector.fit` creates a training DataLoader with
`batch_size=32` and `drop_last=True`. For fewer than 32 normal fit rows, the
training loop has no batch and its later `loss` reference fails. All 60
AutoEncoder/VAE records in the frozen 25% cells for glass, lympho, and
vertebral have that `UnboundLocalError`. This is a batch-size implementation
limit, not a need for outlier labels.

For AutoEncoder and VAE only, the wrapper calls the unchanged primary
`runner.evaluate` and `model_factory.build_model`, then sets the constructed
estimator's **training** `batch_size` to `min(32, fit_normals)` before `fit`.
If that changes the value, a bound `decision_function` wrapper forwards the
original batch size 32 for PyOD's internal fit scoring and the held-out test
scoring. The wrapper counts calls to each model's `training_forward`; a
successful record requires the expected epoch-by-minibatch count. The normal
training split, scaler/PCA fit, X-only model fit, mixed held-out test, seeds,
features, model constructor, epochs, 15-second nominal signal timer, and
measured `fit_seconds + score_seconds <= 15.0` qualification rule are the
primary protocol. The wrapper and count instrumentation are recorded as a
separate intervention and contribute a small amount of measured runtime.

The `plan_manifest.json` fixes 370 cells and 740 new method records: both
methods in every 10% and 5% dataset/seed/representation cell (340 cells),
plus the 30 frozen 25% cells with fewer than 32 fit normals. From the frozen
normal-pool counts, 540 method rows need a reduced training batch (340 at 5%,
140 at 10%, and 60 at 25%); the other 200 10% rows keep 32. The six-size
sensitivity analysis reuses the 1,300 unchanged primary AutoEncoder/VAE rows
at 100%, 75%, 50%, and unaffected 25% cells, with source hashes.

Use Python 3.12 with packages matching `provenance/environment-requirements.txt`.
The following commands run from the public repository root and write
new sensitivity records under `local_runs/adaptive_batch/`:

```sh
python benchmark/odds_oneclass/extreme_retention/adaptive_batch/sweep.py
```

With the published primary grid intact, execute the sensitivity:

```sh
python benchmark/odds_oneclass/extreme_retention/adaptive_batch/sweep.py --execute --workers 8
```

Execution pins hashes of all 1,020 primary raw cells, both primary run
manifests, source files, data, and all three sensitivity scripts in its own
`run_manifest.json`. Each sensitivity cell has two append-only records under
`raw/` and one process log under `logs/`; rerunning the sweep resumes missing
methods after validating completed cells. The run manifest rejects a changed
primary grid or changed code.

After primary six-size analysis and sensitivity execution finish:

```sh
python benchmark/odds_oneclass/extreme_retention/adaptive_batch/analyze.py
```

The analysis writes a 740-row cell comparison, 24 AutoEncoder/VAE method
panels, 312 NSA AUROC/AP fixed-field rank sensitivity rows, and an audit with
all primary/sensitivity hashes and coverage. Its complete-coverage baseline
field is recalculated across all six sizes; an NSA receives a rank only when
its own 85 dataset/seed records qualify in that panel. Changes in rank
denominator are displayed, so a field change is not mistaken for a performance
gain. Partial-case means remain descriptive and are not used for fixed-field
ranking.
