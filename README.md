# Negative selection with scarce normal training data

This repository contains the code and experimental evidence for a benchmark
of negative selection algorithms (NSA) and PyOD detectors when training data
contains only normal observations. The experiment asks a practical question:
what happens when only a small part of the normal training pool is available?

Repository: <https://github.com/kishordgupta/pyod-nsa-normal-only-benchmark>

Authors: Kishor Datta Gupta and Dipankar Dasgupta.

## What the experiment found

Dual NSA became more competitive as the normal training pool shrank after
PCA to two dimensions. Its mean AUROC rose from 0.717 at full retention to
0.760 at 25% retention. At that size, it ranked 11th among the 36 configurations
that completed every dataset/seed run within the measured time budget. KNN
scored 0.800. NSA therefore improved without establishing overall superiority.
The gain was not general on original features, and Dual NSA's average precision
was almost unchanged (0.472 to 0.470).

The table below gives the dataset-level result for **25% retained normal
training data, no training outliers, and PCA(2)**. “Top position” is simply
`rank / 36 × 100`; smaller is better. Each score is the mean across five split
seeds, and all datasets use the same field of 36 eligible configurations.

| Dataset | Dual NSA AUROC | Rank / 36 | Top position |
| --- | ---: | ---: | ---: |
| Arrhythmia | 0.7113 | 3 / 36 | 8.3% |
| Cardio | 0.9274 | 11 / 36 | 30.6% |
| Glass | 0.9122 | 3 / 36 | 8.3% |
| Ionosphere | 0.7433 | 8 / 36 | 22.2% |
| Letter | 0.5155 | 17 / 36 | 47.2% |
| Lympho | 0.7310 | 19 / 36 | 52.8% |
| MNIST | 0.6448 | 21 / 36 | 58.3% |
| Musk | 0.9937 | 12 / 36 | 33.3% |
| Optdigits | 0.2992 | 26 / 36 | 72.2% |
| Pendigits | 0.9045 | 16 / 36 | 44.4% |
| Pima | 0.6606 | 18 / 36 | 50.0% |
| Satellite | 0.8179 | 6 / 36 | 16.7% |
| Satimage-2 | 0.9664 | 9 / 36 | 25.0% |
| Shuttle | 0.9834 | 2 / 36 | 5.6% |
| Vertebral | 0.4327 | 13 / 36 | 36.1% |
| Vowels | 0.7654 | 7 / 36 | 19.4% |
| WBC | 0.9153 | 10 / 36 | 27.8% |

These descriptive ranks vary across datasets and seeds. They are not a claim
that NSA is consistently in a fixed top percentile. The main comparison field
contains 10 NSA configurations and 26 other PyOD detectors. The companion CSV
also includes the rank against the 26 other detectors plus Dual NSA and a
sensitivity calculation with each dataset's locally eligible field.

## Protocol and scope

The primary fresh sweep has **17 ODDS datasets × 5 seeds × 2 feature
representations × 4 retention fractions × 56 configurations = 38,080 records**.
It includes all 11 NSA strategy families separately (13 configurations because
the binary family has three matching rules) and all 43 other registered tabular
PyOD methods at the pinned source revision. “All” refers to that declared
tabular roster, not graph, image, time-series or multimodal detectors.

For each seed, a stratified 60/40 split precedes removal of every anomalous
training row. The original normal training pool is capped at 600 observations
and the mixed test set at 2,000. The 100%, 75%, 50% and 25% subsets are nested.
Scaling and optional PCA are fitted only to the retained normals. Every
detector receives `fit(X)` without training labels. Test anomalies and labels
are used only to calculate AUROC and average precision.

The primary analysis qualifies a successful record when its measured
construction/import/fit plus scoring time is at most 15 seconds. Native code
sometimes swallowed the Python alarm, so this qualification is applied to the
recorded elapsed times after execution. All raw successes and their scores
remain available in a sensitivity analysis. Six stalled LOCI processes were
terminated after at least 600 seconds without raw-record progress; snapshot
and recovery evidence is retained. Two process restarts are also documented.
Failure and inapplicability are never scored as zero AUROC. Complete-coverage
rankings exclude methods without all 85 qualified dataset/seed runs.

XGBOD and DevNet require labeled outliers for their intended supervised
training, so they are inapplicable under this rule. Other errors, timeouts and
dimension restrictions are reported separately. Some query scoring methods
depend on the test feature batch; that limitation is audited explicitly.

The experiment is a bounded default-configuration benchmark, not a tuned
comparison or a proof about every normal-only learning regime. Retention is
relative to the capped normal training pool, rather than the full dataset.

## Six-size extension: 10% and 5% normal training data

The fresh 10% and 5% settings add **19,040 primary records** to the frozen four-size benchmark. The complete six-size grid has **57,120 records**: 17 datasets, five seeds, two representations, six retained fractions, and 56 configurations (13 NSA configurations from 11 strategy families plus 43 other PyOD entries). Initial model fitting and preprocessing use retained normal rows only. The mixed held-out tests are fixed within each dataset and seed. Actual training samples contain **8–60 normals at 10%** and **4–30 at 5%**.

### Dual NSA's standing with the comparison roster held fixed

The primary fields contain other detectors that complete all 85 cases at every one of the six sizes: 19 native-feature baselines or 25 PCA(2) baselines. Each rank inserts only the focal NSA into that field. These are ranks among the stated complete-coverage detectors, rather than all 43 attempted entries.

| Features | Retained | Dual AUROC | Dual AP | Qualified cases | Fixed-field AUROC rank | Fixed-field AP rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 10% | 0.7792 | 0.5112 | 85/85 | 14/20 | 7/20 |
| Native | 5% | 0.7900 | 0.5066 | 85/85 | 10/20 | 4/20 |
| PCA(2) | 100% | 0.7172 | 0.4718 | 85/85 | 13/26 | 6/26 |
| PCA(2) | 25% | 0.7603 | 0.4699 | 85/85 | 10/26 | 3/26 |
| PCA(2) | 10% | 0.7626 | 0.4206 | 85/85 | 6/26 | 6/26 |
| PCA(2) | 5% | 0.7264 | 0.3755 | 85/85 | 5/26 | 5/26 |

All ten continuously scored, six-size-complete NSA configurations in PCA(2) improve their AUROC ranks at both 10% and 5% relative to full retention. At 5%, only three improve their absolute AUROC, and none improves AP. Dual NSA's PCA(2) AUROC falls between 10% and 5% even as its ordinal position improves. The six complete native NSA configurations show no corresponding AUROC rank gain; native Dual is complete at the new sizes but incomplete at 100–25%, so it has no full-coverage native cross-size rank comparison.

KDE has the highest fixed-field AUROC in all four new panels. The findings support conditional relative gains for some NSA strategies and representations. They do not establish broad superiority or consistency across all datasets. Dual's PCA(2) dataset ranks span 1/26 to 16/26 at 10%, and 1/26 to 24/26 at 5%.

### Suitability and a neural training-batch sensitivity

Every method is retained in the outcome tables. Failed or inapplicable cases are marked **unsuitable under the tested configuration**, with exact counts and reasons. Partial-case means remain available but receive no complete-coverage rank. The label covers the recorded constructor, dimensions, sample count, normal-only input, and measured 15-second budget; it does not establish family-wide incapability.

XGBOD and DevNet require labeled anomalies for their intended supervised training in this pinned roster. MAD requires one-dimensional input. Voronoi and deterministic NSA have low-dimensional restrictions, and binary NSA encounters negative-detector generation errors in PCA(2). Other partial failures and timeouts are shown individually.

AutoEncoder and VAE's fixed training batch of 32 with `drop_last=True` creates an empty training loop when fewer than 32 normals are available. A separate **adaptive training-batch sensitivity** has **740 fresh method attempts**, all successful within 15 seconds: all 10%/5% cells plus the affected historical 25% cells. Training batches are capped by the normal sample size; inference batches stay at 32. Successful fits have observed training-forward calls. The remaining 1,300 earlier AE/VAE records are reused with explicit lineage.

The correction adds VAE to the fixed PCA(2) baseline field. Dual's AUROC and AP ranks become **6/27 at 10%** and **5/27 at 5%**; all ten continuous NSA configurations retain their AUROC rank improvement. The native fixed field is unchanged. Both neural methods become operational in all 85 cases of each new panel, so their primary unsuitability reflects the fixed batch configuration.

### Evidence and interpretation limits

- [Full six-size report, including suitability and adaptive sensitivity](results/pyod_nsa_extreme_retention_report.md)
- [All detector scores, coverage, and suitability](results/pyod_nsa_extreme_retention_method_summary.csv)
- [Fixed comparison-field ranks](results/pyod_nsa_extreme_retention_common_field_ranks.csv)
- [Dataset-level NSA results and ranks](results/pyod_nsa_extreme_retention_dataset_nsa_ranks.csv)
- [Relative gaps with the same comparator held fixed](results/pyod_nsa_extreme_retention_relative_change_vs_100.csv)
- [Adaptive neural rank sensitivity](results/pyod_nsa_extreme_retention_adaptive_batch_fixed_rank_sensitivity.csv)

Raw records, sanitized logs, four new ROD watchdog recoveries, split/source audits, and extension provenance accompany these tables. The original four-size records remain byte-identical. AUROC/AP reporting and bootstrap intervals are reconstructed from aggregate records; per-observation prediction vectors were not saved. Twelve implementations use the mixed query batch while scoring, and an inductive subset is reported separately. Intervals are exploratory and unadjusted for multiple comparisons. Fixed hyperparameters, refitted preprocessing, computation limits, and small samples bound the conclusions.

## Find the evidence

| Path | Contents |
| --- | --- |
| [Full report](results/pyod_nsa_full_retention_report.md) | Main report, all-method summaries, paired comparisons and limitations |
| [Dataset ranks](results/pyod_nsa_dual_pca2_retain25_dataset_rank.csv) | Table above, per-seed ranks and comparison-field sensitivities |
| [Method summaries](results/pyod_nsa_full_retention_method_summary.csv) | AUROC, average precision, coverage and status counts for every setting |
| [Overall ranks](results/pyod_nsa_full_retention_coverage_qualified_ranks.csv) | Complete-coverage AUROC/AP rankings |
| [NSA versus other detectors](results/pyod_nsa_full_retention_nsa_vs_baseline.csv) | Matched comparisons and dataset-cluster bootstrap intervals |
| `benchmark/odds_oneclass/full_retention/raw/` | 680 primary JSONL files, 38,080 records |
| `benchmark/odds_oneclass/full_retention/logs/` | Public views of all primary process logs |
| `benchmark/odds_oneclass/full_retention/watchdog/` and `provenance/` | Snapshot and process-restart evidence |
| `benchmark/odds_oneclass/raw/`, `sensitivity/`, `retention/` | Earlier benchmark and retention experiments; historical, not additional primary runs |
| `benchmark/odds_oneclass/` | Runner, model constructors, sweep, analysis, plotting and packaging code |
| `vendor/pyod/` | Pinned PyOD source and tests, with upstream license |
| [Public copy manifest](provenance/public-copy-manifest.json) | Original/public file hashes and privacy changes |

![AUROC change when the normal training pool shrinks](results/pyod_nsa_full_retention_auroc_change.png)

## Reproduce from a clean clone

Use Python 3.12 on a POSIX system. The original execution used Python 3.12 on
macOS; elapsed times and occasional native-library failures are machine
dependent. Exact runtime versions are in `requirements.txt`, and the complete
observed environment is archived in `provenance/environment-requirements.txt`.
If a pinned wheel is unavailable for your platform, preserve the difference
in the new run manifest. The code uses POSIX alarms and does not provide the
same timeout behavior on Windows.

```bash
git clone https://github.com/kishordgupta/pyod-nsa-normal-only-benchmark.git
cd pyod-nsa-normal-only-benchmark
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e vendor/pyod
python scripts/fetch_datasets.py
python scripts/verify_public_tree.py
```

Verify a short execution before launching the full sweep:

```bash
python benchmark/odds_oneclass/runner.py \
  --data-dir vendor/pyod/notebooks/data --dataset glass --seed 42 \
  --representation pca2 --train-fraction 0.25 \
  --train-cap 600 --test-cap 2000 --method-timeout 15 \
  --method-filter nsa --results local_runs/smoke/glass_pca2_seed42_retain25.jsonl
```

Run the complete fresh grid into a new directory:

```bash
python benchmark/odds_oneclass/full_retention_sweep.py \
  --data-dir vendor/pyod/notebooks/data \
  --run-root local_runs/full_retention --workers 4
```

The sweep resumes missing records. It does not overwrite completed records.
Use a separate `--run-root` whenever changing the source, environment or
protocol. Keep the released `benchmark/.../raw` files as historical evidence.
The nominal alarm can be swallowed by native libraries; monitor long-running
cells. The original targeted recovery script is included as provenance but
requires a verified PID and exact preconditions before sending a signal.

Recompute the published full-grid analysis without fitting any model:

```bash
python benchmark/odds_oneclass/full_retention/analyze.py \
  --out-dir reanalysis/full_retention --bootstrap-reps 5000
python benchmark/odds_oneclass/full_retention/plot.py \
  --changes reanalysis/full_retention/full_retention_change_vs_full.csv \
  --output reanalysis/full_retention/full_retention_auroc_change.png
python benchmark/odds_oneclass/full_retention/dataset_rank_consistency.py
```

The last command regenerates the dataset-rank CSV/JSON in `results`. Plot
options are documented by `--help`. For a fresh run's analysis, the historical
watchdog and restart validation in the archived analyzer must be adapted to
the new run's observed interventions; do not copy old recovery events into a
new run. The archived analyzer is deliberately strict about the released
evidence and its exact 680-cell grid.

Public logs and metadata have host paths redacted. Original and public hashes
are recorded, and scientific fields are preserved. See
[publication privacy and portability notes](provenance/PORTABILITY_AND_PRIVACY.md)
and [third-party notices](THIRD_PARTY_NOTICES.md). Before publication, run
`python scripts/scan_public_tree.py`; it reports rule names and filenames,
never matched credential values.
