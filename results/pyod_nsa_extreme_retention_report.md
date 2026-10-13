# Normal-only PyOD retention benchmark: six training sizes

**Private review draft.** This report combines the frozen 100/75/50/25% runs with fresh 10/5% runs. It describes the observed implementation and does not assert general NSA superiority.

## Executive measurements

The following measurements use the strict recorded-success, measured ≤15-second rule. A score marked available can be based on fewer than 85 runs; its coverage is shown, and it receives no full-field rank. The matched change columns compare each method with its own 100% run on the same dataset and seed.

### NSA/dual across all six sizes

| Features | % fit | Strict success | AUROC available | AP available | AUROC rank | AP rank | Paired ΔAUROC vs 100% | Paired ΔAP vs 100% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 100 | 81/85 | 0.7047 | 0.4609 | — | — | reference | reference |
| native | 75 | 80/85 | 0.7352 | 0.5006 | — | — | +0.0354 (80/85) | +0.0303 (80/85) |
| native | 50 | 81/85 | 0.7318 | 0.4966 | — | — | +0.0327 (80/85) | +0.0331 (80/85) |
| native | 25 | 83/85 | 0.7488 | 0.5142 | — | — | +0.0550 (80/85) | +0.0554 (80/85) |
| native | 10 | 85/85 | 0.7792 | 0.5112 | 14/20 | 7/20 | +0.0780 (81/85) | +0.0532 (81/85) |
| native | 5 | 85/85 | 0.7900 | 0.5066 | 10/20 | 4/20 | +0.0903 (81/85) | +0.0493 (81/85) |
| pca2 | 100 | 85/85 | 0.7172 | 0.4718 | 13/26 | 6/26 | reference | reference |
| pca2 | 75 | 85/85 | 0.7268 | 0.4777 | 12/26 | 6/26 | +0.0096 (85/85) | +0.0060 (85/85) |
| pca2 | 50 | 85/85 | 0.7419 | 0.4801 | 12/26 | 6/26 | +0.0247 (85/85) | +0.0083 (85/85) |
| pca2 | 25 | 85/85 | 0.7603 | 0.4699 | 10/26 | 3/26 | +0.0431 (85/85) | -0.0019 (85/85) |
| pca2 | 10 | 85/85 | 0.7626 | 0.4206 | 6/26 | 6/26 | +0.0454 (85/85) | -0.0511 (85/85) |
| pca2 | 5 | 85/85 | 0.7264 | 0.3755 | 5/26 | 5/26 | +0.0092 (85/85) | -0.0962 (85/85) |

### Best complete NSA and fixed-field other detectors at 10% and 5%

These leaders are selected by observed outcomes separately for AUROC and AP. Their paired gaps are exploratory and do not establish general superiority.

| Features | % fit | Metric | Top panel-complete NSA | Top fixed-field other | Matched | NSA − other | 95% CI | AUROC-win datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 10 | AUROC | NSA/dual (0.7792; panel only) | KDE (0.8781) | 85/85 | -0.0989 | [-0.169, -0.043] | 2/17 |
| native | 10 | AP | NSA/dual (0.5112; panel only) | KDE (0.5620) | 85/85 | -0.0508 | [-0.095, -0.007] | — |
| native | 5 | AUROC | NSA/dual (0.7900; panel only) | KDE (0.8577) | 85/85 | -0.0677 | [-0.115, -0.030] | 2/17 |
| native | 5 | AP | NSA/dual (0.5066; panel only) | KDE (0.5237) | 85/85 | -0.0171 | [-0.044, +0.016] | — |
| pca2 | 10 | AUROC | NSA/dual (0.7626; six-size) | KDE (0.7769) | 85/85 | -0.0143 | [-0.032, +0.004] | 4/17 |
| pca2 | 10 | AP | NSA/dual (0.4206; six-size) | KDE (0.4373) | 85/85 | -0.0167 | [-0.040, +0.003] | — |
| pca2 | 5 | AUROC | NSA/dual (0.7264; six-size) | KDE (0.7496) | 85/85 | -0.0232 | [-0.045, -0.002] | 6/17 |
| pca2 | 5 | AP | NSA/dual (0.3755; six-size) | KDE (0.3949) | 85/85 | -0.0194 | [-0.047, +0.005] | — |

### Change in NSA/dual's gap to the same comparator

Relative change is (Dual − comparator) at the reduced size minus (Dual − comparator) at 100%, using only dataset/seed cells where all four outcomes succeeded. A positive value means Dual gained relative to that comparator as normal training rows were removed; it does not necessarily mean Dual won at the reduced size. IForest, KNN, and the strongest fixed-field other method selected at 100% are held by name across sizes. Confidence intervals are descriptive and unadjusted; the full 2,236-row CSV includes every NSA–baseline pairing.

| Features | % fit | Metric | Fixed comparator | Reference | Four-success matches | Datasets | Relative change | 95% CI | Gain datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 10 | AUROC | IForest | IForest | 81/85 | 17 | +0.1270 | [+0.073, +0.181] | 16/17 |
| native | 10 | AUROC | KNN | KNN | 81/85 | 17 | +0.1175 | [+0.073, +0.163] | 15/17 |
| native | 10 | AUROC | LUNAR | top fixed other at 100% | 81/85 | 17 | +0.1106 | [+0.062, +0.164] | 15/17 |
| native | 5 | AUROC | IForest | IForest | 81/85 | 17 | +0.1723 | [+0.105, +0.236] | 15/17 |
| native | 5 | AUROC | KNN | KNN | 81/85 | 17 | +0.1759 | [+0.120, +0.230] | 16/17 |
| native | 5 | AUROC | LUNAR | top fixed other at 100% | 81/85 | 17 | +0.1586 | [+0.109, +0.210] | 16/17 |
| native | 10 | AP | IForest | IForest | 81/85 | 17 | +0.1777 | [+0.095, +0.274] | 15/17 |
| native | 10 | AP | KNN | KNN | 81/85 | 17 | +0.1315 | [+0.075, +0.196] | 14/17 |
| native | 10 | AP | LUNAR | top fixed other at 100% | 81/85 | 17 | +0.1705 | [+0.092, +0.254] | 14/17 |
| native | 5 | AP | IForest | IForest | 81/85 | 17 | +0.2202 | [+0.132, +0.322] | 15/17 |
| native | 5 | AP | KNN | KNN | 81/85 | 17 | +0.1721 | [+0.099, +0.258] | 16/17 |
| native | 5 | AP | LUNAR | top fixed other at 100% | 81/85 | 17 | +0.2198 | [+0.128, +0.325] | 15/17 |
| pca2 | 10 | AUROC | IForest | IForest | 85/85 | 17 | +0.0851 | [+0.035, +0.146] | 14/17 |
| pca2 | 10 | AUROC | KNN | KNN, top fixed other at 100% | 85/85 | 17 | +0.0707 | [+0.013, +0.136] | 11/17 |
| pca2 | 5 | AUROC | IForest | IForest | 85/85 | 17 | +0.0815 | [+0.010, +0.164] | 10/17 |
| pca2 | 5 | AUROC | KNN | KNN, top fixed other at 100% | 85/85 | 17 | +0.0968 | [-0.001, +0.197] | 10/17 |
| pca2 | 10 | AP | IForest | IForest | 85/85 | 17 | +0.0455 | [-0.009, +0.109] | 11/17 |
| pca2 | 10 | AP | KNN | KNN, top fixed other at 100% | 85/85 | 17 | +0.0129 | [-0.031, +0.057] | 10/17 |
| pca2 | 5 | AP | IForest | IForest | 85/85 | 17 | +0.0252 | [-0.041, +0.090] | 9/17 |
| pca2 | 5 | AP | KNN | KNN, top fixed other at 100% | 85/85 | 17 | +0.0193 | [-0.047, +0.077] | 9/17 |

### NSA increases and decreases relative to 100%

Only configurations with all 85 matched dataset/seed pairs enter the increase/decrease counts. Detailed scores, coverage, and intervals for every NSA are below and in the CSV. A positive within-method change does not imply a win over another detector.

| Features | % fit | Metric | Fully matched NSA | Increase | Decrease | Tie | Largest increase | Largest decrease |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 75 | AUROC | 7/13 | 1 | 6 | 0 | NSA/hierarchical +0.0110 | NSA/suppressed -0.0228 |
| native | 75 | AP | 7/13 | 0 | 7 | 0 | — | NSA/suppressed -0.0277 |
| native | 50 | AUROC | 7/13 | 0 | 7 | 0 | — | NSA/grid -0.0403 |
| native | 50 | AP | 7/13 | 0 | 7 | 0 | — | NSA/coverage -0.0585 |
| native | 25 | AUROC | 6/13 | 0 | 6 | 0 | — | NSA/suppressed -0.0964 |
| native | 25 | AP | 6/13 | 0 | 6 | 0 | — | NSA/suppressed -0.0993 |
| native | 10 | AUROC | 7/13 | 0 | 7 | 0 | — | NSA/grid -0.1649 |
| native | 10 | AP | 7/13 | 0 | 7 | 0 | — | NSA/grid -0.1452 |
| native | 5 | AUROC | 7/13 | 1 | 6 | 0 | NSA/hierarchical +0.0121 | NSA/grid -0.2150 |
| native | 5 | AP | 7/13 | 1 | 6 | 0 | NSA/hierarchical +0.0019 | NSA/grid -0.1609 |
| pca2 | 75 | AUROC | 10/13 | 9 | 1 | 0 | NSA/annealed +0.0185 | NSA/hierarchical -0.0078 |
| pca2 | 75 | AP | 10/13 | 6 | 4 | 0 | NSA/suppressed +0.0075 | NSA/hierarchical -0.0128 |
| pca2 | 50 | AUROC | 10/13 | 10 | 0 | 0 | NSA/deterministic +0.0428 | — |
| pca2 | 50 | AP | 10/13 | 10 | 0 | 0 | NSA/deterministic +0.0334 | — |
| pca2 | 25 | AUROC | 10/13 | 10 | 0 | 0 | NSA/fixed +0.0473 | — |
| pca2 | 25 | AP | 10/13 | 1 | 9 | 0 | NSA/fixed +0.0047 | NSA/hierarchical -0.0361 |
| pca2 | 10 | AUROC | 10/13 | 10 | 0 | 0 | NSA/dual +0.0454 | — |
| pca2 | 10 | AP | 10/13 | 0 | 10 | 0 | — | NSA/hierarchical -0.0902 |
| pca2 | 5 | AUROC | 10/13 | 3 | 7 | 0 | NSA/dual +0.0092 | NSA/variable -0.0245 |
| pca2 | 5 | AP | 10/13 | 0 | 10 | 0 | — | NSA/hierarchical -0.1571 |

## Scope and controls

The exact grid contains **57,120 records** in **1,020 dataset/seed/representation/size cells**: 17 ODDS datasets × five seeds (42–46) × native or PCA(2) features × six retained fractions (100, 75, 50, 25, 10, 5%) × 56 fixed configurations (13 NSA configurations from 11 strategy families and 43 other PyOD entries). Dataset labels stratify the split and select normal training rows; anomalies in the original training partition are discarded. Benchmark preprocessing and each initial detector.fit call receive retained normal features only, without a target label. A separate mixed held-out test set uses labels to calculate AUROC and average precision (AP). The same seed-fixed normal parent pool, capped at 600, supplies nested subsets at all six sizes; an implementation minimum of two rows applies. The smallest observed 10/5% fit pool has **4 normal rows**. Preprocessing and PCA are fitted again on each retained normal subset, so representation drift is part of the measured pipeline.

A separate independent audit reconstructed all 340 new cells and confirmed every initial fit pool contains only normal rows; its SHA-256 is `7b496f19f0c866f8f536c0afe70fa5e38affe594266a1c11cbdffe6f4a2933a8`. Raw JSONL retains aggregate AUROC/AP but not per-test score vectors, so that audit cannot recompute those metric values independently.

## Actual source and fit counts

| Dataset | Rows | Features | Normal | Anomaly | 100% fit | 75% fit | 50% fit | 25% fit | 10% fit | 5% fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 452 | 274 | 386 | 66 | 231 | 173 | 115 | 57 | 23 | 11 |
| cardio | 1831 | 21 | 1655 | 176 | 600 | 450 | 300 | 150 | 60 | 30 |
| glass | 214 | 9 | 205 | 9 | 123 | 92 | 61 | 30 | 12 | 6 |
| ionosphere | 351 | 33 | 225 | 126 | 135 | 101 | 67 | 33 | 13 | 6 |
| letter | 1600 | 32 | 1500 | 100 | 600 | 450 | 300 | 150 | 60 | 30 |
| lympho | 148 | 18 | 142 | 6 | 84 | 63 | 42 | 21 | 8 | 4 |
| mnist | 7603 | 100 | 6903 | 700 | 600 | 450 | 300 | 150 | 60 | 30 |
| musk | 3062 | 166 | 2965 | 97 | 600 | 450 | 300 | 150 | 60 | 30 |
| optdigits | 5216 | 64 | 5066 | 150 | 600 | 450 | 300 | 150 | 60 | 30 |
| pendigits | 6870 | 16 | 6714 | 156 | 600 | 450 | 300 | 150 | 60 | 30 |
| pima | 768 | 8 | 500 | 268 | 299 | 224 | 149 | 74 | 29 | 14 |
| satellite | 6435 | 36 | 4399 | 2036 | 600 | 450 | 300 | 150 | 60 | 30 |
| satimage-2 | 5803 | 36 | 5732 | 71 | 600 | 450 | 300 | 150 | 60 | 30 |
| shuttle | 49097 | 9 | 45586 | 3511 | 600 | 450 | 300 | 150 | 60 | 30 |
| vertebral | 240 | 6 | 210 | 30 | 126 | 94 | 63 | 31 | 12 | 6 |
| vowels | 1456 | 12 | 1406 | 50 | 600 | 450 | 300 | 150 | 60 | 30 |
| wbc | 378 | 30 | 357 | 21 | 213 | 159 | 106 | 53 | 21 | 10 |

## Time qualification and coverage

Raw successes: **48,275**; measured ≤15-second qualified successes: **48,131**; successful scores beyond that measured duration: **144** (historical 118, new 26). Scores beyond 15 seconds remain valid normal-only outcomes in the raw sensitivity table but do not enter the strict-budget ranks, means, or paired comparisons. The Python alarm is nominal: native callbacks may swallow it. Measured fit plus score time includes optional Torch setup before the alarm starts. Failures, timeouts, inapplicability, partial AUROC on a timeout, and incomplete seed coverage are never scored as zero.

The six-size field requires **85/85 qualified dataset/seed outcomes at every size** in a representation. A baseline-only field contains the qualifying other detectors; each NSA is inserted individually when it has 85/85 success in that panel. Its focal six-size eligibility is flagged separately; cross-size rank changes require that eligibility. A separate inductive sensitivity removes all 12 audited batch-dependent scoring methods. Panel-specific fields are shown only as sensitivity because their membership changes with size.

| Features | All common | NSA common | Other common | Inductive common | Inductive other | Raw common | Raw NSA common |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 25 | 6 | 19 | 24 | 18 | 27 | 6 |
| pca2 | 35 | 10 | 25 | 30 | 20 | 36 | 10 |

## All configurations at 10% and 5%

Every method is shown, including failed, timed-out, inapplicable, and partially covered configurations. Scores marked available use successful dataset/seed cells only and should not be compared across differing coverage. A six-size all-method rank requires 85/85 qualified outcomes at every size. 'Unsuitable in X/85 tested cases' means X cases did not produce a measured ≤15-second score with these fit sizes, dimensions, constructor settings, and runtime. Zero qualified scores are labeled 'unsuitable under tested configuration.' This does not establish that the detector family cannot train on normal data in another configuration. The cause column counts the derived non-success statuses: labeled-class requirement, dimension mismatch, model construction or fit/scoring error, reported timeout, or measured overbudget success. Partially covered available-case means receive no coverage-qualified rank. Raw status counts and scores are preserved in the JSONL and global audit.

### Suitability coverage overview

| Features | Retained % | 85/85 methods | 1–84/85 methods | 0/85 methods | Methods with 0/85 qualified scores |
| --- | --- | --- | --- | --- | --- |
| native | 10 | 36 | 15 | 5 | NSA/voronoi, NSA/deterministic, MAD, XGBOD, DevNet |
| native | 5 | 37 | 12 | 7 | NSA/voronoi, NSA/deterministic, MAD, XGBOD, AutoEncoder, VAE, DevNet |
| pca2 | 10 | 40 | 10 | 6 | NSA/binary/hamming, NSA/binary/rcontiguous, NSA/binary/rchunk, MAD, XGBOD, DevNet |
| pca2 | 5 | 43 | 5 | 8 | NSA/binary/hamming, NSA/binary/rcontiguous, NSA/binary/rchunk, MAD, XGBOD, AutoEncoder, VAE, DevNet |

### Native features, 10% fit

| Method | Family | ≤15 s success | Raw success | Suitability under tested configuration | AUROC available | AP available | Six-size all rank | Non-success causes and counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.2982 | 0.1093 | 23/25 | none |
| NSA/variable | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4350 | 0.1620 | 21/25 | none |
| NSA/coverage | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4333 | 0.1620 | 22/25 | none |
| NSA/grid | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4323 | 0.1573 | — | none |
| NSA/hierarchical | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.2452 | 0.1031 | 25/25 | none |
| NSA/voronoi | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | two- or three-dimensional input required: 85 |
| NSA/deterministic | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | two-dimensional input required: 85 |
| NSA/suppressed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4366 | 0.1614 | 20/25 | none |
| NSA/dual | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7792 | 0.5112 | — | none |
| NSA/annealed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.2641 | 0.1040 | 24/25 | none |
| NSA/binary/hamming | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5024 | 0.1361 | — | none |
| NSA/binary/rcontiguous | NSA | 75/85 | 75/85 | unsuitable in 10/85 tested cases | 0.5317 | 0.1321 | — | fit or scoring error: 10 |
| NSA/binary/rchunk | NSA | 78/85 | 78/85 | unsuitable in 7/85 tested cases | 0.5350 | 0.1290 | — | fit or scoring error: 7 |
| ECOD | baseline | 77/85 | 85/85 | unsuitable in 8/85 tested cases | 0.7729 | 0.4329 | — | measured success beyond 15 seconds: 8 |
| ABOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8536 | 0.5352 | 5/25 | none |
| COPOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7776 | 0.4474 | 14/25 | none |
| MAD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | one-dimensional input required: 85 |
| SOS | baseline | 84/85 | 84/85 | unsuitable in 1/85 tested cases | 0.5916 | 0.2208 | — | fit or scoring error: 1 |
| QMCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4502 | 0.1812 | — | none |
| KDE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8781 | 0.5620 | 1/25 | none |
| Sampling | baseline | 65/85 | 65/85 | unsuitable in 20/85 tested cases | 0.8732 | 0.5472 | — | fit or scoring error: 20 |
| GMM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8420 | 0.5067 | 6/25 | none |
| PCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8061 | 0.5068 | — | none |
| KPCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8766 | 0.5579 | 2/25 | none |
| MCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8178 | 0.4461 | 9/25 | none |
| CD | baseline | 78/85 | 78/85 | unsuitable in 7/85 tested cases | 0.8216 | 0.4776 | — | fit or scoring error: 6; reported runtime timeout: 1 |
| OCSVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8407 | 0.5353 | 7/25 | none |
| LMDD | baseline | 52/85 | 52/85 | unsuitable in 33/85 tested cases | 0.7443 | 0.4517 | — | reported runtime timeout: 33 |
| LOF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7931 | 0.5063 | 13/25 | none |
| COF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6657 | 0.3202 | — | none |
| CBLOF | baseline | 80/85 | 80/85 | unsuitable in 5/85 tested cases | 0.8580 | 0.5274 | — | fit or scoring error: 5 |
| LOCI | baseline | 29/85 | 29/85 | unsuitable in 56/85 tested cases | 0.7087 | 0.3674 | — | reported runtime timeout: 56 |
| HBOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8043 | 0.4543 | — | none |
| HDBSCAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5446 | 0.1398 | 17/25 | none |
| KNN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8558 | 0.5347 | 4/25 | none |
| SOD | baseline | 58/85 | 67/85 | unsuitable in 27/85 tested cases | 0.7325 | 0.3226 | — | model construction error: 5; measured success beyond 15 seconds: 9; reported runtime timeout: 13 |
| ROD | baseline | 39/85 | 40/85 | unsuitable in 46/85 tested cases | 0.8054 | 0.4483 | — | measured success beyond 15 seconds: 1; reported runtime timeout: 45 |
| IForest | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8021 | 0.3957 | 11/25 | none |
| INNE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8379 | 0.4534 | 8/25 | none |
| DIF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7636 | 0.3574 | — | none |
| FeatureBagging | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8002 | 0.5099 | 12/25 | none |
| LSCP | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.8724 | 0.5188 | — | fit or scoring error: 35 |
| LODA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6620 | 0.1956 | 15/25 | none |
| SUOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8190 | 0.4654 | — | none |
| XGBOD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| AutoEncoder | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.8347 | 0.5080 | — | fit or scoring error: 35 |
| VAE | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.8194 | 0.4915 | — | fit or scoring error: 35 |
| SO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4595 | 0.2057 | 19/25 | none |
| MO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4879 | 0.2013 | 18/25 | none |
| DeepSVDD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8089 | 0.5069 | 10/25 | none |
| AnoGAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7961 | 0.4628 | — | none |
| ALAD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5461 | 0.1544 | 16/25 | none |
| AE1SVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8143 | 0.5101 | — | none |
| DevNet | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| RGraph | baseline | 35/85 | 35/85 | unsuitable in 50/85 tested cases | 0.5518 | 0.2457 | — | reported runtime timeout: 50 |
| LUNAR | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8703 | 0.5564 | 3/25 | none |

### Native features, 5% fit

| Method | Family | ≤15 s success | Raw success | Suitability under tested configuration | AUROC available | AP available | Six-size all rank | Non-success causes and counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.2951 | 0.1173 | 24/25 | none |
| NSA/variable | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3844 | 0.1509 | 20/25 | none |
| NSA/coverage | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3819 | 0.1463 | 21/25 | none |
| NSA/grid | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3823 | 0.1417 | — | none |
| NSA/hierarchical | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3089 | 0.1193 | 23/25 | none |
| NSA/voronoi | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | two- or three-dimensional input required: 85 |
| NSA/deterministic | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | two-dimensional input required: 85 |
| NSA/suppressed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3803 | 0.1483 | 22/25 | none |
| NSA/dual | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7900 | 0.5066 | — | none |
| NSA/annealed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.2564 | 0.0924 | 25/25 | none |
| NSA/binary/hamming | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5036 | 0.1234 | — | none |
| NSA/binary/rcontiguous | NSA | 75/85 | 75/85 | unsuitable in 10/85 tested cases | 0.5437 | 0.1339 | — | fit or scoring error: 10 |
| NSA/binary/rchunk | NSA | 84/85 | 84/85 | unsuitable in 1/85 tested cases | 0.5223 | 0.1186 | — | fit or scoring error: 1 |
| ECOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7686 | 0.4268 | — | none |
| ABOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8227 | 0.4993 | 6/25 | none |
| COPOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7762 | 0.4436 | 12/25 | none |
| MAD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | one-dimensional input required: 85 |
| SOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5928 | 0.2208 | — | none |
| QMCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3821 | 0.1485 | — | none |
| KDE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8577 | 0.5237 | 1/25 | none |
| Sampling | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.8850 | 0.5191 | — | fit or scoring error: 35 |
| GMM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8244 | 0.4648 | 5/25 | none |
| PCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7907 | 0.4861 | — | none |
| KPCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8544 | 0.5214 | 2/25 | none |
| MCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7862 | 0.4051 | 10/25 | none |
| CD | baseline | 79/85 | 79/85 | unsuitable in 6/85 tested cases | 0.8162 | 0.4575 | — | fit or scoring error: 6 |
| OCSVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8258 | 0.5112 | 4/25 | none |
| LMDD | baseline | 55/85 | 55/85 | unsuitable in 30/85 tested cases | 0.7284 | 0.4163 | — | reported runtime timeout: 30 |
| LOF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7739 | 0.4874 | 13/25 | none |
| COF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6500 | 0.2984 | — | none |
| CBLOF | baseline | 65/85 | 65/85 | unsuitable in 20/85 tested cases | 0.8652 | 0.5216 | — | fit or scoring error: 20 |
| LOCI | baseline | 30/85 | 30/85 | unsuitable in 55/85 tested cases | 0.6819 | 0.3476 | — | reported runtime timeout: 55 |
| HBOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7848 | 0.4265 | — | none |
| HDBSCAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5303 | 0.1303 | 16/25 | none |
| KNN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8104 | 0.4913 | 7/25 | none |
| SOD | baseline | 55/85 | 55/85 | unsuitable in 30/85 tested cases | 0.7238 | 0.2796 | — | model construction error: 30 |
| ROD | baseline | 42/85 | 43/85 | unsuitable in 43/85 tested cases | 0.7826 | 0.4245 | — | fit or scoring error: 1; measured success beyond 15 seconds: 1; reported runtime timeout: 41 |
| IForest | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7696 | 0.3516 | 14/25 | none |
| INNE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7993 | 0.3844 | 8/25 | none |
| DIF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7334 | 0.3109 | — | none |
| FeatureBagging | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7809 | 0.4873 | 11/25 | none |
| LSCP | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.8487 | 0.4971 | — | fit or scoring error: 35 |
| LODA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4868 | 0.1312 | 19/25 | none |
| SUOD | baseline | 84/85 | 84/85 | unsuitable in 1/85 tested cases | 0.7981 | 0.4334 | — | fit or scoring error: 1 |
| XGBOD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| AutoEncoder | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| VAE | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| SO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4980 | 0.2088 | 17/25 | none |
| MO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.4889 | 0.2088 | 18/25 | none |
| DeepSVDD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7927 | 0.4683 | 9/25 | none |
| AnoGAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7894 | 0.4754 | — | none |
| ALAD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5458 | 0.1551 | 15/25 | none |
| AE1SVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7921 | 0.4924 | — | none |
| DevNet | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| RGraph | baseline | 54/85 | 54/85 | unsuitable in 31/85 tested cases | 0.5585 | 0.2186 | — | reported runtime timeout: 31 |
| LUNAR | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.8343 | 0.5021 | 3/25 | none |

### PCA(2), 10% fit

| Method | Family | ≤15 s success | Raw success | Suitability under tested configuration | AUROC available | AP available | Six-size all rank | Non-success causes and counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6132 | 0.2789 | 28/35 | none |
| NSA/variable | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7273 | 0.3638 | 10/35 | none |
| NSA/coverage | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7110 | 0.3571 | 14/35 | none |
| NSA/grid | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7243 | 0.3668 | 11/35 | none |
| NSA/hierarchical | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6891 | 0.3316 | 18/35 | none |
| NSA/voronoi | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7361 | 0.3689 | 9/35 | none |
| NSA/deterministic | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6278 | 0.2893 | 24/35 | none |
| NSA/suppressed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7173 | 0.3547 | 13/35 | none |
| NSA/dual | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7626 | 0.4206 | 6/35 | none |
| NSA/annealed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6241 | 0.2869 | 25/35 | none |
| NSA/binary/hamming | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| NSA/binary/rcontiguous | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| NSA/binary/rchunk | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| ECOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6162 | 0.2412 | 26/35 | none |
| ABOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7751 | 0.4221 | 3/35 | none |
| COPOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6611 | 0.2884 | 21/35 | none |
| MAD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | one-dimensional input required: 85 |
| SOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5252 | 0.1472 | 34/35 | none |
| QMCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6058 | 0.2582 | 29/35 | none |
| KDE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7769 | 0.4373 | 1/35 | none |
| Sampling | baseline | 65/85 | 65/85 | unsuitable in 20/85 tested cases | 0.7742 | 0.4442 | — | fit or scoring error: 20 |
| GMM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6418 | 0.3487 | 23/35 | none |
| PCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6504 | 0.3413 | 22/35 | none |
| KPCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7764 | 0.4290 | 2/35 | none |
| MCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7053 | 0.3462 | 15/35 | none |
| CD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5877 | 0.2623 | 31/35 | none |
| OCSVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7663 | 0.4295 | 5/35 | none |
| LMDD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5874 | 0.2547 | — | none |
| LOF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6729 | 0.3576 | 19/35 | none |
| COF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6139 | 0.2165 | 27/35 | none |
| CBLOF | baseline | 80/85 | 80/85 | unsuitable in 5/85 tested cases | 0.7665 | 0.4357 | — | fit or scoring error: 5 |
| LOCI | baseline | 28/85 | 28/85 | unsuitable in 57/85 tested cases | 0.5645 | 0.2412 | — | reported runtime timeout: 57 |
| HBOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7331 | 0.2969 | — | none |
| HDBSCAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5928 | 0.1505 | 30/35 | none |
| KNN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7706 | 0.4232 | 4/35 | none |
| SOD | baseline | 63/85 | 68/85 | unsuitable in 22/85 tested cases | 0.5624 | 0.2075 | — | model construction error: 5; measured success beyond 15 seconds: 5; reported runtime timeout: 12 |
| ROD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6634 | 0.2968 | — | none |
| IForest | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7196 | 0.3056 | 12/35 | none |
| INNE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7452 | 0.4062 | 8/35 | none |
| DIF | baseline | 83/85 | 85/85 | unsuitable in 2/85 tested cases | 0.7250 | 0.3661 | — | measured success beyond 15 seconds: 2 |
| FeatureBagging | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6948 | 0.3703 | 17/35 | none |
| LSCP | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.7968 | 0.4374 | — | fit or scoring error: 35 |
| LODA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6645 | 0.2562 | 20/35 | none |
| SUOD | baseline | 84/85 | 84/85 | unsuitable in 1/85 tested cases | 0.7402 | 0.3490 | — | fit or scoring error: 1 |
| XGBOD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| AutoEncoder | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.6253 | 0.3012 | — | fit or scoring error: 35 |
| VAE | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.6255 | 0.3211 | — | fit or scoring error: 35 |
| SO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5688 | 0.2370 | 32/35 | none |
| MO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5610 | 0.2541 | 33/35 | none |
| DeepSVDD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7010 | 0.3433 | 16/35 | none |
| AnoGAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6425 | 0.2731 | — | none |
| ALAD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5045 | 0.1276 | 35/35 | none |
| AE1SVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6548 | 0.3471 | — | none |
| DevNet | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| RGraph | baseline | 53/85 | 53/85 | unsuitable in 32/85 tested cases | 0.4038 | 0.1112 | — | reported runtime timeout: 32 |
| LUNAR | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7459 | 0.4038 | 7/35 | none |

### PCA(2), 5% fit

| Method | Family | ≤15 s success | Raw success | Suitability under tested configuration | AUROC available | AP available | Six-size all rank | Non-success causes and counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5917 | 0.2200 | 28/35 | none |
| NSA/variable | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6683 | 0.2879 | 13/35 | none |
| NSA/coverage | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6669 | 0.2838 | 14/35 | none |
| NSA/grid | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6775 | 0.3002 | 12/35 | none |
| NSA/hierarchical | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6579 | 0.2647 | 18/35 | none |
| NSA/voronoi | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6882 | 0.3153 | 9/35 | none |
| NSA/deterministic | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6066 | 0.2198 | 24/35 | none |
| NSA/suppressed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6650 | 0.2955 | 17/35 | none |
| NSA/dual | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7264 | 0.3755 | 5/35 | none |
| NSA/annealed | NSA | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5966 | 0.2187 | 27/35 | none |
| NSA/binary/hamming | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| NSA/binary/rcontiguous | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| NSA/binary/rchunk | NSA | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| ECOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6172 | 0.2507 | 23/35 | none |
| ABOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7437 | 0.3791 | 4/35 | none |
| COPOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6529 | 0.2899 | 19/35 | none |
| MAD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | one-dimensional input required: 85 |
| SOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5295 | 0.1529 | 33/35 | none |
| QMCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5224 | 0.2009 | 34/35 | none |
| KDE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7496 | 0.3949 | 1/35 | none |
| Sampling | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.7676 | 0.3969 | — | fit or scoring error: 35 |
| GMM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6403 | 0.3479 | 20/35 | none |
| PCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6657 | 0.3562 | 16/35 | none |
| KPCA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7440 | 0.3837 | 3/35 | none |
| MCD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6805 | 0.3242 | 11/35 | none |
| CD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5975 | 0.2643 | 26/35 | none |
| OCSVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7445 | 0.3855 | 2/35 | none |
| LMDD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5800 | 0.2384 | — | none |
| LOF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6207 | 0.3030 | 22/35 | none |
| COF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6023 | 0.2040 | 25/35 | none |
| CBLOF | baseline | 65/85 | 65/85 | unsuitable in 20/85 tested cases | 0.7703 | 0.4210 | — | fit or scoring error: 20 |
| LOCI | baseline | 30/85 | 30/85 | unsuitable in 55/85 tested cases | 0.5263 | 0.2354 | — | reported runtime timeout: 55 |
| HBOS | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6720 | 0.2230 | — | none |
| HDBSCAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5650 | 0.1351 | 30/35 | none |
| KNN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7082 | 0.3717 | 8/35 | none |
| SOD | baseline | 55/85 | 55/85 | unsuitable in 30/85 tested cases | 0.5845 | 0.2026 | — | model construction error: 30 |
| ROD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6248 | 0.2797 | — | none |
| IForest | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6870 | 0.2809 | 10/35 | none |
| INNE | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7139 | 0.3470 | 6/35 | none |
| DIF | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6941 | 0.3325 | — | none |
| FeatureBagging | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6355 | 0.3239 | 21/35 | none |
| LSCP | baseline | 50/85 | 50/85 | unsuitable in 35/85 tested cases | 0.7475 | 0.3439 | — | fit or scoring error: 35 |
| LODA | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5771 | 0.1873 | 29/35 | none |
| SUOD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6915 | 0.2966 | — | none |
| XGBOD | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| AutoEncoder | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| VAE | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | fit or scoring error: 85 |
| SO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5571 | 0.2157 | 31/35 | none |
| MO_GAAL | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5434 | 0.2227 | 32/35 | none |
| DeepSVDD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6667 | 0.3067 | 15/35 | none |
| AnoGAN | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6493 | 0.2950 | — | none |
| ALAD | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.5133 | 0.1373 | 35/35 | none |
| AE1SVM | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.6558 | 0.3537 | — | none |
| DevNet | baseline | 0/85 | 0/85 | unsuitable under tested configuration (0/85 qualified) | — | — | — | labeled anomalies required by this protocol: 85 |
| RGraph | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.3872 | 0.0982 | — | none |
| LUNAR | baseline | 85/85 | 85/85 | fully operational under tested configuration (85/85) | 0.7106 | 0.3666 | 7/35 | none |

## Fixed-field NSA rank and score

Each cell gives the NSA rank after insertion into the fixed six-size eligible baseline field, its absolute score, and whether that NSA also qualifies at all six sizes. If it lacks 85/85 success in that panel, the rank is blank and coverage is shown. The all-method common-field ranks are exported separately. Competition ranks count scores strictly more than 1e−12 above the focal score; a smaller rank fraction means a higher position, not an accuracy percentage.

### Native features: AUROC

| NSA configuration | 100% | 75% | 50% | 25% | 10% | 5% |
| --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | 20/20 · 0.3597 · six-size ✓ | 20/20 · 0.3504 · six-size ✓ | 20/20 · 0.3460 · six-size ✓ | 20/20 · 0.3297 · six-size ✓ | 20/20 · 0.2982 · six-size ✓ | 20/20 · 0.2951 · six-size ✓ |
| NSA/variable | 17/20 · 0.5931 · six-size ✓ | 17/20 · 0.5821 · six-size ✓ | 17/20 · 0.5531 · six-size ✓ | 20/20 · 0.5025 · six-size ✓ | 20/20 · 0.4350 · six-size ✓ | 20/20 · 0.3844 · six-size ✓ |
| NSA/coverage | 17/20 · 0.5931 · six-size ✓ | 17/20 · 0.5814 · six-size ✓ | 17/20 · 0.5534 · six-size ✓ | 20/20 · 0.5028 · six-size ✓ | 20/20 · 0.4333 · six-size ✓ | 20/20 · 0.3819 · six-size ✓ |
| NSA/grid | 17/20 · 0.5972 · panel only | 17/20 · 0.5880 · panel only | 17/20 · 0.5569 · panel only | — · 84/85 | 20/20 · 0.4323 · panel only | 20/20 · 0.3823 · panel only |
| NSA/hierarchical | 20/20 · 0.2968 · six-size ✓ | 20/20 · 0.3078 · six-size ✓ | 20/20 · 0.2753 · six-size ✓ | 20/20 · 0.2712 · six-size ✓ | 20/20 · 0.2452 · six-size ✓ | 20/20 · 0.3089 · six-size ✓ |
| NSA/voronoi | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/deterministic | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/suppressed | 17/20 · 0.5883 · six-size ✓ | 17/20 · 0.5656 · six-size ✓ | 17/20 · 0.5516 · six-size ✓ | 20/20 · 0.4920 · six-size ✓ | 20/20 · 0.4366 · six-size ✓ | 20/20 · 0.3803 · six-size ✓ |
| NSA/dual | — · 81/85 | — · 80/85 | — · 81/85 | — · 83/85 | 14/20 · 0.7792 · panel only | 10/20 · 0.7900 · panel only |
| NSA/annealed | 20/20 · 0.3161 · six-size ✓ | 20/20 · 0.3112 · six-size ✓ | 20/20 · 0.3023 · six-size ✓ | 20/20 · 0.2789 · six-size ✓ | 20/20 · 0.2641 · six-size ✓ | 20/20 · 0.2564 · six-size ✓ |
| NSA/binary/hamming | — · 76/85 | — · 83/85 | 19/20 · 0.5288 · panel only | 20/20 · 0.5172 · panel only | 18/20 · 0.5024 · panel only | 17/20 · 0.5036 · panel only |
| NSA/binary/rcontiguous | — · 59/85 | — · 60/85 | — · 60/85 | — · 63/85 | — · 75/85 | — · 75/85 |
| NSA/binary/rchunk | — · 75/85 | — · 75/85 | — · 75/85 | — · 75/85 | — · 78/85 | — · 84/85 |

### PCA(2): AUROC

| NSA configuration | 100% | 75% | 50% | 25% | 10% | 5% |
| --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | 22/26 · 0.5827 · six-size ✓ | 21/26 · 0.5923 · six-size ✓ | 20/26 · 0.6202 · six-size ✓ | 19/26 · 0.6299 · six-size ✓ | 19/26 · 0.6132 · six-size ✓ | 19/26 · 0.5917 · six-size ✓ |
| NSA/variable | 15/26 · 0.6928 · six-size ✓ | 14/26 · 0.6982 · six-size ✓ | 13/26 · 0.7139 · six-size ✓ | 11/26 · 0.7240 · six-size ✓ | 8/26 · 0.7273 · six-size ✓ | 10/26 · 0.6683 · six-size ✓ |
| NSA/coverage | 15/26 · 0.6694 · six-size ✓ | 15/26 · 0.6763 · six-size ✓ | 13/26 · 0.7039 · six-size ✓ | 14/26 · 0.6954 · six-size ✓ | 9/26 · 0.7110 · six-size ✓ | 10/26 · 0.6669 · six-size ✓ |
| NSA/grid | 15/26 · 0.6886 · six-size ✓ | 15/26 · 0.6942 · six-size ✓ | 13/26 · 0.7105 · six-size ✓ | 11/26 · 0.7261 · six-size ✓ | 8/26 · 0.7243 · six-size ✓ | 10/26 · 0.6775 · six-size ✓ |
| NSA/hierarchical | 15/26 · 0.6754 · six-size ✓ | 15/26 · 0.6676 · six-size ✓ | 15/26 · 0.6921 · six-size ✓ | 14/26 · 0.6836 · six-size ✓ | 12/26 · 0.6891 · six-size ✓ | 12/26 · 0.6579 · six-size ✓ |
| NSA/voronoi | 15/26 · 0.7043 · six-size ✓ | 14/26 · 0.7164 · six-size ✓ | 12/26 · 0.7315 · six-size ✓ | 11/26 · 0.7432 · six-size ✓ | 8/26 · 0.7361 · six-size ✓ | 8/26 · 0.6882 · six-size ✓ |
| NSA/deterministic | 21/26 · 0.6032 · six-size ✓ | 21/26 · 0.6149 · six-size ✓ | 18/26 · 0.6459 · six-size ✓ | 17/26 · 0.6420 · six-size ✓ | 17/26 · 0.6278 · six-size ✓ | 17/26 · 0.6066 · six-size ✓ |
| NSA/suppressed | 15/26 · 0.6738 · six-size ✓ | 15/26 · 0.6841 · six-size ✓ | 13/26 · 0.7063 · six-size ✓ | 12/26 · 0.7120 · six-size ✓ | 9/26 · 0.7173 · six-size ✓ | 12/26 · 0.6650 · six-size ✓ |
| NSA/dual | 13/26 · 0.7172 · six-size ✓ | 12/26 · 0.7268 · six-size ✓ | 12/26 · 0.7419 · six-size ✓ | 10/26 · 0.7603 · six-size ✓ | 6/26 · 0.7626 · six-size ✓ | 5/26 · 0.7264 · six-size ✓ |
| NSA/annealed | 21/26 · 0.5998 · six-size ✓ | 21/26 · 0.6183 · six-size ✓ | 18/26 · 0.6401 · six-size ✓ | 19/26 · 0.6286 · six-size ✓ | 17/26 · 0.6241 · six-size ✓ | 19/26 · 0.5966 · six-size ✓ |
| NSA/binary/hamming | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/binary/rcontiguous | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/binary/rchunk | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |

### Native features: AP

| NSA configuration | 100% | 75% | 50% | 25% | 10% | 5% |
| --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | 20/20 · 0.1287 · six-size ✓ | 20/20 · 0.1272 · six-size ✓ | 20/20 · 0.1232 · six-size ✓ | 20/20 · 0.1211 · six-size ✓ | 20/20 · 0.1093 · six-size ✓ | 20/20 · 0.1173 · six-size ✓ |
| NSA/variable | 16/20 · 0.3038 · six-size ✓ | 16/20 · 0.2865 · six-size ✓ | 16/20 · 0.2456 · six-size ✓ | 18/20 · 0.2054 · six-size ✓ | 18/20 · 0.1620 · six-size ✓ | 18/20 · 0.1509 · six-size ✓ |
| NSA/coverage | 16/20 · 0.3040 · six-size ✓ | 16/20 · 0.2855 · six-size ✓ | 16/20 · 0.2455 · six-size ✓ | 18/20 · 0.2053 · six-size ✓ | 18/20 · 0.1620 · six-size ✓ | 18/20 · 0.1463 · six-size ✓ |
| NSA/grid | 16/20 · 0.3026 · panel only | 16/20 · 0.2869 · panel only | 16/20 · 0.2510 · panel only | — · 84/85 | 18/20 · 0.1573 · panel only | 18/20 · 0.1417 · panel only |
| NSA/hierarchical | 20/20 · 0.1173 · six-size ✓ | 20/20 · 0.1145 · six-size ✓ | 20/20 · 0.0938 · six-size ✓ | 20/20 · 0.1092 · six-size ✓ | 20/20 · 0.1031 · six-size ✓ | 20/20 · 0.1193 · six-size ✓ |
| NSA/voronoi | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/deterministic | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/suppressed | 16/20 · 0.2982 · six-size ✓ | 16/20 · 0.2705 · six-size ✓ | 16/20 · 0.2440 · six-size ✓ | 18/20 · 0.1989 · six-size ✓ | 18/20 · 0.1614 · six-size ✓ | 18/20 · 0.1483 · six-size ✓ |
| NSA/dual | — · 81/85 | — · 80/85 | — · 81/85 | — · 83/85 | 7/20 · 0.5112 · panel only | 4/20 · 0.5066 · panel only |
| NSA/annealed | 20/20 · 0.1120 · six-size ✓ | 20/20 · 0.1058 · six-size ✓ | 20/20 · 0.1070 · six-size ✓ | 20/20 · 0.0988 · six-size ✓ | 20/20 · 0.1040 · six-size ✓ | 20/20 · 0.0924 · six-size ✓ |
| NSA/binary/hamming | — · 76/85 | — · 83/85 | 20/20 · 0.1573 · panel only | 20/20 · 0.1449 · panel only | 20/20 · 0.1361 · panel only | 20/20 · 0.1234 · panel only |
| NSA/binary/rcontiguous | — · 59/85 | — · 60/85 | — · 60/85 | — · 63/85 | — · 75/85 | — · 75/85 |
| NSA/binary/rchunk | — · 75/85 | — · 75/85 | — · 75/85 | — · 75/85 | — · 78/85 | — · 84/85 |

### PCA(2): AP

| NSA configuration | 100% | 75% | 50% | 25% | 10% | 5% |
| --- | --- | --- | --- | --- | --- | --- |
| NSA/fixed | 16/26 · 0.3427 · six-size ✓ | 17/26 · 0.3328 · six-size ✓ | 14/26 · 0.3564 · six-size ✓ | 14/26 · 0.3474 · six-size ✓ | 16/26 · 0.2789 · six-size ✓ | 19/26 · 0.2200 · six-size ✓ |
| NSA/variable | 10/26 · 0.4441 · six-size ✓ | 10/26 · 0.4435 · six-size ✓ | 10/26 · 0.4466 · six-size ✓ | 8/26 · 0.4204 · six-size ✓ | 9/26 · 0.3638 · six-size ✓ | 15/26 · 0.2879 · six-size ✓ |
| NSA/coverage | 10/26 · 0.4253 · six-size ✓ | 10/26 · 0.4310 · six-size ✓ | 10/26 · 0.4422 · six-size ✓ | 10/26 · 0.4102 · six-size ✓ | 10/26 · 0.3571 · six-size ✓ | 15/26 · 0.2838 · six-size ✓ |
| NSA/grid | 10/26 · 0.4361 · six-size ✓ | 10/26 · 0.4385 · six-size ✓ | 10/26 · 0.4454 · six-size ✓ | 8/26 · 0.4208 · six-size ✓ | 9/26 · 0.3668 · six-size ✓ | 14/26 · 0.3002 · six-size ✓ |
| NSA/hierarchical | 10/26 · 0.4218 · six-size ✓ | 10/26 · 0.4090 · six-size ✓ | 10/26 · 0.4266 · six-size ✓ | 11/26 · 0.3857 · six-size ✓ | 14/26 · 0.3316 · six-size ✓ | 16/26 · 0.2647 · six-size ✓ |
| NSA/voronoi | 9/26 · 0.4521 · six-size ✓ | 8/26 · 0.4590 · six-size ✓ | 9/26 · 0.4550 · six-size ✓ | 8/26 · 0.4347 · six-size ✓ | 9/26 · 0.3689 · six-size ✓ | 12/26 · 0.3153 · six-size ✓ |
| NSA/deterministic | 14/26 · 0.3594 · six-size ✓ | 13/26 · 0.3586 · six-size ✓ | 12/26 · 0.3928 · six-size ✓ | 14/26 · 0.3450 · six-size ✓ | 15/26 · 0.2893 · six-size ✓ | 19/26 · 0.2198 · six-size ✓ |
| NSA/suppressed | 10/26 · 0.4214 · six-size ✓ | 10/26 · 0.4288 · six-size ✓ | 10/26 · 0.4402 · six-size ✓ | 8/26 · 0.4186 · six-size ✓ | 10/26 · 0.3547 · six-size ✓ | 14/26 · 0.2955 · six-size ✓ |
| NSA/dual | 6/26 · 0.4718 · six-size ✓ | 6/26 · 0.4777 · six-size ✓ | 6/26 · 0.4801 · six-size ✓ | 3/26 · 0.4699 · six-size ✓ | 6/26 · 0.4206 · six-size ✓ | 5/26 · 0.3755 · six-size ✓ |
| NSA/annealed | 13/26 · 0.3672 · six-size ✓ | 12/26 · 0.3715 · six-size ✓ | 12/26 · 0.3835 · six-size ✓ | 15/26 · 0.3403 · six-size ✓ | 16/26 · 0.2869 · six-size ✓ | 19/26 · 0.2187 · six-size ✓ |
| NSA/binary/hamming | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/binary/rcontiguous | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |
| NSA/binary/rchunk | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 | — · 0/85 |

## Best other PyOD detectors under fixed and changing fields

The fixed-field leaders are selected descriptively from methods that qualified at all six sizes. The panel field is a changing-field sensitivity.

| Features | % fit | Metric | Top fixed-field other detector | Fixed other N | Top panel-field other detector | Panel other N |
| --- | --- | --- | --- | --- | --- | --- |
| native | 100 | AUROC | LUNAR (0.9058) | 19 | LUNAR (0.9058) | 32 |
| native | 100 | AP | LUNAR (0.6768) | 19 | LUNAR (0.6768) | 32 |
| native | 75 | AUROC | LUNAR (0.9022) | 19 | LUNAR (0.9022) | 29 |
| native | 75 | AP | LUNAR (0.6401) | 19 | LUNAR (0.6401) | 29 |
| native | 50 | AUROC | LUNAR (0.8967) | 19 | LUNAR (0.8967) | 31 |
| native | 50 | AP | LUNAR (0.6285) | 19 | LUNAR (0.6285) | 31 |
| native | 25 | AUROC | KDE (0.8933) | 19 | KDE (0.8933) | 23 |
| native | 25 | AP | KPCA (0.6004) | 19 | KPCA (0.6004) | 23 |
| native | 10 | AUROC | KDE (0.8781) | 19 | KDE (0.8781) | 27 |
| native | 10 | AP | KDE (0.5620) | 19 | KDE (0.5620) | 27 |
| native | 5 | AUROC | KDE (0.8577) | 19 | KDE (0.8577) | 28 |
| native | 5 | AP | KDE (0.5237) | 19 | KDE (0.5237) | 28 |
| pca2 | 100 | AUROC | KNN (0.7958) | 25 | KNN (0.7958) | 38 |
| pca2 | 100 | AP | KNN (0.4872) | 25 | KNN (0.4872) | 38 |
| pca2 | 75 | AUROC | KNN (0.7993) | 25 | KNN (0.7993) | 36 |
| pca2 | 75 | AP | KNN (0.4891) | 25 | KNN (0.4891) | 36 |
| pca2 | 50 | AUROC | KNN (0.8051) | 25 | KNN (0.8051) | 38 |
| pca2 | 50 | AP | KNN (0.4942) | 25 | KNN (0.4942) | 38 |
| pca2 | 25 | AUROC | KNN (0.8005) | 25 | KNN (0.8005) | 26 |
| pca2 | 25 | AP | KDE (0.4711) | 25 | KDE (0.4711) | 26 |
| pca2 | 10 | AUROC | KDE (0.7769) | 25 | KDE (0.7769) | 30 |
| pca2 | 10 | AP | KDE (0.4373) | 25 | KDE (0.4373) | 30 |
| pca2 | 5 | AUROC | KDE (0.7496) | 25 | KDE (0.7496) | 33 |
| pca2 | 5 | AP | KDE (0.3949) | 25 | KDE (0.3949) | 33 |

## Top NSA and top fixed-field other method at each size

Leaders are chosen descriptively using each panel's observed score; the comparator and focal can change between sizes and between AUROC and AP. Each difference is then calculated on matched dataset/seed outcomes, with an unadjusted dataset-cluster interval. This selected leader comparison is exploratory and is not a test of general supremacy.

| Features | % fit | Metric | Top panel-complete NSA | Top fixed-field other | Matched | NSA − other | 95% CI | AUROC-win datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 100 | AUROC | NSA/grid (0.5972; panel only) | LUNAR (0.9058) | 85/85 | -0.3086 | [-0.449, -0.192] | 1/17 |
| native | 100 | AP | NSA/coverage (0.3040; six-size) | LUNAR (0.6768) | 85/85 | -0.3728 | [-0.510, -0.250] | — |
| native | 75 | AUROC | NSA/grid (0.5880; panel only) | LUNAR (0.9022) | 85/85 | -0.3141 | [-0.458, -0.190] | 1/17 |
| native | 75 | AP | NSA/grid (0.2869; panel only) | LUNAR (0.6401) | 85/85 | -0.3531 | [-0.487, -0.231] | — |
| native | 50 | AUROC | NSA/grid (0.5569; panel only) | LUNAR (0.8967) | 85/85 | -0.3398 | [-0.479, -0.219] | 1/17 |
| native | 50 | AP | NSA/grid (0.2510; panel only) | LUNAR (0.6285) | 85/85 | -0.3774 | [-0.518, -0.254] | — |
| native | 25 | AUROC | NSA/binary/hamming (0.5172; panel only) | KDE (0.8933) | 85/85 | -0.3761 | [-0.462, -0.283] | 1/17 |
| native | 25 | AP | NSA/variable (0.2054; six-size) | KPCA (0.6004) | 85/85 | -0.3951 | [-0.549, -0.248] | — |
| native | 10 | AUROC | NSA/dual (0.7792; panel only) | KDE (0.8781) | 85/85 | -0.0989 | [-0.169, -0.043] | 2/17 |
| native | 10 | AP | NSA/dual (0.5112; panel only) | KDE (0.5620) | 85/85 | -0.0508 | [-0.095, -0.007] | — |
| native | 5 | AUROC | NSA/dual (0.7900; panel only) | KDE (0.8577) | 85/85 | -0.0677 | [-0.115, -0.030] | 2/17 |
| native | 5 | AP | NSA/dual (0.5066; panel only) | KDE (0.5237) | 85/85 | -0.0171 | [-0.044, +0.016] | — |
| pca2 | 100 | AUROC | NSA/dual (0.7172; six-size) | KNN (0.7958) | 85/85 | -0.0786 | [-0.141, -0.028] | 3/17 |
| pca2 | 100 | AP | NSA/dual (0.4718; six-size) | KNN (0.4872) | 85/85 | -0.0155 | [-0.047, +0.019] | — |
| pca2 | 75 | AUROC | NSA/dual (0.7268; six-size) | KNN (0.7993) | 85/85 | -0.0725 | [-0.127, -0.025] | 2/17 |
| pca2 | 75 | AP | NSA/dual (0.4777; six-size) | KNN (0.4891) | 85/85 | -0.0113 | [-0.038, +0.012] | — |
| pca2 | 50 | AUROC | NSA/dual (0.7419; six-size) | KNN (0.8051) | 85/85 | -0.0632 | [-0.116, -0.020] | 3/17 |
| pca2 | 50 | AP | NSA/dual (0.4801; six-size) | KNN (0.4942) | 85/85 | -0.0141 | [-0.038, +0.007] | — |
| pca2 | 25 | AUROC | NSA/dual (0.7603; six-size) | KNN (0.8005) | 85/85 | -0.0402 | [-0.092, -0.002] | 6/17 |
| pca2 | 25 | AP | NSA/dual (0.4699; six-size) | KDE (0.4711) | 85/85 | -0.0012 | [-0.018, +0.017] | — |
| pca2 | 10 | AUROC | NSA/dual (0.7626; six-size) | KDE (0.7769) | 85/85 | -0.0143 | [-0.032, +0.004] | 4/17 |
| pca2 | 10 | AP | NSA/dual (0.4206; six-size) | KDE (0.4373) | 85/85 | -0.0167 | [-0.040, +0.003] | — |
| pca2 | 5 | AUROC | NSA/dual (0.7264; six-size) | KDE (0.7496) | 85/85 | -0.0232 | [-0.045, -0.002] | 6/17 |
| pca2 | 5 | AP | NSA/dual (0.3755; six-size) | KDE (0.3949) | 85/85 | -0.0194 | [-0.047, +0.005] | — |

## Fixed-field rank changes versus 100%

| Features | NSA | Metric | % fit | 100% rank | Reduced rank | Rank improvement | Score change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | NSA/fixed | AUROC | 75 | 23/25 | 23/25 | 0 | -0.0093 |
| native | NSA/fixed | AUROC | 50 | 23/25 | 23/25 | 0 | -0.0137 |
| native | NSA/fixed | AUROC | 25 | 23/25 | 23/25 | 0 | -0.0300 |
| native | NSA/fixed | AUROC | 10 | 23/25 | 23/25 | 0 | -0.0615 |
| native | NSA/fixed | AUROC | 5 | 23/25 | 24/25 | -1 | -0.0645 |
| native | NSA/fixed | AP | 75 | 23/25 | 23/25 | 0 | -0.0015 |
| native | NSA/fixed | AP | 50 | 23/25 | 23/25 | 0 | -0.0055 |
| native | NSA/fixed | AP | 25 | 23/25 | 23/25 | 0 | -0.0076 |
| native | NSA/fixed | AP | 10 | 23/25 | 23/25 | 0 | -0.0194 |
| native | NSA/fixed | AP | 5 | 23/25 | 24/25 | -1 | -0.0113 |
| native | NSA/variable | AUROC | 75 | 18/25 | 17/25 | 1 | -0.0110 |
| native | NSA/variable | AUROC | 50 | 18/25 | 18/25 | 0 | -0.0400 |
| native | NSA/variable | AUROC | 25 | 18/25 | 21/25 | -3 | -0.0906 |
| native | NSA/variable | AUROC | 10 | 18/25 | 21/25 | -3 | -0.1581 |
| native | NSA/variable | AUROC | 5 | 18/25 | 20/25 | -2 | -0.2087 |
| native | NSA/variable | AP | 75 | 17/25 | 16/25 | 1 | -0.0173 |
| native | NSA/variable | AP | 50 | 17/25 | 16/25 | 1 | -0.0582 |
| native | NSA/variable | AP | 25 | 17/25 | 18/25 | -1 | -0.0984 |
| native | NSA/variable | AP | 10 | 17/25 | 18/25 | -1 | -0.1418 |
| native | NSA/variable | AP | 5 | 17/25 | 18/25 | -1 | -0.1529 |
| native | NSA/coverage | AUROC | 75 | 17/25 | 18/25 | -1 | -0.0117 |
| native | NSA/coverage | AUROC | 50 | 17/25 | 17/25 | 0 | -0.0397 |
| native | NSA/coverage | AUROC | 25 | 17/25 | 20/25 | -3 | -0.0904 |
| native | NSA/coverage | AUROC | 10 | 17/25 | 22/25 | -5 | -0.1598 |
| native | NSA/coverage | AUROC | 5 | 17/25 | 21/25 | -4 | -0.2112 |
| native | NSA/coverage | AP | 75 | 16/25 | 17/25 | -1 | -0.0184 |
| native | NSA/coverage | AP | 50 | 16/25 | 17/25 | -1 | -0.0585 |
| native | NSA/coverage | AP | 25 | 16/25 | 19/25 | -3 | -0.0987 |
| native | NSA/coverage | AP | 10 | 16/25 | 19/25 | -3 | -0.1420 |
| native | NSA/coverage | AP | 5 | 16/25 | 20/25 | -4 | -0.1577 |
| native | NSA/hierarchical | AUROC | 75 | 25/25 | 25/25 | 0 | +0.0110 |
| native | NSA/hierarchical | AUROC | 50 | 25/25 | 25/25 | 0 | -0.0216 |
| native | NSA/hierarchical | AUROC | 25 | 25/25 | 25/25 | 0 | -0.0256 |
| native | NSA/hierarchical | AUROC | 10 | 25/25 | 25/25 | 0 | -0.0516 |
| native | NSA/hierarchical | AUROC | 5 | 25/25 | 23/25 | 2 | +0.0121 |
| native | NSA/hierarchical | AP | 75 | 24/25 | 24/25 | 0 | -0.0028 |
| native | NSA/hierarchical | AP | 50 | 24/25 | 25/25 | -1 | -0.0236 |
| native | NSA/hierarchical | AP | 25 | 24/25 | 24/25 | 0 | -0.0081 |
| native | NSA/hierarchical | AP | 10 | 24/25 | 25/25 | -1 | -0.0142 |
| native | NSA/hierarchical | AP | 5 | 24/25 | 23/25 | 1 | +0.0019 |
| native | NSA/suppressed | AUROC | 75 | 19/25 | 19/25 | 0 | -0.0228 |
| native | NSA/suppressed | AUROC | 50 | 19/25 | 19/25 | 0 | -0.0368 |
| native | NSA/suppressed | AUROC | 25 | 19/25 | 22/25 | -3 | -0.0964 |
| native | NSA/suppressed | AUROC | 10 | 19/25 | 20/25 | -1 | -0.1518 |
| native | NSA/suppressed | AUROC | 5 | 19/25 | 22/25 | -3 | -0.2080 |
| native | NSA/suppressed | AP | 75 | 18/25 | 18/25 | 0 | -0.0277 |
| native | NSA/suppressed | AP | 50 | 18/25 | 18/25 | 0 | -0.0543 |
| native | NSA/suppressed | AP | 25 | 18/25 | 20/25 | -2 | -0.0993 |
| native | NSA/suppressed | AP | 10 | 18/25 | 20/25 | -2 | -0.1369 |
| native | NSA/suppressed | AP | 5 | 18/25 | 19/25 | -1 | -0.1500 |
| native | NSA/annealed | AUROC | 75 | 24/25 | 24/25 | 0 | -0.0048 |
| native | NSA/annealed | AUROC | 50 | 24/25 | 24/25 | 0 | -0.0138 |
| native | NSA/annealed | AUROC | 25 | 24/25 | 24/25 | 0 | -0.0371 |
| native | NSA/annealed | AUROC | 10 | 24/25 | 24/25 | 0 | -0.0520 |
| native | NSA/annealed | AUROC | 5 | 24/25 | 25/25 | -1 | -0.0597 |
| native | NSA/annealed | AP | 75 | 25/25 | 25/25 | 0 | -0.0062 |
| native | NSA/annealed | AP | 50 | 25/25 | 24/25 | 1 | -0.0050 |
| native | NSA/annealed | AP | 25 | 25/25 | 25/25 | 0 | -0.0132 |
| native | NSA/annealed | AP | 10 | 25/25 | 24/25 | 1 | -0.0080 |
| native | NSA/annealed | AP | 5 | 25/25 | 25/25 | 0 | -0.0196 |
| pca2 | NSA/fixed | AUROC | 75 | 31/35 | 30/35 | 1 | +0.0096 |
| pca2 | NSA/fixed | AUROC | 50 | 31/35 | 29/35 | 2 | +0.0376 |
| pca2 | NSA/fixed | AUROC | 25 | 31/35 | 27/35 | 4 | +0.0473 |
| pca2 | NSA/fixed | AUROC | 10 | 31/35 | 28/35 | 3 | +0.0305 |
| pca2 | NSA/fixed | AUROC | 5 | 31/35 | 28/35 | 3 | +0.0090 |
| pca2 | NSA/fixed | AP | 75 | 25/35 | 26/35 | -1 | -0.0099 |
| pca2 | NSA/fixed | AP | 50 | 25/35 | 23/35 | 2 | +0.0138 |
| pca2 | NSA/fixed | AP | 25 | 25/35 | 21/35 | 4 | +0.0047 |
| pca2 | NSA/fixed | AP | 10 | 25/35 | 25/35 | 0 | -0.0638 |
| pca2 | NSA/fixed | AP | 5 | 25/35 | 26/35 | -1 | -0.1227 |
| pca2 | NSA/variable | AUROC | 75 | 17/35 | 16/35 | 1 | +0.0054 |
| pca2 | NSA/variable | AUROC | 50 | 17/35 | 15/35 | 2 | +0.0212 |
| pca2 | NSA/variable | AUROC | 25 | 17/35 | 14/35 | 3 | +0.0313 |
| pca2 | NSA/variable | AUROC | 10 | 17/35 | 10/35 | 7 | +0.0345 |
| pca2 | NSA/variable | AUROC | 5 | 17/35 | 13/35 | 4 | -0.0245 |
| pca2 | NSA/variable | AP | 75 | 12/35 | 12/35 | 0 | -0.0006 |
| pca2 | NSA/variable | AP | 50 | 12/35 | 12/35 | 0 | +0.0024 |
| pca2 | NSA/variable | AP | 25 | 12/35 | 11/35 | 1 | -0.0237 |
| pca2 | NSA/variable | AP | 10 | 12/35 | 12/35 | 0 | -0.0803 |
| pca2 | NSA/variable | AP | 5 | 12/35 | 19/35 | -7 | -0.1562 |
| pca2 | NSA/coverage | AUROC | 75 | 21/35 | 20/35 | 1 | +0.0069 |
| pca2 | NSA/coverage | AUROC | 50 | 21/35 | 18/35 | 3 | +0.0345 |
| pca2 | NSA/coverage | AUROC | 25 | 21/35 | 19/35 | 2 | +0.0260 |
| pca2 | NSA/coverage | AUROC | 10 | 21/35 | 14/35 | 7 | +0.0417 |
| pca2 | NSA/coverage | AUROC | 5 | 21/35 | 14/35 | 7 | -0.0025 |
| pca2 | NSA/coverage | AP | 75 | 14/35 | 14/35 | 0 | +0.0057 |
| pca2 | NSA/coverage | AP | 50 | 14/35 | 14/35 | 0 | +0.0169 |
| pca2 | NSA/coverage | AP | 25 | 14/35 | 15/35 | -1 | -0.0152 |
| pca2 | NSA/coverage | AP | 10 | 14/35 | 14/35 | 0 | -0.0683 |
| pca2 | NSA/coverage | AP | 5 | 14/35 | 20/35 | -6 | -0.1416 |
| pca2 | NSA/grid | AUROC | 75 | 18/35 | 18/35 | 0 | +0.0057 |
| pca2 | NSA/grid | AUROC | 50 | 18/35 | 16/35 | 2 | +0.0219 |
| pca2 | NSA/grid | AUROC | 25 | 18/35 | 13/35 | 5 | +0.0375 |
| pca2 | NSA/grid | AUROC | 10 | 18/35 | 11/35 | 7 | +0.0357 |
| pca2 | NSA/grid | AUROC | 5 | 18/35 | 12/35 | 6 | -0.0110 |
| pca2 | NSA/grid | AP | 75 | 13/35 | 13/35 | 0 | +0.0024 |
| pca2 | NSA/grid | AP | 50 | 13/35 | 13/35 | 0 | +0.0093 |
| pca2 | NSA/grid | AP | 25 | 13/35 | 10/35 | 3 | -0.0153 |
| pca2 | NSA/grid | AP | 10 | 13/35 | 11/35 | 2 | -0.0693 |
| pca2 | NSA/grid | AP | 5 | 13/35 | 16/35 | -3 | -0.1359 |
| pca2 | NSA/hierarchical | AUROC | 75 | 19/35 | 21/35 | -2 | -0.0078 |
| pca2 | NSA/hierarchical | AUROC | 50 | 19/35 | 21/35 | -2 | +0.0167 |
| pca2 | NSA/hierarchical | AUROC | 25 | 19/35 | 20/35 | -1 | +0.0082 |
| pca2 | NSA/hierarchical | AUROC | 10 | 19/35 | 18/35 | 1 | +0.0137 |
| pca2 | NSA/hierarchical | AUROC | 5 | 19/35 | 18/35 | 1 | -0.0175 |
| pca2 | NSA/hierarchical | AP | 75 | 15/35 | 16/35 | -1 | -0.0128 |
| pca2 | NSA/hierarchical | AP | 50 | 15/35 | 16/35 | -1 | +0.0048 |
| pca2 | NSA/hierarchical | AP | 25 | 15/35 | 17/35 | -2 | -0.0361 |
| pca2 | NSA/hierarchical | AP | 10 | 15/35 | 20/35 | -5 | -0.0902 |
| pca2 | NSA/hierarchical | AP | 5 | 15/35 | 22/35 | -7 | -0.1571 |
| pca2 | NSA/voronoi | AUROC | 75 | 16/35 | 15/35 | 1 | +0.0122 |
| pca2 | NSA/voronoi | AUROC | 50 | 16/35 | 13/35 | 3 | +0.0272 |
| pca2 | NSA/voronoi | AUROC | 25 | 16/35 | 12/35 | 4 | +0.0389 |
| pca2 | NSA/voronoi | AUROC | 10 | 16/35 | 9/35 | 7 | +0.0318 |
| pca2 | NSA/voronoi | AUROC | 5 | 16/35 | 9/35 | 7 | -0.0160 |
| pca2 | NSA/voronoi | AP | 75 | 10/35 | 9/35 | 1 | +0.0069 |
| pca2 | NSA/voronoi | AP | 50 | 10/35 | 10/35 | 0 | +0.0029 |
| pca2 | NSA/voronoi | AP | 25 | 10/35 | 9/35 | 1 | -0.0174 |
| pca2 | NSA/voronoi | AP | 10 | 10/35 | 10/35 | 0 | -0.0833 |
| pca2 | NSA/voronoi | AP | 5 | 10/35 | 13/35 | -3 | -0.1368 |
| pca2 | NSA/deterministic | AUROC | 75 | 28/35 | 29/35 | -1 | +0.0117 |
| pca2 | NSA/deterministic | AUROC | 50 | 28/35 | 25/35 | 3 | +0.0428 |
| pca2 | NSA/deterministic | AUROC | 25 | 28/35 | 24/35 | 4 | +0.0389 |
| pca2 | NSA/deterministic | AUROC | 10 | 28/35 | 24/35 | 4 | +0.0247 |
| pca2 | NSA/deterministic | AUROC | 5 | 28/35 | 24/35 | 4 | +0.0035 |
| pca2 | NSA/deterministic | AP | 75 | 22/35 | 21/35 | 1 | -0.0009 |
| pca2 | NSA/deterministic | AP | 50 | 22/35 | 19/35 | 3 | +0.0334 |
| pca2 | NSA/deterministic | AP | 25 | 22/35 | 22/35 | 0 | -0.0145 |
| pca2 | NSA/deterministic | AP | 10 | 22/35 | 22/35 | 0 | -0.0702 |
| pca2 | NSA/deterministic | AP | 5 | 22/35 | 27/35 | -5 | -0.1396 |
| pca2 | NSA/suppressed | AUROC | 75 | 20/35 | 19/35 | 1 | +0.0102 |
| pca2 | NSA/suppressed | AUROC | 50 | 20/35 | 17/35 | 3 | +0.0324 |
| pca2 | NSA/suppressed | AUROC | 25 | 20/35 | 16/35 | 4 | +0.0382 |
| pca2 | NSA/suppressed | AUROC | 10 | 20/35 | 13/35 | 7 | +0.0435 |
| pca2 | NSA/suppressed | AUROC | 5 | 20/35 | 17/35 | 3 | -0.0088 |
| pca2 | NSA/suppressed | AP | 75 | 16/35 | 15/35 | 1 | +0.0075 |
| pca2 | NSA/suppressed | AP | 50 | 16/35 | 15/35 | 1 | +0.0188 |
| pca2 | NSA/suppressed | AP | 25 | 16/35 | 12/35 | 4 | -0.0028 |
| pca2 | NSA/suppressed | AP | 10 | 16/35 | 15/35 | 1 | -0.0667 |
| pca2 | NSA/suppressed | AP | 5 | 16/35 | 17/35 | -1 | -0.1259 |
| pca2 | NSA/dual | AUROC | 75 | 13/35 | 12/35 | 1 | +0.0096 |
| pca2 | NSA/dual | AUROC | 50 | 13/35 | 12/35 | 1 | +0.0247 |
| pca2 | NSA/dual | AUROC | 25 | 13/35 | 10/35 | 3 | +0.0431 |
| pca2 | NSA/dual | AUROC | 10 | 13/35 | 6/35 | 7 | +0.0454 |
| pca2 | NSA/dual | AUROC | 5 | 13/35 | 5/35 | 8 | +0.0092 |
| pca2 | NSA/dual | AP | 75 | 6/35 | 6/35 | 0 | +0.0060 |
| pca2 | NSA/dual | AP | 50 | 6/35 | 6/35 | 0 | +0.0083 |
| pca2 | NSA/dual | AP | 25 | 6/35 | 3/35 | 3 | -0.0019 |
| pca2 | NSA/dual | AP | 10 | 6/35 | 6/35 | 0 | -0.0511 |
| pca2 | NSA/dual | AP | 5 | 6/35 | 5/35 | 1 | -0.0962 |
| pca2 | NSA/annealed | AUROC | 75 | 29/35 | 28/35 | 1 | +0.0185 |
| pca2 | NSA/annealed | AUROC | 50 | 29/35 | 26/35 | 3 | +0.0403 |
| pca2 | NSA/annealed | AUROC | 25 | 29/35 | 28/35 | 1 | +0.0288 |
| pca2 | NSA/annealed | AUROC | 10 | 29/35 | 25/35 | 4 | +0.0243 |
| pca2 | NSA/annealed | AUROC | 5 | 29/35 | 27/35 | 2 | -0.0032 |
| pca2 | NSA/annealed | AP | 75 | 20/35 | 19/35 | 1 | +0.0043 |
| pca2 | NSA/annealed | AP | 50 | 20/35 | 20/35 | 0 | +0.0163 |
| pca2 | NSA/annealed | AP | 25 | 20/35 | 24/35 | -4 | -0.0270 |
| pca2 | NSA/annealed | AP | 10 | 20/35 | 24/35 | -4 | -0.0803 |
| pca2 | NSA/annealed | AP | 5 | 20/35 | 28/35 | -8 | -0.1485 |

## Paired NSA score changes versus 100%

Differences use only matched successful dataset/seed cells. CIs are descriptive, unadjusted 95% dataset-cluster bootstrap intervals (5,000 draws by default); no interval is estimated from fewer than five paired datasets. Positive Δ means the smaller training pool did better on that metric. A within-method change alone does not establish superiority over another detector.

| Features | NSA configuration | % fit | Matched runs | Datasets | ΔAUROC vs 100% | 95% CI | ΔAP vs 100% | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | NSA/fixed | 75 | 85/85 | 17 | -0.0093 | [-0.023, +0.003] | -0.0015 | [-0.006, +0.004] |
| native | NSA/fixed | 50 | 85/85 | 17 | -0.0137 | [-0.038, +0.017] | -0.0055 | [-0.013, +0.004] |
| native | NSA/fixed | 25 | 85/85 | 17 | -0.0300 | [-0.063, +0.011] | -0.0076 | [-0.016, -0.000] |
| native | NSA/fixed | 10 | 85/85 | 17 | -0.0615 | [-0.099, -0.023] | -0.0194 | [-0.035, -0.004] |
| native | NSA/fixed | 5 | 85/85 | 17 | -0.0645 | [-0.110, -0.017] | -0.0113 | [-0.029, +0.007] |
| native | NSA/variable | 75 | 85/85 | 17 | -0.0110 | [-0.017, -0.006] | -0.0173 | [-0.029, -0.007] |
| native | NSA/variable | 50 | 85/85 | 17 | -0.0400 | [-0.075, -0.009] | -0.0582 | [-0.112, -0.016] |
| native | NSA/variable | 25 | 85/85 | 17 | -0.0906 | [-0.177, -0.019] | -0.0984 | [-0.190, -0.022] |
| native | NSA/variable | 10 | 85/85 | 17 | -0.1581 | [-0.274, -0.058] | -0.1418 | [-0.248, -0.052] |
| native | NSA/variable | 5 | 85/85 | 17 | -0.2087 | [-0.350, -0.088] | -0.1529 | [-0.275, -0.048] |
| native | NSA/coverage | 75 | 85/85 | 17 | -0.0117 | [-0.017, -0.006] | -0.0184 | [-0.029, -0.010] |
| native | NSA/coverage | 50 | 85/85 | 17 | -0.0397 | [-0.075, -0.009] | -0.0585 | [-0.111, -0.017] |
| native | NSA/coverage | 25 | 85/85 | 17 | -0.0904 | [-0.174, -0.018] | -0.0987 | [-0.191, -0.025] |
| native | NSA/coverage | 10 | 85/85 | 17 | -0.1598 | [-0.271, -0.061] | -0.1420 | [-0.247, -0.050] |
| native | NSA/coverage | 5 | 85/85 | 17 | -0.2112 | [-0.349, -0.090] | -0.1577 | [-0.278, -0.059] |
| native | NSA/grid | 75 | 85/85 | 17 | -0.0092 | [-0.021, +0.002] | -0.0156 | [-0.029, -0.004] |
| native | NSA/grid | 50 | 85/85 | 17 | -0.0403 | [-0.078, -0.011] | -0.0515 | [-0.101, -0.012] |
| native | NSA/grid | 25 | 84/85 | 17 | -0.0939 | [-0.178, -0.025] | -0.1027 | [-0.195, -0.036] |
| native | NSA/grid | 10 | 85/85 | 17 | -0.1649 | [-0.277, -0.066] | -0.1452 | [-0.256, -0.060] |
| native | NSA/grid | 5 | 85/85 | 17 | -0.2150 | [-0.348, -0.101] | -0.1609 | [-0.284, -0.059] |
| native | NSA/hierarchical | 75 | 85/85 | 17 | +0.0110 | [-0.037, +0.063] | -0.0028 | [-0.024, +0.016] |
| native | NSA/hierarchical | 50 | 85/85 | 17 | -0.0216 | [-0.068, +0.021] | -0.0236 | [-0.045, -0.007] |
| native | NSA/hierarchical | 25 | 85/85 | 17 | -0.0256 | [-0.077, +0.023] | -0.0081 | [-0.034, +0.015] |
| native | NSA/hierarchical | 10 | 85/85 | 17 | -0.0516 | [-0.111, -0.001] | -0.0142 | [-0.044, +0.014] |
| native | NSA/hierarchical | 5 | 85/85 | 17 | +0.0121 | [-0.049, +0.072] | +0.0019 | [-0.023, +0.026] |
| native | NSA/voronoi | 75 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/voronoi | 50 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/voronoi | 25 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/voronoi | 10 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/voronoi | 5 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/deterministic | 75 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/deterministic | 50 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/deterministic | 25 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/deterministic | 10 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/deterministic | 5 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| native | NSA/suppressed | 75 | 85/85 | 17 | -0.0228 | [-0.041, -0.009] | -0.0277 | [-0.049, -0.010] |
| native | NSA/suppressed | 50 | 85/85 | 17 | -0.0368 | [-0.067, -0.011] | -0.0543 | [-0.105, -0.013] |
| native | NSA/suppressed | 25 | 85/85 | 17 | -0.0964 | [-0.187, -0.023] | -0.0993 | [-0.196, -0.024] |
| native | NSA/suppressed | 10 | 85/85 | 17 | -0.1518 | [-0.260, -0.061] | -0.1369 | [-0.235, -0.056] |
| native | NSA/suppressed | 5 | 85/85 | 17 | -0.2080 | [-0.338, -0.092] | -0.1500 | [-0.261, -0.048] |
| native | NSA/dual | 75 | 80/85 | 16 | +0.0354 | [+0.010, +0.064] | +0.0303 | [+0.006, +0.067] |
| native | NSA/dual | 50 | 80/85 | 16 | +0.0327 | [+0.001, +0.069] | +0.0331 | [+0.003, +0.072] |
| native | NSA/dual | 25 | 80/85 | 16 | +0.0550 | [+0.018, +0.096] | +0.0554 | [+0.013, +0.107] |
| native | NSA/dual | 10 | 81/85 | 17 | +0.0780 | [+0.034, +0.131] | +0.0532 | [+0.014, +0.095] |
| native | NSA/dual | 5 | 81/85 | 17 | +0.0903 | [+0.041, +0.144] | +0.0493 | [+0.006, +0.095] |
| native | NSA/annealed | 75 | 85/85 | 17 | -0.0048 | [-0.016, +0.006] | -0.0062 | [-0.014, +0.000] |
| native | NSA/annealed | 50 | 85/85 | 17 | -0.0138 | [-0.031, +0.001] | -0.0050 | [-0.021, +0.006] |
| native | NSA/annealed | 25 | 85/85 | 17 | -0.0371 | [-0.075, -0.011] | -0.0132 | [-0.023, -0.005] |
| native | NSA/annealed | 10 | 85/85 | 17 | -0.0520 | [-0.095, -0.017] | -0.0080 | [-0.021, +0.005] |
| native | NSA/annealed | 5 | 85/85 | 17 | -0.0597 | [-0.104, -0.022] | -0.0196 | [-0.038, -0.005] |
| native | NSA/binary/hamming | 75 | 76/85 | 16 | -0.0078 | [-0.022, +0.004] | -0.0048 | [-0.011, +0.002] |
| native | NSA/binary/hamming | 50 | 76/85 | 16 | -0.0022 | [-0.023, +0.019] | -0.0141 | [-0.045, +0.006] |
| native | NSA/binary/hamming | 25 | 76/85 | 16 | -0.0149 | [-0.036, +0.003] | -0.0250 | [-0.069, +0.000] |
| native | NSA/binary/hamming | 10 | 76/85 | 16 | -0.0240 | [-0.051, +0.000] | -0.0325 | [-0.090, +0.002] |
| native | NSA/binary/hamming | 5 | 76/85 | 16 | -0.0234 | [-0.063, +0.010] | -0.0458 | [-0.125, -0.000] |
| native | NSA/binary/rcontiguous | 75 | 59/85 | 12 | +0.0076 | [-0.012, +0.030] | -0.0078 | [-0.019, +0.002] |
| native | NSA/binary/rcontiguous | 50 | 59/85 | 12 | -0.0025 | [-0.023, +0.021] | -0.0067 | [-0.012, -0.001] |
| native | NSA/binary/rcontiguous | 25 | 59/85 | 12 | -0.0165 | [-0.045, +0.011] | -0.0272 | [-0.065, -0.005] |
| native | NSA/binary/rcontiguous | 10 | 59/85 | 12 | -0.0122 | [-0.063, +0.046] | -0.0209 | [-0.068, +0.012] |
| native | NSA/binary/rcontiguous | 5 | 59/85 | 12 | -0.0177 | [-0.072, +0.047] | -0.0180 | [-0.055, +0.011] |
| native | NSA/binary/rchunk | 75 | 75/85 | 15 | +0.0023 | [-0.009, +0.018] | -0.0004 | [-0.006, +0.005] |
| native | NSA/binary/rchunk | 50 | 75/85 | 15 | +0.0060 | [-0.008, +0.024] | -0.0050 | [-0.015, +0.002] |
| native | NSA/binary/rchunk | 25 | 75/85 | 15 | -0.0078 | [-0.026, +0.010] | -0.0179 | [-0.041, -0.003] |
| native | NSA/binary/rchunk | 10 | 75/85 | 15 | +0.0052 | [-0.024, +0.035] | -0.0151 | [-0.031, -0.003] |
| native | NSA/binary/rchunk | 5 | 75/85 | 15 | -0.0018 | [-0.030, +0.029] | -0.0234 | [-0.051, -0.005] |
| pca2 | NSA/fixed | 75 | 85/85 | 17 | +0.0096 | [-0.006, +0.027] | -0.0099 | [-0.031, +0.007] |
| pca2 | NSA/fixed | 50 | 85/85 | 17 | +0.0376 | [+0.006, +0.075] | +0.0138 | [-0.019, +0.044] |
| pca2 | NSA/fixed | 25 | 85/85 | 17 | +0.0473 | [-0.013, +0.123] | +0.0047 | [-0.054, +0.061] |
| pca2 | NSA/fixed | 10 | 85/85 | 17 | +0.0305 | [-0.047, +0.113] | -0.0638 | [-0.145, +0.014] |
| pca2 | NSA/fixed | 5 | 85/85 | 17 | +0.0090 | [-0.093, +0.100] | -0.1227 | [-0.226, -0.028] |
| pca2 | NSA/variable | 75 | 85/85 | 17 | +0.0054 | [-0.004, +0.015] | -0.0006 | [-0.016, +0.014] |
| pca2 | NSA/variable | 50 | 85/85 | 17 | +0.0212 | [-0.004, +0.047] | +0.0024 | [-0.024, +0.027] |
| pca2 | NSA/variable | 25 | 85/85 | 17 | +0.0313 | [-0.025, +0.092] | -0.0237 | [-0.080, +0.021] |
| pca2 | NSA/variable | 10 | 85/85 | 17 | +0.0345 | [-0.030, +0.107] | -0.0803 | [-0.153, -0.018] |
| pca2 | NSA/variable | 5 | 85/85 | 17 | -0.0245 | [-0.110, +0.069] | -0.1562 | [-0.257, -0.064] |
| pca2 | NSA/coverage | 75 | 85/85 | 17 | +0.0069 | [-0.001, +0.016] | +0.0057 | [-0.008, +0.019] |
| pca2 | NSA/coverage | 50 | 85/85 | 17 | +0.0345 | [+0.002, +0.072] | +0.0169 | [-0.013, +0.048] |
| pca2 | NSA/coverage | 25 | 85/85 | 17 | +0.0260 | [-0.033, +0.089] | -0.0152 | [-0.068, +0.028] |
| pca2 | NSA/coverage | 10 | 85/85 | 17 | +0.0417 | [-0.028, +0.120] | -0.0683 | [-0.143, -0.004] |
| pca2 | NSA/coverage | 5 | 85/85 | 17 | -0.0025 | [-0.095, +0.098] | -0.1416 | [-0.249, -0.048] |
| pca2 | NSA/grid | 75 | 85/85 | 17 | +0.0057 | [-0.005, +0.016] | +0.0024 | [-0.013, +0.018] |
| pca2 | NSA/grid | 50 | 85/85 | 17 | +0.0219 | [-0.003, +0.049] | +0.0093 | [-0.013, +0.033] |
| pca2 | NSA/grid | 25 | 85/85 | 17 | +0.0375 | [-0.010, +0.094] | -0.0153 | [-0.059, +0.027] |
| pca2 | NSA/grid | 10 | 85/85 | 17 | +0.0357 | [-0.028, +0.104] | -0.0693 | [-0.132, -0.007] |
| pca2 | NSA/grid | 5 | 85/85 | 17 | -0.0110 | [-0.103, +0.077] | -0.1359 | [-0.226, -0.052] |
| pca2 | NSA/hierarchical | 75 | 85/85 | 17 | -0.0078 | [-0.031, +0.012] | -0.0128 | [-0.041, +0.012] |
| pca2 | NSA/hierarchical | 50 | 85/85 | 17 | +0.0167 | [-0.007, +0.041] | +0.0048 | [-0.028, +0.040] |
| pca2 | NSA/hierarchical | 25 | 85/85 | 17 | +0.0082 | [-0.049, +0.058] | -0.0361 | [-0.088, +0.005] |
| pca2 | NSA/hierarchical | 10 | 85/85 | 17 | +0.0137 | [-0.063, +0.092] | -0.0902 | [-0.158, -0.027] |
| pca2 | NSA/hierarchical | 5 | 85/85 | 17 | -0.0175 | [-0.093, +0.064] | -0.1571 | [-0.257, -0.073] |
| pca2 | NSA/voronoi | 75 | 85/85 | 17 | +0.0122 | [+0.002, +0.022] | +0.0069 | [-0.005, +0.020] |
| pca2 | NSA/voronoi | 50 | 85/85 | 17 | +0.0272 | [+0.005, +0.055] | +0.0029 | [-0.024, +0.035] |
| pca2 | NSA/voronoi | 25 | 85/85 | 17 | +0.0389 | [-0.009, +0.095] | -0.0174 | [-0.060, +0.024] |
| pca2 | NSA/voronoi | 10 | 85/85 | 17 | +0.0318 | [-0.031, +0.097] | -0.0833 | [-0.151, -0.017] |
| pca2 | NSA/voronoi | 5 | 85/85 | 17 | -0.0160 | [-0.099, +0.069] | -0.1368 | [-0.226, -0.052] |
| pca2 | NSA/deterministic | 75 | 85/85 | 17 | +0.0117 | [-0.003, +0.027] | -0.0009 | [-0.027, +0.025] |
| pca2 | NSA/deterministic | 50 | 85/85 | 17 | +0.0428 | [+0.000, +0.097] | +0.0334 | [-0.017, +0.087] |
| pca2 | NSA/deterministic | 25 | 85/85 | 17 | +0.0389 | [-0.024, +0.113] | -0.0145 | [-0.078, +0.044] |
| pca2 | NSA/deterministic | 10 | 85/85 | 17 | +0.0247 | [-0.045, +0.101] | -0.0702 | [-0.154, +0.010] |
| pca2 | NSA/deterministic | 5 | 85/85 | 17 | +0.0035 | [-0.075, +0.077] | -0.1396 | [-0.245, -0.038] |
| pca2 | NSA/suppressed | 75 | 85/85 | 17 | +0.0102 | [-0.002, +0.023] | +0.0075 | [-0.009, +0.023] |
| pca2 | NSA/suppressed | 50 | 85/85 | 17 | +0.0324 | [+0.012, +0.058] | +0.0188 | [-0.003, +0.043] |
| pca2 | NSA/suppressed | 25 | 85/85 | 17 | +0.0382 | [-0.009, +0.094] | -0.0028 | [-0.041, +0.033] |
| pca2 | NSA/suppressed | 10 | 85/85 | 17 | +0.0435 | [-0.023, +0.116] | -0.0667 | [-0.143, +0.000] |
| pca2 | NSA/suppressed | 5 | 85/85 | 17 | -0.0088 | [-0.084, +0.072] | -0.1259 | [-0.215, -0.046] |
| pca2 | NSA/dual | 75 | 85/85 | 17 | +0.0096 | [+0.002, +0.017] | +0.0060 | [-0.007, +0.020] |
| pca2 | NSA/dual | 50 | 85/85 | 17 | +0.0247 | [+0.005, +0.048] | +0.0083 | [-0.010, +0.031] |
| pca2 | NSA/dual | 25 | 85/85 | 17 | +0.0431 | [+0.001, +0.094] | -0.0019 | [-0.030, +0.027] |
| pca2 | NSA/dual | 10 | 85/85 | 17 | +0.0454 | [-0.001, +0.100] | -0.0511 | [-0.099, -0.008] |
| pca2 | NSA/dual | 5 | 85/85 | 17 | +0.0092 | [-0.069, +0.083] | -0.0962 | [-0.176, -0.031] |
| pca2 | NSA/annealed | 75 | 85/85 | 17 | +0.0185 | [+0.004, +0.039] | +0.0043 | [-0.016, +0.024] |
| pca2 | NSA/annealed | 50 | 85/85 | 17 | +0.0403 | [-0.002, +0.104] | +0.0163 | [-0.024, +0.055] |
| pca2 | NSA/annealed | 25 | 85/85 | 17 | +0.0288 | [-0.038, +0.114] | -0.0270 | [-0.091, +0.034] |
| pca2 | NSA/annealed | 10 | 85/85 | 17 | +0.0243 | [-0.055, +0.118] | -0.0803 | [-0.168, +0.002] |
| pca2 | NSA/annealed | 5 | 85/85 | 17 | -0.0032 | [-0.082, +0.078] | -0.1485 | [-0.254, -0.049] |
| pca2 | NSA/binary/hamming | 75 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/hamming | 50 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/hamming | 25 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/hamming | 10 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/hamming | 5 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rcontiguous | 75 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rcontiguous | 50 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rcontiguous | 25 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rcontiguous | 10 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rcontiguous | 5 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rchunk | 75 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rchunk | 50 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rchunk | 25 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rchunk | 10 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |
| pca2 | NSA/binary/rchunk | 5 | 0/85 | 0 | — | interval not estimated | — | interval not estimated |

## NSA versus the strongest fixed-field other detector at 10% and 5%

The comparator is chosen by its observed AUROC in the common baseline field at that size. The full CSV contains all 6,708 NSA-versus-other same-size comparisons, including AP and uncertainty. This concise table shows AUROC point differences and matched coverage.

| Features | % fit | NSA | Top fixed other | Matched runs | NSA − other AUROC |
| --- | --- | --- | --- | --- | --- |
| native | 10 | NSA/fixed | KDE | 85/85 | -0.5800 |
| native | 10 | NSA/variable | KDE | 85/85 | -0.4432 |
| native | 10 | NSA/coverage | KDE | 85/85 | -0.4448 |
| native | 10 | NSA/grid | KDE | 85/85 | -0.4458 |
| native | 10 | NSA/hierarchical | KDE | 85/85 | -0.6329 |
| native | 10 | NSA/voronoi | KDE | 0/85 | — |
| native | 10 | NSA/deterministic | KDE | 0/85 | — |
| native | 10 | NSA/suppressed | KDE | 85/85 | -0.4415 |
| native | 10 | NSA/dual | KDE | 85/85 | -0.0989 |
| native | 10 | NSA/annealed | KDE | 85/85 | -0.6140 |
| native | 10 | NSA/binary/hamming | KDE | 85/85 | -0.3758 |
| native | 10 | NSA/binary/rcontiguous | KDE | 75/85 | -0.3804 |
| native | 10 | NSA/binary/rchunk | KDE | 78/85 | -0.3381 |
| native | 5 | NSA/fixed | KDE | 85/85 | -0.5626 |
| native | 5 | NSA/variable | KDE | 85/85 | -0.4734 |
| native | 5 | NSA/coverage | KDE | 85/85 | -0.4758 |
| native | 5 | NSA/grid | KDE | 85/85 | -0.4755 |
| native | 5 | NSA/hierarchical | KDE | 85/85 | -0.5488 |
| native | 5 | NSA/voronoi | KDE | 0/85 | — |
| native | 5 | NSA/deterministic | KDE | 0/85 | — |
| native | 5 | NSA/suppressed | KDE | 85/85 | -0.4774 |
| native | 5 | NSA/dual | KDE | 85/85 | -0.0677 |
| native | 5 | NSA/annealed | KDE | 85/85 | -0.6013 |
| native | 5 | NSA/binary/hamming | KDE | 85/85 | -0.3541 |
| native | 5 | NSA/binary/rcontiguous | KDE | 75/85 | -0.3520 |
| native | 5 | NSA/binary/rchunk | KDE | 84/85 | -0.3352 |
| pca2 | 10 | NSA/fixed | KDE | 85/85 | -0.1638 |
| pca2 | 10 | NSA/variable | KDE | 85/85 | -0.0496 |
| pca2 | 10 | NSA/coverage | KDE | 85/85 | -0.0659 |
| pca2 | 10 | NSA/grid | KDE | 85/85 | -0.0526 |
| pca2 | 10 | NSA/hierarchical | KDE | 85/85 | -0.0878 |
| pca2 | 10 | NSA/voronoi | KDE | 85/85 | -0.0408 |
| pca2 | 10 | NSA/deterministic | KDE | 85/85 | -0.1491 |
| pca2 | 10 | NSA/suppressed | KDE | 85/85 | -0.0596 |
| pca2 | 10 | NSA/dual | KDE | 85/85 | -0.0143 |
| pca2 | 10 | NSA/annealed | KDE | 85/85 | -0.1528 |
| pca2 | 10 | NSA/binary/hamming | KDE | 0/85 | — |
| pca2 | 10 | NSA/binary/rcontiguous | KDE | 0/85 | — |
| pca2 | 10 | NSA/binary/rchunk | KDE | 0/85 | — |
| pca2 | 5 | NSA/fixed | KDE | 85/85 | -0.1579 |
| pca2 | 5 | NSA/variable | KDE | 85/85 | -0.0813 |
| pca2 | 5 | NSA/coverage | KDE | 85/85 | -0.0827 |
| pca2 | 5 | NSA/grid | KDE | 85/85 | -0.0720 |
| pca2 | 5 | NSA/hierarchical | KDE | 85/85 | -0.0917 |
| pca2 | 5 | NSA/voronoi | KDE | 85/85 | -0.0613 |
| pca2 | 5 | NSA/deterministic | KDE | 85/85 | -0.1430 |
| pca2 | 5 | NSA/suppressed | KDE | 85/85 | -0.0845 |
| pca2 | 5 | NSA/dual | KDE | 85/85 | -0.0232 |
| pca2 | 5 | NSA/annealed | KDE | 85/85 | -0.1530 |
| pca2 | 5 | NSA/binary/hamming | KDE | 0/85 | — |
| pca2 | 5 | NSA/binary/rcontiguous | KDE | 0/85 | — |
| pca2 | 5 | NSA/binary/rchunk | KDE | 0/85 | — |

## Dataset strategy coverage at the new sizes

All 13 configurations, representing all 11 strategy families, remain in the dataset-level CSV. A dataset rank is blank if fewer than five qualified seed outcomes are available. Fixed baseline-plus-focal ranks hold the baseline comparison field across all six sizes; locally qualified ranks use a changing field and are explicitly secondary.

| Features | % fit | NSA | Datasets with 5/5 | Datasets in six-size common field | Successful seed runs / 85 |
| --- | --- | --- | --- | --- | --- |
| native | 10 | NSA/fixed | 17 | 17 | 85 |
| native | 10 | NSA/variable | 17 | 17 | 85 |
| native | 10 | NSA/coverage | 17 | 17 | 85 |
| native | 10 | NSA/grid | 17 | 0 | 85 |
| native | 10 | NSA/hierarchical | 17 | 17 | 85 |
| native | 10 | NSA/voronoi | 0 | 0 | 0 |
| native | 10 | NSA/deterministic | 0 | 0 | 0 |
| native | 10 | NSA/suppressed | 17 | 17 | 85 |
| native | 10 | NSA/dual | 17 | 0 | 85 |
| native | 10 | NSA/annealed | 17 | 17 | 85 |
| native | 10 | NSA/binary/hamming | 17 | 0 | 85 |
| native | 10 | NSA/binary/rcontiguous | 15 | 0 | 75 |
| native | 10 | NSA/binary/rchunk | 15 | 0 | 78 |
| native | 5 | NSA/fixed | 17 | 17 | 85 |
| native | 5 | NSA/variable | 17 | 17 | 85 |
| native | 5 | NSA/coverage | 17 | 17 | 85 |
| native | 5 | NSA/grid | 17 | 0 | 85 |
| native | 5 | NSA/hierarchical | 17 | 17 | 85 |
| native | 5 | NSA/voronoi | 0 | 0 | 0 |
| native | 5 | NSA/deterministic | 0 | 0 | 0 |
| native | 5 | NSA/suppressed | 17 | 17 | 85 |
| native | 5 | NSA/dual | 17 | 0 | 85 |
| native | 5 | NSA/annealed | 17 | 17 | 85 |
| native | 5 | NSA/binary/hamming | 17 | 0 | 85 |
| native | 5 | NSA/binary/rcontiguous | 15 | 0 | 75 |
| native | 5 | NSA/binary/rchunk | 16 | 0 | 84 |
| pca2 | 10 | NSA/fixed | 17 | 17 | 85 |
| pca2 | 10 | NSA/variable | 17 | 17 | 85 |
| pca2 | 10 | NSA/coverage | 17 | 17 | 85 |
| pca2 | 10 | NSA/grid | 17 | 17 | 85 |
| pca2 | 10 | NSA/hierarchical | 17 | 17 | 85 |
| pca2 | 10 | NSA/voronoi | 17 | 17 | 85 |
| pca2 | 10 | NSA/deterministic | 17 | 17 | 85 |
| pca2 | 10 | NSA/suppressed | 17 | 17 | 85 |
| pca2 | 10 | NSA/dual | 17 | 17 | 85 |
| pca2 | 10 | NSA/annealed | 17 | 17 | 85 |
| pca2 | 10 | NSA/binary/hamming | 0 | 0 | 0 |
| pca2 | 10 | NSA/binary/rcontiguous | 0 | 0 | 0 |
| pca2 | 10 | NSA/binary/rchunk | 0 | 0 | 0 |
| pca2 | 5 | NSA/fixed | 17 | 17 | 85 |
| pca2 | 5 | NSA/variable | 17 | 17 | 85 |
| pca2 | 5 | NSA/coverage | 17 | 17 | 85 |
| pca2 | 5 | NSA/grid | 17 | 17 | 85 |
| pca2 | 5 | NSA/hierarchical | 17 | 17 | 85 |
| pca2 | 5 | NSA/voronoi | 17 | 17 | 85 |
| pca2 | 5 | NSA/deterministic | 17 | 17 | 85 |
| pca2 | 5 | NSA/suppressed | 17 | 17 | 85 |
| pca2 | 5 | NSA/dual | 17 | 17 | 85 |
| pca2 | 5 | NSA/annealed | 17 | 17 | 85 |
| pca2 | 5 | NSA/binary/hamming | 0 | 0 | 0 |
| pca2 | 5 | NSA/binary/rcontiguous | 0 | 0 | 0 |
| pca2 | 5 | NSA/binary/rchunk | 0 | 0 | 0 |

## NSA/dual by dataset at 10% and 5%

Each score is the mean across all five seeds only when the dataset has 5/5 strict successes. The comparator field is the same six-size eligible baseline roster plus NSA/dual; its size is printed with each rank. Top % is a rank-derived display value, 100 × (field size − rank + 1) / field size. It is not a statistical confidence level. Missing ranks or scores mean incomplete coverage, not zero performance.

### Native features, 10% fit

| Dataset | Strict success | AUROC | AP | AUROC rank | AUROC top % | AP rank | AP top % | Statuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 5/5 | 0.7755 | 0.4616 | 15/20 | 30.0% | 14/20 | 35.0% | {"success": 5} |
| cardio | 5/5 | 0.9081 | 0.6820 | 14/20 | 35.0% | 4/20 | 85.0% | {"success": 5} |
| glass | 5/5 | 0.7378 | 0.1447 | 6/20 | 75.0% | 9/20 | 60.0% | {"success": 5} |
| ionosphere | 5/5 | 0.9262 | 0.8993 | 8/20 | 65.0% | 8/20 | 65.0% | {"success": 5} |
| letter | 5/5 | 0.5589 | 0.1161 | 13/20 | 40.0% | 11/20 | 50.0% | {"success": 5} |
| lympho | 5/5 | 0.9862 | 0.8133 | 2/20 | 95.0% | 2/20 | 95.0% | {"success": 5} |
| mnist | 5/5 | 0.8086 | 0.3928 | 13/20 | 40.0% | 10/20 | 55.0% | {"success": 5} |
| musk | 5/5 | 1.0000 | 0.9988 | 8/20 | 65.0% | 8/20 | 65.0% | {"success": 5} |
| optdigits | 5/5 | 0.3639 | 0.0212 | 19/20 | 10.0% | 20/20 | 5.0% | {"success": 5} |
| pendigits | 5/5 | 0.6806 | 0.1179 | 16/20 | 25.0% | 13/20 | 40.0% | {"success": 5} |
| pima | 5/5 | 0.6509 | 0.4889 | 16/20 | 25.0% | 16/20 | 25.0% | {"success": 5} |
| satellite | 5/5 | 0.6963 | 0.6913 | 14/20 | 35.0% | 14/20 | 35.0% | {"success": 5} |
| satimage-2 | 5/5 | 0.9769 | 0.9408 | 14/20 | 35.0% | 8/20 | 65.0% | {"success": 5} |
| shuttle | 5/5 | 0.9962 | 0.8727 | 1/20 | 100.0% | 4/20 | 85.0% | {"success": 5} |
| vertebral | 5/5 | 0.4603 | 0.1266 | 17/20 | 20.0% | 14/20 | 35.0% | {"success": 5} |
| vowels | 5/5 | 0.8334 | 0.2571 | 9/20 | 60.0% | 9/20 | 60.0% | {"success": 5} |
| wbc | 5/5 | 0.8868 | 0.6653 | 14/20 | 35.0% | 8/20 | 65.0% | {"success": 5} |

### Native features, 5% fit

| Dataset | Strict success | AUROC | AP | AUROC rank | AUROC top % | AP rank | AP top % | Statuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 5/5 | 0.7804 | 0.4754 | 14/20 | 35.0% | 13/20 | 40.0% | {"success": 5} |
| cardio | 5/5 | 0.9060 | 0.6542 | 12/20 | 45.0% | 5/20 | 80.0% | {"success": 5} |
| glass | 5/5 | 0.7018 | 0.1023 | 8/20 | 65.0% | 11/20 | 50.0% | {"success": 5} |
| ionosphere | 5/5 | 0.9050 | 0.8524 | 6/20 | 75.0% | 6/20 | 75.0% | {"success": 5} |
| letter | 5/5 | 0.5741 | 0.1313 | 11/20 | 50.0% | 7/20 | 70.0% | {"success": 5} |
| lympho | 5/5 | 0.9793 | 0.8508 | 2/20 | 95.0% | 2/20 | 95.0% | {"success": 5} |
| mnist | 5/5 | 0.8140 | 0.3842 | 14/20 | 35.0% | 8/20 | 65.0% | {"success": 5} |
| musk | 5/5 | 0.9983 | 0.9345 | 7/20 | 70.0% | 6/20 | 75.0% | {"success": 5} |
| optdigits | 5/5 | 0.4443 | 0.0249 | 18/20 | 15.0% | 18/20 | 15.0% | {"success": 5} |
| pendigits | 5/5 | 0.7701 | 0.1710 | 14/20 | 35.0% | 9/20 | 60.0% | {"success": 5} |
| pima | 5/5 | 0.6516 | 0.4929 | 13/20 | 40.0% | 11/20 | 50.0% | {"success": 5} |
| satellite | 5/5 | 0.7518 | 0.7275 | 11/20 | 50.0% | 11/20 | 50.0% | {"success": 5} |
| satimage-2 | 5/5 | 0.9695 | 0.9331 | 12/20 | 45.0% | 6/20 | 75.0% | {"success": 5} |
| shuttle | 5/5 | 0.9967 | 0.8899 | 2/20 | 95.0% | 5/20 | 80.0% | {"success": 5} |
| vertebral | 5/5 | 0.4250 | 0.1141 | 10/20 | 55.0% | 10/20 | 55.0% | {"success": 5} |
| vowels | 5/5 | 0.8456 | 0.2589 | 7/20 | 70.0% | 7/20 | 70.0% | {"success": 5} |
| wbc | 5/5 | 0.9165 | 0.6147 | 7/20 | 70.0% | 7/20 | 70.0% | {"success": 5} |

### PCA(2), 10% fit

| Dataset | Strict success | AUROC | AP | AUROC rank | AUROC top % | AP rank | AP top % | Statuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 5/5 | 0.6401 | 0.3116 | 16/26 | 42.3% | 19/26 | 30.8% | {"success": 5} |
| cardio | 5/5 | 0.8860 | 0.5817 | 16/26 | 42.3% | 13/26 | 53.8% | {"success": 5} |
| glass | 5/5 | 0.8470 | 0.2364 | 5/26 | 84.6% | 6/26 | 80.8% | {"success": 5} |
| ionosphere | 5/5 | 0.7478 | 0.5792 | 5/26 | 84.6% | 6/26 | 80.8% | {"success": 5} |
| letter | 5/5 | 0.6229 | 0.1049 | 6/26 | 80.8% | 4/26 | 88.5% | {"success": 5} |
| lympho | 5/5 | 0.7207 | 0.2885 | 12/26 | 57.7% | 4/26 | 88.5% | {"success": 5} |
| mnist | 5/5 | 0.6640 | 0.2105 | 15/26 | 46.2% | 15/26 | 46.2% | {"success": 5} |
| musk | 5/5 | 0.9797 | 0.8122 | 8/26 | 73.1% | 7/26 | 76.9% | {"success": 5} |
| optdigits | 5/5 | 0.5067 | 0.0307 | 11/26 | 61.5% | 10/26 | 65.4% | {"success": 5} |
| pendigits | 5/5 | 0.8600 | 0.2298 | 15/26 | 46.2% | 7/26 | 76.9% | {"success": 5} |
| pima | 5/5 | 0.6628 | 0.5040 | 12/26 | 57.7% | 12/26 | 57.7% | {"success": 5} |
| satellite | 5/5 | 0.8136 | 0.7728 | 3/26 | 92.3% | 1/26 | 100.0% | {"success": 5} |
| satimage-2 | 5/5 | 0.9721 | 0.8509 | 2/26 | 96.2% | 5/26 | 84.6% | {"success": 5} |
| shuttle | 5/5 | 0.9660 | 0.7471 | 10/26 | 65.4% | 9/26 | 69.2% | {"success": 5} |
| vertebral | 5/5 | 0.4613 | 0.1229 | 8/26 | 73.1% | 12/26 | 57.7% | {"success": 5} |
| vowels | 5/5 | 0.7337 | 0.1234 | 1/26 | 100.0% | 1/26 | 100.0% | {"success": 5} |
| wbc | 5/5 | 0.8799 | 0.6439 | 11/26 | 61.5% | 10/26 | 65.4% | {"success": 5} |

### PCA(2), 5% fit

| Dataset | Strict success | AUROC | AP | AUROC rank | AUROC top % | AP rank | AP top % | Statuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 5/5 | 0.6220 | 0.3307 | 14/26 | 50.0% | 12/26 | 57.7% | {"success": 5} |
| cardio | 5/5 | 0.8666 | 0.5725 | 16/26 | 42.3% | 11/26 | 61.5% | {"success": 5} |
| glass | 5/5 | 0.7799 | 0.1717 | 4/26 | 88.5% | 7/26 | 76.9% | {"success": 5} |
| ionosphere | 5/5 | 0.7147 | 0.5790 | 5/26 | 84.6% | 4/26 | 88.5% | {"success": 5} |
| letter | 5/5 | 0.6014 | 0.1007 | 2/26 | 96.2% | 2/26 | 96.2% | {"success": 5} |
| lympho | 5/5 | 0.4121 | 0.0465 | 24/26 | 11.5% | 23/26 | 15.4% | {"success": 5} |
| mnist | 5/5 | 0.6626 | 0.1920 | 18/26 | 34.6% | 20/26 | 26.9% | {"success": 5} |
| musk | 5/5 | 0.9350 | 0.5974 | 7/26 | 76.9% | 6/26 | 80.8% | {"success": 5} |
| optdigits | 5/5 | 0.6226 | 0.0474 | 1/26 | 100.0% | 1/26 | 100.0% | {"success": 5} |
| pendigits | 5/5 | 0.8256 | 0.1608 | 17/26 | 38.5% | 11/26 | 61.5% | {"success": 5} |
| pima | 5/5 | 0.5903 | 0.4427 | 16/26 | 42.3% | 13/26 | 53.8% | {"success": 5} |
| satellite | 5/5 | 0.8185 | 0.7658 | 1/26 | 100.0% | 1/26 | 100.0% | {"success": 5} |
| satimage-2 | 5/5 | 0.9544 | 0.7986 | 4/26 | 88.5% | 5/26 | 84.6% | {"success": 5} |
| shuttle | 5/5 | 0.9616 | 0.7142 | 4/26 | 88.5% | 5/26 | 84.6% | {"success": 5} |
| vertebral | 5/5 | 0.3853 | 0.1125 | 19/26 | 30.8% | 17/26 | 38.5% | {"success": 5} |
| vowels | 5/5 | 0.7107 | 0.1065 | 1/26 | 100.0% | 1/26 | 100.0% | {"success": 5} |
| wbc | 5/5 | 0.8856 | 0.6450 | 12/26 | 57.7% | 3/26 | 92.3% | {"success": 5} |

## Raw-success sensitivity at the new sizes

These full-coverage leaders use every recorded successful score, including successes over the measured 15-second budget. Their field can differ from the strict field. Available-case scores of incomplete methods are present in the CSV, with their coverage.

| Features | % fit | Qualification | Family | Full coverage methods | Top AUROC method | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| native | 10 | raw recorded | NSA | 9 | NSA/dual | 0.7792 |
| native | 10 | raw recorded | Other | 28 | KDE | 0.8781 |
| native | 10 | strict ≤15 s | NSA | 9 | NSA/dual | 0.7792 |
| native | 10 | strict ≤15 s | Other | 27 | KDE | 0.8781 |
| native | 5 | raw recorded | NSA | 9 | NSA/dual | 0.7900 |
| native | 5 | raw recorded | Other | 28 | KDE | 0.8577 |
| native | 5 | strict ≤15 s | NSA | 9 | NSA/dual | 0.7900 |
| native | 5 | strict ≤15 s | Other | 28 | KDE | 0.8577 |
| pca2 | 10 | raw recorded | NSA | 10 | NSA/dual | 0.7626 |
| pca2 | 10 | raw recorded | Other | 31 | KDE | 0.7769 |
| pca2 | 10 | strict ≤15 s | NSA | 10 | NSA/dual | 0.7626 |
| pca2 | 10 | strict ≤15 s | Other | 30 | KDE | 0.7769 |
| pca2 | 5 | raw recorded | NSA | 10 | NSA/dual | 0.7264 |
| pca2 | 5 | raw recorded | Other | 33 | KDE | 0.7496 |
| pca2 | 5 | strict ≤15 s | NSA | 10 | NSA/dual | 0.7264 |
| pca2 | 5 | strict ≤15 s | Other | 33 | KDE | 0.7496 |

### Raw versus strict fixed-baseline ranks

| Features | % fit | NSA | Raw rank | Raw AUROC | Strict rank | Strict AUROC | Raw six-size eligible | Strict six-size eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 10 | NSA/fixed | 22/22 | 0.2982 | 20/20 | 0.2982 | True | True |
| native | 10 | NSA/variable | 22/22 | 0.4350 | 20/20 | 0.4350 | True | True |
| native | 10 | NSA/coverage | 22/22 | 0.4333 | 20/20 | 0.4333 | True | True |
| native | 10 | NSA/grid | 22/22 | 0.4323 | 20/20 | 0.4323 | False | False |
| native | 10 | NSA/hierarchical | 22/22 | 0.2452 | 20/20 | 0.2452 | True | True |
| native | 10 | NSA/suppressed | 22/22 | 0.4366 | 20/20 | 0.4366 | True | True |
| native | 10 | NSA/dual | 14/22 | 0.7792 | 14/20 | 0.7792 | False | False |
| native | 10 | NSA/annealed | 22/22 | 0.2641 | 20/20 | 0.2641 | True | True |
| native | 10 | NSA/binary/hamming | 20/22 | 0.5024 | 18/20 | 0.5024 | False | False |
| native | 5 | NSA/fixed | 22/22 | 0.2951 | 20/20 | 0.2951 | True | True |
| native | 5 | NSA/variable | 22/22 | 0.3844 | 20/20 | 0.3844 | True | True |
| native | 5 | NSA/coverage | 22/22 | 0.3819 | 20/20 | 0.3819 | True | True |
| native | 5 | NSA/grid | 22/22 | 0.3823 | 20/20 | 0.3823 | False | False |
| native | 5 | NSA/hierarchical | 22/22 | 0.3089 | 20/20 | 0.3089 | True | True |
| native | 5 | NSA/suppressed | 22/22 | 0.3803 | 20/20 | 0.3803 | True | True |
| native | 5 | NSA/dual | 10/22 | 0.7900 | 10/20 | 0.7900 | False | False |
| native | 5 | NSA/annealed | 22/22 | 0.2564 | 20/20 | 0.2564 | True | True |
| native | 5 | NSA/binary/hamming | 19/22 | 0.5036 | 17/20 | 0.5036 | False | False |
| pca2 | 10 | NSA/fixed | 20/27 | 0.6132 | 19/26 | 0.6132 | True | True |
| pca2 | 10 | NSA/variable | 8/27 | 0.7273 | 8/26 | 0.7273 | True | True |
| pca2 | 10 | NSA/coverage | 10/27 | 0.7110 | 9/26 | 0.7110 | True | True |
| pca2 | 10 | NSA/grid | 8/27 | 0.7243 | 8/26 | 0.7243 | True | True |
| pca2 | 10 | NSA/hierarchical | 13/27 | 0.6891 | 12/26 | 0.6891 | True | True |
| pca2 | 10 | NSA/voronoi | 8/27 | 0.7361 | 8/26 | 0.7361 | True | True |
| pca2 | 10 | NSA/deterministic | 18/27 | 0.6278 | 17/26 | 0.6278 | True | True |
| pca2 | 10 | NSA/suppressed | 10/27 | 0.7173 | 9/26 | 0.7173 | True | True |
| pca2 | 10 | NSA/dual | 6/27 | 0.7626 | 6/26 | 0.7626 | True | True |
| pca2 | 10 | NSA/annealed | 18/27 | 0.6241 | 17/26 | 0.6241 | True | True |
| pca2 | 5 | NSA/fixed | 20/27 | 0.5917 | 19/26 | 0.5917 | True | True |
| pca2 | 5 | NSA/variable | 11/27 | 0.6683 | 10/26 | 0.6683 | True | True |
| pca2 | 5 | NSA/coverage | 11/27 | 0.6669 | 10/26 | 0.6669 | True | True |
| pca2 | 5 | NSA/grid | 11/27 | 0.6775 | 10/26 | 0.6775 | True | True |
| pca2 | 5 | NSA/hierarchical | 13/27 | 0.6579 | 12/26 | 0.6579 | True | True |
| pca2 | 5 | NSA/voronoi | 9/27 | 0.6882 | 8/26 | 0.6882 | True | True |
| pca2 | 5 | NSA/deterministic | 18/27 | 0.6066 | 17/26 | 0.6066 | True | True |
| pca2 | 5 | NSA/suppressed | 13/27 | 0.6650 | 12/26 | 0.6650 | True | True |
| pca2 | 5 | NSA/dual | 5/27 | 0.7264 | 5/26 | 0.7264 | True | True |
| pca2 | 5 | NSA/annealed | 20/27 | 0.5966 | 19/26 | 0.5966 | True | True |

## Isolated adaptive-batch sensitivity: AutoEncoder and VAE

This separate sensitivity contains **740 fresh rows** in **370 cells**; the training-batch adapter applied to **540 rows**. It changes only the two declared PyOD training batch sizes to min(32, retained normal rows), keeps their inference batch size at 32, and records observed training forward calls. Other methods and all primary scores are reused by checked hash. The following ranks belong to a newly qualified six-size baseline field; they never overwrite the primary ranking.

### Fixed baseline field membership

| Features | Primary other N | Adaptive other N | Added | Removed |
| --- | --- | --- | --- | --- |
| native | 19 | 19 | — | — |
| pca2 | 25 | 26 | VAE | — |

### AutoEncoder and VAE coverage and scores

An available-case score with less than 85/85 qualified outcomes is descriptive and never receives a complete-field rank.

| Features | % fit | Method | Primary strict | Adaptive strict | Fresh adaptive rows | Adaptive AUROC available | Adaptive AP available | Adaptive statuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 100 | AutoEncoder | 85/85 | 85/85 | 0 | 0.8684 | 0.5702 | {"success": 85} |
| native | 100 | VAE | 85/85 | 85/85 | 0 | 0.8299 | 0.5466 | {"success": 85} |
| native | 75 | AutoEncoder | 85/85 | 85/85 | 0 | 0.8627 | 0.5625 | {"success": 85} |
| native | 75 | VAE | 85/85 | 85/85 | 0 | 0.8234 | 0.5340 | {"success": 85} |
| native | 50 | AutoEncoder | 85/85 | 85/85 | 0 | 0.8517 | 0.5511 | {"success": 85} |
| native | 50 | VAE | 85/85 | 85/85 | 0 | 0.8141 | 0.5253 | {"success": 85} |
| native | 25 | AutoEncoder | 57/85 | 72/85 | 15 | 0.8309 | 0.5239 | {"overbudget_success": 1, "success": 72, "timeout": 12} |
| native | 25 | VAE | 69/85 | 84/85 | 15 | 0.8084 | 0.5115 | {"error": 1, "success": 84} |
| native | 10 | AutoEncoder | 50/85 | 85/85 | 85 | 0.8119 | 0.5071 | {"success": 85} |
| native | 10 | VAE | 50/85 | 85/85 | 85 | 0.8021 | 0.4992 | {"success": 85} |
| native | 5 | AutoEncoder | 0/85 | 85/85 | 85 | 0.7952 | 0.4945 | {"success": 85} |
| native | 5 | VAE | 0/85 | 85/85 | 85 | 0.7863 | 0.4860 | {"success": 85} |
| pca2 | 100 | AutoEncoder | 85/85 | 85/85 | 0 | 0.6830 | 0.3647 | {"success": 85} |
| pca2 | 100 | VAE | 85/85 | 85/85 | 0 | 0.6718 | 0.3697 | {"success": 85} |
| pca2 | 75 | AutoEncoder | 85/85 | 85/85 | 0 | 0.6767 | 0.3495 | {"success": 85} |
| pca2 | 75 | VAE | 85/85 | 85/85 | 0 | 0.6690 | 0.3697 | {"success": 85} |
| pca2 | 50 | AutoEncoder | 85/85 | 85/85 | 0 | 0.6591 | 0.3211 | {"success": 85} |
| pca2 | 50 | VAE | 85/85 | 85/85 | 0 | 0.6623 | 0.3577 | {"success": 85} |
| pca2 | 25 | AutoEncoder | 60/85 | 75/85 | 15 | 0.6531 | 0.3081 | {"overbudget_success": 3, "success": 75, "timeout": 7} |
| pca2 | 25 | VAE | 70/85 | 85/85 | 15 | 0.6453 | 0.3524 | {"success": 85} |
| pca2 | 10 | AutoEncoder | 50/85 | 85/85 | 85 | 0.6306 | 0.3183 | {"success": 85} |
| pca2 | 10 | VAE | 50/85 | 85/85 | 85 | 0.6297 | 0.3295 | {"success": 85} |
| pca2 | 5 | AutoEncoder | 0/85 | 85/85 | 85 | 0.6400 | 0.3151 | {"success": 85} |
| pca2 | 5 | VAE | 0/85 | 85/85 | 85 | 0.6243 | 0.3205 | {"success": 85} |

### NSA/dual scores and primary versus adaptive AUROC/AP ranks

The NSA/dual scores are identical in both columns of this sensitivity; rank or field-size changes reflect the separately trained baselines.

| Features | % fit | Dual strict | Dual AUROC | Dual AP | Primary AUROC rank | Adaptive AUROC rank | Primary AP rank | Adaptive AP rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 100 | 81/85 | 0.7047 | 0.4609 | — | — | — | — |
| native | 75 | 80/85 | 0.7352 | 0.5006 | — | — | — | — |
| native | 50 | 81/85 | 0.7318 | 0.4966 | — | — | — | — |
| native | 25 | 83/85 | 0.7488 | 0.5142 | — | — | — | — |
| native | 10 | 85/85 | 0.7792 | 0.5112 | 14/20 | 14/20 | 7/20 | 7/20 |
| native | 5 | 85/85 | 0.7900 | 0.5066 | 10/20 | 10/20 | 4/20 | 4/20 |
| pca2 | 100 | 85/85 | 0.7172 | 0.4718 | 13/26 | 13/27 | 6/26 | 6/27 |
| pca2 | 75 | 85/85 | 0.7268 | 0.4777 | 12/26 | 12/27 | 6/26 | 6/27 |
| pca2 | 50 | 85/85 | 0.7419 | 0.4801 | 12/26 | 12/27 | 6/26 | 6/27 |
| pca2 | 25 | 85/85 | 0.7603 | 0.4699 | 10/26 | 10/27 | 3/26 | 3/27 |
| pca2 | 10 | 85/85 | 0.7626 | 0.4206 | 6/26 | 6/27 | 6/26 | 6/27 |
| pca2 | 5 | 85/85 | 0.7264 | 0.3755 | 5/26 | 5/27 | 5/26 | 5/27 |

### All-six-eligible NSA adaptive rank changes versus 100%

Each count uses the same adaptive six-size baseline roster in its 100% and reduced panels. Positive rank change means a better ordinal position. NSA configurations without full six-size coverage are excluded from these rank-change counts, though they remain in the 312-row sensitivity CSV.

| Features | % fit | Metric | Eligible NSA | Better | Worse | Same | Improved methods and positions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 10 | AUROC | 6 | 0 | 3 | 3 | — |
| native | 10 | AP | 6 | 0 | 3 | 3 | — |
| native | 5 | AUROC | 6 | 0 | 3 | 3 | — |
| native | 5 | AP | 6 | 0 | 3 | 3 | — |
| pca2 | 10 | AUROC | 10 | 10 | 0 | 0 | NSA/fixed (+3), NSA/variable (+7), NSA/coverage (+7), NSA/grid (+7), NSA/hierarchical (+3), NSA/voronoi (+7), NSA/deterministic (+4), NSA/suppressed (+6), NSA/dual (+7), NSA/annealed (+4) |
| pca2 | 10 | AP | 10 | 2 | 3 | 5 | NSA/variable (+1), NSA/grid (+1) |
| pca2 | 5 | AUROC | 10 | 10 | 0 | 0 | NSA/fixed (+3), NSA/variable (+5), NSA/coverage (+6), NSA/grid (+5), NSA/hierarchical (+3), NSA/voronoi (+7), NSA/deterministic (+4), NSA/suppressed (+3), NSA/dual (+8), NSA/annealed (+2) |
| pca2 | 5 | AP | 10 | 1 | 9 | 0 | NSA/dual (+1) |

### Best complete NSA against the adaptive fixed-field baseline leader

Leaders are selected by their observed macro score separately per metric and panel. The gap is a descriptive difference of complete 17-dataset macro means without a paired interval in this sensitivity.

| Features | % fit | Metric | Top NSA | NSA score | Adaptive rank | Top adaptive other | Other score | NSA − other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 10 | AUROC | NSA/dual | 0.7792 | 14/20 | KDE | 0.8781 | -0.0989 |
| native | 10 | AP | NSA/dual | 0.5112 | 7/20 | KDE | 0.5620 | -0.0508 |
| native | 5 | AUROC | NSA/dual | 0.7900 | 10/20 | KDE | 0.8577 | -0.0677 |
| native | 5 | AP | NSA/dual | 0.5066 | 4/20 | KDE | 0.5237 | -0.0171 |
| pca2 | 10 | AUROC | NSA/dual | 0.7626 | 6/27 | KDE | 0.7769 | -0.0143 |
| pca2 | 10 | AP | NSA/dual | 0.4206 | 6/27 | KDE | 0.4373 | -0.0167 |
| pca2 | 5 | AUROC | NSA/dual | 0.7264 | 5/27 | KDE | 0.7496 | -0.0232 |
| pca2 | 5 | AP | NSA/dual | 0.3755 | 5/27 | KDE | 0.3949 | -0.0194 |

These ranks remain conditional on normal-only initial fitting, refitted PCA where applicable, the measured ≤15-second rule, and query-batch behavior of the 12 audited non-pointwise scorers. Dataset variation and incomplete outcomes remain visible in the primary tables. A separate startup-timing sidecar records possible brief overlap between an independent source/PCA audit and the first eight adaptive worker cells. Exact per-method start times were not logged; all 16 methods in those cells succeeded in under 5.1 seconds measured, below the 15-second qualification threshold. Their runtime values may reflect transient load. The sidecar SHA-256 is `e852b5bfab58eff05697bfc8fbc984607f74eafdef2f17bb7aa1a07f5fe9f5ea`.

## Recovery provenance and interpretation limits

The new 10/5% root has **4** externally recorded watchdog intervention(s). The active monitor mode recorded in its supervision manifest is **execute_verified_recovery**. A read-only monitor observation never creates an outcome row. Any intervention counted here requires a new-root snapshot, hash, ≥600-second raw-file age, and matching no-score timeout row; it is separate from the frozen historical recovery evidence. The historical results and 118 measured overbudget successes were read from the unchanged four-size files after checking their published package hashes.

XGBOD and DevNet need labeled anomalies for their intended supervised training, so they are inapplicable under this protocol; supplying all-zero labels would not create the missing anomaly class. Other failures may arise from geometry, tiny samples, implementation errors, or runtime. MAD is one-dimensional, Voronoi NSA accepts two or three features, and deterministic NSA needs two; binary NSA can fail under the PCA(2) constructor. LUNAR and GAAL variants may synthesize negative examples internally without receiving real anomaly rows.

AutoEncoder and VAE retain the pinned PyOD default training batch size of 32. Their shared deep-learning loader uses drop_last=True; when fewer than 32 normal rows remain, it yields no training batch and the training loop errors before learning. Such errors show a mini-batch constraint, not a need for labeled outliers. Any separate adaptive-batch sensitivity is outside these primary fixed-constructor results and must be labeled separately.

For 12 audited batch-dependent scorers, decision_function can depend on other rows in the mixed test query batch. The normal-only constraint describes benchmark preprocessing and the initial fit call, not every internal calculation during scoring. The inductive-only field removes these 12 configurations for a stricter query-independent comparison.

The frozen constructors include reduced training epochs or ensembles for some expensive methods. Query-batch scoring, the 600/2,000 caps, refitted PCA, one clean-normal selection rule, fixed hyperparameters, and very small five-percent pools limit generalization. Rankings are conditional on coverage and field definition. There are many exploratory comparisons, and the intervals are not multiplicity adjusted. No detector is declared generally superior by this benchmark alone.

## Machine-readable evidence

The `analysis/` directory contains `extreme_retention_method_summary.csv`, `extreme_retention_common_field_ranks.csv`, `extreme_retention_panel_ranks.csv`, `extreme_retention_rank_change.csv`, `extreme_retention_change_vs_100.csv`, `extreme_retention_nsa_vs_baseline.csv`, `extreme_retention_relative_change_vs_100.csv`, `extreme_retention_dataset_nsa_ranks.csv`, `extreme_retention_dataset_inventory.csv`, `extreme_retention_raw_vs_strict.csv`, `extreme_retention_raw_common_ranks.csv`, `extreme_retention_focal_metrics.json`, and the analysis/recovery audit JSON. The companion `plot.py` renders native and PCA(2) PNG summaries from these validated tables using only fully covered points. Historical raw rows and the new 10/5% raw rows remain in separate roots. This report accompanies the public benchmark evidence.

The isolated `adaptive_batch/analysis/` directory additionally contains the 740-row cell comparison, 24 AE/VAE panel summaries, 312 focal NSA AUROC/AP rank rows, and its SHA-validated audit.
