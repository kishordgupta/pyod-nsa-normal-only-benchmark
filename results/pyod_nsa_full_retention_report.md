# Fresh full-roster PyOD NSA benchmark under normal-only training

**Review report for the user. Nothing was posted to GitHub.**

## Scope and evaluation rule

The experiment reruns all 56 predeclared configurations from the tested PyOD checkout: 13 NSA configurations spanning 11 strategy families, and all 43 other registered tabular methods. Four training sizes retain 100%, 75%, 50%, or 25% of a capped pool of normal rows. All four sizes are fresh runs. Every fitted model receives feature rows from the normal class only. The split uses labels to stratify and discard anomalies from the training partition; held-out labels are used only to calculate AUROC and average precision (AP). No anomalous feature row or label array is passed to `model.fit`.

An unsuccessful or inapplicable configuration is recorded with its status and has no scored AUROC/AP pair. Raw successes whose measured fit-plus-score duration exceeds 15 seconds are classified as `overbudget_success` for analysis, while their raw status and scores remain in JSONL. A method with partial coverage is never ranked by its available-case score. All size changes and NSA/other-method comparisons below use only matched successful dataset and seed cells; the matching denominator is always visible.

## Main observations

| Features | % retained | NSA configs with 85 paired runs | Positive mean ΔAUROC | Positive mean ΔAP | AUROC interval above zero | Partial or no pairs |
| --- | --- | --- | --- | --- | --- | --- |
| native | 75 | 7/13 | 1 | 0 | 0 | 6 |
| native | 50 | 7/13 | 0 | 0 | 0 | 6 |
| native | 25 | 6/13 | 0 | 0 | 0 | 7 |
| pca2 | 75 | 10/13 | 9 | 6 | 3 | 3 |
| pca2 | 50 | 10/13 | 10 | 10 | 6 | 3 |
| pca2 | 25 | 10/13 | 10 | 1 | 1 | 3 |

A positive mean counts only a configuration with complete paired coverage at that size. The 95% intervals are descriptive dataset-cluster bootstrap intervals; no multiplicity adjustment was applied. Intervals are not estimated with fewer than five paired datasets. A positive interval therefore does not establish a universal or confirmatory training-size benefit.

### Which complete NSA configurations improved or declined?

| Features | % retained | Complete NSA configs with positive mean ΔAUROC | Complete NSA configs with negative mean ΔAUROC | Unadjusted 95% interval wholly above zero |
| --- | --- | --- | --- | --- |
| native | 75 | hierarchical +0.011 | suppressed -0.023, coverage -0.012, variable -0.011, fixed -0.009, grid -0.009, annealed -0.005 | none |
| native | 50 | none | grid -0.040, variable -0.040, coverage -0.040, suppressed -0.037, hierarchical -0.022, annealed -0.014, fixed -0.014 | none |
| native | 25 | none | suppressed -0.096, variable -0.091, coverage -0.090, annealed -0.037, fixed -0.030, hierarchical -0.026 | none |
| pca2 | 75 | annealed +0.019, voronoi +0.012, deterministic +0.012, suppressed +0.010, fixed +0.010, dual +0.010, coverage +0.007, grid +0.006, variable +0.005 | hierarchical -0.008 | voronoi, dual, annealed |
| pca2 | 50 | deterministic +0.043, annealed +0.040, fixed +0.038, coverage +0.035, suppressed +0.032, voronoi +0.027, dual +0.025, grid +0.022, variable +0.021, hierarchical +0.017 | none | fixed, coverage, voronoi, deterministic, suppressed, dual |
| pca2 | 25 | fixed +0.047, dual +0.043, voronoi +0.039, deterministic +0.039, suppressed +0.038, grid +0.037, variable +0.031, annealed +0.029, coverage +0.026, hierarchical +0.008 | none | dual |

Positive and negative are based on paired mean AUROC versus each configuration's fresh 100% result. Methods with partial or no matched coverage are listed separately in the failure table and are not counted here. AP changes appear in the full family tables below.

### Highest observed methods with complete coverage

| Features | % retained | Highest AUROC NSA, full coverage | Highest AUROC other PyOD, full coverage | Matched NSA − other AUROC |
| --- | --- | --- | --- | --- |
| native | 100 | grid 0.597 | LUNAR 0.906 | -0.309 [-0.449, -0.192] (85/85) |
| native | 75 | grid 0.588 | LUNAR 0.902 | -0.314 [-0.458, -0.190] (85/85) |
| native | 50 | grid 0.557 | LUNAR 0.897 | -0.340 [-0.479, -0.219] (85/85) |
| native | 25 | binary · hamming 0.517 | KDE 0.893 | -0.376 [-0.462, -0.283] (85/85) |
| pca2 | 100 | dual 0.717 | KNN 0.796 | -0.079 [-0.141, -0.028] (85/85) |
| pca2 | 75 | dual 0.727 | KNN 0.799 | -0.072 [-0.127, -0.025] (85/85) |
| pca2 | 50 | dual 0.742 | KNN 0.805 | -0.063 [-0.116, -0.020] (85/85) |
| pca2 | 25 | dual 0.760 | KNN 0.800 | -0.040 [-0.092, -0.002] (85/85) |

Each row selects the highest mean AUROC method **after observing the test results** within each family and training size. It is a descriptive ranking, not a model-selection procedure or an independent estimate of a selected method. The last column uses matched cells for the displayed methods.

### Raw all-success normal-only performance sensitivity

| Features | % retained | Highest raw-success NSA AUROC / AP | Highest raw-success other PyOD AUROC / AP |
| --- | --- | --- | --- |
| native | 100 | grid 0.597 / 0.303 (7/13 full) | LUNAR 0.906 / 0.677 (33/43 full) |
| native | 75 | grid 0.588 / 0.287 (7/13 full) | LUNAR 0.902 / 0.640 (29/43 full) |
| native | 50 | grid 0.557 / 0.251 (8/13 full) | LUNAR 0.897 / 0.628 (31/43 full) |
| native | 25 | binary · hamming 0.517 / 0.145 (7/13 full) | KDE 0.893 / 0.596 (24/43 full) |
| pca2 | 100 | dual 0.717 / 0.472 (10/13 full) | KNN 0.796 / 0.487 (38/43 full) |
| pca2 | 75 | dual 0.727 / 0.478 (10/13 full) | KNN 0.799 / 0.489 (36/43 full) |
| pca2 | 50 | dual 0.742 / 0.480 (10/13 full) | KNN 0.805 / 0.494 (38/43 full) |
| pca2 | 25 | dual 0.760 / 0.470 (10/13 full) | KNN 0.800 / 0.464 (27/43 full) |

This second view uses the unchanged raw successful scores, including successes that finished after 15 measured seconds. It ranks only methods with all 85 raw dataset/seed successes in the panel. The leaders are selected descriptively from held-out outcomes and need not match the strict-budget leaders above. `full_retention_recorded_vs_qualified.csv` contains both raw and qualified AUROC/AP and coverage for **all 56 methods** in all eight panels. The raw scores answer the normal-only detection-performance question without filtering on runtime; strict scores answer the controlled 15-second-budget question.

### Comparison with the strongest fully observed other method

| Features | % retained | Highest full-coverage other AUROC | Highest full-coverage other AP | NSA above AUROC leader on matched cells | NSA above AP leader on matched cells |
| --- | --- | --- | --- | --- | --- |
| native | 100 | LUNAR 0.906 | LUNAR 0.677 | 0/7 point; 0/7 interval > 0 | 0/7 point; 0/7 interval > 0 |
| native | 75 | LUNAR 0.902 | LUNAR 0.640 | 0/7 point; 0/7 interval > 0 | 0/7 point; 0/7 interval > 0 |
| native | 50 | LUNAR 0.897 | LUNAR 0.628 | 0/8 point; 0/8 interval > 0 | 0/8 point; 0/8 interval > 0 |
| native | 25 | KDE 0.893 | KPCA 0.600 | 0/7 point; 0/7 interval > 0 | 0/7 point; 0/7 interval > 0 |
| pca2 | 100 | KNN 0.796 | KNN 0.487 | 0/10 point; 0/10 interval > 0 | 0/10 point; 0/10 interval > 0 |
| pca2 | 75 | KNN 0.799 | KNN 0.489 | 0/10 point; 0/10 interval > 0 | 0/10 point; 0/10 interval > 0 |
| pca2 | 50 | KNN 0.805 | KNN 0.494 | 0/10 point; 0/10 interval > 0 | 0/10 point; 0/10 interval > 0 |
| pca2 | 25 | KNN 0.800 | KDE 0.471 | 0/10 point; 0/10 interval > 0 | 0/10 point; 0/10 interval > 0 |

The AUROC and AP leaders are chosen separately within each feature representation and training size after observing the test results. Counts use NSA configurations with complete coverage and matched cells with the displayed leader. A positive point difference is weaker evidence than a descriptive interval wholly above zero. The many comparisons and selection of a leader from test outcomes prevent a confirmatory superiority claim.

### Dataset concentration of the largest 25% NSA gain

| Features | Largest fully paired NSA gain at 25% | Full 17-dataset mean ΔAUROC | Largest single-dataset influence | Leave-one-out mean range | Positive after exclusion (of 17) |
| --- | --- | --- | --- | --- | --- |
| native | No complete NSA gain | — | — | — | — |
| pca2 | fixed | +0.047 | glass +0.027 | [+0.021, +0.064] | 17 |

This is an exploratory leave-one-dataset-out sensitivity check. The highlighted configuration is selected after seeing its gain; all complete method/size combinations were also processed in `full_retention_leave_one_out.csv`. A strong effect driven by one dataset should be interpreted narrowly.

## Protocol and audit

- **Grid:** 17 ODDS datasets × 5 split seeds (42–46) × two feature representations × four retained-normal fractions × 56 configurations = **38,080** fresh records. All are present before this report is generated.
- **Split:** 60/40 stratified train/test. Every anomaly in the train partition is discarded. The original normal pool is capped at 600 rows; the mixed held-out test set is capped at 2,000, preserving its class mix. Retention uses nested, seed-fixed normal subsets and keeps each subset in the original parent fit-row order.
- **Features:** original dimensions and two-dimensional PCA. StandardScaler and PCA are fitted again using the retained normal rows at each size. The test rows and labels stay fixed within each dataset/seed comparison.
- **Models:** the same fixed constructors and label-free neighbor limits are used at every size. PyOD's `contamination=0.1` is retained for its score cutoff, but no training anomalies are inserted; AUROC/AP use `decision_function` scores. A nominal 15-second Python signal timer begins before lazy model-class import, construction, fitting, and scoring. Optional torch setup happens before that timer starts. Primary analysis requires the recorded fit-plus-score elapsed time, which includes torch setup, to be at most 15 seconds.
- **Audit:** dataset SHA-256 hashes, exact full grid, zero anomalous fit rows, retained normal row identities and order, fixed held-out row identities, metadata, status validity, and transform consistency across methods were checked. The machine-readable audit file records total success and each failure status.

The source checkout is pinned to commit `0d6c901652f82c813da45875873ef4a8560cf049` (tree `57ec5c9ea2800b559019adf0457d7e5b858fc153`). The dataset manifest records each ODDS file's source path and SHA-256. Anomalies in the mixed test set are necessary to measure detection; they never enter fitting.

### Dataset inventory and actual training sizes

Source row, feature, and class counts come from the pinned dataset manifest and were checked against the hashed source arrays. Fit counts come from the audited fresh normal-only records; a range shows variation across the five split seeds. The same counts apply to original features and PCA(2). The 100% pool is capped at 600 normals before the smaller nested subsets are drawn.

| Dataset | Source rows | Features | Normals | Anomalies | Fit normals 100% | 75% | 50% | 25% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrhythmia | 452 | 274 | 386 | 66 | 231 | 173 | 115 | 57 |
| cardio | 1831 | 21 | 1655 | 176 | 600 | 450 | 300 | 150 |
| glass | 214 | 9 | 205 | 9 | 123 | 92 | 61 | 30 |
| ionosphere | 351 | 33 | 225 | 126 | 135 | 101 | 67 | 33 |
| letter | 1600 | 32 | 1500 | 100 | 600 | 450 | 300 | 150 |
| lympho | 148 | 18 | 142 | 6 | 84 | 63 | 42 | 21 |
| mnist | 7603 | 100 | 6903 | 700 | 600 | 450 | 300 | 150 |
| musk | 3062 | 166 | 2965 | 97 | 600 | 450 | 300 | 150 |
| optdigits | 5216 | 64 | 5066 | 150 | 600 | 450 | 300 | 150 |
| pendigits | 6870 | 16 | 6714 | 156 | 600 | 450 | 300 | 150 |
| pima | 768 | 8 | 500 | 268 | 299 | 224 | 149 | 74 |
| satellite | 6435 | 36 | 4399 | 2036 | 600 | 450 | 300 | 150 |
| satimage-2 | 5803 | 36 | 5732 | 71 | 600 | 450 | 300 | 150 |
| shuttle | 49097 | 9 | 45586 | 3511 | 600 | 450 | 300 | 150 |
| vertebral | 240 | 6 | 210 | 30 | 126 | 94 | 63 | 31 |
| vowels | 1456 | 12 | 1406 | 50 | 600 | 450 | 300 | 150 |
| wbc | 378 | 30 | 357 | 21 | 213 | 159 | 106 | 53 |

## Coverage-qualified ranking

| Features | % retained | Group | Fully observed | Top five by AUROC among fully observed |
| --- | --- | --- | --- | --- |
| native | 100 | NSA | 7/13 | grid 0.597, coverage 0.593, variable 0.593, suppressed 0.588, fixed 0.360 |
| native | 100 | Other PyOD | 32/43 | LUNAR 0.906, KPCA 0.902, KNN 0.897, KDE 0.897, ABOD 0.895 |
| native | 75 | NSA | 7/13 | grid 0.588, variable 0.582, coverage 0.581, suppressed 0.566, fixed 0.350 |
| native | 75 | Other PyOD | 29/43 | LUNAR 0.902, KPCA 0.899, KDE 0.895, KNN 0.894, ABOD 0.893 |
| native | 50 | NSA | 8/13 | grid 0.557, coverage 0.553, variable 0.553, suppressed 0.552, binary · hamming 0.529 |
| native | 50 | Other PyOD | 31/43 | LUNAR 0.897, KPCA 0.897, KDE 0.893, KNN 0.890, ABOD 0.884 |
| native | 25 | NSA | 7/13 | binary · hamming 0.517, coverage 0.503, variable 0.503, suppressed 0.492, fixed 0.330 |
| native | 25 | Other PyOD | 23/43 | KDE 0.893, KPCA 0.893, LUNAR 0.887, KNN 0.878, ABOD 0.872 |
| pca2 | 100 | NSA | 10/13 | dual 0.717, voronoi 0.704, variable 0.693, grid 0.689, hierarchical 0.675 |
| pca2 | 100 | Other PyOD | 38/43 | KNN 0.796, LSCP 0.789, KDE 0.789, KPCA 0.786, ABOD 0.785 |
| pca2 | 75 | NSA | 10/13 | dual 0.727, voronoi 0.716, variable 0.698, grid 0.694, suppressed 0.684 |
| pca2 | 75 | Other PyOD | 36/43 | KNN 0.799, KDE 0.791, KPCA 0.789, LSCP 0.789, LOF 0.782 |
| pca2 | 50 | NSA | 10/13 | dual 0.742, voronoi 0.731, variable 0.714, grid 0.710, suppressed 0.706 |
| pca2 | 50 | Other PyOD | 38/43 | KNN 0.805, KPCA 0.798, KDE 0.796, LSCP 0.793, OCSVM 0.792 |
| pca2 | 25 | NSA | 10/13 | dual 0.760, voronoi 0.743, grid 0.726, variable 0.724, suppressed 0.712 |
| pca2 | 25 | Other PyOD | 26/43 | KNN 0.800, KDE 0.795, KPCA 0.794, ABOD 0.789, OCSVM 0.779 |

The complete coverage-qualified AUROC **and AP** rankings are in `full_retention_coverage_qualified_ranks.csv`. These rankings include only methods with 85 successful dataset/seed runs for the given feature representation and size. A batch-dependent detector can still appear in this descriptive ranking; inference semantics are discussed under limitations.

## All NSA strategies: AUROC, AP, and change versus fresh 100% training

Each score is an equal-dataset-weight mean over successful runs, followed by successful runs out of 85. A dagger (†) marks a partial-coverage mean; it cannot be compared directly across sizes if the successful cells differ. The paired tables solve that problem. Binary is one NSA family with three matching rules shown separately.

### Original features

| Method | 100%: AUROC / AP | 75%: AUROC / AP | 50%: AUROC / AP | 25%: AUROC / AP |
| --- | --- | --- | --- | --- |
| fixed | 0.360 (85/85) / 0.129 (85/85) | 0.350 (85/85) / 0.127 (85/85) | 0.346 (85/85) / 0.123 (85/85) | 0.330 (85/85) / 0.121 (85/85) |
| variable | 0.593 (85/85) / 0.304 (85/85) | 0.582 (85/85) / 0.287 (85/85) | 0.553 (85/85) / 0.246 (85/85) | 0.503 (85/85) / 0.205 (85/85) |
| coverage | 0.593 (85/85) / 0.304 (85/85) | 0.581 (85/85) / 0.286 (85/85) | 0.553 (85/85) / 0.246 (85/85) | 0.503 (85/85) / 0.205 (85/85) |
| grid | 0.597 (85/85) / 0.303 (85/85) | 0.588 (85/85) / 0.287 (85/85) | 0.557 (85/85) / 0.251 (85/85) | 0.503† (84/85) / 0.199† (84/85) |
| hierarchical | 0.297 (85/85) / 0.117 (85/85) | 0.308 (85/85) / 0.115 (85/85) | 0.275 (85/85) / 0.094 (85/85) | 0.271 (85/85) / 0.109 (85/85) |
| voronoi | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| deterministic | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| suppressed | 0.588 (85/85) / 0.298 (85/85) | 0.566 (85/85) / 0.271 (85/85) | 0.552 (85/85) / 0.244 (85/85) | 0.492 (85/85) / 0.199 (85/85) |
| dual | 0.705† (81/85) / 0.461† (81/85) | 0.735† (80/85) / 0.501† (80/85) | 0.732† (81/85) / 0.497† (81/85) | 0.749† (83/85) / 0.514† (83/85) |
| annealed | 0.316 (85/85) / 0.112 (85/85) | 0.311 (85/85) / 0.106 (85/85) | 0.302 (85/85) / 0.107 (85/85) | 0.279 (85/85) / 0.099 (85/85) |
| binary · hamming | 0.526† (76/85) / 0.168† (76/85) | 0.523† (83/85) / 0.164† (83/85) | 0.529 (85/85) / 0.157 (85/85) | 0.517 (85/85) / 0.145 (85/85) |
| binary · rcontiguous | 0.535† (59/85) / 0.158† (59/85) | 0.540† (60/85) / 0.150† (60/85) | 0.533† (60/85) / 0.151† (60/85) | 0.522† (63/85) / 0.120† (63/85) |
| binary · rchunk | 0.527† (75/85) / 0.126† (75/85) | 0.529† (75/85) / 0.126† (75/85) | 0.533† (75/85) / 0.121† (75/85) | 0.519† (75/85) / 0.108† (75/85) |

**Paired AUROC change versus 100% retained**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.009 [-0.024, +0.002] (85/85) | -0.014 [-0.039, +0.017] (85/85) | -0.030 [-0.064, +0.010] (85/85) |
| variable | -0.011 [-0.017, -0.005] (85/85) | -0.040 [-0.075, -0.008] (85/85) | -0.091 [-0.175, -0.018] (85/85) |
| coverage | -0.012 [-0.017, -0.006] (85/85) | -0.040 [-0.075, -0.009] (85/85) | -0.090 [-0.177, -0.017] (85/85) |
| grid | -0.009 [-0.021, +0.002] (85/85) | -0.040 [-0.079, -0.010] (85/85) | -0.094 [-0.181, -0.025] (84/85) |
| hierarchical | +0.011 [-0.035, +0.061] (85/85) | -0.022 [-0.066, +0.023] (85/85) | -0.026 [-0.077, +0.023] (85/85) |
| voronoi | — (0/85) | — (0/85) | — (0/85) |
| deterministic | — (0/85) | — (0/85) | — (0/85) |
| suppressed | -0.023 [-0.041, -0.009] (85/85) | -0.037 [-0.067, -0.011] (85/85) | -0.096 [-0.187, -0.023] (85/85) |
| dual | +0.035 [+0.010, +0.065] (80/85) | +0.033 [+0.001, +0.069] (80/85) | +0.055 [+0.018, +0.097] (80/85) |
| annealed | -0.005 [-0.016, +0.007] (85/85) | -0.014 [-0.030, +0.001] (85/85) | -0.037 [-0.075, -0.011] (85/85) |
| binary · hamming | -0.008 [-0.022, +0.004] (76/85) | -0.002 [-0.023, +0.019] (76/85) | -0.015 [-0.037, +0.003] (76/85) |
| binary · rcontiguous | +0.008 [-0.012, +0.029] (59/85) | -0.003 [-0.024, +0.021] (59/85) | -0.017 [-0.044, +0.011] (59/85) |
| binary · rchunk | +0.002 [-0.009, +0.018] (75/85) | +0.006 [-0.008, +0.024] (75/85) | -0.008 [-0.025, +0.009] (75/85) |

**Paired AP change versus 100% retained**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.001 [-0.006, +0.005] (85/85) | -0.005 [-0.013, +0.004] (85/85) | -0.008 [-0.016, 0.000] (85/85) |
| variable | -0.017 [-0.028, -0.007] (85/85) | -0.058 [-0.111, -0.016] (85/85) | -0.098 [-0.192, -0.026] (85/85) |
| coverage | -0.018 [-0.029, -0.010] (85/85) | -0.058 [-0.110, -0.016] (85/85) | -0.099 [-0.191, -0.024] (85/85) |
| grid | -0.016 [-0.029, -0.004] (85/85) | -0.052 [-0.101, -0.012] (85/85) | -0.103 [-0.196, -0.035] (84/85) |
| hierarchical | -0.003 [-0.023, +0.016] (85/85) | -0.024 [-0.045, -0.006] (85/85) | -0.008 [-0.034, +0.014] (85/85) |
| voronoi | — (0/85) | — (0/85) | — (0/85) |
| deterministic | — (0/85) | — (0/85) | — (0/85) |
| suppressed | -0.028 [-0.050, -0.010] (85/85) | -0.054 [-0.108, -0.013] (85/85) | -0.099 [-0.194, -0.025] (85/85) |
| dual | +0.030 [+0.006, +0.069] (80/85) | +0.033 [+0.002, +0.073] (80/85) | +0.055 [+0.013, +0.109] (80/85) |
| annealed | -0.006 [-0.014, 0.000] (85/85) | -0.005 [-0.022, +0.006] (85/85) | -0.013 [-0.023, -0.005] (85/85) |
| binary · hamming | -0.005 [-0.011, +0.002] (76/85) | -0.014 [-0.045, +0.006] (76/85) | -0.025 [-0.068, 0.000] (76/85) |
| binary · rcontiguous | -0.008 [-0.019, +0.002] (59/85) | -0.007 [-0.012, -0.002] (59/85) | -0.027 [-0.066, -0.005] (59/85) |
| binary · rchunk | 0.000 [-0.005, +0.004] (75/85) | -0.005 [-0.014, +0.002] (75/85) | -0.018 [-0.041, -0.002] (75/85) |

### PCA to two dimensions

| Method | 100%: AUROC / AP | 75%: AUROC / AP | 50%: AUROC / AP | 25%: AUROC / AP |
| --- | --- | --- | --- | --- |
| fixed | 0.583 (85/85) / 0.343 (85/85) | 0.592 (85/85) / 0.333 (85/85) | 0.620 (85/85) / 0.356 (85/85) | 0.630 (85/85) / 0.347 (85/85) |
| variable | 0.693 (85/85) / 0.444 (85/85) | 0.698 (85/85) / 0.443 (85/85) | 0.714 (85/85) / 0.447 (85/85) | 0.724 (85/85) / 0.420 (85/85) |
| coverage | 0.669 (85/85) / 0.425 (85/85) | 0.676 (85/85) / 0.431 (85/85) | 0.704 (85/85) / 0.442 (85/85) | 0.695 (85/85) / 0.410 (85/85) |
| grid | 0.689 (85/85) / 0.436 (85/85) | 0.694 (85/85) / 0.438 (85/85) | 0.710 (85/85) / 0.445 (85/85) | 0.726 (85/85) / 0.421 (85/85) |
| hierarchical | 0.675 (85/85) / 0.422 (85/85) | 0.668 (85/85) / 0.409 (85/85) | 0.692 (85/85) / 0.427 (85/85) | 0.684 (85/85) / 0.386 (85/85) |
| voronoi | 0.704 (85/85) / 0.452 (85/85) | 0.716 (85/85) / 0.459 (85/85) | 0.731 (85/85) / 0.455 (85/85) | 0.743 (85/85) / 0.435 (85/85) |
| deterministic | 0.603 (85/85) / 0.359 (85/85) | 0.615 (85/85) / 0.359 (85/85) | 0.646 (85/85) / 0.393 (85/85) | 0.642 (85/85) / 0.345 (85/85) |
| suppressed | 0.674 (85/85) / 0.421 (85/85) | 0.684 (85/85) / 0.429 (85/85) | 0.706 (85/85) / 0.440 (85/85) | 0.712 (85/85) / 0.419 (85/85) |
| dual | 0.717 (85/85) / 0.472 (85/85) | 0.727 (85/85) / 0.478 (85/85) | 0.742 (85/85) / 0.480 (85/85) | 0.760 (85/85) / 0.470 (85/85) |
| annealed | 0.600 (85/85) / 0.367 (85/85) | 0.618 (85/85) / 0.371 (85/85) | 0.640 (85/85) / 0.384 (85/85) | 0.629 (85/85) / 0.340 (85/85) |
| binary · hamming | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| binary · rcontiguous | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| binary · rchunk | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |

**Paired AUROC change versus 100% retained**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| fixed | +0.010 [-0.006, +0.026] (85/85) | +0.038 [+0.007, +0.075] (85/85) | +0.047 [-0.015, +0.118] (85/85) |
| variable | +0.005 [-0.005, +0.015] (85/85) | +0.021 [-0.005, +0.048] (85/85) | +0.031 [-0.024, +0.095] (85/85) |
| coverage | +0.007 [-0.002, +0.016] (85/85) | +0.035 [+0.003, +0.071] (85/85) | +0.026 [-0.031, +0.086] (85/85) |
| grid | +0.006 [-0.005, +0.017] (85/85) | +0.022 [-0.003, +0.050] (85/85) | +0.037 [-0.012, +0.095] (85/85) |
| hierarchical | -0.008 [-0.032, +0.013] (85/85) | +0.017 [-0.007, +0.042] (85/85) | +0.008 [-0.046, +0.059] (85/85) |
| voronoi | +0.012 [+0.003, +0.023] (85/85) | +0.027 [+0.005, +0.055] (85/85) | +0.039 [-0.010, +0.097] (85/85) |
| deterministic | +0.012 [-0.003, +0.027] (85/85) | +0.043 [+0.001, +0.094] (85/85) | +0.039 [-0.025, +0.115] (85/85) |
| suppressed | +0.010 [-0.001, +0.023] (85/85) | +0.032 [+0.011, +0.058] (85/85) | +0.038 [-0.009, +0.093] (85/85) |
| dual | +0.010 [+0.002, +0.017] (85/85) | +0.025 [+0.004, +0.048] (85/85) | +0.043 [+0.001, +0.095] (85/85) |
| annealed | +0.019 [+0.004, +0.039] (85/85) | +0.040 [-0.003, +0.106] (85/85) | +0.029 [-0.039, +0.111] (85/85) |
| binary · hamming | — (0/85) | — (0/85) | — (0/85) |
| binary · rcontiguous | — (0/85) | — (0/85) | — (0/85) |
| binary · rchunk | — (0/85) | — (0/85) | — (0/85) |

**Paired AP change versus 100% retained**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.010 [-0.031, +0.007] (85/85) | +0.014 [-0.021, +0.044] (85/85) | +0.005 [-0.053, +0.063] (85/85) |
| variable | -0.001 [-0.016, +0.014] (85/85) | +0.002 [-0.023, +0.027] (85/85) | -0.024 [-0.078, +0.020] (85/85) |
| coverage | +0.006 [-0.009, +0.019] (85/85) | +0.017 [-0.013, +0.047] (85/85) | -0.015 [-0.068, +0.027] (85/85) |
| grid | +0.002 [-0.013, +0.018] (85/85) | +0.009 [-0.013, +0.033] (85/85) | -0.015 [-0.059, +0.027] (85/85) |
| hierarchical | -0.013 [-0.041, +0.011] (85/85) | +0.005 [-0.029, +0.040] (85/85) | -0.036 [-0.088, +0.006] (85/85) |
| voronoi | +0.007 [-0.005, +0.020] (85/85) | +0.003 [-0.024, +0.034] (85/85) | -0.017 [-0.059, +0.023] (85/85) |
| deterministic | -0.001 [-0.028, +0.025] (85/85) | +0.033 [-0.014, +0.084] (85/85) | -0.014 [-0.080, +0.044] (85/85) |
| suppressed | +0.007 [-0.010, +0.023] (85/85) | +0.019 [-0.003, +0.044] (85/85) | -0.003 [-0.042, +0.032] (85/85) |
| dual | +0.006 [-0.007, +0.020] (85/85) | +0.008 [-0.010, +0.032] (85/85) | -0.002 [-0.031, +0.028] (85/85) |
| annealed | +0.004 [-0.015, +0.025] (85/85) | +0.016 [-0.024, +0.055] (85/85) | -0.027 [-0.090, +0.033] (85/85) |
| binary · hamming | — (0/85) | — (0/85) | — (0/85) |
| binary · rcontiguous | — (0/85) | — (0/85) | — (0/85) |
| binary · rchunk | — (0/85) | — (0/85) | — (0/85) |

## All 43 other PyOD methods

The same normal-only subsets, transforms, timeout, and held-out rows are used for every registered method. The method tables include exclusions and failures through their 0/85 or partial-coverage denominators.

### Original features

| Method | 100%: AUROC / AP | 75%: AUROC / AP | 50%: AUROC / AP | 25%: AUROC / AP |
| --- | --- | --- | --- | --- |
| ECOD | 0.786† (77/85) / 0.474† (77/85) | 0.782 (85/85) / 0.463 (85/85) | 0.778 (85/85) / 0.452 (85/85) | 0.774 (85/85) / 0.440 (85/85) |
| ABOD | 0.895 (85/85) / 0.628 (85/85) | 0.893 (85/85) / 0.612 (85/85) | 0.884 (85/85) / 0.585 (85/85) | 0.872 (85/85) / 0.558 (85/85) |
| COPOD | 0.787 (85/85) / 0.457 (85/85) | 0.783 (85/85) / 0.459 (85/85) | 0.784 (85/85) / 0.455 (85/85) | 0.780 (85/85) / 0.446 (85/85) |
| MAD | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| SOS | 0.587 (85/85) / 0.222 (85/85) | 0.588† (84/85) / 0.228† (84/85) | 0.590 (85/85) / 0.230 (85/85) | 0.588† (82/85) / 0.220† (82/85) |
| QMCD | 0.538 (85/85) / 0.237 (85/85) | 0.526† (84/85) / 0.217† (84/85) | 0.512 (85/85) / 0.198 (85/85) | 0.468 (85/85) / 0.177 (85/85) |
| KDE | 0.897 (85/85) / 0.644 (85/85) | 0.895 (85/85) / 0.631 (85/85) | 0.893 (85/85) / 0.619 (85/85) | 0.893 (85/85) / 0.596 (85/85) |
| Sampling | 0.862 (85/85) / 0.560 (85/85) | 0.867 (85/85) / 0.554 (85/85) | 0.868 (85/85) / 0.555 (85/85) | 0.865 (85/85) / 0.546 (85/85) |
| GMM | 0.865 (85/85) / 0.550 (85/85) | 0.862 (85/85) / 0.544 (85/85) | 0.864 (85/85) / 0.542 (85/85) | 0.861 (85/85) / 0.527 (85/85) |
| PCA | 0.807† (82/85) / 0.530† (82/85) | 0.806† (81/85) / 0.527† (81/85) | 0.799† (80/85) / 0.515† (80/85) | 0.811 (85/85) / 0.523 (85/85) |
| KPCA | 0.902 (85/85) / 0.646 (85/85) | 0.899 (85/85) / 0.630 (85/85) | 0.897 (85/85) / 0.622 (85/85) | 0.893 (85/85) / 0.600 (85/85) |
| MCD | 0.843 (85/85) / 0.470 (85/85) | 0.845 (85/85) / 0.469 (85/85) | 0.844 (85/85) / 0.468 (85/85) | 0.842 (85/85) / 0.462 (85/85) |
| CD | 0.819† (79/85) / 0.472† (79/85) | 0.817† (79/85) / 0.470† (79/85) | 0.817† (75/85) / 0.468† (75/85) | 0.818† (78/85) / 0.460† (78/85) |
| OCSVM | 0.841 (85/85) / 0.567 (85/85) | 0.840 (85/85) / 0.561 (85/85) | 0.840 (85/85) / 0.563 (85/85) | 0.842 (85/85) / 0.556 (85/85) |
| LMDD | 0.759† (60/85) / 0.452† (60/85) | 0.756† (59/85) / 0.445† (59/85) | 0.711† (50/85) / 0.424† (50/85) | 0.752† (53/85) / 0.439† (53/85) |
| LOF | 0.888 (85/85) / 0.579 (85/85) | 0.883 (85/85) / 0.578 (85/85) | 0.874 (85/85) / 0.565 (85/85) | 0.855 (85/85) / 0.548 (85/85) |
| COF | 0.675 (85/85) / 0.343 (85/85) | 0.675 (85/85) / 0.340 (85/85) | 0.673 (85/85) / 0.340 (85/85) | 0.675† (81/85) / 0.341† (81/85) |
| CBLOF | 0.880† (84/85) / 0.570† (84/85) | 0.882† (83/85) / 0.569† (83/85) | 0.881 (85/85) / 0.571 (85/85) | 0.871† (75/85) / 0.552† (75/85) |
| LOCI | 0.677† (30/85) / 0.333† (30/85) | 0.677† (28/85) / 0.322† (28/85) | 0.702† (22/85) / 0.364† (22/85) | 0.713† (24/85) / 0.376† (24/85) |
| HBOS | 0.813 (85/85) / 0.495 (85/85) | 0.813 (85/85) / 0.492 (85/85) | 0.813 (85/85) / 0.486 (85/85) | 0.812† (73/85) / 0.476† (73/85) |
| HDBSCAN | 0.695 (85/85) / 0.239 (85/85) | 0.673 (85/85) / 0.222 (85/85) | 0.641 (85/85) / 0.202 (85/85) | 0.610 (85/85) / 0.170 (85/85) |
| KNN | 0.897 (85/85) / 0.616 (85/85) | 0.894 (85/85) / 0.603 (85/85) | 0.890 (85/85) / 0.594 (85/85) | 0.878 (85/85) / 0.573 (85/85) |
| SOD | 0.755 (85/85) / 0.352 (85/85) | 0.754† (78/85) / 0.339† (78/85) | 0.721† (16/85) / 0.311† (16/85) | 0.749† (60/85) / 0.342† (60/85) |
| ROD | 0.824† (42/85) / 0.511† (42/85) | 0.810† (35/85) / 0.469† (35/85) | 0.792† (31/85) / 0.464† (31/85) | 0.790† (38/85) / 0.476† (38/85) |
| IForest | 0.853 (85/85) / 0.526 (85/85) | 0.853 (85/85) / 0.531 (85/85) | 0.848 (85/85) / 0.516 (85/85) | 0.837 (85/85) / 0.479 (85/85) |
| INNE | 0.873 (85/85) / 0.576 (85/85) | 0.872 (85/85) / 0.569 (85/85) | 0.868 (85/85) / 0.552 (85/85) | 0.863 (85/85) / 0.529 (85/85) |
| DIF | 0.819 (85/85) / 0.443 (85/85) | 0.811 (85/85) / 0.424 (85/85) | 0.808 (85/85) / 0.420 (85/85) | 0.803† (72/85) / 0.397† (72/85) |
| FeatureBagging | 0.889 (85/85) / 0.586 (85/85) | 0.882 (85/85) / 0.577 (85/85) | 0.872 (85/85) / 0.567 (85/85) | 0.854 (85/85) / 0.551 (85/85) |
| LSCP | 0.892 (85/85) / 0.591 (85/85) | 0.887 (85/85) / 0.584 (85/85) | 0.878 (85/85) / 0.576 (85/85) | 0.853† (72/85) / 0.528† (72/85) |
| LODA | 0.743 (85/85) / 0.415 (85/85) | 0.737 (85/85) / 0.399 (85/85) | 0.725 (85/85) / 0.379 (85/85) | 0.722 (85/85) / 0.336 (85/85) |
| SUOD | 0.850 (85/85) / 0.544 (85/85) | 0.845 (85/85) / 0.538 (85/85) | 0.844† (69/85) / 0.533† (69/85) | 0.831† (61/85) / 0.507† (61/85) |
| XGBOD | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| AutoEncoder | 0.868 (85/85) / 0.570 (85/85) | 0.863 (85/85) / 0.562 (85/85) | 0.852 (85/85) / 0.551 (85/85) | 0.856† (57/85) / 0.561† (57/85) |
| VAE | 0.830 (85/85) / 0.547 (85/85) | 0.823 (85/85) / 0.534 (85/85) | 0.814 (85/85) / 0.525 (85/85) | 0.832† (69/85) / 0.544† (69/85) |
| SO_GAAL | 0.529 (85/85) / 0.269 (85/85) | 0.515 (85/85) / 0.235 (85/85) | 0.523 (85/85) / 0.224 (85/85) | 0.526 (85/85) / 0.244 (85/85) |
| MO_GAAL | 0.544 (85/85) / 0.259 (85/85) | 0.516 (85/85) / 0.227 (85/85) | 0.544 (85/85) / 0.234 (85/85) | 0.539 (85/85) / 0.250 (85/85) |
| DeepSVDD | 0.865 (85/85) / 0.591 (85/85) | 0.855 (85/85) / 0.576 (85/85) | 0.854 (85/85) / 0.570 (85/85) | 0.840 (85/85) / 0.543 (85/85) |
| AnoGAN | 0.790 (85/85) / 0.484 (85/85) | 0.785† (83/85) / 0.478† (83/85) | 0.794† (83/85) / 0.489† (83/85) | 0.800† (77/85) / 0.465† (77/85) |
| ALAD | 0.541 (85/85) / 0.147 (85/85) | 0.549 (85/85) / 0.158 (85/85) | 0.549 (85/85) / 0.157 (85/85) | 0.558 (85/85) / 0.163 (85/85) |
| AE1SVM | 0.836 (85/85) / 0.546 (85/85) | 0.829 (85/85) / 0.528 (85/85) | 0.828 (85/85) / 0.529 (85/85) | 0.818† (79/85) / 0.499† (79/85) |
| DevNet | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| RGraph | 0.384† (26/85) / 0.168† (26/85) | 0.471† (21/85) / 0.198† (21/85) | 0.424† (21/85) / 0.213† (21/85) | 0.504† (31/85) / 0.255† (31/85) |
| LUNAR | 0.906 (85/85) / 0.677 (85/85) | 0.902 (85/85) / 0.640 (85/85) | 0.897 (85/85) / 0.628 (85/85) | 0.887 (85/85) / 0.596 (85/85) |

**Paired AUROC change versus 100% retained**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| ECOD | -0.003 [-0.004, -0.002] (77/85) | -0.007 [-0.010, -0.004] (77/85) | -0.011 [-0.016, -0.007] (77/85) |
| ABOD | -0.002 [-0.006, +0.001] (85/85) | -0.011 [-0.018, -0.004] (85/85) | -0.023 [-0.039, -0.008] (85/85) |
| COPOD | -0.004 [-0.008, 0.000] (85/85) | -0.003 [-0.005, -0.001] (85/85) | -0.007 [-0.016, +0.001] (85/85) |
| MAD | — (0/85) | — (0/85) | — (0/85) |
| SOS | +0.001 [-0.002, +0.005] (84/85) | +0.003 [0.000, +0.008] (85/85) | +0.001 [-0.004, +0.009] (82/85) |
| QMCD | -0.011 [-0.023, -0.001] (84/85) | -0.025 [-0.056, +0.012] (85/85) | -0.070 [-0.139, +0.007] (85/85) |
| KDE | -0.002 [-0.005, 0.000] (85/85) | -0.004 [-0.010, +0.003] (85/85) | -0.004 [-0.016, +0.009] (85/85) |
| Sampling | +0.005 [-0.005, +0.015] (85/85) | +0.006 [-0.004, +0.018] (85/85) | +0.003 [-0.008, +0.014] (85/85) |
| GMM | -0.004 [-0.010, 0.000] (85/85) | -0.002 [-0.011, +0.009] (85/85) | -0.004 [-0.016, +0.009] (85/85) |
| PCA | -0.001 [-0.002, -0.001] (81/85) | +0.002 [-0.002, +0.007] (80/85) | +0.004 [-0.003, +0.012] (82/85) |
| KPCA | -0.004 [-0.006, -0.001] (85/85) | -0.006 [-0.010, -0.001] (85/85) | -0.010 [-0.018, -0.002] (85/85) |
| MCD | +0.002 [-0.004, +0.009] (85/85) | +0.002 [-0.010, +0.015] (85/85) | -0.001 [-0.023, +0.021] (85/85) |
| CD | -0.002 [-0.011, +0.007] (79/85) | -0.002 [-0.008, +0.004] (75/85) | -0.001 [-0.017, +0.015] (78/85) |
| OCSVM | -0.001 [-0.003, 0.000] (85/85) | 0.000 [-0.006, +0.004] (85/85) | +0.001 [-0.014, +0.017] (85/85) |
| LMDD | -0.002 [-0.006, +0.001] (59/85) | -0.006 [-0.018, +0.006] (50/85) | -0.008 [-0.020, +0.004] (53/85) |
| LOF | -0.005 [-0.011, +0.001] (85/85) | -0.014 [-0.029, -0.001] (85/85) | -0.033 [-0.060, -0.007] (85/85) |
| COF | +0.001 [-0.001, +0.002] (85/85) | -0.002 [-0.008, +0.002] (85/85) | 0.000 [-0.005, +0.004] (81/85) |
| CBLOF | +0.002 [-0.008, +0.013] (82/85) | +0.001 [-0.004, +0.006] (84/85) | -0.008 [-0.014, -0.002] (74/85) |
| LOCI | +0.004 [-0.005, +0.013] (28/85) | +0.026 [-0.003, +0.073] (22/85) | +0.025 [-0.015, +0.083] (24/85) |
| HBOS | 0.000 [-0.003, +0.003] (85/85) | 0.000 [-0.005, +0.005] (85/85) | -0.002 [-0.010, +0.008] (73/85) |
| HDBSCAN | -0.021 [-0.043, -0.001] (85/85) | -0.053 [-0.096, -0.020] (85/85) | -0.085 [-0.143, -0.034] (85/85) |
| KNN | -0.004 [-0.006, -0.001] (85/85) | -0.008 [-0.014, -0.002] (85/85) | -0.019 [-0.034, -0.007] (85/85) |
| SOD | 0.000 [-0.007, +0.007] (78/85) | -0.007 [-0.027, +0.012] (16/85) | -0.008 [-0.021, +0.002] (60/85) |
| ROD | -0.003 [-0.010, +0.001] (35/85) | -0.004 [-0.016, +0.004] (31/85) | 0.000 [-0.010, +0.008] (35/85) |
| IForest | -0.001 [-0.005, +0.003] (85/85) | -0.005 [-0.011, 0.000] (85/85) | -0.017 [-0.027, -0.007] (85/85) |
| INNE | -0.001 [-0.004, +0.001] (85/85) | -0.005 [-0.012, -0.001] (85/85) | -0.010 [-0.022, 0.000] (85/85) |
| DIF | -0.009 [-0.017, 0.000] (85/85) | -0.011 [-0.020, -0.004] (85/85) | -0.013 [-0.022, -0.004] (72/85) |
| FeatureBagging | -0.007 [-0.014, -0.001] (85/85) | -0.017 [-0.036, -0.002] (85/85) | -0.035 [-0.063, -0.010] (85/85) |
| LSCP | -0.006 [-0.011, -0.001] (85/85) | -0.014 [-0.026, -0.004] (85/85) | -0.033 [-0.054, -0.015] (72/85) |
| LODA | -0.005 [-0.023, +0.010] (85/85) | -0.018 [-0.067, +0.012] (85/85) | -0.020 [-0.072, +0.026] (85/85) |
| SUOD | -0.005 [-0.009, -0.002] (85/85) | -0.008 [-0.016, -0.002] (69/85) | -0.018 [-0.034, -0.007] (61/85) |
| XGBOD | — (0/85) | — (0/85) | — (0/85) |
| AutoEncoder | -0.006 [-0.010, -0.002] (85/85) | -0.017 [-0.029, -0.006] (85/85) | -0.040 [-0.073, -0.011] (57/85) |
| VAE | -0.007 [-0.013, -0.001] (85/85) | -0.016 [-0.034, -0.001] (85/85) | -0.029 [-0.061, -0.005] (69/85) |
| SO_GAAL | -0.014 [-0.085, +0.049] (85/85) | -0.005 [-0.032, +0.021] (85/85) | -0.003 [-0.059, +0.054] (85/85) |
| MO_GAAL | -0.028 [-0.096, +0.029] (85/85) | 0.000 [-0.024, +0.026] (85/85) | -0.005 [-0.063, +0.052] (85/85) |
| DeepSVDD | -0.010 [-0.020, -0.002] (85/85) | -0.011 [-0.022, 0.000] (85/85) | -0.026 [-0.050, -0.004] (85/85) |
| AnoGAN | -0.004 [-0.015, +0.005] (83/85) | +0.004 [-0.006, +0.012] (83/85) | +0.009 [-0.003, +0.021] (77/85) |
| ALAD | +0.008 [-0.011, +0.030] (85/85) | +0.008 [-0.014, +0.034] (85/85) | +0.017 [-0.004, +0.040] (85/85) |
| AE1SVM | -0.008 [-0.014, -0.002] (85/85) | -0.008 [-0.014, -0.003] (85/85) | -0.012 [-0.025, -0.002] (79/85) |
| DevNet | — (0/85) | — (0/85) | — (0/85) |
| RGraph | +0.017 [interval not estimated] (20/85) | +0.044 [interval not estimated] (20/85) | +0.076 [-0.014, +0.178] (26/85) |
| LUNAR | -0.004 [-0.007, -0.001] (85/85) | -0.009 [-0.015, -0.002] (85/85) | -0.019 [-0.030, -0.009] (85/85) |

**Paired AP change versus 100% retained**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| ECOD | -0.010 [-0.018, -0.004] (77/85) | -0.021 [-0.038, -0.010] (77/85) | -0.034 [-0.061, -0.015] (77/85) |
| ABOD | -0.016 [-0.027, -0.006] (85/85) | -0.043 [-0.071, -0.018] (85/85) | -0.070 [-0.123, -0.026] (85/85) |
| COPOD | +0.001 [-0.004, +0.010] (85/85) | -0.002 [-0.008, +0.005] (85/85) | -0.011 [-0.019, -0.003] (85/85) |
| MAD | — (0/85) | — (0/85) | — (0/85) |
| SOS | +0.007 [0.000, +0.018] (84/85) | +0.008 [-0.003, +0.026] (85/85) | -0.002 [-0.006, +0.001] (82/85) |
| QMCD | -0.020 [-0.040, -0.006] (84/85) | -0.039 [-0.090, 0.000] (85/85) | -0.060 [-0.134, +0.002] (85/85) |
| KDE | -0.013 [-0.024, -0.004] (85/85) | -0.025 [-0.044, -0.005] (85/85) | -0.048 [-0.084, -0.013] (85/85) |
| Sampling | -0.005 [-0.033, +0.017] (85/85) | -0.005 [-0.027, +0.014] (85/85) | -0.013 [-0.041, +0.011] (85/85) |
| GMM | -0.005 [-0.012, 0.000] (85/85) | -0.008 [-0.023, +0.011] (85/85) | -0.023 [-0.046, +0.006] (85/85) |
| PCA | -0.006 [-0.009, -0.003] (81/85) | +0.002 [-0.011, +0.016] (80/85) | -0.009 [-0.030, +0.012] (82/85) |
| KPCA | -0.016 [-0.026, -0.008] (85/85) | -0.024 [-0.044, -0.007] (85/85) | -0.046 [-0.085, -0.012] (85/85) |
| MCD | -0.001 [-0.033, +0.020] (85/85) | -0.002 [-0.035, +0.026] (85/85) | -0.008 [-0.048, +0.030] (85/85) |
| CD | -0.001 [-0.014, +0.011] (79/85) | -0.004 [-0.012, +0.003] (75/85) | -0.012 [-0.032, +0.007] (78/85) |
| OCSVM | -0.006 [-0.011, -0.002] (85/85) | -0.004 [-0.016, +0.008] (85/85) | -0.012 [-0.029, +0.006] (85/85) |
| LMDD | -0.007 [-0.017, 0.000] (59/85) | -0.006 [-0.024, +0.011] (50/85) | -0.015 [-0.037, +0.004] (53/85) |
| LOF | 0.000 [-0.009, +0.010] (85/85) | -0.014 [-0.041, +0.014] (85/85) | -0.031 [-0.074, +0.010] (85/85) |
| COF | -0.003 [-0.007, 0.000] (85/85) | -0.002 [-0.011, +0.003] (85/85) | -0.002 [-0.012, +0.009] (81/85) |
| CBLOF | 0.000 [-0.015, +0.020] (82/85) | +0.001 [-0.015, +0.018] (84/85) | -0.018 [-0.040, +0.003] (74/85) |
| LOCI | -0.008 [-0.033, +0.012] (28/85) | +0.038 [-0.009, +0.099] (22/85) | +0.041 [-0.013, +0.122] (24/85) |
| HBOS | -0.003 [-0.007, +0.001] (85/85) | -0.009 [-0.017, -0.003] (85/85) | -0.020 [-0.043, -0.003] (73/85) |
| HDBSCAN | -0.017 [-0.040, +0.008] (85/85) | -0.037 [-0.076, +0.008] (85/85) | -0.069 [-0.112, -0.030] (85/85) |
| KNN | -0.013 [-0.021, -0.005] (85/85) | -0.022 [-0.043, -0.003] (85/85) | -0.043 [-0.080, -0.008] (85/85) |
| SOD | -0.009 [-0.025, +0.003] (78/85) | -0.001 [-0.011, +0.010] (16/85) | -0.015 [-0.027, -0.005] (60/85) |
| ROD | -0.003 [-0.012, +0.009] (35/85) | +0.004 [-0.018, +0.029] (31/85) | +0.003 [-0.018, +0.029] (35/85) |
| IForest | +0.005 [-0.007, +0.023] (85/85) | -0.010 [-0.021, 0.000] (85/85) | -0.047 [-0.075, -0.024] (85/85) |
| INNE | -0.007 [-0.014, 0.000] (85/85) | -0.024 [-0.042, -0.009] (85/85) | -0.047 [-0.072, -0.025] (85/85) |
| DIF | -0.018 [-0.034, -0.005] (85/85) | -0.023 [-0.045, -0.005] (85/85) | -0.043 [-0.079, -0.015] (72/85) |
| FeatureBagging | -0.009 [-0.019, +0.001] (85/85) | -0.019 [-0.053, +0.011] (85/85) | -0.035 [-0.081, +0.008] (85/85) |
| LSCP | -0.007 [-0.016, +0.001] (85/85) | -0.016 [-0.034, +0.004] (85/85) | -0.049 [-0.085, -0.014] (72/85) |
| LODA | -0.016 [-0.033, -0.003] (85/85) | -0.036 [-0.061, -0.014] (85/85) | -0.079 [-0.138, -0.025] (85/85) |
| SUOD | -0.006 [-0.012, 0.000] (85/85) | -0.009 [-0.026, +0.005] (69/85) | -0.036 [-0.074, -0.007] (61/85) |
| XGBOD | — (0/85) | — (0/85) | — (0/85) |
| AutoEncoder | -0.008 [-0.014, -0.001] (85/85) | -0.019 [-0.033, -0.004] (85/85) | -0.045 [-0.086, -0.013] (57/85) |
| VAE | -0.013 [-0.025, -0.003] (85/85) | -0.021 [-0.041, -0.006] (85/85) | -0.039 [-0.065, -0.017] (69/85) |
| SO_GAAL | -0.034 [-0.096, +0.020] (85/85) | -0.046 [-0.118, +0.007] (85/85) | -0.025 [-0.093, +0.034] (85/85) |
| MO_GAAL | -0.032 [-0.084, +0.013] (85/85) | -0.025 [-0.072, +0.022] (85/85) | -0.010 [-0.054, +0.042] (85/85) |
| DeepSVDD | -0.014 [-0.028, -0.001] (85/85) | -0.021 [-0.042, -0.002] (85/85) | -0.048 [-0.089, -0.013] (85/85) |
| AnoGAN | -0.005 [-0.025, +0.009] (83/85) | +0.006 [-0.007, +0.019] (83/85) | -0.022 [-0.061, +0.009] (77/85) |
| ALAD | +0.011 [-0.005, +0.029] (85/85) | +0.010 [0.000, +0.023] (85/85) | +0.016 [+0.001, +0.035] (85/85) |
| AE1SVM | -0.018 [-0.028, -0.009] (85/85) | -0.017 [-0.031, -0.004] (85/85) | -0.024 [-0.043, -0.007] (79/85) |
| DevNet | — (0/85) | — (0/85) | — (0/85) |
| RGraph | -0.001 [interval not estimated] (20/85) | +0.080 [interval not estimated] (20/85) | +0.079 [-0.006, +0.192] (26/85) |
| LUNAR | -0.037 [-0.062, -0.015] (85/85) | -0.048 [-0.080, -0.022] (85/85) | -0.081 [-0.134, -0.034] (85/85) |

### PCA to two dimensions

| Method | 100%: AUROC / AP | 75%: AUROC / AP | 50%: AUROC / AP | 25%: AUROC / AP |
| --- | --- | --- | --- | --- |
| ECOD | 0.620 (85/85) / 0.270 (85/85) | 0.620 (85/85) / 0.272 (85/85) | 0.618 (85/85) / 0.278 (85/85) | 0.620 (85/85) / 0.258 (85/85) |
| ABOD | 0.785 (85/85) / 0.483 (85/85) | 0.781 (85/85) / 0.483 (85/85) | 0.790 (85/85) / 0.486 (85/85) | 0.789 (85/85) / 0.471 (85/85) |
| COPOD | 0.663 (85/85) / 0.312 (85/85) | 0.662 (85/85) / 0.311 (85/85) | 0.669 (85/85) / 0.324 (85/85) | 0.663 (85/85) / 0.300 (85/85) |
| MAD | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| SOS | 0.533 (85/85) / 0.155 (85/85) | 0.540 (85/85) / 0.168 (85/85) | 0.535 (85/85) / 0.165 (85/85) | 0.528 (85/85) / 0.160 (85/85) |
| QMCD | 0.658 (85/85) / 0.338 (85/85) | 0.657 (85/85) / 0.338 (85/85) | 0.661 (85/85) / 0.329 (85/85) | 0.635 (85/85) / 0.283 (85/85) |
| KDE | 0.789 (85/85) / 0.482 (85/85) | 0.791 (85/85) / 0.487 (85/85) | 0.796 (85/85) / 0.491 (85/85) | 0.795 (85/85) / 0.471 (85/85) |
| Sampling | 0.761 (85/85) / 0.433 (85/85) | 0.753 (85/85) / 0.439 (85/85) | 0.743 (85/85) / 0.419 (85/85) | 0.765 (85/85) / 0.433 (85/85) |
| GMM | 0.635 (85/85) / 0.357 (85/85) | 0.633 (85/85) / 0.360 (85/85) | 0.637 (85/85) / 0.376 (85/85) | 0.650 (85/85) / 0.364 (85/85) |
| PCA | 0.666 (85/85) / 0.361 (85/85) | 0.658 (85/85) / 0.358 (85/85) | 0.647 (85/85) / 0.352 (85/85) | 0.650 (85/85) / 0.357 (85/85) |
| KPCA | 0.786 (85/85) / 0.485 (85/85) | 0.789 (85/85) / 0.488 (85/85) | 0.798 (85/85) / 0.492 (85/85) | 0.794 (85/85) / 0.468 (85/85) |
| MCD | 0.706 (85/85) / 0.367 (85/85) | 0.698 (85/85) / 0.353 (85/85) | 0.703 (85/85) / 0.362 (85/85) | 0.707 (85/85) / 0.354 (85/85) |
| CD | 0.583 (85/85) / 0.294 (85/85) | 0.581 (85/85) / 0.295 (85/85) | 0.587 (85/85) / 0.303 (85/85) | 0.591 (85/85) / 0.270 (85/85) |
| OCSVM | 0.780 (85/85) / 0.480 (85/85) | 0.780 (85/85) / 0.484 (85/85) | 0.792 (85/85) / 0.491 (85/85) | 0.779 (85/85) / 0.466 (85/85) |
| LMDD | 0.558 (85/85) / 0.245 (85/85) | 0.572 (85/85) / 0.253 (85/85) | 0.590 (85/85) / 0.263 (85/85) | 0.577† (80/85) / 0.255† (80/85) |
| LOF | 0.778 (85/85) / 0.460 (85/85) | 0.782 (85/85) / 0.454 (85/85) | 0.783 (85/85) / 0.458 (85/85) | 0.765 (85/85) / 0.413 (85/85) |
| COF | 0.614 (85/85) / 0.225 (85/85) | 0.622 (85/85) / 0.232 (85/85) | 0.621 (85/85) / 0.238 (85/85) | 0.626 (85/85) / 0.242 (85/85) |
| CBLOF | 0.781 (85/85) / 0.453 (85/85) | 0.778 (85/85) / 0.451 (85/85) | 0.786 (85/85) / 0.482 (85/85) | 0.774† (64/85) / 0.453† (64/85) |
| LOCI | 0.558† (30/85) / 0.244† (30/85) | 0.552† (28/85) / 0.261† (28/85) | 0.579† (29/85) / 0.262† (29/85) | 0.461† (14/85) / 0.215† (14/85) |
| HBOS | 0.758 (85/85) / 0.390 (85/85) | 0.749 (85/85) / 0.377 (85/85) | 0.759 (85/85) / 0.377 (85/85) | 0.756† (68/85) / 0.355† (68/85) |
| HDBSCAN | 0.716 (85/85) / 0.239 (85/85) | 0.725 (85/85) / 0.226 (85/85) | 0.693 (85/85) / 0.200 (85/85) | 0.634 (85/85) / 0.179 (85/85) |
| KNN | 0.796 (85/85) / 0.487 (85/85) | 0.799 (85/85) / 0.489 (85/85) | 0.805 (85/85) / 0.494 (85/85) | 0.800 (85/85) / 0.464 (85/85) |
| SOD | 0.575 (85/85) / 0.207 (85/85) | 0.564† (82/85) / 0.206† (82/85) | 0.567 (85/85) / 0.215 (85/85) | 0.559† (18/85) / 0.220† (18/85) |
| ROD | 0.665 (85/85) / 0.313 (85/85) | 0.666 (85/85) / 0.320 (85/85) | 0.671 (85/85) / 0.320 (85/85) | 0.670† (72/85) / 0.315† (72/85) |
| IForest | 0.759 (85/85) / 0.402 (85/85) | 0.757 (85/85) / 0.393 (85/85) | 0.760 (85/85) / 0.393 (85/85) | 0.752 (85/85) / 0.344 (85/85) |
| INNE | 0.779 (85/85) / 0.468 (85/85) | 0.781 (85/85) / 0.471 (85/85) | 0.783 (85/85) / 0.474 (85/85) | 0.771 (85/85) / 0.447 (85/85) |
| DIF | 0.758 (85/85) / 0.422 (85/85) | 0.754 (85/85) / 0.432 (85/85) | 0.764 (85/85) / 0.435 (85/85) | 0.760† (74/85) / 0.399† (74/85) |
| FeatureBagging | 0.770 (85/85) / 0.450 (85/85) | 0.769 (85/85) / 0.444 (85/85) | 0.773 (85/85) / 0.450 (85/85) | 0.764 (85/85) / 0.418 (85/85) |
| LSCP | 0.789 (85/85) / 0.476 (85/85) | 0.789 (85/85) / 0.478 (85/85) | 0.793 (85/85) / 0.467 (85/85) | 0.793† (77/85) / 0.452† (77/85) |
| LODA | 0.727 (85/85) / 0.346 (85/85) | 0.724 (85/85) / 0.336 (85/85) | 0.725 (85/85) / 0.348 (85/85) | 0.709 (85/85) / 0.306 (85/85) |
| SUOD | 0.758 (85/85) / 0.398 (85/85) | 0.760† (84/85) / 0.405† (84/85) | 0.762 (85/85) / 0.410 (85/85) | 0.761† (35/85) / 0.378† (35/85) |
| XGBOD | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| AutoEncoder | 0.683 (85/85) / 0.365 (85/85) | 0.677 (85/85) / 0.350 (85/85) | 0.659 (85/85) / 0.321 (85/85) | 0.665† (60/85) / 0.332† (60/85) |
| VAE | 0.672 (85/85) / 0.370 (85/85) | 0.669 (85/85) / 0.370 (85/85) | 0.662 (85/85) / 0.358 (85/85) | 0.646† (70/85) / 0.387† (70/85) |
| SO_GAAL | 0.582 (85/85) / 0.207 (85/85) | 0.551 (85/85) / 0.207 (85/85) | 0.562 (85/85) / 0.215 (85/85) | 0.553 (85/85) / 0.208 (85/85) |
| MO_GAAL | 0.572 (85/85) / 0.234 (85/85) | 0.564 (85/85) / 0.248 (85/85) | 0.564 (85/85) / 0.243 (85/85) | 0.527 (85/85) / 0.226 (85/85) |
| DeepSVDD | 0.737 (85/85) / 0.391 (85/85) | 0.743 (85/85) / 0.389 (85/85) | 0.745 (85/85) / 0.401 (85/85) | 0.724 (85/85) / 0.389 (85/85) |
| AnoGAN | 0.625 (85/85) / 0.280 (85/85) | 0.632 (85/85) / 0.282 (85/85) | 0.629 (85/85) / 0.291 (85/85) | 0.643† (77/85) / 0.281† (77/85) |
| ALAD | 0.505 (85/85) / 0.130 (85/85) | 0.523 (85/85) / 0.134 (85/85) | 0.528 (85/85) / 0.146 (85/85) | 0.519 (85/85) / 0.136 (85/85) |
| AE1SVM | 0.657 (85/85) / 0.336 (85/85) | 0.647 (85/85) / 0.338 (85/85) | 0.653 (85/85) / 0.350 (85/85) | 0.664† (80/85) / 0.345† (80/85) |
| DevNet | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) | — (0/85) / — (0/85) |
| RGraph | 0.436† (35/85) / 0.154† (35/85) | 0.418† (34/85) / 0.154† (34/85) | 0.415† (45/85) / 0.124† (45/85) | 0.403† (32/85) / 0.152† (32/85) |
| LUNAR | 0.775 (85/85) / 0.458 (85/85) | 0.778 (85/85) / 0.463 (85/85) | 0.791 (85/85) / 0.476 (85/85) | 0.778 (85/85) / 0.445 (85/85) |

**Paired AUROC change versus 100% retained**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| ECOD | 0.000 [-0.005, +0.005] (85/85) | -0.003 [-0.014, +0.009] (85/85) | 0.000 [-0.019, +0.020] (85/85) |
| ABOD | -0.005 [-0.019, +0.007] (85/85) | +0.005 [-0.009, +0.022] (85/85) | +0.004 [-0.009, +0.018] (85/85) |
| COPOD | -0.001 [-0.008, +0.005] (85/85) | +0.005 [-0.009, +0.021] (85/85) | 0.000 [-0.018, +0.019] (85/85) |
| MAD | — (0/85) | — (0/85) | — (0/85) |
| SOS | +0.008 [-0.003, +0.023] (85/85) | +0.002 [-0.005, +0.010] (85/85) | -0.005 [-0.017, +0.007] (85/85) |
| QMCD | -0.001 [-0.008, +0.006] (85/85) | +0.003 [-0.014, +0.019] (85/85) | -0.023 [-0.071, +0.020] (85/85) |
| KDE | +0.002 [-0.002, +0.008] (85/85) | +0.007 [-0.002, +0.018] (85/85) | +0.006 [-0.008, +0.022] (85/85) |
| Sampling | -0.008 [-0.022, +0.005] (85/85) | -0.018 [-0.036, +0.002] (85/85) | +0.004 [-0.018, +0.026] (85/85) |
| GMM | -0.002 [-0.009, +0.006] (85/85) | +0.002 [-0.010, +0.017] (85/85) | +0.015 [-0.002, +0.036] (85/85) |
| PCA | -0.007 [-0.028, +0.012] (85/85) | -0.018 [-0.039, +0.001] (85/85) | -0.016 [-0.052, +0.016] (85/85) |
| KPCA | +0.003 [-0.005, +0.010] (85/85) | +0.012 [+0.002, +0.022] (85/85) | +0.008 [-0.014, +0.029] (85/85) |
| MCD | -0.008 [-0.023, +0.004] (85/85) | -0.004 [-0.015, +0.008] (85/85) | +0.001 [-0.018, +0.018] (85/85) |
| CD | -0.002 [-0.013, +0.006] (85/85) | +0.004 [-0.012, +0.020] (85/85) | +0.008 [-0.024, +0.040] (85/85) |
| OCSVM | 0.000 [-0.004, +0.004] (85/85) | +0.012 [+0.001, +0.025] (85/85) | -0.001 [-0.022, +0.017] (85/85) |
| LMDD | +0.014 [-0.003, +0.030] (85/85) | +0.032 [+0.001, +0.075] (85/85) | +0.030 [-0.012, +0.081] (80/85) |
| LOF | +0.004 [-0.009, +0.016] (85/85) | +0.005 [-0.010, +0.018] (85/85) | -0.013 [-0.043, +0.013] (85/85) |
| COF | +0.007 [-0.002, +0.019] (85/85) | +0.007 [-0.006, +0.023] (85/85) | +0.012 [-0.005, +0.031] (85/85) |
| CBLOF | -0.003 [-0.015, +0.012] (85/85) | +0.004 [-0.012, +0.024] (85/85) | -0.004 [-0.026, +0.020] (64/85) |
| LOCI | -0.004 [-0.018, +0.009] (28/85) | +0.019 [-0.005, +0.049] (29/85) | -0.040 [-0.128, +0.014] (14/85) |
| HBOS | -0.009 [-0.019, -0.001] (85/85) | +0.001 [-0.013, +0.015] (85/85) | +0.003 [-0.031, +0.036] (68/85) |
| HDBSCAN | +0.009 [-0.007, +0.026] (85/85) | -0.023 [-0.069, +0.017] (85/85) | -0.082 [-0.146, -0.019] (85/85) |
| KNN | +0.003 [-0.004, +0.014] (85/85) | +0.009 [-0.003, +0.022] (85/85) | +0.005 [-0.011, +0.023] (85/85) |
| SOD | -0.009 [-0.031, +0.011] (82/85) | -0.007 [-0.029, +0.015] (85/85) | -0.002 [-0.108, +0.088] (18/85) |
| ROD | +0.001 [-0.008, +0.014] (85/85) | +0.006 [-0.005, +0.017] (85/85) | +0.007 [-0.021, +0.030] (72/85) |
| IForest | -0.002 [-0.009, +0.004] (85/85) | 0.000 [-0.008, +0.009] (85/85) | -0.007 [-0.018, +0.004] (85/85) |
| INNE | +0.002 [-0.009, +0.013] (85/85) | +0.004 [-0.008, +0.016] (85/85) | -0.008 [-0.035, +0.014] (85/85) |
| DIF | -0.004 [-0.014, +0.006] (85/85) | +0.007 [-0.009, +0.025] (85/85) | +0.001 [-0.017, +0.020] (74/85) |
| FeatureBagging | -0.001 [-0.012, +0.012] (85/85) | +0.003 [-0.013, +0.018] (85/85) | -0.006 [-0.036, +0.021] (85/85) |
| LSCP | 0.000 [-0.011, +0.011] (85/85) | +0.004 [-0.008, +0.016] (85/85) | +0.008 [-0.009, +0.027] (77/85) |
| LODA | -0.003 [-0.012, +0.006] (85/85) | -0.002 [-0.017, +0.012] (85/85) | -0.018 [-0.051, +0.012] (85/85) |
| SUOD | 0.000 [-0.005, +0.005] (84/85) | +0.004 [-0.002, +0.011] (85/85) | -0.010 [-0.026, +0.007] (35/85) |
| XGBOD | — (0/85) | — (0/85) | — (0/85) |
| AutoEncoder | -0.006 [-0.025, +0.013] (85/85) | -0.024 [-0.072, +0.017] (85/85) | -0.031 [-0.060, -0.004] (60/85) |
| VAE | -0.003 [-0.019, +0.012] (85/85) | -0.010 [-0.027, +0.007] (85/85) | -0.043 [-0.079, -0.013] (70/85) |
| SO_GAAL | -0.031 [-0.086, +0.012] (85/85) | -0.020 [-0.058, +0.018] (85/85) | -0.028 [-0.077, +0.017] (85/85) |
| MO_GAAL | -0.007 [-0.040, +0.021] (85/85) | -0.008 [-0.045, +0.026] (85/85) | -0.045 [-0.076, -0.016] (85/85) |
| DeepSVDD | +0.007 [-0.009, +0.024] (85/85) | +0.008 [-0.019, +0.037] (85/85) | -0.013 [-0.055, +0.024] (85/85) |
| AnoGAN | +0.008 [-0.009, +0.025] (85/85) | +0.004 [-0.022, +0.033] (85/85) | +0.021 [-0.015, +0.059] (77/85) |
| ALAD | +0.018 [0.000, +0.040] (85/85) | +0.023 [-0.002, +0.054] (85/85) | +0.014 [-0.007, +0.038] (85/85) |
| AE1SVM | -0.010 [-0.032, +0.008] (85/85) | -0.004 [-0.028, +0.020] (85/85) | +0.005 [-0.017, +0.025] (80/85) |
| DevNet | — (0/85) | — (0/85) | — (0/85) |
| RGraph | -0.018 [-0.046, +0.002] (34/85) | -0.027 [-0.068, +0.006] (35/85) | -0.032 [-0.090, +0.019] (32/85) |
| LUNAR | +0.004 [-0.009, +0.018] (85/85) | +0.017 [+0.001, +0.035] (85/85) | +0.004 [-0.012, +0.023] (85/85) |

**Paired AP change versus 100% retained**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| ECOD | +0.002 [-0.007, +0.015] (85/85) | +0.008 [-0.009, +0.033] (85/85) | -0.011 [-0.032, +0.008] (85/85) |
| ABOD | 0.000 [-0.008, +0.009] (85/85) | +0.003 [-0.009, +0.019] (85/85) | -0.013 [-0.032, +0.007] (85/85) |
| COPOD | 0.000 [-0.011, +0.010] (85/85) | +0.012 [-0.008, +0.034] (85/85) | -0.012 [-0.038, +0.015] (85/85) |
| MAD | — (0/85) | — (0/85) | — (0/85) |
| SOS | +0.013 [+0.001, +0.034] (85/85) | +0.011 [-0.001, +0.029] (85/85) | +0.005 [-0.002, +0.015] (85/85) |
| QMCD | 0.000 [-0.017, +0.019] (85/85) | -0.009 [-0.045, +0.023] (85/85) | -0.055 [-0.131, +0.013] (85/85) |
| KDE | +0.006 [-0.004, +0.017] (85/85) | +0.009 [-0.006, +0.028] (85/85) | -0.011 [-0.033, +0.008] (85/85) |
| Sampling | +0.005 [-0.022, +0.031] (85/85) | -0.014 [-0.039, +0.011] (85/85) | 0.000 [-0.035, +0.034] (85/85) |
| GMM | +0.004 [-0.007, +0.016] (85/85) | +0.019 [0.000, +0.047] (85/85) | +0.008 [-0.012, +0.025] (85/85) |
| PCA | -0.002 [-0.015, +0.011] (85/85) | -0.009 [-0.023, +0.006] (85/85) | -0.004 [-0.023, +0.016] (85/85) |
| KPCA | +0.003 [-0.007, +0.014] (85/85) | +0.006 [-0.009, +0.026] (85/85) | -0.018 [-0.044, +0.006] (85/85) |
| MCD | -0.015 [-0.040, +0.004] (85/85) | -0.005 [-0.025, +0.010] (85/85) | -0.013 [-0.039, +0.008] (85/85) |
| CD | +0.001 [-0.007, +0.010] (85/85) | +0.008 [-0.008, +0.025] (85/85) | -0.024 [-0.068, +0.013] (85/85) |
| OCSVM | +0.003 [-0.004, +0.012] (85/85) | +0.011 [-0.005, +0.030] (85/85) | -0.015 [-0.038, +0.006] (85/85) |
| LMDD | +0.008 [-0.008, +0.027] (85/85) | +0.018 [+0.003, +0.035] (85/85) | +0.016 [-0.012, +0.043] (80/85) |
| LOF | -0.006 [-0.017, +0.005] (85/85) | -0.003 [-0.029, +0.025] (85/85) | -0.048 [-0.088, -0.011] (85/85) |
| COF | +0.007 [0.000, +0.015] (85/85) | +0.013 [-0.006, +0.044] (85/85) | +0.017 [+0.002, +0.037] (85/85) |
| CBLOF | -0.003 [-0.022, +0.017] (85/85) | +0.028 [-0.004, +0.076] (85/85) | 0.000 [-0.028, +0.031] (64/85) |
| LOCI | +0.011 [-0.016, +0.037] (28/85) | +0.014 [+0.004, +0.026] (29/85) | -0.015 [-0.040, +0.015] (14/85) |
| HBOS | -0.013 [-0.032, +0.005] (85/85) | -0.013 [-0.040, +0.013] (85/85) | -0.030 [-0.085, +0.032] (68/85) |
| HDBSCAN | -0.013 [-0.038, +0.011] (85/85) | -0.039 [-0.078, -0.006] (85/85) | -0.060 [-0.107, -0.016] (85/85) |
| KNN | +0.002 [-0.009, +0.013] (85/85) | +0.007 [-0.014, +0.033] (85/85) | -0.023 [-0.053, +0.007] (85/85) |
| SOD | -0.001 [-0.011, +0.007] (82/85) | +0.008 [-0.024, +0.042] (85/85) | +0.017 [-0.042, +0.072] (18/85) |
| ROD | +0.007 [-0.008, +0.027] (85/85) | +0.007 [-0.009, +0.026] (85/85) | +0.006 [-0.021, +0.034] (72/85) |
| IForest | -0.009 [-0.037, +0.012] (85/85) | -0.009 [-0.034, +0.018] (85/85) | -0.058 [-0.107, -0.015] (85/85) |
| INNE | +0.003 [-0.006, +0.014] (85/85) | +0.006 [-0.011, +0.029] (85/85) | -0.022 [-0.048, -0.002] (85/85) |
| DIF | +0.010 [-0.002, +0.026] (85/85) | +0.013 [-0.008, +0.034] (85/85) | -0.028 [-0.071, +0.010] (74/85) |
| FeatureBagging | -0.006 [-0.014, +0.001] (85/85) | 0.000 [-0.019, +0.019] (85/85) | -0.032 [-0.064, -0.001] (85/85) |
| LSCP | +0.003 [-0.011, +0.017] (85/85) | -0.009 [-0.029, +0.011] (85/85) | -0.030 [-0.061, -0.002] (77/85) |
| LODA | -0.009 [-0.022, +0.002] (85/85) | +0.002 [-0.015, +0.018] (85/85) | -0.040 [-0.094, +0.008] (85/85) |
| SUOD | +0.007 [-0.009, +0.025] (84/85) | +0.012 [-0.013, +0.042] (85/85) | -0.036 [-0.072, -0.006] (35/85) |
| XGBOD | — (0/85) | — (0/85) | — (0/85) |
| AutoEncoder | -0.015 [-0.050, +0.027] (85/85) | -0.044 [-0.127, +0.024] (85/85) | -0.076 [-0.165, -0.004] (60/85) |
| VAE | 0.000 [-0.019, +0.018] (85/85) | -0.012 [-0.042, +0.017] (85/85) | -0.023 [-0.056, +0.008] (70/85) |
| SO_GAAL | 0.000 [-0.028, +0.025] (85/85) | +0.009 [-0.014, +0.034] (85/85) | +0.002 [-0.014, +0.018] (85/85) |
| MO_GAAL | +0.014 [-0.007, +0.036] (85/85) | +0.009 [-0.005, +0.024] (85/85) | -0.008 [-0.028, +0.009] (85/85) |
| DeepSVDD | -0.001 [-0.016, +0.015] (85/85) | +0.010 [-0.020, +0.048] (85/85) | -0.002 [-0.039, +0.032] (85/85) |
| AnoGAN | +0.002 [-0.010, +0.013] (85/85) | +0.011 [-0.009, +0.030] (85/85) | -0.004 [-0.034, +0.024] (77/85) |
| ALAD | +0.004 [-0.004, +0.013] (85/85) | +0.016 [-0.004, +0.043] (85/85) | +0.006 [-0.006, +0.023] (85/85) |
| AE1SVM | +0.002 [-0.021, +0.022] (85/85) | +0.014 [-0.012, +0.043] (85/85) | +0.014 [-0.011, +0.046] (80/85) |
| DevNet | — (0/85) | — (0/85) | — (0/85) |
| RGraph | 0.000 [-0.003, +0.002] (34/85) | 0.000 [-0.005, +0.005] (35/85) | -0.002 [-0.012, +0.008] (32/85) |
| LUNAR | +0.006 [-0.005, +0.018] (85/85) | +0.018 [-0.007, +0.055] (85/85) | -0.013 [-0.025, -0.001] (85/85) |

## NSA versus other methods at the same training size

The complete `full_retention_nsa_vs_baseline.csv` contains every 13 × 43 NSA/other comparison at all four sizes and both feature representations: matched AUROC and AP mean differences, 95% dataset-cluster intervals, seed-level and dataset-level win counts, and coverage. Differences within 1e-12 AUROC count as ties. Positive values favor NSA. This is **4,472 comparisons**; reporting only the largest positive ones in prose would select on test outcomes. The two fixed controls below make the comparison readable without changing the full table.

### native: NSA minus IForest at 75%, 50%, and 25% retained

| NSA configuration | 75% retained: NSA − IForest ΔAUROC / ΔAP | 50% retained: NSA − IForest ΔAUROC / ΔAP | 25% retained: NSA − IForest ΔAUROC / ΔAP |
| --- | --- | --- | --- |
| fixed | -0.502 [-0.647, -0.358] (85/85) / AP -0.404 (85/85); AUROC-win datasets 1/17 | -0.502 [-0.650, -0.349] (85/85) / AP -0.393 (85/85); AUROC-win datasets 1/17 | -0.507 [-0.660, -0.349] (85/85) / AP -0.358 (85/85); AUROC-win datasets 1/17 |
| variable | -0.271 [-0.414, -0.143] (85/85) / AP -0.244 (85/85); AUROC-win datasets 4/17 | -0.295 [-0.447, -0.163] (85/85) / AP -0.270 (85/85); AUROC-win datasets 4/17 | -0.334 [-0.491, -0.184] (85/85) / AP -0.273 (85/85); AUROC-win datasets 5/17 |
| coverage | -0.271 [-0.417, -0.144] (85/85) / AP -0.246 (85/85); AUROC-win datasets 4/17 | -0.294 [-0.437, -0.160] (85/85) / AP -0.270 (85/85); AUROC-win datasets 4/17 | -0.334 [-0.486, -0.184] (85/85) / AP -0.273 (85/85); AUROC-win datasets 5/17 |
| grid | -0.265 [-0.417, -0.131] (85/85) / AP -0.244 (85/85); AUROC-win datasets 3/17 | -0.291 [-0.437, -0.152] (85/85) / AP -0.265 (85/85); AUROC-win datasets 4/17 | -0.335 [-0.483, -0.189] (84/85) / AP -0.279 (84/85); AUROC-win datasets 3/17 |
| hierarchical | -0.545 [-0.670, -0.416] (85/85) / AP -0.417 (85/85); AUROC-win datasets 0/17 | -0.573 [-0.696, -0.444] (85/85) / AP -0.422 (85/85); AUROC-win datasets 1/17 | -0.566 [-0.700, -0.424] (85/85) / AP -0.369 (85/85); AUROC-win datasets 1/17 |
| voronoi | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| deterministic | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| suppressed | -0.287 [-0.430, -0.157] (85/85) / AP -0.261 (85/85); AUROC-win datasets 4/17 | -0.296 [-0.442, -0.165] (85/85) / AP -0.272 (85/85); AUROC-win datasets 3/17 | -0.345 [-0.504, -0.188] (85/85) / AP -0.280 (85/85); AUROC-win datasets 3/17 |
| dual | -0.117 [-0.187, -0.045] (80/85) / AP -0.038 (80/85); AUROC-win datasets 5/16 | -0.117 [-0.190, -0.051] (81/85) / AP -0.020 (81/85); AUROC-win datasets 5/17 | -0.088 [-0.170, -0.015] (83/85) / AP +0.037 (83/85); AUROC-win datasets 5/17 |
| annealed | -0.541 [-0.677, -0.406] (85/85) / AP -0.425 (85/85); AUROC-win datasets 1/17 | -0.546 [-0.676, -0.403] (85/85) / AP -0.409 (85/85); AUROC-win datasets 1/17 | -0.558 [-0.691, -0.414] (85/85) / AP -0.380 (85/85); AUROC-win datasets 1/17 |
| binary · hamming | -0.331 [-0.417, -0.242] (83/85) / AP -0.367 (83/85); AUROC-win datasets 1/17 | -0.319 [-0.410, -0.219] (85/85) / AP -0.358 (85/85); AUROC-win datasets 1/17 | -0.320 [-0.409, -0.219] (85/85) / AP -0.334 (85/85); AUROC-win datasets 1/17 |
| binary · rcontiguous | -0.358 [-0.419, -0.291] (60/85) / AP -0.433 (60/85); AUROC-win datasets 0/12 | -0.360 [-0.424, -0.296] (60/85) / AP -0.419 (60/85); AUROC-win datasets 0/12 | -0.334 [-0.395, -0.268] (63/85) / AP -0.345 (63/85); AUROC-win datasets 0/14 |
| binary · rchunk | -0.359 [-0.410, -0.303] (75/85) / AP -0.429 (75/85); AUROC-win datasets 0/15 | -0.350 [-0.405, -0.289] (75/85) / AP -0.418 (75/85); AUROC-win datasets 0/15 | -0.352 [-0.406, -0.292] (75/85) / AP -0.389 (75/85); AUROC-win datasets 0/15 |

### native: NSA minus KNN at 75%, 50%, and 25% retained

| NSA configuration | 75% retained: NSA − KNN ΔAUROC / ΔAP | 50% retained: NSA − KNN ΔAUROC / ΔAP | 25% retained: NSA − KNN ΔAUROC / ΔAP |
| --- | --- | --- | --- |
| fixed | -0.543 [-0.676, -0.415] (85/85) / AP -0.476 (85/85); AUROC-win datasets 1/17 | -0.544 [-0.682, -0.401] (85/85) / AP -0.471 (85/85); AUROC-win datasets 1/17 | -0.548 [-0.697, -0.399] (85/85) / AP -0.452 (85/85); AUROC-win datasets 1/17 |
| variable | -0.312 [-0.447, -0.188] (85/85) / AP -0.317 (85/85); AUROC-win datasets 1/17 | -0.337 [-0.477, -0.213] (85/85) / AP -0.349 (85/85); AUROC-win datasets 1/17 | -0.376 [-0.523, -0.231] (85/85) / AP -0.368 (85/85); AUROC-win datasets 1/17 |
| coverage | -0.312 [-0.453, -0.191] (85/85) / AP -0.318 (85/85); AUROC-win datasets 1/17 | -0.336 [-0.473, -0.211] (85/85) / AP -0.349 (85/85); AUROC-win datasets 1/17 | -0.375 [-0.530, -0.230] (85/85) / AP -0.368 (85/85); AUROC-win datasets 1/17 |
| grid | -0.306 [-0.446, -0.179] (85/85) / AP -0.316 (85/85); AUROC-win datasets 1/17 | -0.333 [-0.474, -0.206] (85/85) / AP -0.343 (85/85); AUROC-win datasets 1/17 | -0.376 [-0.526, -0.232] (84/85) / AP -0.374 (84/85); AUROC-win datasets 2/17 |
| hierarchical | -0.586 [-0.700, -0.464] (85/85) / AP -0.489 (85/85); AUROC-win datasets 0/17 | -0.614 [-0.731, -0.486] (85/85) / AP -0.500 (85/85); AUROC-win datasets 0/17 | -0.607 [-0.738, -0.466] (85/85) / AP -0.464 (85/85); AUROC-win datasets 1/17 |
| voronoi | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| deterministic | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| suppressed | -0.328 [-0.464, -0.207] (85/85) / AP -0.333 (85/85); AUROC-win datasets 1/17 | -0.338 [-0.482, -0.214] (85/85) / AP -0.350 (85/85); AUROC-win datasets 1/17 | -0.386 [-0.540, -0.232] (85/85) / AP -0.374 (85/85); AUROC-win datasets 2/17 |
| dual | -0.156 [-0.226, -0.090] (80/85) / AP -0.106 (80/85); AUROC-win datasets 2/16 | -0.158 [-0.226, -0.097] (81/85) / AP -0.098 (81/85); AUROC-win datasets 1/17 | -0.130 [-0.196, -0.067] (83/85) / AP -0.060 (83/85); AUROC-win datasets 2/17 |
| annealed | -0.582 [-0.706, -0.456] (85/85) / AP -0.498 (85/85); AUROC-win datasets 1/17 | -0.587 [-0.712, -0.454] (85/85) / AP -0.487 (85/85); AUROC-win datasets 1/17 | -0.599 [-0.730, -0.461] (85/85) / AP -0.474 (85/85); AUROC-win datasets 1/17 |
| binary · hamming | -0.371 [-0.451, -0.287] (83/85) / AP -0.439 (83/85); AUROC-win datasets 1/17 | -0.361 [-0.445, -0.265] (85/85) / AP -0.437 (85/85); AUROC-win datasets 1/17 | -0.361 [-0.445, -0.270] (85/85) / AP -0.428 (85/85); AUROC-win datasets 1/17 |
| binary · rcontiguous | -0.391 [-0.443, -0.338] (60/85) / AP -0.504 (60/85); AUROC-win datasets 0/12 | -0.394 [-0.446, -0.340] (60/85) / AP -0.497 (60/85); AUROC-win datasets 0/12 | -0.381 [-0.426, -0.337] (63/85) / AP -0.459 (63/85); AUROC-win datasets 0/14 |
| binary · rchunk | -0.403 [-0.443, -0.364] (75/85) / AP -0.511 (75/85); AUROC-win datasets 0/15 | -0.395 [-0.437, -0.354] (75/85) / AP -0.505 (75/85); AUROC-win datasets 0/15 | -0.395 [-0.437, -0.355] (75/85) / AP -0.494 (75/85); AUROC-win datasets 0/15 |

### pca2: NSA minus IForest at 75%, 50%, and 25% retained

| NSA configuration | 75% retained: NSA − IForest ΔAUROC / ΔAP | 50% retained: NSA − IForest ΔAUROC / ΔAP | 25% retained: NSA − IForest ΔAUROC / ΔAP |
| --- | --- | --- | --- |
| fixed | -0.165 [-0.225, -0.103] (85/85) / AP -0.060 (85/85); AUROC-win datasets 1/17 | -0.140 [-0.191, -0.088] (85/85) / AP -0.037 (85/85); AUROC-win datasets 2/17 | -0.122 [-0.193, -0.060] (85/85) / AP +0.003 (85/85); AUROC-win datasets 3/17 |
| variable | -0.059 [-0.102, -0.022] (85/85) / AP +0.051 (85/85); AUROC-win datasets 6/17 | -0.046 [-0.088, -0.006] (85/85) / AP +0.053 (85/85); AUROC-win datasets 5/17 | -0.028 [-0.085, +0.022] (85/85) / AP +0.076 (85/85); AUROC-win datasets 8/17 |
| coverage | -0.081 [-0.124, -0.041] (85/85) / AP +0.038 (85/85); AUROC-win datasets 3/17 | -0.056 [-0.106, -0.008] (85/85) / AP +0.049 (85/85); AUROC-win datasets 6/17 | -0.057 [-0.111, -0.007] (85/85) / AP +0.066 (85/85); AUROC-win datasets 6/17 |
| grid | -0.063 [-0.105, -0.025] (85/85) / AP +0.046 (85/85); AUROC-win datasets 6/17 | -0.049 [-0.093, -0.008] (85/85) / AP +0.052 (85/85); AUROC-win datasets 5/17 | -0.026 [-0.077, +0.023] (85/85) / AP +0.077 (85/85); AUROC-win datasets 8/17 |
| hierarchical | -0.089 [-0.142, -0.042] (85/85) / AP +0.016 (85/85); AUROC-win datasets 5/17 | -0.068 [-0.121, -0.018] (85/85) / AP +0.033 (85/85); AUROC-win datasets 5/17 | -0.068 [-0.135, -0.012] (85/85) / AP +0.042 (85/85); AUROC-win datasets 7/17 |
| voronoi | -0.041 [-0.075, -0.006] (85/85) / AP +0.066 (85/85); AUROC-win datasets 7/17 | -0.028 [-0.064, +0.006] (85/85) / AP +0.062 (85/85); AUROC-win datasets 6/17 | -0.009 [-0.051, +0.033] (85/85) / AP +0.091 (85/85); AUROC-win datasets 9/17 |
| deterministic | -0.142 [-0.200, -0.086] (85/85) / AP -0.034 (85/85); AUROC-win datasets 2/17 | -0.114 [-0.173, -0.058] (85/85) / AP 0.000 (85/85); AUROC-win datasets 4/17 | -0.110 [-0.181, -0.047] (85/85) / AP +0.001 (85/85); AUROC-win datasets 6/17 |
| suppressed | -0.073 [-0.107, -0.040] (85/85) / AP +0.036 (85/85); AUROC-win datasets 2/17 | -0.053 [-0.089, -0.020] (85/85) / AP +0.047 (85/85); AUROC-win datasets 4/17 | -0.040 [-0.088, +0.006] (85/85) / AP +0.075 (85/85); AUROC-win datasets 7/17 |
| dual | -0.030 [-0.062, -0.002] (85/85) / AP +0.085 (85/85); AUROC-win datasets 6/17 | -0.018 [-0.049, +0.012] (85/85) / AP +0.087 (85/85); AUROC-win datasets 7/17 | +0.008 [-0.028, +0.042] (85/85) / AP +0.126 (85/85); AUROC-win datasets 12/17 |
| annealed | -0.139 [-0.196, -0.085] (85/85) / AP -0.021 (85/85); AUROC-win datasets 2/17 | -0.120 [-0.175, -0.070] (85/85) / AP -0.010 (85/85); AUROC-win datasets 2/17 | -0.123 [-0.186, -0.063] (85/85) / AP -0.004 (85/85); AUROC-win datasets 4/17 |
| binary · hamming | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| binary · rcontiguous | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| binary · rchunk | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |

### pca2: NSA minus KNN at 75%, 50%, and 25% retained

| NSA configuration | 75% retained: NSA − KNN ΔAUROC / ΔAP | 50% retained: NSA − KNN ΔAUROC / ΔAP | 25% retained: NSA − KNN ΔAUROC / ΔAP |
| --- | --- | --- | --- |
| fixed | -0.207 [-0.289, -0.124] (85/85) / AP -0.156 (85/85); AUROC-win datasets 1/17 | -0.185 [-0.257, -0.119] (85/85) / AP -0.138 (85/85); AUROC-win datasets 2/17 | -0.171 [-0.254, -0.094] (85/85) / AP -0.117 (85/85); AUROC-win datasets 3/17 |
| variable | -0.101 [-0.164, -0.045] (85/85) / AP -0.046 (85/85); AUROC-win datasets 3/17 | -0.091 [-0.153, -0.040] (85/85) / AP -0.048 (85/85); AUROC-win datasets 3/17 | -0.076 [-0.140, -0.022] (85/85) / AP -0.044 (85/85); AUROC-win datasets 5/17 |
| coverage | -0.123 [-0.192, -0.062] (85/85) / AP -0.058 (85/85); AUROC-win datasets 2/17 | -0.101 [-0.172, -0.043] (85/85) / AP -0.052 (85/85); AUROC-win datasets 2/17 | -0.105 [-0.172, -0.046] (85/85) / AP -0.054 (85/85); AUROC-win datasets 4/17 |
| grid | -0.105 [-0.169, -0.047] (85/85) / AP -0.051 (85/85); AUROC-win datasets 3/17 | -0.095 [-0.156, -0.042] (85/85) / AP -0.049 (85/85); AUROC-win datasets 3/17 | -0.074 [-0.135, -0.021] (85/85) / AP -0.044 (85/85); AUROC-win datasets 5/17 |
| hierarchical | -0.132 [-0.202, -0.065] (85/85) / AP -0.080 (85/85); AUROC-win datasets 3/17 | -0.113 [-0.181, -0.051] (85/85) / AP -0.068 (85/85); AUROC-win datasets 2/17 | -0.117 [-0.191, -0.050] (85/85) / AP -0.079 (85/85); AUROC-win datasets 5/17 |
| voronoi | -0.083 [-0.141, -0.032] (85/85) / AP -0.030 (85/85); AUROC-win datasets 3/17 | -0.074 [-0.125, -0.031] (85/85) / AP -0.039 (85/85); AUROC-win datasets 4/17 | -0.057 [-0.102, -0.016] (85/85) / AP -0.030 (85/85); AUROC-win datasets 5/17 |
| deterministic | -0.184 [-0.261, -0.108] (85/85) / AP -0.131 (85/85); AUROC-win datasets 2/17 | -0.159 [-0.230, -0.092] (85/85) / AP -0.101 (85/85); AUROC-win datasets 2/17 | -0.158 [-0.242, -0.082] (85/85) / AP -0.119 (85/85); AUROC-win datasets 3/17 |
| suppressed | -0.115 [-0.174, -0.065] (85/85) / AP -0.060 (85/85); AUROC-win datasets 1/17 | -0.099 [-0.151, -0.053] (85/85) / AP -0.054 (85/85); AUROC-win datasets 0/17 | -0.088 [-0.146, -0.038] (85/85) / AP -0.046 (85/85); AUROC-win datasets 3/17 |
| dual | -0.072 [-0.127, -0.025] (85/85) / AP -0.011 (85/85); AUROC-win datasets 2/17 | -0.063 [-0.116, -0.020] (85/85) / AP -0.014 (85/85); AUROC-win datasets 3/17 | -0.040 [-0.092, -0.002] (85/85) / AP +0.005 (85/85); AUROC-win datasets 6/17 |
| annealed | -0.181 [-0.260, -0.104] (85/85) / AP -0.118 (85/85); AUROC-win datasets 2/17 | -0.165 [-0.234, -0.098] (85/85) / AP -0.111 (85/85); AUROC-win datasets 1/17 | -0.172 [-0.251, -0.099] (85/85) / AP -0.124 (85/85); AUROC-win datasets 1/17 |
| binary · hamming | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| binary · rcontiguous | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |
| binary · rchunk | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 | — (0/85) / AP — (0/85); AUROC-win datasets 0/0 |

A positive same-size difference is different from a positive **training-size interaction**. The latter is `(NSA reduced − NSA 100%) − (other reduced − other 100%)` and says who changed more, not who has the higher absolute AUROC. All **3,354** four-fit matched interactions with AUROC/AP and coverage are in `full_retention_interaction.csv`.

## Dataset-level diagnostic view

This table shows paired AUROC change at 25% retained versus the fresh 100% run for four fixed methods. Each cell is the mean over matched successful seeds within that dataset, with matched seeds out of five. The complete `full_retention_dataset_changes.csv` has every method, fraction, dataset, and status.

| Features | Dataset | Dual NSA ΔAUROC | Variable NSA ΔAUROC | IForest ΔAUROC | KNN ΔAUROC |
| --- | --- | --- | --- | --- | --- |
| native | arrhythmia | +0.218 (5/5) | -0.152 (5/5) | -0.002 (5/5) | +0.007 (5/5) |
| native | cardio | +0.119 (5/5) | -0.127 (5/5) | -0.018 (5/5) | -0.008 (5/5) |
| native | glass | +0.041 (5/5) | +0.159 (5/5) | -0.046 (5/5) | -0.087 (5/5) |
| native | ionosphere | +0.030 (5/5) | -0.154 (5/5) | -0.059 (5/5) | -0.036 (5/5) |
| native | letter | +0.066 (5/5) | +0.025 (5/5) | -0.007 (5/5) | -0.066 (5/5) |
| native | lympho | +0.055 (5/5) | -0.467 (5/5) | -0.014 (5/5) | +0.016 (5/5) |
| native | mnist | — (0/5) | -0.070 (5/5) | +0.003 (5/5) | -0.021 (5/5) |
| native | musk | 0.000 (5/5) | 0.000 (5/5) | -0.061 (5/5) | 0.000 (5/5) |
| native | optdigits | -0.033 (5/5) | +0.025 (5/5) | +0.022 (5/5) | -0.063 (5/5) |
| native | pendigits | +0.031 (5/5) | +0.025 (5/5) | -0.011 (5/5) | -0.021 (5/5) |
| native | pima | +0.060 (5/5) | +0.032 (5/5) | -0.002 (5/5) | +0.004 (5/5) |
| native | satellite | +0.006 (5/5) | -0.021 (5/5) | -0.004 (5/5) | -0.007 (5/5) |
| native | satimage-2 | +0.006 (5/5) | -0.022 (5/5) | -0.001 (5/5) | 0.000 (5/5) |
| native | shuttle | 0.000 (5/5) | -0.504 (5/5) | -0.005 (5/5) | -0.001 (5/5) |
| native | vertebral | -0.071 (5/5) | -0.076 (5/5) | -0.033 (5/5) | -0.008 (5/5) |
| native | vowels | +0.233 (5/5) | -0.002 (5/5) | -0.030 (5/5) | -0.025 (5/5) |
| native | wbc | +0.119 (5/5) | -0.210 (5/5) | -0.013 (5/5) | -0.009 (5/5) |
| pca2 | arrhythmia | +0.075 (5/5) | +0.075 (5/5) | +0.033 (5/5) | +0.023 (5/5) |
| pca2 | cardio | -0.015 (5/5) | -0.015 (5/5) | -0.010 (5/5) | -0.010 (5/5) |
| pca2 | glass | +0.150 (5/5) | +0.154 (5/5) | +0.007 (5/5) | -0.046 (5/5) |
| pca2 | ionosphere | +0.050 (5/5) | +0.070 (5/5) | -0.029 (5/5) | -0.049 (5/5) |
| pca2 | letter | +0.142 (5/5) | +0.136 (5/5) | +0.003 (5/5) | -0.019 (5/5) |
| pca2 | lympho | -0.119 (5/5) | -0.128 (5/5) | -0.031 (5/5) | +0.048 (5/5) |
| pca2 | mnist | +0.024 (5/5) | +0.041 (5/5) | -0.002 (5/5) | +0.032 (5/5) |
| pca2 | musk | -0.006 (5/5) | -0.005 (5/5) | -0.015 (5/5) | -0.001 (5/5) |
| pca2 | optdigits | -0.050 (5/5) | 0.000 (5/5) | +0.041 (5/5) | +0.055 (5/5) |
| pca2 | pendigits | -0.006 (5/5) | -0.007 (5/5) | -0.034 (5/5) | -0.008 (5/5) |
| pca2 | pima | +0.026 (5/5) | +0.026 (5/5) | +0.006 (5/5) | +0.008 (5/5) |
| pca2 | satellite | +0.001 (5/5) | -0.021 (5/5) | -0.019 (5/5) | -0.009 (5/5) |
| pca2 | satimage-2 | +0.040 (5/5) | -0.237 (5/5) | -0.022 (5/5) | -0.010 (5/5) |
| pca2 | shuttle | +0.005 (5/5) | -0.002 (5/5) | -0.016 (5/5) | -0.015 (5/5) |
| pca2 | vertebral | +0.051 (5/5) | +0.059 (5/5) | +0.032 (5/5) | +0.100 (5/5) |
| pca2 | vowels | +0.350 (5/5) | +0.379 (5/5) | -0.043 (5/5) | -0.019 (5/5) |
| pca2 | wbc | +0.015 (5/5) | +0.006 (5/5) | -0.027 (5/5) | -0.001 (5/5) |

## Methods excluded or failing under this protocol

**XGBOD and DevNet require labeled outliers for their intended supervised training.** They are included in all 56-method denominators as `requires_labeled_outliers` and are not assigned scores. Their Python `fit` methods may be callable with all-zero labels; that is not the intended outlier-discrimination training and is not evidence that they work under the stated protocol. LUNAR, SO_GAAL, and MO_GAAL may synthesize negatives from normal data; no real anomalous row is provided to them.

MAD requires one feature and is inapplicable to both tested feature panels. NSA Voronoi requires 2 or 3 features; deterministic requires exactly 2. Some binary matching configurations can fail on two PCA features because the bit signature is too short for their construction. Actual outcomes appear below.

### Measured-time qualification of raw successes

The raw JSONL has **32,510** successes. After the strict measured-time rule, **32,392** remain qualified and **118** are marked `overbudget_success` for analysis. Every excluded row retains its original raw `success` status, AUROC, and AP in the JSONL. The longest measured successful run was **250.751s** (ROD, optdigits, seed 43, native, 25% retained).

| Method | Raw successes over 15s | Longest measured seconds | By representation | By retained % |
| --- | --- | --- | --- | --- |
| SOD | 55 | 65.270 | {"native": 40, "pca2": 15} | {"25": 20, "50": 30, "75": 5} |
| DIF | 24 | 24.468 | {"native": 13, "pca2": 11} | {"25": 24} |
| HBOS | 9 | 21.908 | {"native": 2, "pca2": 7} | {"25": 9} |
| ECOD | 8 | 16.474 | {"native": 8} | {"100": 8} |
| ROD | 8 | 250.751 | {"native": 3, "pca2": 5} | {"25": 8} |
| LOCI | 5 | 56.925 | {"native": 2, "pca2": 3} | {"25": 5} |
| AutoEncoder | 4 | 20.701 | {"native": 1, "pca2": 3} | {"25": 4} |
| CBLOF | 4 | 21.191 | {"pca2": 4} | {"25": 4} |
| SUOD | 1 | 31.196 | {"native": 1} | {"25": 1} |

**All-success and strict-budget scores for affected method panels**

| Features | % retained | Method | Overbudget raw successes | Raw all-success AUROC / AP | Strict ≤15s AUROC / AP |
| --- | --- | --- | --- | --- | --- |
| native | 100 | ECOD | 8 | 0.785 / 0.473 (85/85) | 0.786 / 0.474 (77/85) |
| native | 75 | SOD | 4 | 0.756 / 0.346 (82/85) | 0.754 / 0.339 (78/85) |
| native | 50 | SOD | 30 | 0.752 / 0.358 (46/85) | 0.721 / 0.311 (16/85) |
| native | 25 | AutoEncoder | 1 | 0.858 / 0.563 (58/85) | 0.856 / 0.561 (57/85) |
| native | 25 | DIF | 13 | 0.806 / 0.397 (85/85) | 0.803 / 0.397 (72/85) |
| native | 25 | HBOS | 2 | 0.812 / 0.475 (75/85) | 0.812 / 0.476 (73/85) |
| native | 25 | LOCI | 2 | 0.716 / 0.378 (26/85) | 0.713 / 0.376 (24/85) |
| native | 25 | ROD | 3 | 0.789 / 0.442 (41/85) | 0.790 / 0.476 (38/85) |
| native | 25 | SOD | 6 | 0.748 / 0.343 (66/85) | 0.749 / 0.342 (60/85) |
| native | 25 | SUOD | 1 | 0.831 / 0.507 (62/85) | 0.831 / 0.507 (61/85) |
| pca2 | 75 | SOD | 1 | 0.565 / 0.206 (83/85) | 0.564 / 0.206 (82/85) |
| pca2 | 25 | AutoEncoder | 3 | 0.664 / 0.332 (63/85) | 0.665 / 0.332 (60/85) |
| pca2 | 25 | CBLOF | 4 | 0.774 / 0.451 (68/85) | 0.774 / 0.453 (64/85) |
| pca2 | 25 | DIF | 11 | 0.758 / 0.394 (85/85) | 0.760 / 0.399 (74/85) |
| pca2 | 25 | HBOS | 7 | 0.761 / 0.358 (75/85) | 0.756 / 0.355 (68/85) |
| pca2 | 25 | LOCI | 3 | 0.530 / 0.256 (17/85) | 0.461 / 0.215 (14/85) |
| pca2 | 25 | ROD | 5 | 0.670 / 0.314 (77/85) | 0.670 / 0.315 (72/85) |
| pca2 | 25 | SOD | 14 | 0.599 / 0.248 (32/85) | 0.559 / 0.220 (18/85) |

Both score columns are descriptive equal-dataset-weight available-case means, with successful dataset/seed rows out of 85. The raw column includes valid normal-only fits that finished after the measured budget; the strict column qualifies runs by time. These two columns may use different successful cells, so paired comparisons in the primary tables remain the controlled comparison.

The measured rule uses the **recorded fit seconds plus score seconds > 15.0**. The 15-second target was fixed before the sweep; the strict measured-time qualification was applied after discovering that the nominal Python signal timer could be ignored in native callbacks. This is a post-hoc enforcement decision, reported alongside the unfiltered raw-success sensitivity above. Overbudget scores remain valid normal-only outcomes; they are excluded from every strict-budget primary method mean, rank, paired change, same-size comparison, and interaction. The row-level `full_retention_overbudget_success.csv` records both raw and analysis status and the preserved raw scores. `full_retention_recorded_vs_qualified.csv` shows the descriptive available-case mean sensitivity for every method and panel; differing successful cell sets limit direct interpretation of those mean shifts.

One timeout fired between AUROC and AP calculation, leaving an intermediate AUROC in its raw JSONL row. It is excluded from every score and comparison; `full_retention_partial_metric_timeouts.csv` identifies the 1 affected row(s).

### External watchdog interventions

An external watchdog intervened in **6** stalled LOCI cell(s) only after the raw file had not advanced for at least 600 seconds. Before each intervention, it saved and hashed the partial raw JSONL and log; the final grid contains a matching LOCI `timeout` row with null scores and an `ExternalWatchdogTimeout:` reason. The row is a recorded failed attempt, not an inferred AUROC or AP. The runner then resumes the remaining methods without rerunning completed records.

| Intervention ID | Dataset | Seed | Features | % retained | Raw age before action (s) | Signal |
| --- | --- | --- | --- | --- | --- | --- |
| 20261012T212828933736Z_letter_native_seed46_retain25_pid85334 | letter | 46 | native | 25 | 913.1 | SIGTERM |
| 20261012T213003272305Z_mnist_native_seed46_retain25_pid85463 | mnist | 46 | native | 25 | 839.4 | SIGTERM |
| 20261012T213030279209Z_satimage-2_native_seed46_retain25_pid85532 | satimage-2 | 46 | native | 25 | 820.7 | SIGTERM |
| 20261012T214135192368Z_satellite_pca2_seed42_retain25_pid86775 | satellite | 42 | pca2 | 25 | 626.3 | SIGTERM |
| 20261012T214429169321Z_vowels_pca2_seed42_retain25_pid87040 | vowels | 42 | pca2 | 25 | 627.1 | SIGTERM |
| 20261012T215037874354Z_optdigits_pca2_seed43_retain25_pid87436 | optdigits | 43 | pca2 | 25 | 641.3 | SIGTERM |

`watchdog_intervention_audit.json` verifies unique event IDs, the ≥600-second age rule, snapshot hashes, the appended LOCI row hash, and matching final raw rows. The reproducibility bundle includes the watchdog script, intervention JSON, and before snapshots for review. These events do not contribute a score to the primary or raw all-success performance tables.

### Process restart provenance

Two 25%-retention PCA(2) cells have two identical logged runner commands: glass/seed 43 resumed after 36 committed records, and pendigits/seed 43 resumed after 30. Each later command completed its cell to 56 method records without rewriting the committed prefix. Separate sidecars in `provenance/` hash both invocation-log segments, the preserved raw prefix, and the final raw/log files; `process_restart_audit.json` verifies both. The saved logs do not establish the first process exit codes or identify which methods caused the processes to end. No score or failure status was imputed from either restart.

| Features | % retained | Group | Successful runs | Fully observed methods | Partial | Zero success | Other outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 100 | NSA | 886/1105 | 7 | 4 | 2 | error 49, requires_2d 85, requires_2d_or_3d 85 |
| native | 100 | Other PyOD | 3200/3655 | 32 | 8 | 3 | error 10, overbudget_success 8, requires_1d 85, requires_labeled_outliers 170, timeout 182 |
| native | 75 | NSA | 893/1105 | 7 | 4 | 2 | error 42, requires_2d 85, requires_2d_or_3d 85 |
| native | 75 | Other PyOD | 3180/3655 | 29 | 11 | 3 | build_error 1, error 10, overbudget_success 4, requires_1d 85, requires_labeled_outliers 170, timeout 205 |
| native | 50 | NSA | 896/1105 | 8 | 3 | 2 | error 39, requires_2d 85, requires_2d_or_3d 85 |
| native | 50 | Other PyOD | 3082/3655 | 31 | 9 | 3 | error 7, overbudget_success 30, requires_1d 85, requires_labeled_outliers 170, timeout 281 |
| native | 25 | NSA | 900/1105 | 7 | 4 | 2 | error 34, requires_2d 85, requires_2d_or_3d 85, timeout 1 |
| native | 25 | Other PyOD | 3037/3655 | 23 | 17 | 3 | error 48, overbudget_success 28, requires_1d 85, requires_labeled_outliers 170, timeout 287 |
| pca2 | 100 | NSA | 850/1105 | 10 | 0 | 3 | error 255 |
| pca2 | 100 | Other PyOD | 3295/3655 | 38 | 2 | 3 | requires_1d 85, requires_labeled_outliers 170, timeout 105 |
| pca2 | 75 | NSA | 850/1105 | 10 | 0 | 3 | error 255 |
| pca2 | 75 | Other PyOD | 3288/3655 | 36 | 4 | 3 | error 2, overbudget_success 1, requires_1d 85, requires_labeled_outliers 170, timeout 109 |
| pca2 | 50 | NSA | 850/1105 | 10 | 0 | 3 | error 255 |
| pca2 | 50 | Other PyOD | 3304/3655 | 38 | 2 | 3 | requires_1d 85, requires_labeled_outliers 170, timeout 96 |
| pca2 | 25 | NSA | 850/1105 | 10 | 0 | 3 | error 255 |
| pca2 | 25 | Other PyOD | 3031/3655 | 26 | 14 | 3 | error 40, overbudget_success 47, requires_1d 85, requires_labeled_outliers 170, timeout 282 |

| Features | Method | Analysis statuses at sizes with incomplete coverage |
| --- | --- | --- |
| native | grid | 25%: success 84, timeout 1 |
| native | voronoi | 100%: requires_2d_or_3d 85; 75%: requires_2d_or_3d 85; 50%: requires_2d_or_3d 85; 25%: requires_2d_or_3d 85 |
| native | deterministic | 100%: requires_2d 85; 75%: requires_2d 85; 50%: requires_2d 85; 25%: requires_2d 85 |
| native | dual | 100%: error 4, success 81; 75%: error 5, success 80; 50%: error 4, success 81; 25%: error 2, success 83 |
| native | binary · hamming | 100%: error 9, success 76; 75%: error 2, success 83 |
| native | binary · rcontiguous | 100%: error 26, success 59; 75%: error 25, success 60; 50%: error 25, success 60; 25%: error 22, success 63 |
| native | binary · rchunk | 100%: error 10, success 75; 75%: error 10, success 75; 50%: error 10, success 75; 25%: error 10, success 75 |
| native | ECOD | 100%: overbudget_success 8, success 77 |
| native | MAD | 100%: requires_1d 85; 75%: requires_1d 85; 50%: requires_1d 85; 25%: requires_1d 85 |
| native | SOS | 75%: success 84, timeout 1; 25%: error 2, success 82, timeout 1 |
| native | QMCD | 75%: success 84, timeout 1 |
| native | PCA | 100%: error 3, success 82; 75%: error 4, success 81; 50%: error 5, success 80 |
| native | CD | 100%: error 6, success 79; 75%: error 5, success 79, timeout 1; 50%: error 1, success 75, timeout 9; 25%: error 4, success 78, timeout 3 |
| native | LMDD | 100%: success 60, timeout 25; 75%: success 59, timeout 26; 50%: success 50, timeout 35; 25%: success 53, timeout 32 |
| native | COF | 25%: success 81, timeout 4 |
| native | CBLOF | 100%: error 1, success 84; 75%: build_error 1, success 83, timeout 1; 25%: success 75, timeout 10 |
| native | LOCI | 100%: success 30, timeout 55; 75%: success 28, timeout 57; 50%: success 22, timeout 63; 25%: overbudget_success 2, success 24, timeout 59 |
| native | HBOS | 25%: overbudget_success 2, success 73, timeout 10 |
| native | SOD | 75%: overbudget_success 4, success 78, timeout 3; 50%: overbudget_success 30, success 16, timeout 39; 25%: overbudget_success 6, success 60, timeout 19 |
| native | ROD | 100%: success 42, timeout 43; 75%: error 1, success 35, timeout 49; 50%: success 31, timeout 54; 25%: overbudget_success 3, success 38, timeout 44 |
| native | DIF | 25%: overbudget_success 13, success 72 |
| native | LSCP | 25%: error 5, success 72, timeout 8 |
| native | SUOD | 50%: success 69, timeout 16; 25%: overbudget_success 1, success 61, timeout 23 |
| native | XGBOD | 100%: requires_labeled_outliers 85; 75%: requires_labeled_outliers 85; 50%: requires_labeled_outliers 85; 25%: requires_labeled_outliers 85 |
| native | AutoEncoder | 25%: error 15, overbudget_success 1, success 57, timeout 12 |
| native | VAE | 25%: error 16, success 69 |
| native | AnoGAN | 75%: success 83, timeout 2; 50%: success 83, timeout 2; 25%: success 77, timeout 8 |
| native | AE1SVM | 25%: error 6, success 79 |
| native | DevNet | 100%: requires_labeled_outliers 85; 75%: requires_labeled_outliers 85; 50%: requires_labeled_outliers 85; 25%: requires_labeled_outliers 85 |
| native | RGraph | 100%: success 26, timeout 59; 75%: success 21, timeout 64; 50%: error 1, success 21, timeout 63; 25%: success 31, timeout 54 |
| pca2 | binary · hamming | 100%: error 85; 75%: error 85; 50%: error 85; 25%: error 85 |
| pca2 | binary · rcontiguous | 100%: error 85; 75%: error 85; 50%: error 85; 25%: error 85 |
| pca2 | binary · rchunk | 100%: error 85; 75%: error 85; 50%: error 85; 25%: error 85 |
| pca2 | MAD | 100%: requires_1d 85; 75%: requires_1d 85; 50%: requires_1d 85; 25%: requires_1d 85 |
| pca2 | LMDD | 25%: success 80, timeout 5 |
| pca2 | CBLOF | 25%: overbudget_success 4, success 64, timeout 17 |
| pca2 | LOCI | 100%: success 30, timeout 55; 75%: success 28, timeout 57; 50%: success 29, timeout 56; 25%: overbudget_success 3, success 14, timeout 68 |
| pca2 | HBOS | 25%: overbudget_success 7, success 68, timeout 10 |
| pca2 | SOD | 75%: overbudget_success 1, success 82, timeout 2; 25%: overbudget_success 14, success 18, timeout 53 |
| pca2 | ROD | 25%: overbudget_success 5, success 72, timeout 8 |
| pca2 | DIF | 25%: overbudget_success 11, success 74 |
| pca2 | LSCP | 25%: error 5, success 77, timeout 3 |
| pca2 | SUOD | 75%: error 1, success 84; 25%: success 35, timeout 50 |
| pca2 | XGBOD | 100%: requires_labeled_outliers 85; 75%: requires_labeled_outliers 85; 50%: requires_labeled_outliers 85; 25%: requires_labeled_outliers 85 |
| pca2 | AutoEncoder | 25%: error 15, overbudget_success 3, success 60, timeout 7 |
| pca2 | VAE | 25%: error 15, success 70 |
| pca2 | AnoGAN | 25%: success 77, timeout 8 |
| pca2 | AE1SVM | 25%: error 5, success 80 |
| pca2 | DevNet | 100%: requires_labeled_outliers 85; 75%: requires_labeled_outliers 85; 50%: requires_labeled_outliers 85; 25%: requires_labeled_outliers 85 |
| pca2 | RGraph | 100%: success 35, timeout 50; 75%: error 1, success 34, timeout 50; 50%: success 45, timeout 40; 25%: success 32, timeout 53 |

No error, timeout, geometry exclusion, or overbudget success was replaced with AUROC/AP zero. A 0/85 method is unranked. A partial-coverage result can be inspected in the full status and matched-pair tables but cannot be generalized to the omitted cells.

## Interpretation limits

- The 600-normal training and 2,000-row test caps make this a controlled CPU screen, not a full-data replication of the original papers. In small datasets, 25% retained can leave only a few dozen normals. Five split seeds do not create five independent datasets; the bootstrap resamples whole datasets and retains the seeds inside each.
- The nominal 15-second Python signal timer includes lazy model-class import and construction, but optional torch import and seed setup occur before it. The recorded fit time includes that setup. Native callbacks may not stop when the signal is raised; some run logs show Numba/llvmlite callbacks ignoring a `FitTimeout` and later returning a raw success. Those logs do not establish the cause of every overbudget result. The strict **measured end-to-end ≤15s** qualification was introduced after observing this behavior and removes such overbudget successes from primary comparisons; both raw and qualified results are reported. A timeout or overbudget success establishes only a budget outcome in this environment, not that the detector is intrinsically unusable. Methods run in a fixed roster order within each cell. Lazy import caching, execution order, and concurrent machine load can affect which fits reach the 15-second timeout. The recorded primary scores and paired comparisons remain the results for this run.
- Deep-model compute overrides differ from paper defaults: DIF uses 10 ensembles instead of 50; DeepSVDD 50 epochs instead of 100; AnoGAN 50 training and 5 query epochs instead of 500 and 20; ALAD 50 instead of 200 epochs; LUNAR 50 instead of 200 epochs. Fixed constructors are not optimized for each dataset or training size.
- PCA is refitted on the retained normal set at every size. A change in PCA(2) performance can reflect altered scaling or axes as well as altered detector learning. Two PCA dimensions can discard anomaly-relevant directions.
- COF, LOCI, SOS, SOD, ECOD, COPOD, RGraph, SUOD, LMDD, DIF, CD, and ROD have batch-dependent scoring behavior in this evaluation. Their test scores can depend on other test rows; they are not strictly pointwise novelty predictors like NSA. The analysis CSV marks these methods. The active model factory metadata under-flags LMDD, DIF, CD, and ROD as inductive; the analysis-level correction is recorded in `batch_scoring_audit.json`. The active model factory was not changed.
- AP depends on anomaly prevalence in the held-out set. Interpret AP changes within matched splits. All intervals are descriptive and unadjusted for thousands of method, fraction, representation, and metric comparisons. Coverage-qualified ranks are descriptive and cannot establish a generally best method.

## Reproducibility files

The raw JSONL includes success and non-success records for every cell. `full_retention_analysis_audit.json` records the validation and overall status counts. `batch_scoring_audit.json` records scoring-semantics metadata corrections. `watchdog_intervention_audit.json` links any external LOCI intervention to its snapshots and final timeout row. `process_restart_audit.json` links the glass and pendigits PCA(2)/seed 43/25% restart sidecars to their raw and log evidence. The overbudget row list and recorded-versus-qualified sensitivity tables preserve the distinction between raw and derived status. The method summary, matched changes, dataset changes, complete NSA/other pairwise table, interactions, coverage-qualified ranks, leave-one-dataset-out results, and 17-dataset source/training-size inventory are separate CSVs. The dataset manifest pins every source file; the run manifest and source scripts pin the tested checkout, constructors, and code. No PR response or other external post was made.
