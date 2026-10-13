# NSA normal-training-size experiment — report

**Review copy. Nothing in this report has been posted to GitHub.**

## Question and answer

This experiment tests whether reducing the number of normal training examples improves the 11 NSA strategy families in [PyOD PR #747](https://github.com/yzhao062/pyod/pull/747). The binary family has three matching rules, so the tables contain 13 distinct NSA configurations. Anomalous feature rows are held out for evaluation; labels stratify the split and identify training anomalies to discard, but are never passed to model fitting. A positive paired difference means that the reduced normal-only training set scored higher AUROC or AP than the matched 100% run.

In native, no NSA configuration had positive paired mean AUROC changes at all three reduced sizes with complete coverage.
In native, no configuration had an exploratory 95% interval entirely above zero at every reduced size.
In native, no NSA configuration had higher same-size mean AUROC than IForest at all three reduced sizes with complete paired coverage.
In pca2, fixed, variable, coverage, grid, voronoi, deterministic, suppressed, dual, annealed had positive paired mean AUROC changes at all three reduced sizes with complete coverage.
In pca2, no configuration had an exploratory 95% interval entirely above zero at every reduced size.
In pca2, no NSA configuration had higher same-size mean AUROC than IForest at all three reduced sizes with complete paired coverage.

These are descriptive outcomes of the tested datasets, splits, implementations, and fixed settings. A positive result for one configuration does not establish general NSA superiority; the same-size baseline comparisons below test that separate claim.

### Notable measured results

- **PCA2 dual:** mean AUROC was 0.714 (51/51) at 100% and 0.762 (51/51) at 25% retained. Its matched AUROC change was +0.048 [+0.008, +0.106] (51/51); matched AP change was -0.000 [-0.035, +0.031] (51/51).
- **PCA2 voronoi:** mean AUROC was 0.703 (51/51) at 100% and 0.750 (51/51) at 25% retained. Its matched AUROC change was +0.047 [+0.001, +0.109] (51/51); matched AP change was -0.010 [-0.048, +0.027] (51/51).
- **AP qualification:** neither PCA2 dual nor PCA2 Voronoi improved average precision at 25% retained in the paired mean; both AP bootstrap intervals include zero. An AUROC gain therefore does not translate into an observed AP gain here.
- **Same-size controls for PCA2 dual at 25% retained:** NSA − IForest was +0.012 [-0.026, +0.048] (51/51); NSA − KNN was -0.038 [-0.086, +0.001] (51/51). KNN's mean AUROC was 0.800 (51/51). A gain over dual's own 100% result alone does not show that it outperforms these controls.
- **Dataset concentration:** the two largest positive PCA2 dual changes at 25% retained were vowels +0.418 (3/3 seeds), letter +0.175 (3/3 seeds). These are inspected contributors to the macro mean, not preselected success cases or an independent validation. Because scaling and PCA are refit, representation drift may contribute.

## Experimental design

- **Data:** 17 ODDS `.mat` datasets bundled with PyOD's [benchmark collection](https://github.com/yzhao062/pyod/blob/master/notebooks/benchmark.py). Dataset hashes and original source links are in `dataset_manifest.json`; the tested PyOD tree is `57ec5c9ea2800b559019adf0457d7e5b858fc153`.
- **Splits:** seeds 42, 43, and 44; each uses a 60/40 stratified train/test split. Every anomalous row in the training partition is discarded before preprocessing or fitting. The original fit pool is capped at 600 normals; the mixed held-out test set is capped at 2,000 rows with its class mix preserved.
- **Retention:** 100%, 75%, 50%, and 25% of the original capped normal fit pool. One seed-fixed random permutation of the parent normal indices supplies nested reduced subsets. The order of selected rows follows the parent fit set. Thus '25% retained' also means '75% removed'.
- **Features:** original features and PCA to two dimensions. StandardScaler and PCA are refit using only the normals retained at each size. Every model is fitted with `fit(X)`; no anomaly label or anomaly feature row is passed to the model. The test indices and labels remain fixed across sizes.
- **Methods:** all 11 PR NSA families (13 configurations because binary has Hamming, r-contiguous, and r-chunk rules) plus five predeclared controls: IForest, ECOD, OCSVM, KNN, ABOD. Constructor settings match the previous experiment; each method has a 15-second wall-time limit.
- **Evaluation:** held-out mixed-class AUROC and average precision (AP). No test label is used for fitting or hyperparameter selection. Run failures and geometry exclusions stay in the denominator and never become zero metric scores.

The analysis checks the complete 1,836-row 100% reference grid and 5,508-row reduced grid, source dataset hashes, zero fitted outliers, exact retained normal index sets and order, unchanged test indices, and status/metric validity. The reference uses the prior full-roster seed-42 runs and predeclared selected-method seed-43/44 runs. An independent numerical audit verified that the 100% native and PCA2 arrays match the previous preprocessing in every dataset/seed cell.

## How to read the statistics

A score cell below gives the equal-dataset-weight mean among successful runs followed by successful dataset/seed runs out of 51. Different score cells can use different successful subsets; use the paired-change tables for direct size comparisons. Each paired difference is computed within the same dataset and seed, averaged over available seeds in each dataset, and then averaged equally over datasets. Brackets are descriptive 95% dataset-cluster bootstrap intervals (5,000 resamples). They are not adjusted for the many strategy, size, representation, and baseline comparisons.

### At-a-glance NSA AUROC changes

| Features | % retained | Complete NSA configs | Positive mean | Negative mean | Intervals > 0 | Intervals < 0 | Partial | No pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 75 | 7/13 | 1 | 6 | 0 | 2 | 4 | 2 |
| native | 50 | 7/13 | 0 | 7 | 0 | 2 | 4 | 2 |
| native | 25 | 7/13 | 0 | 7 | 0 | 5 | 4 | 2 |
| pca2 | 75 | 10/13 | 9 | 1 | 1 | 0 | 0 | 3 |
| pca2 | 50 | 10/13 | 10 | 0 | 6 | 0 | 0 | 3 |
| pca2 | 25 | 10/13 | 10 | 0 | 2 | 0 | 0 | 3 |

Only complete-coverage configurations contribute to the positive/negative and interval counts in this overview. Partial and zero-pair configurations are counted separately.

## Results by NSA family

### Original features

The first table shows mean AUROC / AP at each retained size. The next two tables compare each reduced size with its matched 100% run.

| Method | 100% retained: AUROC / AP | 75% retained: AUROC / AP | 50% retained: AUROC / AP | 25% retained: AUROC / AP |
| --- | --- | --- | --- | --- |
| fixed | 0.360 (51/51) / 0.128 (51/51) | 0.347 (51/51) / 0.125 (51/51) | 0.347 (51/51) / 0.123 (51/51) | 0.336 (51/51) / 0.118 (51/51) |
| variable | 0.599 (51/51) / 0.304 (51/51) | 0.591 (51/51) / 0.291 (51/51) | 0.569 (51/51) / 0.246 (51/51) | 0.525 (51/51) / 0.213 (51/51) |
| coverage | 0.599 (51/51) / 0.304 (51/51) | 0.590 (51/51) / 0.289 (51/51) | 0.569 (51/51) / 0.246 (51/51) | 0.524 (51/51) / 0.212 (51/51) |
| grid | 0.610 (51/51) / 0.311 (51/51) | 0.599 (51/51) / 0.298 (51/51) | 0.579 (51/51) / 0.256 (51/51) | 0.527 (51/51) / 0.207 (51/51) |
| hierarchical | 0.290 (51/51) / 0.105 (51/51) | 0.316 (51/51) / 0.110 (51/51) | 0.270 (51/51) / 0.095 (51/51) | 0.273 (51/51) / 0.107 (51/51) |
| voronoi | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) |
| deterministic | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) |
| suppressed | 0.597 (51/51) / 0.300 (51/51) | 0.568 (51/51) / 0.270 (51/51) | 0.564 (51/51) / 0.248 (51/51) | 0.511 (51/51) / 0.202 (51/51) |
| dual | 0.710 (49/51) / 0.461 (49/51) | 0.729 (48/51) / 0.482 (48/51) | 0.722 (48/51) / 0.493 (48/51) | 0.737 (49/51) / 0.498 (49/51) |
| annealed | 0.314 (51/51) / 0.110 (51/51) | 0.307 (51/51) / 0.105 (51/51) | 0.300 (51/51) / 0.102 (51/51) | 0.286 (51/51) / 0.091 (51/51) |
| binary · hamming | 0.528 (45/51) / 0.150 (45/51) | 0.528 (49/51) / 0.160 (49/51) | 0.545 (51/51) / 0.160 (51/51) | 0.531 (51/51) / 0.146 (51/51) |
| binary · rcontiguous | 0.557 (36/51) / 0.166 (36/51) | 0.571 (36/51) / 0.159 (36/51) | 0.563 (36/51) / 0.152 (36/51) | 0.530 (39/51) / 0.117 (39/51) |
| binary · rchunk | 0.536 (45/51) / 0.125 (45/51) | 0.544 (45/51) / 0.124 (45/51) | 0.550 (45/51) / 0.122 (45/51) | 0.522 (45/51) / 0.107 (45/51) |

**Paired AUROC change versus 100% training**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.013 [-0.033, +0.001] (51/51) | -0.013 [-0.049, +0.031] (51/51) | -0.025 [-0.062, +0.018] (51/51) |
| variable | -0.008 [-0.017, +0.001] (51/51) | -0.030 [-0.072, +0.004] (51/51) | -0.074 [-0.144, -0.018] (51/51) |
| coverage | -0.009 [-0.017, -0.000] (51/51) | -0.030 [-0.071, +0.004] (51/51) | -0.074 [-0.146, -0.018] (51/51) |
| grid | -0.010 [-0.029, +0.004] (51/51) | -0.030 [-0.067, -0.003] (51/51) | -0.083 [-0.145, -0.030] (51/51) |
| hierarchical | +0.027 [-0.014, +0.079] (51/51) | -0.019 [-0.064, +0.021] (51/51) | -0.017 [-0.064, +0.027] (51/51) |
| voronoi | — (0/51) | — (0/51) | — (0/51) |
| deterministic | — (0/51) | — (0/51) | — (0/51) |
| suppressed | -0.028 [-0.053, -0.012] (51/51) | -0.033 [-0.062, -0.009] (51/51) | -0.086 [-0.154, -0.029] (51/51) |
| dual | +0.024 [-0.008, +0.059] (48/51) | +0.017 [-0.032, +0.062] (48/51) | +0.028 [-0.012, +0.074] (48/51) |
| annealed | -0.006 [-0.021, +0.008] (51/51) | -0.014 [-0.036, +0.004] (51/51) | -0.028 [-0.053, -0.007] (51/51) |
| binary · hamming | +0.001 [-0.011, +0.014] (45/51) | +0.008 [-0.016, +0.041] (45/51) | -0.001 [-0.020, +0.020] (45/51) |
| binary · rcontiguous | +0.014 [-0.011, +0.040] (36/51) | +0.006 [-0.017, +0.032] (36/51) | -0.027 [-0.048, -0.010] (36/51) |
| binary · rchunk | +0.008 [-0.005, +0.022] (45/51) | +0.014 [-0.002, +0.030] (45/51) | -0.014 [-0.037, +0.008] (45/51) |

**Paired AP change versus 100% training**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.003 [-0.010, +0.004] (51/51) | -0.005 [-0.014, +0.005] (51/51) | -0.010 [-0.020, -0.001] (51/51) |
| variable | -0.013 [-0.024, -0.002] (51/51) | -0.058 [-0.114, -0.017] (51/51) | -0.092 [-0.173, -0.024] (51/51) |
| coverage | -0.015 [-0.024, -0.006] (51/51) | -0.058 [-0.114, -0.017] (51/51) | -0.092 [-0.174, -0.025] (51/51) |
| grid | -0.013 [-0.030, -0.000] (51/51) | -0.056 [-0.108, -0.016] (51/51) | -0.104 [-0.183, -0.040] (51/51) |
| hierarchical | +0.005 [-0.010, +0.022] (51/51) | -0.011 [-0.029, +0.005] (51/51) | +0.002 [-0.015, +0.020] (51/51) |
| voronoi | — (0/51) | — (0/51) | — (0/51) |
| deterministic | — (0/51) | — (0/51) | — (0/51) |
| suppressed | -0.030 [-0.062, -0.008] (51/51) | -0.052 [-0.097, -0.013] (51/51) | -0.098 [-0.182, -0.033] (51/51) |
| dual | +0.012 [-0.019, +0.048] (48/51) | +0.022 [-0.020, +0.068] (48/51) | +0.034 [-0.008, +0.087] (48/51) |
| annealed | -0.005 [-0.012, +0.001] (51/51) | -0.008 [-0.029, +0.004] (51/51) | -0.018 [-0.042, -0.003] (51/51) |
| binary · hamming | -0.001 [-0.005, +0.003] (45/51) | -0.009 [-0.030, +0.004] (45/51) | -0.020 [-0.060, +0.002] (45/51) |
| binary · rcontiguous | -0.007 [-0.016, +0.001] (36/51) | -0.014 [-0.039, +0.002] (36/51) | -0.039 [-0.094, -0.009] (36/51) |
| binary · rchunk | -0.001 [-0.008, +0.006] (45/51) | -0.004 [-0.010, +0.003] (45/51) | -0.019 [-0.043, -0.002] (45/51) |

### PCA to two dimensions

The first table shows mean AUROC / AP at each retained size. The next two tables compare each reduced size with its matched 100% run.

| Method | 100% retained: AUROC / AP | 75% retained: AUROC / AP | 50% retained: AUROC / AP | 25% retained: AUROC / AP |
| --- | --- | --- | --- | --- |
| fixed | 0.597 (51/51) / 0.337 (51/51) | 0.598 (51/51) / 0.319 (51/51) | 0.629 (51/51) / 0.351 (51/51) | 0.636 (51/51) / 0.348 (51/51) |
| variable | 0.696 (51/51) / 0.444 (51/51) | 0.703 (51/51) / 0.431 (51/51) | 0.717 (51/51) / 0.455 (51/51) | 0.735 (51/51) / 0.425 (51/51) |
| coverage | 0.679 (51/51) / 0.427 (51/51) | 0.690 (51/51) / 0.427 (51/51) | 0.701 (51/51) / 0.449 (51/51) | 0.701 (51/51) / 0.411 (51/51) |
| grid | 0.695 (51/51) / 0.439 (51/51) | 0.702 (51/51) / 0.431 (51/51) | 0.716 (51/51) / 0.450 (51/51) | 0.738 (51/51) / 0.427 (51/51) |
| hierarchical | 0.681 (51/51) / 0.421 (51/51) | 0.679 (51/51) / 0.416 (51/51) | 0.695 (51/51) / 0.424 (51/51) | 0.692 (51/51) / 0.393 (51/51) |
| voronoi | 0.703 (51/51) / 0.450 (51/51) | 0.708 (51/51) / 0.445 (51/51) | 0.734 (51/51) / 0.460 (51/51) | 0.750 (51/51) / 0.440 (51/51) |
| deterministic | 0.621 (51/51) / 0.362 (51/51) | 0.635 (51/51) / 0.359 (51/51) | 0.657 (51/51) / 0.392 (51/51) | 0.657 (51/51) / 0.347 (51/51) |
| suppressed | 0.680 (51/51) / 0.426 (51/51) | 0.685 (51/51) / 0.420 (51/51) | 0.708 (51/51) / 0.445 (51/51) | 0.719 (51/51) / 0.416 (51/51) |
| dual | 0.714 (51/51) / 0.465 (51/51) | 0.717 (51/51) / 0.456 (51/51) | 0.734 (51/51) / 0.468 (51/51) | 0.762 (51/51) / 0.465 (51/51) |
| annealed | 0.606 (51/51) / 0.361 (51/51) | 0.635 (51/51) / 0.367 (51/51) | 0.657 (51/51) / 0.387 (51/51) | 0.635 (51/51) / 0.337 (51/51) |
| binary · hamming | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) |
| binary · rcontiguous | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) |
| binary · rchunk | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) | — (0/51) / — (0/51) |

**Paired AUROC change versus 100% training**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| fixed | +0.001 [-0.013, +0.013] (51/51) | +0.032 [+0.003, +0.071] (51/51) | +0.039 [-0.016, +0.094] (51/51) |
| variable | +0.007 [-0.005, +0.021] (51/51) | +0.022 [-0.002, +0.044] (51/51) | +0.039 [-0.016, +0.104] (51/51) |
| coverage | +0.011 [-0.003, +0.026] (51/51) | +0.022 [-0.002, +0.046] (51/51) | +0.022 [-0.027, +0.071] (51/51) |
| grid | +0.007 [-0.005, +0.019] (51/51) | +0.021 [-0.001, +0.043] (51/51) | +0.043 [-0.009, +0.104] (51/51) |
| hierarchical | -0.002 [-0.025, +0.020] (51/51) | +0.014 [-0.010, +0.035] (51/51) | +0.011 [-0.033, +0.056] (51/51) |
| voronoi | +0.005 [-0.009, +0.016] (51/51) | +0.031 [+0.009, +0.060] (51/51) | +0.047 [+0.001, +0.109] (51/51) |
| deterministic | +0.013 [-0.004, +0.031] (51/51) | +0.036 [+0.003, +0.076] (51/51) | +0.035 [-0.015, +0.092] (51/51) |
| suppressed | +0.005 [-0.013, +0.019] (51/51) | +0.028 [+0.008, +0.050] (51/51) | +0.040 [-0.010, +0.100] (51/51) |
| dual | +0.003 [-0.008, +0.013] (51/51) | +0.020 [+0.002, +0.038] (51/51) | +0.048 [+0.008, +0.106] (51/51) |
| annealed | +0.028 [+0.008, +0.053] (51/51) | +0.051 [+0.003, +0.121] (51/51) | +0.029 [-0.034, +0.105] (51/51) |
| binary · hamming | — (0/51) | — (0/51) | — (0/51) |
| binary · rcontiguous | — (0/51) | — (0/51) | — (0/51) |
| binary · rchunk | — (0/51) | — (0/51) | — (0/51) |

**Paired AP change versus 100% training**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.018 [-0.034, -0.004] (51/51) | +0.014 [-0.022, +0.053] (51/51) | +0.011 [-0.049, +0.070] (51/51) |
| variable | -0.012 [-0.059, +0.021] (51/51) | +0.012 [-0.008, +0.036] (51/51) | -0.019 [-0.067, +0.021] (51/51) |
| coverage | -0.000 [-0.028, +0.023] (51/51) | +0.022 [-0.002, +0.052] (51/51) | -0.016 [-0.063, +0.026] (51/51) |
| grid | -0.008 [-0.053, +0.023] (51/51) | +0.012 [-0.013, +0.042] (51/51) | -0.012 [-0.056, +0.029] (51/51) |
| hierarchical | -0.005 [-0.050, +0.038] (51/51) | +0.003 [-0.035, +0.044] (51/51) | -0.028 [-0.076, +0.013] (51/51) |
| voronoi | -0.005 [-0.044, +0.021] (51/51) | +0.010 [-0.015, +0.043] (51/51) | -0.010 [-0.048, +0.027] (51/51) |
| deterministic | -0.003 [-0.031, +0.025] (51/51) | +0.030 [-0.011, +0.079] (51/51) | -0.015 [-0.070, +0.034] (51/51) |
| suppressed | -0.005 [-0.040, +0.020] (51/51) | +0.019 [-0.001, +0.048] (51/51) | -0.009 [-0.050, +0.027] (51/51) |
| dual | -0.010 [-0.057, +0.026] (51/51) | +0.003 [-0.025, +0.035] (51/51) | -0.000 [-0.035, +0.031] (51/51) |
| annealed | +0.006 [-0.011, +0.025] (51/51) | +0.026 [-0.014, +0.069] (51/51) | -0.024 [-0.087, +0.032] (51/51) |
| binary · hamming | — (0/51) | — (0/51) | — (0/51) |
| binary · rcontiguous | — (0/51) | — (0/51) | — (0/51) |
| binary · rchunk | — (0/51) | — (0/51) | — (0/51) |

## Baseline controls

The same train-size reduction is applied to each control. ECOD can use the whole test batch during scoring, so its evaluation is not strictly pointwise inductive; IForest, OCSVM, KNN, and ABOD are the primary inductive controls here. The table cells keep the same coverage convention.

### Original features

| Method | 100% retained: AUROC / AP | 75% retained: AUROC / AP | 50% retained: AUROC / AP | 25% retained: AUROC / AP |
| --- | --- | --- | --- | --- |
| IForest | 0.852 (51/51) / 0.518 (51/51) | 0.849 (51/51) / 0.527 (51/51) | 0.844 (51/51) / 0.518 (51/51) | 0.833 (51/51) / 0.484 (51/51) |
| ECOD | 0.782 (51/51) / 0.468 (51/51) | 0.779 (51/51) / 0.458 (51/51) | 0.774 (51/51) / 0.448 (51/51) | 0.770 (51/51) / 0.436 (51/51) |
| OCSVM | 0.843 (51/51) / 0.562 (51/51) | 0.841 (51/51) / 0.553 (51/51) | 0.843 (51/51) / 0.556 (51/51) | 0.842 (51/51) / 0.551 (51/51) |
| KNN | 0.895 (51/51) / 0.610 (51/51) | 0.890 (51/51) / 0.597 (51/51) | 0.887 (51/51) / 0.589 (51/51) | 0.875 (51/51) / 0.570 (51/51) |
| ABOD | 0.892 (51/51) / 0.627 (51/51) | 0.889 (51/51) / 0.603 (51/51) | 0.878 (51/51) / 0.573 (51/51) | 0.870 (51/51) / 0.552 (51/51) |

**Controls' paired AUROC change versus 100% training**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| IForest | -0.003 [-0.010, +0.003] (51/51) | -0.008 [-0.017, -0.001] (51/51) | -0.018 [-0.030, -0.009] (51/51) |
| ECOD | -0.003 [-0.005, -0.002] (51/51) | -0.008 [-0.012, -0.004] (51/51) | -0.012 [-0.018, -0.007] (51/51) |
| OCSVM | -0.003 [-0.007, -0.000] (51/51) | -0.001 [-0.004, +0.003] (51/51) | -0.002 [-0.010, +0.006] (51/51) |
| KNN | -0.004 [-0.008, -0.001] (51/51) | -0.008 [-0.013, -0.003] (51/51) | -0.019 [-0.035, -0.005] (51/51) |
| ABOD | -0.003 [-0.006, +0.000] (51/51) | -0.013 [-0.023, -0.006] (51/51) | -0.022 [-0.036, -0.009] (51/51) |

**Controls' paired AP change versus 100% training**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| IForest | +0.009 [-0.015, +0.048] (51/51) | +0.000 [-0.017, +0.022] (51/51) | -0.033 [-0.049, -0.018] (51/51) |
| ECOD | -0.010 [-0.018, -0.004] (51/51) | -0.020 [-0.038, -0.009] (51/51) | -0.032 [-0.059, -0.015] (51/51) |
| OCSVM | -0.009 [-0.017, -0.003] (51/51) | -0.006 [-0.018, +0.004] (51/51) | -0.011 [-0.030, +0.011] (51/51) |
| KNN | -0.013 [-0.021, -0.006] (51/51) | -0.021 [-0.040, -0.006] (51/51) | -0.040 [-0.074, -0.008] (51/51) |
| ABOD | -0.025 [-0.037, -0.014] (51/51) | -0.055 [-0.086, -0.026] (51/51) | -0.075 [-0.132, -0.027] (51/51) |

### PCA to two dimensions

| Method | 100% retained: AUROC / AP | 75% retained: AUROC / AP | 50% retained: AUROC / AP | 25% retained: AUROC / AP |
| --- | --- | --- | --- | --- |
| IForest | 0.762 (51/51) / 0.402 (51/51) | 0.759 (51/51) / 0.395 (51/51) | 0.762 (51/51) / 0.407 (51/51) | 0.751 (51/51) / 0.356 (51/51) |
| ECOD | 0.621 (51/51) / 0.268 (51/51) | 0.621 (51/51) / 0.260 (51/51) | 0.624 (51/51) / 0.276 (51/51) | 0.622 (51/51) / 0.264 (51/51) |
| OCSVM | 0.784 (51/51) / 0.473 (51/51) | 0.777 (51/51) / 0.465 (51/51) | 0.793 (51/51) / 0.492 (51/51) | 0.782 (51/51) / 0.466 (51/51) |
| KNN | 0.799 (51/51) / 0.483 (51/51) | 0.802 (51/51) / 0.479 (51/51) | 0.806 (51/51) / 0.491 (51/51) | 0.800 (51/51) / 0.468 (51/51) |
| ABOD | 0.791 (51/51) / 0.477 (51/51) | 0.773 (51/51) / 0.460 (51/51) | 0.790 (51/51) / 0.476 (51/51) | 0.789 (51/51) / 0.468 (51/51) |

**Controls' paired AUROC change versus 100% training**

| Method | 75% retained: paired ΔAUROC [95% interval] | 50% retained: paired ΔAUROC [95% interval] | 25% retained: paired ΔAUROC [95% interval] |
| --- | --- | --- | --- |
| IForest | -0.004 [-0.012, +0.004] (51/51) | -0.000 [-0.012, +0.013] (51/51) | -0.012 [-0.022, -0.001] (51/51) |
| ECOD | -0.000 [-0.007, +0.005] (51/51) | +0.002 [-0.011, +0.019] (51/51) | +0.000 [-0.010, +0.011] (51/51) |
| OCSVM | -0.007 [-0.022, +0.004] (51/51) | +0.009 [-0.003, +0.024] (51/51) | -0.002 [-0.010, +0.007] (51/51) |
| KNN | +0.003 [-0.004, +0.011] (51/51) | +0.007 [-0.004, +0.019] (51/51) | +0.001 [-0.017, +0.021] (51/51) |
| ABOD | -0.018 [-0.056, +0.007] (51/51) | -0.001 [-0.015, +0.016] (51/51) | -0.002 [-0.013, +0.010] (51/51) |

**Controls' paired AP change versus 100% training**

| Method | 75% retained: paired ΔAP [95% interval] | 50% retained: paired ΔAP [95% interval] | 25% retained: paired ΔAP [95% interval] |
| --- | --- | --- | --- |
| IForest | -0.007 [-0.030, +0.014] (51/51) | +0.005 [-0.016, +0.028] (51/51) | -0.046 [-0.094, -0.001] (51/51) |
| ECOD | -0.008 [-0.020, +0.003] (51/51) | +0.007 [-0.011, +0.030] (51/51) | -0.004 [-0.037, +0.034] (51/51) |
| OCSVM | -0.009 [-0.032, +0.007] (51/51) | +0.019 [-0.008, +0.053] (51/51) | -0.008 [-0.025, +0.010] (51/51) |
| KNN | -0.004 [-0.020, +0.012] (51/51) | +0.008 [-0.018, +0.036] (51/51) | -0.014 [-0.047, +0.021] (51/51) |
| ABOD | -0.017 [-0.050, +0.006] (51/51) | -0.000 [-0.015, +0.019] (51/51) | -0.009 [-0.035, +0.027] (51/51) |

## NSA versus controls at the same training size

This comparison asks whether NSA has higher held-out AUROC than a control trained on the same retained normal set. Positive values favor NSA. Each comparison is paired by dataset, seed, representation, and test set; the full table also includes AP.

| Features | % retained | Control | NSA configurations with pairs | Full coverage | Positive NSA − control mean | Intervals above zero | Intervals below zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 75 | IForest | 11/13 | 7/13 | 0/11 | 0 | 11 |
| native | 75 | ECOD | 11/13 | 7/13 | 0/11 | 0 | 10 |
| native | 75 | OCSVM | 11/13 | 7/13 | 0/11 | 0 | 11 |
| native | 75 | KNN | 11/13 | 7/13 | 0/11 | 0 | 11 |
| native | 75 | ABOD | 11/13 | 7/13 | 0/11 | 0 | 11 |
| native | 50 | IForest | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 50 | ECOD | 11/13 | 8/13 | 0/11 | 0 | 10 |
| native | 50 | OCSVM | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 50 | KNN | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 50 | ABOD | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 25 | IForest | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 25 | ECOD | 11/13 | 8/13 | 0/11 | 0 | 10 |
| native | 25 | OCSVM | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 25 | KNN | 11/13 | 8/13 | 0/11 | 0 | 11 |
| native | 25 | ABOD | 11/13 | 8/13 | 0/11 | 0 | 11 |
| pca2 | 75 | IForest | 10/13 | 10/13 | 0/10 | 0 | 10 |
| pca2 | 75 | ECOD | 10/13 | 10/13 | 9/10 | 4 | 0 |
| pca2 | 75 | OCSVM | 10/13 | 10/13 | 0/10 | 0 | 10 |
| pca2 | 75 | KNN | 10/13 | 10/13 | 0/10 | 0 | 10 |
| pca2 | 75 | ABOD | 10/13 | 10/13 | 0/10 | 0 | 9 |
| pca2 | 50 | IForest | 10/13 | 10/13 | 0/10 | 0 | 8 |
| pca2 | 50 | ECOD | 10/13 | 10/13 | 10/10 | 6 | 0 |
| pca2 | 50 | OCSVM | 10/13 | 10/13 | 0/10 | 0 | 10 |
| pca2 | 50 | KNN | 10/13 | 10/13 | 0/10 | 0 | 10 |
| pca2 | 50 | ABOD | 10/13 | 10/13 | 0/10 | 0 | 10 |
| pca2 | 25 | IForest | 10/13 | 10/13 | 1/10 | 0 | 4 |
| pca2 | 25 | ECOD | 10/13 | 10/13 | 10/10 | 5 | 0 |
| pca2 | 25 | OCSVM | 10/13 | 10/13 | 0/10 | 0 | 8 |
| pca2 | 25 | KNN | 10/13 | 10/13 | 0/10 | 0 | 9 |
| pca2 | 25 | ABOD | 10/13 | 10/13 | 0/10 | 0 | 9 |

### Original features: NSA minus IForest at the same training size

| NSA configuration | 75% retained: AUROC difference [95% interval] | 50% retained: AUROC difference [95% interval] | 25% retained: AUROC difference [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.501 [-0.644, -0.352] (51/51) | -0.496 [-0.645, -0.344] (51/51) | -0.498 [-0.650, -0.346] (51/51) |
| variable | -0.258 [-0.407, -0.129] (51/51) | -0.274 [-0.420, -0.141] (51/51) | -0.308 [-0.455, -0.163] (51/51) |
| coverage | -0.259 [-0.408, -0.127] (51/51) | -0.275 [-0.422, -0.141] (51/51) | -0.309 [-0.457, -0.166] (51/51) |
| grid | -0.249 [-0.404, -0.118] (51/51) | -0.264 [-0.415, -0.128] (51/51) | -0.306 [-0.452, -0.169] (51/51) |
| hierarchical | -0.532 [-0.668, -0.390] (51/51) | -0.573 [-0.706, -0.432] (51/51) | -0.560 [-0.700, -0.416] (51/51) |
| voronoi | — (0/51) | — (0/51) | — (0/51) |
| deterministic | — (0/51) | — (0/51) | — (0/51) |
| suppressed | -0.280 [-0.423, -0.153] (51/51) | -0.280 [-0.425, -0.145] (51/51) | -0.323 [-0.469, -0.181] (51/51) |
| dual | -0.118 [-0.190, -0.049] (48/51) | -0.120 [-0.199, -0.050] (48/51) | -0.095 [-0.184, -0.015] (49/51) |
| annealed | -0.541 [-0.676, -0.407] (51/51) | -0.544 [-0.679, -0.396] (51/51) | -0.548 [-0.679, -0.409] (51/51) |
| binary · hamming | -0.321 [-0.414, -0.221] (49/51) | -0.299 [-0.402, -0.182] (51/51) | -0.303 [-0.402, -0.203] (51/51) |
| binary · rcontiguous | -0.328 [-0.385, -0.268] (36/51) | -0.332 [-0.391, -0.272] (36/51) | -0.328 [-0.391, -0.265] (39/51) |
| binary · rchunk | -0.343 [-0.394, -0.286] (45/51) | -0.333 [-0.381, -0.278] (45/51) | -0.348 [-0.406, -0.281] (45/51) |

### PCA to two dimensions: NSA minus IForest at the same training size

| NSA configuration | 75% retained: AUROC difference [95% interval] | 50% retained: AUROC difference [95% interval] | 25% retained: AUROC difference [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.161 [-0.213, -0.109] (51/51) | -0.133 [-0.180, -0.086] (51/51) | -0.115 [-0.175, -0.057] (51/51) |
| variable | -0.056 [-0.099, -0.015] (51/51) | -0.045 [-0.089, -0.005] (51/51) | -0.016 [-0.068, +0.029] (51/51) |
| coverage | -0.069 [-0.116, -0.030] (51/51) | -0.061 [-0.114, -0.016] (51/51) | -0.049 [-0.103, +0.001] (51/51) |
| grid | -0.057 [-0.101, -0.016] (51/51) | -0.046 [-0.092, -0.006] (51/51) | -0.012 [-0.059, +0.034] (51/51) |
| hierarchical | -0.079 [-0.130, -0.035] (51/51) | -0.067 [-0.116, -0.022] (51/51) | -0.058 [-0.116, -0.006] (51/51) |
| voronoi | -0.050 [-0.093, -0.008] (51/51) | -0.028 [-0.064, +0.006] (51/51) | -0.000 [-0.041, +0.042] (51/51) |
| deterministic | -0.124 [-0.172, -0.077] (51/51) | -0.105 [-0.156, -0.055] (51/51) | -0.094 [-0.158, -0.033] (51/51) |
| suppressed | -0.074 [-0.115, -0.034] (51/51) | -0.054 [-0.089, -0.021] (51/51) | -0.031 [-0.077, +0.014] (51/51) |
| dual | -0.042 [-0.080, -0.007] (51/51) | -0.028 [-0.065, +0.004] (51/51) | +0.012 [-0.026, +0.048] (51/51) |
| annealed | -0.124 [-0.168, -0.081] (51/51) | -0.105 [-0.152, -0.061] (51/51) | -0.116 [-0.179, -0.057] (51/51) |
| binary · hamming | — (0/51) | — (0/51) | — (0/51) |
| binary · rcontiguous | — (0/51) | — (0/51) | — (0/51) |
| binary · rchunk | — (0/51) | — (0/51) | — (0/51) |

## Relative benefit compared with the controls

For each NSA/control pair, the interaction is **(NSA at reduced size − NSA at 100%) − (control at reduced size − control at 100%)**, using only dataset/seed cells where all four fits succeeded. Positive values mean NSA gained more (or lost less) from reducing training data than that control. This is a relative training-size effect; it does not mean NSA has the higher absolute AUROC. The first table includes all five controls; the next tables show the per-configuration comparison with IForest. The complete 390-row interaction CSV contains every NSA/control interval and coverage count.

| Features | % retained | Control | NSA configurations with pairs | Full coverage | Positive mean interaction | Intervals above zero | Intervals below zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 75 | IForest | 11/13 | 7/13 | 5/11 | 0 | 1 |
| native | 75 | ECOD | 11/13 | 7/13 | 5/11 | 0 | 1 |
| native | 75 | OCSVM | 11/13 | 7/13 | 5/11 | 0 | 1 |
| native | 75 | KNN | 11/13 | 7/13 | 5/11 | 0 | 1 |
| native | 75 | ABOD | 11/13 | 7/13 | 5/11 | 0 | 1 |
| native | 50 | IForest | 11/13 | 7/13 | 4/11 | 1 | 1 |
| native | 50 | ECOD | 11/13 | 7/13 | 4/11 | 1 | 1 |
| native | 50 | OCSVM | 11/13 | 7/13 | 4/11 | 0 | 2 |
| native | 50 | KNN | 11/13 | 7/13 | 4/11 | 1 | 1 |
| native | 50 | ABOD | 11/13 | 7/13 | 5/11 | 1 | 0 |
| native | 25 | IForest | 11/13 | 7/13 | 4/11 | 1 | 2 |
| native | 25 | ECOD | 11/13 | 7/13 | 2/11 | 1 | 4 |
| native | 25 | OCSVM | 11/13 | 7/13 | 1/11 | 0 | 6 |
| native | 25 | KNN | 11/13 | 7/13 | 4/11 | 1 | 2 |
| native | 25 | ABOD | 11/13 | 7/13 | 4/11 | 1 | 1 |
| pca2 | 75 | IForest | 10/13 | 10/13 | 10/10 | 1 | 0 |
| pca2 | 75 | ECOD | 10/13 | 10/13 | 9/10 | 1 | 0 |
| pca2 | 75 | OCSVM | 10/13 | 10/13 | 10/10 | 6 | 0 |
| pca2 | 75 | KNN | 10/13 | 10/13 | 7/10 | 1 | 0 |
| pca2 | 75 | ABOD | 10/13 | 10/13 | 10/10 | 6 | 0 |
| pca2 | 50 | IForest | 10/13 | 10/13 | 10/10 | 5 | 0 |
| pca2 | 50 | ECOD | 10/13 | 10/13 | 10/10 | 5 | 0 |
| pca2 | 50 | OCSVM | 10/13 | 10/13 | 10/10 | 0 | 0 |
| pca2 | 50 | KNN | 10/13 | 10/13 | 10/10 | 1 | 0 |
| pca2 | 50 | ABOD | 10/13 | 10/13 | 10/10 | 3 | 0 |
| pca2 | 25 | IForest | 10/13 | 10/13 | 10/10 | 3 | 0 |
| pca2 | 25 | ECOD | 10/13 | 10/13 | 10/10 | 1 | 0 |
| pca2 | 25 | OCSVM | 10/13 | 10/13 | 10/10 | 2 | 0 |
| pca2 | 25 | KNN | 10/13 | 10/13 | 10/10 | 0 | 0 |
| pca2 | 25 | ABOD | 10/13 | 10/13 | 10/10 | 1 | 0 |

### Original features: NSA training-size interaction versus IForest

| NSA configuration | 75% retained: AUROC interaction [95% interval] | 50% retained: AUROC interaction [95% interval] | 25% retained: AUROC interaction [95% interval] |
| --- | --- | --- | --- |
| fixed | -0.010 [-0.033, +0.007] (51/51) | -0.005 [-0.045, +0.042] (51/51) | -0.006 [-0.049, +0.045] (51/51) |
| variable | -0.005 [-0.015, +0.006] (51/51) | -0.022 [-0.067, +0.016] (51/51) | -0.056 [-0.126, +0.006] (51/51) |
| coverage | -0.006 [-0.016, +0.005] (51/51) | -0.022 [-0.066, +0.017] (51/51) | -0.056 [-0.128, +0.005] (51/51) |
| grid | -0.007 [-0.027, +0.009] (51/51) | -0.022 [-0.060, +0.009] (51/51) | -0.064 [-0.131, -0.010] (51/51) |
| hierarchical | +0.030 [-0.011, +0.081] (51/51) | -0.011 [-0.051, +0.027] (51/51) | +0.002 [-0.040, +0.046] (51/51) |
| voronoi | — (0/51) | — (0/51) | — (0/51) |
| deterministic | — (0/51) | — (0/51) | — (0/51) |
| suppressed | -0.025 [-0.052, -0.007] (51/51) | -0.024 [-0.055, -0.001] (51/51) | -0.068 [-0.139, -0.008] (51/51) |
| dual | +0.027 [-0.005, +0.063] (48/51) | +0.025 [-0.018, +0.069] (48/51) | +0.048 [+0.006, +0.095] (48/51) |
| annealed | -0.003 [-0.016, +0.010] (51/51) | -0.006 [-0.024, +0.015] (51/51) | -0.010 [-0.034, +0.011] (51/51) |
| binary · hamming | +0.003 [-0.013, +0.020] (45/51) | +0.014 [-0.014, +0.055] (45/51) | +0.017 [-0.007, +0.050] (45/51) |
| binary · rcontiguous | +0.011 [-0.012, +0.038] (36/51) | +0.007 [-0.014, +0.035] (36/51) | -0.015 [-0.040, +0.007] (36/51) |
| binary · rchunk | +0.009 [-0.007, +0.026] (45/51) | +0.020 [+0.001, +0.040] (45/51) | +0.004 [-0.023, +0.031] (45/51) |

### PCA to two dimensions: NSA training-size interaction versus IForest

| NSA configuration | 75% retained: AUROC interaction [95% interval] | 50% retained: AUROC interaction [95% interval] | 25% retained: AUROC interaction [95% interval] |
| --- | --- | --- | --- |
| fixed | +0.005 [-0.009, +0.020] (51/51) | +0.032 [+0.003, +0.077] (51/51) | +0.051 [-0.002, +0.108] (51/51) |
| variable | +0.011 [-0.003, +0.027] (51/51) | +0.022 [-0.005, +0.052] (51/51) | +0.051 [-0.004, +0.121] (51/51) |
| coverage | +0.014 [-0.001, +0.031] (51/51) | +0.022 [-0.004, +0.050] (51/51) | +0.034 [-0.019, +0.088] (51/51) |
| grid | +0.011 [-0.001, +0.025] (51/51) | +0.021 [-0.004, +0.049] (51/51) | +0.055 [+0.003, +0.124] (51/51) |
| hierarchical | +0.002 [-0.019, +0.023] (51/51) | +0.014 [-0.007, +0.037] (51/51) | +0.023 [-0.021, +0.072] (51/51) |
| voronoi | +0.009 [-0.004, +0.022] (51/51) | +0.031 [+0.006, +0.065] (51/51) | +0.059 [+0.010, +0.126] (51/51) |
| deterministic | +0.017 [-0.001, +0.036] (51/51) | +0.036 [+0.004, +0.075] (51/51) | +0.047 [-0.004, +0.104] (51/51) |
| suppressed | +0.009 [-0.008, +0.026] (51/51) | +0.028 [+0.004, +0.055] (51/51) | +0.051 [-0.000, +0.116] (51/51) |
| dual | +0.007 [-0.004, +0.020] (51/51) | +0.020 [-0.001, +0.045] (51/51) | +0.060 [+0.018, +0.119] (51/51) |
| annealed | +0.032 [+0.011, +0.058] (51/51) | +0.051 [+0.007, +0.112] (51/51) | +0.041 [-0.023, +0.117] (51/51) |
| binary · hamming | — (0/51) | — (0/51) | — (0/51) |
| binary · rcontiguous | — (0/51) | — (0/51) | — (0/51) |
| binary · rchunk | — (0/51) | — (0/51) | — (0/51) |

The same-size comparisons above assess absolute relative AUROC at each reduced size; a positive interaction alone cannot establish that claim.

## Exploratory dataset exclusion check

The PCA2 dual and Voronoi gains at 25% retained were inspected after seeing the dataset-level changes. The table compares the original 17-dataset mean with a recomputation excluding vowels and letter, the two largest positive contributors for dual. This exclusion was chosen **after** observing outcomes; it is a stress check, not a prespecified test or independent replication.

| NSA configuration | Datasets excluded | Datasets left | Paired ΔAUROC [95% interval] | Paired ΔAP [95% interval] | Datasets with AUROC gain |
| --- | --- | --- | --- | --- | --- |
| dual | None | 17 | +0.048 [+0.008, +0.106] | -0.000 [-0.035, +0.031] | 12 |
| dual | Vowels and letter | 15 | +0.015 [-0.001, +0.031] | -0.009 [-0.043, +0.025] | 10 |
| voronoi | None | 17 | +0.047 [+0.001, +0.109] | -0.010 [-0.048, +0.027] | 10 |
| voronoi | Vowels and letter | 15 | +0.011 [-0.012, +0.034] | -0.021 [-0.061, +0.018] | 8 |

Both AUROC intervals include zero after excluding those two datasets. The full `retention_exclusion_sensitivity.csv` also reports each one-dataset exclusion for these two configurations.

## Completion and failure accounting

| Features | % retained | Group | Successful runs | Other recorded outcomes |
| --- | --- | --- | --- | --- |
| native | 100 | NSA | 532/663 | error 29, requires_2d 51, requires_2d_or_3d 51 |
| native | 100 | Control | 255/255 | none |
| native | 75 | NSA | 535/663 | error 26, requires_2d 51, requires_2d_or_3d 51 |
| native | 75 | Control | 255/255 | none |
| native | 50 | NSA | 537/663 | error 24, requires_2d 51, requires_2d_or_3d 51 |
| native | 50 | Control | 255/255 | none |
| native | 25 | NSA | 541/663 | error 20, requires_2d 51, requires_2d_or_3d 51 |
| native | 25 | Control | 255/255 | none |
| pca2 | 100 | NSA | 510/663 | error 153 |
| pca2 | 100 | Control | 255/255 | none |
| pca2 | 75 | NSA | 510/663 | error 153 |
| pca2 | 75 | Control | 255/255 | none |
| pca2 | 50 | NSA | 510/663 | error 153 |
| pca2 | 50 | Control | 255/255 | none |
| pca2 | 25 | NSA | 510/663 | error 153 |
| pca2 | 25 | Control | 255/255 | none |

A no-pair cell means no valid within-dataset/seed reduced-versus-full comparison was available. Geometry exclusions (for example, deterministic NSA outside two dimensions), detector-generation errors, and timeouts are reported separately in the underlying status counts. A method that cannot fit is not assigned an AUROC or AP of zero.

## Limits on interpretation

- This is a fixed-parameter screen on 17 datasets with three splits and one nested subset draw per split. Its many comparisons are exploratory; the 95% intervals are descriptive and unadjusted.
- The 600-normal fit cap and 2,000-row test cap reduce runtime and do not reproduce full-dataset training. The smallest 25% fit sets have 21 normals on lympho, 30 on glass, and 31 on vertebral.
- Refit scaling and PCA make this a test of the full low-data pipeline. In the PCA panel, training-size changes can also rotate or rescale the two-dimensional representation. PCA may discard anomaly-relevant directions, especially in high-dimensional datasets.
- A method with partial coverage can have an optimistic or pessimistic available-case mean. Its paired denominator and failure counts must accompany any claim.
- The five control algorithms are a selected follow-up panel. The earlier 100%-training report contains the wider 43-detector comparison. This reduced-data experiment does not rerun all 43.
- AP depends on held-out anomaly prevalence and is interpreted within the matched split. No outlier was available for training, but outliers in the test set are necessary to score AUROC/AP.
- The PCA2 dual and Voronoi 25% AUROC gains have substantial contributions from vowels and letter. Their post-hoc exclusion intervals include zero; this limits claims of a stable cross-dataset gain.

## Conclusion

In native, no NSA configuration had positive paired mean AUROC changes at all three reduced sizes with complete coverage.
In native, no configuration had an exploratory 95% interval entirely above zero at every reduced size.
In native, no NSA configuration had higher same-size mean AUROC than IForest at all three reduced sizes with complete paired coverage.
In pca2, fixed, variable, coverage, grid, voronoi, deterministic, suppressed, dual, annealed had positive paired mean AUROC changes at all three reduced sizes with complete coverage.
In pca2, no configuration had an exploratory 95% interval entirely above zero at every reduced size.
In pca2, no NSA configuration had higher same-size mean AUROC than IForest at all three reduced sizes with complete paired coverage.

The evidence supports only the family, size, representation, and coverage-qualified statements shown above. PCA2 dual and Voronoi showed AUROC gains at 25% retained, but neither showed a paired AP gain; both AUROC gains weakened under the post-hoc vowels-and-letter exclusion. Compare the same-size NSA/control differences before claiming superiority. No PR response or external post was made from this report.

## Reproducibility files

- `retention/analysis/retention_method_summary.csv`: AUROC/AP, coverage, and status counts for 100%, 75%, 50%, and 25% training.
- `retention/analysis/retention_change_vs_full.csv`: all paired size changes and bootstrap intervals.
- `retention/analysis/retention_dataset_changes.csv`: dataset-level matched changes and statuses.
- `retention/analysis/retention_nsa_vs_baseline.csv`: all same-size NSA/control comparisons.
- `retention/analysis/retention_interaction.csv`: all relative training-size effects.
- `retention/analysis/retention_exclusion_sensitivity.csv`: exploratory leave-one-dataset and vowels-plus-letter exclusion checks for PCA2 dual and Voronoi at 25%.
- `retention/retention_manifest.json`: frozen source, protocol, package, input-hash, and output-hash provenance for this follow-up experiment.
- `retention/analysis/retention_auroc_change.png`: per-configuration AUROC change figure with partial-coverage markers.
- `dataset_manifest.json`, `run_manifest.json`, the raw JSONL records, `retention_analysis.py`, `runner.py`, `model_factory.py`, and `retention_plot.py`: source and reproducibility information.
