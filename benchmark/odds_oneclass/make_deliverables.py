"""Create a reviewable report and evidence bundle from validated benchmark outputs."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent.parent / "outputs"
ANALYSIS = ROOT / "analysis"
SENSITIVITY = ROOT / "sensitivity" / "analysis"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    def safe(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        ["| " + " | ".join(map(safe, headers)) + " |",
         "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(map(safe, row)) + " |" for row in rows]
    )


def number(value: object, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def signed(value: object, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):+.{digits}f}"


def count_status(group: pd.DataFrame) -> str:
    counts = group.status.value_counts().to_dict()
    return ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))


def score_cell(row: pd.Series) -> str:
    if row.success_datasets == 0:
        statuses = json.loads(row.status_counts)
        return "; ".join(f"{status} {n}/17" for status, n in statuses.items())
    return (f"{number(row.mean_auroc)} / {number(row.mean_ap)} "
            f"({int(row.success_datasets)}/17)")


def run() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(ANALYSIS / "all_results.csv")
    assert len(full) == 1904
    assert full.dataset.nunique() == 17 and full.method.nunique() == 56
    assert (full.fit_outliers == 0).all()
    assert set(full.representation) == {"native", "pca2"}
    assert set(full.seed) == {42}
    assert not full.duplicated(["representation", "dataset", "method", "seed"]).any()
    selected = pd.read_csv(SENSITIVITY / "sensitivity_records.csv")
    assert len(selected) == 1836
    assert (selected.fit_outliers == 0).all()
    assert set(selected.seed) == {42, 43, 44}
    native_m = pd.read_csv(ANALYSIS / "native_methods.csv").set_index("method")
    pca_m = pd.read_csv(ANALYSIS / "pca2_methods.csv").set_index("method")
    native_d = pd.read_csv(ANALYSIS / "native_datasets.csv").set_index("dataset")
    pca_d = pd.read_csv(ANALYSIS / "pca2_datasets.csv").set_index("dataset")
    sens_m = pd.read_csv(SENSITIVITY / "sensitivity_methods.csv")
    sens_p = pd.read_csv(SENSITIVITY / "sensitivity_pairwise.csv")
    sens_lookup = {(row.representation, row.method): row
                   for row in sens_m.itertuples(index=False)}
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    features = {item["name"]: item["n_features"] for item in manifest["datasets"]}

    parts: list[str] = []
    parts.append("# PyOD NSA: normal-only training benchmark and PR evidence report\n")
    parts.append("**Prepared 12 October 2026.** This is a local report for review; nothing was posted to GitHub. "
                 "It evaluates the NSA implementation at PR #747's source tree "
                 "`57ec5c9ea2800b559019adf0457d7e5b858fc153`.\n")
    parts.append("## Result in brief\n")
    parts.append("The requested clean-normal training condition was met in every recorded fit: "
                 "**zero anomalous training rows**. Outliers remained in a separate held-out test partition. "
                 "The results **do not establish general NSA superiority**. Most eligible PyOD detectors learned "
                 "and scored with normal-only training data. Only XGBOD and DevNet in the 43-method tabular "
                 "baseline roster require outlier labels for their intended supervised task.\n")
    parts.append(md_table(
        ["Three-split summary", "Best fully observed NSA", "Inductive comparators", "Interpretation"],
        [["Original features", "NSA/grid 0.610 AUROC (51/51)",
          "KNN 0.895; ABOD 0.892; IForest 0.852 (each 51/51)",
          "NSA/dual reached 0.710 on 49/51 runs, so its mean has incomplete coverage."],
         ["PCA to 2D", "NSA/dual 0.714 AUROC (51/51)",
          "KNN 0.799; ABOD 0.791; OCSVM 0.784; IForest 0.762 (each 51/51)",
          "NSA trails each of these predeclared inductive comparators."]]
    ) + "\n")
    parts.append("Means give each of the 17 datasets equal weight, with seeds 42–44 averaged within dataset. "
                 "The complete 43-baseline/13-NSA, seed-42 screen and the 18-method/three-seed sensitivity panel "
                 "appear below and in the attached machine-readable files.\n")

    parts.append("## The 11 NSA strategy families, compared separately\n")
    mechanisms = [
        ("binary", "Median-threshold bits; Hamming, contiguous-run, or chunk matching"),
        ("fixed", "Fixed-radius nonself spheres"),
        ("variable", "Nearest-self variable-radius spheres"),
        ("coverage", "Variable spheres with statistical coverage stopping"),
        ("grid", "Sparse spatial grid and radius filtering"),
        ("hierarchical", "Coarse-to-fine self clusters"),
        ("voronoi", "Bounded Voronoi centers in 2D or 3D"),
        ("deterministic", "2D lattice with boundary repulsion"),
        ("suppressed", "Negative spheres plus reverse self detectors"),
        ("dual", "Cluster-ball envelope plus negative detectors"),
        ("annealed", "Monte Carlo population estimate plus position annealing"),
    ]

    def family_score(representation: str, family: str) -> str:
        if family == "binary":
            variants = [("hamming", "hamming"),
                        ("contiguous", "rcontiguous"), ("chunk", "rchunk")]
            pieces = []
            for label, rule in variants:
                row = sens_lookup[(representation, f"NSA/binary/{rule}")]
                pieces.append(f"{label} {number(row.macro_auroc_available)} "
                              f"({row.successful_rows}/51)")
            return "; ".join(pieces)
        row = sens_lookup[(representation, f"NSA/{family}")]
        return f"{number(row.macro_auroc_available)} ({row.successful_rows}/51)"

    family_rows = [[family, mechanism, family_score("native", family),
                    family_score("pca2", family)]
                   for family, mechanism in mechanisms]
    parts.append(md_table(["NSA family", "Implemented mechanism", "Original AUROC (runs)",
                           "PCA(2) AUROC (runs)"], family_rows) + "\n")
    parts.append("This table compares the PR's **11 generation strategies** under the same normal-only "
                 "training splits. Binary is one family with three matching rules, displayed separately "
                 "instead of averaging away rule-specific failures. A score of “—” means no valid scored "
                 "run. The 13-configuration tables below add AP and all failure outcomes. These are "
                 "constructor-default strategies, not tuned optima.\n")

    parts.append("## Data and protocol\n")
    parts.append("1. **Source and coverage.** The 17 ODDS `.mat` datasets bundled in the "
                 "[official PyOD benchmark notebook](https://github.com/yzhao062/pyod/blob/master/notebooks/benchmark.py) "
                 "were checked against PyOD commit `ef6d5e8dc828615f498286c1d6a22cbd02fd4cd2`; "
                 "the manifest records source URL and SHA-256 for every file. The tested roster is all 44 "
                 "**shipped** tabular entries in the checked-out `algorithms.json`: 43 baselines plus NSA. "
                 "The registry's planned LLMAD entry has no implementation in this checkout and was "
                 "excluded as unavailable. NSA's 11 "
                 "strategies become 13 configurations because binary matching has three rules.\n")
    parts.append("2. **Split before fitting.** Each dataset used a 60/40 stratified train/test split. Every `y=1` "
                 "row in the training partition was discarded. Up to 600 normal training rows and 2,000 "
                 "held-out test rows were retained. For the test cap, stratified subsampling preserved the "
                 "partition's approximate prevalence. Seeds 42, 43 and 44 were used in the sensitivity panel; "
                 "the full roster used seed 42.\n")
    parts.append("3. **Preprocessing.** StandardScaler was fit on normal training data only. The original-feature "
                 "panel used all features. A second panel applied PCA to two components fit on normal training "
                 "data only, allowing NSA's two-dimensional Voronoi and deterministic strategies to run. "
                 "Test data were transformed by the already fit scaler/projector. PCA(2) can discard most "
                 "variance, so those results describe a different representation.\n")
    parts.append("4. **Models and metrics.** All attempted models received only `fit(X_normal)`, without `y`. "
                 "They scored the same held-out, mixed-label test set with `decision_function`. AUROC and "
                 "average precision (AP) were computed from the test labels after scoring. PyOD's default "
                 "`contamination=0.1` was retained where applicable; this parameter did not supply "
                 "anomalous training rows or labels. Ranked scores, rather than binary predictions, "
                 "were evaluated. No hyperparameters "
                 "were selected using test labels. A 15-second wall-time budget applied to each model/dataset "
                 "attempt, including lazy import, construction, fit and score.\n")
    parts.append("5. **Predeclared practical settings.** NSA used its constructor defaults (100 detectors and "
                 "10,000 candidate draws) except strategy, seed, and a fixed binary matching length of "
                 "40% of feature count for contiguous/chunk rules. Neighbor counts were bounded by the "
                 "available normal training rows. LSCP used a fixed LOF/KNN/HBOS base list because it has no "
                 "default list. To keep the CPU-wide screen feasible, DIF used 10 ensembles instead of 50; "
                 "DeepSVDD 50 epochs instead of 100; AnoGAN 50 training and 5 query epochs instead of "
                 "500/20; ALAD 50 instead of 200; LUNAR 50 instead of 200. These changes limit "
                 "paper-default comparisons. The code and exact versions are in the evidence bundle.\n")
    parts.append("6. **Outcome accounting.** Every attempted method has a status. A timeout, geometry exclusion, "
                 "or score error is never imputed as AUROC zero. The analysis checks one record for every "
                 "planned method/dataset/representation cell, validates the zero-outlier fit count, and "
                 "keeps failures in denominators. Reported runtime includes lazy import/construction; it "
                 "is not a comparable fit-only speed benchmark.\n")

    parts.append("## Main screen: complete method outcomes, seed 42\n")
    status_rows = []
    for representation in ("native", "pca2"):
        for family in ("NSA", "baseline"):
            block = full[(full.representation == representation) & (full.family == family)]
            status_rows.append(["Original features" if representation == "native" else "PCA to 2D",
                                family, str(len(block)), count_status(block)])
    parts.append(md_table(["Representation", "Family", "Planned records", "Outcomes"], status_rows) + "\n")
    parts.append("The following tables include **every** configured NSA method and every tabular baseline. "
                 "Cells show mean AUROC / mean AP and successful datasets out of 17. Means with fewer than "
                 "17 successes are conditional on success and should not be ranked against complete-coverage "
                 "means. AP depends on the anomaly prevalence in each test split.\n")

    nsa_names = [name for name in native_m.index if name.startswith("NSA/")]
    nsa_names = sorted(nsa_names)
    nsa_rows = [[name, score_cell(native_m.loc[name]), score_cell(pca_m.loc[name])]
                for name in nsa_names]
    parts.append("### NSA: 13 configurations\n")
    parts.append(md_table(["NSA configuration", "Original AUROC / AP (coverage)",
                           "PCA(2) AUROC / AP (coverage)"], nsa_rows) + "\n")
    parts.append("NSA/voronoi requires two or three features; NSA/deterministic requires exactly two. "
                 "All original-feature ODDS files here have more than three features. All three binary "
                 "matching rules failed to generate a usable detector in the PCA(2) panel. On original "
                 "features, binary rules also failed on some datasets, and NSA/dual failed on MNIST. "
                 "These are NSA applicability or implementation outcomes, not missing outlier labels.\n")

    base_names = sorted(name for name in native_m.index if not name.startswith("NSA/"))
    base_rows = []
    for name in base_names:
        row = native_m.loc[name]
        flags = []
        if name in {"XGBOD", "DevNet"}:
            flags.append("requires anomaly labels")
        elif name == "MAD":
            flags.append("one feature only")
        elif bool(row.batch_dependent):
            flags.append("batch")
        if bool(row.stochastic_inference):
            flags.append("stochastic")
        base_rows.append([name, score_cell(row), score_cell(pca_m.loc[name]),
                          ", ".join(flags) if flags else "pointwise"])
    parts.append("### Other PyOD tabular methods: all 43 registry entries\n")
    parts.append(md_table(["Method", "Original AUROC / AP (coverage)",
                           "PCA(2) AUROC / AP (coverage)", "Fit/score note"], base_rows) + "\n")
    parts.append("“Batch” means this implementation's score for a query may depend on other query rows; "
                 "those rows were unlabeled, but this is not the same strictly inductive pointwise task "
                 "as NSA. The batch-dependent group is COF, LOCI, SOS, SOD, ECOD, COPOD, RGraph, SUOD, "
                 "LMDD, DIF, CD and ROD. VAE, LSCP and ALAD have stochastic or call-order-sensitive "
                 "inference here. The four predeclared repeated-run inductive comparators were IForest, "
                 "OCSVM, KNN and ABOD; ECOD was repeated as a batch-dependent comparator.\n")
    parts.append("For the original-feature screen, the best *fully observed* NSA method was grid (0.622), "
                 "while LUNAR, KPCA, KDE, KNN and ABOD scored 0.904, 0.903, 0.902, 0.897 and 0.891, "
                 "respectively, each on all 17 datasets. NSA/dual scored 0.728 on 16/17. For PCA(2), "
                 "NSA/dual scored 0.723 versus KDE 0.800, KPCA 0.795, ABOD 0.794 and KNN 0.794, "
                 "each on all 17.\n")

    parts.append("## Per-dataset screen: descriptive best observed strategy\n")
    parts.append("These tables select each family's best method **after reading test labels**. They are "
                 "optimistic diagnostics, not a usable strategy-selection procedure. Eligible baseline "
                 "selection excludes the batch-dependent methods above. The listed test anomaly count "
                 "is from seed 42 and can be very small.\n")
    for representation, data in (("Original features", native_d), ("PCA to 2D", pca_d)):
        rows = []
        for name, row in data.sort_index().iterrows():
            delta = row.oracle_best_nsa_auroc - row.oracle_best_inductive_baseline_auroc
            rows.append([name, str(features[name]), str(int(row.fit_normals)),
                         str(int(row.test_outliers)),
                         (number(row.retained_variance) if representation == "PCA to 2D" else "1.000"),
                         f"{row.oracle_best_nsa.removeprefix('NSA/')} {number(row.oracle_best_nsa_auroc)}",
                         f"{row.oracle_best_inductive_baseline} {number(row.oracle_best_inductive_baseline_auroc)}",
                         signed(delta)])
        parts.append(f"### {representation}\n")
        parts.append(md_table(["Dataset", "Features", "Fit normals", "Test anomalies",
                               "Variance kept", "Best NSA", "Best inductive baseline",
                               "NSA − baseline"], rows) + "\n")
    parts.append("The hindsight NSA winner beat the hindsight eligible baseline on 1 of 17 original-feature "
                 "datasets (vertebral), tied on two (lympho, musk), and lost on 14. In PCA(2), it won "
                 "on two (cardio, glass) and lost on 15. The glass test split contains only four anomalies; "
                 "lympho contains two.\n")

    parts.append("## Three-split sensitivity and paired uncertainty\n")
    parts.append("The prespecified repeated subset was all 13 NSA configurations and IForest, ECOD, "
                 "OCSVM, KNN and ABOD on seeds 42–44. This is 17 datasets × 18 configurations × 3 seeds "
                 "× 2 representations = **1,836 validated records**. Means below average seed scores "
                 "within dataset, then give each dataset equal weight. Coverage is successful runs out of 51.\n")
    order = ["NSA/fixed", "NSA/variable", "NSA/coverage", "NSA/grid",
             "NSA/hierarchical", "NSA/voronoi", "NSA/deterministic",
             "NSA/suppressed", "NSA/dual", "NSA/annealed",
             "NSA/binary/hamming", "NSA/binary/rcontiguous", "NSA/binary/rchunk",
             "IForest", "ECOD", "OCSVM", "KNN", "ABOD"]
    rows = []
    for name in order:
        n = sens_lookup[("native", name)]
        p = sens_lookup[("pca2", name)]
        rows.append([name,
                     f"{number(n.macro_auroc_available)} / {number(n.macro_ap_available)} ({n.successful_rows}/51)",
                     f"{number(p.macro_auroc_available)} / {number(p.macro_ap_available)} ({p.successful_rows}/51)"])
    parts.append(md_table(["Method", "Original AUROC / AP (coverage)",
                           "PCA(2) AUROC / AP (coverage)"], rows) + "\n")

    p_lookup = {(row.representation, row.nsa, row.baseline): row
                for row in sens_p.itertuples(index=False)}
    pair_rows = []
    for name in order[:13]:
        cells = []
        for representation in ("native", "pca2"):
            r = p_lookup[(representation, name, "IForest")]
            if r.paired_seed_rows == 0:
                cells.append("—")
            else:
                cells.append(f"{signed(r.macro_mean_delta_auroc)} "
                             f"[{signed(r.cluster_bootstrap_95_low_auroc)}, "
                             f"{signed(r.cluster_bootstrap_95_high_auroc)}] "
                             f"({r.paired_seed_rows}/51)")
        pair_rows.append([name, *cells])
    parts.append("### Paired NSA − IForest AUROC\n")
    parts.append(md_table(["NSA method", "Original mean [95% interval] (pairs)",
                           "PCA(2) mean [95% interval] (pairs)"], pair_rows) + "\n")
    parts.append("Intervals are percentile bootstrap intervals from 5,000 resamples of the **17 datasets**, "
                 "preserving the matched seeds inside each dataset. They describe uncertainty across "
                 "datasets; they are unadjusted and should not be read as multiple-comparison tests. "
                 "Every complete-coverage NSA–IForest AUROC interval is below zero in both representations. "
                 "NSA/dual versus IForest was −0.142 [−0.221, −0.067] on 49/51 original-feature pairs and "
                 "−0.048 [−0.094, −0.010] on all 51 PCA(2) pairs. The PCA(2) dual AP mean, 0.465, was "
                 "above IForest's 0.402; KNN, ABOD and OCSVM had higher AP means of 0.483, 0.477 and "
                 "0.473. AP is metric- and prevalence-sensitive, so this does not reverse the AUROC result.\n")
    parts.append("![Three-split paired AUROC differences between NSA strategies and IForest]"
                 "(pyod_nsa_auroc_vs_iforest.png)\n")

    parts.append("## Which PyOD detectors need outlier training labels?\n")
    parts.append("Within the **complete 43-entry tabular baseline registry used here**, the methods whose "
                 "intended supervised training needs labeled anomalous examples are **XGBOD** and "
                 "**DevNet**. Both have a required `y` argument and `fit(X_normal)` raised `TypeError` "
                 "in a separate 24-normal-row execution probe. Each was recorded as "
                 "`requires_labeled_outliers` in 17 original-feature and 17 PCA(2) cells, for "
                 "**34 cells per method**. "
                 "No test labels were passed into their training.\n")
    parts.append("The all-zero-label probe is an important qualification: `fit(X_normal, y=zeros)` **did "
                 "execute** for both. XGBOD exposed only class 0 and gave one constant anomaly score on "
                 "the 24 rows. DevNet ran one epoch and returned varying scores despite reporting zero "
                 "outliers. Thus the exact claim is that they **cannot be trained for their intended "
                 "labeled-outlier discrimination under this protocol**, not that every possible all-zero "
                 "call throws. The probe results and constructor settings are saved separately.\n")
    parts.append("The other 41 baseline registry entries do not require anomaly examples at fit time. "
                 "Some, including LUNAR and the GAAL variants, create synthetic negative examples "
                 "internally from normal-only data; this obeys the restriction on using real dataset "
                 "outliers during training. Non-tabular wrappers such as EmbeddingOD and MultiModalOD "
                 "may inherit a label requirement if configured with XGBOD or DevNet, but they add no "
                 "separate always-label-dependent detector to this tabular roster. The registered "
                 "graph-only SCAN_Graph detector accepts optional `y` but needs graph structure, so it was also "
                 "outside this tabular benchmark. "
                 "MAD was excluded here because it accepts a single feature, not because it needs "
                 "outliers. LOCI, RGraph, ROD and LMDD had 15-second timeouts; CD and PCA had isolated "
                 "nonfinite-score errors in the original-feature panel. These outcomes do **not** imply "
                 "that the methods need anomalous training rows. PyOD explicitly distinguishes "
                 "[novelty detection from outlier detection](https://pyod.readthedocs.io/en/latest/relevant_knowledge.html), "
                 "and its [API documentation](https://pyod.readthedocs.io/en/latest/) notes that `y` is "
                 "ignored by unsupervised detectors.\n")

    parts.append("## Interpretation for PR #747\n")
    parts.append("The [open PR](https://github.com/yzhao062/pyod/pull/747) review asks for a defined operating "
                 "regime and evidence. This experiment supplies the requested clean-normal training "
                 "test, but it **cannot support a broad claim that NSA is superior because other PyOD "
                 "detectors cannot work without training outliers**. Many competitors succeeded.\n")
    parts.append("There are exploratory, dataset-specific leads: on PCA(2) cardio, NSA/deterministic averaged "
                 "0.942 AUROC over the three selected splits versus IForest 0.928, while the single-split "
                 "all-baseline best was INNE at 0.942 versus NSA/deterministic 0.950. On original-feature "
                 "vertebral, NSA/fixed averaged 0.554 versus IForest 0.431, but its absolute discrimination "
                 "was weak. On glass the seed-42 PCA(2) oracle NSA advantage used only four anomalous "
                 "test rows. These observations need a prospective strategy rule and more data before "
                 "becoming an operating-regime claim.\n")
    parts.append("The report addresses the reviewer's evidence question only. The separate requests to "
                 "remove the package-level NSA export and reduce the PR's size remain code-review work. "
                 "No GitHub comment, commit, or PR update was made while preparing this report.\n")

    parts.append("## Limitations and evidence files\n")
    parts.append("- The 600-normal-row and 2,000-test-row caps, 15-second budget and reduced deep-model "
                 "budgets favor tractable breadth over paper-default optimization. Methods were not "
                 "individually tuned.\n")
    parts.append("- PCA(2) retained only 16.2% of normal-training variance for arrhythmia, 21.4% for "
                 "MNIST and 24.4% for optdigits; inspect the per-dataset table before generalizing its "
                 "results back to the original feature space.\n")
    parts.append("- Some PyOD `decision_function` implementations use the whole unlabeled query batch, "
                 "so batch-dependent baselines do not measure the exact same inductive prediction task. "
                 "Test labels were never used for model fitting.\n")
    parts.append("- The best-per-dataset tables select with held-out labels and deliberately overstate what "
                 "a deployable selection policy could achieve. Rare test anomalies make some individual "
                 "dataset comparisons unstable.\n")
    parts.append("- This is a fixed PyOD source-tree snapshot and an explicitly specified CPU screen, not "
                 "a claim about every future PyOD release or every hyperparameter setting. PYTHONPATH "
                 "resolved to the checked-out PR source reporting version 3.6.6; the virtual environment's "
                 "installed-distribution metadata reports 3.6.5. The source tree hash is the controlling "
                 "implementation identifier.\n")
    parts.append("The accompanying CSVs include all 1,904 primary outcomes, both panels' 56-method "
                 "summaries, per-dataset diagnostics, all primary NSA–baseline pairings, the 1,836 "
                 "repeated-run records, and all repeated-run pairings. The reproducibility ZIP includes "
                 "raw JSONL records, scripts, manifests, and the synthetic supervised probe. Original "
                 "ODDS data bytes are linked and hashed in the dataset manifest, rather than repackaged.\n")

    report_path = OUTPUTS / "pyod_nsa_oneclass_report.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")

    family_data = sens_m[sens_m.family == "NSA"].copy()
    family_data.insert(2, "strategy_family",
                       family_data.method.map(lambda name: name.split("/")[1]))
    family_data.insert(3, "binary_match_rule",
                       family_data.method.map(
                           lambda name: name.split("/")[2] if name.startswith("NSA/binary/") else ""))
    family_csv = SENSITIVITY / "nsa_family_comparison.csv"
    family_data.to_csv(family_csv, index=False)

    copies = {
        ANALYSIS / "all_results.csv": "pyod_nsa_all_results.csv",
        ANALYSIS / "native_methods.csv": "pyod_nsa_native_methods.csv",
        ANALYSIS / "pca2_methods.csv": "pyod_nsa_pca2_methods.csv",
        ANALYSIS / "native_datasets.csv": "pyod_nsa_native_datasets.csv",
        ANALYSIS / "pca2_datasets.csv": "pyod_nsa_pca2_datasets.csv",
        ANALYSIS / "native_pairwise.csv": "pyod_nsa_native_pairwise.csv",
        ANALYSIS / "pca2_pairwise.csv": "pyod_nsa_pca2_pairwise.csv",
        ANALYSIS / "paired_auroc_vs_iforest.png": "pyod_nsa_auroc_vs_iforest.png",
        SENSITIVITY / "sensitivity_records.csv": "pyod_nsa_sensitivity_records.csv",
        SENSITIVITY / "sensitivity_methods.csv": "pyod_nsa_sensitivity_methods.csv",
        SENSITIVITY / "sensitivity_pairwise.csv": "pyod_nsa_sensitivity_pairwise.csv",
        family_csv: "pyod_nsa_family_comparison.csv",
        ROOT / "dataset_manifest.json": "pyod_nsa_dataset_manifest.json",
        ROOT / "run_manifest.json": "pyod_nsa_run_manifest.json",
        ROOT / "supervised_probe_results.json": "pyod_nsa_supervised_probe_results.json",
    }
    for source, name in copies.items():
        shutil.copy2(source, OUTPUTS / name)

    archive = OUTPUTS / "pyod_nsa_reproducibility.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=7) as bundle:
        bundle.write(report_path, arcname="report.md")
        for source in copies:
            bundle.write(source, arcname=str(source.relative_to(ROOT)))
        for name in ("loader.py", "build_manifest.py", "model_factory.py",
                     "runner.py", "scheduler.py", "selected_runner.py",
                     "analyze.py", "sensitivity_analysis.py", "plot.py",
                     "supervised_probe.py", "make_deliverables.py"):
            bundle.write(ROOT / name, arcname=f"scripts/{name}")
        for directory, label in ((ROOT / "raw", "raw/primary"),
                                 (ROOT / "sensitivity" / "raw", "raw/sensitivity")):
            for source in sorted(directory.glob("*.jsonl")):
                bundle.write(source, arcname=f"{label}/{source.name}")
        bundle.writestr(
            "README.txt",
            "PyOD NSA normal-only training benchmark, 2026-10-12.\n"
            "Read report.md and the two manifests first. The original ODDS .mat files "
            "are not included; dataset_manifest.json has pinned URLs and SHA-256 values. "
            "To rerun, set PYTHONPATH to the PR checkout at the recorded source tree, "
            "install the package versions in run_manifest.json, and run scripts/runner.py "
            "and scripts/selected_runner.py using a directory of the verified .mat files. "
            "scripts/analyze.py and scripts/sensitivity_analysis.py validate coverage.\n",
        )
    print(report_path)
    print(archive)
    print(f"report lines: {len(report_path.read_text().splitlines())}")


if __name__ == "__main__":
    run()
