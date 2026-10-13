"""Build a review-only Markdown report from the validated retention tables.

Run retention_analysis.py first. This generator reads its complete CSV outputs,
checks the expected grids, and writes a local draft. It does not fit models,
alter a pull request, or publish results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NSA = (
    "NSA/fixed", "NSA/variable", "NSA/coverage", "NSA/grid",
    "NSA/hierarchical", "NSA/voronoi", "NSA/deterministic",
    "NSA/suppressed", "NSA/dual", "NSA/annealed",
    "NSA/binary/hamming", "NSA/binary/rcontiguous", "NSA/binary/rchunk",
)
BASELINES = ("IForest", "ECOD", "OCSVM", "KNN", "ABOD")
METHODS = NSA + BASELINES
REPS = ("native", "pca2")
SIZES = (100, 75, 50, 25)
REDUCED = (75, 50, 25)
DISPLAY = {method: method.removeprefix("NSA/").replace("binary/", "binary · ")
           for method in NSA}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"validated analysis table not yet present: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader)


def _index(rows: list[dict[str, str]], fields: tuple[str, ...],
           expected: set[tuple], label: str) -> dict[tuple, dict[str, str]]:
    found: dict[tuple, dict[str, str]] = {}
    for row in rows:
        key = tuple(int(row[field]) if field == "retention_pct" else row[field]
                    for field in fields)
        if key in found:
            raise ValueError(f"duplicate {label} row: {key}")
        found[key] = row
    if set(found) != expected:
        raise ValueError(f"{label} grid incomplete: expected {len(expected)}, "
                         f"found {len(found)}, missing={sorted(expected-set(found))[:3]}, "
                         f"extra={sorted(set(found)-expected)[:3]}")
    return found


def load_tables(analysis_dir: Path) -> dict[str, dict]:
    summary = _read_csv(analysis_dir / "retention_method_summary.csv")
    change = _read_csv(analysis_dir / "retention_change_vs_full.csv")
    datasets = _read_csv(analysis_dir / "retention_dataset_changes.csv")
    cross = _read_csv(analysis_dir / "retention_nsa_vs_baseline.csv")
    interaction = _read_csv(analysis_dir / "retention_interaction.csv")
    exclusion = _read_csv(analysis_dir / "retention_exclusion_sensitivity.csv")
    table = {
        "summary": _index(summary, ("representation", "retention_pct", "method"),
                          {(r, p, m) for r in REPS for p in SIZES for m in METHODS},
                          "method summary"),
        "change": _index(change, ("representation", "retention_pct", "method"),
                         {(r, p, m) for r in REPS for p in REDUCED for m in METHODS},
                         "paired change"),
        "cross": _index(cross, ("representation", "retention_pct", "nsa", "baseline"),
                        {(r, p, n, b) for r in REPS for p in REDUCED
                         for n in NSA for b in BASELINES}, "NSA versus baseline"),
        "interaction": _index(
            interaction, ("representation", "retention_pct", "nsa", "baseline"),
            {(r, p, n, b) for r in REPS for p in REDUCED
             for n in NSA for b in BASELINES}, "training-size interaction"),
    }
    names = json.loads((ROOT / "dataset_manifest.json").read_text())["datasets"]
    table["dataset"] = _index(
        datasets, ("representation", "retention_pct", "method", "dataset"),
        {(r, p, m, d["name"]) for r in REPS for p in REDUCED
         for m in METHODS for d in names}, "dataset-level paired change")
    excluded_sets = {"none", "vowels+letter", *(item["name"] for item in names)}
    table["exclusion"] = _index(
        exclusion, ("representation", "retention_pct", "method", "excluded_datasets"),
        {("pca2", 25, method, excluded) for method in ("NSA/dual", "NSA/voronoi")
         for excluded in excluded_sets}, "post-hoc exclusion sensitivity")
    for key, rows in table.items():
        if key in {"dataset", "exclusion"}:
            continue
        for identity, row in rows.items():
            if key == "summary":
                n = int(row["success_seed_rows"])
                denom = int(row["expected_seed_rows"])
            else:
                n = int(row["paired_seed_rows"])
                denom = int(row["expected_pairs"])
            if denom != 51 or not 0 <= n <= denom:
                raise ValueError(f"invalid coverage in {key}: {identity}")
            declared = row["full_coverage"].lower() == "true"
            if declared != (n == denom):
                raise ValueError(f"full_coverage mismatch in {key}: {identity}")
    for identity, row in table["exclusion"].items():
        excluded = identity[-1]
        count = 17 if excluded == "none" else 15 if excluded == "vowels+letter" else 16
        if int(row["remaining_datasets"]) != count or row["exploratory_posthoc"].lower() != "true":
            raise ValueError(f"invalid exclusion sensitivity metadata: {identity}")
    return table


def _number(text: str | None) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    value = float(text)
    if not -1 <= value <= 1:
        raise ValueError(f"out-of-range metric or paired difference: {value}")
    return value


def _fmt(value: float | None, signed: bool = False) -> str:
    return "—" if value is None else (f"{value:+.3f}" if signed else f"{value:.3f}")


def _score(row: dict[str, str], metric: str) -> str:
    value = _number(row[f"macro_{metric}_available"])
    n = int(row["success_seed_rows"])
    denom = int(row["expected_seed_rows"])
    if (value is None) != (n == 0):
        raise ValueError(f"score/coverage mismatch: {row['method']} {metric}")
    return f"{_fmt(value)} ({n}/{denom})"


def _delta(row: dict[str, str], metric: str, ci: bool = True) -> str:
    value = _number(row[f"macro_mean_delta_{metric}"])
    n = int(row["paired_seed_rows"])
    denom = int(row["expected_pairs"])
    if (value is None) != (n == 0):
        raise ValueError(f"paired score/coverage mismatch: {row}")
    if value is None:
        return f"— (0/{denom})"
    result = _fmt(value, signed=True)
    if ci:
        low = _number(row[f"cluster_bootstrap_95_low_{metric}"])
        high = _number(row[f"cluster_bootstrap_95_high_{metric}"])
        if low is None or high is None or low > high:
            raise ValueError(f"invalid paired interval: {row}")
        result += f" [{_fmt(low, True)}, {_fmt(high, True)}]"
    return result + f" ({n}/{denom})"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    escape = lambda cell: str(cell).replace("|", "\\|").replace("\n", " ")
    return "\n".join(
        ["| " + " | ".join(map(escape, headers)) + " |",
         "| " + " | ".join("---" for _ in headers) + " |", *(
             "| " + " | ".join(map(escape, row)) + " |" for row in rows
         )]
    )


def _method_scores(table: dict, rep: str, methods: tuple[str, ...]) -> str:
    rows = []
    for method in methods:
        name = DISPLAY.get(method, method)
        cells = [name]
        for pct in SIZES:
            row = table["summary"][(rep, pct, method)]
            cells.append(f"{_score(row, 'auroc')} / {_score(row, 'ap')}")
        rows.append(cells)
    return _table(["Method", *[f"{p}% retained: AUROC / AP" for p in SIZES]], rows)


def _method_changes(table: dict, rep: str, methods: tuple[str, ...],
                    metric: str) -> str:
    rows = []
    for method in methods:
        cells = [DISPLAY.get(method, method)]
        for pct in REDUCED:
            cells.append(_delta(table["change"][(rep, pct, method)], metric))
        rows.append(cells)
    return _table(["Method", *[f"{p}% retained: paired Δ{metric.upper()} [95% interval]"
                               for p in REDUCED]], rows)


def _status_table(table: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in SIZES:
            for label, methods in (("NSA", NSA), ("Control", BASELINES)):
                counts = Counter()
                for method in methods:
                    record = table["summary"][(rep, pct, method)]
                    statuses = json.loads(record["status_counts"])
                    if sum(int(x) for x in statuses.values()) != 51:
                        raise ValueError(f"status counts incomplete: {rep}/{pct}/{method}")
                    counts.update({key: int(value) for key, value in statuses.items()})
                expected = 51 * len(methods)
                if sum(counts.values()) != expected:
                    raise AssertionError("status count total drift")
                issues = ", ".join(f"{key} {counts[key]}" for key in sorted(counts)
                                   if key != "success") or "none"
                rows.append([rep, str(pct), label, f"{counts['success']}/{expected}", issues])
    return _table(["Features", "% retained", "Group", "Successful runs", "Other recorded outcomes"], rows)


def _interaction_summary(table: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in REDUCED:
            for baseline in BASELINES:
                entries = [table["interaction"][(rep, pct, method, baseline)]
                           for method in NSA]
                available = [row for row in entries if int(row["paired_seed_rows"]) > 0]
                full = sum(row["full_coverage"].lower() == "true" for row in entries)
                positive = sum(_number(row["macro_mean_delta_auroc"]) > 0
                               for row in available)
                above = sum(_number(row["cluster_bootstrap_95_low_auroc"]) > 0
                            for row in available)
                below = sum(_number(row["cluster_bootstrap_95_high_auroc"]) < 0
                            for row in available)
                rows.append([rep, str(pct), baseline, f"{len(available)}/13",
                             f"{full}/13", f"{positive}/{len(available)}",
                             str(above), str(below)])
    return _table(["Features", "% retained", "Control", "NSA configurations with pairs",
                   "Full coverage", "Positive mean interaction", "Intervals above zero",
                   "Intervals below zero"], rows)


def _cross_summary(table: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in REDUCED:
            for baseline in BASELINES:
                entries = [table["cross"][(rep, pct, method, baseline)]
                           for method in NSA]
                available = [row for row in entries if int(row["paired_seed_rows"]) > 0]
                full = sum(row["full_coverage"].lower() == "true" for row in entries)
                positive = sum(_number(row["macro_mean_delta_auroc"]) > 0
                               for row in available)
                above = sum(_number(row["cluster_bootstrap_95_low_auroc"]) > 0
                            for row in available)
                below = sum(_number(row["cluster_bootstrap_95_high_auroc"]) < 0
                            for row in available)
                rows.append([rep, str(pct), baseline, f"{len(available)}/13",
                             f"{full}/13", f"{positive}/{len(available)}",
                             str(above), str(below)])
    return _table(["Features", "% retained", "Control", "NSA configurations with pairs",
                   "Full coverage", "Positive NSA − control mean", "Intervals above zero",
                   "Intervals below zero"], rows)


def _overview(table: dict) -> tuple[str, list[str]]:
    rows = []
    conclusions = []
    for rep in REPS:
        for pct in REDUCED:
            entries = [table["change"][(rep, pct, method)] for method in NSA]
            complete = [row for row in entries if row["full_coverage"].lower() == "true"]
            partial = [row for row in entries if 0 < int(row["paired_seed_rows"]) < 51]
            zero = sum(int(row["paired_seed_rows"]) == 0 for row in entries)
            up = sum(_number(row["macro_mean_delta_auroc"]) > 0 for row in complete)
            down = sum(_number(row["macro_mean_delta_auroc"]) < 0 for row in complete)
            ci_up = sum(_number(row["cluster_bootstrap_95_low_auroc"]) > 0
                        for row in complete)
            ci_down = sum(_number(row["cluster_bootstrap_95_high_auroc"]) < 0
                          for row in complete)
            rows.append([rep, str(pct), f"{len(complete)}/13", str(up), str(down),
                         str(ci_up), str(ci_down), str(len(partial)), str(zero)])
        consistent = [method for method in NSA if all(
            table["change"][(rep, pct, method)]["full_coverage"].lower() == "true"
            and _number(table["change"][(rep, pct, method)]["macro_mean_delta_auroc"]) > 0
            for pct in REDUCED)]
        intervals = [method for method in NSA if all(
            table["change"][(rep, pct, method)]["full_coverage"].lower() == "true"
            and _number(table["change"][(rep, pct, method)]["cluster_bootstrap_95_low_auroc"]) > 0
            for pct in REDUCED)]
        if consistent:
            conclusions.append(f"In {rep}, {', '.join(DISPLAY[m] for m in consistent)} "
                               "had positive paired mean AUROC changes at all three reduced sizes "
                               "with complete coverage.")
        else:
            conclusions.append(f"In {rep}, no NSA configuration had positive paired mean "
                               "AUROC changes at all three reduced sizes with complete coverage.")
        if intervals:
            conclusions.append(f"In {rep}, the exploratory 95% interval was entirely "
                               "above zero at all three sizes for "
                               f"{', '.join(DISPLAY[m] for m in intervals)}.")
        else:
            conclusions.append(f"In {rep}, no configuration had an exploratory 95% "
                               "interval entirely above zero at every reduced size.")
        iforest_better = [method for method in NSA if all(
            table["cross"][(rep, pct, method, "IForest")]["full_coverage"].lower() == "true"
            and _number(table["cross"][(rep, pct, method, "IForest")]["macro_mean_delta_auroc"]) > 0
            for pct in REDUCED)]
        if iforest_better:
            conclusions.append(f"In {rep}, {', '.join(DISPLAY[m] for m in iforest_better)} "
                               "had higher same-size mean AUROC than IForest at all three reduced "
                               "sizes with complete paired coverage.")
        else:
            conclusions.append(f"In {rep}, no NSA configuration had higher same-size mean "
                               "AUROC than IForest at all three reduced sizes with complete "
                               "paired coverage.")
    overview = _table(["Features", "% retained", "Complete NSA configs", "Positive mean",
                       "Negative mean", "Intervals > 0", "Intervals < 0",
                       "Partial", "No pairs"], rows)
    return overview, conclusions


def _notable_pca2_results(table: dict) -> list[str]:
    """Describe predeclared family results without selecting a universal winner."""
    lines = []
    for method in ("NSA/dual", "NSA/voronoi"):
        full = table["summary"][("pca2", 100, method)]
        quarter = table["summary"][("pca2", 25, method)]
        change = table["change"][("pca2", 25, method)]
        lines.append(
            f"- **PCA2 {DISPLAY[method]}:** mean AUROC was "
            f"{_score(full, 'auroc')} at 100% and {_score(quarter, 'auroc')} "
            f"at 25% retained. Its matched AUROC change was "
            f"{_delta(change, 'auroc')}; matched AP change was "
            f"{_delta(change, 'ap')}."
        )
    lines.append(
        "- **AP qualification:** neither PCA2 dual nor PCA2 Voronoi improved "
        "average precision at 25% retained in the paired mean; both AP "
        "bootstrap intervals include zero. An AUROC gain therefore does not "
        "translate into an observed AP gain here."
    )
    dual_iforest = table["cross"][("pca2", 25, "NSA/dual", "IForest")]
    dual_knn = table["cross"][("pca2", 25, "NSA/dual", "KNN")]
    knn = table["summary"][("pca2", 25, "KNN")]
    lines.append(
        "- **Same-size controls for PCA2 dual at 25% retained:** NSA − IForest "
        f"was {_delta(dual_iforest, 'auroc')}; NSA − KNN was "
        f"{_delta(dual_knn, 'auroc')}. KNN's mean AUROC was "
        f"{_score(knn, 'auroc')}. A gain over dual's own 100% result alone "
        "does not show that it outperforms these controls."
    )
    rows = [row for (rep, pct, method, _dataset), row in table["dataset"].items()
            if rep == "pca2" and pct == 25 and method == "NSA/dual"
            and row["mean_delta_auroc"]]
    if rows:
        top = sorted(rows, key=lambda row: float(row["mean_delta_auroc"]), reverse=True)[:2]
        examples = ", ".join(
            f"{row['dataset']} {_fmt(_number(row['mean_delta_auroc']), True)} "
            f"({row['paired_success_seed_rows']}/3 seeds)" for row in top
        )
        lines.append(
            f"- **Dataset concentration:** the two largest positive PCA2 dual changes "
            f"at 25% retained were {examples}. These are inspected contributors "
            "to the macro mean, not preselected success cases or an independent validation. "
            "Because scaling and PCA are refit, representation drift may contribute."
        )
    return lines


def _exclusion_section(table: dict) -> str:
    rows = []
    for method in ("NSA/dual", "NSA/voronoi"):
        for excluded in ("none", "vowels+letter"):
            row = table["exclusion"][("pca2", 25, method, excluded)]
            auc = _number(row["macro_mean_delta_auroc"])
            lo = _number(row["cluster_bootstrap_95_low_auroc"])
            hi = _number(row["cluster_bootstrap_95_high_auroc"])
            ap = _number(row["macro_mean_delta_ap"])
            ap_lo = _number(row["cluster_bootstrap_95_low_ap"])
            ap_hi = _number(row["cluster_bootstrap_95_high_ap"])
            if None in (auc, lo, hi, ap, ap_lo, ap_hi):
                raise ValueError(f"missing exclusion sensitivity metric: {method}/{excluded}")
            rows.append([DISPLAY[method],
                         "None" if excluded == "none" else "Vowels and letter",
                         row["remaining_datasets"],
                         f"{_fmt(auc, True)} [{_fmt(lo, True)}, {_fmt(hi, True)}]",
                         f"{_fmt(ap, True)} [{_fmt(ap_lo, True)}, {_fmt(ap_hi, True)}]",
                         row["remaining_dataset_wins_auroc"]])
    return _table(["NSA configuration", "Datasets excluded", "Datasets left",
                   "Paired ΔAUROC [95% interval]", "Paired ΔAP [95% interval]",
                   "Datasets with AUROC gain"], rows)


def make_report(table: dict, dataset_manifest: dict, run_manifest: dict) -> str:
    overview, conclusions = _overview(table)
    n_datasets = len(dataset_manifest["datasets"])
    if n_datasets != 17:
        raise ValueError("dataset manifest no longer lists 17 ODDS files")
    parts = [
        "# NSA normal-training-size experiment — report",
        "",
        "**Review copy. Nothing in this report has been posted to GitHub.**",
        "",
        "## Question and answer",
        "",
        "This experiment tests whether reducing the number of normal training examples "
        "improves the 11 NSA strategy families in [PyOD PR #747](https://github.com/yzhao062/pyod/pull/747). "
        "The binary family has three matching rules, so the tables contain 13 distinct NSA configurations. "
        "Anomalous feature rows are held out for evaluation; labels stratify the split and identify "
        "training anomalies to discard, but are never passed to model fitting. A positive paired difference "
        "means that the reduced normal-only training set scored higher AUROC or AP than the matched 100% run.",
        "",
        *conclusions,
        "",
        "These are descriptive outcomes of the tested datasets, splits, implementations, and fixed settings. "
        "A positive result for one configuration does not establish general NSA superiority; "
        "the same-size baseline comparisons below test that separate claim.",
        "",
        "### Notable measured results",
        "",
        *_notable_pca2_results(table),
        "",
        "## Experimental design",
        "",
        f"- **Data:** {n_datasets} ODDS `.mat` datasets bundled with PyOD's "
        "[benchmark collection](https://github.com/yzhao062/pyod/blob/master/notebooks/benchmark.py). "
        f"Dataset hashes and original source links are in `dataset_manifest.json`; the tested PyOD tree is "
        f"`{run_manifest['source_git_tree']}`.",
        "- **Splits:** seeds 42, 43, and 44; each uses a 60/40 stratified train/test split. "
        "Every anomalous row in the training partition is discarded before preprocessing or fitting. "
        "The original fit pool is capped at 600 normals; the mixed held-out test set is capped at 2,000 rows "
        "with its class mix preserved.",
        "- **Retention:** 100%, 75%, 50%, and 25% of the original capped normal fit pool. "
        "One seed-fixed random permutation of the parent normal indices supplies nested reduced subsets. "
        "The order of selected rows follows the parent fit set. Thus '25% retained' also means '75% removed'.",
        "- **Features:** original features and PCA to two dimensions. StandardScaler and PCA are refit "
        "using only the normals retained at each size. Every model is fitted with `fit(X)`; no anomaly label "
        "or anomaly feature row is passed to the model. The test indices and labels remain fixed across sizes.",
        "- **Methods:** all 11 PR NSA families (13 configurations because binary has Hamming, "
        "r-contiguous, and r-chunk rules) plus five predeclared controls: IForest, ECOD, OCSVM, KNN, ABOD. "
        "Constructor settings match the previous experiment; each method has a 15-second wall-time limit.",
        "- **Evaluation:** held-out mixed-class AUROC and average precision (AP). "
        "No test label is used for fitting or hyperparameter selection. Run failures and geometry exclusions "
        "stay in the denominator and never become zero metric scores.",
        "",
        "The analysis checks the complete 1,836-row 100% reference grid and 5,508-row reduced grid, "
        "source dataset hashes, zero fitted outliers, exact retained normal index sets and order, "
        "unchanged test indices, and status/metric validity. The reference uses the prior full-roster seed-42 "
        "runs and predeclared selected-method seed-43/44 runs. An independent numerical audit verified that "
        "the 100% native and PCA2 arrays match the previous preprocessing in every dataset/seed cell.",
        "",
        "## How to read the statistics",
        "",
        "A score cell below gives the equal-dataset-weight mean among successful runs followed by successful "
        "dataset/seed runs out of 51. Different score cells can use different successful subsets; use the "
        "paired-change tables for direct size comparisons. Each paired difference is computed within the "
        "same dataset and seed, averaged over available seeds in each dataset, and then averaged equally over "
        "datasets. Brackets are descriptive 95% dataset-cluster bootstrap intervals (5,000 resamples). "
        "They are not adjusted for the many strategy, size, representation, and baseline comparisons.",
        "",
        "### At-a-glance NSA AUROC changes",
        "",
        overview,
        "",
        "Only complete-coverage configurations contribute to the positive/negative and interval counts "
        "in this overview. Partial and zero-pair configurations are counted separately.",
        "",
        "## Results by NSA family",
        "",
    ]
    for rep, title in (("native", "Original features"), ("pca2", "PCA to two dimensions")):
        parts += [f"### {title}", "",
                  "The first table shows mean AUROC / AP at each retained size. "
                  "The next two tables compare each reduced size with its matched 100% run.", "",
                  _method_scores(table, rep, NSA), "",
                  "**Paired AUROC change versus 100% training**", "",
                  _method_changes(table, rep, NSA, "auroc"), "",
                  "**Paired AP change versus 100% training**", "",
                  _method_changes(table, rep, NSA, "ap"), ""]
    parts += [
        "## Baseline controls",
        "",
        "The same train-size reduction is applied to each control. ECOD can use the whole test batch "
        "during scoring, so its evaluation is not strictly pointwise inductive; IForest, OCSVM, KNN, "
        "and ABOD are the primary inductive controls here. The table cells keep the same coverage convention.",
        "",
    ]
    for rep, title in (("native", "Original features"), ("pca2", "PCA to two dimensions")):
        parts += [f"### {title}", "", _method_scores(table, rep, BASELINES), "",
                  "**Controls' paired AUROC change versus 100% training**", "",
                  _method_changes(table, rep, BASELINES, "auroc"), "",
                  "**Controls' paired AP change versus 100% training**", "",
                  _method_changes(table, rep, BASELINES, "ap"), ""]
    parts += [
        "## NSA versus controls at the same training size",
        "",
        "This comparison asks whether NSA has higher held-out AUROC than a control trained "
        "on the same retained normal set. Positive values favor NSA. Each comparison is paired "
        "by dataset, seed, representation, and test set; the full table also includes AP.",
        "",
        _cross_summary(table), "",
    ]
    for rep, title in (("native", "Original features"), ("pca2", "PCA to two dimensions")):
        rows = [[DISPLAY[method], *[
            _delta(table["cross"][(rep, pct, method, "IForest")], "auroc")
            for pct in REDUCED]] for method in NSA]
        parts += [f"### {title}: NSA minus IForest at the same training size", "",
                  _table(["NSA configuration", *[f"{pct}% retained: AUROC difference [95% interval]"
                                                for pct in REDUCED]], rows), ""]
    parts += [
        "## Relative benefit compared with the controls",
        "",
        "For each NSA/control pair, the interaction is **(NSA at reduced size − NSA at 100%) "
        "− (control at reduced size − control at 100%)**, using only dataset/seed cells where all four "
        "fits succeeded. Positive values mean NSA gained more (or lost less) from reducing training data "
        "than that control. This is a relative training-size effect; it does not mean NSA has the higher "
        "absolute AUROC. The first table includes all five controls; the next tables show the "
        "per-configuration comparison with IForest. The complete 390-row interaction CSV contains every "
        "NSA/control interval and coverage count.",
        "",
        _interaction_summary(table), "",
    ]
    for rep, title in (("native", "Original features"), ("pca2", "PCA to two dimensions")):
        rows = [[DISPLAY[method], *[
            _delta(table["interaction"][(rep, pct, method, "IForest")], "auroc")
            for pct in REDUCED]] for method in NSA]
        parts += [f"### {title}: NSA training-size interaction versus IForest", "",
                  _table(["NSA configuration", *[f"{pct}% retained: AUROC interaction [95% interval]"
                                                for pct in REDUCED]], rows), ""]
    parts += [
        "The same-size comparisons above assess absolute relative AUROC at each reduced size; "
        "a positive interaction alone cannot establish that claim.",
        "",
        "## Exploratory dataset exclusion check",
        "",
        "The PCA2 dual and Voronoi gains at 25% retained were inspected after seeing "
        "the dataset-level changes. The table compares the original 17-dataset mean with "
        "a recomputation excluding vowels and letter, the two largest positive "
        "contributors for dual. This exclusion was chosen **after** observing outcomes; "
        "it is a stress check, not a prespecified test or independent replication.",
        "",
        _exclusion_section(table),
        "",
        "Both AUROC intervals include zero after excluding those two datasets. "
        "The full `retention_exclusion_sensitivity.csv` also reports each one-dataset "
        "exclusion for these two configurations.",
        "",
        "## Completion and failure accounting",
        "",
        _status_table(table),
        "",
        "A no-pair cell means no valid within-dataset/seed reduced-versus-full comparison was available. "
        "Geometry exclusions (for example, deterministic NSA outside two dimensions), detector-generation "
        "errors, and timeouts are reported separately in the underlying status counts. A method that "
        "cannot fit is not assigned an AUROC or AP of zero.",
        "",
        "## Limits on interpretation",
        "",
        "- This is a fixed-parameter screen on 17 datasets with three splits and one nested subset draw per split. "
        "Its many comparisons are exploratory; the 95% intervals are descriptive and unadjusted.",
        "- The 600-normal fit cap and 2,000-row test cap reduce runtime and do not reproduce full-dataset training. "
        "The smallest 25% fit sets have 21 normals on lympho, 30 on glass, and 31 on vertebral.",
        "- Refit scaling and PCA make this a test of the full low-data pipeline. In the PCA panel, "
        "training-size changes can also rotate or rescale the two-dimensional representation. PCA may "
        "discard anomaly-relevant directions, especially in high-dimensional datasets.",
        "- A method with partial coverage can have an optimistic or pessimistic available-case mean. "
        "Its paired denominator and failure counts must accompany any claim.",
        "- The five control algorithms are a selected follow-up panel. The earlier 100%-training report "
        "contains the wider 43-detector comparison. This reduced-data experiment does not rerun all 43.",
        "- AP depends on held-out anomaly prevalence and is interpreted within the matched split. "
        "No outlier was available for training, but outliers in the test set are necessary to score AUROC/AP.",
        "- The PCA2 dual and Voronoi 25% AUROC gains have substantial contributions from "
        "vowels and letter. Their post-hoc exclusion intervals include zero; this limits "
        "claims of a stable cross-dataset gain.",
        "",
        "## Conclusion",
        "",
        *conclusions,
        "",
        "The evidence supports only the family, size, representation, and coverage-qualified statements "
        "shown above. PCA2 dual and Voronoi showed AUROC gains at 25% retained, "
        "but neither showed a paired AP gain; both AUROC gains weakened under the "
        "post-hoc vowels-and-letter exclusion. Compare the same-size NSA/control differences "
        "before claiming superiority. "
        "No PR response or external post was made from this report.",
        "",
        "## Reproducibility files",
        "",
        "- `retention/analysis/retention_method_summary.csv`: AUROC/AP, coverage, and status counts "
        "for 100%, 75%, 50%, and 25% training.",
        "- `retention/analysis/retention_change_vs_full.csv`: all paired size changes and bootstrap intervals.",
        "- `retention/analysis/retention_dataset_changes.csv`: dataset-level matched changes and statuses.",
        "- `retention/analysis/retention_nsa_vs_baseline.csv`: all same-size NSA/control comparisons.",
        "- `retention/analysis/retention_interaction.csv`: all relative training-size effects.",
        "- `retention/analysis/retention_exclusion_sensitivity.csv`: exploratory "
        "leave-one-dataset and vowels-plus-letter exclusion checks for PCA2 dual and Voronoi at 25%.",
        "- `retention/retention_manifest.json`: frozen source, protocol, package, "
        "input-hash, and output-hash provenance for this follow-up experiment.",
        "- `retention/analysis/retention_auroc_change.png`: per-configuration AUROC "
        "change figure with partial-coverage markers.",
        "- `dataset_manifest.json`, `run_manifest.json`, the raw JSONL records, "
        "`retention_analysis.py`, `runner.py`, `model_factory.py`, and `retention_plot.py`: "
        "source and reproducibility information.",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=ROOT / "retention" / "analysis")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "retention" / "report_draft.md")
    args = parser.parse_args()
    table = load_tables(args.analysis_dir)
    dataset_manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    run_manifest = json.loads((ROOT / "run_manifest.json").read_text())
    report = make_report(table, dataset_manifest, run_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote validated review draft ({len(report):,} characters) to {args.output}")


if __name__ == "__main__":
    main()
