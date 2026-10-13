"""Audit and analyze the reduced clean-normal training experiment.

The reduced panel has 17 ODDS datasets, 3 retention fractions (25, 50, 75%),
3 fixed splits, 2 feature representations, and 18 predeclared methods.  The
100% reference uses the original seed-42 screen and seed-43/44 sensitivity
panel.  Every comparison is paired on dataset, split seed, representation,
method, and original held-out row indices.  Failed fits stay in the coverage
denominator.  Bootstrap intervals resample whole datasets, retaining the
matched seeds within each dataset, and are descriptive rather than a
multiple-comparison-adjusted hypothesis test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from model_factory import METHOD_META, NSA_METHODS
from sensitivity_analysis import BASELINES, SELECTED_METHODS, _load_jsonl


REPRESENTATIONS = ("native", "pca2")
SEEDS = (42, 43, 44)
RETENTION_PCTS = (25, 50, 75)
EXPECTED_DATASETS = 17
EXPECTED_METHODS = 18
EXPECTED_REFERENCE_ROWS = EXPECTED_DATASETS * len(SEEDS) * len(REPRESENTATIONS) * EXPECTED_METHODS
EXPECTED_REDUCED_ROWS = EXPECTED_REFERENCE_ROWS * len(RETENTION_PCTS)


def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.sort(np.asarray(indices, dtype="<i8")).tobytes()).hexdigest()


def _hash_ordered_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _read_manifest(path: Path) -> tuple[list[str], Path, dict[str, dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items = manifest["datasets"]
    names = sorted(item["name"] for item in items)
    if len(names) != EXPECTED_DATASETS or len(set(names)) != EXPECTED_DATASETS:
        raise ValueError("dataset manifest must contain 17 distinct datasets")
    if len(SELECTED_METHODS) != EXPECTED_METHODS or len(set(SELECTED_METHODS)) != EXPECTED_METHODS:
        raise ValueError("selected roster must contain 18 distinct methods")
    return names, Path(manifest["source"]["local_directory"]), {item["name"]: item for item in items}


def _expected_splits(names: list[str], data_dir: Path, items: dict[str, dict]) -> dict:
    """Reconstruct legacy test and full normal fit indices from source files."""
    expected = {}
    for dataset in names:
        path = data_dir / items[dataset]["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != items[dataset]["sha256"]:
            raise ValueError(f"dataset checksum changed: {path}")
        y = np.asarray(loadmat(path)["y"]).ravel().astype(np.int8)
        if set(np.unique(y)) != {0, 1} or len(y) != items[dataset]["n_samples"]:
            raise ValueError(f"invalid source labels: {dataset}")
        for seed in SEEDS:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
            fit_idx = train_idx[y[train_idx] == 0]
            if len(fit_idx) > 600:
                rng = np.random.RandomState(seed + 10_000)
                fit_idx = np.sort(rng.choice(fit_idx, size=600, replace=False))
            if len(test_idx) > 2000:
                _, chosen = train_test_split(
                    np.arange(len(test_idx)), test_size=2000,
                    stratify=y[test_idx], random_state=seed + 20_000)
                test_idx = np.sort(test_idx[chosen])
            expected[(dataset, seed)] = {
                # Preserve the legacy split order below the 600-row cap:
                # retention_runner permutes this original order before taking
                # a prefix. The hash itself sorts indices for canonical form.
                "fit_indices": fit_idx.copy(),
                "fit_normals": len(fit_idx),
                "fit_parent_index_sha256": _hash_indices(fit_idx),
                "test_index_sha256": _hash_indices(test_idx),
                "test_normals": int(np.sum(y[test_idx] == 0)),
                "test_outliers": int(np.sum(y[test_idx] == 1)),
                "discarded_train_outliers": int(np.sum(y[train_idx] == 1)),
                "dataset_rows": len(y),
                "source_features": items[dataset]["n_features"],
            }
    return expected


def _index(rows: list[dict], include_pct: bool = False) -> dict:
    counts = Counter()
    indexed = {}
    for row in rows:
        try:
            key = (row["dataset"], row["representation"], int(row["seed"]), row["method"])
            if include_pct:
                pct = int(round(100 * float(row["retention_fraction"])))
                key = (*key, pct)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed record at {row.get('_source_file')}:{row.get('_source_line')}: {exc}") from exc
        counts[key] += 1
        indexed[key] = row
    duplicates = [key for key, n in counts.items() if n != 1]
    if duplicates:
        raise ValueError(f"duplicate records: {duplicates[:5]}")
    return indexed


def _validate_common(row: dict, key: tuple, split: dict) -> None:
    if row.get("fit_outliers") != 0:
        raise ValueError(f"outlier leaked into fit: {key}")
    if row.get("train_cap") != 600 or row.get("test_cap") != 2000:
        raise ValueError(f"protocol cap drift: {key}")
    for field in ("test_normals", "test_outliers", "discarded_train_outliers",
                  "dataset_rows", "source_features"):
        if row.get(field) != split[field]:
            raise ValueError(f"split/source metadata drift in {field}: {key}")
    if row.get("status") == "success":
        for metric in ("auroc", "ap"):
            value = row.get(metric)
            if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {metric} for successful {key}")


def load_and_validate(primary_raw_dir: Path, sensitivity_raw_dir: Path,
                      retention_raw_dir: Path, dataset_manifest: Path) -> tuple[list[dict], list[dict]]:
    names, data_dir, items = _read_manifest(dataset_manifest)
    expected_splits = _expected_splits(names, data_dir, items)
    reference = [r for r in _load_jsonl(primary_raw_dir, {42}) if r.get("method") in SELECTED_METHODS]
    reference.extend(_load_jsonl(sensitivity_raw_dir, {43, 44}))
    reduced = _load_jsonl(retention_raw_dir, set(SEEDS))
    reference_index = _index(reference)
    reduced_index = _index(reduced, include_pct=True)
    expected_reference = {(d, r, s, m) for d in names for r in REPRESENTATIONS
                          for s in SEEDS for m in SELECTED_METHODS}
    expected_reduced = {(*key, pct) for key in expected_reference for pct in RETENTION_PCTS}
    if set(reference_index) != expected_reference or len(reference) != EXPECTED_REFERENCE_ROWS:
        raise ValueError(f"100% reference grid incomplete: rows={len(reference)}, "
                         f"missing={list(expected_reference-set(reference_index))[:4]}, "
                         f"extra={list(set(reference_index)-expected_reference)[:4]}")
    if set(reduced_index) != expected_reduced or len(reduced) != EXPECTED_REDUCED_ROWS:
        raise ValueError(f"reduced grid incomplete: rows={len(reduced)}, "
                         f"missing={list(expected_reduced-set(reduced_index))[:4]}, "
                         f"extra={list(set(reduced_index)-expected_reduced)[:4]}")

    for key, row in reference_index.items():
        split = expected_splits[(key[0], key[2])]
        _validate_common(row, key, split)
        if row.get("fit_normals") != split["fit_normals"]:
            raise ValueError(f"100% fit count drift: {key}")

    for key, row in reduced_index.items():
        split = expected_splits[(key[0], key[2])]
        _validate_common(row, key, split)
        pct = key[-1]
        if pct not in RETENTION_PCTS or row.get("retention_fraction") != pct / 100:
            raise ValueError(f"retention fraction drift: {key}")
        if row.get("reduction_fraction") != 1 - pct / 100:
            raise ValueError(f"reduction fraction drift: {key}")
        if row.get("protocol_version") != "retention-v2-parent-order":
            raise ValueError(f"wrong retention run protocol: {key}")
        if row.get("original_fit_normals") != split["fit_normals"]:
            raise ValueError(f"parent fit count drift: {key}")
        if row.get("fit_parent_index_sha256") != split["fit_parent_index_sha256"]:
            raise ValueError(f"full-fit row identity drift: {key}")
        if row.get("test_index_sha256") != split["test_index_sha256"]:
            raise ValueError(f"held-out test rows changed: {key}")
        parent = split["fit_indices"]
        expected_count = max(2, int(np.floor(len(parent) * pct / 100)))
        if row.get("fit_normals") != expected_count:
            raise ValueError(f"reduced fit count drift: {key}")
        chosen = np.random.RandomState(key[2] + 30_000).permutation(parent)[:expected_count]
        retained = parent[np.isin(parent, chosen)]
        if row.get("fit_index_sha256") != _hash_indices(retained):
            raise ValueError(f"retained normal rows changed: {key}")
        if row.get("fit_index_ordered_sha256") != _hash_ordered_indices(retained):
            raise ValueError(f"retained normal row order changed: {key}")
        # A fraction uses the same retained training rows and held-out test
        # rows across all 18 methods and both representations. The transform
        # may differ because it is refit using only retained normals.
        ref = reference_index[key[:-1]]
        for field in ("test_normals", "test_outliers", "discarded_train_outliers",
                      "dataset_rows", "source_features"):
            if row[field] != ref[field]:
                raise ValueError(f"reduced/reference mismatch in {field}: {key}")

    # Verify that a given retention draw is identical across methods and
    # representations as a second guard on the raw run records.
    by_draw = defaultdict(set)
    for key, row in reduced_index.items():
        by_draw[(key[0], key[2], key[-1])].add((row["fit_normals"], row["fit_index_sha256"]))
    if any(len(variants) != 1 for variants in by_draw.values()):
        raise ValueError("retained normal rows differ across method or representation")
    return reference, reduced


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _macro_success(rows: list[dict], metric: str) -> tuple[float | None, int]:
    by_dataset = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            by_dataset[row["dataset"]].append(float(row[metric]))
    return _mean(np.mean(values) for values in by_dataset.values()), len(by_dataset)


def summarize_methods(reference: list[dict], reduced: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in reference:
        grouped[(row["representation"], 100, row["method"])].append(row)
    for row in reduced:
        grouped[(row["representation"], int(round(100 * row["retention_fraction"])), row["method"])].append(row)
    output = []
    for representation in REPRESENTATIONS:
        for pct in (100, *RETENTION_PCTS):
            for method in SELECTED_METHODS:
                rows = grouped[(representation, pct, method)]
                statuses = Counter(row["status"] for row in rows)
                auc, auc_datasets = _macro_success(rows, "auroc")
                ap, ap_datasets = _macro_success(rows, "ap")
                if len(rows) != EXPECTED_DATASETS * len(SEEDS) or auc_datasets != ap_datasets:
                    raise AssertionError("unexpected summary coverage")
                output.append({
                    "representation": representation,
                    "retention_pct": pct,
                    "reduction_pct": 100 - pct,
                    "method": method,
                    "family": METHOD_META[method]["family"],
                    "strategy": METHOD_META[method].get("strategy", ""),
                    "expected_seed_rows": len(rows),
                    "success_seed_rows": statuses.get("success", 0),
                    "success_datasets": auc_datasets,
                    "full_coverage": statuses.get("success", 0) == len(rows),
                    "macro_auroc_available": auc,
                    "macro_ap_available": ap,
                    "macro_auroc_full_coverage": auc if statuses.get("success", 0) == len(rows) else None,
                    "macro_ap_full_coverage": ap if statuses.get("success", 0) == len(rows) else None,
                    "status_counts": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
                })
    return output


def _cluster_interval(dataset_values: list[float], seed_key: str,
                      reps: int) -> tuple[float | None, float | None]:
    if not dataset_values:
        return None, None
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    values = np.asarray(dataset_values, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(values), size=(reps, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def _paired_summary(pairs: list[tuple[str, int, float, float]],
                    reps: int, seed_key: str) -> dict:
    by_dataset = defaultdict(list)
    for dataset, _seed, delta_auc, delta_ap in pairs:
        by_dataset[dataset].append((delta_auc, delta_ap))
    dataset_auc = [float(np.mean([v[0] for v in by_dataset[d]])) for d in sorted(by_dataset)]
    dataset_ap = [float(np.mean([v[1] for v in by_dataset[d]])) for d in sorted(by_dataset)]
    auc_low, auc_high = _cluster_interval(dataset_auc, seed_key + "|auroc", reps)
    ap_low, ap_high = _cluster_interval(dataset_ap, seed_key + "|ap", reps)
    return {
        "paired_seed_rows": len(pairs),
        "paired_datasets": len(by_dataset),
        "first_higher_seed_rows": sum(p[2] > 0 for p in pairs),
        "ties_seed_rows": sum(p[2] == 0 for p in pairs),
        "second_higher_seed_rows": sum(p[2] < 0 for p in pairs),
        "macro_mean_delta_auroc": _mean(dataset_auc),
        "macro_mean_delta_ap": _mean(dataset_ap),
        "cluster_bootstrap_95_low_auroc": auc_low,
        "cluster_bootstrap_95_high_auroc": auc_high,
        "cluster_bootstrap_95_low_ap": ap_low,
        "cluster_bootstrap_95_high_ap": ap_high,
        "bootstrap_reps": reps,
    }


def summarize_change_vs_full(reference: list[dict], reduced: list[dict],
                             reps: int, seed: int) -> list[dict]:
    full = _index(reference)
    small = _index(reduced, True)
    datasets = sorted({row["dataset"] for row in reference})
    output = []
    for representation in REPRESENTATIONS:
        for pct in RETENTION_PCTS:
            for method in SELECTED_METHODS:
                paired = []
                for dataset in datasets:
                    for split_seed in SEEDS:
                        key = (dataset, representation, split_seed, method)
                        f, r = full[key], small[(*key, pct)]
                        if f["status"] == r["status"] == "success":
                            paired.append((dataset, split_seed,
                                           r["auroc"] - f["auroc"],
                                           r["ap"] - f["ap"]))
                stats = _paired_summary(paired, reps,
                                        f"{seed}|vsfull|{representation}|{pct}|{method}")
                output.append({
                    "representation": representation,
                    "retention_pct": pct,
                    "reduction_pct": 100 - pct,
                    "method": method,
                    "family": METHOD_META[method]["family"],
                    "strategy": METHOD_META[method].get("strategy", ""),
                    "expected_pairs": EXPECTED_DATASETS * len(SEEDS),
                    "full_coverage": len(paired) == EXPECTED_DATASETS * len(SEEDS),
                    **stats,
                })
    return output


def summarize_dataset_changes(reference: list[dict], reduced: list[dict]) -> list[dict]:
    """Retain dataset-level paired evidence behind each macro comparison."""
    full = _index(reference)
    small = _index(reduced, True)
    datasets = sorted({row["dataset"] for row in reference})
    output = []
    for representation in REPRESENTATIONS:
        for pct in RETENTION_PCTS:
            for method in SELECTED_METHODS:
                for dataset in datasets:
                    matched = []
                    full_status, small_status = Counter(), Counter()
                    for split_seed in SEEDS:
                        key = (dataset, representation, split_seed, method)
                        f, r = full[key], small[(*key, pct)]
                        full_status[f["status"]] += 1
                        small_status[r["status"]] += 1
                        if f["status"] == r["status"] == "success":
                            matched.append((f["auroc"], r["auroc"], f["ap"], r["ap"]))
                    output.append({
                        "representation": representation,
                        "retention_pct": pct,
                        "reduction_pct": 100 - pct,
                        "method": method,
                        "dataset": dataset,
                        "expected_seed_pairs": len(SEEDS),
                        "paired_success_seed_rows": len(matched),
                        "full_status_counts": json.dumps(dict(sorted(full_status.items())), sort_keys=True),
                        "reduced_status_counts": json.dumps(dict(sorted(small_status.items())), sort_keys=True),
                        "mean_auroc_full_on_paired_seeds": _mean(v[0] for v in matched),
                        "mean_auroc_reduced_on_paired_seeds": _mean(v[1] for v in matched),
                        "mean_delta_auroc": _mean(v[1] - v[0] for v in matched),
                        "mean_ap_full_on_paired_seeds": _mean(v[2] for v in matched),
                        "mean_ap_reduced_on_paired_seeds": _mean(v[3] for v in matched),
                        "mean_delta_ap": _mean(v[3] - v[2] for v in matched),
                    })
    return output


def summarize_exclusion_sensitivity(dataset_changes: list[dict], reps: int,
                                    seed: int) -> list[dict]:
    """Exploratory robustness for the two strongest PCA2/25% NSA changes.

    The leave-one-dataset-out sweep and combined vowels+letter exclusion were
    requested after seeing the primary results. They should not be treated as
    predeclared confirmatory comparisons.
    """
    output = []
    for method in ("NSA/dual", "NSA/voronoi"):
        chosen = {r["dataset"]: r for r in dataset_changes
                  if r["representation"] == "pca2"
                  and r["retention_pct"] == 25 and r["method"] == method}
        if len(chosen) != EXPECTED_DATASETS or any(
                row["paired_success_seed_rows"] != len(SEEDS) for row in chosen.values()):
            raise ValueError(f"exclusion audit requires complete paired rows for {method}")
        exclusions = [(), *((d,) for d in sorted(chosen)), ("vowels", "letter")]
        for excluded in exclusions:
            included = [r for d, r in sorted(chosen.items()) if d not in excluded]
            auc = [float(r["mean_delta_auroc"]) for r in included]
            ap = [float(r["mean_delta_ap"]) for r in included]
            label = "+".join(excluded) if excluded else "none"
            # The no-exclusion row must reproduce the primary paired interval
            # exactly. Other exclusions get their own deterministic draw.
            interval_key = (f"{seed}|vsfull|pca2|25|{method}" if not excluded
                            else f"{seed}|exclusion|{method}|{label}")
            auc_low, auc_high = _cluster_interval(
                auc, interval_key + "|auroc", reps)
            ap_low, ap_high = _cluster_interval(
                ap, interval_key + "|ap", reps)
            output.append({
                "representation": "pca2",
                "retention_pct": 25,
                "method": method,
                "excluded_datasets": label,
                "remaining_datasets": len(included),
                "remaining_dataset_wins_auroc": sum(value > 0 for value in auc),
                "macro_mean_delta_auroc": _mean(auc),
                "cluster_bootstrap_95_low_auroc": auc_low,
                "cluster_bootstrap_95_high_auroc": auc_high,
                "macro_mean_delta_ap": _mean(ap),
                "cluster_bootstrap_95_low_ap": ap_low,
                "cluster_bootstrap_95_high_ap": ap_high,
                "bootstrap_reps": reps,
                "exploratory_posthoc": True,
            })
    return output


def summarize_nsa_vs_baseline(reference: list[dict], reduced: list[dict],
                              reps: int, seed: int) -> tuple[list[dict], list[dict]]:
    full = _index(reference)
    small = _index(reduced, True)
    datasets = sorted({row["dataset"] for row in reference})
    cross, interactions = [], []
    for representation in REPRESENTATIONS:
        for pct in RETENTION_PCTS:
            for nsa in NSA_METHODS:
                for baseline in BASELINES:
                    at_reduced, interaction = [], []
                    for dataset in datasets:
                        for split_seed in SEEDS:
                            nkey = (dataset, representation, split_seed, nsa)
                            bkey = (dataset, representation, split_seed, baseline)
                            nr, br = small[(*nkey, pct)], small[(*bkey, pct)]
                            nf, bf = full[nkey], full[bkey]
                            if nr["status"] == br["status"] == "success":
                                at_reduced.append((dataset, split_seed,
                                                   nr["auroc"] - br["auroc"],
                                                   nr["ap"] - br["ap"]))
                            if all(row["status"] == "success" for row in (nr, br, nf, bf)):
                                interaction.append((
                                    dataset, split_seed,
                                    (nr["auroc"] - nf["auroc"]) - (br["auroc"] - bf["auroc"]),
                                    (nr["ap"] - nf["ap"]) - (br["ap"] - bf["ap"])))
                    common = {
                        "representation": representation,
                        "retention_pct": pct,
                        "reduction_pct": 100 - pct,
                        "nsa": nsa,
                        "baseline": baseline,
                        "expected_pairs": EXPECTED_DATASETS * len(SEEDS),
                    }
                    cross.append({**common,
                                  "full_coverage": len(at_reduced) == EXPECTED_DATASETS * len(SEEDS),
                                  **_paired_summary(at_reduced, reps,
                                                    f"{seed}|cross|{representation}|{pct}|{nsa}|{baseline}")})
                    interactions.append({**common,
                                         "full_coverage": len(interaction) == EXPECTED_DATASETS * len(SEEDS),
                                         **_paired_summary(interaction, reps,
                                                           f"{seed}|interaction|{representation}|{pct}|{nsa}|{baseline}")})
    return cross, interactions


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no records for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-raw-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-raw-dir", type=Path, required=True)
    parser.add_argument("--retention-raw-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.bootstrap_reps < 1000:
        parser.error("--bootstrap-reps must be at least 1000")
    reference, reduced = load_and_validate(args.primary_raw_dir,
                                            args.sensitivity_raw_dir,
                                            args.retention_raw_dir,
                                            args.dataset_manifest)
    methods = summarize_methods(reference, reduced)
    changes = summarize_change_vs_full(reference, reduced,
                                       args.bootstrap_reps, args.bootstrap_seed)
    dataset_changes = summarize_dataset_changes(reference, reduced)
    robustness = summarize_exclusion_sensitivity(dataset_changes,
                                                 args.bootstrap_reps,
                                                 args.bootstrap_seed)
    cross, interactions = summarize_nsa_vs_baseline(reference, reduced,
                                                    args.bootstrap_reps,
                                                    args.bootstrap_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "retention_method_summary.csv", methods)
    _write_csv(args.out_dir / "retention_change_vs_full.csv", changes)
    _write_csv(args.out_dir / "retention_dataset_changes.csv", dataset_changes)
    _write_csv(args.out_dir / "retention_exclusion_sensitivity.csv", robustness)
    _write_csv(args.out_dir / "retention_nsa_vs_baseline.csv", cross)
    _write_csv(args.out_dir / "retention_interaction.csv", interactions)
    print(f"Validated {len(reference)} full and {len(reduced)} reduced rows; "
          f"wrote {len(methods)} method, {len(changes)} paired-change, "
          f"{len(dataset_changes)} dataset-change, {len(robustness)} robustness, "
          f"{len(cross)} cross-method and {len(interactions)} interaction rows")


if __name__ == "__main__":
    main()
