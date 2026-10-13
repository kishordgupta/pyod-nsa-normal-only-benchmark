"""Audit and aggregate the predeclared three-seed clean-normal experiment.

Seed 42 is read from the full-roster primary panel; seeds 43 and 44 are read
from the selected-method sensitivity panel. Dataset, method, representation,
and seed coverage must be exact before any output is written. Outcomes with
failed fits remain visible in status counts and never become zero scores.

The paired interval resamples *datasets*, keeping a dataset's matched seed
differences together. It is a percentile cluster-bootstrap interval for the
macro (equal-dataset-weight) mean AUROC difference, not a significance test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from model_factory import METHOD_META, NSA_METHODS


REPRESENTATIONS = ("pca2", "native")
SEEDS = (42, 43, 44)
BASELINES = ("IForest", "ECOD", "OCSVM", "KNN", "ABOD")
SELECTED_METHODS = tuple(NSA_METHODS) + BASELINES
EXPECTED_DATASETS = 17
EXPECTED_METHODS = 18
EXPECTED_RECORDS = EXPECTED_DATASETS * EXPECTED_METHODS * len(SEEDS) * len(REPRESENTATIONS)


def _load_jsonl(directory: Path, seeds: set[int]) -> list[dict]:
    if not directory.is_dir():
        raise ValueError(f"missing raw directory: {directory}")
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if row.get("seed") not in seeds:
                    continue
                row["_source_file"] = str(path)
                row["_source_line"] = line_number
                records.append(row)
    return records


def load_and_validate(primary_raw_dir: Path, sensitivity_raw_dir: Path,
                      dataset_manifest: Path) -> list[dict]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    datasets = tuple(sorted(item["name"] for item in manifest["datasets"]))
    if len(datasets) != EXPECTED_DATASETS or len(set(datasets)) != EXPECTED_DATASETS:
        raise ValueError("expected exactly 17 distinct manifest datasets")
    if len(SELECTED_METHODS) != EXPECTED_METHODS or len(set(SELECTED_METHODS)) != EXPECTED_METHODS:
        raise ValueError("selected method roster must have 18 distinct entries")
    records = []
    for row in _load_jsonl(primary_raw_dir, {42}):
        if row.get("method") in SELECTED_METHODS:
            records.append(row)
    for row in _load_jsonl(sensitivity_raw_dir, {43, 44}):
        records.append(row)

    expected = {(d, r, s, m) for d in datasets for r in REPRESENTATIONS
                for s in SEEDS for m in SELECTED_METHODS}
    counts = Counter()
    for row in records:
        try:
            key = (row["dataset"], row["representation"], row["seed"],
                   row["method"])
        except KeyError as exc:
            raise ValueError(f"malformed row at {row['_source_file']}:{row['_source_line']}: {exc}") from exc
        counts[key] += 1
        if row.get("fit_outliers") != 0:
            raise ValueError(f"outlier leaked into fitting for {key}")
        if row.get("train_cap") != 600 or row.get("test_cap") != 2000:
            raise ValueError(f"protocol cap drift for {key}")
        if row["status"] == "success":
            for metric in ("auroc", "ap"):
                value = row.get(metric)
                if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid {metric} on successful row {key}")
    duplicates = {k: n for k, n in counts.items() if n != 1}
    missing = expected - set(counts)
    extra = set(counts) - expected
    if duplicates or missing or extra or len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"sensitivity grid incomplete: expected {EXPECTED_RECORDS}, "
            f"found {len(records)}, missing {len(missing)}, extra {len(extra)}, "
            f"duplicates {len(duplicates)}; examples "
            f"missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]}")
    return sorted(records, key=lambda row: (row["representation"],
                                           row["dataset"], row["seed"],
                                           row["method"]))


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _macro_metric(rows: list[dict], metric: str) -> tuple[float | None, int]:
    by_dataset = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            by_dataset[row["dataset"]].append(row[metric])
    return _mean(np.mean(values) for values in by_dataset.values()), len(by_dataset)


def summarize_methods(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["representation"], row["method"])].append(row)
    output = []
    for representation in REPRESENTATIONS:
        for method in SELECTED_METHODS:
            rows = grouped[(representation, method)]
            statuses = Counter(row["status"] for row in rows)
            auroc, n_auc_datasets = _macro_metric(rows, "auroc")
            ap, n_ap_datasets = _macro_metric(rows, "ap")
            if n_auc_datasets != n_ap_datasets:
                raise AssertionError("AUROC/AP coverage mismatch")
            output.append({
                "representation": representation,
                "method": method,
                "family": METHOD_META[method]["family"],
                "inference_mode": METHOD_META[method].get("inference_mode", "unknown"),
                "expected_rows": EXPECTED_DATASETS * len(SEEDS),
                "successful_rows": statuses.get("success", 0),
                "successful_datasets": n_auc_datasets,
                "full_coverage": statuses.get("success", 0) == EXPECTED_DATASETS * len(SEEDS),
                "macro_auroc_available": auroc,
                "macro_ap_available": ap,
                "macro_auroc_full_coverage": auroc if statuses.get("success", 0) == EXPECTED_DATASETS * len(SEEDS) else None,
                "macro_ap_full_coverage": ap if statuses.get("success", 0) == EXPECTED_DATASETS * len(SEEDS) else None,
                "status_counts": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
            })
    return output


def _bootstrap_interval(dataset_deltas: list[float], seed: int,
                        reps: int) -> tuple[float | None, float | None]:
    if not dataset_deltas:
        return None, None
    values = np.asarray(dataset_deltas, dtype=float)
    rng = np.random.default_rng(seed)
    # Each sampled index is an entire dataset cluster, including its matched
    # seed observations through the already averaged dataset delta.
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_pairs(records: list[dict], bootstrap_reps: int,
                    bootstrap_seed: int) -> list[dict]:
    indexed = {(r["representation"], r["dataset"], r["seed"], r["method"]): r
               for r in records}
    datasets = sorted({r["dataset"] for r in records})
    output = []
    for representation in REPRESENTATIONS:
        for nsa in NSA_METHODS:
            for baseline in BASELINES:
                paired = []
                for dataset in datasets:
                    for seed in SEEDS:
                        n = indexed[(representation, dataset, seed, nsa)]
                        b = indexed[(representation, dataset, seed, baseline)]
                        if n["status"] == b["status"] == "success":
                            paired.append((dataset, seed, n["auroc"] - b["auroc"],
                                           n["ap"] - b["ap"]))
                by_dataset = defaultdict(list)
                for dataset, _seed, delta_auc, delta_ap in paired:
                    by_dataset[dataset].append((delta_auc, delta_ap))
                dataset_auc = [float(np.mean([x[0] for x in by_dataset[d]]))
                               for d in sorted(by_dataset)]
                dataset_ap = [float(np.mean([x[1] for x in by_dataset[d]]))
                              for d in sorted(by_dataset)]
                key = f"{bootstrap_seed}|{representation}|{nsa}|{baseline}".encode()
                stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
                lower, upper = _bootstrap_interval(dataset_auc, stable_seed,
                                                    bootstrap_reps)
                output.append({
                    "representation": representation,
                    "nsa": nsa,
                    "baseline": baseline,
                    "baseline_inference_mode": METHOD_META[baseline].get("inference_mode", "unknown"),
                    "expected_pairs": EXPECTED_DATASETS * len(SEEDS),
                    "paired_seed_rows": len(paired),
                    "paired_datasets": len(by_dataset),
                    "full_coverage": len(paired) == EXPECTED_DATASETS * len(SEEDS),
                    "nsa_higher_seed_rows": sum(x[2] > 0 for x in paired),
                    "ties_seed_rows": sum(x[2] == 0 for x in paired),
                    "baseline_higher_seed_rows": sum(x[2] < 0 for x in paired),
                    "macro_mean_delta_auroc": _mean(dataset_auc),
                    "macro_mean_delta_ap": _mean(dataset_ap),
                    "cluster_bootstrap_95_low_auroc": lower,
                    "cluster_bootstrap_95_high_auroc": upper,
                    "bootstrap_reps": bootstrap_reps,
                    "bootstrap_seed": bootstrap_seed,
                })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-raw-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-raw-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.bootstrap_reps < 1000:
        parser.error("--bootstrap-reps must be at least 1000")

    records = load_and_validate(args.primary_raw_dir,
                                args.sensitivity_raw_dir,
                                args.dataset_manifest)
    methods = summarize_methods(records)
    pairs = summarize_pairs(records, args.bootstrap_reps,
                            args.bootstrap_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "sensitivity_records.csv", records)
    _write_csv(args.out_dir / "sensitivity_methods.csv", methods)
    _write_csv(args.out_dir / "sensitivity_pairwise.csv", pairs)
    print(f"Validated {len(records)} rows, wrote {len(methods)} method rows "
          f"and {len(pairs)} paired comparisons to {args.out_dir}")


if __name__ == "__main__":
    main()
