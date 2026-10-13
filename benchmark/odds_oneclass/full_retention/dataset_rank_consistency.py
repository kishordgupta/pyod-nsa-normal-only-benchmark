"""Audit Dual NSA dataset ranks for normal-only PCA(2), 25% retention."""

import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualified(row):
    return (row["status"] == "success"
            and row["fit_seconds"] + row["score_seconds"] <= 15.0)


def rank(value, values):
    return 1 + sum(other > value + 1e-12 for other in values)


def main():
    workspace = Path(__file__).resolve().parents[3]
    outputs = workspace / "results"
    ranking_file = outputs / "pyod_nsa_full_retention_coverage_qualified_ranks.csv"
    with ranking_file.open(newline="") as stream:
        field = [row["method"] for row in csv.DictReader(stream)
                 if row["representation"] == "pca2"
                 and row["retention_pct"] == "25" and row["metric"] == "auroc"]
    assert len(field) == len(set(field)) == 36
    files = sorted(Path(__file__).parent.joinpath("raw").glob(
        "*_pca2_seed*_retain25.jsonl"))
    assert len(files) == 85
    index = {}
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["fit_outliers"] == 0
            assert row["representation"] == "pca2"
            assert row["retention_fraction"] == 0.25
            key = (row["dataset"], row["seed"], row["method"])
            assert key not in index
            index[key] = row
    seeds = [42, 43, 44, 45, 46]
    datasets = sorted({key[0] for key in index})
    assert len(datasets) == 17
    all_methods = sorted({key[2] for key in index})
    rows = []
    for dataset in datasets:
        assert all(qualified(index[dataset, seed, method])
                   for method in field for seed in seeds)
        means = {method: statistics.mean(index[dataset, seed, method]["auroc"]
                                         for seed in seeds) for method in field}
        dual = means["NSA/dual"]
        overall_rank = rank(dual, means.values())
        seed_ranks = [rank(index[dataset, seed, "NSA/dual"]["auroc"],
                           [index[dataset, seed, method]["auroc"] for method in field])
                      for seed in seeds]
        local = {method: statistics.mean(index[dataset, seed, method]["auroc"]
                                         for seed in seeds) for method in all_methods
                 if all(qualified(index[dataset, seed, method]) for seed in seeds)}
        local_rank = rank(dual, local.values())
        baseline_values = [score for method, score in means.items()
                           if not method.startswith("NSA/")]
        assert len(baseline_values) == 26
        baseline_rank = rank(dual, [dual, *baseline_values])
        rows.append({
            "dataset": dataset, "dual_mean_auroc": dual,
            "overall_rank_fixed_field": overall_rank, "fixed_field_size": len(field),
            "rank_fraction_fixed_field": overall_rank / len(field),
            "top35_fixed_field": overall_rank / len(field) <= 0.35,
            "top35_seed_count": sum(value / len(field) <= 0.35 for value in seed_ranks),
            "seed_count": len(seeds),
            **{f"rank_seed{seed}": value for seed, value in zip(seeds, seed_ranks)},
            "baseline_plus_dual_rank": baseline_rank, "baseline_plus_dual_field_size": 27,
            "top35_baseline_plus_dual": baseline_rank / 27 <= 0.35,
            "rank_dataset_qualified_field": local_rank,
            "dataset_qualified_field_size": len(local),
            "top35_dataset_qualified_field": local_rank / len(local) <= 0.35,
        })
    prefix = outputs / "pyod_nsa_dual_pca2_retain25_dataset_rank"
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "setting": "PCA(2); 25% retained normal training pool; zero training outliers",
        "metric": "mean AUROC over five seeds for each dataset",
        "rank_rule": "competition rank; 1 + count(score > Dual NSA score + 1e-12)",
        "top35_rule": "rank / eligible field size <= 0.35",
        "primary_field": field, "primary_field_size": len(field),
        "primary_qualification": "all 85 runs raw success and measured fit+score <=15 seconds",
        "datasets": len(datasets), "seeds": seeds,
        "top35_datasets_fixed_field": sum(row["top35_fixed_field"] for row in rows),
        "top35_all_five_seeds_datasets": sum(row["top35_seed_count"] == 5 for row in rows),
        "top35_seed_runs": sum(row["top35_seed_count"] for row in rows),
        "top35_datasets_dataset_qualified_field": sum(
            row["top35_dataset_qualified_field"] for row in rows),
        "top35_datasets_baseline_plus_dual": sum(
            row["top35_baseline_plus_dual"] for row in rows),
        "dual_macro_auroc": statistics.mean(row["dual_mean_auroc"] for row in rows),
        "ranking_file_sha256": sha(ranking_file), "csv_sha256": sha(csv_path),
        "raw_files_sha256": {path.name: sha(path) for path in files},
        "script_sha256": sha(Path(__file__)),
        "limitations": "Descriptive dataset ranks; ties, failures, chosen field, and small test sets affect rank. No universal or confirmatory superiority claim.",
    }
    audit_path = prefix.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items()
                      if key != "raw_files_sha256"}, indent=2))
    print(csv_path)
    print(audit_path)


if __name__ == "__main__":
    main()
