"""Package the validated NSA normal-retention experiment for user review.

This script never reruns models. It requires the strict retention analysis CSVs
and the exact 306 corrected JSONL files. The provisional pre-order-fix files
are kept in a visibly excluded archive directory and are not aggregated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RETENTION = ROOT / "retention"
ANALYSIS = RETENTION / "analysis"
PREFIX = "pyod_nsa_training_retention_"
ANALYSIS_FILES = (
    "retention_method_summary.csv",
    "retention_change_vs_full.csv",
    "retention_dataset_changes.csv",
    "retention_nsa_vs_baseline.csv",
    "retention_interaction.csv",
    "retention_exclusion_sensitivity.csv",
)
CODE_FILES = (
    "runner.py", "selected_runner.py", "scheduler.py", "retention_sweep.py",
    "model_factory.py", "sensitivity_analysis.py", "retention_analysis.py",
    "retention_plot.py", "retention_report.py", "retention_package.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["raw_file"] = path.name
                row["raw_line"] = line_number
                rows.append(row)
    return rows


def write_raw_csv(path: Path, rows: list[dict]) -> None:
    front = [
        "dataset", "representation", "seed", "retention_fraction",
        "reduction_fraction", "method", "family", "status", "auroc", "ap",
        "fit_normals", "original_fit_normals", "fit_outliers",
        "test_normals", "test_outliers", "raw_file", "raw_line",
    ]
    fields = front + sorted(set().union(*(r.keys() for r in rows)) - set(front))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (
                r["dataset"], r["representation"], r["seed"],
                r["retention_fraction"], r["method"])):
            writer.writerow({key: (json.dumps(value, sort_keys=True)
                                  if isinstance(value, (list, dict)) else value)
                             for key, value in row.items()})


def collect_files(root: Path, folder: str, pattern: str = "*") -> list[Path]:
    path = root / folder
    if not path.is_dir():
        raise ValueError(f"missing directory: {path}")
    return sorted(p for p in path.glob(pattern) if p.is_file())


def build_manifest(raw_files: list[Path], rows: list[dict],
                   raw_csv: Path, report_path: Path) -> dict:
    original = json.loads((ROOT / "run_manifest.json").read_text())
    source = json.loads((ROOT / "dataset_manifest.json").read_text())
    source_dir = Path(source["source"]["local_directory"])
    for item in source["datasets"]:
        if sha256(source_dir / item["filename"]) != item["sha256"]:
            raise ValueError(f"dataset checksum changed: {item['name']}")
    status_counts = Counter((r["representation"], int(100 * r["retention_fraction"]),
                             r["status"]) for r in rows)
    statuses = {}
    for (representation, pct, status), count in sorted(status_counts.items()):
        statuses.setdefault(f"{representation}_retained_{pct}", {})[status] = count
    old_files = collect_files(RETENTION, "pre_order_fix/raw", "*.jsonl")
    old_rows = sum(sum(bool(line.strip()) for line in path.read_text().splitlines())
                   for path in old_files)
    source_root = Path(original["resolved_nsa_module"]).parents[2]
    if not (source_root / "pyod/models/nsa.py").is_file():
        raise ValueError("pinned PR source checkout not found")
    code_hashes = {name: sha256(ROOT / name) for name in CODE_FILES}
    if code_hashes["model_factory.py"] != original["model_factory_sha256"]:
        raise ValueError("method roster changed since 100% reference")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Normal-only training-size sensitivity experiment for the NSA PyOD PR; review only, no GitHub posting",
        "validation": {
            "result": "passed",
            "command": "python retention_analysis.py --primary-raw-dir raw --sensitivity-raw-dir sensitivity/raw --retention-raw-dir retention/raw --dataset-manifest dataset_manifest.json --out-dir retention/analysis",
            "reference_100pct_rows": 1836,
            "corrected_reduced_rows": len(rows),
            "corrected_reduced_jsonl_files": len(raw_files),
            "all_corrected_fit_outliers_zero": all(r["fit_outliers"] == 0 for r in rows),
            "checked_source_dataset_sha256": True,
            "checked_test_row_identity": True,
            "checked_fit_normal_row_set_and_order": True,
            "checked_exact_method_dataset_fraction_seed_representation_grid": True,
        },
        "protocol": {
            "version": "retention-v2-parent-order",
            "datasets": len(source["datasets"]),
            "seeds": [42, 43, 44],
            "representations": ["native", "pca2"],
            "retention_fractions": [0.25, 0.50, 0.75],
            "reduction_fractions": [0.75, 0.50, 0.25],
            "nsa_configurations": 13,
            "nsa_strategy_families": 11,
            "controls": ["IForest", "ECOD", "OCSVM", "KNN", "ABOD"],
            "methods_per_cell": 18,
            "split": "60/40 stratified on original binary labels; discard training outliers before every fit",
            "normal_parent_cap": 600,
            "mixed_test_cap": 2000,
            "selection": "From the original capped normal training rows, seed+30000 permutes indices. Nested prefixes of floor(N*fraction), at least 2, select 25/50/75% normal subsets; selected rows are restored to their original parent fit order.",
            "preprocessing": "StandardScaler fit only on retained normals; PCA2, when selected, refit only on retained normals; same original held-out test indices at every fraction.",
            "fit": "model.fit(X_fit) without y for all applicable methods",
            "evaluation": "AUROC and average precision on identical mixed held-out test rows within dataset/seed; statuses retained for failed fits",
            "per_method_wall_time_seconds": 15,
            "bootstrap": "5000 percentile draws of whole datasets; average matched split deltas within each dataset; unadjusted descriptive intervals",
        },
        "source": {
            "repository_url": source["source"]["repository_url"],
            "source_git_commit": original["source_git_commit"],
            "source_git_tree": original["source_git_tree"],
            "source_pyod_version": original["source_pyod_version"],
            "dataset_manifest_sha256": sha256(ROOT / "dataset_manifest.json"),
            "datasets": [{"name": item["name"], "sha256": item["sha256"],
                          "source_url": item["source_url"]} for item in source["datasets"]],
            "pr_nsa_module_sha256": sha256(source_root / "pyod/models/nsa.py"),
            "registry_sha256": sha256(source_root / "pyod/utils/knowledge/algorithms.json"),
        },
        "environment": {
            "python_version": original["python_version"],
            "platform": original["platform"],
            "packages": original["packages"],
        },
        "reference_100pct": {
            "source": "original full-roster seed-42 raw files plus selected-method seed-43 and seed-44 sensitivity raw files",
            "original_run_manifest_sha256": sha256(ROOT / "run_manifest.json"),
            "original_runner_sha256": original["runner_sha256"],
            "selected_reference_csv_sha256": sha256(ROOT / "sensitivity/analysis/sensitivity_records.csv"),
        },
        "corrected_reduced": {
            "raw_status_counts": statuses,
            "raw_csv_sha256": sha256(raw_csv),
            "raw_jsonl_sha256": {p.name: sha256(p) for p in raw_files},
            "code_sha256": code_hashes,
            "analysis_sha256": {name: sha256(ANALYSIS / name) for name in ANALYSIS_FILES},
            "plot_sha256": sha256(ANALYSIS / "retention_auroc_change.png"),
        },
        "report": {
            "filename": report_path.name,
            "sha256": sha256(report_path),
        },
        "excluded_pre_order_fix": {
            "excluded_from_all_scored_tables": True,
            "reason": "Provisional partial runs sorted retained rows rather than preserving the original parent fit order; they were archived before the corrected sweep.",
            "archive_directory": "excluded_pre_order_fix/",
            "raw_jsonl_files": len(old_files),
            "raw_rows": old_rows,
        },
        "outputs_prefix": PREFIX,
    }
    if len(rows) != 5508 or len(raw_files) != 306:
        raise ValueError("corrected retention grid incomplete")
    if not manifest["validation"]["all_corrected_fit_outliers_zero"]:
        raise ValueError("outlier found in retained normal training")
    return manifest


def make_zip(path: Path, raw_csv: Path, manifest_path: Path,
             reference_csv: Path, report_path: Path) -> tuple[int, str]:
    items: list[tuple[Path, str]] = []
    def add_file(src: Path, dest: str) -> None:
        if not src.is_file():
            raise ValueError(f"missing archive input: {src}")
        items.append((src, f"pyod_nsa_training_retention/{dest}"))
    add_file(manifest_path, "manifests/retention_manifest.json")
    add_file(ROOT / "dataset_manifest.json", "manifests/dataset_manifest.json")
    add_file(ROOT / "run_manifest.json", "manifests/reference_run_manifest.json")
    add_file(raw_csv, "tables/reduced_raw_5508.csv")
    add_file(reference_csv, "tables/reference_100pct_selected_1836.csv")
    add_file(report_path, "report/pyod_nsa_training_retention_report.md")
    for name in ANALYSIS_FILES:
        add_file(ANALYSIS / name, f"tables/{name}")
    add_file(ANALYSIS / "retention_auroc_change.png", "figures/retention_auroc_change.png")
    for name in CODE_FILES:
        add_file(ROOT / name, f"scripts/{name}")
    for file in collect_files(RETENTION, "raw", "*.jsonl"):
        add_file(file, f"reduced_raw/{file.name}")
    for file in collect_files(ROOT, "raw", "*.jsonl"):
        add_file(file, f"reference_100pct/seed42_full_roster/{file.name}")
    for file in collect_files(ROOT, "sensitivity/raw", "*.jsonl"):
        add_file(file, f"reference_100pct/seed43_44_selected/{file.name}")
    for file in collect_files(RETENTION, "logs", "*.log"):
        add_file(file, f"logs/corrected/{file.name}")
    for file in collect_files(RETENTION, "sweep_logs", "*.log"):
        add_file(file, f"logs/sweep/{file.name}")
    for file in sorted((RETENTION / "pre_order_fix").rglob("*")):
        if file.is_file():
            add_file(file, f"excluded_pre_order_fix/{file.relative_to(RETENTION / 'pre_order_fix')}")
    source = json.loads((ROOT / "dataset_manifest.json").read_text())
    data_dir = Path(source["source"]["local_directory"])
    for item in source["datasets"]:
        add_file(data_dir / item["filename"], f"data/{item['filename']}")
    run = json.loads((ROOT / "run_manifest.json").read_text())
    source_root = Path(run["resolved_nsa_module"]).parents[2]
    add_file(source_root / "pyod/models/nsa.py", "source_snapshot/pyod/models/nsa.py")
    add_file(source_root / "pyod/utils/knowledge/algorithms.json",
             "source_snapshot/pyod/utils/knowledge/algorithms.json")
    readme = (
        "# PyOD NSA normal-retention evidence bundle\n\n"
        "The corrected experiment used only normal rows for training. `reduced_raw/` "
        "contains 5,508 scored run records. `reference_100pct/` contains the "
        "earlier raw records; the selected 1,836-row reference is in `tables/`. "
        "`data/` stores the 17 exact source MAT files, with hashes in the manifests.\n\n"
        "Run `scripts/retention_analysis.py` with the raw directories and dataset "
        "manifest to regenerate the analysis tables. The analysis script expects "
        "the repository's pinned PyOD PR checkout for imports; the tested source "
        "tree and NSA module hash are recorded in the manifest.\n\n"
        "`excluded_pre_order_fix/` contains a provisional partial run. It used a "
        "different fit-row order and was excluded from every scored table.\n"
    ).encode()
    excluded_readme = (
        "EXCLUDED FROM ALL SCORED RESULTS. These provisional partial files sorted "
        "selected normal rows. The corrected v2 sweep preserves original parent "
        "fit order and is in `reduced_raw/`. Do not mix these records.\n"
    ).encode()
    extra = {
        "pyod_nsa_training_retention/README.md": readme,
        "pyod_nsa_training_retention/excluded_pre_order_fix/EXCLUDED_README.txt": excluded_readme,
    }
    names = [dest for _src, dest in items] + list(extra)
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive member name")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for src, dest in sorted(items, key=lambda item: item[1]):
            archive.write(src, dest)
        for dest, payload in sorted(extra.items()):
            archive.writestr(dest, payload)
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"archive CRC failure: {corrupt}")
        if set(archive.namelist()) != set(names):
            raise ValueError("archive members changed")
        for src, dest in items:
            if hashlib.sha256(archive.read(dest)).hexdigest() != sha256(src):
                raise ValueError(f"archive SHA256 mismatch: {dest}")
    return len(names), sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT.parents[1] / "outputs")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = collect_files(RETENTION, "raw", "*.jsonl")
    rows = read_rows(raw_files)
    raw_csv = out_dir / f"{PREFIX}raw.csv"
    write_raw_csv(raw_csv, rows)
    report_path = out_dir / f"{PREFIX}report.md"
    if not report_path.is_file():
        raise ValueError(f"final report missing: {report_path}")
    reference_csv = ROOT / "sensitivity/analysis/sensitivity_records.csv"
    if sum(1 for _ in reference_csv.open(encoding="utf-8")) != 1837:
        raise ValueError("selected 100% reference table incomplete")
    manifest = build_manifest(raw_files, rows, raw_csv, report_path)
    manifest_path = RETENTION / "retention_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    shutil.copy2(manifest_path, out_dir / f"{PREFIX}manifest.json")
    shutil.copy2(reference_csv, out_dir / f"{PREFIX}reference_100pct.csv")
    for name in ANALYSIS_FILES:
        shutil.copy2(ANALYSIS / name, out_dir / f"{PREFIX}{name.removeprefix('retention_')}")
    shutil.copy2(ANALYSIS / "retention_auroc_change.png",
                 out_dir / f"{PREFIX}auroc_change.png")
    zip_path = out_dir / f"{PREFIX}reproducibility.zip"
    members, archive_sha = make_zip(zip_path, raw_csv, manifest_path,
                                    reference_csv, report_path)
    print(f"Packaged {len(rows)} corrected rows and {members} archive members; "
          f"ZIP SHA256 {archive_sha}")


if __name__ == "__main__":
    main()
