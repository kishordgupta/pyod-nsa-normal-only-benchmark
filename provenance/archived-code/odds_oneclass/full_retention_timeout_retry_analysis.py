"""Validate and summarize the separate 60-second timeout feasibility panel.

This script requires a complete primary grid and exactly one retry result for
every primary timeout. It writes only under timeout_retry/analysis and never
changes or combines with the authoritative 15-second ranking results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import runner
from full_retention_timeout_retry import (
    DEFAULT_PRIMARY, inspect_primary, sha256, verify_primary_code,
    verify_primary_source,
)
from model_factory import METHODS
from timeout_retry_runner import META_KEYS, read_rows, validate_rows


DETAIL_FIELDS = (
    "dataset", "representation", "seed", "retention_fraction",
    "fit_normals", "test_normals", "test_outliers", "method",
    "primary_status", "retry_status", "outcome_group", "auroc", "ap",
    "fit_seconds", "score_seconds", "error", "primary_record_sha256",
    "data_sha256",
)
SUMMARY_FIELDS = (
    "method", "representation", "retention_fraction",
    "primary_timeouts", "retry_success", "retry_error", "retry_timeout",
    "success_mean_auroc", "success_mean_ap",
)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_status(status: str) -> str:
    if status == "success":
        return "success"
    if status == "timeout":
        return "timeout"
    if status in {"error", "build_error"}:
        return "error"
    raise ValueError(f"unexpected retry status {status!r}")


def validate_cell(cell: dict, primary_manifest: dict, retry_root: Path,
                  data_hash: str) -> list[dict]:
    """Return audited detail rows for one primary-timeout cell."""
    name = cell["name"]
    primary_path = cell["primary"]
    raw_path = retry_root / "raw" / f"{name}.jsonl"
    sidecar_path = raw_path.with_suffix(".retry.json")
    if not raw_path.exists() or not sidecar_path.exists():
        raise ValueError(f"{name}: missing raw retry outcome or provenance sidecar")

    data_path = Path(primary_manifest["data_dir"]) / f"{cell['dataset']}.mat"
    _, _, _, expected = runner.split_data(
        data_path, cell["seed"], primary_manifest["train_cap"],
        primary_manifest["test_cap"], cell["representation"], cell["fraction"],
    )
    if expected["fit_outliers"] != 0:
        raise ValueError(f"{name}: reconstructed training set contains anomalies")

    primary = read_rows(primary_path)
    validate_rows(primary, primary_path, expected, set(METHODS), require_all=True)
    primary_by_method = {row["method"]: row for row in primary}
    selected = [method for method in METHODS
                if primary_by_method[method]["status"] == "timeout"]
    if selected != cell["methods"]:
        raise ValueError(f"{name}: timeout method set changed")
    retry = read_rows(raw_path)
    validate_rows(retry, raw_path, expected, set(selected), require_all=True)
    retry_by_method = {row["method"]: row for row in retry}

    sidecar = json.loads(sidecar_path.read_text())
    expected_sidecar = {
        "primary_record": str(primary_path.resolve()),
        "primary_record_sha256": cell["primary_sha256"],
        "primary_method_status": "timeout",
        "method_timeout_seconds": 60.0,
        "methods": selected,
        "data_path": str(data_path.resolve()),
        "data_sha256": data_hash,
        "split_meta": {key: expected[key] for key in META_KEYS},
    }
    for key, value in expected_sidecar.items():
        if sidecar.get(key) != value:
            raise ValueError(f"{name}: sidecar mismatch at {key}")

    detail: list[dict] = []
    for method in selected:
        old, new = primary_by_method[method], retry_by_method[method]
        if old["status"] != "timeout":
            raise ValueError(f"{name}/{method}: non-timeout primary method retried")
        if any(old.get(key) != new.get(key) for key in META_KEYS):
            raise ValueError(f"{name}/{method}: retry split differs from primary")
        outcome = classify_status(new["status"])
        if outcome == "success":
            if not (new.get("auroc") is not None
                    and new.get("ap") is not None
                    and 0 <= new["auroc"] <= 1
                    and 0 <= new["ap"] <= 1):
                raise ValueError(f"{name}/{method}: invalid successful scores")
        elif new.get("auroc") is not None or new.get("ap") is not None:
            raise ValueError(f"{name}/{method}: failed retry has performance scores")
        detail.append({
            "dataset": cell["dataset"],
            "representation": cell["representation"],
            "seed": cell["seed"],
            "retention_fraction": cell["fraction"],
            "fit_normals": new["fit_normals"],
            "test_normals": new["test_normals"],
            "test_outliers": new["test_outliers"],
            "method": method,
            "primary_status": old["status"],
            "retry_status": new["status"],
            "outcome_group": outcome,
            "auroc": new["auroc"],
            "ap": new["ap"],
            "fit_seconds": new.get("fit_seconds"),
            "score_seconds": new.get("score_seconds"),
            "error": new.get("error"),
            "primary_record_sha256": cell["primary_sha256"],
            "data_sha256": data_hash,
        })
    return detail


def summarize(detail: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in detail:
        grouped[(row["method"], row["representation"],
                 row["retention_fraction"])].append(row)
    output = []
    for (method, representation, fraction), rows in sorted(grouped.items()):
        counts = Counter(row["outcome_group"] for row in rows)
        successes = [row for row in rows if row["outcome_group"] == "success"]
        output.append({
            "method": method,
            "representation": representation,
            "retention_fraction": fraction,
            "primary_timeouts": len(rows),
            "retry_success": counts["success"],
            "retry_error": counts["error"],
            "retry_timeout": counts["timeout"],
            "success_mean_auroc": mean(row["auroc"] for row in successes)
            if successes else None,
            "success_mean_ap": mean(row["ap"] for row in successes)
            if successes else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--retry-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    primary_root = args.primary_root.resolve()
    retry_root = (args.retry_root or primary_root / "timeout_retry").resolve()
    output_dir = (args.output_dir or retry_root / "analysis").resolve()
    if output_dir in {primary_root, primary_root / "raw", primary_root / "logs",
                      retry_root, retry_root / "raw", retry_root / "logs"}:
        parser.error("analysis output must be separate from primary and retry records")

    primary_manifest_path = primary_root / "run_manifest.json"
    retry_manifest_path = retry_root / "run_manifest.json"
    primary_manifest = json.loads(primary_manifest_path.read_text())
    retry_manifest = json.loads(retry_manifest_path.read_text())
    if retry_manifest.get("primary_manifest_sha256") != sha256(primary_manifest_path):
        raise ValueError("retry manifest does not match primary manifest")
    if retry_manifest.get("retry_method_timeout_seconds") != 60:
        raise ValueError("retry manifest does not declare a 60-second budget")
    current_primary_code = verify_primary_code(primary_manifest)
    if retry_manifest.get("primary_code_sha256") != current_primary_code:
        raise ValueError("retry manifest code hashes differ from the audited primary code")
    current_primary_source = verify_primary_source(primary_manifest)
    if retry_manifest.get("primary_source") != current_primary_source:
        raise ValueError("retry manifest PyOD source differs from the audited primary source")
    for field, path in (
        ("retry_runner_sha256", Path(__file__).resolve().parent / "timeout_retry_runner.py"),
        ("retry_orchestrator_sha256", Path(__file__).resolve().parent / "full_retention_timeout_retry.py"),
    ):
        if retry_manifest.get(field) != sha256(path):
            raise ValueError(f"retry code changed after execution: {path.name}")
    cells, incomplete = inspect_primary(primary_manifest, primary_root)
    if incomplete or len(cells) != primary_manifest["expected_cells"]:
        raise ValueError(f"primary grid incomplete: {len(incomplete)} missing/partial cells")
    targets = [cell for cell in cells if cell["methods"]]
    planned = [{"name": cell["name"],
                "primary_sha256": cell["primary_sha256"],
                "methods": cell["methods"]} for cell in targets]
    if retry_manifest.get("primary_timeout_cells") != planned:
        raise ValueError("retry manifest target set differs from primary timeouts")
    expected_files = {f"{cell['name']}.jsonl" for cell in targets}
    observed_files = {path.name for path in (retry_root / "raw").glob("*.jsonl")}
    if observed_files != expected_files:
        raise ValueError(f"retry raw files differ: missing {len(expected_files - observed_files)}, "
                         f"unexpected {len(observed_files - expected_files)}")

    data_hashes = {}
    for entry in primary_manifest["datasets"]:
        path = Path(primary_manifest["data_dir"]) / f"{entry['name']}.mat"
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{entry['name']}: dataset SHA-256 differs from primary manifest")
        data_hashes[entry["name"]] = actual
    detail = []
    for cell in targets:
        detail.extend(validate_cell(
            cell, primary_manifest, retry_root, data_hashes[cell["dataset"]]
        ))
    expected_count = sum(len(cell["methods"]) for cell in targets)
    if len(detail) != expected_count:
        raise ValueError(f"retry record count {len(detail)} != {expected_count}")
    detail.sort(key=lambda row: (
        row["representation"], row["retention_fraction"],
        row["dataset"], row["seed"], row["method"]
    ))
    summary = summarize(detail)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "retry_outcomes.csv", detail, DETAIL_FIELDS)
    write_csv(output_dir / "retry_feasibility_summary.csv", summary, SUMMARY_FIELDS)
    outcomes = Counter(row["outcome_group"] for row in detail)
    validation = {
        "purpose": "60-second timeout feasibility only; excluded from primary 15-second rankings",
        "primary_manifest_sha256": sha256(primary_manifest_path),
        "retry_manifest_sha256": sha256(retry_manifest_path),
        "validated_primary_timeout_cells": len(targets),
        "validated_retry_records": len(detail),
        "outcomes": dict(sorted(outcomes.items())),
        "dataset_sha256": dict(sorted(data_hashes.items())),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(
        "# 60-second timeout feasibility panel\n\n"
        "Every row here began as a timeout in the complete 15-second primary "
        "normal-only benchmark. The retry used the identical data, normal-only "
        "fit indices and order, scaler/PCA protocol, held-out test indices, "
        "method configuration, and seed. Only the wall-time cap rose to 60 "
        "seconds. All data and split hashes were validated.\n\n"
        "`retry_outcomes.csv` lists every original timeout and its retry "
        "status. `retry_feasibility_summary.csv` counts success, error, and "
        "timeout by method, retained training fraction, and feature "
        "representation. Its mean AUROC/AP fields describe only successful "
        "retries among previously timed-out cells. They are not comparable "
        "full-grid rankings and must not be merged into the 15-second results.\n"
    )
    print(f"Validated {len(detail)} retries in {len(targets)} timeout cells: "
          f"{dict(outcomes)}. Wrote {output_dir}")


if __name__ == "__main__":
    main()
