"""Verify the public 5%/10% extension and frozen four-size benchmark.

Run from any directory after the append-only extension has been assembled:
    python scripts/verify_extreme_extension.py

This verifies published bytes and recorded result invariants. Raw JSONL files
contain aggregate metrics, not test score vectors, so this is not an AUROC/AP
recalculation or a fresh model run.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "provenance/public-copy-manifest.json"
EXTENSION_MANIFEST = ROOT / "provenance/extreme-retention-extension-manifest.json"
PROJECT = ROOT / "benchmark/odds_oneclass"
FULL = PROJECT / "full_retention"
EXTREME = PROJECT / "extreme_retention"
ANALYSIS = EXTREME / "analysis"
ADAPTIVE = EXTREME / "adaptive_batch"
OLD_PCTS = (100, 75, 50, 25)
NEW_PCTS = (10, 5)
PCTS = OLD_PCTS + NEW_PCTS
REPS = ("native", "pca2")
SEEDS = (42, 43, 44, 45, 46)
TERMINAL = {
    "success", "error", "timeout", "build_error", "requires_labeled_outliers",
    "requires_1d", "requires_2d", "requires_2d_or_3d",
}
TABLES = (
    "method_summary", "common_field_ranks", "raw_common_ranks", "panel_ranks",
    "rank_change", "change_vs_100", "nsa_vs_baseline", "relative_change_vs_100",
    "dataset_nsa_ranks", "dataset_inventory", "raw_vs_strict",
)
FIXED_TABLE_ROWS = {
    "method_summary": 672, "change_vs_100": 560, "nsa_vs_baseline": 6708,
    "relative_change_vs_100": 2236, "dataset_nsa_ranks": 2652,
    "dataset_inventory": 17, "raw_vs_strict": 672,
}
SPLIT_HASHES = (
    "fit_index_sha256", "fit_index_ordered_sha256", "fit_parent_index_sha256",
    "test_index_sha256",
)
ADAPTIVE_SPLIT_FIELDS = (
    "dataset", "seed", "representation", "retention_fraction", "reduction_fraction",
    "protocol_version", "dataset_rows", "source_features", "fit_normals",
    "original_fit_normals", "fit_outliers", "discarded_train_outliers",
    "test_normals", "test_outliers", "train_cap", "test_cap", "retained_variance",
    "fit_index_sha256", "fit_index_ordered_sha256", "fit_parent_index_sha256",
    "test_index_sha256", "method", "family",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_tree_hash(files: dict[str, str]) -> str:
    data = "".join(f"{name}\0{digest}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def public_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or name != relative.as_posix():
        raise ValueError(f"unsafe public manifest path: {name}")
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"public manifest path escapes repository: {name}")
    return path


def verify_files() -> tuple[dict, dict]:
    extension = json.loads(EXTENSION_MANIFEST.read_text(encoding="utf-8"))
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    if (extension.get("schema_version") != 1
            or extension.get("completion") != {
                "new_cells": 340, "new_records": 19040,
                "combined_cells": 1020, "combined_records": 57120,
                "adaptive_sensitivity_cells": 370,
                "adaptive_sensitivity_records": 740,
            }
            or extension.get("base_public_copy_manifest_sha256") != sha(BASE_MANIFEST)):
        raise ValueError("extension manifest or frozen public-copy reference changed")
    protected = extension.get("protected_existing_benchmark_result_file_sha256", {})
    if (len(protected) != extension.get("protected_existing_benchmark_result_file_count")
            or len(protected) != 2387
            or canonical_tree_hash(protected)
                != extension.get("protected_existing_benchmark_result_tree_sha256")):
        raise ValueError("frozen benchmark/result tree declaration changed")
    for name, digest in protected.items():
        if not name.startswith(("benchmark/", "results/")) or sha(public_path(name)) != digest:
            raise ValueError(f"frozen benchmark/result file changed: {name}")
    for item in base["files"]:
        name = item["public_file"]
        if sha(public_path(name)) != item["public_sha256"]:
            raise ValueError(f"prior public-copy file changed: {name}")
    entries = extension.get("files", [])
    names = [item["public_file"] for item in entries]
    if len(names) != len(set(names)) or len(entries) < 700:
        raise ValueError("extension entries are missing or duplicated")
    for item in entries:
        name = item["public_file"]
        if name in protected or sha(public_path(name)) != item["public_sha256"]:
            raise ValueError(f"extension file changed: {name}")
    for name, pair in extension.get("review_files", {}).items():
        if sha(public_path(name)) != pair["after_sha256"]:
            raise ValueError(f"review file changed: {name}")
    if set(extension.get("review_files", {})) != {
            "README.md", "provenance/PORTABILITY_AND_PRIVACY.md"}:
        raise ValueError("review-file allowlist changed")
    listed = set(protected) | set(names)
    actual = {path.relative_to(ROOT).as_posix()
              for directory in (ROOT / "benchmark", ROOT / "results")
              for path in directory.rglob("*")
              if path.is_file()
              and not any(part in {"__pycache__", ".pytest_cache"}
                          for part in path.relative_to(ROOT).parts)
              and path.suffix not in {".pyc", ".pyo"}}
    if actual != {name for name in listed if name.startswith(("benchmark/", "results/"))}:
        raise ValueError("unmanifested benchmark/result file or missing protected file")
    return extension, base


def expected_stem(dataset: str, rep: str, seed: int, pct: int) -> str:
    stem = f"{dataset}_{rep}_seed{seed}"
    return stem if pct == 100 else f"{stem}_retain{pct}"


def finite_unit(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and -1e-12 <= value <= 1.0 + 1e-12


def valid_duration(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value >= 0.0


def verify_grid() -> tuple[dict, dict, dict]:
    dataset_manifest = json.loads((PROJECT / "dataset_manifest.json").read_text())
    datasets = sorted(item["name"] for item in dataset_manifest["datasets"])
    historical = json.loads((FULL / "run_manifest.json").read_text())
    extreme = json.loads((EXTREME / "run_manifest.json").read_text())
    methods = historical["methods"]
    if (len(datasets) != 17 or len(set(datasets)) != 17
            or len(methods) != 56 or len(set(methods)) != 56
            or extreme["methods"] != methods
            or historical["expected_cells"] != 680
            or historical["expected_records"] != 38080
            or extreme["expected_cells"] != 340
            or extreme["expected_records"] != 19040
            or historical["method_timeout_seconds"] != 15.0
            or extreme["method_timeout_seconds"] != 15.0
            or historical["protocol_version"] != extreme["protocol_version"]
            or historical["protocol_version"] != "retention-v2-parent-order"
            or historical["seeds"] != list(SEEDS) or extreme["seeds"] != list(SEEDS)
            or historical["representations"] != list(REPS)
            or extreme["representations"] != list(REPS)
            or historical["fractions_retained"] != [1.0, 0.75, 0.5, 0.25]
            or extreme["fractions_retained"] != [0.1, 0.05]
            or sorted(item["name"] for item in historical["datasets"]) != datasets
            or sorted(item["name"] for item in extreme["datasets"]) != datasets):
        raise ValueError("published six-size grid manifests differ from declared protocol")
    expected_old = {expected_stem(dataset, rep, seed, pct) + ".jsonl":
                    (dataset, rep, seed, pct)
                    for dataset in datasets for rep in REPS for seed in SEEDS for pct in OLD_PCTS}
    expected_new = {expected_stem(dataset, rep, seed, pct) + ".jsonl":
                    (dataset, rep, seed, pct)
                    for dataset in datasets for rep in REPS for seed in SEEDS for pct in NEW_PCTS}
    old_files = {path.name: path for path in (FULL / "raw").glob("*.jsonl")}
    new_files = {path.name: path for path in (EXTREME / "raw").glob("*.jsonl")}
    if set(old_files) != set(expected_old) or set(new_files) != set(expected_new):
        raise ValueError("raw cell filenames differ from the exact six-size grid")
    status = Counter()
    new_status = Counter()
    strict = Counter()
    old_rows = new_rows = 0
    new_hashes: dict[str, str] = {}
    for files, expected, is_new in ((old_files, expected_old, False),
                                    (new_files, expected_new, True)):
        for name, (dataset, rep, seed, pct) in sorted(expected.items()):
            path = files[name]
            content = path.read_bytes()
            if not content.endswith(b"\n"):
                raise ValueError(f"unterminated cell file: {path}")
            lines = content.splitlines()
            if len(lines) != 56:
                raise ValueError(f"wrong method count in {path}")
            if is_new:
                new_hashes[name] = hashlib.sha256(content).hexdigest()
            for method, line in zip(methods, lines, strict=True):
                row = json.loads(line)
                if (row.get("dataset") != dataset or row.get("seed") != seed
                        or row.get("representation") != rep
                        or row.get("retention_fraction") != pct / 100
                        or row.get("method") != method
                        or row.get("protocol_version") != "retention-v2-parent-order"
                        or row.get("fit_outliers") != 0
                        or row.get("train_cap") != 600 or row.get("test_cap") != 2000
                        or not isinstance(row.get("fit_normals"), int)
                        or row["fit_normals"] < 2 or row["fit_normals"] > 600
                        or not isinstance(row.get("original_fit_normals"), int)
                        or row["original_fit_normals"] < row["fit_normals"]
                        or not isinstance(row.get("test_normals"), int)
                        or not isinstance(row.get("test_outliers"), int)
                        or row["test_normals"] < 1 or row["test_outliers"] < 1
                        or row["test_normals"] + row["test_outliers"] > 2000
                        or row.get("status") not in TERMINAL
                        or any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get(field)))
                               for field in SPLIT_HASHES)):
                    raise ValueError(f"row identity, split, or status differs in {path}: {method}")
                state = row["status"]
                if state == "success":
                    if (not finite_unit(row.get("auroc")) or not finite_unit(row.get("ap"))
                            or not valid_duration(row.get("fit_seconds"))
                            or not valid_duration(row.get("score_seconds"))
                            or not isinstance(row.get("score_unique"), int)
                            or not 1 <= row["score_unique"]
                                <= row["test_normals"] + row["test_outliers"]):
                        raise ValueError(f"scored success has invalid metric/timing: {path}: {method}")
                    strict["overbudget_success" if row["fit_seconds"] + row["score_seconds"] > 15.0
                           else "success"] += 1
                else:
                    # A timeout can interrupt after AUROC but before AP. It
                    # remains a timeout and is never counted as scored success.
                    if (row.get("auroc") is not None or row.get("ap") is not None) and not (
                            state == "timeout" and row.get("ap") is None
                            and finite_unit(row.get("auroc"))):
                        raise ValueError(f"non-success has an invalid partial metric: {path}: {method}")
                    strict[state] += 1
                status[state] += 1
                if is_new:
                    new_status[state] += 1
                    new_rows += 1
                else:
                    old_rows += 1
    if old_rows != 38080 or new_rows != 19040 or sum(status.values()) != 57120:
        raise ValueError("combined raw record count differs from 57,120")
    return dict(status), dict(strict), {
        "new_status": dict(new_status), "new_hashes": new_hashes,
        "old_files": len(old_files), "new_files": len(new_files),
        "old_rows": old_rows, "new_rows": new_rows,
    }


def verify_analysis(status: dict, strict: dict, details: dict) -> dict:
    audit_path = ANALYSIS / "extreme_retention_analysis_audit.json"
    independent_path = ANALYSIS / "independent_audit.json"
    focal_path = ANALYSIS / "extreme_retention_focal_metrics.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    focal = json.loads(focal_path.read_text(encoding="utf-8"))
    if (audit.get("expected_cells") != 1020 or audit.get("expected_records") != 57120
            or audit.get("raw_status_counts") != status
            or audit.get("strict_status_counts") != strict
            or audit.get("raw_success_records") != status.get("success", 0)
            or audit.get("strict_success_records") != strict.get("success", 0)
            or audit.get("total_overbudget_success_records")
                != strict.get("overbudget_success", 0)
            or audit.get("new_extreme", {}).get("raw_files") != 340
            or audit["new_extreme"].get("records") != 19040
            or audit["new_extreme"].get("raw_file_sha256") != details["new_hashes"]
            or audit.get("new_run_manifest", {}).get("sha256")
                != sha(EXTREME / "run_manifest.json")
            or audit.get("new_supervision_manifest", {}).get("sha256")
                != sha(EXTREME / "supervision_manifest.json")
            or audit.get("dataset_manifest_sha256") != sha(PROJECT / "dataset_manifest.json")
            or audit.get("focal_metrics_sha256") != sha(focal_path)
            or audit.get("focal_metrics_panels") != 12
            or independent.get("completion_mode") != "complete"
            or independent.get("observed_records") != 19040
            or independent.get("complete_cells") != 340
            or independent.get("raw_status_counts") != details["new_status"]
            or independent.get("all_fit_rows_independently_verified_normal") is not True
            or independent.get("nested_split_rule_verified") is not True
            or independent.get("required_historical_crosscheck") is not True
            or independent.get("historical_size_cells_crosschecked") != 680
            or independent.get("source_manifest_sha256")
                != sha(PROJECT / "dataset_manifest.json")
            or independent.get("runner_sha256") != sha(PROJECT / "runner.py")
            or independent.get("model_factory_sha256") != sha(PROJECT / "model_factory.py")):
        raise ValueError("published six-size analysis and independent audit disagree with raw grid")
    if set(audit.get("tables", {})) != set(TABLES):
        raise ValueError("analysis audit does not list all eleven tables")
    for name in TABLES:
        path = ANALYSIS / f"extreme_retention_{name}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        if (count < 1 or count != audit["tables"][name]
                or (name in FIXED_TABLE_ROWS and count != FIXED_TABLE_ROWS[name])):
            raise ValueError(f"table count differs from audit: {name}")
    panels = focal.get("panels", [])
    if (len(panels) != 12
            or {(p["representation"], p["retention_pct"]) for p in panels}
                != {(rep, pct) for rep in REPS for pct in PCTS}
            or any(len(panel["nsa"]) != 13 for panel in panels)
            or any(set(item["metrics"]) != {"auroc", "ap"}
                   or any("relative_change_vs_100" not in item["metrics"][metric]
                          or "relative_change_vs_100_ci95" not in item["metrics"][metric]
                          for metric in ("auroc", "ap"))
                   for panel in panels for item in panel["nsa"])):
        raise ValueError("focal metric panels or relative-change fields are incomplete")
    for rep in REPS:
        plot = ANALYSIS / f"extreme_retention_dual_{rep}.png"
        if not plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"missing published {rep} plot")
    supervision = json.loads((EXTREME / "supervision_manifest.json").read_text())
    permission_for_mode = {
        "read_only_detection_only": "recovery_inactive",
        "execute_verified_recovery": "recovery_enabled",
    }
    monitor_mode = supervision.get("active_monitor_mode")
    expected_permission = permission_for_mode.get(monitor_mode)
    if (expected_permission is None
            or supervision.get("active_recovery_permission") != expected_permission
            or audit["new_supervision_manifest"].get("active_recovery_permission")
                != expected_permission
            or audit["new_supervision_manifest"].get("active_monitor_mode")
                != monitor_mode):
        raise ValueError("published supervision state is inconsistent")
    return {
        "analysis_tables": len(TABLES), "focal_panels": len(panels),
        "raw_status_counts": status, "strict_status_counts": strict,
    }


def verify_watchdog() -> dict:
    """Check every saved external-timeout snapshot against the final public row."""
    root = EXTREME / "watchdog/interventions"
    if not root.exists():
        return {"watchdog_interventions_verified": 0}
    methods = json.loads((FULL / "run_manifest.json").read_text())["methods"]
    count = 0
    for directory in sorted(root.iterdir()):
        if (not directory.is_dir()
                or {path.name for path in directory.iterdir()} != {
                    "intervention.json", "raw.before.jsonl", "log.before.log"}):
            raise ValueError(f"incomplete watchdog snapshot: {directory}")
        event = json.loads((directory / "intervention.json").read_text())
        before, after = event["before"], event["after"]
        cell, method = event["cell"]["name"], event["cell"]["method"]
        raw = (directory / "raw.before.jsonl").read_bytes()
        log = (directory / "log.before.log").read_bytes()
        final = (EXTREME / "raw" / f"{cell}.jsonl").read_bytes()
        position = methods.index(method)
        lines = final.splitlines(keepends=True)
        if (event.get("state") != "completed"
                or event.get("public_log_digest_scope") is None
                or len(lines) != 56 or len(raw.splitlines()) != position
                or (raw and not raw.endswith(b"\n"))
                or not final.startswith(raw + lines[position])
                or json.loads(lines[position]).get("method") != method
                or json.loads(lines[position]).get("status") != "timeout"
                or before.get("raw_record_count") != position
                or before.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
                or before.get("log_sha256") != hashlib.sha256(log).hexdigest()
                or event.get("appended_row_sha256") != hashlib.sha256(lines[position]).hexdigest()
                or after.get("raw_record_count") != position + 1
                or after.get("raw_sha256") != hashlib.sha256(raw + lines[position]).hexdigest()
                or after.get("appended_method") != method
                or after.get("appended_status") != "timeout"):
            raise ValueError(f"watchdog snapshot differs from final public row: {directory}")
        count += 1
    return {"watchdog_interventions_verified": count}


def verify_adaptive() -> dict:
    """Verify the separate AE/VAE minibatch sensitivity and primary lineage."""
    plan = json.loads((ADAPTIVE / "plan_manifest.json").read_text(encoding="utf-8"))
    manifest_path = ADAPTIVE / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((ADAPTIVE / "analysis/adaptive_batch_audit.json")
                       .read_text(encoding="utf-8"))
    fixed_keys = (
        "sensitivity_protocol", "new_fractions", "historical_fraction", "seeds",
        "representations", "methods", "training_batch_rule", "inference_batch_size",
        "expected_new_cells", "expected_old25_small_fit_cells", "expected_cells",
        "expected_records", "adaptive_cells", "historical_small_fit_cells",
    )
    code = manifest.get("code_sha256", {})
    code_files = {
        "primary_runner.py": PROJECT / "runner.py",
        "primary_model_factory.py": PROJECT / "model_factory.py",
        "adaptive_runner.py": ADAPTIVE / "runner.py",
        "adaptive_sweep.py": ADAPTIVE / "sweep.py",
        "adaptive_analyze.py": ADAPTIVE / "analyze.py",
    }
    if (manifest.get("sensitivity_protocol") != "adaptive-training-batch-v1"
            or manifest.get("methods") != ["AutoEncoder", "VAE"]
            or manifest.get("expected_cells") != 370
            or manifest.get("expected_records") != 740
            or manifest.get("train_cap") != 600 or manifest.get("test_cap") != 2000
            or manifest.get("method_timeout_seconds") != 15.0
            or any(manifest.get(key) != plan.get(key) for key in fixed_keys)
            or set(code) != set(code_files)
            or any(code[name] != sha(path) for name, path in code_files.items())
            or manifest.get("primary_historical_manifest_sha256")
                != sha(FULL / "run_manifest.json")
            or manifest.get("primary_extreme_manifest_sha256")
                != sha(EXTREME / "run_manifest.json")
            or manifest.get("dataset_manifest_sha256")
                != sha(PROJECT / "dataset_manifest.json")):
        raise ValueError("adaptive plan, pinned code, or primary manifest lineage changed")
    datasets = json.loads((PROJECT / "dataset_manifest.json").read_text())["datasets"]
    if manifest.get("dataset_sha256") != {item["filename"]: item["sha256"]
                                               for item in datasets}:
        raise ValueError("adaptive dataset hash list differs from frozen ODDS manifest")
    source = ROOT / manifest["source_repo"]
    provenance = json.loads((source / "SOURCE_PROVENANCE.json").read_text())
    if (manifest.get("source_git_commit") != provenance["commit"]
            or manifest.get("source_git_tree") != provenance["tree"]
            or set(manifest.get("source_pyod_sha256", {})) != {
                "pyod/models/base_dl.py", "pyod/models/auto_encoder.py",
                "pyod/models/vae.py", "pyod/models/nsa.py"}
            or any(sha(source / name) != digest
                   for name, digest in manifest["source_pyod_sha256"].items())):
        raise ValueError("adaptive PyOD source lineage changed")
    primary_files = {path.name: path for directory in (FULL / "raw", EXTREME / "raw")
                     for path in directory.glob("*.jsonl")}
    primary_hashes = {name: sha(path) for name, path in primary_files.items()}
    if (len(primary_files) != 1020
            or manifest.get("primary_raw_cell_sha256") != primary_hashes):
        raise ValueError("adaptive manifest does not pin all 1,020 primary raw cells")
    new_stems = {expected_stem(item["name"], rep, seed, pct)
                 for item in datasets for rep in REPS for seed in SEEDS
                 for pct in NEW_PCTS}
    old_small = {path.stem for path in (FULL / "raw").glob("*_retain25.jsonl")
                 if json.loads(path.read_text(encoding="utf-8").splitlines()[0])["fit_normals"] < 32}
    expected = new_stems | old_small
    raw = {path.stem: path for path in (ADAPTIVE / "raw").glob("*.jsonl")}
    logs = {path.stem: path for path in (ADAPTIVE / "logs").glob("*.log")}
    if (len(new_stems) != 340 or len(old_small) != 30 or len(expected) != 370
            or set(manifest["adaptive_cells"]) != expected
            or set(manifest["historical_small_fit_cells"]) != old_small
            or set(raw) != expected or set(logs) != expected):
        raise ValueError("adaptive raw/log grid differs from 340 new plus 30 old small-fit cells")
    statuses = Counter()
    strict = Counter()
    adapted_count = 0
    adapted_by_pct = Counter()
    trained_successes = 0
    adaptive_hashes: dict[str, str] = {}
    for stem in sorted(expected):
        primary_path = primary_files[f"{stem}.jsonl"]
        original = {row["method"]: row for row in
                    (json.loads(line) for line in primary_path.read_bytes().splitlines())}
        content = raw[stem].read_bytes()
        lines = content.splitlines()
        if len(lines) != 2 or not content.endswith(b"\n"):
            raise ValueError(f"adaptive cell is not two terminated rows: {stem}")
        adaptive_hashes[f"{stem}.jsonl"] = hashlib.sha256(content).hexdigest()
        rows = [json.loads(line) for line in lines]
        if [row.get("method") for row in rows] != ["AutoEncoder", "VAE"]:
            raise ValueError(f"adaptive method roster/order changed: {stem}")
        for row in rows:
            reference = original[row["method"]]
            if any(row.get(field) != reference.get(field) for field in ADAPTIVE_SPLIT_FIELDS):
                raise ValueError(f"adaptive normal-only split differs from primary: {stem}/{row['method']}")
            meta = row.get("adaptive_batch")
            if not isinstance(meta, dict):
                raise ValueError(f"adaptive batch metadata missing: {stem}/{row['method']}")
            effective = min(32, row["fit_normals"])
            if (meta.get("sensitivity_protocol") != "adaptive-training-batch-v1"
                    or meta.get("original_training_batch_size") != 32
                    or meta.get("effective_training_batch_size") != effective
                    or meta.get("inference_batch_size") != 32
                    or meta.get("adapter_applied") is not (effective < 32)
                    or meta.get("full_training_minibatches_per_epoch")
                        != row["fit_normals"] // effective
                    or meta.get("primary_cell_sha256") != primary_hashes[primary_path.name]
                    or meta.get("primary_method_status") != reference["status"]
                    or meta.get("source_runner_sha256") != sha(PROJECT / "runner.py")
                    or meta.get("source_model_factory_sha256")
                        != sha(PROJECT / "model_factory.py")
                    or not isinstance(meta.get("observed_training_forward_calls"), int)
                    or meta["observed_training_forward_calls"] < 0):
                raise ValueError(f"adaptive batch or hash lineage invalid: {stem}/{row['method']}")
            adapted_count += int(effective < 32)
            if effective < 32:
                adapted_by_pct[str(round(100 * row["retention_fraction"]))] += 1
            state = row.get("status")
            if state not in {"success", "timeout", "error", "build_error"}:
                raise ValueError(f"adaptive method status invalid: {stem}/{row['method']}")
            if state == "success":
                expected_epochs = 10 if row["method"] == "AutoEncoder" else 30
                if (meta.get("intended_epochs") != expected_epochs
                        or meta["observed_training_forward_calls"]
                            != expected_epochs * meta["full_training_minibatches_per_epoch"]
                        or not finite_unit(row.get("auroc"))
                        or not finite_unit(row.get("ap"))
                        or not valid_duration(row.get("fit_seconds"))
                        or not valid_duration(row.get("score_seconds"))):
                    raise ValueError(f"adaptive success lacks completed training or valid metrics: {stem}")
                trained_successes += 1
                strict["overbudget_success" if row["fit_seconds"] + row["score_seconds"] > 15.0
                       else "success"] += 1
            else:
                if (row.get("auroc") is not None or row.get("ap") is not None) and not (
                        state == "timeout" and row.get("ap") is None
                        and finite_unit(row.get("auroc"))):
                    raise ValueError(f"adaptive failure has scored outcome: {stem}/{row['method']}")
                strict[state] += 1
            statuses[state] += 1
    if adapted_count != 540 or dict(adapted_by_pct) != {"5": 340, "10": 140, "25": 60}:
        raise ValueError("adaptive batch eligibility differs from frozen fit-normal counts")
    table_rows = {
        "adaptive_batch_cell_comparison.csv": 740,
        "adaptive_batch_method_summary.csv": 24,
        "adaptive_batch_fixed_rank_sensitivity.csv": 312,
    }
    for name, expected_count in table_rows.items():
        path = ADAPTIVE / "analysis" / name
        with path.open(newline="", encoding="utf-8") as stream:
            actual_count = sum(1 for _ in csv.DictReader(stream))
        if (actual_count != expected_count
                or audit.get("output_rows", {}).get(name) != expected_count
                or audit.get("output_sha256", {}).get(name) != sha(path)):
            raise ValueError(f"adaptive analysis table changed: {name}")
    primary_audit_path = ANALYSIS / "extreme_retention_analysis_audit.json"
    primary_audit = json.loads(primary_audit_path.read_text(encoding="utf-8"))
    if (audit.get("expected_cells") != 370 or audit.get("actual_cells") != 370
            or audit.get("expected_records") != 740 or audit.get("actual_records") != 740
            or audit.get("primary_raw_files") != 1020
            or audit.get("primary_aevae_rows") != 2040
            or audit.get("selected_adaptive_aevae_rows") != 740
            or audit.get("selected_reused_primary_aevae_rows") != 1300
            or audit.get("selected_total_aevae_rows") != 2040
            or audit.get("adapter_applied_rows") != adapted_count
            or audit.get("adapter_applied_rows_by_retention_pct") != dict(adapted_by_pct)
            or audit.get("successful_rows_with_observed_training") != trained_successes
            or audit.get("raw_status_counts") != dict(statuses)
            or audit.get("strict_status_counts") != dict(strict)
            or audit.get("primary_raw_cell_sha256") != primary_hashes
            or audit.get("adaptive_raw_cell_sha256") != adaptive_hashes
            or audit.get("primary_run_manifest_sha256") != sha(manifest_path)
            or audit.get("primary_analysis_audit_sha256") != sha(primary_audit_path)
            or audit.get("primary_method_summary_sha256")
                != sha(ANALYSIS / "extreme_retention_method_summary.csv")
            or audit.get("fixed_fields", {}).get("primary") != {
                rep: primary_audit["common_fields"][rep]["baseline"] for rep in REPS}):
        raise ValueError("adaptive audit or primary-selection lineage differs from raw evidence")
    overlap_path = ADAPTIVE / "provenance/startup_timing_overlap.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    affected = overlap.get("affected_first_wave_cells", [])
    expected_first_wave = {
        f"{dataset}_native_seed42_retain10"
        for dataset in ("lympho", "cardio", "arrhythmia", "glass", "mnist", "letter",
                        "ionosphere", "musk")
    }
    if (overlap.get("schema_version") != 1 or len(affected) != 8
            or {item.get("cell") for item in affected} != expected_first_wave):
        raise ValueError("adaptive startup timing observation cohort changed")
    finished = datetime.fromisoformat(overlap["independent_source_pca_audit_finished_utc"])
    started = datetime.fromisoformat(overlap["adaptive_run_manifest_created_utc"])
    if finished < started:
        raise ValueError("adaptive startup timing observation chronology changed")
    for item in affected:
        stem = item["cell"]
        observations = item.get("method_status_and_measured_seconds", [])
        cell_rows = [json.loads(line) for line in raw[stem].read_bytes().splitlines()]
        if (item.get("raw_sha256") != adaptive_hashes[f"{stem}.jsonl"]
                or len(observations) != 2
                or datetime.fromisoformat(item["raw_last_write_utc"]) < finished):
            raise ValueError("adaptive startup timing observation differs from published raw")
        for observation, row in zip(observations, cell_rows, strict=True):
            seconds = row["fit_seconds"] + row["score_seconds"]
            if (observation.get("method") != row["method"]
                    or observation.get("status") != row["status"]
                    or row["status"] != "success"
                    or observation.get("fit_plus_score_seconds") != seconds
                    or not seconds < 5.1):
                raise ValueError("adaptive startup timing observation eligibility changed")
    return {
        "adaptive_cells": len(raw), "adaptive_records": sum(statuses.values()),
        "adaptive_training_batch_changed_rows": adapted_count,
        "adaptive_successes_with_forward_calls": trained_successes,
        "adaptive_reused_primary_rows": 1300,
        "adaptive_startup_timing_observation_cells": len(affected),
        "adaptive_startup_timing_observation_records": 2 * len(affected),
    }


def main() -> None:
    extension, base = verify_files()
    status, strict, details = verify_grid()
    analysis = verify_analysis(status, strict, details)
    watchdog = verify_watchdog()
    adaptive = verify_adaptive()
    print(json.dumps({
        "verified": True,
        "frozen_base_manifest_files": len(base["files"]),
        "frozen_benchmark_result_files": len(
            extension["protected_existing_benchmark_result_file_sha256"]),
        "extension_manifest_files": len(extension["files"]),
        "old_cells": details["old_files"], "new_cells": details["new_files"],
        "old_records": details["old_rows"], "new_records": details["new_rows"],
        "combined_records": details["old_rows"] + details["new_rows"],
        "zero_fit_outliers": True,
        **analysis,
        **watchdog,
        **adaptive,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
