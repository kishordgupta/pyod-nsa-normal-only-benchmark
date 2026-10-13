"""Independent data and record audit for the 10%/5% normal-only ODDS sweep.

This reconstructs split indices and preprocessing from hashed source MAT files
without importing runner.split_data. It validates the raw roster and metric
metadata, and optionally ties the low-retention rows to the four historical
sizes. Raw JSONL stores aggregate AUROC/AP, not per-test scores, so this audit
cannot numerically recompute those metrics; it checks their range and verifies
the frozen runner computes them from anomaly-oriented decision_function scores.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from model_factory import METHODS, METHOD_META, NSA_METHODS  # noqa: E402

LOW_PCTS = (10, 5)
HISTORICAL_PCTS = (100, 75, 50, 25)
ALL_PCTS = HISTORICAL_PCTS + LOW_PCTS
SEEDS = (42, 43, 44, 45, 46)
REPS = ("native", "pca2")
PROTOCOL = "retention-v2-parent-order"
TERMINAL = {
    "success", "error", "timeout", "build_error", "requires_labeled_outliers",
    "requires_1d", "requires_2d", "requires_2d_or_3d",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_digest(indices: np.ndarray, ordered: bool = False) -> str:
    values = np.asarray(indices, dtype="<i8")
    if not ordered:
        values = np.sort(values)
    return hashlib.sha256(values.tobytes()).hexdigest()


def cell_name(dataset: str, rep: str, seed: int, pct: int) -> str:
    return f"{dataset}_{rep}_seed{seed}_retain{pct}"


def assert_scoring_orientation(runner_path: Path) -> None:
    """Check that the frozen runner evaluates its direct outlier score as-is."""
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    evaluate = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "evaluate"), None)
    if evaluate is None:
        raise ValueError("runner.evaluate is missing")
    calls = [node for node in ast.walk(evaluate) if isinstance(node, ast.Call)]
    fit = [node for node in calls if isinstance(node.func, ast.Attribute)
           and isinstance(node.func.value, ast.Name)
           and node.func.value.id == "model" and node.func.attr == "fit"]
    score = [node for node in calls if isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "model" and node.func.attr == "decision_function"]
    def direct_metric(name: str) -> bool:
        return any(isinstance(node.func, ast.Name) and node.func.id == name
                   and len(node.args) >= 2
                   and isinstance(node.args[0], ast.Name) and node.args[0].id == "y_test"
                   and isinstance(node.args[1], ast.Name) and node.args[1].id == "raw_scores"
                   for node in calls)
    if (len(fit) != 1 or len(fit[0].args) != 1
            or not isinstance(fit[0].args[0], ast.Name)
            or fit[0].args[0].id != "X_fit"
            or len(score) != 1 or len(score[0].args) != 1
            or not isinstance(score[0].args[0], ast.Name)
            or score[0].args[0].id != "X_test"
            or not direct_metric("roc_auc_score")
            or not direct_metric("average_precision_score")):
        raise ValueError("runner score/fit call structure changed; orientation needs review")


def source_expectations(dataset_manifest: Path, data_dir: Path) -> tuple[list[str], dict, dict]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    entries = manifest["datasets"]
    names = sorted(entry["name"] for entry in entries)
    if len(names) != 17 or len(set(names)) != 17 or len(METHODS) != 56 or len(NSA_METHODS) != 13:
        raise ValueError("Expected 17 datasets, 56 methods, and 13 NSA configurations")
    expected: dict[tuple[str, int, int], dict] = {}
    sizes: dict[str, dict[str, int]] = {}
    for entry in entries:
        name = entry["name"]
        path = data_dir / entry["filename"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {path}")
        mat = loadmat(path)
        X = np.asarray(mat["X"], dtype=np.float64)
        y = np.asarray(mat["y"]).ravel().astype(np.int8)
        if (X.shape != (entry["n_samples"], entry["n_features"])
                or not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}
                or int(np.sum(y == 0)) != entry["n_normal"]
                or int(np.sum(y == 1)) != entry["n_anomaly"]):
            raise ValueError(f"source array/label mismatch: {name}")
        for seed in SEEDS:
            train, test = train_test_split(np.arange(len(y)), test_size=0.4,
                                           stratify=y, random_state=seed)
            parent = train[y[train] == 0]
            if len(parent) > 600:
                parent = np.sort(np.random.RandomState(seed + 10_000).choice(
                    parent, size=600, replace=False))
            if len(test) > 2000:
                _, selected = train_test_split(np.arange(len(test)), test_size=2000,
                                               stratify=y[test], random_state=seed + 20_000)
                test = np.sort(test[selected])
            permutation = np.random.RandomState(seed + 30_000).permutation(parent)
            fits = {}
            for pct in ALL_PCTS:
                if pct == 100:
                    fit = parent.copy()
                else:
                    n_keep = max(2, int(np.floor(len(parent) * pct / 100)))
                    fit = parent[np.isin(parent, permutation[:n_keep])]
                if len(fit) != (len(parent) if pct == 100 else n_keep) or np.any(y[fit] != 0):
                    raise ValueError(f"reconstructed normal-only fit mismatch: {name}/{seed}/{pct}")
                fits[pct] = fit
            for higher, lower in zip(ALL_PCTS, ALL_PCTS[1:]):
                if not set(fits[lower]).issubset(set(fits[higher])):
                    raise ValueError(f"retention subsets are not nested: {name}/{seed}/{lower}")
            sizes[f"{name}/seed{seed}"] = {
                "parent": len(parent), "retain10": len(fits[10]), "retain5": len(fits[5]),
            }
            for pct in ALL_PCTS:
                fit = fits[pct]
                item = {
                    "dataset_rows": len(y),
                    "source_features": X.shape[1],
                    "original_fit_normals": len(parent),
                    "fit_normals": len(fit),
                    "fit_outliers": 0,
                    "discarded_train_outliers": int(np.sum(y[train] == 1)),
                    "test_normals": int(np.sum(y[test] == 0)),
                    "test_outliers": int(np.sum(y[test] == 1)),
                    "fit_index_sha256": index_digest(fit),
                    "fit_index_ordered_sha256": index_digest(fit, ordered=True),
                    "fit_parent_index_sha256": index_digest(parent),
                    "test_index_sha256": index_digest(test),
                }
                if pct in LOW_PCTS:
                    scaled = StandardScaler().fit_transform(X[fit])
                    pca = PCA(n_components=2, svd_solver="full").fit(scaled)
                    item["pca2_retained_variance"] = float(
                        np.sum(pca.explained_variance_ratio_))
                expected[(name, seed, pct)] = item
    return names, expected, sizes


def verify_run_manifest(path: Path, names: list[str], dataset_manifest: Path,
                        *, partial: bool) -> dict | None:
    if not path.exists():
        if partial:
            return None
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("protocol_version") != PROTOCOL
            or sorted(item["name"] for item in manifest.get("datasets", [])) != names
            or set(manifest.get("seeds", [])) != set(SEEDS)
            or set(manifest.get("representations", [])) != set(REPS)
            or set(manifest.get("fractions_retained", [])) != {0.1, 0.05}
            or manifest.get("methods") != METHODS
            or manifest.get("train_cap") != 600 or manifest.get("test_cap") != 2000
            or manifest.get("method_timeout_seconds") != 15.0
            or manifest.get("expected_cells") != 340
            or manifest.get("expected_records") != 19_040
            or manifest.get("dataset_manifest_sha256") != digest(dataset_manifest)):
        raise ValueError("extreme run manifest differs from the 17×5×2×2×56 protocol")
    code_hashes = manifest.get("code_sha256", {})
    for filename in ("runner.py", "model_factory.py", "loader.py"):
        if filename not in code_hashes or code_hashes[filename] != digest(PROJECT / filename):
            raise ValueError(f"execution code hash changed: {filename}")
    return manifest


def verify_log(path: Path, raw_path: Path, fraction: float) -> int:
    if not path.is_file():
        raise ValueError(f"missing cell invocation log: {path}")
    commands = [line.removeprefix("COMMAND ") for line in
                path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.startswith("COMMAND ")]
    if not commands or len(set(commands)) != 1:
        raise ValueError(f"missing or non-identical resume commands: {path}")
    for command in commands:
        tokens = shlex.split(command)
        def flag(name: str) -> str:
            try:
                return tokens[tokens.index(name) + 1]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"missing {name} in {path}") from exc
        if (not any(Path(token).name == "runner.py" for token in tokens)
                or (PROJECT.parents[1] / flag("--results")).resolve() != raw_path.resolve()
                or not math.isclose(float(flag("--train-fraction")), fraction)
                or float(flag("--method-timeout")) != 15.0
                or flag("--method-filter") != "all"
                or flag("--train-cap") != "600"
                or flag("--test-cap") != "2000"):
            raise ValueError(f"cell command differs from expected protocol: {path}")
    return len(commands)


def verify_history(full_root: Path, observed_cells: set[tuple[str, str, int]],
                   expected: dict, *, require: bool) -> int:
    compared = 0
    package_path = PROJECT.parents[1] / "results" / "pyod_nsa_full_retention_package_audit.json"
    if require and not package_path.is_file():
        raise ValueError(f"frozen historical package audit is missing: {package_path}")
    packaged_hashes = (json.loads(package_path.read_text(encoding="utf-8"))
                       .get("members_sha256", {}) if package_path.is_file() else {})
    fields = ("fit_normals", "fit_outliers", "fit_index_sha256",
              "fit_index_ordered_sha256", "fit_parent_index_sha256",
              "test_index_sha256", "test_normals", "test_outliers")
    for dataset, rep, seed in sorted(observed_cells):
        for pct in HISTORICAL_PCTS:
            suffix = "" if pct == 100 else f"_retain{pct}"
            path = full_root / "raw" / f"{dataset}_{rep}_seed{seed}{suffix}.jsonl"
            if not path.is_file():
                if require:
                    raise ValueError(f"historical nested-size cell is missing: {path}")
                continue
            archived_hash = packaged_hashes.get(f"production/raw/{path.name}")
            if require and archived_hash is None:
                raise ValueError(f"historical cell is absent from frozen package: {path}")
            if archived_hash is not None and digest(path) != archived_hash:
                raise ValueError(f"historical raw differs from frozen package: {path}")
            with path.open(encoding="utf-8") as f:
                first = f.readline()
            if not first:
                if require:
                    raise ValueError(f"historical cell is empty: {path}")
                continue
            row = json.loads(first)
            ref = expected[(dataset, seed, pct)]
            if (row.get("dataset") != dataset or row.get("representation") != rep
                    or row.get("seed") != seed or row.get("retention_fraction") != pct / 100
                    or any(row.get(field) != ref[field] for field in fields)):
                raise ValueError(f"frozen historical split differs from reconstructed split: {path}")
            compared += 1
    return compared


def audit(run_root: Path, dataset_manifest: Path, data_dir: Path,
          full_root: Path, *, partial: bool, require_history: bool) -> dict:
    names, expected, sizes = source_expectations(dataset_manifest, data_dir)
    assert_scoring_orientation(PROJECT / "runner.py")
    manifest = verify_run_manifest(run_root / "run_manifest.json", names,
                                   dataset_manifest, partial=partial)
    if manifest is not None:
        historical_manifest = json.loads((full_root / "run_manifest.json").read_text())
        for key in ("source_git_commit", "source_git_tree", "source_nsa_sha256",
                    "dataset_manifest_sha256"):
            if manifest.get(key) != historical_manifest.get(key):
                raise ValueError(f"extreme and historical runs differ at {key}")
        for filename in ("runner.py", "model_factory.py", "loader.py"):
            if (manifest["code_sha256"][filename]
                    != historical_manifest.get("code_sha256", {}).get(filename)):
                raise ValueError(f"extreme and historical execution code differ: {filename}")
    expected_files = {
        f"{cell_name(d, rep, seed, pct)}.jsonl"
        for d in names for rep in REPS for seed in SEEDS for pct in LOW_PCTS
    }
    raw_dir = run_root / "raw"
    raw_files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    found_files = {path.name for path in raw_files}
    if found_files - expected_files or (not partial and found_files != expected_files):
        raise ValueError(f"extreme cell file grid mismatch: found={len(found_files)} "
                         f"missing={len(expected_files - found_files)} "
                         f"extra={sorted(found_files - expected_files)[:5]}")
    statuses: Counter = Counter()
    panel_statuses: dict[str, Counter] = defaultdict(Counter)
    method_panels: dict[str, Counter] = defaultdict(Counter)
    overbudget_by_method_panel: Counter = Counter()
    qualified_success_by_method_panel: Counter = Counter()
    overbudget = 0
    records = 0
    complete_cells = 0
    log_invocations = 0
    observed_cells: set[tuple[str, str, int]] = set()
    for path in raw_files:
        stem = path.stem
        found = False
        for pct in LOW_PCTS:
            suffix = f"_retain{pct}"
            if stem.endswith(suffix):
                prefix = stem.removesuffix(suffix)
                for rep in REPS:
                    for seed in SEEDS:
                        tail = f"_{rep}_seed{seed}"
                        if prefix.endswith(tail):
                            dataset = prefix.removesuffix(tail)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        if not found or dataset not in names:
            raise ValueError(f"unrecognized extreme cell filename: {path.name}")
        fraction = pct / 100
        observed_cells.add((dataset, rep, seed))
        ref = expected[(dataset, seed, pct)]
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > len(METHODS) or (not partial and len(lines) != len(METHODS)):
            raise ValueError(f"wrong method record count: {path}: {len(lines)}")
        if len(lines) == len(METHODS):
            complete_cells += 1
        log_invocations += verify_log(run_root / "logs" / f"{stem}.log", path, fraction)
        for line_number, line in enumerate(lines, 1):
            if not line.endswith("\n"):
                raise ValueError(f"unterminated JSONL row: {path}:{line_number}")
            row = json.loads(line)
            method = METHODS[line_number - 1]
            if row.get("method") != method:
                raise ValueError(f"method roster order/identity mismatch: {path}:{line_number}")
            if (row.get("dataset") != dataset or row.get("representation") != rep
                    or row.get("seed") != seed
                    or row.get("retention_fraction") != fraction
                    or not math.isclose(row.get("reduction_fraction", -1), 1 - fraction)
                    or row.get("protocol_version") != PROTOCOL
                    or row.get("train_cap") != 600 or row.get("test_cap") != 2000
                    or row.get("family") != METHOD_META[method]["family"]):
                raise ValueError(f"identity/protocol mismatch: {path}:{line_number}")
            for field in ("dataset_rows", "source_features", "fit_normals",
                          "original_fit_normals", "fit_outliers", "discarded_train_outliers",
                          "test_normals", "test_outliers", "fit_index_sha256",
                          "fit_index_ordered_sha256", "fit_parent_index_sha256",
                          "test_index_sha256"):
                if row.get(field) != ref[field]:
                    raise ValueError(f"{field} split mismatch: {path}:{line_number}")
            variance = row.get("retained_variance")
            target_variance = 1.0 if rep == "native" else ref["pca2_retained_variance"]
            if (not isinstance(variance, (int, float)) or not math.isfinite(variance)
                    or not math.isclose(variance, target_variance, rel_tol=1e-8,
                                        abs_tol=1e-10)):
                raise ValueError(f"normal-only transform variance mismatch: {path}:{line_number}")
            status = row.get("status")
            if status not in TERMINAL:
                raise ValueError(f"unknown/nonterminal status: {path}:{line_number}: {status}")
            expected_inapplicable = METHOD_META[method].get("applicability", "applicable")
            if expected_inapplicable != "applicable" and status != expected_inapplicable:
                raise ValueError(f"labeled-outlier requirement not honored: {path}:{line_number}")
            if method == "MAD" and status != "requires_1d":
                raise ValueError(f"MAD geometry requirement not honored: {path}:{line_number}")
            if method == "NSA/voronoi" and (2 if rep == "pca2" else ref["source_features"]) not in (2, 3) \
                    and status != "requires_2d_or_3d":
                raise ValueError(f"Voronoi geometry requirement not honored: {path}:{line_number}")
            if method == "NSA/deterministic" and (2 if rep == "pca2" else ref["source_features"]) != 2 \
                    and status != "requires_2d":
                raise ValueError(f"deterministic geometry requirement not honored: {path}:{line_number}")
            if status == "success":
                for metric in ("auroc", "ap"):
                    value = row.get(metric)
                    if (not isinstance(value, (int, float)) or not math.isfinite(value)
                            or not -1e-12 <= value <= 1 + 1e-12):
                        raise ValueError(f"invalid {metric}: {path}:{line_number}")
                for timing in ("fit_seconds", "score_seconds"):
                    value = row.get(timing)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {timing}: {path}:{line_number}")
                if not isinstance(row.get("score_unique"), int) or not 1 <= row["score_unique"] <= ref["test_normals"] + ref["test_outliers"]:
                    raise ValueError(f"invalid score uniqueness count: {path}:{line_number}")
                method_panel = f"{rep}/retain{pct}/{method}"
                if row["fit_seconds"] + row["score_seconds"] > 15.0:
                    overbudget += 1
                    overbudget_by_method_panel[method_panel] += 1
                else:
                    qualified_success_by_method_panel[method_panel] += 1
            else:
                auc, ap = row.get("auroc"), row.get("ap")
                if (auc is not None or ap is not None) and not (
                        status == "timeout" and ap is None
                        and isinstance(auc, (int, float)) and math.isfinite(auc)
                        and -1e-12 <= auc <= 1 + 1e-12):
                    raise ValueError(f"non-success row has a scored outcome: {path}:{line_number}")
            statuses[status] += 1
            panel = f"{rep}/retain{pct}"
            panel_statuses[panel][status] += 1
            method_panels[f"{panel}/{method}"][status] += 1
            records += 1
    if not partial and (records != 19_040 or complete_cells != 340):
        raise ValueError(f"extreme grid incomplete: {complete_cells}/340 cells, {records}/19040 rows")
    historical_compared = verify_history(full_root, observed_cells, expected,
                                         require=require_history)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL,
        "completion_mode": "partial" if partial else "complete",
        "expected_cells": 340,
        "observed_cells": len(found_files),
        "complete_cells": complete_cells,
        "expected_records": 19_040,
        "observed_records": records,
        "source_datasets_verified": len(names),
        "source_manifest_sha256": digest(dataset_manifest),
        "runner_sha256": digest(PROJECT / "runner.py"),
        "model_factory_sha256": digest(PROJECT / "model_factory.py"),
        "score_orientation_check": "model.fit(X_fit), direct model.decision_function(X_test), roc_auc_score(y_test, raw_scores), average_precision_score(y_test, raw_scores)",
        "metric_recalculation_limit": "Raw records do not retain per-test score vectors; aggregate AUROC/AP cannot be numerically recomputed from JSONL alone.",
        "fit_size_bounds": {
            "retain10_min": min(row["retain10"] for row in sizes.values()),
            "retain10_max": max(row["retain10"] for row in sizes.values()),
            "retain5_min": min(row["retain5"] for row in sizes.values()),
            "retain5_max": max(row["retain5"] for row in sizes.values()),
        },
        "all_fit_rows_independently_verified_normal": True,
        "nested_split_rule_verified": True,
        "historical_size_cells_crosschecked": historical_compared,
        "required_historical_crosscheck": require_history,
        "log_invocations": log_invocations,
        "raw_status_counts": dict(sorted(statuses.items())),
        "panel_status_counts": {k: dict(sorted(v.items())) for k, v in sorted(panel_statuses.items())},
        "method_panel_status_counts": {k: dict(sorted(v.items())) for k, v in sorted(method_panels.items())},
        "raw_successes_measured_over_15_seconds": overbudget,
        "measured_overbudget_by_method_panel": dict(sorted(overbudget_by_method_panel.items())),
        "qualified_success_by_method_panel": dict(sorted(qualified_success_by_method_panel.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=HERE)
    parser.add_argument("--dataset-manifest", type=Path, default=PROJECT / "dataset_manifest.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--full-root", type=Path, default=PROJECT / "full_retention")
    parser.add_argument("--partial", action="store_true", help="Audit available in-progress cell prefixes")
    parser.add_argument("--require-history", action="store_true", help="Require all four prior sizes for each observed dataset/seed/representation")
    parser.add_argument("--output", type=Path, help="Optional JSON audit destination")
    args = parser.parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        data_dir = PROJECT.parents[1] / manifest["source"]["local_directory"]
    result = audit(args.run_root, args.dataset_manifest, data_dir, args.full_root,
                   partial=args.partial, require_history=args.require_history)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
