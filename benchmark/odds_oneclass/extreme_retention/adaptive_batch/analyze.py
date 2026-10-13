"""Audit the separate AutoEncoder/VAE training-batch sensitivity.

The 56-method primary results remain unchanged. This analysis substitutes
only the two declared adaptive PyOD baseline records, then recalculates
six-size complete-coverage comparison fields and NSA AUROC/AP ranks.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import importlib.util
import importlib.metadata as package_metadata
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXTREME = HERE.parent
PROJECT = EXTREME.parent
sys.path.insert(0, str(PROJECT))

from model_factory import METHODS, NSA_METHODS  # noqa: E402

_spec = importlib.util.spec_from_file_location("_adaptive_batch_wrapper", HERE / "runner.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot import the adaptive batch wrapper")
adaptive_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adaptive_runner)

ADAPTIVE = adaptive_runner.METHODS
PCTS = (100, 75, 50, 25, 10, 5)
SEEDS = (42, 43, 44, 45, 46)
REPS = ("native", "pca2")
EXPECTED_PRIMARY_RECORDS = 57120
EXPECTED_ADAPTIVE_RECORDS = 740
TIME_BUDGET = 15.0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def macro(rows: list[dict], metric: str):
    datasets = defaultdict(list)
    for row in rows:
        datasets[row["dataset"]].append(float(row[metric]))
    return mean(mean(datasets[name]) for name in sorted(datasets))


def maybe_float(value):
    return None if value in (None, "") else float(value)


def strict_status(row: dict) -> str:
    if row["status"] == "success" and row["fit_seconds"] + row["score_seconds"] > TIME_BUDGET:
        return "overbudget_success"
    return row["status"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_manifest(root: Path) -> dict:
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    plan_path = root / "plan_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (manifest["sensitivity_protocol"] != "adaptive-training-batch-v1"
            or manifest["expected_cells"] != 370
            or manifest["expected_records"] != EXPECTED_ADAPTIVE_RECORDS
            or len(manifest["adaptive_cells"]) != 370
            or len(set(manifest["adaptive_cells"])) != 370
            or len(manifest["primary_raw_cell_sha256"]) != 1020
            or any(manifest.get(key) != value for key, value in plan.items())):
        raise ValueError("adaptive manifest does not match 370-cell fixed plan")
    code = manifest["code_sha256"]
    expected_code = {"primary_runner.py": PROJECT / "runner.py",
                     "primary_model_factory.py": PROJECT / "model_factory.py",
                     "adaptive_runner.py": HERE / "runner.py",
                     "adaptive_sweep.py": HERE / "sweep.py",
                     "adaptive_analyze.py": Path(__file__)}
    if set(code) != set(expected_code) or any(code[k] != sha(v) for k, v in expected_code.items()):
        raise ValueError("adaptive or primary code SHA changed")
    old_manifest = PROJECT / "full_retention" / "run_manifest.json"
    new_manifest = EXTREME / "run_manifest.json"
    if (manifest["primary_historical_manifest_sha256"] != sha(old_manifest)
            or manifest["primary_extreme_manifest_sha256"] != sha(new_manifest)
            or manifest["dataset_manifest_sha256"] != sha(PROJECT / "dataset_manifest.json")):
        raise ValueError("primary data/manifest lineage changed")
    for package, version in manifest["packages"].items():
        try:
            installed = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            raise ValueError(f"installed {package} version differs from adaptive run")
    source = PROJECT.parents[1] / manifest["source_repo"]
    if set(manifest["source_pyod_sha256"]) != {
            "pyod/models/base_dl.py", "pyod/models/auto_encoder.py",
            "pyod/models/vae.py", "pyod/models/nsa.py"}:
        raise ValueError("adaptive PyOD source SHA inventory incomplete")
    for rel, digest in manifest["source_pyod_sha256"].items():
        if sha(source / rel) != digest:
            raise ValueError(f"PyOD source changed: {rel}")
    provenance = json.loads((source / "SOURCE_PROVENANCE.json").read_text())
    head, tree = provenance["commit"], provenance["tree"]
    if (head != manifest["source_git_commit"] or tree != manifest["source_git_tree"]):
        raise ValueError("PyOD Git lineage changed")
    for filename, digest in manifest["dataset_sha256"].items():
        if sha(PROJECT.parents[1] / manifest["data_dir"] / filename) != digest:
            raise ValueError(f"source data changed: {filename}")
    return manifest


def load_primary(manifest: dict) -> tuple[dict, dict, dict]:
    roots = ((PROJECT.parents[1] / manifest["primary_historical_raw"], 680),
             (PROJECT.parents[1] / manifest["primary_extreme_raw"], 340))
    mapping: dict[tuple, dict] = {}
    hashes = {}
    filenames = set()
    for root, expected_files in roots:
        paths = sorted(root.glob("*.jsonl"))
        if len(paths) != expected_files:
            raise ValueError(f"primary raw file count differs: {root}")
        for path in paths:
            if path.name in filenames:
                raise ValueError(f"duplicate primary filename: {path.name}")
            filenames.add(path.name)
            digest = sha(path)
            if digest != manifest["primary_raw_cell_sha256"].get(path.name):
                raise ValueError(f"primary raw changed: {path}")
            hashes[path.name] = digest
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            if len(rows) != 56 or [r["method"] for r in rows] != METHODS:
                raise ValueError(f"primary cell incomplete: {path}")
            for row in rows:
                if row["method"] not in ADAPTIVE:
                    continue
                pct = round(100 * row["retention_fraction"])
                key = (row["dataset"], row["seed"], row["representation"], pct, row["method"])
                if key in mapping or row["fit_outliers"] != 0:
                    raise ValueError(f"primary AE/VAE duplicate or fitted outlier: {key}")
                mapping[key] = row
    if len(hashes) != 1020 or len(mapping) != 2 * 1020:
        raise ValueError("primary AE/VAE lineage incomplete")
    return mapping, hashes, {"files": len(hashes), "adaptive_method_rows": len(mapping)}


def load_adaptive(root: Path, manifest: dict, primary: dict) -> tuple[dict, dict, list[dict]]:
    raw_root = root / "raw"
    paths = sorted(raw_root.glob("*.jsonl"))
    if len(paths) != 370 or {p.stem for p in paths} != set(manifest["adaptive_cells"]):
        raise ValueError("adaptive raw file grid incomplete or unexpected")
    rows_by_key = {}
    hashes = {}
    comparison = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 2 or [r["method"] for r in rows] != list(ADAPTIVE):
            raise ValueError(f"adaptive cell incomplete: {path}")
        hashes[path.name] = sha(path)
        for row in rows:
            pct = round(100 * row["retention_fraction"])
            key = (row["dataset"], row["seed"], row["representation"], pct, row["method"])
            if key in rows_by_key or key not in primary:
                raise ValueError(f"adaptive duplicate or unknown row: {key}")
            if path.stem != adaptive_runner.cell_name(row["dataset"], row["representation"],
                                                     row["seed"], row["retention_fraction"]):
                raise ValueError(f"adaptive filename/cell mismatch: {path}")
            parent = primary[key]
            if (row["fit_outliers"] != 0
                    or not adaptive_runner.split_matches(row, parent)):
                raise ValueError(f"adaptive normal-only split drifted: {key}")
            meta = row.get("adaptive_batch")
            if not isinstance(meta, dict):
                raise ValueError(f"missing adaptive batch metadata: {key}")
            expected_batch = min(32, row["fit_normals"])
            primary_digest = manifest["primary_raw_cell_sha256"][path.name]
            if (meta.get("sensitivity_protocol") != manifest["sensitivity_protocol"]
                    or meta.get("primary_cell_sha256") != primary_digest
                    or meta.get("primary_method_status") != parent["status"]
                    or meta.get("source_runner_sha256") != manifest["code_sha256"]["primary_runner.py"]
                    or meta.get("source_model_factory_sha256") != manifest["code_sha256"]["primary_model_factory.py"]
                    or meta.get("original_training_batch_size") != 32
                    or meta.get("effective_training_batch_size") != expected_batch
                    or meta.get("inference_batch_size") != 32
                    or meta.get("adapter_applied") != (expected_batch < 32)
                    or meta.get("full_training_minibatches_per_epoch") != row["fit_normals"] // expected_batch):
                raise ValueError(f"adaptive batch metadata mismatch: {key}")
            calls = meta.get("observed_training_forward_calls")
            if not isinstance(calls, int) or calls < 0:
                raise ValueError(f"invalid observed training calls: {key}")
            if row["status"] == "success":
                if (calls < 1 or calls != meta["intended_epochs"] *
                        meta["full_training_minibatches_per_epoch"]
                        or not all(math.isfinite(float(row[m])) for m in ("auroc", "ap"))):
                    raise ValueError(f"successful adaptive row lacks actual training/finite metrics: {key}")
            if pct == 25 and row["fit_normals"] >= 32:
                raise ValueError(f"historical 25% sensitivity run was not needed: {key}")
            rows_by_key[key] = row
            comparison.append({"dataset": key[0], "seed": key[1],
                               "representation": key[2], "retention_pct": key[3],
                               "method": key[4], "fit_normals": row["fit_normals"],
                               "original_batch_size": 32,
                               "effective_training_batch_size": expected_batch,
                               "inference_batch_size": 32,
                               "adapter_applied": expected_batch < 32,
                               "training_forward_calls": calls,
                               "primary_raw_status": parent["status"],
                               "adaptive_raw_status": row["status"],
                               "primary_strict_status": strict_status(parent),
                               "adaptive_strict_status": strict_status(row),
                               "primary_auroc": parent["auroc"],
                               "adaptive_auroc": row["auroc"],
                               "primary_ap": parent["ap"],
                               "adaptive_ap": row["ap"],
                               "primary_fit_seconds": parent["fit_seconds"],
                               "adaptive_fit_seconds": row["fit_seconds"],
                               "primary_score_seconds": parent["score_seconds"],
                               "adaptive_score_seconds": row["score_seconds"],
                               "primary_cell_sha256": primary_digest,
                               "adaptive_cell_sha256": hashes[path.name]})
    if len(rows_by_key) != EXPECTED_ADAPTIVE_RECORDS:
        raise ValueError("adaptive record grid incomplete")
    return rows_by_key, hashes, comparison


def summarize(primary: dict, adaptive: dict, primary_summary: dict) -> tuple[list[dict], dict]:
    result = []
    changed = {}
    datasets = sorted({key[0] for key in primary})
    for rep in REPS:
        for pct in PCTS:
            for method in ADAPTIVE:
                chosen = [adaptive.get((dataset, seed, rep, pct, method),
                                       primary[(dataset, seed, rep, pct, method)])
                          for dataset in datasets for seed in SEEDS]
                original = [primary[(dataset, seed, rep, pct, method)]
                            for dataset in datasets for seed in SEEDS]
                if len(chosen) != 85:
                    raise ValueError("adaptive panel has fewer than 85 cells")
                good = [r for r in chosen if strict_status(r) == "success"]
                recorded = [r for r in chosen if r["status"] == "success"]
                original_good = [r for r in original if strict_status(r) == "success"]
                old_summary = primary_summary[rep, pct, method]
                for metric in ("auroc", "ap"):
                    expected = maybe_float(old_summary[f"macro_{metric}_qualified_available"])
                    actual = macro(original_good, metric)
                    if expected is None:
                        if actual is not None:
                            raise ValueError("primary summary/row macro mismatch")
                    elif actual is None or not math.isclose(expected, actual, abs_tol=1e-12, rel_tol=0):
                        raise ValueError("primary summary/row macro mismatch")
                record = {"representation": rep, "retention_pct": pct, "method": method,
                          "expected_seed_rows": 85,
                          "adaptive_source_seed_rows": sum((dataset, seed, rep, pct, method) in adaptive
                                                           for dataset in datasets for seed in SEEDS),
                          "reused_primary_seed_rows": sum((dataset, seed, rep, pct, method) not in adaptive
                                                          for dataset in datasets for seed in SEEDS),
                          "original_raw_success_seed_rows": sum(r["status"] == "success" for r in original),
                          "original_qualified_success_seed_rows": len(original_good),
                          "adaptive_raw_success_seed_rows": len(recorded),
                          "adaptive_qualified_success_seed_rows": len(good),
                          "adaptive_status_counts": json.dumps(dict(sorted(Counter(strict_status(r) for r in chosen).items()))),
                          "original_macro_auroc_qualified_available": macro(original_good, "auroc"),
                          "original_macro_ap_qualified_available": macro(original_good, "ap"),
                          "adaptive_macro_auroc_qualified_available": macro(good, "auroc"),
                          "adaptive_macro_ap_qualified_available": macro(good, "ap"),
                          "adaptive_macro_auroc_raw_available": macro(recorded, "auroc"),
                          "adaptive_macro_ap_raw_available": macro(recorded, "ap"),
                          "full_qualified_coverage": len(good) == 85}
                result.append(record)
                changed[rep, pct, method] = record
    if len(result) != 24 or sum(r["adaptive_source_seed_rows"] for r in result) != 740:
        raise ValueError("adaptive panel provenance incomplete")
    return result, changed


def fixed_ranks(primary_summary: dict, primary_audit: dict,
                changed: dict) -> tuple[list[dict], dict]:
    expected_base = {}
    sensitive_base = {}
    baselines = [method for method in METHODS if method not in NSA_METHODS]
    for rep in REPS:
        original = [m for m in baselines
                    if all(int(primary_summary[rep, pct, m]["qualified_success_seed_rows"]) == 85
                           for pct in PCTS)]
        if original != primary_audit["common_fields"][rep]["baseline"]:
            raise ValueError("primary fixed baseline field/summary mismatch")
        expected_base[rep] = original
        sensitive_base[rep] = [m for m in baselines
                               if all((changed[rep, pct, m]["adaptive_qualified_success_seed_rows"]
                                       if m in ADAPTIVE else
                                       int(primary_summary[rep, pct, m]["qualified_success_seed_rows"])) == 85
                                      for pct in PCTS)]
    rows = []
    for rep in REPS:
        for pct in PCTS:
            for method in NSA_METHODS:
                own = primary_summary[rep, pct, method]
                coverage = int(own["qualified_success_seed_rows"])
                for metric in ("auroc", "ap"):
                    focal = maybe_float(own[f"macro_{metric}_qualified_available"])
                    if coverage == 85 and focal is None:
                        raise ValueError("complete NSA has no macro score")

                    def score(m):
                        if m in ADAPTIVE:
                            value = changed[rep, pct, m][f"adaptive_macro_{metric}_qualified_available"]
                        else:
                            value = maybe_float(primary_summary[rep, pct, m][f"macro_{metric}_qualified_available"])
                        if value is None:
                            raise ValueError(f"complete field member {m} has no score")
                        return value

                    def primary_score(m):
                        value = maybe_float(primary_summary[rep, pct, m][f"macro_{metric}_qualified_available"])
                        if value is None:
                            raise ValueError(f"primary field member {m} has no score")
                        return value

                    strict_rank = (1 + sum(primary_score(m) > focal + 1e-12
                                           for m in expected_base[rep])) if coverage == 85 else None
                    adapted_rank = (1 + sum(score(m) > focal + 1e-12
                                            for m in sensitive_base[rep])) if coverage == 85 else None
                    rows.append({"representation": rep, "retention_pct": pct,
                                 "metric": metric, "nsa": method,
                                 "nsa_qualified_seed_rows": coverage,
                                 "nsa_macro_score": focal,
                                 "primary_fixed_baseline_rank": strict_rank,
                                 "primary_field_size_with_focal": len(expected_base[rep]) + 1,
                                 "adaptive_fixed_baseline_rank": adapted_rank,
                                 "adaptive_field_size_with_focal": len(sensitive_base[rep]) + 1,
                                 "primary_baseline_count": len(expected_base[rep]),
                                 "adaptive_baseline_count": len(sensitive_base[rep]),
                                 "field_added": ",".join(m for m in sensitive_base[rep]
                                                         if m not in expected_base[rep]),
                                 "field_removed": ",".join(m for m in expected_base[rep]
                                                           if m not in sensitive_base[rep]),
                                 "rank_rule": "competition rank; blank if focal is not 85/85 qualified"})
    if len(rows) != 2 * 6 * 13 * 2:
        raise ValueError("fixed rank sensitivity incomplete")
    return rows, {"primary": expected_base, "adaptive": sensitive_base}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT.parents[1] / "local_runs/adaptive_batch")
    parser.add_argument("--primary-analysis", type=Path, default=EXTREME / "analysis")
    parser.add_argument("--out-dir", type=Path, default=PROJECT.parents[1] / "local_runs/adaptive_batch/analysis")
    args = parser.parse_args()
    manifest = check_manifest(args.run_root)
    primary, primary_hashes, primary_size = load_primary(manifest)
    adaptive, adaptive_hashes, comparison = load_adaptive(args.run_root, manifest, primary)
    audit_path = args.primary_analysis / "extreme_retention_analysis_audit.json"
    summary_path = args.primary_analysis / "extreme_retention_method_summary.csv"
    primary_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (primary_audit["expected_records"] != EXPECTED_PRIMARY_RECORDS
            or primary_audit["retention_percent"] != list(PCTS)
            or primary_audit["methods"] != METHODS):
        raise ValueError("primary six-size analysis incomplete")
    with summary_path.open(newline="", encoding="utf-8") as stream:
        primary_rows = list(csv.DictReader(stream))
    primary_summary = {(r["representation"], int(r["retention_pct"]), r["method"]): r
                       for r in primary_rows}
    if len(primary_rows) != 672 or len(primary_summary) != 672:
        raise ValueError("primary method summary incomplete or duplicated")
    method_summary, changed = summarize(primary, adaptive, primary_summary)
    ranks, fields = fixed_ranks(primary_summary, primary_audit, changed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"adaptive_batch_cell_comparison.csv": comparison,
               "adaptive_batch_method_summary.csv": method_summary,
               "adaptive_batch_fixed_rank_sensitivity.csv": ranks}
    for filename, rows in outputs.items():
        write_csv(args.out_dir / filename, rows)
    raw_status = Counter(row["status"] for row in adaptive.values())
    strict_counts = Counter(strict_status(row) for row in adaptive.values())
    selected = {**primary, **adaptive}
    selected_reused = sum(key not in adaptive for key in primary)
    applied_by_pct = Counter(round(100 * row["retention_fraction"])
                             for row in adaptive.values()
                             if row["adaptive_batch"]["adapter_applied"])
    if applied_by_pct != Counter({5: 340, 10: 140, 25: 60}):
        raise ValueError(f"unexpected adaptive batch intervention grid: {applied_by_pct}")
    audit = {
        "schema_version": 1,
        "sensitivity_protocol": manifest["sensitivity_protocol"],
        "expected_cells": 370,
        "expected_records": EXPECTED_ADAPTIVE_RECORDS,
        "actual_cells": len(adaptive_hashes),
        "actual_records": len(adaptive),
        "historical_small_fit_cells": len(manifest["historical_small_fit_cells"]),
        "primary_raw_files": primary_size["files"],
        "primary_aevae_rows": primary_size["adaptive_method_rows"],
        "selected_adaptive_aevae_rows": len(adaptive),
        "selected_reused_primary_aevae_rows": selected_reused,
        "selected_total_aevae_rows": len(selected),
        "adapter_applied_rows": sum(row["adaptive_batch"]["adapter_applied"]
                                    for row in adaptive.values()),
        "adapter_applied_rows_by_retention_pct": dict(sorted(applied_by_pct.items())),
        "successful_rows_with_observed_training": sum(
            row["status"] == "success" and row["adaptive_batch"]["observed_training_forward_calls"] > 0
            for row in adaptive.values()),
        "raw_status_counts": dict(sorted(raw_status.items())),
        "strict_status_counts": dict(sorted(strict_counts.items())),
        "full_qualified_adaptive_panels": sum(r["full_qualified_coverage"] for r in method_summary),
        "fixed_fields": fields,
        "primary_run_manifest_sha256": sha(args.run_root / "run_manifest.json"),
        "primary_analysis_audit_sha256": sha(audit_path),
        "primary_method_summary_sha256": sha(summary_path),
        "primary_raw_cell_sha256": primary_hashes,
        "adaptive_raw_cell_sha256": adaptive_hashes,
        "output_sha256": {name: sha(args.out_dir / name) for name in outputs},
        "output_rows": {name: len(rows) for name, rows in outputs.items()},
        "qualification": "raw success and measured fit_seconds + score_seconds <= 15.0",
        "interpretation": "sensitivity only; original primary rows and their ranks remain unchanged; field sizes may differ",
    }
    (args.out_dir / "adaptive_batch_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(adaptive_hashes), "records": len(adaptive),
                      "adapter_applied": audit["adapter_applied_rows"],
                      "strict_success": strict_counts["success"],
                      "fixed_fields": {rep: {k: len(v[rep]) for k, v in fields.items()}
                                       for rep in REPS}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
