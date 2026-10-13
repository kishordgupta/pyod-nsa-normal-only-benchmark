"""Prepare, then execute the isolated AutoEncoder/VAE batch sensitivity.

Preparation writes the fixed 370-cell plan while the primary benchmark runs.
Execution is deliberately explicit and requires the 1,020 primary raw cells
to be complete before pinning their hashes in a separate run manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXTREME = HERE.parent
PROJECT = EXTREME.parent
sys.path.insert(0, str(PROJECT))

from loader import read_manifest  # noqa: E402
from model_factory import METHODS  # noqa: E402

METHODS_ADAPTIVE = ("AutoEncoder", "VAE")
FRACTIONS = (.10, .05)
SEEDS = (42, 43, 44, 45, 46)
REPS = ("native", "pca2")
EXPECTED_NEW_CELLS = 340
EXPECTED_OLD25_CELLS = 30
EXPECTED_ADAPTIVE_CELLS = 370
EXPECTED_ADAPTIVE_RECORDS = 740


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def name_of(dataset: str, rep: str, seed: int, fraction: float) -> str:
    return f"{dataset}_{rep}_seed{seed}_retain{round(100 * fraction)}"


def complete_cell(path: Path, roster: list[str] | tuple[str, ...]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len(roster) or [r["method"] for r in rows] != list(roster):
        raise ValueError(f"cell not complete in pinned roster order: {path}")
    return rows


def plan(datasets: list[str], historical_raw: Path) -> dict:
    new = [(dataset, rep, seed, fraction)
           for fraction in FRACTIONS for rep in REPS
           for seed in SEEDS for dataset in datasets]
    if len(new) != EXPECTED_NEW_CELLS:
        raise ValueError("unexpected 5/10% adaptive cell count")
    old = []
    for rep in REPS:
        for seed in SEEDS:
            for dataset in datasets:
                cell = (dataset, rep, seed, .25)
                path = historical_raw / (name_of(*cell) + ".jsonl")
                rows = complete_cell(path, METHODS)
                if rows[0]["fit_normals"] < 32:
                    old.append(cell)
    if len(old) != EXPECTED_OLD25_CELLS:
        raise ValueError(f"expected 30 historical 25% small-fit cells, got {len(old)}")
    cells = new + old
    names = [name_of(*cell) for cell in cells]
    if len(cells) != EXPECTED_ADAPTIVE_CELLS or len(set(names)) != len(names):
        raise ValueError("adaptive grid duplicates or omissions")
    return {"schema_version": 1, "sensitivity_protocol": "adaptive-training-batch-v1",
            "purpose": "AutoEncoder/VAE training batch capped at normal fit size; inference batch held at default 32",
            "new_fractions": list(FRACTIONS), "historical_fraction": .25,
            "seeds": list(SEEDS), "representations": list(REPS),
            "methods": list(METHODS_ADAPTIVE),
            "training_batch_rule": "min(original default 32, fit_normals)",
            "inference_batch_size": 32,
            "expected_new_cells": EXPECTED_NEW_CELLS,
            "expected_old25_small_fit_cells": EXPECTED_OLD25_CELLS,
            "expected_cells": EXPECTED_ADAPTIVE_CELLS,
            "expected_records": EXPECTED_ADAPTIVE_RECORDS,
            "adaptive_cells": names,
            "historical_small_fit_cells": [name_of(*cell) for cell in old]}


def prepare(args: argparse.Namespace) -> tuple[dict, list[tuple]]:
    data_manifest = read_manifest()
    datasets = sorted(item["name"] for item in data_manifest["datasets"])
    if len(datasets) != 17:
        raise ValueError("expected 17 ODDS datasets")
    historical_raw = Path(args.historical_raw).resolve()
    fixed = plan(datasets, historical_raw)
    cells = [(dataset, rep, seed, fraction)
             for fraction in FRACTIONS for rep in REPS
             for seed in SEEDS for dataset in datasets]
    cells += [(dataset, rep, seed, .25)
              for rep in REPS for seed in SEEDS for dataset in datasets
              if name_of(dataset, rep, seed, .25) in fixed["historical_small_fit_cells"]]
    plan_path = args.run_root / "plan_manifest.json"
    args.run_root.mkdir(parents=True, exist_ok=True)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != fixed:
            raise ValueError(f"fixed adaptive plan changed: {plan_path}")
    else:
        plan_path.write_text(json.dumps(fixed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fixed, cells


def primary_snapshot(args: argparse.Namespace, datasets: list[str]) -> dict[str, str]:
    old_raw = Path(args.historical_raw).resolve()
    new_raw = Path(args.extreme_raw).resolve()
    snapshot = {}
    for root, fractions, expected in ((old_raw, (1.0, .75, .50, .25), 680),
                                      (new_raw, FRACTIONS, 340)):
        files = sorted(root.glob("*.jsonl"))
        if len(files) != expected:
            raise ValueError(f"primary raw grid incomplete: {root}, {len(files)}/{expected}")
        for fraction in fractions:
            for rep in REPS:
                for seed in SEEDS:
                    for dataset in datasets:
                        stem = (f"{dataset}_{rep}_seed{seed}" if fraction == 1.0
                                else name_of(dataset, rep, seed, fraction))
                        path = root / f"{stem}.jsonl"
                        complete_cell(path, METHODS)
                        if path.name in snapshot:
                            raise ValueError(f"duplicate primary filename {path.name}")
                        snapshot[path.name] = sha(path)
    if len(snapshot) != 1020:
        raise ValueError("primary six-size snapshot incomplete")
    return snapshot


def run_manifest(args: argparse.Namespace, fixed: dict,
                 snapshot: dict[str, str], data_manifest: dict) -> dict:
    historical_manifest = PROJECT / "full_retention" / "run_manifest.json"
    extreme_manifest = EXTREME / "run_manifest.json"
    old = json.loads(historical_manifest.read_text(encoding="utf-8"))
    new = json.loads(extreme_manifest.read_text(encoding="utf-8"))
    if (old["source_git_commit"] != new["source_git_commit"]
            or old["source_git_tree"] != new["source_git_tree"]
            or old["source_git_status"] != new["source_git_status"]
            or old["code_sha256"]["runner.py"] != sha(PROJECT / "runner.py")
            or new["code_sha256"]["runner.py"] != sha(PROJECT / "runner.py")
            or old["code_sha256"]["model_factory.py"] != sha(PROJECT / "model_factory.py")
            or new["code_sha256"]["model_factory.py"] != sha(PROJECT / "model_factory.py")):
        raise ValueError("primary source lineage changed")
    if old["packages"] != new["packages"]:
        raise ValueError("historical and extreme primary package environments differ")
    for package, version in old["packages"].items():
        try:
            installed = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            raise ValueError(f"installed {package} version differs from primary manifest")
    source = PROJECT.parents[1] / old["source_repo"]
    pinned = ("pyod/models/base_dl.py", "pyod/models/auto_encoder.py",
              "pyod/models/vae.py", "pyod/models/nsa.py")
    source_sha = {rel: sha(source / rel) for rel in pinned}
    if source_sha["pyod/models/nsa.py"] != old["source_nsa_sha256"]:
        raise ValueError("PyOD NSA source differs from frozen primary")
    provenance = json.loads((source / "SOURCE_PROVENANCE.json").read_text())
    source_head, source_tree = provenance["commit"], provenance["tree"]
    if (source_head != old["source_git_commit"]
            or source_tree != old["source_git_tree"]):
        raise ValueError("PyOD source revision changed")
    for entry in data_manifest["datasets"]:
        if sha(Path(args.data_dir) / entry["filename"]) != entry["sha256"]:
            raise ValueError(f"ODDS data changed: {entry['filename']}")
    code = {"primary_runner.py": sha(PROJECT / "runner.py"),
            "primary_model_factory.py": sha(PROJECT / "model_factory.py"),
            "adaptive_runner.py": sha(HERE / "runner.py"),
            "adaptive_sweep.py": sha(Path(__file__)),
            "adaptive_analyze.py": sha(HERE / "analyze.py")}
    return {**fixed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "primary_protocol_version": old["protocol_version"],
            "primary_historical_manifest_sha256": sha(historical_manifest),
            "primary_extreme_manifest_sha256": sha(extreme_manifest),
            "primary_historical_raw": str(Path(args.historical_raw).resolve()),
            "primary_extreme_raw": str(Path(args.extreme_raw).resolve()),
            "primary_raw_cell_sha256": snapshot,
            "source_repo": str(source.resolve()),
            "source_git_commit": source_head, "source_git_tree": source_tree,
            "source_git_status": old["source_git_status"],
            "source_pyod_sha256": source_sha,
            "packages": old["packages"],
            "code_sha256": code,
            "dataset_manifest_sha256": sha(PROJECT / "dataset_manifest.json"),
            "data_dir": str(Path(args.data_dir).resolve()),
            "dataset_sha256": {entry["filename"]: entry["sha256"]
                               for entry in data_manifest["datasets"]},
            "train_cap": 600, "test_cap": 2000,
            "method_timeout_seconds": 15.0,
            "strict_time_qualification": "raw success and measured fit_seconds + score_seconds <=15.0",
            "python": sys.executable,
            "workers": args.workers,
            "runtime_rule": "same pinned primary evaluate; batch cap set after build and before fit; original query batch retained"}


def validate_run_manifest(path: Path, proposed: dict) -> dict:
    if not path.exists():
        path.write_text(json.dumps(proposed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return proposed
    actual = json.loads(path.read_text(encoding="utf-8"))
    for key, value in proposed.items():
        if key == "created_utc":
            continue
        if actual.get(key) != value:
            raise ValueError(f"adaptive run manifest changed at {key}: {path}")
    return actual


def execute_cell(cell: tuple, args: argparse.Namespace, manifest: dict) -> tuple:
    dataset, rep, seed, fraction = cell
    stem = name_of(*cell)
    raw_path = args.run_root / "raw" / f"{stem}.jsonl"
    log_path = args.run_root / "logs" / f"{stem}.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) == 2 and [r["method"] for r in rows] == list(METHODS_ADAPTIVE):
            if all(r.get("adaptive_batch", {}).get("primary_cell_sha256")
                   == manifest["primary_raw_cell_sha256"][f"{stem}.jsonl"] for r in rows):
                return stem, 0, 2, "already_complete"
        elif len(rows) > 2:
            raise ValueError(f"too many adaptive rows: {raw_path}")
    command = [sys.executable, str(HERE / "runner.py"),
               "--manifest", str(args.run_root / "run_manifest.json"),
               "--dataset", dataset, "--seed", str(seed),
               "--representation", rep, "--fraction", str(fraction),
               "--results", str(raw_path)]
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    for subdir in ("matplotlib", "xdg", "staging"):
        (args.run_root / ("cache" if subdir != "staging" else "") / subdir).mkdir(
            parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(args.run_root / "cache" / "matplotlib")
    env["XDG_CACHE_HOME"] = str(args.run_root / "cache" / "xdg")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        outcome = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if outcome.returncode != 0:
        return stem, outcome.returncode, None, "runner_failed"
    rows = complete_cell(raw_path, METHODS_ADAPTIVE)
    return stem, 0, len(rows), "finished"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_manifest = read_manifest()
    parser.add_argument("--run-root", type=Path, default=PROJECT.parents[1] / "local_runs/adaptive_batch")
    parser.add_argument("--historical-raw", type=Path,
                        default=PROJECT / "full_retention" / "raw")
    parser.add_argument("--extreme-raw", type=Path, default=EXTREME / "raw")
    parser.add_argument("--data-dir", type=Path,
                        default=PROJECT.parents[1] / data_manifest["source"]["local_directory"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true",
                        help="Run only after the primary 1,020-cell grid is complete")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.run_root = args.run_root.resolve()
    args.historical_raw = args.historical_raw.resolve()
    args.extreme_raw = args.extreme_raw.resolve()
    args.data_dir = args.data_dir.resolve()
    fixed, cells = prepare(args)
    if not args.execute:
        print(json.dumps({"prepared_plan": str(args.run_root / "plan_manifest.json"),
                          "cells": fixed["expected_cells"],
                          "records": fixed["expected_records"],
                          "executed": False}, indent=2), flush=True)
        return
    if sys.version_info[:2] != (3, 12):
        raise ValueError("public reproduction requires Python 3.12")
    datasets = sorted(item["name"] for item in data_manifest["datasets"])
    snapshot = primary_snapshot(args, datasets)
    proposed = run_manifest(args, fixed, snapshot, data_manifest)
    manifest = validate_run_manifest(args.run_root / "run_manifest.json", proposed)
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_cell, cell, args, manifest): cell for cell in cells}
        for future in as_completed(futures):
            try:
                stem, code, count, status = future.result()
                print(f"{stem}: exit={code} records={count} state={status}", flush=True)
                if code:
                    errors.append((stem, code))
            except Exception as exc:
                errors.append((futures[future], f"{type(exc).__name__}: {exc}"))
                print(f"ERROR {futures[future]}: {exc}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} adaptive cells require review/resume: {errors[:8]}")
    print(f"COMPLETE adaptive cells={len(cells)} records={len(cells) * 2}", flush=True)


if __name__ == "__main__":
    main()
