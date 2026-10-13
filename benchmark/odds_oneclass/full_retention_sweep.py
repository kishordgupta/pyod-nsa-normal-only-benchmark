"""Run the complete PyOD/NSA normal-only retention benchmark.

One subprocess evaluates all 56 declared methods for a dataset/seed/feature
representation/retention cell. Each cell has a separate, append-only JSONL
file; rerunning this command resumes missing methods without changing completed
records. The same 100% filenames as scheduler.py are used so an already
started full-training cell is resumed, rather than copied or duplicated.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loader import read_manifest
from model_factory import METHODS, METHOD_META, NSA_METHODS, BASELINE_CLASSES
from runner import PROTOCOL_VERSION, split_data


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "full_retention"
FRACTIONS = (1.0, 0.75, 0.5, 0.25)
SEEDS = (42, 43, 44, 45, 46)
REPRESENTATIONS = ("native", "pca2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(path: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True,
            stderr=subprocess.DEVNULL, timeout=20,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def record_name(dataset: str, representation: str, seed: int,
                fraction: float) -> str:
    suffix = "" if fraction == 1.0 else f"_retain{round(fraction * 100)}"
    return f"{dataset}_{representation}_seed{seed}{suffix}"


def validate_existing(path: Path, dataset: str, representation: str,
                      seed: int, fraction: float, data_path: Path,
                      train_cap: int, test_cap: int) -> int:
    if not path.exists():
        return 0
    # Existing output is only safe to resume when it belongs to this exact
    # split/protocol. In particular, a retained set must preserve parent order.
    _, _, _, expected = split_data(
        data_path, seed, train_cap, test_cap, representation, fraction
    )
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed {path}:{line_number}: {exc}") from exc
            method = row.get("method")
            if method not in METHODS or method in seen:
                raise ValueError(f"Unexpected/duplicate method in {path}:{line_number}: {method}")
            seen.add(method)
            checks = (
                "dataset", "representation", "seed", "retention_fraction",
                "protocol_version", "fit_index_sha256",
                "fit_index_ordered_sha256", "fit_parent_index_sha256",
                "test_index_sha256", "fit_normals", "fit_outliers",
                "test_normals", "test_outliers", "train_cap", "test_cap",
            )
            mismatch = [key for key in checks if row.get(key) != expected.get(key)]
            if mismatch:
                raise ValueError(
                    f"Incompatible record {path}:{line_number}, method={method}, "
                    f"fields={mismatch}; preserve it separately before rerunning"
                )
    return len(seen)


def make_manifest(args: argparse.Namespace, datasets: list[str],
                  data_manifest: dict) -> dict:
    source_repo = Path(sys.executable).resolve().parents[3] / "pyod-nsa"
    # The editable install is authoritative if it differs from the usual
    # project layout. Read the path without importing the entire model stack.
    try:
        import pyod
        source_repo = Path(pyod.__file__).resolve().parents[1]
    except ImportError:
        pass
    packages = {}
    for name in ("pyod", "numpy", "scipy", "scikit-learn", "torch",
                 "xgboost", "combo", "suod", "pythresh"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    dataset_entries = {entry["name"]: entry for entry in data_manifest["datasets"]}
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol": "60/40 stratified ODDS split; discard all train-partition anomalies; cap original normal fit at 600 and mixed test at 2000; draw nested retained normal subsets; fit scaler and optional PCA2 on retained normals only; model.fit receives X only; held-out mixed test supplies labels for AUROC/AP only",
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_repo": str(source_repo),
        "source_git_commit": json.loads((source_repo / "SOURCE_PROVENANCE.json").read_text())["commit"],
        "source_git_tree": json.loads((source_repo / "SOURCE_PROVENANCE.json").read_text())["tree"],
        "source_git_status": "vendored source snapshot; verify public-copy-manifest.json",
        "source_nsa_sha256": sha256(source_repo / "pyod/models/nsa.py"),
        "code_sha256": {p.name: sha256(p) for p in (
            HERE / "runner.py", HERE / "model_factory.py", HERE / "loader.py",
            HERE / "full_retention_sweep.py")},
        "dataset_manifest_sha256": sha256(HERE / "dataset_manifest.json"),
        "datasets": [{"name": name, "filename": dataset_entries[name]["filename"],
                      "sha256": dataset_entries[name]["sha256"]} for name in datasets],
        "data_dir": str(args.data_dir.resolve()),
        "train_cap": args.train_cap,
        "test_cap": args.test_cap,
        "method_timeout_seconds": args.method_timeout,
        "fit_seconds_note": "Includes lazy model import and construction as well as fit, and is not isolated fit time.",
        "fractions_retained": list(args.fractions),
        "seeds": list(args.seeds),
        "representations": list(args.representations),
        "methods": METHODS,
        "method_meta": METHOD_META,
        "method_count": len(METHODS),
        "nsa_configuration_count": len(NSA_METHODS),
        "baseline_count": len(BASELINE_CLASSES),
        "expected_cells": len(datasets) * len(args.seeds) * len(args.fractions) * len(args.representations),
        "expected_records": len(datasets) * len(args.seeds) * len(args.fractions) * len(args.representations) * len(METHODS),
        "result_naming": "{dataset}_{representation}_seed{seed}.jsonl at 100%; append _retain{percent} otherwise",
    }


def validate_manifest(existing_path: Path, proposed: dict) -> None:
    if not existing_path.exists():
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text(json.dumps(proposed, indent=2, sort_keys=True) + "\n")
        return
    existing = json.loads(existing_path.read_text())
    for key, value in proposed.items():
        if key == "created_utc":
            continue
        if existing.get(key) != value:
            raise ValueError(f"{existing_path} differs at {key}; use a separate --run-root")


def run_cell(cell: tuple[str, str, int, float], args: argparse.Namespace) -> tuple:
    dataset, representation, seed, fraction = cell
    name = record_name(dataset, representation, seed, fraction)
    raw_path = args.run_root / "raw" / f"{name}.jsonl"
    log_path = args.run_root / "logs" / f"{name}.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_existing = validate_existing(
        raw_path, dataset, representation, seed, fraction,
        args.data_dir / f"{dataset}.mat", args.train_cap, args.test_cap,
    )
    if n_existing == len(METHODS):
        return name, 0, len(METHODS), "complete"
    cmd = [sys.executable, str(HERE / "runner.py"),
           "--data-dir", str(args.data_dir), "--dataset", dataset,
           "--results", str(raw_path), "--seed", str(seed),
           "--representation", representation,
           "--train-cap", str(args.train_cap),
           "--test-cap", str(args.test_cap),
           "--train-fraction", str(fraction),
           "--method-timeout", str(args.method_timeout),
           "--method-filter", "all"]
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    cache_dir = args.run_root / "cache"
    for subdir in ("matplotlib", "xdg"):
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    env["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(cmd) + "\n")
        stream.flush()
        result = subprocess.run(cmd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode == 0:
        count = validate_existing(
            raw_path, dataset, representation, seed, fraction,
            args.data_dir / f"{dataset}.mat", args.train_cap, args.test_cap,
        )
        if count != len(METHODS):
            raise ValueError(f"{name}: only {count}/{len(METHODS)} methods recorded")
        return name, 0, count, "finished"
    return name, result.returncode, n_existing, "runner_failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    manifest = read_manifest()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(manifest["source"]["local_directory"]))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--fractions", nargs="+", type=float, default=list(FRACTIONS))
    parser.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-active-full-seed42", action="store_true",
                        help="Temporarily omit 100%%-retained seed-42 cells while a separate scheduler writes them; rerun without this flag afterward")
    parser.add_argument("--train-cap", type=int, default=600)
    parser.add_argument("--test-cap", type=int, default=2000)
    parser.add_argument("--method-timeout", type=float, default=15.0)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.data_dir = args.data_dir.resolve()
    datasets = sorted(entry["name"] for entry in manifest["datasets"])
    if args.datasets is not None:
        if set(args.datasets) - set(datasets):
            parser.error(f"Unknown datasets: {sorted(set(args.datasets) - set(datasets))}")
        datasets = sorted(set(args.datasets))
    if not args.fractions or set(args.fractions) - set(FRACTIONS):
        parser.error("Fractions must be chosen from 1.0, 0.75, 0.5, 0.25")
    if not args.representations or set(args.representations) - set(REPRESENTATIONS):
        parser.error("Representations must be native and/or pca2")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be a nonempty unique list")
    if len(set(args.fractions)) != len(args.fractions):
        parser.error("Fractions must not repeat")
    if len(set(args.representations)) != len(args.representations):
        parser.error("Representations must not repeat")
    if args.workers < 1 or args.train_cap < 2 or args.test_cap < 2 or args.method_timeout <= 0:
        parser.error("Workers, caps, and timeout must be positive")
    if len(METHODS) != 56 or len(NSA_METHODS) != 13 or len(BASELINE_CLASSES) != 43:
        raise ValueError("Model roster has changed from the declared 56-configuration grid")
    entries = {entry["name"]: entry for entry in manifest["datasets"]}
    for dataset in datasets:
        source = args.data_dir / entries[dataset]["filename"]
        if sha256(source) != entries[dataset]["sha256"]:
            raise ValueError(f"Dataset SHA256 differs from manifest: {source}")
    proposed = make_manifest(args, datasets, manifest)
    validate_manifest(args.run_root / "run_manifest.json", proposed)
    all_cells = [(dataset, representation, seed, fraction)
             for fraction in args.fractions
             for representation in args.representations
             for seed in args.seeds
             for dataset in datasets]
    cells = [cell for cell in all_cells
             if not (args.skip_active_full_seed42
                     and cell[2] == 42 and cell[3] == 1.0)]
    skipped = len(all_cells) - len(cells)
    print(f"GRID total_cells={len(all_cells)} submitted_cells={len(cells)} "
          f"skipped_active={skipped} methods={len(METHODS)} "
          f"expected_total_records={len(all_cells) * len(METHODS)} "
          f"run_root={args.run_root}", flush=True)
    errors = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, cell, args): cell for cell in cells}
        for future in cf.as_completed(futures):
            try:
                name, code, count, state = future.result()
            except Exception as exc:
                errors.append((futures[future], str(exc)))
                print(f"ERROR {futures[future]}: {exc}", flush=True)
                continue
            print(f"{name}: exit={code} records={count} state={state}", flush=True)
            if code != 0:
                errors.append((name, f"exit={code}"))
    if errors:
        raise SystemExit(f"{len(errors)} cells failed; rerun to resume: {errors[:5]}")
    if skipped:
        print(f"BATCH COMPLETE {len(cells)} submitted cells; {skipped} "
              "temporarily skipped cells require a later unfiltered resume", flush=True)
    else:
        print(f"COMPLETE {len(cells)} cells and {len(cells) * len(METHODS)} records", flush=True)


if __name__ == "__main__":
    main()
