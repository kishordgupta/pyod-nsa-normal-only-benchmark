"""Optional 60-second feasibility panel for primary full-roster timeouts.

Run without ``--execute`` to audit readiness. Execution is allowed only after
all primary 15-second cells are complete. Retries are isolated under
``full_retention/timeout_retry`` and never replace the primary results.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from model_factory import METHODS


HERE = Path(__file__).resolve().parent
DEFAULT_PRIMARY = HERE / "full_retention"
RETRY_SECONDS = 60.0
PRIMARY_CODE_FILES = ("runner.py", "model_factory.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_primary_code(manifest: dict) -> dict[str, str]:
    """Require unchanged split/evaluation and model-construction code."""
    pinned = manifest.get("code_sha256")
    if not isinstance(pinned, dict):
        raise ValueError("primary manifest has no code SHA-256 inventory")
    verified: dict[str, str] = {}
    for name in PRIMARY_CODE_FILES:
        expected = pinned.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"primary manifest has no valid SHA-256 for {name}")
        actual = sha256(HERE / name)
        if actual != expected:
            raise ValueError(
                f"{name} changed since the primary 15-second run; "
                "retry would change more than the timeout"
            )
        verified[name] = actual
    return verified


def verify_primary_source(manifest: dict) -> dict[str, str]:
    """Require the tested clean editable PyOD checkout, not another install."""
    import pyod

    repo = Path(manifest["source_repo"]).resolve()
    imported_repo = Path(pyod.__file__).resolve().parents[1]
    if imported_repo != repo:
        raise ValueError(
            f"editable PyOD source differs: imported {imported_repo}, "
            f"primary used {repo}"
        )

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True,
            stderr=subprocess.DEVNULL, timeout=20,
        ).strip()

    current = {
        "source_repo": str(repo),
        "source_git_commit": git("rev-parse", "HEAD"),
        "source_git_tree": git("rev-parse", "HEAD^{tree}"),
        "source_git_status": git("status", "--porcelain"),
        "source_nsa_sha256": sha256(repo / "pyod" / "models" / "nsa.py"),
    }
    for field, actual in current.items():
        expected = str(Path(manifest[field]).resolve()) if field == "source_repo" \
            else manifest.get(field)
        if actual != expected:
            raise ValueError(
                f"PyOD source {field} differs from the primary run manifest"
            )
    if current["source_git_status"]:
        raise ValueError("PyOD source checkout must be clean for a controlled retry")
    return current


def record_name(dataset: str, representation: str, seed: int,
                fraction: float) -> str:
    suffix = "" if fraction == 1.0 else f"_retain{round(100 * fraction)}"
    return f"{dataset}_{representation}_seed{seed}{suffix}"


def inspect_primary(manifest: dict, primary_root: Path) -> tuple[list[dict], list[str]]:
    expected_methods = set(METHODS)
    if manifest.get("method_timeout_seconds") != 15:
        raise ValueError("primary manifest must declare a 15-second method cap")
    if manifest.get("methods") != METHODS:
        raise ValueError("primary method roster differs from installed factory")
    cells: list[dict] = []
    incomplete: list[str] = []
    for entry in manifest["datasets"]:
        dataset = entry["name"]
        for seed in manifest["seeds"]:
            for representation in manifest["representations"]:
                for fraction in manifest["fractions_retained"]:
                    name = record_name(dataset, representation, seed, fraction)
                    primary = primary_root / "raw" / f"{name}.jsonl"
                    if not primary.exists():
                        incomplete.append(f"{name}: missing primary file")
                        continue
                    try:
                        with primary.open(encoding="utf-8") as stream:
                            rows = [json.loads(line) for line in stream if line.strip()]
                    except (OSError, json.JSONDecodeError) as exc:
                        incomplete.append(f"{name}: unreadable primary file ({exc})")
                        continue
                    methods = [row.get("method") for row in rows]
                    if (len(rows) != len(METHODS)
                            or len(set(methods)) != len(METHODS)
                            or set(methods) != expected_methods):
                        incomplete.append(f"{name}: {len(rows)}/{len(METHODS)} methods")
                        continue
                    if any(row.get("dataset") != dataset
                           or row.get("representation") != representation
                           or row.get("seed") != seed
                           or row.get("retention_fraction") != fraction
                           for row in rows):
                        incomplete.append(f"{name}: cell fields inconsistent")
                        continue
                    timeouts = [name for name in METHODS if any(
                        row["method"] == name and row["status"] == "timeout"
                        for row in rows)]
                    cells.append({
                        "name": name, "dataset": dataset, "seed": seed,
                        "representation": representation, "fraction": fraction,
                        "primary": primary, "primary_sha256": sha256(primary),
                        "methods": timeouts,
                    })
    return cells, incomplete


def run_cell(cell: dict, manifest: dict, retry_root: Path) -> tuple[str, int, int]:
    name = cell["name"]
    raw = retry_root / "raw" / f"{name}.jsonl"
    log = retry_root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(HERE / "timeout_retry_runner.py"),
        "--primary-record", str(cell["primary"]),
        "--data-dir", manifest["data_dir"],
        "--dataset", cell["dataset"],
        "--representation", cell["representation"],
        "--seed", str(cell["seed"]),
        "--train-fraction", str(cell["fraction"]),
        "--train-cap", str(manifest["train_cap"]),
        "--test-cap", str(manifest["test_cap"]),
        "--method-timeout", str(RETRY_SECONDS),
        "--results", str(raw),
    ]
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    cache = retry_root / "cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache / "matplotlib")
    env["XDG_CACHE_HOME"] = str(cache / "xdg")
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(cmd) + "\n")
        stream.flush()
        proc = subprocess.run(cmd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT)
    return name, proc.returncode, len(cell["methods"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--retry-root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--execute", action="store_true",
                        help="Run only after all primary cells are complete")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    primary_root = args.primary_root.resolve()
    retry_root = (args.retry_root or primary_root / "timeout_retry").resolve()
    if retry_root == primary_root or retry_root == primary_root / "raw" \
            or retry_root == primary_root / "logs":
        parser.error("retry output must be separate from primary raw and logs")
    primary_manifest_path = primary_root / "run_manifest.json"
    manifest = json.loads(primary_manifest_path.read_text())
    primary_code_sha256 = verify_primary_code(manifest)
    primary_source = verify_primary_source(manifest)
    cells, incomplete = inspect_primary(manifest, primary_root)
    targets = [cell for cell in cells if cell["methods"]]
    print(f"Primary cells complete: {len(cells)}/{manifest['expected_cells']}")
    print(f"Primary timeout cells: {len(targets)}; method retries: "
          f"{sum(len(cell['methods']) for cell in targets)}")
    if incomplete:
        print("Incomplete examples: " + "; ".join(incomplete[:8]))
    if not args.execute:
        print("Dry run only. Pass --execute after the primary grid finishes.")
        return
    if incomplete or len(cells) != manifest["expected_cells"]:
        raise ValueError("full primary grid must finish before timeout retries")
    retry_root.mkdir(parents=True, exist_ok=True)
    retry_manifest_path = retry_root / "run_manifest.json"
    retry_manifest = {
        "purpose": "descriptive 60-second feasibility check; primary 15-second results remain authoritative",
        "primary_manifest": str(primary_manifest_path),
        "primary_manifest_sha256": sha256(primary_manifest_path),
        "primary_method_timeout_seconds": 15,
        "retry_method_timeout_seconds": RETRY_SECONDS,
        "primary_code_sha256": primary_code_sha256,
        "primary_source": primary_source,
        "retry_runner_sha256": sha256(HERE / "timeout_retry_runner.py"),
        "retry_orchestrator_sha256": sha256(Path(__file__)),
        "primary_timeout_cells": [
            {"name": cell["name"], "primary_sha256": cell["primary_sha256"],
             "methods": cell["methods"]} for cell in targets
        ],
    }
    if retry_manifest_path.exists():
        if json.loads(retry_manifest_path.read_text()) != retry_manifest:
            raise ValueError("retry manifest differs; use another retry directory")
    else:
        retry_manifest_path.write_text(
            json.dumps(retry_manifest, indent=2, sort_keys=True) + "\n"
        )
    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cell, cell, manifest, retry_root)
                   for cell in targets]
        for future in cf.as_completed(futures):
            name, code, count = future.result()
            print(f"{name}: exit={code}, retried={count}", flush=True)
            if code:
                failures.append(name)
    if failures:
        raise SystemExit(f"{len(failures)} retry cells failed: {failures[:8]}")


if __name__ == "__main__":
    main()
