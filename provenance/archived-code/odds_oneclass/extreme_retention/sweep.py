"""Fixed 10%/5% normal-only PyOD/NSA extension of the frozen 15 s grid.

The scientific runner and 56-model roster are imported unchanged from the
primary benchmark. Each cell has its own append-only raw JSONL and log; a
second invocation resumes missing methods after the first queue completes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
HERE = ROOT.parent
sys.path.insert(0, str(HERE))

import full_retention_sweep as base  # noqa: E402
from loader import read_manifest  # noqa: E402
from model_factory import METHODS, NSA_METHODS, BASELINE_CLASSES  # noqa: E402


FRACTIONS = (0.10, 0.05)
SEEDS = (42, 43, 44, 45, 46)
REPRESENTATIONS = ("native", "pca2")
TRAIN_CAP = 600
TEST_CAP = 2000
METHOD_TIMEOUT_SECONDS = 15.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_manifest = read_manifest()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(data_manifest["source"]["local_directory"]))
    parser.add_argument("--run-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.run_root = args.run_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.train_cap = TRAIN_CAP
    args.test_cap = TEST_CAP
    args.method_timeout = METHOD_TIMEOUT_SECONDS
    args.fractions = FRACTIONS
    args.seeds = SEEDS
    args.representations = REPRESENTATIONS
    if len(METHODS) != 56 or len(NSA_METHODS) != 13 or len(BASELINE_CLASSES) != 43:
        raise ValueError("The pinned 56-configuration roster changed")

    datasets = sorted(item["name"] for item in data_manifest["datasets"])
    if len(datasets) != 17:
        raise ValueError("Expected the 17 pinned ODDS datasets")
    entries = {item["name"]: item for item in data_manifest["datasets"]}
    for dataset in datasets:
        source = args.data_dir / entries[dataset]["filename"]
        if base.sha256(source) != entries[dataset]["sha256"]:
            raise ValueError(f"Dataset SHA256 mismatch: {source}")
    historical_manifest = HERE / "full_retention" / "run_manifest.json"
    historical = json.loads(historical_manifest.read_text(encoding="utf-8"))
    proposed = base.make_manifest(args, datasets, data_manifest)
    proposed["code_sha256"]["extreme_retention/sweep.py"] = base.sha256(Path(__file__))
    proposed["historical_run_manifest_sha256"] = base.sha256(historical_manifest)
    proposed["historical_run_root"] = str(historical_manifest.parent.resolve())
    proposed["extension_name"] = "normal-only-retention-10-and-5-percent"
    proposed["result_naming"] = "{dataset}_{representation}_seed{seed}_retain{percent}.jsonl"
    if (proposed["expected_cells"] != 340 or proposed["expected_records"] != 19_040
            or proposed["protocol_version"] != "retention-v2-parent-order"
            or proposed["method_timeout_seconds"] != 15.0):
        raise ValueError("Unexpected extreme retention manifest dimensions")
    for key in ("source_git_commit", "source_git_tree", "source_nsa_sha256",
                "dataset_manifest_sha256", "train_cap", "test_cap"):
        if proposed[key] != historical[key]:
            raise ValueError(f"Extension differs from frozen historical run at {key}")
    if proposed["methods"] != historical["methods"]:
        raise ValueError("Extension method roster differs from historical run")
    base.validate_manifest(args.run_root / "run_manifest.json", proposed)

    cells = [(dataset, rep, seed, fraction)
             for fraction in FRACTIONS for rep in REPRESENTATIONS
             for seed in SEEDS for dataset in datasets]
    print(f"EXTREME GRID cells={len(cells)} records={len(cells) * len(METHODS)} "
          f"workers={args.workers} run_root={args.run_root}", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(base.run_cell, cell, args): cell for cell in cells}
        for future in as_completed(futures):
            try:
                name, code, count, state = future.result()
            except Exception as exc:
                errors.append((futures[future], f"{type(exc).__name__}: {exc}"))
                print(f"ERROR {futures[future]}: {exc}", flush=True)
                continue
            print(f"{name}: exit={code} records={count} state={state}", flush=True)
            if code != 0:
                errors.append((name, f"exit={code}"))
    if errors:
        raise SystemExit(f"{len(errors)} cells need resume: {errors[:8]}")
    print(f"COMPLETE {len(cells)} cells and {len(cells) * len(METHODS)} records",
          flush=True)


if __name__ == "__main__":
    main()
