"""Retry only primary 15-second timeouts in an isolated 60-second panel.

This imports the unchanged benchmark runner, narrows its method roster to the
methods whose *same cell* had status ``timeout``, and preserves its normal-only
split, preprocessing, scoring, and JSONL schema. Primary files are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import runner


META_KEYS = (
    "dataset", "representation", "seed", "retention_fraction",
    "protocol_version", "fit_index_sha256", "fit_index_ordered_sha256",
    "fit_parent_index_sha256", "test_index_sha256", "fit_normals",
    "fit_outliers", "test_normals", "test_outliers", "train_cap", "test_cap",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_rows(rows: list[dict], path: Path, expected: dict,
                  allowed_methods: set[str], require_all: bool) -> None:
    seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        method = row.get("method")
        if method not in allowed_methods or method in seen:
            raise ValueError(f"{path}:{number}: unexpected or duplicate method {method}")
        seen.add(method)
        mismatch = [key for key in META_KEYS if row.get(key) != expected.get(key)]
        if mismatch:
            raise ValueError(f"{path}:{number}: split/protocol mismatch {mismatch}")
    if require_all and seen != allowed_methods:
        raise ValueError(
            f"{path}: incomplete method roster; missing {sorted(allowed_methods - seen)}"
        )


def run(args: argparse.Namespace) -> list[str]:
    data_path = args.data_dir / f"{args.dataset}.mat"
    _, _, _, expected = runner.split_data(
        data_path, args.seed, args.train_cap, args.test_cap,
        args.representation, args.train_fraction,
    )
    primary = read_rows(args.primary_record)
    validate_rows(primary, args.primary_record, expected,
                  set(runner.METHODS), require_all=True)
    primary_by_method = {row["method"]: row for row in primary}
    selected = [name for name in runner.METHODS
                if primary_by_method[name]["status"] == "timeout"]
    if args.method:
        if args.method not in selected:
            raise ValueError(f"{args.method}: no primary timeout in {args.primary_record}")
        selected = [args.method]
    if not selected:
        raise ValueError(f"{args.primary_record}: no primary timeouts to retry")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.results.with_suffix(".retry.json")
    metadata = {
        "purpose": "descriptive feasibility retry; primary 15-second outcomes remain authoritative",
        "primary_record": str(args.primary_record.resolve()),
        "primary_record_sha256": sha256(args.primary_record),
        "primary_method_status": "timeout",
        "method_timeout_seconds": args.method_timeout,
        "methods": selected,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256(data_path),
        "split_meta": {key: expected[key] for key in META_KEYS},
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f"{metadata_path}: retry metadata changed; use new result path")
    elif args.results.exists():
        raise ValueError(f"{args.results}: existing raw rows lack retry metadata")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    existing = read_rows(args.results) if args.results.exists() else []
    validate_rows(existing, args.results, expected, set(selected), require_all=False)
    runner.METHODS = selected
    runner.evaluate(
        data_path, args.seed, args.train_cap, args.test_cap,
        args.representation, args.method_timeout, args.results,
        method_filter="all", train_fraction=args.train_fraction,
    )
    final = read_rows(args.results)
    validate_rows(final, args.results, expected, set(selected), require_all=True)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-record", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", required=True, choices=("native", "pca2"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-fraction", required=True, type=float)
    parser.add_argument("--train-cap", type=int, default=600)
    parser.add_argument("--test-cap", type=int, default=2000)
    parser.add_argument("--method-timeout", type=float, default=60.0)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--method", help="Smoke-test one primary timeout method")
    args = parser.parse_args()
    if args.method_timeout <= 15:
        parser.error("retry timeout must exceed the primary 15-second cap")
    selected = run(args)
    print(f"RETRIED {args.dataset} {args.representation} seed={args.seed} "
          f"retained={args.train_fraction}: {', '.join(selected)}", flush=True)


if __name__ == "__main__":
    main()
