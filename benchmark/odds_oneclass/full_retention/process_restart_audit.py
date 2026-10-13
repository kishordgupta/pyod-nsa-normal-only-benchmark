"""Verify observed runner process restarts without inferring their causes.

The glass and pendigits PCA(2), seed-43, 25%-retention logs each contain two
runner command lines. Their first invocations stopped after 36 and 30 complete
JSONL records, respectively, and their second invocations resumed the same
append-only files. An operating-system exit code was not preserved in these
artifacts, so this audit makes no exit-code or method-culpability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_factory import METHODS


CELL = "glass_pca2_seed43_retain25"
PENDIGITS_CELL = "pendigits_pca2_seed43_retain25"
EXPECTED_RECORDS = len(METHODS)
RESTART_AFTER_RECORD = METHODS.index("ROD")
PENDIGITS_RESTART_AFTER_RECORD = METHODS.index("CBLOF")
RESTART_CELLS = (CELL, PENDIGITS_CELL)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(run_root: Path) -> dict:
    raw_path = run_root / "raw" / f"{CELL}.jsonl"
    log_path = run_root / "logs" / f"{CELL}.log"
    raw_bytes, log_bytes = raw_path.read_bytes(), log_path.read_bytes()
    if not raw_bytes.endswith(b"\n"):
        raise ValueError("glass raw JSONL has an unterminated record")
    lines = raw_bytes.splitlines(keepends=True)
    if len(lines) != EXPECTED_RECORDS:
        raise ValueError(f"glass cell has {len(lines)}/{EXPECTED_RECORDS} records")
    rows = [json.loads(line) for line in lines]
    if [row.get("method") for row in rows] != METHODS:
        raise ValueError("glass method order is incomplete or duplicated")
    if (RESTART_AFTER_RECORD != 36
            or rows[RESTART_AFTER_RECORD - 1]["method"] != "SOD"
            or rows[RESTART_AFTER_RECORD - 1]["status"] != "timeout"
            or rows[RESTART_AFTER_RECORD]["method"] != "ROD"):
        raise ValueError("glass restart boundary is not after SOD")
    reference = rows[0]
    fields = ("dataset", "seed", "representation", "retention_fraction",
              "protocol_version", "fit_index_sha256",
              "fit_index_ordered_sha256", "fit_parent_index_sha256",
              "test_index_sha256", "fit_normals", "test_normals",
              "test_outliers", "train_cap", "test_cap")
    for row in rows:
        if row.get("fit_outliers") != 0 or any(
                row.get(field) != reference.get(field) for field in fields):
            raise ValueError("glass resumed rows use a different split/protocol")
    if (reference["dataset"] != "glass" or reference["seed"] != 43
            or reference["representation"] != "pca2"
            or reference["retention_fraction"] != 0.25):
        raise ValueError("glass cell identity differs from requested restart audit")

    command_starts = [match.start() for match in re.finditer(br"(?m)^COMMAND ", log_bytes)]
    if len(command_starts) != 2 or command_starts[0] != 0:
        raise ValueError("glass log must have exactly two runner invocations")
    first_log = log_bytes[:command_starts[1]]
    second_log = log_bytes[command_starts[1]:]
    first_command = first_log.splitlines()[0].decode("utf-8", errors="replace")
    second_command = second_log.splitlines()[0].decode("utf-8", errors="replace")
    if first_command != second_command:
        raise ValueError("glass resume command differs from initial command")
    if (b"FitTimeout: per-method wall-time budget exceeded" not in first_log
            or b"Exception ignored in:" not in first_log
            or b"AttributeError: 'PipelineTuningOptions' object has no attribute '_as_parameter_'"
                not in first_log
            or b"glass        pca2   SOD                      timeout" not in first_log):
        raise ValueError("glass first log segment lacks the observed native-callback evidence")
    first_records = _logged_records(first_log, "glass", "pca2")
    second_records = _logged_records(second_log, "glass", "pca2")
    if (first_records != [(row["method"], row["status"]) for row in rows[:RESTART_AFTER_RECORD]]
            or second_records != [(row["method"], row["status"]) for row in rows[RESTART_AFTER_RECORD:]]):
        raise ValueError("glass log segments disagree with the saved restart boundary")
    first_raw = b"".join(lines[:RESTART_AFTER_RECORD])
    return {
        "schema_version": 1,
        "purpose": "process restart provenance only; primary method outcomes remain authoritative",
        "cell": CELL,
        "raw_file": f"raw/{CELL}.jsonl",
        "log_file": f"logs/{CELL}.log",
        "observed_invocations": 2,
        "commands_identical": True,
        "command": first_command,
        "first_invocation": {
            "completed_record_count": RESTART_AFTER_RECORD,
            "last_record_method": "SOD",
            "last_record_status": "timeout",
            "raw_prefix_sha256": sha(first_raw),
            "raw_prefix_bytes": len(first_raw),
            "log_segment_sha256": sha(first_log),
            "log_segment_bytes": len(first_log),
            "ignored_fit_timeout_observed": True,
            "llvmlite_cleanup_attribute_error_observed": True,
        },
        "resumed_invocation": {
            "first_new_record_method": rows[RESTART_AFTER_RECORD]["method"],
            "first_new_record_status": rows[RESTART_AFTER_RECORD]["status"],
            "log_segment_sha256": sha(second_log),
            "log_segment_bytes": len(second_log),
        },
        "final": {
            "record_count": len(rows),
            "unique_method_count": len({row["method"] for row in rows}),
            "all_fit_outliers_zero": True,
            "raw_sha256": sha(raw_bytes),
            "log_sha256": sha(log_bytes),
            "fit_index_sha256": reference["fit_index_sha256"],
            "fit_index_ordered_sha256": reference["fit_index_ordered_sha256"],
            "test_index_sha256": reference["test_index_sha256"],
        },
        "limits": [
            "The saved raw and log artifacts do not contain the first process exit code.",
            "The log does not establish which method caused the first process to end.",
            "No score or failure outcome was imputed from the restart itself.",
        ],
    }


def _logged_records(segment: bytes, dataset: str, representation: str) -> list[tuple[str, str]]:
    """Extract only the runner's per-method status lines from one log segment."""
    records = []
    for line in segment.decode("utf-8", errors="replace").splitlines():
        fields = line.split()
        if (len(fields) >= 4 and fields[0] == dataset
                and fields[1] == representation and fields[2] in METHODS):
            records.append((fields[2], fields[3]))
    return records


def verify_pendigits(run_root: Path) -> dict:
    """Audit the observed 30-record pendigits restart and preserved append."""
    raw_path = run_root / "raw" / f"{PENDIGITS_CELL}.jsonl"
    log_path = run_root / "logs" / f"{PENDIGITS_CELL}.log"
    raw_bytes, log_bytes = raw_path.read_bytes(), log_path.read_bytes()
    if not raw_bytes.endswith(b"\n"):
        raise ValueError("pendigits raw JSONL has an unterminated record")
    lines = raw_bytes.splitlines(keepends=True)
    if len(lines) != EXPECTED_RECORDS:
        raise ValueError(f"pendigits cell has {len(lines)}/{EXPECTED_RECORDS} records")
    rows = [json.loads(line) for line in lines]
    if [row.get("method") for row in rows] != METHODS:
        raise ValueError("pendigits method order is incomplete or duplicated")
    if (PENDIGITS_RESTART_AFTER_RECORD != 30
            or rows[29]["method"] != "COF"
            or rows[29]["status"] != "success"
            or rows[30]["method"] != "CBLOF"):
        raise ValueError("pendigits restart boundary is not after COF")
    reference = rows[0]
    fields = ("dataset", "seed", "representation", "retention_fraction",
              "protocol_version", "fit_index_sha256",
              "fit_index_ordered_sha256", "fit_parent_index_sha256",
              "test_index_sha256", "fit_normals", "test_normals",
              "test_outliers", "train_cap", "test_cap")
    for row in rows:
        if row.get("fit_outliers") != 0 or any(
                row.get(field) != reference.get(field) for field in fields):
            raise ValueError("pendigits resumed rows use a different split/protocol")
    if (reference["dataset"] != "pendigits" or reference["seed"] != 43
            or reference["representation"] != "pca2"
            or reference["retention_fraction"] != 0.25):
        raise ValueError("pendigits cell identity differs from requested restart audit")

    command_starts = [match.start() for match in re.finditer(br"(?m)^COMMAND ", log_bytes)]
    if len(command_starts) != 2 or command_starts[0] != 0:
        raise ValueError("pendigits log must have exactly two runner invocations")
    first_log = log_bytes[:command_starts[1]]
    second_log = log_bytes[command_starts[1]:]
    first_command = first_log.splitlines()[0].decode("utf-8", errors="replace")
    second_command = second_log.splitlines()[0].decode("utf-8", errors="replace")
    if first_command != second_command:
        raise ValueError("pendigits resume command differs from initial command")
    first_records = _logged_records(first_log, "pendigits", "pca2")
    second_records = _logged_records(second_log, "pendigits", "pca2")
    if (first_records != [(row["method"], row["status"]) for row in rows[:30]]
            or second_records != [(row["method"], row["status"]) for row in rows[30:]]):
        raise ValueError("pendigits log segments disagree with the saved restart boundary")
    first_raw = b"".join(lines[:PENDIGITS_RESTART_AFTER_RECORD])
    return {
        "schema_version": 1,
        "purpose": "process restart provenance only; primary method outcomes remain authoritative",
        "cell": PENDIGITS_CELL,
        "raw_file": f"raw/{PENDIGITS_CELL}.jsonl",
        "log_file": f"logs/{PENDIGITS_CELL}.log",
        "observed_invocations": 2,
        "commands_identical": True,
        "command": first_command,
        "first_invocation": {
            "completed_record_count": PENDIGITS_RESTART_AFTER_RECORD,
            "last_record_method": "COF",
            "last_record_status": "success",
            "raw_prefix_sha256": sha(first_raw),
            "raw_prefix_bytes": len(first_raw),
            "log_segment_sha256": sha(first_log),
            "log_segment_bytes": len(first_log),
        },
        "resumed_invocation": {
            "first_new_record_method": rows[30]["method"],
            "first_new_record_status": rows[30]["status"],
            "log_segment_sha256": sha(second_log),
            "log_segment_bytes": len(second_log),
        },
        "final": {
            "record_count": len(rows),
            "unique_method_count": len({row["method"] for row in rows}),
            "all_fit_outliers_zero": True,
            "raw_sha256": sha(raw_bytes),
            "log_sha256": sha(log_bytes),
            "fit_index_sha256": reference["fit_index_sha256"],
            "fit_index_ordered_sha256": reference["fit_index_ordered_sha256"],
            "test_index_sha256": reference["test_index_sha256"],
        },
        "limits": [
            "The saved raw and log artifacts do not contain the first process exit code.",
            "The log does not establish which method caused the first process to end.",
            "No score or failure outcome was imputed from the restart itself.",
        ],
    }


def verify_all(run_root: Path) -> dict[str, dict]:
    return {CELL: verify(run_root), PENDIGITS_CELL: verify_pendigits(run_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell", choices=RESTART_CELLS, default=CELL)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = (args.output or run_root / "provenance" / f"{args.cell}_restart.json").resolve()
    data = verify(run_root) if args.cell == CELL else verify_pendigits(run_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise ValueError(f"{output} exists with different restart evidence")
    output.write_text(content)
    print(f"Validated {data['cell']}: {data['final']['record_count']} records and two invocations: {output}")


if __name__ == "__main__":
    main()
