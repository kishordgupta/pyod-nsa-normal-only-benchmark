"""Recover one demonstrably stalled extreme-retention method externally.

This is a single-use, opt-in intervention. With no --execute it only reports
preflight evidence. It never launches a replacement runner: after the timeout
row is appended, the ordinary *unfiltered* sweep can resume missing methods.
The process must still hold this exact cell's raw JSONL as fd 3 and its log as
fd 1 and 2. This lsof identity check works even where ps/psutil is unavailable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path(__file__).resolve().parent
MIN_RAW_AGE_SECONDS = 600.0
EXPECTED_TIMEOUT_SECONDS = 15.0
SIGNAL_WAIT_SECONDS = 20.0
CELL_REPRESENTATIONS = {"native", "pca2"}
CELL_FRACTIONS = {0.10, 0.05}

sys.path.insert(0, str(HERE))
from model_factory import METHODS, METHOD_META  # noqa: E402
from runner import PROTOCOL_VERSION, split_data  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def name_for(dataset: str, representation: str, seed: int, fraction: float) -> str:
    suffix = "" if fraction == 1.0 else f"_retain{round(fraction * 100)}"
    return f"{dataset}_{representation}_seed{seed}{suffix}"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def fd_identity(pid: int, raw_path: Path, log_path: Path) -> dict:
    """Require the live PID to hold exact writable cell files on runner FDs."""
    result = subprocess.run(
        ["lsof", "-n", "-a", "-p", str(pid), "-d", "1,2,3"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    if result.returncode != 0:
        raise ValueError(f"Cannot verify live PID {pid} using lsof: {result.stderr.strip()}")
    found = {}
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(maxsplit=8)
        if len(parts) != 9 or parts[1] != str(pid):
            continue
        fd = parts[3]
        if fd in {"1w", "2w", "3w"}:
            found[fd] = {"path": parts[8], "inode": int(parts[7]),
                         "command": parts[0]}
    wanted = {"1w": log_path, "2w": log_path, "3w": raw_path}
    for fd, path in wanted.items():
        observed = found.get(fd)
        if (observed is None or Path(observed["path"]).resolve() != path
                or observed["inode"] != path.stat().st_ino
                or not observed["command"].startswith("python")):
            raise ValueError(f"PID {pid} is not the exact cell runner: {fd}={observed}")
    return found


def assert_no_live_cell_writer(raw_path: Path, log_path: Path) -> None:
    """A resumed/new runner would make appending unsafe."""
    result = subprocess.run(["lsof", "-n", str(raw_path), str(log_path)],
                            capture_output=True, text=True, check=False,
                            timeout=15)
    if result.returncode not in (0, 1):
        raise ValueError(f"Cannot inspect cell writers: {result.stderr.strip()}")
    writers = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(maxsplit=8)
        if len(parts) == 9 and parts[3] in {"1w", "2w", "3w"}:
            writers.append({"pid": parts[1], "fd": parts[3], "path": parts[8]})
    if writers:
        raise ValueError(f"Cell still has a live runner writer: {writers}")


def parse_last_command(log_text: str, manifest: dict, *, raw_path: Path,
                       dataset: str, representation: str, seed: int,
                       fraction: float) -> list[str]:
    commands = [line for line in log_text.splitlines()
                if line.startswith("COMMAND ") or line.startswith("COMMAND: ")]
    if not commands:
        raise ValueError("Cell log has no runner COMMAND line")
    argv = shlex.split(commands[-1].split(" ", 1)[1])
    if len(argv) < 4 or Path(argv[0]).resolve() != Path(manifest["python"]).resolve() \
            or Path(argv[1]).resolve() != HERE / "runner.py":
        raise ValueError("Last logged command is not the pinned runner and Python")
    opts = {}
    for index in range(2, len(argv), 2):
        if index + 1 >= len(argv) or not argv[index].startswith("--") \
                or argv[index] in opts:
            raise ValueError("Malformed or duplicated runner CLI option")
        opts[argv[index]] = argv[index + 1]
    required = {"--data-dir", "--dataset", "--results", "--seed",
                "--representation", "--train-cap", "--test-cap",
                "--method-timeout", "--method-filter"}
    if not required.issubset(opts) or set(opts) - (required | {"--train-fraction"}):
        raise ValueError("Last runner command has unexpected options")
    checks = (
        Path(opts["--data-dir"]).resolve() == Path(manifest["data_dir"]).resolve(),
        opts["--dataset"] == dataset,
        Path(opts["--results"]).resolve() == raw_path,
        int(opts["--seed"]) == seed,
        opts["--representation"] == representation,
        int(opts["--train-cap"]) == manifest["train_cap"],
        int(opts["--test-cap"]) == manifest["test_cap"],
        float(opts["--method-timeout"]) == EXPECTED_TIMEOUT_SECONDS,
        opts["--method-filter"] == "all",
        float(opts.get("--train-fraction", "1.0")) == fraction,
    )
    if not all(checks):
        raise ValueError("Last logged runner command differs from exact cell/protocol")
    return argv


def inspect_cell(args: argparse.Namespace) -> dict:
    run_root = args.run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("protocol_version") != PROTOCOL_VERSION
            or manifest.get("method_timeout_seconds") != EXPECTED_TIMEOUT_SECONDS
            or manifest.get("methods") != METHODS):
        raise ValueError("Primary manifest protocol, timer, or method order differs")
    if args.representation not in CELL_REPRESENTATIONS or args.fraction not in CELL_FRACTIONS \
            or args.seed not in manifest["seeds"]:
        raise ValueError("Cell representation, fraction, or seed is outside the primary grid")
    entry = next((entry for entry in manifest["datasets"]
                  if entry["name"] == args.dataset), None)
    if entry is None or args.fraction not in manifest["fractions_retained"] \
            or args.representation not in manifest["representations"]:
        raise ValueError("Cell is absent from primary manifest")
    for script in ("runner.py", "model_factory.py", "full_retention_sweep.py"):
        if file_digest(HERE / script) != manifest["code_sha256"][script]:
            raise ValueError(f"Pinned benchmark code changed: {script}")
    if (file_digest(DEFAULT_ROOT / "sweep.py")
            != manifest["code_sha256"]["extreme_retention/sweep.py"]):
        raise ValueError("Pinned extreme scheduler changed")
    data_path = Path(manifest["data_dir"]) / entry["filename"]
    if file_digest(data_path) != entry["sha256"]:
        raise ValueError("Dataset bytes differ from primary manifest")
    cell_name = name_for(args.dataset, args.representation, args.seed,
                         args.fraction)
    raw_path = run_root / "raw" / f"{cell_name}.jsonl"
    log_path = run_root / "logs" / f"{cell_name}.log"
    raw_stat = raw_path.stat()
    raw_age = time.time() - raw_stat.st_mtime
    if raw_age < MIN_RAW_AGE_SECONDS:
        raise ValueError(f"Raw file age {raw_age:.1f}s is below 600s")
    raw_bytes, log_bytes = raw_path.read_bytes(), log_path.read_bytes()
    if raw_bytes and not raw_bytes.endswith(b"\n"):
        raise ValueError("Partial raw JSONL lacks a complete final line")
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines()]
    method_position = METHODS.index(args.method)
    preceding_method = METHODS[method_position - 1] if method_position else None
    if (len(rows) != method_position
            or [row.get("method") for row in rows] != METHODS[:method_position]
            or (preceding_method and rows[-1]["method"] != preceding_method)):
        raise ValueError(f"Raw method prefix does not end immediately before {args.method}")
    _, _, _, split_meta = split_data(
        data_path, args.seed, manifest["train_cap"], manifest["test_cap"],
        args.representation, args.fraction)
    for row in rows:
        if any(row.get(key) != value for key, value in split_meta.items()):
            raise ValueError(f"Existing row differs from exact split: {row.get('method')}")
    log_text = log_bytes.decode("utf-8", errors="replace")
    argv = parse_last_command(
        log_text, manifest, raw_path=raw_path, dataset=args.dataset,
        representation=args.representation, seed=args.seed,
        fraction=args.fraction)
    # A runner may have been restarted after a crash. The last completed
    # method can be in an earlier invocation while the current invocation
    # has no outcome yet. Search all invocation segments for the final outcome.
    method_lines = [line for line in log_text.splitlines()
                    if re.match(rf"^{re.escape(args.dataset)}\s+{args.representation}\s+", line)]
    if preceding_method:
        last = rows[-1]
        if (not method_lines
                or not re.search(rf"\s{re.escape(preceding_method)}\s+"
                                 rf"{re.escape(last['status'])}\s+",
                                 method_lines[-1])):
            raise ValueError("Log does not end at the last raw method/status")
    elif method_lines:
        raise ValueError("Log has method outcomes but raw has none")
    command_positions = [match.start() for match in
                         re.finditer(r"(?m)^COMMAND(?::)? ", log_text)]
    current_invocation_tail = log_text[command_positions[-1]:]
    if preceding_method:
        last_outcome_position = current_invocation_tail.rfind(method_lines[-1])
        evidence_tail = (current_invocation_tail[last_outcome_position
                                                  + len(method_lines[-1]):]
                         if last_outcome_position >= 0 else current_invocation_tail)
    else:
        evidence_tail = current_invocation_tail
    evidence = (
        ("Exception ignored on calling ctypes callback function" in evidence_tail
         or "Exception ignored in:" in evidence_tail)
        and ("llvmlite/binding/executionengine.py" in evidence_tail
             or "numba/core/typing/context.py" in evidence_tail)
        and "runner.py" in evidence_tail
        and "FitTimeout: per-method wall-time budget exceeded" in evidence_tail
    )
    if evidence:
        evidence_kind = "ignored_signal_callback"
    else:
        if "FitTimeout: per-method wall-time budget exceeded" in evidence_tail:
            raise ValueError("Ambiguous partial FitTimeout traceback in log tail")
        evidence_kind = "stalled_without_callback"
    fd = fd_identity(args.pid, raw_path, log_path)
    return {
        "run_root": run_root, "manifest_path": manifest_path,
        "manifest": manifest, "cell_name": cell_name,
        "raw_path": raw_path, "log_path": log_path,
        "raw_bytes": raw_bytes, "log_bytes": log_bytes,
        "raw_sha256": digest(raw_bytes), "log_sha256": digest(log_bytes),
        "raw_inode": raw_stat.st_ino, "raw_mtime_ns": raw_stat.st_mtime_ns,
        "raw_age_seconds": round(raw_age, 3), "rows": rows,
        "split_meta": split_meta, "runner_argv": argv,
        "fd_identity": fd,
        "evidence_kind": evidence_kind,
        "signal_evidence_excerpt": evidence_tail[-1200:],
    }


def wait_until_fd_closed(pid: int, raw_path: Path, log_path: Path,
                         seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            fd_identity(pid, raw_path, log_path)
        except ValueError:
            assert_no_live_cell_writer(raw_path, log_path)
            return True
        time.sleep(0.25)
    return False


def execute(args: argparse.Namespace, pre: dict) -> dict:
    now = datetime.now(timezone.utc)
    intervention_id = (now.strftime("%Y%m%dT%H%M%S%fZ") + "_"
                       + pre["cell_name"] + f"_pid{args.pid}")
    evidence_dir = pre["run_root"] / "watchdog" / "interventions" / intervention_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    (evidence_dir / "raw.before.jsonl").write_bytes(pre["raw_bytes"])
    (evidence_dir / "log.before.log").write_bytes(pre["log_bytes"])
    if (file_digest(evidence_dir / "raw.before.jsonl") != pre["raw_sha256"]
            or file_digest(evidence_dir / "log.before.log") != pre["log_sha256"]):
        raise ValueError("Evidence snapshots failed SHA256 verification")
    if pre["evidence_kind"] == "ignored_signal_callback":
        reason_text = (f"runner remained at {args.method} after an ignored "
                       "15 s SIGALRM and at least 600 s without a raw row")
    else:
        reason_text = (f"runner remained at {args.method} for at least 600 s "
                       "without a raw row under the nominal 15 s method budget; "
                       "no ignored-signal callback was observed")
    row = {
        **pre["split_meta"], "method": args.method,
        "family": METHOD_META[args.method]["family"],
        "status": "timeout", "fit_seconds": None, "score_seconds": None,
        "auroc": None, "ap": None,
        "error": f"ExternalWatchdogTimeout: {reason_text}",
        "external_watchdog": {
            "intervention_id": intervention_id,
            "audit_path": str(evidence_dir.relative_to(pre["run_root"]) / "intervention.json"),
            "target_pid": args.pid,
            "minimum_raw_age_seconds": MIN_RAW_AGE_SECONDS,
            "reason": pre["evidence_kind"],
        },
    }
    row_bytes = (json.dumps(row, allow_nan=False, sort_keys=True) + "\n").encode()
    audit = {
        "schema_version": 1, "intervention_id": intervention_id,
        "state": "prepared", "created_utc": now.isoformat(),
        "cell": {"dataset": args.dataset, "representation": args.representation,
                 "seed": args.seed, "retention_fraction": args.fraction,
                 "name": pre["cell_name"], "method": args.method},
        "target": {"pid": args.pid, "fd_identity": pre["fd_identity"],
                   "runner_argv": pre["runner_argv"]},
        "before": {"raw_sha256": pre["raw_sha256"],
                   "log_sha256": pre["log_sha256"],
                   "raw_age_seconds": pre["raw_age_seconds"],
                   "raw_record_count": len(pre["rows"]),
                   "raw_snapshot": "raw.before.jsonl",
                   "log_snapshot": "log.before.log",
                   "evidence_kind": pre["evidence_kind"],
                   "signal_evidence_excerpt": pre["signal_evidence_excerpt"]},
        "action": {"signal": "SIGTERM", "termination_confirmed": False,
                   "primary_method_timeout_seconds": EXPECTED_TIMEOUT_SECONDS},
        "appended_row_sha256": digest(row_bytes),
        "resume_state": "pending_unfiltered_sweep",
    }
    audit_path = evidence_dir / "intervention.json"
    atomic_json(audit_path, audit)
    try:
        # Re-read all mutable evidence immediately before signaling. A normal
        # completion or another intervention must stop this run without a row.
        again = inspect_cell(args)
        if (again["raw_sha256"] != pre["raw_sha256"]
                or again["log_sha256"] != pre["log_sha256"]
                or again["raw_inode"] != pre["raw_inode"]
                or again["fd_identity"] != pre["fd_identity"]):
            raise ValueError("Cell changed between snapshot and signal")
        os.kill(args.pid, signal.SIGTERM)
        audit["action"]["signal_sent_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(audit_path, audit)
        if not wait_until_fd_closed(args.pid, pre["raw_path"], pre["log_path"],
                                    SIGNAL_WAIT_SECONDS):
            raise RuntimeError("SIGTERM did not close runner FDs within 20 s; no row appended")
        audit["action"]["termination_confirmed"] = True
        audit["action"]["runner_fds_closed_utc"] = datetime.now(timezone.utc).isoformat()
        if pre["raw_path"].stat().st_ino != pre["raw_inode"] \
                or file_digest(pre["raw_path"]) != pre["raw_sha256"]:
            raise ValueError("Raw changed during termination; no watchdog row appended")
        assert_no_live_cell_writer(pre["raw_path"], pre["log_path"])
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(pre["raw_path"], flags)
        try:
            if os.fstat(fd).st_ino != pre["raw_inode"]:
                raise ValueError("Raw inode changed before append")
            if os.write(fd, row_bytes) != len(row_bytes):
                raise OSError("Partial watchdog row write")
            os.fsync(fd)
        finally:
            os.close(fd)
        final_bytes = pre["raw_path"].read_bytes()
        if final_bytes != pre["raw_bytes"] + row_bytes:
            raise ValueError("Final raw differs from snapshot plus one timeout row")
        audit["after"] = {
            "raw_sha256": digest(final_bytes),
            "log_sha256": file_digest(pre["log_path"]),
            "raw_record_count": len(pre["rows"]) + 1,
            "appended_method": args.method, "appended_status": "timeout",
        }
        audit["state"] = "completed"
        audit["completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(audit_path, audit)
        return audit
    except BaseException as exc:
        audit["state"] = "interrupted"
        audit["interruption"] = f"{type(exc).__name__}: {exc}"
        atomic_json(audit_path, audit)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", required=True,
                        choices=sorted(CELL_REPRESENTATIONS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, required=True,
                        choices=sorted(CELL_FRACTIONS))
    parser.add_argument("--pid", type=int, required=True,
                        help="PID independently identified from lsof on this exact cell log")
    parser.add_argument("--method", choices=METHODS, required=True,
                        help="Next method to mark timed out after its exact roster prefix stalls")
    parser.add_argument("--execute", action="store_true",
                        help="After all checks, snapshot evidence, terminate child, append timeout")
    args = parser.parse_args()
    if args.pid <= 1 or args.pid == os.getpid():
        parser.error("Provide a live, non-self runner child PID")
    pre = inspect_cell(args)
    if not args.execute:
        print(json.dumps({
            "ready": True, "execute": False, "cell": pre["cell_name"],
            "pid": args.pid, "raw_age_seconds": pre["raw_age_seconds"],
            "existing_rows": len(pre["rows"]), "next_method": args.method,
            "evidence_kind": pre["evidence_kind"],
            "raw_sha256": pre["raw_sha256"],
            "log_sha256": pre["log_sha256"],
            "fd_identity": pre["fd_identity"],
        }, indent=2, sort_keys=True))
        return
    audit = execute(args, pre)
    print(json.dumps({"intervention_id": audit["intervention_id"],
                      "state": audit["state"], "cell": audit["cell"],
                      "resume_state": audit["resume_state"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
