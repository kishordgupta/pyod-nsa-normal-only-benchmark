"""Watch the 10%/5% sweep for a runner with no completed method for 600 s.

The monitor acts only when --execute is supplied. It delegates every mutation
to watchdog_recover.py, which rechecks the exact child PID, open file
descriptors, split metadata, log command, and unchanged raw bytes. A killed
cell can be resumed by rerunning the fixed 340-cell sweep after its queue ends.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
HERE = ROOT.parent
sys.path.insert(0, str(HERE))
from model_factory import METHODS  # noqa: E402


STEM = re.compile(r"^(.+)_(native|pca2)_seed(4[2-6])_retain(10|5)$")
MIN_IDLE_SECONDS = 600.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def child_pid_for_log(path: Path) -> int | None:
    result = subprocess.run(["lsof", "-n", str(path)], capture_output=True,
                            text=True, timeout=15, check=False)
    if result.returncode == 1:
        return None
    if result.returncode != 0:
        raise RuntimeError(f"lsof failed for {path}: {result.stderr.strip()}")
    fds: dict[int, set[str]] = {}
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(maxsplit=8)
        if len(parts) == 9 and parts[1].isdigit() and parts[8] == str(path):
            fds.setdefault(int(parts[1]), set()).add(parts[3])
    children = [pid for pid, held in fds.items() if {"1w", "2w"}.issubset(held)]
    if len(children) > 1:
        raise RuntimeError(f"More than one child has log stdout/stderr: {path}: {children}")
    return children[0] if children else None


def inspect_candidates(root: Path) -> list[dict]:
    now = time.time()
    candidates = []
    for raw in sorted((root / "raw").glob("*.jsonl")):
        match = STEM.fullmatch(raw.stem)
        if match is None:
            raise ValueError(f"Unexpected extreme raw filename: {raw}")
        dataset, rep, seed_text, pct_text = match.groups()
        log = root / "logs" / f"{raw.stem}.log"
        if not log.is_file():
            continue
        raw_age = now - raw.stat().st_mtime
        log_age = now - log.stat().st_mtime
        if raw_age < MIN_IDLE_SECONDS or log_age < MIN_IDLE_SECONDS:
            continue
        raw_bytes = raw.read_bytes()
        if raw_bytes and not raw_bytes.endswith(b"\n"):
            raise ValueError(f"Incomplete JSONL line in stalled cell: {raw}")
        rows = [json.loads(line) for line in raw_bytes.splitlines()]
        if [row.get("method") for row in rows] != METHODS[:len(rows)]:
            raise ValueError(f"Stalled cell has non-prefix method rows: {raw}")
        if len(rows) >= len(METHODS):
            continue
        pid = child_pid_for_log(log)
        if pid is None:
            continue
        candidates.append({
            "dataset": dataset, "representation": rep, "seed": int(seed_text),
            "fraction": int(pct_text) / 100, "method": METHODS[len(rows)],
            "pid": pid, "raw_age_seconds": round(raw_age, 1),
            "log_age_seconds": round(log_age, 1),
            "completed_rows": len(rows), "raw_path": str(raw),
        })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Run the externally audited watchdog on eligible live children")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("poll-seconds must be at least 5")
    root = args.run_root.resolve()
    if not (root / "run_manifest.json").is_file():
        raise FileNotFoundError(root / "run_manifest.json")
    print(f"{utc_now()} MONITOR root={root} execute={args.execute} "
          f"minimum_raw_and_log_idle_seconds={MIN_IDLE_SECONDS}", flush=True)
    seen: set[tuple] = set()
    while True:
        for candidate in inspect_candidates(root):
            key = (candidate["raw_path"], candidate["pid"], candidate["method"])
            if key in seen:
                continue
            seen.add(key)
            print(f"{utc_now()} ELIGIBLE {json.dumps(candidate, sort_keys=True)}",
                  flush=True)
            if not args.execute:
                continue
            command = [
                sys.executable, str(ROOT / "watchdog_recover.py"),
                "--run-root", str(root),
                "--dataset", candidate["dataset"],
                "--representation", candidate["representation"],
                "--seed", str(candidate["seed"]),
                "--fraction", str(candidate["fraction"]),
                "--method", candidate["method"],
                "--pid", str(candidate["pid"]), "--execute",
            ]
            result = subprocess.run(command, text=True, capture_output=True,
                                    check=False, timeout=90)
            print(f"{utc_now()} WATCHDOG exit={result.returncode} "
                  f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}",
                  flush=True)
            if result.returncode != 0:
                raise RuntimeError("Watchdog intervention failed; monitor stopped for review")
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
