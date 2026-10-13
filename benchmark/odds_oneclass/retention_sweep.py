"""Run the predeclared normal-retention grid with resumable per-cell records."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    log_root = root / "retention" / "sweep_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for fraction in args.fractions:
        if fraction not in (0.25, 0.5, 0.75):
            raise ValueError("fractions must be 0.25, 0.50 or 0.75")
        for representation in ("native", "pca2"):
            for seed in args.seeds:
                name = f"retain{round(fraction * 100)}_{representation}_seed{seed}"
                cmd = [sys.executable, str(root / "scheduler.py"),
                       "--data-dir", str(args.data_dir),
                       "--representation", representation,
                       "--seed", str(seed),
                       "--train-fraction", str(fraction),
                       "--train-cap", "600", "--test-cap", "2000",
                       "--method-timeout", "15", "--workers", str(args.workers),
                       "--runner-script", "selected_runner.py"]
                print(f"START {name}", flush=True)
                with (log_root / f"{name}.log").open("a", encoding="utf-8") as stream:
                    stream.write("COMMAND " + " ".join(cmd) + "\n")
                    stream.flush()
                    result = subprocess.run(cmd, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT)
                print(f"DONE {name} exit={result.returncode}", flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
