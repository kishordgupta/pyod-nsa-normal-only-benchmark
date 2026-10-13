"""Run the one-class PyOD benchmark dataset by dataset with bounded parallelism."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
from pathlib import Path


def run_one(dataset: str, args):
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    reduced = args.train_fraction != 1.0
    result_root = Path(args.result_root) if args.result_root else root / ("retention/raw" if reduced else "raw")
    log_root = Path(args.log_root) if args.log_root else root / ("retention/logs" if reduced else "logs")
    suffix = f"_retain{round(100 * args.train_fraction)}" if reduced else ""
    result = result_root / f"{dataset}_{args.representation}_seed{args.seed}{suffix}.jsonl"
    log = log_root / f"{dataset}_{args.representation}_seed{args.seed}{suffix}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(root / args.runner_script),
           "--data-dir", str(data_dir), "--dataset", dataset,
           "--results", str(result), "--seed", str(args.seed),
           "--representation", args.representation,
           "--train-cap", str(args.train_cap),
           "--test-cap", str(args.test_cap),
           "--method-timeout", str(args.method_timeout),
           "--method-filter", args.method_filter]
    if reduced:
        cmd.extend(["--train-fraction", str(args.train_fraction)])
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    cache_dir = root / "cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "xdg").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    env["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    with log.open("a") as stream:
        stream.write("COMMAND: " + " ".join(cmd) + "\n")
        stream.flush()
        process = subprocess.run(cmd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT)
    return dataset, process.returncode, result, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--representation", choices=["native", "pca2", "pca1"],
                        default="pca2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cap", type=int, default=600)
    parser.add_argument("--test-cap", type=int, default=2000)
    parser.add_argument("--train-fraction", type=float,
                        choices=[0.25, 0.5, 0.75, 1.0], default=1.0,
                        help="fraction of the original capped normal fit pool to retain")
    parser.add_argument("--method-timeout", type=float, default=20)
    parser.add_argument("--method-filter", choices=["all", "nsa", "baseline"],
                        default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--runner-script", default="runner.py")
    parser.add_argument("--result-root")
    parser.add_argument("--log-root")
    parser.add_argument("datasets", nargs="*")
    args = parser.parse_args()
    if not args.datasets:
        args.datasets = sorted(p.stem for p in Path(args.data_dir).glob("*.mat"))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, dataset, args): dataset
                   for dataset in args.datasets}
        for future in cf.as_completed(futures):
            dataset, code, result, log = future.result()
            print(f"{dataset}: exit={code}, results={result}, log={log}", flush=True)
    if any(future.result()[1] != 0 for future in futures):
        sys.exit(1)


if __name__ == "__main__":
    main()
