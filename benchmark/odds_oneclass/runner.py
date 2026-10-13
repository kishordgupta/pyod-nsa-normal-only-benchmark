"""Normal-only fit benchmark on the ODDS files bundled with PyOD.

Each dataset is split before preprocessing. All anomalous rows in the training
partition are discarded. Every estimator receives X only at fit time. The
separate test partition keeps labels for AUROC/AP evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_factory import METHODS, METHOD_META, ModelBuildError, build_model


class FitTimeout(TimeoutError):
    pass


PROTOCOL_VERSION = "retention-v2-parent-order"


def _on_alarm(_signum, _frame):
    raise FitTimeout("per-method wall-time budget exceeded")


def _index_sha256(indices: np.ndarray) -> str:
    """Hash a row-index *set* in a platform-independent binary encoding."""
    ordered = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _ordered_index_sha256(indices: np.ndarray) -> str:
    """Hash row indices in the order supplied to model.fit."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def split_data(path: Path, seed: int, train_cap: int, test_cap: int,
               representation: str, train_fraction: float = 1.0):
    if not (0 < train_fraction <= 1):
        raise ValueError("train_fraction must be in (0, 1]")
    z = loadmat(path)
    X = np.asarray(z["X"], dtype=np.float64)
    y = np.asarray(z["y"]).ravel().astype(np.int8)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(X).all():
        raise ValueError("expected finite X and binary ODDS y")

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed
    )
    fit_idx = train_idx[y[train_idx] == 0]
    discarded_outliers = int(np.sum(y[train_idx] == 1))
    if len(fit_idx) > train_cap:
        rng = np.random.RandomState(seed + 10_000)
        fit_idx = np.sort(rng.choice(fit_idx, size=train_cap, replace=False))
    if len(test_idx) > test_cap:
        _, chosen = train_test_split(
            np.arange(len(test_idx)), test_size=test_cap,
            stratify=y[test_idx], random_state=seed + 20_000,
        )
        test_idx = np.sort(test_idx[chosen])

    # Draw one label-free permutation of the original capped normal pool. Its
    # prefixes give nested 25%, 50%, and 75% training sets. Leave the 100%
    # ordering intact so the original benchmark's fit and scores reproduce.
    parent_fit_idx = fit_idx.copy()
    if train_fraction < 1:
        n_keep = max(2, int(np.floor(len(parent_fit_idx) * train_fraction)))
        permuted = np.random.RandomState(seed + 30_000).permutation(parent_fit_idx)
        # Select a nested prefix *set*, then retain the parent's row order.
        # This matters for uncapped datasets, where the original 100% split
        # keeps train_test_split's order rather than sorting fit indices.
        fit_idx = parent_fit_idx[np.isin(parent_fit_idx, permuted[:n_keep])]

    X_fit, X_test, y_test = X[fit_idx], X[test_idx], y[test_idx]
    scaler = StandardScaler().fit(X_fit)
    X_fit = scaler.transform(X_fit)
    X_test = scaler.transform(X_test)
    if representation == "pca2":
        projector = PCA(n_components=2, svd_solver="full").fit(X_fit)
        X_fit = projector.transform(X_fit)
        X_test = projector.transform(X_test)
        retained_variance = float(np.sum(projector.explained_variance_ratio_))
    elif representation == "pca1":
        projector = PCA(n_components=1, svd_solver="full").fit(X_fit)
        X_fit = projector.transform(X_fit)
        X_test = projector.transform(X_test)
        retained_variance = float(np.sum(projector.explained_variance_ratio_))
    elif representation == "native":
        retained_variance = 1.0
    else:
        raise ValueError(representation)
    if np.any(y[fit_idx] != 0):
        raise AssertionError("outlier leaked into training")
    assert len(np.unique(y_test)) == 2
    meta = {
        "dataset": path.stem,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_rows": int(len(y)),
        "source_features": int(X.shape[1]),
        "fit_normals": int(len(fit_idx)),
        "original_fit_normals": int(len(parent_fit_idx)),
        "fit_outliers": 0,
        "retention_fraction": float(train_fraction),
        "reduction_fraction": float(1 - train_fraction),
        "fit_index_sha256": _index_sha256(fit_idx),
        "fit_index_ordered_sha256": _ordered_index_sha256(fit_idx),
        "fit_parent_index_sha256": _index_sha256(parent_fit_idx),
        "test_index_sha256": _index_sha256(test_idx),
        "discarded_train_outliers": discarded_outliers,
        "test_normals": int(np.sum(y_test == 0)),
        "test_outliers": int(np.sum(y_test == 1)),
        "representation": representation,
        "retained_variance": retained_variance,
        "seed": seed,
        "train_cap": train_cap,
        "test_cap": test_cap,
    }
    return X_fit, X_test, y_test, meta


def evaluate(path: Path, seed: int, train_cap: int, test_cap: int,
             representation: str, method_timeout: float, results_path: Path,
             method_filter: str = "all", train_fraction: float = 1.0):
    X_fit, X_test, y_test, split_meta = split_data(
        path, seed, train_cap, test_cap, representation, train_fraction
    )
    existing = set()
    if results_path.exists():
        with results_path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (train_fraction < 1
                        and float(row.get("retention_fraction", -1)) == train_fraction
                        and (row.get("protocol_version") != PROTOCOL_VERSION
                             or not row.get("fit_index_ordered_sha256"))):
                    raise ValueError(
                        f"stale retention record in {results_path}; "
                        "move the preliminary file before rerunning")
                try:
                    existing.add((row["dataset"], row["seed"],
                                  row["representation"], row["method"],
                                  float(row.get("retention_fraction", 1.0))))
                except (ValueError, KeyError):
                    continue
    results_path.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _on_alarm)
    with results_path.open("a", buffering=1) as out:
        for method in METHODS:
            if method_filter == "nsa" and not method.startswith("NSA/"):
                continue
            if method_filter == "baseline" and method.startswith("NSA/"):
                continue
            key = (path.stem, seed, representation, method, float(train_fraction))
            if key in existing:
                continue
            method_meta = METHOD_META[method]
            row = {**split_meta, "method": method,
                   "family": method_meta["family"],
                   "status": "pending", "fit_seconds": None,
                   "score_seconds": None, "auroc": None, "ap": None,
                   "error": None}
            applicability = method_meta.get("applicability", "applicable")
            if applicability != "applicable":
                row["status"] = applicability
                row["error"] = method_meta.get("reason")
            elif method == "MAD" and X_fit.shape[1] != 1:
                row["status"] = "requires_1d"
                row["error"] = "MAD accepts only one input feature"
            elif method == "NSA/voronoi" and X_fit.shape[1] not in (2, 3):
                row["status"] = "requires_2d_or_3d"
                row["error"] = "Voronoi NSA accepts 2D or 3D features"
            elif method == "NSA/deterministic" and X_fit.shape[1] != 2:
                row["status"] = "requires_2d"
                row["error"] = "Deterministic NSA accepts 2D features"
            else:
                t0 = time.perf_counter()
                try:
                    random.seed(seed)
                    np.random.seed(seed)
                    if "torch" in method_meta.get("requires", []):
                        import torch
                        torch.manual_seed(seed)
                    signal.setitimer(signal.ITIMER_REAL, method_timeout)
                    model = build_model(method, seed, len(X_fit), X_fit.shape[1])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(X_fit)
                        t1 = time.perf_counter()
                        raw_scores = np.asarray(
                            model.decision_function(X_test), dtype=np.float64
                        ).ravel()
                        t2 = time.perf_counter()
                    if raw_scores.shape != y_test.shape:
                        raise ValueError("score shape mismatch")
                    if not np.isfinite(raw_scores).all():
                        raise ValueError("nonfinite test score")
                    row["fit_seconds"] = round(t1 - t0, 6)
                    row["score_seconds"] = round(t2 - t1, 6)
                    row["auroc"] = float(roc_auc_score(y_test, raw_scores))
                    row["ap"] = float(average_precision_score(y_test, raw_scores))
                    row["score_unique"] = int(len(np.unique(raw_scores)))
                    row["warning_count"] = len(caught)
                    row["warnings"] = [str(w.message)[:300] for w in caught[:3]]
                    row["status"] = "success"
                except FitTimeout as exc:
                    row["status"] = "timeout"
                    row["error"] = str(exc)
                except ModelBuildError as exc:
                    row["status"] = "build_error"
                    row["error"] = str(exc)[:500]
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            print(f"{path.stem:12s} {representation:6s} {method:24s} "
                  f"{row['status']:18s} {row['auroc']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation", choices=["native", "pca2", "pca1"],
                        default="pca2")
    parser.add_argument("--train-cap", type=int, default=600)
    parser.add_argument("--test-cap", type=int, default=2000)
    parser.add_argument("--method-timeout", type=float, default=20.0)
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of the original capped normal-only fit pool to retain")
    parser.add_argument("--method-filter", choices=["all", "nsa", "baseline"],
                        default="all")
    args = parser.parse_args()
    evaluate(args.data_dir / f"{args.dataset}.mat", args.seed, args.train_cap,
             args.test_cap, args.representation, args.method_timeout,
             args.results, args.method_filter, args.train_fraction)


if __name__ == "__main__":
    main()
