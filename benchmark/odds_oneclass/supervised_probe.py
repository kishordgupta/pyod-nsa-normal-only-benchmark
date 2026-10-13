"""Probe supervised PyOD detectors with normal-only training data.

This is an API/behavior check, not a benchmark. It uses 24 synthetic normal
rows, never supplies test labels, and does not compute detection metrics.
Run with the benchmark venv; the script locates that venv's bundled libomp on
macOS before importing XGBoost.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import warnings


EXPECTED_SOURCE = Path(__file__).resolve().parents[2] / "vendor/pyod/pyod"


def bootstrap_libomp() -> None:
    """Make the locally bundled OpenMP library visible to XGBoost on macOS."""
    if sys.platform != "darwin" or os.environ.get("NSA_PROBE_LIBOMP_READY"):
        return
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    torch_root = Path(next(iter(spec.submodule_search_locations)))
    libomp = torch_root / "lib" / "libomp.dylib"
    if not libomp.is_file():
        return
    lib_dir = str(libomp.parent)
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if lib_dir in existing.split(os.pathsep):
        return
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(x for x in (lib_dir, existing) if x)
    env["NSA_PROBE_LIBOMP_READY"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def exception_record(call) -> dict:
    try:
        call()
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": None, "message": "fit(X) unexpectedly completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("supervised_probe_results.json"),
    )
    args = parser.parse_args()
    bootstrap_libomp()

    import numpy as np
    import pyod
    import sklearn
    import torch
    import xgboost
    from pyod.models.devnet import DevNet
    from pyod.models.iforest import IForest
    from pyod.models.xgbod import XGBOD

    pyod_source = Path(pyod.__file__).resolve().parent
    if pyod_source != EXPECTED_SOURCE:
        raise RuntimeError(f"Wrong PyOD source: {pyod_source}; expected {EXPECTED_SOURCE}")

    X = np.random.RandomState(7).normal(size=(24, 3))
    y = np.zeros(24, dtype=np.int64)
    result = {
        "purpose": "normal-only execution probe; no anomaly test labels or AUC",
        "input": {
            "rows": 24, "features": 3, "normal_labels": 24,
            "outlier_labels": 0, "feature_seed": 7,
            "feature_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyod_version": pyod.__version__,
            "pyod_source": str(pyod_source),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "torch_version": torch.__version__,
            "libomp_directory": os.environ.get("DYLD_LIBRARY_PATH", ""),
        },
        "models": {},
    }

    xgb_settings = {
        "estimator_list": "[IForest(n_estimators=8, random_state=7)]",
        "standardization_flag_list": [False],
        "n_estimators": 2, "n_jobs": 1, "random_state": 7,
    }
    xgb = XGBOD(
        estimator_list=[IForest(n_estimators=8, random_state=7)],
        standardization_flag_list=[False], n_estimators=2, n_jobs=1,
        random_state=7,
    )
    xgb_record = {"constructor_settings": xgb_settings}
    xgb_record["fit_X_only"] = exception_record(lambda: xgb.fit(X))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xgb.fit(X, y)
        xgb_scores = np.asarray(xgb.decision_function(X), dtype=float)
    xgb_record["fit_X_all_zero_y"] = {
        "status": "completed",
        "classifier_classes": np.asarray(xgb.clf_.classes_).tolist(),
        "score_count": int(xgb_scores.size),
        "score_unique": int(np.unique(xgb_scores).size),
        "score_min": float(xgb_scores.min()),
        "score_max": float(xgb_scores.max()),
        "warnings": [str(w.message) for w in caught],
    }
    result["models"]["XGBOD"] = xgb_record

    dev_settings = {
        "network_depth": 1, "batch_size": 8, "epochs": 1,
        "device": "cpu", "random_seed": 7,
    }
    torch.manual_seed(42)
    dev = DevNet(**dev_settings)
    dev_record = {"constructor_settings": dev_settings, "torch_seed": 42}
    dev_record["fit_X_only"] = exception_record(lambda: dev.fit(X.astype("float32")))
    fit_stdout = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, redirect_stdout(fit_stdout):
        warnings.simplefilter("always")
        dev.fit(X.astype("float32"), y)
        dev_scores = np.asarray(dev.decision_function(X.astype("float32")), dtype=float)
    dev_record["fit_X_all_zero_y"] = {
        "status": "completed",
        "score_count": int(dev_scores.size),
        "score_unique": int(np.unique(dev_scores).size),
        "score_min": float(dev_scores.min()),
        "score_max": float(dev_scores.max()),
        "stdout": fit_stdout.getvalue().strip().splitlines(),
        "warnings": [str(w.message) for w in caught],
    }
    result["models"]["DevNet"] = dev_record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
