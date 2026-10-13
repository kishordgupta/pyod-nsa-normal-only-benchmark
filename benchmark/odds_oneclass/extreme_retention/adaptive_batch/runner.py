"""Isolated AutoEncoder/VAE training-batch sensitivity using the pinned runner.

The primary ``runner.evaluate`` performs the split, preprocessing, timer,
fit, scoring, and metric calculation. Only the constructed estimator's
training batch is capped at the number of normal fitting rows. Its original
inference batch remains 32, including PyOD's scoring inside ``fit``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from model_factory import build_model as pinned_build_model  # noqa: E402

_spec = importlib.util.spec_from_file_location("_adaptive_pinned_runner", PROJECT / "runner.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot import the pinned primary runner")
primary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(primary)

METHODS = ("AutoEncoder", "VAE")
SPLIT_KEYS = ("dataset", "protocol_version", "dataset_rows", "source_features",
              "fit_normals", "original_fit_normals", "fit_outliers",
              "retention_fraction", "reduction_fraction", "fit_index_sha256",
              "fit_index_ordered_sha256", "fit_parent_index_sha256",
              "test_index_sha256", "discarded_train_outliers", "test_normals",
              "test_outliers", "representation", "retained_variance", "seed",
              "train_cap", "test_cap")


def split_matches(row: dict, expected: dict) -> bool:
    for key in SPLIT_KEYS:
        if key == "retained_variance":
            if not math.isclose(float(row.get(key, float("nan"))),
                                float(expected.get(key, float("nan"))),
                                rel_tol=0, abs_tol=1e-12):
                return False
        elif row.get(key) != expected.get(key):
            return False
    return True


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_name(dataset: str, representation: str, seed: int, fraction: float) -> str:
    return f"{dataset}_{representation}_seed{seed}_retain{round(fraction * 100)}"


def primary_path(manifest: dict, name: str, fraction: float) -> Path:
    root = Path(manifest["primary_historical_raw"] if fraction == .25
                else manifest["primary_extreme_raw"])
    return (root if root.is_absolute() else PROJECT.parents[1] / root) / f"{name}.jsonl"


def validate_primary(path: Path, expected_sha: str, expected: dict) -> dict[str, dict]:
    if sha(path) != expected_sha:
        raise ValueError(f"primary raw cell changed: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 56 or len({r["method"] for r in rows}) != 56:
        raise ValueError(f"primary raw cell is not a complete 56-method cell: {path}")
    for row in rows:
        if not split_matches(row, expected):
            raise ValueError(f"primary split differs for {path}:{row['method']}")
    return {r["method"]: r for r in rows}


def validate_existing(path: Path, expected: dict, primary_sha: str,
                      runner_sha: str, factory_sha: str) -> list[str]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            method = row.get("method")
            if len(rows) >= len(METHODS) or method != METHODS[len(rows)]:
                raise ValueError(f"wrong/duplicate adaptive method {path}:{line_no}: {method}")
            if not split_matches(row, expected):
                raise ValueError(f"adaptive split differs at {path}:{line_no}")
            meta = row.get("adaptive_batch")
            if not isinstance(meta, dict) or meta.get("primary_cell_sha256") != primary_sha:
                raise ValueError(f"missing primary lineage at {path}:{line_no}")
            if (meta.get("source_runner_sha256") != runner_sha
                    or meta.get("source_model_factory_sha256") != factory_sha
                    or meta.get("inference_batch_size") != 32):
                raise ValueError(f"adaptive code or inference provenance differs at {path}:{line_no}")
            expected_train = min(32, expected["fit_normals"])
            if (meta.get("original_training_batch_size") != 32
                    or meta.get("effective_training_batch_size") != expected_train
                    or meta.get("adapter_applied") != (expected_train < 32)):
                raise ValueError(f"adaptive training batch differs at {path}:{line_no}")
            rows.append(method)
    return rows


def evaluate_one(data_path: Path, seed: int, representation: str,
                 fraction: float, method: str, stage: Path) -> tuple[dict, dict]:
    state: dict = {"method": method, "observed_training_forward_calls": 0,
                   "model_built": False}

    def adaptive_build(name: str, model_seed: int, n_train: int, n_features: int):
        if name != method:
            raise ValueError(f"unexpected method {name}")
        model = pinned_build_model(name, model_seed, n_train, n_features)
        original_batch = int(model.batch_size)
        if original_batch != 32:
            raise ValueError(f"expected PyOD {name} default batch 32, got {original_batch}")
        train_batch = min(original_batch, n_train)
        model.batch_size = train_batch
        original_decision = model.decision_function
        original_forward = model.training_forward

        def fixed_inference_batch(X, batch_size=None):
            return original_decision(X, batch_size=original_batch if batch_size is None else batch_size)

        def observed_training_forward(*args, **kwargs):
            state["observed_training_forward_calls"] += 1
            return original_forward(*args, **kwargs)

        if train_batch != original_batch:
            # BaseDeepLearningDetector otherwise reuses self.batch_size for
            # inference. Preserve its original query batch only when changed.
            model.decision_function = fixed_inference_batch
        model.training_forward = observed_training_forward
        state.update(model_built=True, original_training_batch_size=original_batch,
                     effective_training_batch_size=train_batch,
                     inference_batch_size=original_batch,
                     full_training_minibatches_per_epoch=n_train // train_batch,
                     intended_epochs=int(model.epoch_num))
        return model

    saved_methods, saved_factory = primary.METHODS, primary.build_model
    primary.METHODS = (method,)
    primary.build_model = adaptive_build
    try:
        primary.evaluate(data_path, seed, 600, 2000, representation, 15.0,
                         stage, "all", fraction)
    finally:
        primary.METHODS, primary.build_model = saved_methods, saved_factory
    rows = [json.loads(line) for line in stage.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1 or rows[0]["method"] != method:
        raise ValueError(f"pinned runner did not produce exactly one {method} row")
    row = rows[0]
    if row["status"] == "success":
        expected_calls = (state["intended_epochs"] *
                          state["full_training_minibatches_per_epoch"])
        if state["observed_training_forward_calls"] != expected_calls:
            raise ValueError(f"{method} success ran {state['observed_training_forward_calls']} "
                             f"training batches, expected {expected_calls}")
    return row, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--representation", choices=("native", "pca2"), required=True)
    parser.add_argument("--fraction", type=float, choices=(.25, .10, .05), required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["sensitivity_protocol"] != "adaptive-training-batch-v1":
        raise ValueError("wrong sensitivity manifest")
    if (manifest["code_sha256"]["primary_runner.py"] != sha(PROJECT / "runner.py")
            or manifest["code_sha256"]["primary_model_factory.py"] != sha(PROJECT / "model_factory.py")
            or manifest["code_sha256"]["adaptive_runner.py"] != sha(Path(__file__))):
        raise ValueError("pinned runner/factory/adaptive wrapper changed")
    name = cell_name(args.dataset, args.representation, args.seed, args.fraction)
    expected_name = name + ".jsonl"
    if args.results.name != expected_name:
        raise ValueError(f"expected result filename {expected_name}")
    allowed = set(manifest["adaptive_cells"])
    if name not in allowed:
        raise ValueError(f"cell is outside the predeclared adaptive grid: {name}")
    data_root = Path(manifest["data_dir"])
    data_path = (data_root if data_root.is_absolute() else PROJECT.parents[1] / data_root) / f"{args.dataset}.mat"
    X_fit, _, _, split = primary.split_data(
        data_path, args.seed, 600, 2000, args.representation, args.fraction)
    if len(X_fit) != split["fit_normals"] or split["fit_outliers"] != 0:
        raise ValueError("normal-only split validation failed")
    path = primary_path(manifest, name, args.fraction)
    expected_primary_sha = manifest["primary_raw_cell_sha256"][path.name]
    parent = validate_primary(path, expected_primary_sha, split)
    if args.fraction == .25 and len(X_fit) >= 32:
        raise ValueError("25% adaptation only includes cells with fewer than 32 normals")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    completed = validate_existing(args.results, split, expected_primary_sha,
                                  manifest["code_sha256"]["primary_runner.py"],
                                  manifest["code_sha256"]["primary_model_factory.py"])
    (args.results.parent.parent / "staging").mkdir(parents=True, exist_ok=True)
    for method in METHODS[len(completed):]:
        with tempfile.TemporaryDirectory(prefix=f"{name}_{method}_", dir=str(args.results.parent.parent / "staging")) as temp:
            stage = Path(temp) / "method.jsonl"
            row, state = evaluate_one(data_path, args.seed, args.representation,
                                      args.fraction, method, stage)
        if not split_matches(row, split):
            raise ValueError(f"pinned runner split drifted: {name}/{method}")
        row["adaptive_batch"] = {
            "sensitivity_protocol": "adaptive-training-batch-v1",
            "original_training_batch_size": state.get("original_training_batch_size", 32),
            "effective_training_batch_size": state.get("effective_training_batch_size", min(32, len(X_fit))),
            "inference_batch_size": state.get("inference_batch_size", 32),
            "adapter_applied": len(X_fit) < 32,
            "observed_training_forward_calls": state["observed_training_forward_calls"],
            "full_training_minibatches_per_epoch": state.get("full_training_minibatches_per_epoch", max(1, len(X_fit) // 32)),
            "intended_epochs": state.get("intended_epochs"),
            "primary_cell_sha256": expected_primary_sha,
            "primary_method_status": parent[method]["status"],
            "source_runner_sha256": manifest["code_sha256"]["primary_runner.py"],
            "source_model_factory_sha256": manifest["code_sha256"]["primary_model_factory.py"],
        }
        if row["status"] == "success" and row["adaptive_batch"]["observed_training_forward_calls"] < 1:
            raise ValueError(f"successful {method} has no observed optimizer batch")
        with args.results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"{name} {method} {row['status']} batches={state['observed_training_forward_calls']}", flush=True)


if __name__ == "__main__":
    main()
