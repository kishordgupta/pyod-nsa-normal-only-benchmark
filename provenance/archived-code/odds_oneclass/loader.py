"""Load the 17 PyOD ODDS benchmark files for normal-only training.

The source MAT files are read in place. Labels are returned separately and are
never included in the feature matrix. A stratified 60/40 split is made before
training-pool anomalies are discarded, so evaluation retains both classes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


MANIFEST_PATH = Path(__file__).with_name("dataset_manifest.json")


@dataclass(frozen=True)
class OneClassSplit:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    excluded_train_anomaly_indices: np.ndarray

    @property
    def y_train(self) -> np.ndarray:
        """Labels for the retained normal-only training partition."""
        return np.zeros(len(self.train_indices), dtype=np.uint8)


def read_manifest(manifest_path: Path | str = MANIFEST_PATH) -> dict:
    with Path(manifest_path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    return manifest


def list_datasets(manifest_path: Path | str = MANIFEST_PATH) -> list[str]:
    return [entry["name"] for entry in read_manifest(manifest_path)["datasets"]]


def load_dataset(
    name: str,
    *,
    manifest_path: Path | str = MANIFEST_PATH,
    data_root: Path | str | None = None,
    verify_sha256: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric X and binary y (0 normal, 1 anomaly)."""
    manifest = read_manifest(manifest_path)
    entries = {entry["name"]: entry for entry in manifest["datasets"]}
    if name not in entries:
        raise KeyError(f"Unknown dataset {name!r}; choices: {sorted(entries)}")
    entry = entries[name]
    root = Path(data_root or manifest["source"]["local_directory"])
    path = root / entry["filename"]
    if verify_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
    mat = loadmat(path)
    X = np.asarray(mat["X"], dtype=np.float64)
    y = np.asarray(mat["y"]).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError(f"Invalid feature matrix in {path}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"Labels must be binary in {path}")
    y = y.astype(np.uint8)
    if X.shape != (entry["n_samples"], entry["n_features"]):
        raise ValueError(f"Shape differs from manifest for {path}")
    if int((y == 0).sum()) != entry["n_normal"] or int((y == 1).sum()) != entry["n_anomaly"]:
        raise ValueError(f"Class counts differ from manifest for {path}")
    return X, y


def make_oneclass_split(
    name: str,
    *,
    seed: int = 42,
    test_size: float = 0.4,
    manifest_path: Path | str = MANIFEST_PATH,
    data_root: Path | str | None = None,
    verify_sha256: bool = True,
) -> OneClassSplit:
    """Stratify full data, remove anomalies from train, retain mixed test.

    The 60/40 ratio and seed follow PyOD's published benchmark convention.
    Stratification is an explicit addition; these are not PyOD's original
    per-dataset random splits.
    """
    X, y = load_dataset(
        name,
        manifest_path=manifest_path,
        data_root=data_root,
        verify_sha256=verify_sha256,
    )
    pool_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    normal = y[pool_indices] == 0
    train_indices = pool_indices[normal]
    excluded = pool_indices[~normal]
    return OneClassSplit(
        name=name,
        X_train=X[train_indices],
        X_test=X[test_indices],
        y_test=y[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        excluded_train_anomaly_indices=excluded,
    )
