"""Rebuild the manifest for the exact MAT files retained in PyOD notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import loadmat


ORIGINAL_DATA_COMMIT = "ef6d5e8dc828615f498286c1d6a22cbd02fd4cd2"
DEFAULT_REPO = Path(
    "vendor/pyod"
)
DEFAULT_OUTPUT = Path(__file__).with_name("dataset_manifest.json")


def git(*args: str, repo: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True
    ).strip()


def build(repo: Path) -> dict:
    root = repo / "notebooks" / "data"
    records = []
    for path in sorted(root.glob("*.mat")):
        relative = f"notebooks/data/{path.name}"
        expected_blob = git("rev-parse", f"{ORIGINAL_DATA_COMMIT}:{relative}", repo=repo)
        actual_blob = git("hash-object", str(path), repo=repo)
        if actual_blob != expected_blob:
            raise ValueError(f"{path} differs from original PyOD dataset blob")
        mat = loadmat(path)
        X = np.asarray(mat["X"])
        y = np.asarray(mat["y"]).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y) or not np.isfinite(X).all():
            raise ValueError(f"Invalid data in {path}")
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"Unexpected labels in {path}")
        records.append(
            {
                "name": path.stem,
                "filename": path.name,
                "source_url": (
                    "https://github.com/yzhao062/pyod/blob/"
                    f"{ORIGINAL_DATA_COMMIT}/{relative}"
                ),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "git_blob_sha1": actual_blob,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_normal": int((y == 0).sum()),
                "n_anomaly": int((y == 1).sum()),
            }
        )
    if len(records) != 17:
        raise ValueError(f"Expected 17 PyOD benchmark files, got {len(records)}")
    return {
        "schema_version": 1,
        "source": {
            "repository_url": "https://github.com/yzhao062/pyod",
            "original_data_commit": ORIGINAL_DATA_COMMIT,
            "odds_catalog_url": "https://shebuti.com/outlier-detection-datasets-odds/",
            "local_directory": str(root.resolve()),
            "note": (
                "These 17 local files are byte-identical to their blobs at "
                "the original PyOD dataset commit. No downloaded copies "
                "or feature/label transformations are stored here."
            ),
        },
        "labels": {"normal": 0, "anomaly": 1},
        "datasets": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.repo)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['datasets'])} records to {args.output}")
