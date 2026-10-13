"""Aggregate JSONL benchmark records without treating failures as losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from model_factory import METHODS, METHOD_META


def load_records(raw_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in {raw_dir}")
    frame = pd.DataFrame.from_records(records)
    key = ["dataset", "seed", "representation", "method"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate task records: {frame.loc[duplicates, key].to_dict('records')[:5]}")
    frame["batch_dependent"] = frame.method.map(
        lambda method: METHOD_META[method].get("batch_dependent", False)
        or method in {"LMDD", "DIF", "CD", "ROD"}
    )
    frame["stochastic_inference"] = frame.method.isin({"VAE", "LSCP", "ALAD"})
    return frame.sort_values(["representation", "dataset", "method"])


def summarize(frame: pd.DataFrame, representation: str):
    panel = frame[frame.representation == representation].copy()
    good = panel[panel.status == "success"].copy()
    datasets = sorted(panel.dataset.unique())
    rows = []
    for method, group in panel.groupby("method", sort=True):
        ok = group[group.status == "success"]
        rows.append({
            "method": method,
            "family": group.family.iloc[0],
            "attempted_datasets": len(group),
            "success_datasets": len(ok),
            "mean_auroc": ok.auroc.mean() if len(ok) else np.nan,
            "median_auroc": ok.auroc.median() if len(ok) else np.nan,
            "mean_ap": ok.ap.mean() if len(ok) else np.nan,
            "median_fit_seconds": ok.fit_seconds.median() if len(ok) else np.nan,
            "status_counts": json.dumps(group.status.value_counts().to_dict(),
                                        sort_keys=True),
            "batch_dependent": bool(group.batch_dependent.iloc[0]),
            "stochastic_inference": bool(group.stochastic_inference.iloc[0]),
        })
    methods = pd.DataFrame(rows).sort_values(
        ["success_datasets", "mean_auroc"], ascending=[False, False]
    )
    dataset_rows = []
    for dataset, group in good.groupby("dataset", sort=True):
        nsa = group[group.method.str.startswith("NSA/")]
        base = group[(~group.method.str.startswith("NSA/")) &
                     (~group.batch_dependent)]
        iforest = base[base.method == "IForest"]
        best_nsa = nsa.sort_values("auroc", ascending=False).head(1)
        best_base = base.sort_values("auroc", ascending=False).head(1)
        dataset_rows.append({
            "dataset": dataset,
            "fit_normals": int(group.fit_normals.iloc[0]),
            "test_normals": int(group.test_normals.iloc[0]),
            "test_outliers": int(group.test_outliers.iloc[0]),
            "retained_variance": float(group.retained_variance.iloc[0]),
            "nsa_success": len(nsa),
            "baseline_success": len(base),
            "oracle_best_nsa": best_nsa.method.iloc[0] if len(best_nsa) else None,
            "oracle_best_nsa_auroc": best_nsa.auroc.iloc[0] if len(best_nsa) else np.nan,
            "oracle_best_inductive_baseline": best_base.method.iloc[0] if len(best_base) else None,
            "oracle_best_inductive_baseline_auroc": best_base.auroc.iloc[0] if len(best_base) else np.nan,
            "iforest_auroc": iforest.auroc.iloc[0] if len(iforest) else np.nan,
        })
    dataset_summary = pd.DataFrame(dataset_rows)
    pair_rows = []
    nsa_methods = sorted(x for x in good.method.unique() if x.startswith("NSA/"))
    baselines = sorted(x for x in good.method.unique() if not x.startswith("NSA/"))
    for a in nsa_methods:
        for b in baselines:
            left = good[good.method == a][["dataset", "seed", "auroc", "ap"]]
            right = good[good.method == b][["dataset", "seed", "auroc", "ap"]]
            merged = left.merge(right, on=["dataset", "seed"], suffixes=("_nsa", "_baseline"))
            delta = merged.auroc_nsa - merged.auroc_baseline
            pair_rows.append({
                "nsa": a, "baseline": b,
                "baseline_batch_dependent": bool(METHOD_META[b].get("batch_dependent", False)
                                                 or b in {"LMDD", "DIF", "CD", "ROD"}),
                "paired_datasets": len(delta),
                "nsa_higher": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "baseline_higher": int((delta < 0).sum()),
                "mean_delta_auroc": delta.mean() if len(delta) else np.nan,
                "median_delta_auroc": delta.median() if len(delta) else np.nan,
                "mean_delta_ap": (merged.ap_nsa - merged.ap_baseline).mean()
                if len(merged) else np.nan,
            })
    pairs = pd.DataFrame(pair_rows)
    return methods, dataset_summary, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representations", nargs="+", default=["pca2", "native"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_records(args.raw_dir)
    manifest = json.loads(args.dataset_manifest.read_text())
    datasets_expected = {entry["name"] for entry in manifest["datasets"]}
    expected = {(d, args.seed, r, m)
                for d in datasets_expected for r in args.representations
                for m in METHODS}
    observed = set(zip(frame.dataset, frame.seed,
                       frame.representation, frame.method))
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(f"Incomplete/inconsistent grid: {len(missing)} missing, "
                         f"{len(extra)} unexpected; first missing={sorted(missing)[:5]}")
    for field, value in {"train_cap": 600, "test_cap": 2000,
                         "fit_outliers": 0}.items():
        if not (frame[field] == value).all():
            raise ValueError(f"inconsistent protocol field {field}")
    good = frame[frame.status == "success"]
    if not good.auroc.between(0, 1).all() or not good.ap.between(0, 1).all():
        raise ValueError("invalid successful AUROC/AP values")
    frame.to_csv(args.out_dir / "all_results.csv", index=False)
    for representation in sorted(frame.representation.unique()):
        methods, datasets, pairs = summarize(frame, representation)
        methods.to_csv(args.out_dir / f"{representation}_methods.csv", index=False)
        datasets.to_csv(args.out_dir / f"{representation}_datasets.csv", index=False)
        pairs.to_csv(args.out_dir / f"{representation}_pairwise.csv", index=False)
        print(representation, len(datasets), "datasets", len(methods), "methods",
              (frame.representation == representation).sum(), "records")


if __name__ == "__main__":
    main()
