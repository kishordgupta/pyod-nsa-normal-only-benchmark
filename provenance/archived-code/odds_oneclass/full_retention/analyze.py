"""Audit and analyze the fresh, full-roster normal-only retention benchmark.

All 56 predeclared configurations are run afresh at each of four normal-training
sizes. Failure, timeout, and inapplicability records remain in every denominator.
Pairwise summaries use only matched successful dataset/seed cells. Dataset-cluster
bootstrap intervals are descriptive and are not multiple-testing adjusted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_factory import METHODS, METHOD_META, NSA_METHODS  # noqa: E402
from process_restart_audit import RESTART_CELLS, verify_all as verify_process_restarts  # noqa: E402

FRACTIONS = (100, 75, 50, 25)
REPS = ("native", "pca2")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
PROTOCOL = "retention-v2-parent-order"
BATCH_DEPENDENT = {"LMDD", "DIF", "CD", "ROD"}
MEASURED_TIME_BUDGET_SECONDS = 15.0
TERMINAL_STATUSES = {
    "success", "error", "timeout", "build_error", "requires_labeled_outliers",
    "requires_1d", "requires_2d", "requires_2d_or_3d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_hash(indices: np.ndarray, ordered: bool = False) -> str:
    values = np.asarray(indices, dtype="<i8")
    if not ordered:
        values = np.sort(values)
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(raw_dir: Path) -> list[dict]:
    # Production records are flat. Deliberately exclude smoke/ and archived
    # preliminary material beneath the experiment directory.
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"no JSONL files under {raw_dir}")
    rows = []
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"malformed {path}:{line_number}") from exc
                    row["_source_file"] = str(path.relative_to(raw_dir))
                    row["_source_line"] = line_number
                    rows.append(row)
    return rows


def expected_splits(manifest_path: Path, seeds: tuple[int, ...]) -> tuple[list[str], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["datasets"]
    names = sorted(item["name"] for item in items)
    if len(names) != 17 or len(set(names)) != 17:
        raise ValueError("dataset manifest must contain exactly 17 distinct ODDS datasets")
    source_dir = Path(manifest["source"]["local_directory"])
    expected = {}
    for item in items:
        dataset = item["name"]
        path = source_dir / item["filename"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"source dataset checksum changed: {path}")
        z = loadmat(path)
        X = np.asarray(z["X"])
        y = np.asarray(z["y"]).ravel().astype(np.int8)
        if (X.shape != (item["n_samples"], item["n_features"])
                or set(np.unique(y)) != {0, 1}
                or int(np.sum(y == 0)) != item["n_normal"]
                or int(np.sum(y == 1)) != item["n_anomaly"]
                or item["n_normal"] + item["n_anomaly"] != item["n_samples"]):
            raise ValueError(f"source dataset shape or labels changed: {dataset}")
        for seed in seeds:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
            parent = train_idx[y[train_idx] == 0]
            if len(parent) > 600:
                parent = np.sort(np.random.RandomState(seed + 10_000).choice(
                    parent, size=600, replace=False))
            if len(test_idx) > 2000:
                _, chosen = train_test_split(
                    np.arange(len(test_idx)), test_size=2000,
                    stratify=y[test_idx], random_state=seed + 20_000)
                test_idx = np.sort(test_idx[chosen])
            permutation = np.random.RandomState(seed + 30_000).permutation(parent)
            for pct in FRACTIONS:
                if pct == 100:
                    fit_idx = parent
                else:
                    keep = max(2, int(np.floor(len(parent) * pct / 100)))
                    fit_idx = parent[np.isin(parent, permutation[:keep])]
                if np.any(y[fit_idx] != 0):
                    raise AssertionError(f"reconstructed fit leakage: {dataset}/{seed}/{pct}")
                expected[(dataset, seed, pct)] = {
                    "dataset_rows": len(y),
                    "source_features": X.shape[1],
                    "fit_normals": len(fit_idx),
                    "original_fit_normals": len(parent),
                    "fit_outliers": 0,
                    "discarded_train_outliers": int(np.sum(y[train_idx] == 1)),
                    "test_normals": int(np.sum(y[test_idx] == 0)),
                    "test_outliers": int(np.sum(y[test_idx] == 1)),
                    "fit_index_sha256": index_hash(fit_idx),
                    "fit_index_ordered_sha256": index_hash(fit_idx, ordered=True),
                    "fit_parent_index_sha256": index_hash(parent),
                    "test_index_sha256": index_hash(test_idx),
                }
    return names, expected


def load_validate(raw_dir: Path, manifest_path: Path, seeds: tuple[int, ...]) -> tuple[dict, list[str]]:
    names, split = expected_splits(manifest_path, seeds)
    expected_files = {
        f"{d}_{rep}_seed{seed}{'' if pct == 100 else f'_retain{pct}'}.jsonl"
        for d in names for seed in seeds for rep in REPS for pct in FRACTIONS
    }
    found_files = {p.name for p in raw_dir.glob("*.jsonl")}
    if found_files != expected_files:
        raise ValueError(f"production JSONL file grid incomplete: expected {len(expected_files)}, "
                         f"found {len(found_files)}, missing={sorted(expected_files-found_files)[:5]}, "
                         f"extra={sorted(found_files-expected_files)[:5]}")
    raw = read_jsonl(raw_dir)
    expected = {(d, seed, rep, pct, method)
                for d in names for seed in seeds for rep in REPS
                for pct in FRACTIONS for method in METHODS}
    found = {}
    file_methods = defaultdict(set)
    file_counts = Counter()
    for row in raw:
        pct_float = float(row.get("retention_fraction", -1)) * 100
        pct = int(round(pct_float))
        if not math.isclose(pct_float, pct, abs_tol=1e-9):
            raise ValueError(f"bad retention fraction: {row['_source_file']}:{row['_source_line']}")
        key = (row.get("dataset"), row.get("seed"), row.get("representation"), pct, row.get("method"))
        expected_name = f"{key[0]}_{key[2]}_seed{key[1]}{'' if pct == 100 else f'_retain{pct}'}.jsonl"
        if row["_source_file"] != expected_name:
            raise ValueError(f"row/file membership mismatch: {key} in {row['_source_file']}")
        if key in found:
            raise ValueError(f"duplicate result key {key}: {row['_source_file']}:{row['_source_line']}")
        found[key] = row
        file_methods[row["_source_file"]].add(row.get("method"))
        file_counts[row["_source_file"]] += 1
    if set(found) != expected or len(raw) != len(expected):
        raise ValueError(f"incomplete result grid: expected {len(expected)}, observed {len(raw)}, "
                         f"missing={sorted(expected-set(found))[:5]}, extra={sorted(set(found)-expected)[:5]}")
    if any(file_counts[name] != len(METHODS) or file_methods[name] != set(METHODS)
           for name in expected_files):
        raise ValueError("one or more production JSONL files lack exactly 56 unique methods")
    representation_variance = defaultdict(set)
    for key, row in found.items():
        dataset, seed, rep, pct, method = key
        ref = split[(dataset, seed, pct)]
        for field, value in ref.items():
            if row.get(field) != value:
                raise ValueError(f"{field} mismatch: {key}; observed={row.get(field)!r}, expected={value!r}")
        if row.get("protocol_version") != PROTOCOL:
            raise ValueError(f"wrong protocol: {key}")
        if row.get("train_cap") != 600 or row.get("test_cap") != 2000:
            raise ValueError(f"cap mismatch: {key}")
        if row.get("retention_fraction") != pct / 100 or not math.isclose(
                row.get("reduction_fraction", -1), 1 - pct / 100, abs_tol=1e-12):
            raise ValueError(f"retention/reduction mismatch: {key}")
        if row.get("family") != METHOD_META[method]["family"]:
            raise ValueError(f"method family mismatch: {key}")
        if row.get("status") not in TERMINAL_STATUSES:
            raise ValueError(f"unknown or nonterminal status: {key}: {row.get('status')!r}")
        if row.get("status") == "success":
            for metric in ("auroc", "ap"):
                value = row.get(metric)
                # sklearn can return AP=1.0000000000000002 for a perfect
                # ranking because of floating-point summation. Accept only
                # machine-scale overshoot; retain the raw metric unchanged.
                if (not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not -1e-12 <= value <= 1 + 1e-12):
                    raise ValueError(f"invalid success {metric}: {key}")
            for timing in ("fit_seconds", "score_seconds"):
                value = row.get(timing)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid success {timing}: {key}")
        elif row.get("auroc") is not None or row.get("ap") is not None:
            # The alarm can fire after AUROC has been calculated but before
            # AP finishes. This is an incomplete timeout, never a scored run.
            partial_auc = row.get("auroc")
            if (row["status"] != "timeout" or row.get("ap") is not None
                    or not isinstance(partial_auc, (int, float))
                    or not math.isfinite(partial_auc)
                    or not -1e-12 <= partial_auc <= 1 + 1e-12):
                raise ValueError(f"non-success row has metric: {key}")
            row["_partial_auroc_timeout"] = True
        variance = row.get("retained_variance")
        if not isinstance(variance, (int, float)) or not np.isfinite(variance) or not 0 <= variance <= 1.000001:
            raise ValueError(f"invalid retained variance: {key}")
        representation_variance[(dataset, seed, rep, pct)].add(round(float(variance), 12))
    if any(len(variances) != 1 for variances in representation_variance.values()):
        raise ValueError("feature transform differs across methods for a common split")
    return found, names


def apply_measured_time_budget(index: dict, seconds: float = MEASURED_TIME_BUDGET_SECONDS) -> list[dict]:
    """Qualify raw successes by the recorded t0-to-t2 elapsed time.

    The original JSONL status and metrics are left intact on disk. In-memory
    analysis status becomes ``overbudget_success`` when fit+score time exceeds
    the 15-second policy. The recorded fit duration includes pre-timer torch
    setup, while the OS alarm itself starts just before model construction.
    """
    if seconds <= 0:
        raise ValueError("measured time budget must be positive")
    excluded = []
    for key, row in index.items():
        raw_status = row["status"]
        row["_raw_status"] = raw_status
        row["_analysis_status"] = raw_status
        if raw_status != "success":
            continue
        elapsed = float(row["fit_seconds"]) + float(row["score_seconds"])
        row["_measured_end_to_end_seconds"] = elapsed
        if elapsed > seconds:
            row["status"] = "overbudget_success"
            row["_analysis_status"] = "overbudget_success"
            excluded.append({
                "dataset": key[0], "seed": key[1], "representation": key[2],
                "retention_pct": key[3], "method": key[4],
                "raw_status": raw_status, "analysis_status": row["status"],
                "fit_seconds": row["fit_seconds"],
                "score_seconds": row["score_seconds"],
                "measured_end_to_end_seconds": elapsed,
                "measured_time_budget_seconds": seconds,
                "raw_auroc": row["auroc"], "raw_ap": row["ap"],
                "source_file": row["_source_file"],
                "source_line": row["_source_line"],
            })
    return sorted(excluded, key=lambda r: (r["dataset"], r["seed"],
                                            r["representation"], r["retention_pct"],
                                            r["method"]))


def audit_external_watchdog(intervention_root: Path, raw_dir: Path,
                            index: dict) -> dict:
    """Match externally terminated LOCI attempts to immutable snapshots and final rows."""
    event_paths = sorted(intervention_root.glob("*/intervention.json"))
    all_ids = set()
    summarized = []
    matched_keys = set()
    for event_path in event_paths:
        event = json.loads(event_path.read_text(encoding="utf-8"))
        intervention_id = event.get("intervention_id")
        if (event.get("schema_version") != 1
                or not isinstance(intervention_id, str) or not intervention_id
                or intervention_id != event_path.parent.name
                or intervention_id in all_ids or event.get("state") != "completed"):
            raise ValueError(f"incomplete or duplicate watchdog intervention: {event_path}")
        all_ids.add(intervention_id)
        cell = event["cell"]
        dataset = cell["dataset"]
        representation = cell["representation"]
        seed = int(cell["seed"])
        fraction = float(cell["retention_fraction"])
        pct = int(round(100 * fraction))
        expected_stem = f"{dataset}_{representation}_seed{seed}{'' if pct == 100 else f'_retain{pct}'}"
        if (pct not in FRACTIONS or not math.isclose(fraction, pct / 100, abs_tol=1e-12)
                or cell.get("method") != "LOCI" or cell.get("name") != expected_stem):
            raise ValueError(f"invalid watchdog retention fraction: {event_path}")
        key = (dataset, seed, representation, pct, "LOCI")
        if key not in index or key in matched_keys:
            raise ValueError(f"watchdog cell absent or duplicated in final grid: {key}")
        matched_keys.add(key)
        row = index[key]
        marker = row.get("external_watchdog")
        if (row["status"] != "timeout"
                or not str(row.get("error", "")).startswith("ExternalWatchdogTimeout:")
                or not isinstance(marker, dict)
                or marker.get("intervention_id") != intervention_id
                or marker.get("audit_path") != str(event_path.relative_to(intervention_root.parent.parent))
                or marker.get("target_pid") != event.get("target", {}).get("pid")):
            raise ValueError(f"watchdog event lacks matching final LOCI timeout row: {key}")
        before = event["before"]
        age = float(before["raw_age_seconds"])
        if (not math.isfinite(age) or age < 600
                or before.get("raw_record_count") != 31
                or before.get("raw_snapshot") != "raw.before.jsonl"
                or before.get("log_snapshot") != "log.before.log"
                or event.get("after", {}).get("raw_record_count") != 32
                or event["after"].get("appended_method") != "LOCI"
                or event["after"].get("appended_status") != "timeout"
                or event.get("action", {}).get("signal") != "SIGTERM"
                or event["action"].get("termination_confirmed") is not True
                or event["action"].get("primary_method_timeout_seconds") != 15.0
                or event.get("resume_state") != "pending_unfiltered_sweep"
                or not isinstance(event.get("target", {}).get("pid"), int)
                or event["target"]["pid"] < 1):
            raise ValueError(f"watchdog acted before 600 seconds: {event_path}")
        raw_snapshot = event_path.parent / "raw.before.jsonl"
        log_snapshot = event_path.parent / "log.before.log"
        if (not raw_snapshot.is_file() or not log_snapshot.is_file()
                or sha(raw_snapshot) != before["raw_sha256"]
                or sha(log_snapshot) != before["log_sha256"]):
            raise ValueError(f"watchdog snapshot missing or hash mismatch: {event_path}")
        raw_before = raw_snapshot.read_bytes()
        log_before = log_snapshot.read_bytes()
        if len(raw_before.splitlines()) != 31:
            raise ValueError(f"watchdog before snapshot lacks 31 completed methods: {event_path}")
        raw_final = (raw_dir / row["_source_file"]).read_bytes()
        log_final = (intervention_root.parent.parent / "logs" /
                     f"{Path(row['_source_file']).stem}.log").read_bytes()
        if not log_final.startswith(log_before):
            raise ValueError(f"watchdog log snapshot is not a prefix of final log: {event_path}")
        lines = raw_final.splitlines(keepends=True)
        source_line = int(row["_source_line"])
        if source_line != 32 or source_line > len(lines):
            raise ValueError(f"watchdog final row line not found: {event_path}")
        appended = lines[source_line - 1]
        appended_sha = hashlib.sha256(appended).hexdigest()
        if appended_sha != event["appended_row_sha256"]:
            raise ValueError(f"watchdog appended row hash mismatch: {event_path}")
        if not raw_final.startswith(raw_before + appended):
            raise ValueError(f"watchdog raw snapshot is not followed by appended LOCI row: {event_path}")
        if hashlib.sha256(raw_before + appended).hexdigest() != event["after"]["raw_sha256"]:
            raise ValueError(f"watchdog immediate-after raw hash mismatch: {event_path}")
        for line in raw_before.splitlines():
            if line.strip() and json.loads(line).get("method") == "LOCI":
                raise ValueError(f"watchdog before snapshot already scored LOCI: {event_path}")
        log_evidence = log_before.decode("utf-8", errors="replace")
        signal_evidence = (
            ("Exception ignored on calling ctypes callback function" in log_evidence
             or "Exception ignored in:" in log_evidence)
            and ("llvmlite/binding/executionengine.py" in log_evidence
                 or "numba/core/typing/context.py" in log_evidence)
            and "runner.py" in log_evidence
            and "FitTimeout: per-method wall-time budget exceeded" in log_evidence
        )
        if not signal_evidence:
            raise ValueError(f"watchdog event lacks swallowed-signal evidence: {event_path}")
        summarized.append({
            "intervention_id": intervention_id,
            "dataset": dataset, "seed": seed,
            "representation": representation, "retention_pct": pct,
            "method": "LOCI", "raw_age_seconds_before_action": age,
            "target_pid": event.get("target", {}).get("pid"),
            "action_signal": event.get("action", {}).get("signal"),
            "raw_before_sha256": before["raw_sha256"],
            "log_before_sha256": before["log_sha256"],
            "appended_row_sha256": appended_sha,
            "raw_immediate_after_sha256": event["after"]["raw_sha256"],
            "resume_state_at_intervention": event.get("resume_state"),
            "evidence_path": str(event_path.relative_to(intervention_root.parent.parent)),
        })
    marked = {key for key, row in index.items()
              if isinstance(row.get("external_watchdog"), dict)
              or str(row.get("error", "")).startswith("ExternalWatchdogTimeout:")}
    if marked != matched_keys:
        raise ValueError(f"orphan watchdog event or raw row: events={len(matched_keys)}, rows={len(marked)}")
    return {
        "event_count": len(summarized),
        "all_snapshots_sha256_verified": True,
        "all_final_loci_rows_matched": True,
        "minimum_required_raw_age_seconds": 600,
        "events": sorted(summarized, key=lambda e: e["intervention_id"]),
    }


def mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def cluster_ci(dataset_values: list[float], key: str, reps: int) -> tuple[float | None, float | None]:
    # With very few independent datasets, a cluster bootstrap interval is
    # especially misleading (one cluster produces a degenerate interval).
    if len(dataset_values) < 5:
        return None, None
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    values = np.asarray(dataset_values, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(values), size=(reps, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), (0.025, 0.975))
    return float(low), float(high)


def paired_stats(pairs: list[tuple[str, int, float, float]], key: str, reps: int) -> dict:
    # Algebraically equivalent interaction formulas can differ at machine
    # precision. Treat differences below metric precision as ties.
    tie_tolerance = 1e-12
    by_dataset = defaultdict(list)
    for dataset, seed, auc, ap in pairs:
        by_dataset[dataset].append((auc, ap))
    auc_ds = [float(np.mean([v[0] for v in by_dataset[d]])) for d in sorted(by_dataset)]
    ap_ds = [float(np.mean([v[1] for v in by_dataset[d]])) for d in sorted(by_dataset)]
    auc_lo, auc_hi = cluster_ci(auc_ds, key + "|auroc", reps)
    ap_lo, ap_hi = cluster_ci(ap_ds, key + "|ap", reps)
    return {
        "paired_seed_rows": len(pairs),
        "paired_datasets": len(by_dataset),
        "first_higher_seed_rows": sum(v[2] > tie_tolerance for v in pairs),
        "ties_seed_rows": sum(abs(v[2]) <= tie_tolerance for v in pairs),
        "second_higher_seed_rows": sum(v[2] < -tie_tolerance for v in pairs),
        "first_higher_datasets": sum(v > tie_tolerance for v in auc_ds),
        "tie_datasets": sum(abs(v) <= tie_tolerance for v in auc_ds),
        "second_higher_datasets": sum(v < -tie_tolerance for v in auc_ds),
        "macro_mean_delta_auroc": mean(auc_ds),
        "macro_mean_delta_ap": mean(ap_ds),
        "cluster_bootstrap_95_low_auroc": auc_lo,
        "cluster_bootstrap_95_high_auroc": auc_hi,
        "cluster_bootstrap_95_low_ap": ap_lo,
        "cluster_bootstrap_95_high_ap": ap_hi,
        "bootstrap_reps": reps,
    }


def summarize(index: dict, names: list[str], seeds: tuple[int, ...], reps: int,
              bootstrap_seed: int) -> dict[str, list[dict]]:
    n_expected = len(names) * len(seeds)
    baselines = [m for m in METHODS if m not in NSA_METHODS]
    summary, changes, dataset_changes, pairs_same, interactions = [], [], [], [], []
    for rep in REPS:
        for pct in FRACTIONS:
            for method in METHODS:
                rows = [index[(d, seed, rep, pct, method)] for d in names for seed in seeds]
                statuses = Counter(r["status"] for r in rows)
                successful = [r for r in rows if r["status"] == "success"]
                metric_by_ds = defaultdict(list)
                for r in successful:
                    metric_by_ds[r["dataset"]].append(r)
                auc = mean(np.mean([r["auroc"] for r in metric_by_ds[d]]) for d in sorted(metric_by_ds))
                ap = mean(np.mean([r["ap"] for r in metric_by_ds[d]]) for d in sorted(metric_by_ds))
                meta = METHOD_META[method]
                summary.append({
                    "representation": rep, "retention_pct": pct, "reduction_pct": 100-pct,
                    "method": method, "family": meta["family"],
                    "strategy": meta.get("strategy", ""),
                    "applicability": meta["applicability"],
                    "batch_dependent": bool(meta.get("batch_dependent") or method in BATCH_DEPENDENT),
                    "compute_overrides": json.dumps(meta.get("compute_overrides", {}), sort_keys=True),
                    "expected_seed_rows": n_expected, "success_seed_rows": len(successful),
                    "success_datasets": len(metric_by_ds),
                    "full_coverage": len(successful) == n_expected,
                    "macro_auroc_available": auc, "macro_ap_available": ap,
                    "macro_auroc_full_coverage": auc if len(successful) == n_expected else None,
                    "macro_ap_full_coverage": ap if len(successful) == n_expected else None,
                    "median_fit_seconds_success": mean([]) if not successful else float(np.median([
                        r["fit_seconds"] for r in successful if r.get("fit_seconds") is not None])),
                    "status_counts": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
                })
            print(f"Summarized {rep} {pct}%", flush=True)
    for rep in REPS:
        for pct in FRACTIONS:
            if pct != 100:
                for method in METHODS:
                    matched = []
                    for dataset in names:
                        ds_pairs = []
                        full_status, reduced_status = Counter(), Counter()
                        for seed in seeds:
                            a = index[(dataset, seed, rep, pct, method)]
                            b = index[(dataset, seed, rep, 100, method)]
                            reduced_status[a["status"]] += 1
                            full_status[b["status"]] += 1
                            if a["status"] == b["status"] == "success":
                                ds_pairs.append((dataset, seed, a["auroc"] - b["auroc"],
                                                 a["ap"] - b["ap"]))
                                matched.append(ds_pairs[-1])
                        dataset_changes.append({
                            "representation": rep, "retention_pct": pct, "method": method,
                            "dataset": dataset, "expected_seed_pairs": len(seeds),
                            "paired_success_seed_rows": len(ds_pairs),
                            "full_status_counts": json.dumps(dict(sorted(full_status.items()))),
                            "reduced_status_counts": json.dumps(dict(sorted(reduced_status.items()))),
                            "mean_delta_auroc": mean(p[2] for p in ds_pairs),
                            "mean_delta_ap": mean(p[3] for p in ds_pairs),
                        })
                    changes.append({
                        "representation": rep, "retention_pct": pct, "reduction_pct": 100-pct,
                        "method": method, "family": METHOD_META[method]["family"],
                        "expected_pairs": n_expected,
                        "full_coverage": len(matched) == n_expected,
                        **paired_stats(matched, f"{bootstrap_seed}|change|{rep}|{pct}|{method}", reps),
                    })
            for nsa in NSA_METHODS:
                for baseline in baselines:
                    matched, interaction = [], []
                    for dataset in names:
                        for seed in seeds:
                            a = index[(dataset, seed, rep, pct, nsa)]
                            b = index[(dataset, seed, rep, pct, baseline)]
                            if a["status"] == b["status"] == "success":
                                matched.append((dataset, seed, a["auroc"]-b["auroc"],
                                                a["ap"]-b["ap"]))
                                if pct != 100:
                                    af = index[(dataset, seed, rep, 100, nsa)]
                                    bf = index[(dataset, seed, rep, 100, baseline)]
                                    if af["status"] == bf["status"] == "success":
                                        interaction.append((dataset, seed,
                                            (a["auroc"]-af["auroc"])-(b["auroc"]-bf["auroc"]),
                                            (a["ap"]-af["ap"])-(b["ap"]-bf["ap"])))
                    common = {
                        "representation": rep, "retention_pct": pct,
                        "reduction_pct": 100-pct, "nsa": nsa, "baseline": baseline,
                        "baseline_batch_dependent": bool(METHOD_META[baseline].get("batch_dependent")
                                                         or baseline in BATCH_DEPENDENT),
                        "expected_pairs": n_expected,
                    }
                    pairs_same.append({**common, "full_coverage": len(matched) == n_expected,
                        **paired_stats(matched,
                            f"{bootstrap_seed}|same|{rep}|{pct}|{nsa}|{baseline}", reps)})
                    if pct != 100:
                        interactions.append({**common,
                            "full_coverage": len(interaction) == n_expected,
                            **paired_stats(interaction,
                                f"{bootstrap_seed}|interaction|{rep}|{pct}|{nsa}|{baseline}", reps)})
            print(f"Paired {rep} {pct}%", flush=True)
    ranks = []
    for rep in REPS:
        for pct in FRACTIONS:
            available = [r for r in summary if r["representation"] == rep
                         and r["retention_pct"] == pct and r["full_coverage"]]
            for metric in ("auroc", "ap"):
                ordered = sorted(available,
                                 key=lambda r: (-r[f"macro_{metric}_full_coverage"], r["method"]))
                within_family = Counter()
                for overall_rank, row in enumerate(ordered, 1):
                    within_family[row["family"]] += 1
                    ranks.append({
                        "representation": rep, "retention_pct": pct, "metric": metric,
                        "method": row["method"], "family": row["family"],
                        "score": row[f"macro_{metric}_full_coverage"],
                        "rank_overall_full_coverage": overall_rank,
                        "rank_family_full_coverage": within_family[row["family"]],
                        "full_coverage_methods_overall": len(available),
                        "full_coverage_methods_family": sum(r["family"] == row["family"] for r in available),
                        "expected_seed_rows": n_expected,
                    })
    # Leave-one-dataset-out checks are calculated for every method with a
    # complete 100%-versus-reduced paired grid, rather than selected only after
    # observing which NSA family has the largest apparent gain.
    by_change = defaultdict(list)
    for row in dataset_changes:
        by_change[(row["representation"], row["retention_pct"], row["method"])].append(row)
    leave_one_out = []
    for (rep, pct, method), rows in sorted(by_change.items()):
        if len(rows) != len(names) or any(r["paired_success_seed_rows"] != len(seeds)
                                           for r in rows):
            continue
        full_auc = mean(r["mean_delta_auroc"] for r in rows)
        full_ap = mean(r["mean_delta_ap"] for r in rows)
        for excluded in names:
            remainder = [r for r in rows if r["dataset"] != excluded]
            auc_loo = mean(r["mean_delta_auroc"] for r in remainder)
            ap_loo = mean(r["mean_delta_ap"] for r in remainder)
            leave_one_out.append({
                "representation": rep, "retention_pct": pct, "method": method,
                "excluded_dataset": excluded, "remaining_datasets": len(remainder),
                "full_macro_delta_auroc": full_auc,
                "leave_one_out_macro_delta_auroc": auc_loo,
                "removed_dataset_influence_auroc": full_auc - auc_loo,
                "full_macro_delta_ap": full_ap,
                "leave_one_out_macro_delta_ap": ap_loo,
                "removed_dataset_influence_ap": full_ap - ap_loo,
                "exploratory_diagnostic": True,
            })
    return {
        "method_summary": summary,
        "change_vs_full": changes,
        "dataset_changes": dataset_changes,
        "nsa_vs_baseline": pairs_same,
        "interaction": interactions,
        "coverage_qualified_ranks": ranks,
        "leave_one_out": leave_one_out,
    }


def dataset_inventory(index: dict, manifest_path: Path,
                      seeds: tuple[int, ...]) -> list[dict]:
    """Source class counts plus actual audited fit-row ranges across seeds."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in sorted(manifest["datasets"], key=lambda x: x["name"]):
        dataset = item["name"]
        record = {
            "dataset": dataset,
            "source_rows": item["n_samples"],
            "source_features": item["n_features"],
            "source_normals": item["n_normal"],
            "source_anomalies": item["n_anomaly"],
            "fit_seed_count": len(seeds),
        }
        for pct in FRACTIONS:
            counts = []
            for seed in seeds:
                native = index[(dataset, seed, "native", pct, METHODS[0])]["fit_normals"]
                pca2 = index[(dataset, seed, "pca2", pct, METHODS[0])]["fit_normals"]
                if native != pca2:
                    raise ValueError(f"representation fit-row count mismatch: {dataset}/{seed}/{pct}")
                counts.append(native)
            record[f"fit_normals_{pct}_min"] = min(counts)
            record[f"fit_normals_{pct}_max"] = max(counts)
        rows.append(record)
    if len(rows) != 17:
        raise ValueError("dataset inventory must have 17 rows")
    return rows


def _macro_metric(rows: list[dict], metric: str) -> float | None:
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(float(row[metric]))
    return mean(np.mean(by_dataset[d]) for d in sorted(by_dataset))


def recorded_vs_qualified(index: dict, names: list[str],
                          seeds: tuple[int, ...]) -> list[dict]:
    """Descriptive available-case score sensitivity to measured-time filtering."""
    out = []
    for rep in REPS:
        for pct in FRACTIONS:
            for method in METHODS:
                rows = [index[(d, seed, rep, pct, method)] for d in names for seed in seeds]
                recorded = [r for r in rows if r["_raw_status"] == "success"]
                qualified = [r for r in rows if r["status"] == "success"]
                raw_auc = _macro_metric(recorded, "auroc")
                good_auc = _macro_metric(qualified, "auroc")
                raw_ap = _macro_metric(recorded, "ap")
                good_ap = _macro_metric(qualified, "ap")
                out.append({
                    "representation": rep, "retention_pct": pct, "method": method,
                    "expected_seed_rows": len(rows),
                    "raw_success_seed_rows": len(recorded),
                    "qualified_success_seed_rows": len(qualified),
                    "overbudget_success_seed_rows": len(recorded) - len(qualified),
                    "macro_auroc_all_recorded_successes": raw_auc,
                    "macro_auroc_time_qualified": good_auc,
                    "qualified_minus_recorded_auroc":
                        good_auc - raw_auc if raw_auc is not None and good_auc is not None else None,
                    "macro_ap_all_recorded_successes": raw_ap,
                    "macro_ap_time_qualified": good_ap,
                    "qualified_minus_recorded_ap":
                        good_ap - raw_ap if raw_ap is not None and good_ap is not None else None,
                    "interpretation": "descriptive available-case sensitivity; successful cell sets can differ",
                })
    return out


def overbudget_method_counts(excluded: list[dict]) -> list[dict]:
    by_method = defaultdict(list)
    for row in excluded:
        by_method[row["method"]].append(row)
    out = []
    for method in METHODS:
        rows = by_method[method]
        elapsed = [r["measured_end_to_end_seconds"] for r in rows]
        out.append({
            "method": method, "family": METHOD_META[method]["family"],
            "overbudget_success_records": len(rows),
            "max_measured_end_to_end_seconds": max(elapsed) if elapsed else None,
            "mean_measured_end_to_end_seconds": mean(elapsed),
            "by_representation": json.dumps(dict(sorted(Counter(
                r["representation"] for r in rows).items())), sort_keys=True),
            "by_retention_pct": json.dumps(dict(sorted(Counter(
                r["retention_pct"] for r in rows).items())), sort_keys=True),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "dataset_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "analysis")
    parser.add_argument("--watchdog-interventions", type=Path,
                        default=Path(__file__).parent / "watchdog" / "interventions")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or args.bootstrap_reps < 1000:
        parser.error("seeds must be unique; bootstrap-reps must be >= 1000")
    seeds = tuple(args.seeds)
    index, names = load_validate(args.raw_dir, args.dataset_manifest, seeds)
    watchdog = audit_external_watchdog(args.watchdog_interventions, args.raw_dir, index)
    provenance_dir = Path(__file__).parent / "provenance"
    restart_paths = sorted(provenance_dir.glob("*_restart.json"))
    if [path.name for path in restart_paths] != sorted(
            f"{cell}_restart.json" for cell in RESTART_CELLS):
        raise ValueError("process restart sidecars are incomplete or unexpected")
    verified_restarts = verify_process_restarts(Path(__file__).parent)
    restart_events = []
    for path in restart_paths:
        saved_restart = json.loads(path.read_text(encoding="utf-8"))
        cell = saved_restart["cell"]
        if cell not in verified_restarts or saved_restart != verified_restarts[cell]:
            raise ValueError(f"process restart sidecar differs from final raw/log evidence: {path}")
        restart_events.append({
            "cell": cell,
            "sidecar_sha256": sha(path),
            "first_completed_record_count": saved_restart["first_invocation"]["completed_record_count"],
            "final_record_count": saved_restart["final"]["record_count"],
        })
    restart_audit = {
        "event_count": len(restart_events),
        "events": restart_events,
        "raw_and_log_hashes_verified": True,
    }
    excluded = apply_measured_time_budget(index)
    partial_metric_timeouts = sorted(({
        "dataset": key[0], "seed": key[1], "representation": key[2],
        "retention_pct": key[3], "method": key[4],
        "status": row["status"], "intermediate_auroc": row["auroc"],
        "ap": row.get("ap"), "source_file": row["_source_file"],
        "source_line": row["_source_line"],
        "interpretation": "timeout during metrics; not a scored AUROC/AP pair",
    } for key, row in index.items() if row.get("_partial_auroc_timeout")),
        key=lambda r: (r["dataset"], r["seed"], r["representation"],
                       r["retention_pct"], r["method"]))
    tables = summarize(index, names, seeds, args.bootstrap_reps, args.bootstrap_seed)
    tables["dataset_inventory"] = dataset_inventory(index, args.dataset_manifest, seeds)
    tables["overbudget_success"] = excluded
    tables["overbudget_method_counts"] = overbudget_method_counts(excluded)
    tables["recorded_vs_qualified"] = recorded_vs_qualified(index, names, seeds)
    if partial_metric_timeouts:
        tables["partial_metric_timeouts"] = partial_metric_timeouts
    for name, rows in tables.items():
        write_csv(args.out_dir / f"full_retention_{name}.csv", rows)
    audit = {
        "protocol_version": PROTOCOL,
        "datasets": names,
        "seeds": seeds,
        "representations": REPS,
        "retention_percent": FRACTIONS,
        "methods": METHODS,
        "expected_records": len(index),
        "success_records": sum(r["status"] == "success" for r in index.values()),
        "status_counts": dict(sorted(Counter(r["status"] for r in index.values()).items())),
        "raw_success_records": sum(r["_raw_status"] == "success" for r in index.values()),
        "raw_status_counts": dict(sorted(Counter(r["_raw_status"] for r in index.values()).items())),
        "qualified_status_counts": dict(sorted(Counter(r["status"] for r in index.values()).items())),
        "overbudget_success_records": len(excluded),
        "partial_metric_timeout_rows": len(partial_metric_timeouts),
        "external_watchdog_intervention_count": watchdog["event_count"],
        "external_watchdog_loci_timeout_rows": watchdog["event_count"],
        "process_restart_event_count": restart_audit["event_count"],
        "measured_time_budget_seconds": MEASURED_TIME_BUDGET_SECONDS,
        "time_qualification_rule": "raw success is analysis overbudget_success iff recorded fit_seconds + score_seconds > 15.0; original JSONL status and scores are preserved",
        "source_dataset_manifest_sha256": sha(args.dataset_manifest),
        "raw_jsonl_files": len(list(args.raw_dir.glob("*.jsonl"))),
        "analysis_tables": {k: len(v) for k, v in tables.items()},
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "win_tie_tolerance_auroc": 1e-12,
        "validation": "exact full grid; source hashes; split and fit/test row identity/order; zero fit outliers; fixed test rows; valid success metrics; transform consistency",
    }
    (args.out_dir / "full_retention_analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "watchdog_intervention_audit.json").write_text(
        json.dumps(watchdog, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "process_restart_audit.json").write_text(
        json.dumps(restart_audit, indent=2) + "\n", encoding="utf-8")
    metadata_flagged = sorted(m for m in METHODS if METHOD_META[m].get("batch_dependent"))
    overrides = sorted(BATCH_DEPENDENT)
    batch_audit = {
        "metadata_flagged": metadata_flagged,
        "analysis_override_for_underflagged_metadata": overrides,
        "all_batch_dependent_scoring_methods": sorted(set(metadata_flagged) | BATCH_DEPENDENT),
        "count": len(set(metadata_flagged) | BATCH_DEPENDENT),
        "note": "LMDD, DIF, CD, and ROD are marked inductive in the active model_factory metadata, but source-level scoring review found dependence on the query batch. The active model factory is unchanged; analysis flags correct reporting only.",
    }
    if len(metadata_flagged) != 8 or len(batch_audit["all_batch_dependent_scoring_methods"]) != 12:
        raise ValueError("batch scoring method inventory changed; revisit audit")
    (args.out_dir / "batch_scoring_audit.json").write_text(
        json.dumps(batch_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
