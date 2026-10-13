"""Audit the 5/10% normal-only runs and compare all six retained sizes.

The original 100/75/50/25% JSONL files and published output tables are read
only. Rankings use a fixed field of methods with 85 time-qualified successes
at *every* size in a representation. Panel-specific fields are exported only
as a sensitivity analysis. No failed or incomplete score is imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT.parents[1]
FULL = PROJECT / "full_retention"
DEFAULT_OLD_RAW = FULL / "raw"
DEFAULT_NEW_RAW = HERE / "raw"
DEFAULT_DATASET_MANIFEST = PROJECT / "dataset_manifest.json"
DEFAULT_OLD_OUTPUTS = WORKSPACE / "outputs"
OLD_FRACTIONS = (100, 75, 50, 25)
NEW_FRACTIONS = (10, 5)
FRACTIONS = (*OLD_FRACTIONS, *NEW_FRACTIONS)
REPS = ("native", "pca2")
SEEDS = (42, 43, 44, 45, 46)
TIME_BUDGET = 15.0
BOOTSTRAP_REPS = 5000
RANK_TOLERANCE = 1e-12

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(FULL))
from model_factory import METHODS, METHOD_META, NSA_METHODS  # noqa: E402

_spec = importlib.util.spec_from_file_location("_historical_retention_analyze", FULL / "analyze.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load the frozen full-retention validator")
historic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(historic)
BATCH_SCORERS = {m for m in METHODS if METHOD_META[m].get("batch_dependent")} | historic.BATCH_DEPENDENT
if len(BATCH_SCORERS) != 12:
    raise ValueError("batch-dependent scorer inventory changed; review source semantics")

STATUS_CAUSE = {
    "requires_labeled_outliers": "labeled anomalies required by this protocol",
    "requires_1d": "one-dimensional input required",
    "requires_2d": "two-dimensional input required",
    "requires_2d_or_3d": "two- or three-dimensional input required",
    "build_error": "model construction error",
    "error": "fit or scoring error",
    "timeout": "reported runtime timeout",
    "overbudget_success": "measured success beyond 15 seconds",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def macro(rows: list[dict], metric: str) -> float | None:
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(float(row[metric]))
    return mean(mean(by_dataset[dataset]) for dataset in sorted(by_dataset))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical_rows(raw_dir: Path, manifest: Path, old_outputs: Path) -> tuple[dict, list[str], dict]:
    """Use the frozen validator and published package hashes before combining rows."""
    package = json.loads((old_outputs / "pyod_nsa_full_retention_package_audit.json").read_text())
    analysis = json.loads((old_outputs / "pyod_nsa_full_retention_analysis_audit.json").read_text())
    members = package["members_sha256"]
    files = sorted(raw_dir.glob("*.jsonl"))
    if len(files) != 680 or analysis["expected_records"] != 38080:
        raise ValueError("frozen four-size grid is incomplete")
    for path in files:
        if sha(path) != members.get(f"production/raw/{path.name}"):
            raise ValueError(f"frozen raw file differs from published package: {path.name}")
    if (sha(old_outputs / "pyod_nsa_full_retention_analysis_audit.json")
            != members.get("analysis/full_retention_analysis_audit.json")):
        raise ValueError("published four-size analysis audit differs from package")
    index, names = historic.load_validate(raw_dir, manifest, SEEDS)
    excluded = historic.apply_measured_time_budget(index, TIME_BUDGET)
    if (len(index) != 38080 or len(excluded) != analysis["overbudget_success_records"]
            or sum(r["status"] == "success" for r in index.values())
                != analysis["success_records"]
            or names != analysis["datasets"]):
        raise ValueError("reconstructed historical outcomes differ from frozen audit")
    return index, names, {"raw_files": len(files), "records": len(index),
                          "overbudget_successes": len(excluded),
                          "package_audit_sha256": sha(old_outputs / "pyod_nsa_full_retention_package_audit.json"),
                          "analysis_audit_sha256": sha(old_outputs / "pyod_nsa_full_retention_analysis_audit.json")}


def new_rows(raw_dir: Path, manifest: Path, historical: dict,
             names: list[str]) -> tuple[dict, dict]:
    """Reuse the exact split/row validator with only its fraction grid swapped."""
    original = historic.FRACTIONS
    try:
        historic.FRACTIONS = NEW_FRACTIONS
        index, new_names = historic.load_validate(raw_dir, manifest, SEEDS)
    finally:
        historic.FRACTIONS = original
    if new_names != names or len(index) != 17 * 5 * 2 * 2 * len(METHODS):
        raise ValueError("new fraction grid or dataset names differ from historical grid")
    for key, row in index.items():
        if row["_source_line"] != METHODS.index(key[-1]) + 1:
            raise ValueError(f"new row departed from the fixed method order: {key}")
    for dataset in names:
        for seed in SEEDS:
            for rep in REPS:
                old = historical[(dataset, seed, rep, 100, METHODS[0])]
                for pct in NEW_FRACTIONS:
                    row = index[(dataset, seed, rep, pct, METHODS[0])]
                    keep = max(2, math.floor(old["fit_normals"] * pct / 100))
                    if (row["fit_normals"] != keep
                            or row["original_fit_normals"] != old["original_fit_normals"]
                            or row["fit_parent_index_sha256"] != old["fit_parent_index_sha256"]
                            or row["test_index_sha256"] != old["test_index_sha256"]
                            or row["test_normals"] != old["test_normals"]
                            or row["test_outliers"] != old["test_outliers"]
                            or row["fit_outliers"] != 0):
                        raise ValueError(f"new row does not share historical parent/test: {dataset}/{seed}/{rep}/{pct}")
                if index[(dataset, seed, rep, 5, METHODS[0])]["fit_normals"] > \
                        index[(dataset, seed, rep, 10, METHODS[0])]["fit_normals"]:
                    raise ValueError("5% train size exceeds 10% train size")
    files = sorted(raw_dir.glob("*.jsonl"))
    return index, {"raw_files": len(files), "records": len(index),
                   "raw_file_sha256": {path.name: sha(path) for path in files}}


def audit_new_recovery(root: Path, raw_dir: Path, index: dict) -> dict:
    """Match any new-root watchdog intervention to its no-score timeout row."""
    event_root = root / "watchdog" / "interventions"
    event_paths = sorted(event_root.glob("*/intervention.json"))
    events = []
    seen = set()
    for path in event_paths:
        event = json.loads(path.read_text())
        eid = event.get("intervention_id")
        cell = event.get("cell", {})
        method = cell.get("method")
        pct = round(100 * float(cell.get("retention_fraction", -1)))
        key = (cell.get("dataset"), cell.get("seed"), cell.get("representation"), pct, method)
        if (event.get("schema_version") != 1 or event.get("state") != "completed"
                or eid != path.parent.name or eid in seen or method not in METHODS
                or pct not in NEW_FRACTIONS or key not in index):
            raise ValueError(f"invalid new recovery intervention: {path}")
        seen.add(eid)
        row = index[key]
        marker = row.get("external_watchdog")
        if (row["status"] != "timeout" or row.get("auroc") is not None
                or row.get("ap") is not None
                or row.get("fit_seconds") is not None
                or row.get("score_seconds") is not None
                or not str(row.get("error", "")).startswith("ExternalWatchdogTimeout:")
                or not isinstance(marker, dict) or marker.get("intervention_id") != eid
                or marker.get("audit_path") != str(path.relative_to(root))
                or marker.get("target_pid") != event.get("target", {}).get("pid")):
            raise ValueError(f"new recovery row has no matching no-score marker: {key}")
        before = event["before"]
        after = event["after"]
        raw_snapshot = path.parent / before["raw_snapshot"]
        log_snapshot = path.parent / before["log_snapshot"]
        if (float(before["raw_age_seconds"]) < 600
                or before["raw_record_count"] != METHODS.index(method)
                or after["raw_record_count"] != before["raw_record_count"] + 1
                or after["appended_method"] != method or after["appended_status"] != "timeout"
                or event.get("action", {}).get("signal") != "SIGTERM"
                or event["action"].get("termination_confirmed") is not True
                or event["action"].get("primary_method_timeout_seconds") != TIME_BUDGET
                or not raw_snapshot.is_file() or not log_snapshot.is_file()
                or sha(raw_snapshot) != before["raw_sha256"]
                or sha(log_snapshot) != before["log_sha256"]):
            raise ValueError(f"new recovery snapshot/action mismatch: {path}")
        raw_before = raw_snapshot.read_bytes()
        final_raw = (raw_dir / row["_source_file"]).read_bytes()
        final_log = (root / "logs" / f"{Path(row['_source_file']).stem}.log").read_bytes()
        if (not final_log.startswith(log_snapshot.read_bytes())
                or len(raw_before.splitlines()) != before["raw_record_count"]
                or row["_source_line"] != before["raw_record_count"] + 1):
            raise ValueError(f"new recovery row order/log prefix mismatch: {path}")
        appended = final_raw.splitlines(keepends=True)[row["_source_line"] - 1]
        if (not final_raw.startswith(raw_before + appended)
                or sha_bytes(appended) != event["appended_row_sha256"]
                or sha_bytes(raw_before + appended) != after["raw_sha256"]):
            raise ValueError(f"new recovery appended-row hash mismatch: {path}")
        reason = marker.get("reason")
        if (reason not in {"ignored_signal_callback", "stalled_without_callback"}
                or reason != before.get("evidence_kind")):
            raise ValueError(f"new recovery reason mismatch: {path}")
        if reason == "ignored_signal_callback" and b"FitTimeout" not in log_snapshot.read_bytes():
            raise ValueError(f"new callback recovery lacks signal evidence: {path}")
        events.append({"intervention_id": eid, "dataset": key[0], "seed": key[1],
                       "representation": key[2], "retention_pct": pct,
                       "method": method, "reason": reason,
                       "raw_age_seconds": before["raw_age_seconds"],
                       "evidence_path": str(path.relative_to(root))})
    marked = {key for key, row in index.items()
              if isinstance(row.get("external_watchdog"), dict)
              or str(row.get("error", "")).startswith("ExternalWatchdogTimeout:")}
    if marked != {(e["dataset"], e["seed"], e["representation"], e["retention_pct"], e["method"])
                  for e in events}:
        raise ValueError("new recovery evidence and marked timeout rows differ")
    return {"new_event_count": len(events), "events": events,
            "historical_events_not_reused": True}


def method_panels(index: dict, names: list[str]) -> tuple[list[dict], dict]:
    summary = []
    by_panel = {}
    for rep in REPS:
        for pct in FRACTIONS:
            for method in METHODS:
                rows = [index[(dataset, seed, rep, pct, method)]
                        for dataset in names for seed in SEEDS]
                qualified = [r for r in rows if r["status"] == "success"]
                recorded = [r for r in rows if r["_raw_status"] == "success"]
                counts = Counter(r["status"] for r in rows)
                unsuitable = 85 - len(qualified)
                if unsuitable == 0:
                    suitability = "fully operational under tested configuration (85/85)"
                elif len(qualified) == 0:
                    suitability = "unsuitable under tested configuration (0/85 qualified)"
                else:
                    suitability = f"unsuitable in {unsuitable}/85 tested cases"
                causes = "; ".join(f"{STATUS_CAUSE.get(status, status)}: {count}"
                                     for status, count in sorted(counts.items())
                                     if status != "success") or "none"
                meta = METHOD_META[method]
                record = {
                    "representation": rep, "retention_pct": pct,
                    "reduction_pct": 100 - pct, "method": method,
                    "family": meta["family"], "strategy": meta.get("strategy", ""),
                    "applicability": meta["applicability"],
                    "expected_seed_rows": 85,
                    "qualified_success_seed_rows": len(qualified),
                    "raw_success_seed_rows": len(recorded),
                    "overbudget_success_seed_rows": len(recorded) - len(qualified),
                    "qualified_success_datasets": len({r["dataset"] for r in qualified}),
                    "full_qualified_coverage": len(qualified) == 85,
                    "full_raw_coverage": len(recorded) == 85,
                    "unsuitable_cases": unsuitable,
                    "suitability_label": suitability,
                    "unsuitability_causes": causes,
                    "macro_auroc_qualified_available": macro(qualified, "auroc"),
                    "macro_ap_qualified_available": macro(qualified, "ap"),
                    "macro_auroc_raw_available": macro(recorded, "auroc"),
                    "macro_ap_raw_available": macro(recorded, "ap"),
                    "status_counts_qualified": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                    "source_panel": "frozen_historical" if pct in OLD_FRACTIONS else "new_extreme",
                }
                summary.append(record)
                by_panel[(rep, pct, method)] = record
    if len(summary) != 2 * 6 * 56:
        raise AssertionError("method panel table incomplete")
    return summary, by_panel


def common_fields(panel: dict) -> dict:
    fields = {}
    for rep in REPS:
        all_common = [method for method in METHODS
                      if all(panel[(rep, pct, method)]["full_qualified_coverage"] for pct in FRACTIONS)]
        base_common = [method for method in all_common if method not in NSA_METHODS]
        inductive = [method for method in all_common if method not in BATCH_SCORERS]
        inductive_baselines = [method for method in base_common if method not in BATCH_SCORERS]
        fields[rep] = {"all": all_common, "baseline": base_common,
                       "nsa": [method for method in all_common if method in NSA_METHODS],
                       "inductive_all": inductive,
                       "inductive_baseline": inductive_baselines}
    return fields


def raw_common_fields(panel: dict) -> dict:
    fields = {}
    for rep in REPS:
        all_common = [method for method in METHODS
                      if all(panel[(rep, pct, method)]["full_raw_coverage"] for pct in FRACTIONS)]
        fields[rep] = {"all": all_common,
                       "baseline": [method for method in all_common if method not in NSA_METHODS],
                       "nsa": [method for method in all_common if method in NSA_METHODS]}
    return fields


def rank(value: float, values: list[float]) -> int:
    return 1 + sum(score > value + RANK_TOLERANCE for score in values)


def ranking_rows(panel: dict, fields: dict) -> tuple[list[dict], list[dict], list[dict]]:
    common, changing, changes = [], [], []
    for rep in REPS:
        for pct in FRACTIONS:
            for field_name, roster in (("all_common", fields[rep]["all"]),
                                       ("common_baselines", fields[rep]["baseline"]),
                                       ("inductive_all_common", fields[rep]["inductive_all"]),
                                       ("inductive_common_baselines", fields[rep]["inductive_baseline"])):
                for metric in ("auroc", "ap"):
                    scores = {m: panel[(rep, pct, m)][f"macro_{metric}_qualified_available"]
                              for m in roster}
                    for method in roster:
                        common.append({"representation": rep, "retention_pct": pct,
                                       "metric": metric, "field": field_name,
                                       "method": method, "family": METHOD_META[method]["family"],
                                       "score": scores[method], "rank": rank(scores[method], list(scores.values())),
                                       "field_size": len(roster), "field_fixed_across_six": True,
                                       "focal_eligible_all_six": True})
                    if field_name in ("common_baselines", "inductive_common_baselines"):
                        for nsa in (m for m in NSA_METHODS
                                    if panel[(rep, pct, m)]["full_qualified_coverage"]):
                            score = panel[(rep, pct, nsa)][f"macro_{metric}_qualified_available"]
                            common.append({"representation": rep, "retention_pct": pct,
                                           "metric": metric,
                                           "field": ("common_baselines_plus_focal_nsa" if field_name == "common_baselines"
                                                     else "inductive_common_baselines_plus_focal_nsa"),
                                           "method": nsa, "family": "NSA", "score": score,
                                           "rank": rank(score, [score, *scores.values()]),
                                           "field_size": len(roster) + 1,
                                           "field_fixed_across_six": True,
                                           "focal_eligible_all_six": nsa in fields[rep]["nsa"]})
            available = [m for m in METHODS if panel[(rep, pct, m)]["full_qualified_coverage"]]
            available_baselines = [m for m in available if m not in NSA_METHODS]
            for field_name, roster in (("all_panel", available),
                                       ("panel_baselines", available_baselines)):
                for metric in ("auroc", "ap"):
                    scores = {m: panel[(rep, pct, m)][f"macro_{metric}_qualified_available"]
                              for m in roster}
                    for method in roster:
                        changing.append({"representation": rep, "retention_pct": pct,
                                         "metric": metric, "field": field_name,
                                         "method": method, "family": METHOD_META[method]["family"],
                                         "score": scores[method], "rank": rank(scores[method], list(scores.values())),
                                         "field_size": len(roster), "field_fixed_across_six": False,
                                         "focal_eligible_all_six": method in fields[rep]["all"]})
                    if field_name == "panel_baselines":
                        for nsa in (m for m in available if m in NSA_METHODS):
                            score = panel[(rep, pct, nsa)][f"macro_{metric}_qualified_available"]
                            changing.append({"representation": rep, "retention_pct": pct,
                                             "metric": metric, "field": "panel_baselines_plus_focal_nsa",
                                             "method": nsa, "family": "NSA", "score": score,
                                             "rank": rank(score, [score, *scores.values()]),
                                             "field_size": len(roster) + 1,
                                             "field_fixed_across_six": False,
                                             "focal_eligible_all_six": nsa in fields[rep]["nsa"]})
    common_index = {(r["representation"], r["retention_pct"], r["metric"], r["field"], r["method"]): r
                    for r in common}
    for row in common:
        if row["retention_pct"] == 100 or not row["focal_eligible_all_six"]:
            continue
        full = common_index[(row["representation"], 100, row["metric"], row["field"], row["method"])]
        changes.append({"representation": row["representation"], "retention_pct": row["retention_pct"],
                        "metric": row["metric"], "field": row["field"], "method": row["method"],
                        "rank_100": full["rank"], "rank_reduced": row["rank"],
                        "rank_improvement": full["rank"] - row["rank"],
                        "score_100": full["score"], "score_reduced": row["score"],
                        "score_change": row["score"] - full["score"],
                        "field_size": row["field_size"]})
    return common, changing, changes


def raw_common_rankings(panel: dict, raw_fields: dict) -> list[dict]:
    """Six-size raw-success field, explicitly separate from strict-budget ranks."""
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            for metric in ("auroc", "ap"):
                all_scores = {m: panel[(rep, pct, m)][f"macro_{metric}_raw_available"]
                              for m in raw_fields[rep]["all"]}
                baseline_scores = {m: all_scores[m] for m in raw_fields[rep]["baseline"]}
                for field_name, scores in (("raw_all_common", all_scores),
                                           ("raw_common_baselines", baseline_scores)):
                    for method, score in scores.items():
                        rows.append({"representation": rep, "retention_pct": pct,
                                     "metric": metric, "field": field_name,
                                     "method": method, "family": METHOD_META[method]["family"],
                                     "score": score, "rank": rank(score, list(scores.values())),
                                     "field_size": len(scores), "focal_eligible_all_six": True})
                for nsa in NSA_METHODS:
                    if not panel[(rep, pct, nsa)]["full_raw_coverage"]:
                        continue
                    score = panel[(rep, pct, nsa)][f"macro_{metric}_raw_available"]
                    rows.append({"representation": rep, "retention_pct": pct,
                                 "metric": metric, "field": "raw_common_baselines_plus_focal_nsa",
                                 "method": nsa, "family": "NSA", "score": score,
                                 "rank": rank(score, [score, *baseline_scores.values()]),
                                 "field_size": len(baseline_scores) + 1,
                                 "focal_eligible_all_six": nsa in raw_fields[rep]["nsa"]})
    return rows


def paired_change(index: dict, names: list[str], reps: int,
                  bootstrap_seed: int) -> list[dict]:
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            if pct == 100:
                continue
            for method in METHODS:
                pairs = []
                statuses_reduced, statuses_full = Counter(), Counter()
                for dataset in names:
                    for seed in SEEDS:
                        reduced = index[(dataset, seed, rep, pct, method)]
                        full = index[(dataset, seed, rep, 100, method)]
                        statuses_reduced[reduced["status"]] += 1
                        statuses_full[full["status"]] += 1
                        if reduced["status"] == full["status"] == "success":
                            pairs.append((dataset, seed,
                                          reduced["auroc"] - full["auroc"],
                                          reduced["ap"] - full["ap"]))
                stats = historic.paired_stats(
                    pairs, f"{bootstrap_seed}|sixchange|{rep}|{pct}|{method}", reps)
                rows.append({"representation": rep, "retention_pct": pct,
                             "reduction_pct": 100 - pct, "method": method,
                             "family": METHOD_META[method]["family"],
                             "expected_pairs": 85, "full_coverage": len(pairs) == 85,
                             "reduced_status_counts": json.dumps(dict(sorted(statuses_reduced.items()))),
                             "full_status_counts": json.dumps(dict(sorted(statuses_full.items()))),
                             **stats})
    if len(rows) != 2 * 5 * 56:
        raise AssertionError("paired change table incomplete")
    return rows


def same_size_pairs(index: dict, names: list[str], old_outputs: Path,
                    reps: int, bootstrap_seed: int) -> list[dict]:
    """Preserve the published four-size pair table and append 5/10% pairs."""
    old_file = old_outputs / "pyod_nsa_full_retention_nsa_vs_baseline.csv"
    package = json.loads((old_outputs / "pyod_nsa_full_retention_package_audit.json").read_text())
    if sha(old_file) != package["members_sha256"].get("analysis/full_retention_nsa_vs_baseline.csv"):
        raise ValueError("published same-size pair table changed")
    with old_file.open(newline="", encoding="utf-8") as stream:
        old_rows = list(csv.DictReader(stream))
    if len(old_rows) != 2 * 4 * len(NSA_METHODS) * (len(METHODS) - len(NSA_METHODS)):
        raise ValueError("published four-size same-size pair grid incomplete")
    new = []
    baselines = [method for method in METHODS if method not in NSA_METHODS]
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            for nsa in NSA_METHODS:
                for baseline in baselines:
                    pairs = []
                    for dataset in names:
                        for seed in SEEDS:
                            a = index[(dataset, seed, rep, pct, nsa)]
                            b = index[(dataset, seed, rep, pct, baseline)]
                            if a["status"] == b["status"] == "success":
                                pairs.append((dataset, seed,
                                              a["auroc"] - b["auroc"],
                                              a["ap"] - b["ap"]))
                    new.append({"representation": rep, "retention_pct": pct,
                                "reduction_pct": 100 - pct, "nsa": nsa,
                                "baseline": baseline,
                                "baseline_batch_dependent": baseline in BATCH_SCORERS,
                                "expected_pairs": 85, "full_coverage": len(pairs) == 85,
                                **historic.paired_stats(
                                    pairs, f"{bootstrap_seed}|same|{rep}|{pct}|{nsa}|{baseline}", reps)})
    if len(new) != 2 * 2 * 13 * 43:
        raise AssertionError("new same-size pair table incomplete")
    old_keys = {(r["representation"], int(r["retention_pct"]), r["nsa"], r["baseline"])
                for r in old_rows}
    if len(old_keys) != len(old_rows):
        raise ValueError("published pair table contains duplicate key")
    rows = old_rows + new
    if len(rows) != 2 * 6 * 13 * 43:
        raise AssertionError("combined same-size pair table incomplete")
    return rows


def relative_change_vs_100(index: dict, names: list[str], reps: int,
                           bootstrap_seed: int) -> list[dict]:
    """Four-success matched change in the NSA-minus-baseline gap.

    The baseline method is held fixed across sizes. Positive means NSA gains
    relative to that baseline as the normal fit pool is reduced. This is a
    difference of differences, not a new fit or an independent comparison.
    """
    rows = []
    baselines = [method for method in METHODS if method not in NSA_METHODS]
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            for nsa in NSA_METHODS:
                for baseline in baselines:
                    pairs = []
                    for dataset in names:
                        for seed in SEEDS:
                            a_small = index[(dataset, seed, rep, pct, nsa)]
                            b_small = index[(dataset, seed, rep, pct, baseline)]
                            a_full = index[(dataset, seed, rep, 100, nsa)]
                            b_full = index[(dataset, seed, rep, 100, baseline)]
                            if all(r["status"] == "success" for r in
                                   (a_small, b_small, a_full, b_full)):
                                pairs.append((dataset, seed,
                                              (a_small["auroc"] - b_small["auroc"])
                                              - (a_full["auroc"] - b_full["auroc"]),
                                              (a_small["ap"] - b_small["ap"])
                                              - (a_full["ap"] - b_full["ap"])))
                    ap_by_dataset = defaultdict(list)
                    for dataset, _, _, ap_delta in pairs:
                        ap_by_dataset[dataset].append(ap_delta)
                    ap_dataset_means = [float(np.mean(values))
                                        for _, values in sorted(ap_by_dataset.items())]
                    rows.append({
                        "representation": rep, "retention_pct": pct,
                        "reduction_pct": 100 - pct, "nsa": nsa, "baseline": baseline,
                        "baseline_batch_dependent": baseline in BATCH_SCORERS,
                        "expected_four_success_pairs": 85,
                        "full_coverage": len(pairs) == 85,
                        "definition": "(NSA_reduced-baseline_reduced)-(NSA_100-baseline_100)",
                        "ap_gain_datasets": sum(value > RANK_TOLERANCE for value in ap_dataset_means),
                        "ap_tie_datasets": sum(abs(value) <= RANK_TOLERANCE for value in ap_dataset_means),
                        "ap_loss_datasets": sum(value < -RANK_TOLERANCE for value in ap_dataset_means),
                        **historic.paired_stats(
                            pairs,
                            f"{bootstrap_seed}|relativechange|{rep}|{pct}|{nsa}|{baseline}",
                            reps),
                    })
    if len(rows) != 2 * 2 * len(NSA_METHODS) * len(baselines):
        raise AssertionError("new-size relative-change grid incomplete")
    return rows


def dataset_nsa_ranks(index: dict, names: list[str], fields: dict) -> list[dict]:
    """Show every NSA configuration, including incomplete and inapplicable ones."""
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            for dataset in names:
                by_method = {}
                for method in METHODS:
                    seed_rows = [index[(dataset, seed, rep, pct, method)] for seed in SEEDS]
                    qualified = [r for r in seed_rows if r["status"] == "success"]
                    by_method[method] = (seed_rows, qualified)
                local = [m for m in METHODS if len(by_method[m][1]) == 5]
                for nsa in NSA_METHODS:
                    seed_rows, qualified = by_method[nsa]
                    full = len(qualified) == 5
                    status_counts = json.dumps(dict(sorted(Counter(r["status"] for r in seed_rows).items())))
                    record = {
                        "representation": rep, "retention_pct": pct, "dataset": dataset,
                        "method": nsa, "strategy": METHOD_META[nsa].get("strategy", ""),
                        "qualified_seed_rows": len(qualified),
                        "raw_success_seed_rows": sum(r["_raw_status"] == "success" for r in seed_rows),
                        "status_counts": status_counts, "full_dataset_coverage": full,
                        "mean_auroc_five_seeds": mean(r["auroc"] for r in qualified) if full else None,
                        "mean_ap_five_seeds": mean(r["ap"] for r in qualified) if full else None,
                        "rank_fixed_all_common": None,
                        "rank_fixed_all_common_ap": None,
                        "fixed_all_common_field_size": len(fields[rep]["all"]),
                        "rank_fixed_baselines_plus_focal": None,
                        "rank_fixed_baselines_plus_focal_ap": None,
                        "fixed_baselines_plus_focal_size": len(fields[rep]["baseline"]) + 1,
                        "rank_fixed_inductive_baselines_plus_focal": None,
                        "rank_fixed_inductive_baselines_plus_focal_ap": None,
                        "fixed_inductive_baselines_plus_focal_size": len(fields[rep]["inductive_baseline"]) + 1,
                        "rank_local_panel_qualified": None,
                        "rank_local_panel_qualified_ap": None,
                        "local_panel_qualified_field_size": len(local),
                        "rank_rule": "competition rank, 1e-12 metric tie tolerance; blank when <5 successful seeds",
                    }
                    if full:
                        for metric, suffix in (("auroc", ""), ("ap", "_ap")):
                            score = record[f"mean_{metric}_five_seeds"]
                            def scores(methods):
                                return [mean(r[metric] for r in by_method[m][1]) for m in methods]
                            if nsa in fields[rep]["all"]:
                                record[f"rank_fixed_all_common{suffix}"] = rank(
                                    score, scores(fields[rep]["all"]))
                            record[f"rank_fixed_baselines_plus_focal{suffix}"] = rank(
                                score, [score, *scores(fields[rep]["baseline"])])
                            record[f"rank_fixed_inductive_baselines_plus_focal{suffix}"] = rank(
                                score, [score, *scores(fields[rep]["inductive_baseline"])])
                            record[f"rank_local_panel_qualified{suffix}"] = rank(score, scores(local))
                    rows.append(record)
    if len(rows) != 2 * 6 * 17 * 13:
        raise AssertionError("dataset NSA strategy table incomplete")
    return rows


def dataset_inventory(index: dict, manifest_path: Path, names: list[str]) -> list[dict]:
    source = json.loads(manifest_path.read_text())
    items = {item["name"]: item for item in source["datasets"]}
    result = []
    for dataset in names:
        item = items[dataset]
        row = {"dataset": dataset, "source_rows": item["n_samples"],
               "source_features": item["n_features"],
               "source_normals": item["n_normal"],
               "source_anomalies": item["n_anomaly"],
               "fit_seed_count": len(SEEDS)}
        for pct in FRACTIONS:
            counts = []
            for seed in SEEDS:
                normal_native = index[(dataset, seed, "native", pct, METHODS[0])]["fit_normals"]
                normal_pca = index[(dataset, seed, "pca2", pct, METHODS[0])]["fit_normals"]
                if normal_native != normal_pca:
                    raise ValueError(f"representation fit size differs: {dataset}/{seed}/{pct}")
                counts.append(normal_native)
            row[f"fit_normals_{pct}_min"] = min(counts)
            row[f"fit_normals_{pct}_max"] = max(counts)
        result.append(row)
    if len(result) != 17:
        raise AssertionError("dataset inventory incomplete")
    return result


def raw_sensitivity(summary: list[dict]) -> list[dict]:
    rows = []
    for r in summary:
        auc_raw = r["macro_auroc_raw_available"]
        auc_strict = r["macro_auroc_qualified_available"]
        ap_raw = r["macro_ap_raw_available"]
        ap_strict = r["macro_ap_qualified_available"]
        rows.append({"representation": r["representation"], "retention_pct": r["retention_pct"],
                     "method": r["method"], "expected_seed_rows": 85,
                     "raw_success_seed_rows": r["raw_success_seed_rows"],
                     "strict_success_seed_rows": r["qualified_success_seed_rows"],
                     "overbudget_success_seed_rows": r["overbudget_success_seed_rows"],
                     "raw_macro_auroc_available": auc_raw,
                     "strict_macro_auroc_available": auc_strict,
                     "strict_minus_raw_auroc": auc_strict - auc_raw if auc_raw is not None and auc_strict is not None else None,
                     "raw_macro_ap_available": ap_raw,
                     "strict_macro_ap_available": ap_strict,
                     "strict_minus_raw_ap": ap_strict - ap_raw if ap_raw is not None and ap_strict is not None else None,
                     "interpretation": "descriptive available-case sensitivity; successful cell sets can differ"})
    return rows


def focal_metrics(tables: dict[str, list[dict]], fields: dict, raw_fields: dict) -> dict:
    """Compact, audited factual index for the private manuscript components.

    A leader is selected separately for AUROC and AP from the fixed six-size
    baseline field. This is descriptive selection on the observed outcomes.
    All contrasts retain matched coverage and uncertainty, including blanks.
    """
    summary = {(r["representation"], int(r["retention_pct"]), r["method"]): r
               for r in tables["method_summary"]}
    ranks = {(r["representation"], int(r["retention_pct"]), r["metric"],
              r["field"], r["method"]): r for r in tables["common_field_ranks"]}
    raw_ranks = {(r["representation"], int(r["retention_pct"]), r["metric"],
                  r["field"], r["method"]): r for r in tables["raw_common_ranks"]}
    changes = {(r["representation"], int(r["retention_pct"]), r["method"]): r
               for r in tables["change_vs_100"]}
    pairs = {(r["representation"], int(r["retention_pct"]), r["nsa"], r["baseline"]): r
             for r in tables["nsa_vs_baseline"]}
    relative = {(r["representation"], int(r["retention_pct"]), r["nsa"], r["baseline"]): r
                for r in tables["relative_change_vs_100"]}

    def num(value):
        return None if value is None or value == "" else float(value)

    leaders_100 = {}
    for rep in REPS:
        for metric in ("auroc", "ap"):
            candidates = [r for r in tables["common_field_ranks"]
                          if r["representation"] == rep and int(r["retention_pct"]) == 100
                          and r["metric"] == metric and r["field"] == "common_baselines"]
            top = max(candidates, key=lambda r: float(r["score"])) if candidates else None
            leaders_100[(rep, metric)] = top["method"] if top else None
    panels = []
    for rep in REPS:
        for pct in FRACTIONS:
            leaders = {}
            for metric in ("auroc", "ap"):
                candidates = [r for r in tables["common_field_ranks"]
                              if r["representation"] == rep and int(r["retention_pct"]) == pct
                              and r["metric"] == metric and r["field"] == "common_baselines"]
                top = max(candidates, key=lambda r: float(r["score"])) if candidates else None
                leaders[metric] = None if top is None else {
                    "method": top["method"], "score": num(top["score"]),
                    "field_size": int(top["field_size"]),
                }
            focal = []
            for method in NSA_METHODS:
                cell = summary[(rep, pct, method)]
                item = {
                    "method": method, "strategy": METHOD_META[method].get("strategy", ""),
                    "qualified_success_seed_rows": int(cell["qualified_success_seed_rows"]),
                    "raw_success_seed_rows": int(cell["raw_success_seed_rows"]),
                    "full_qualified_coverage": bool(cell["full_qualified_coverage"]),
                    "full_raw_coverage": bool(cell["full_raw_coverage"]),
                    "eligible_all_six_strict": method in fields[rep]["nsa"],
                    "eligible_all_six_raw": method in raw_fields[rep]["nsa"],
                    "metrics": {},
                }
                for metric in ("auroc", "ap"):
                    fixed = ranks.get((rep, pct, metric, "common_baselines_plus_focal_nsa", method))
                    fixed_inductive = ranks.get(
                        (rep, pct, metric, "inductive_common_baselines_plus_focal_nsa", method))
                    all_common = ranks.get((rep, pct, metric, "all_common", method))
                    raw = raw_ranks.get((rep, pct, metric, "raw_common_baselines_plus_focal_nsa", method))
                    own_change = changes.get((rep, pct, method))
                    comparator = leaders[metric]
                    paired = pairs.get((rep, pct, method, comparator["method"])) if comparator else None
                    reference_baseline = leaders_100[(rep, metric)]
                    interaction = relative.get((rep, pct, method, reference_baseline)) \
                        if reference_baseline else None
                    item["metrics"][metric] = {
                        "qualified_macro_score_available": num(cell[f"macro_{metric}_qualified_available"]),
                        "raw_macro_score_available": num(cell[f"macro_{metric}_raw_available"]),
                        "strict_fixed_baselines_plus_focal_rank": int(fixed["rank"]) if fixed else None,
                        "strict_fixed_baselines_plus_focal_field_size": int(fixed["field_size"]) if fixed else None,
                        "strict_fixed_inductive_baselines_plus_focal_rank": (
                            int(fixed_inductive["rank"]) if fixed_inductive else None),
                        "strict_fixed_inductive_baselines_plus_focal_field_size": (
                            int(fixed_inductive["field_size"]) if fixed_inductive else None),
                        "strict_all_common_rank": int(all_common["rank"]) if all_common else None,
                        "strict_all_common_field_size": int(all_common["field_size"]) if all_common else None,
                        "raw_fixed_baselines_plus_focal_rank": int(raw["rank"]) if raw else None,
                        "raw_fixed_baselines_plus_focal_field_size": int(raw["field_size"]) if raw else None,
                        "paired_vs_100_seed_rows": int(own_change["paired_seed_rows"]) if own_change else None,
                        "paired_vs_100_datasets": int(own_change["paired_datasets"]) if own_change else None,
                        "delta_vs_100": num(own_change[f"macro_mean_delta_{metric}"]) if own_change else None,
                        "delta_vs_100_ci95": ([num(own_change[f"cluster_bootstrap_95_low_{metric}"]),
                                               num(own_change[f"cluster_bootstrap_95_high_{metric}"])]
                                              if own_change else None),
                        "selected_baseline_method": comparator["method"] if comparator else None,
                        "paired_vs_selected_baseline_seed_rows": (
                            int(paired["paired_seed_rows"]) if paired else None),
                        "paired_vs_selected_baseline_datasets": (
                            int(paired["paired_datasets"]) if paired else None),
                        "delta_vs_selected_baseline": (
                            num(paired[f"macro_mean_delta_{metric}"]) if paired else None),
                        "delta_vs_selected_baseline_ci95": (
                            [num(paired[f"cluster_bootstrap_95_low_{metric}"]),
                             num(paired[f"cluster_bootstrap_95_high_{metric}"])] if paired else None),
                        "reference_baseline_100pct": reference_baseline,
                        "relative_change_vs_100_seed_rows": (
                            int(interaction["paired_seed_rows"]) if interaction else None),
                        "relative_change_vs_100_datasets": (
                            int(interaction["paired_datasets"]) if interaction else None),
                        "relative_change_vs_100": (
                            num(interaction[f"macro_mean_delta_{metric}"]) if interaction else None),
                        "relative_change_vs_100_ci95": (
                            [num(interaction[f"cluster_bootstrap_95_low_{metric}"]),
                             num(interaction[f"cluster_bootstrap_95_high_{metric}"])]
                            if interaction else None),
                    }
                focal.append(item)
            panels.append({"representation": rep, "retention_pct": pct,
                           "fixed_baseline_leaders": leaders, "nsa": focal})
    return {
        "schema_version": 1,
        "qualification": "recorded success and fit_seconds+score_seconds <=15.0; raw success is separate",
        "selection": "baseline leader selected descriptively by observed score, separately per metric and panel",
        "relative_change": "(NSA_reduced-baseline_reduced)-(NSA_100-baseline_100) on four-success matched cells; baseline selected at 100% and held fixed",
        "rank": "competition rank within six-size fixed baseline field plus one panel-complete focal NSA",
        "uncertainty": "unadjusted dataset-cluster bootstrap 95% CI; null endpoints if fewer than five paired datasets",
        "representations": list(REPS), "retention_percent": list(FRACTIONS),
        "fixed_fields": fields, "raw_fixed_fields": raw_fields,
        "panels": panels,
    }


def validate_new_run_manifest(path: Path, old_manifest: Path,
                              dataset_manifest: Path) -> dict:
    new = json.loads(path.read_text())
    old = json.loads(old_manifest.read_text())
    if (new.get("source_git_commit") != old["source_git_commit"]
            or new.get("source_git_tree") != old["source_git_tree"]
            or new.get("protocol_version") != historic.PROTOCOL
            or new.get("expected_cells") != 340
            or new.get("expected_records") != 19040
            or new.get("seeds") != list(SEEDS)
            or new.get("representations") != list(REPS)
            or new.get("fractions_retained") != [0.1, 0.05]
            or new.get("method_timeout_seconds") != TIME_BUDGET
            or new.get("historical_run_manifest_sha256") != sha(old_manifest)
            or new.get("methods") != METHODS):
        raise ValueError("extreme run manifest does not match the frozen source and 340-cell protocol")
    if ("dataset_manifest_sha256" in new
            and new["dataset_manifest_sha256"] != sha(dataset_manifest)):
        raise ValueError("extreme run dataset manifest hash changed")
    return {"path": str(path), "sha256": sha(path),
            "source_git_commit": new["source_git_commit"],
            "source_git_tree": new["source_git_tree"],
            "expected_cells": new["expected_cells"],
            "expected_records": new["expected_records"]}


def validate_supervision_manifest(root: Path, run_manifest: Path) -> dict:
    path = root / "supervision_manifest.json"
    manifest = json.loads(path.read_text())
    code = manifest.get("code_sha256", {})
    if (manifest.get("schema_version") != 1
            or manifest.get("primary_run_manifest_sha256") != sha(run_manifest)
            or manifest.get("minimum_raw_idle_seconds") != 600
            or manifest.get("minimum_log_idle_seconds") != 600
            or manifest.get("poll_seconds") != 30
            or set(code) != {"monitor.py", "watchdog_recover.py"}
            or any(code[name] != sha(root / name) for name in code)):
        raise ValueError("new-root watchdog supervision manifest changed")
    mode = manifest.get("active_monitor_mode")
    if mode not in {"read_only_detection_only", "execute_verified_recovery"}:
        raise ValueError("unknown extreme monitor action mode")
    return {"path": str(path), "sha256": sha(path),
            "minimum_raw_idle_seconds": 600, "minimum_log_idle_seconds": 600,
            "poll_seconds": 30, "code_sha256": code,
            "active_monitor_mode": mode,
            "active_recovery_permission": manifest.get("active_recovery_permission")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-raw", type=Path, default=DEFAULT_OLD_RAW)
    parser.add_argument("--new-root", type=Path, default=HERE)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--historical-outputs", type=Path, default=DEFAULT_OLD_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=HERE / "analysis")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.bootstrap_reps != BOOTSTRAP_REPS or args.bootstrap_seed != 20261012:
        parser.error("published historical contrasts require 5000 draws and seed 20261012")
    old_index, names, old_audit = historical_rows(
        args.historical_raw, args.dataset_manifest, args.historical_outputs)
    new_raw = args.new_root / "raw"
    new_index, new_audit = new_rows(new_raw, args.dataset_manifest, old_index, names)
    recovery = audit_new_recovery(args.new_root, new_raw, new_index)
    new_overbudget = historic.apply_measured_time_budget(new_index, TIME_BUDGET)
    index = {**old_index, **new_index}
    if len(index) != 17 * 5 * 2 * 6 * 56:
        raise ValueError("combined six-size grid has duplicate or missing keys")
    manifest_audit = validate_new_run_manifest(
        args.new_root / "run_manifest.json", FULL / "run_manifest.json",
        args.dataset_manifest)
    supervision_audit = validate_supervision_manifest(
        args.new_root, args.new_root / "run_manifest.json")
    summary, panel = method_panels(index, names)
    fields = common_fields(panel)
    raw_fields = raw_common_fields(panel)
    common_ranks, panel_ranks, rank_changes = ranking_rows(panel, fields)
    tables = {
        "method_summary": summary,
        "common_field_ranks": common_ranks,
        "raw_common_ranks": raw_common_rankings(panel, raw_fields),
        "panel_ranks": panel_ranks,
        "rank_change": rank_changes,
        "change_vs_100": paired_change(index, names, args.bootstrap_reps, args.bootstrap_seed),
        "nsa_vs_baseline": same_size_pairs(index, names, args.historical_outputs,
                                           args.bootstrap_reps, args.bootstrap_seed),
        "relative_change_vs_100": relative_change_vs_100(
            index, names, args.bootstrap_reps, args.bootstrap_seed),
        "dataset_nsa_ranks": dataset_nsa_ranks(index, names, fields),
        "dataset_inventory": dataset_inventory(index, args.dataset_manifest, names),
        "raw_vs_strict": raw_sensitivity(summary),
    }
    if len(tables["nsa_vs_baseline"]) != 6708:
        raise AssertionError("six-size NSA versus baseline table incomplete")
    if len(tables["relative_change_vs_100"]) != 2236:
        raise AssertionError("new-size relative-change table incomplete")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.out_dir / f"extreme_retention_{name}.csv", rows)
    paper = focal_metrics(tables, fields, raw_fields)
    paper_path = args.out_dir / "extreme_retention_focal_metrics.json"
    paper_path.write_text(json.dumps(paper, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raw_status = Counter(row["_raw_status"] for row in index.values())
    strict_status = Counter(row["status"] for row in index.values())
    audit = {
        "schema_version": 1,
        "source_git_commit": manifest_audit["source_git_commit"],
        "source_git_tree": manifest_audit["source_git_tree"],
        "protocol_version": historic.PROTOCOL,
        "dataset_manifest_sha256": sha(args.dataset_manifest),
        "datasets": names, "seeds": list(SEEDS),
        "representations": list(REPS),
        "retention_percent": list(FRACTIONS), "methods": METHODS,
        "expected_cells": 1020, "expected_records": len(index),
        "historical": old_audit, "new_extreme": new_audit,
        "new_run_manifest": manifest_audit,
        "new_supervision_manifest": supervision_audit,
        "raw_status_counts": dict(sorted(raw_status.items())),
        "strict_status_counts": dict(sorted(strict_status.items())),
        "raw_success_records": raw_status["success"],
        "strict_success_records": strict_status["success"],
        "historical_overbudget_success_records": old_audit["overbudget_successes"],
        "new_overbudget_success_records": len(new_overbudget),
        "total_overbudget_success_records": strict_status["overbudget_success"],
        "strict_measured_time_budget_seconds": TIME_BUDGET,
        "strict_time_rule": "raw success qualifies iff recorded fit_seconds + score_seconds <= 15.0; unqualified successes retain their scores in the raw files",
        "common_fields": fields,
        "raw_common_fields": raw_fields,
        "batch_dependent_scoring_methods": sorted(BATCH_SCORERS),
        "ranking_rule": "competition rank, 1 + count(other score > focal score + 1e-12)",
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "dataset cluster; interval blank for fewer than five paired datasets",
        "relative_change_definition": "(NSA_reduced-baseline_reduced)-(NSA_100-baseline_100); same baseline and dataset/seed in all four successful outcomes",
        "new_recovery": recovery,
        "historical_recovery_not_reused": True,
        "tables": {name: len(rows) for name, rows in tables.items()},
        "focal_metrics_sha256": sha(paper_path),
        "focal_metrics_panels": len(paper["panels"]),
        "scope": "17 ODDS datasets; five seeds; native/PCA2; six retained normal-pool sizes; 56 fixed configurations; no fitted outliers",
    }
    (args.out_dir / "extreme_retention_analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "extreme_retention_recovery_audit.json").write_text(
        json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"expected_records": audit["expected_records"],
                      "historical_records": old_audit["records"],
                      "new_records": new_audit["records"],
                      "strict_success_records": audit["strict_success_records"],
                      "new_recovery_events": recovery["new_event_count"],
                      "common_fields": {rep: {name: len(roster) for name, roster in f.items()}
                                        for rep, f in fields.items()},
                      "tables": audit["tables"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
