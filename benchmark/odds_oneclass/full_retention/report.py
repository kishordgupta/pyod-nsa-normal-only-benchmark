"""Generate a review-only report from audited full-roster retention tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_factory import METHODS, NSA_METHODS  # noqa: E402

REPS = ("native", "pca2")
FRACTIONS = (100, 75, 50, 25)
REDUCED = (75, 50, 25)
BASELINES = tuple(method for method in METHODS if method not in NSA_METHODS)
DISPLAY = {method: method.removeprefix("NSA/").replace("binary/", "binary · ")
           for method in NSA_METHODS}
RESTART_BOUNDARIES = {
    "glass_pca2_seed43_retain25": 36,
    "pendigits_pca2_seed43_retain25": 30,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"missing rows: {path}")
    return rows


def index_rows(rows: list[dict], fields: tuple[str, ...], expected: int, label: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(int(row[field]) if field == "retention_pct" else row[field]
                    for field in fields)
        if key in result:
            raise ValueError(f"duplicate {label}: {key}")
        result[key] = row
    if len(result) != expected:
        raise ValueError(f"{label} incomplete: expected {expected}, got {len(result)}")
    return result


def load(analysis_dir: Path) -> dict:
    audit = json.loads((analysis_dir / "full_retention_analysis_audit.json").read_text())
    batch_audit = json.loads((analysis_dir / "batch_scoring_audit.json").read_text())
    watchdog_audit = json.loads((analysis_dir / "watchdog_intervention_audit.json").read_text())
    restart_audit = json.loads((analysis_dir / "process_restart_audit.json").read_text())
    restart_events = []
    for cell, boundary in sorted(RESTART_BOUNDARIES.items()):
        sidecar_path = Path(__file__).parent / "provenance" / f"{cell}_restart.json"
        sidecar_bytes = sidecar_path.read_bytes()
        sidecar = json.loads(sidecar_bytes)
        if (sidecar.get("cell") != cell
                or sidecar.get("first_invocation", {}).get("completed_record_count") != boundary
                or sidecar.get("final", {}).get("record_count") != 56):
            raise ValueError(f"invalid process restart sidecar: {sidecar_path}")
        restart_events.append({
            "cell": cell,
            "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "first_completed_record_count": boundary,
            "final_record_count": 56,
        })
    run_manifest = json.loads((Path(__file__).parent / "run_manifest.json").read_text())
    if (audit["expected_records"] != 17 * 5 * 2 * 4 * 56
            or audit["raw_jsonl_files"] != 17 * 5 * 2 * 4
            or audit["seeds"] != [42, 43, 44, 45, 46]
            or audit["representations"] != list(REPS)
            or audit["retention_percent"] != list(FRACTIONS)
            or audit["methods"] != METHODS
            or audit.get("measured_time_budget_seconds") != 15.0
            or audit.get("raw_success_records") - audit.get("success_records")
                != audit.get("overbudget_success_records")
            or audit.get("qualified_status_counts") != audit.get("status_counts")
            or audit.get("qualified_status_counts", {}).get("overbudget_success", 0)
                != audit.get("overbudget_success_records")
            or audit.get("external_watchdog_intervention_count") != watchdog_audit.get("event_count")
            or audit.get("external_watchdog_loci_timeout_rows") != watchdog_audit.get("event_count")
            or watchdog_audit.get("all_snapshots_sha256_verified") is not True
            or watchdog_audit.get("all_final_loci_rows_matched") is not True
            or audit.get("process_restart_event_count") != len(RESTART_BOUNDARIES)
            or restart_audit.get("event_count") != len(RESTART_BOUNDARIES)
            or restart_audit.get("events") != restart_events
            or restart_audit.get("raw_and_log_hashes_verified") is not True
            ):
        raise ValueError("report requires the five-seed full-roster audit")
    if (run_manifest["source_git_commit"] != "0d6c901652f82c813da45875873ef4a8560cf049"
            or run_manifest["source_git_tree"] != "57ec5c9ea2800b559019adf0457d7e5b858fc153"
            or run_manifest["seeds"] != audit["seeds"]
            or run_manifest["representations"] != audit["representations"]
            or run_manifest["fractions_retained"] != [1.0, 0.75, 0.5, 0.25]
            or run_manifest["protocol_version"] != audit["protocol_version"]
            or run_manifest["method_timeout_seconds"] != 15.0
            or run_manifest["methods"] != audit["methods"]
            or run_manifest["expected_cells"] != audit["raw_jsonl_files"]
            or run_manifest["expected_records"] != audit["expected_records"]
            or run_manifest["dataset_manifest_sha256"] != audit["source_dataset_manifest_sha256"]):
        raise ValueError("run manifest does not match the audited full-roster grid")
    if batch_audit["count"] != 12 or set(batch_audit["analysis_override_for_underflagged_metadata"]) != {
            "LMDD", "DIF", "CD", "ROD"}:
        raise ValueError("batch scoring audit incomplete")
    paths = {name: analysis_dir / f"full_retention_{name}.csv" for name in (
        "method_summary", "change_vs_full", "dataset_changes",
        "nsa_vs_baseline", "interaction", "coverage_qualified_ranks", "leave_one_out",
        "dataset_inventory", "overbudget_success", "overbudget_method_counts",
        "recorded_vs_qualified", "partial_metric_timeouts")}
    raw = {name: read_csv(path) for name, path in paths.items()}
    expected = {
        "method_summary": 2*4*56,
        "change_vs_full": 2*3*56,
        "dataset_changes": 2*3*56*17,
        "nsa_vs_baseline": 2*4*13*43,
        "interaction": 2*3*13*43,
        "dataset_inventory": 17,
        "overbudget_success": audit["overbudget_success_records"],
        "overbudget_method_counts": 56,
        "recorded_vs_qualified": 448,
        "partial_metric_timeouts": audit["partial_metric_timeout_rows"],
    }
    for name, count in expected.items():
        if len(raw[name]) != count:
            raise ValueError(f"incomplete {name}: {len(raw[name])} vs {count}")
    if (sum(audit["raw_status_counts"].values()) != audit["expected_records"]
            or sum(audit["qualified_status_counts"].values()) != audit["expected_records"]
            or sum(int(r["overbudget_success_records"]) for r in raw["overbudget_method_counts"])
                != audit["overbudget_success_records"]):
        raise ValueError("raw/qualified status totals or overbudget method counts disagree")
    for row in raw["overbudget_success"]:
        elapsed = float(row["fit_seconds"]) + float(row["score_seconds"])
        if (row["raw_status"] != "success"
                or row["analysis_status"] != "overbudget_success"
                or elapsed <= 15.0
                or abs(elapsed - float(row["measured_end_to_end_seconds"])) > 1e-9):
            raise ValueError(f"invalid measured-time exclusion row: {row}")
    for row in raw["partial_metric_timeouts"]:
        if (row["status"] != "timeout" or row["ap"] not in ("", None)
                or row["interpretation"] != "timeout during metrics; not a scored AUROC/AP pair"
                or not 0 <= float(row["intermediate_auroc"]) <= 1 + 1e-12):
            raise ValueError(f"invalid partial timeout metric row: {row}")
    return {
        "audit": audit,
        "batch_audit": batch_audit,
        "watchdog_audit": watchdog_audit,
        "restart_audit": restart_audit,
        "run_manifest": run_manifest,
        "summary": index_rows(raw["method_summary"],
                              ("representation", "retention_pct", "method"), 448, "summary"),
        "change": index_rows(raw["change_vs_full"],
                             ("representation", "retention_pct", "method"), 336, "change"),
        "dataset": index_rows(raw["dataset_changes"],
                              ("representation", "retention_pct", "method", "dataset"),
                              2*3*56*17, "dataset changes"),
        "same": index_rows(raw["nsa_vs_baseline"],
                           ("representation", "retention_pct", "nsa", "baseline"),
                           2*4*13*43, "NSA/baseline"),
        "interaction": index_rows(raw["interaction"],
                                  ("representation", "retention_pct", "nsa", "baseline"),
                                  2*3*13*43, "interaction"),
        "ranks": raw["coverage_qualified_ranks"],
        "loo": raw["leave_one_out"],
        "inventory": raw["dataset_inventory"],
        "overbudget": raw["overbudget_success"],
        "overbudget_methods": raw["overbudget_method_counts"],
        "recorded_vs_qualified": raw["recorded_vs_qualified"],
        "partial_metric_timeouts": raw["partial_metric_timeouts"],
    }


def num(text: str | None) -> float | None:
    return None if text in (None, "") else float(text)


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    if abs(value) < 0.0005:
        return "0.000"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def table(headers: list[str], rows: list[list]) -> str:
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(escape, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(escape, row)) + " |" for row in rows),
    ])


def score(row: dict, metric: str) -> str:
    value = num(row[f"macro_{metric}_available"])
    n = int(row["success_seed_rows"])
    if (value is None) != (n == 0):
        raise ValueError(f"score/coverage mismatch: {row['method']}")
    marker = "†" if 0 < n < int(row["expected_seed_rows"]) else ""
    return f"{fmt(value)}{marker} ({n}/{row['expected_seed_rows']})"


def delta(row: dict, metric: str, interval: bool = True) -> str:
    value = num(row[f"macro_mean_delta_{metric}"])
    n = int(row["paired_seed_rows"])
    denom = row["expected_pairs"]
    if (value is None) != (n == 0):
        raise ValueError("paired delta/coverage mismatch")
    result = fmt(value, signed=True)
    if value is not None and interval:
        low = num(row[f"cluster_bootstrap_95_low_{metric}"])
        high = num(row[f"cluster_bootstrap_95_high_{metric}"])
        result += (" [interval not estimated]" if low is None or high is None
                   else f" [{fmt(low, True)}, {fmt(high, True)}]")
    return f"{result} ({n}/{denom})"


def method_score_table(data: dict, rep: str, methods: tuple[str, ...]) -> str:
    rows = []
    for method in methods:
        cells = [DISPLAY.get(method, method)]
        for pct in FRACTIONS:
            row = data["summary"][(rep, pct, method)]
            cells.append(f"{score(row, 'auroc')} / {score(row, 'ap')}")
        rows.append(cells)
    return table(["Method", *[f"{p}%: AUROC / AP" for p in FRACTIONS]], rows)


def method_change_table(data: dict, rep: str, methods: tuple[str, ...], metric: str) -> str:
    rows = []
    for method in methods:
        rows.append([DISPLAY.get(method, method), *[
            delta(data["change"][(rep, pct, method)], metric) for pct in REDUCED]])
    return table(["Method", *[f"{pct}% retained: paired Δ{metric.upper()} [95% interval]"
                               for pct in REDUCED]], rows)


def coverage_table(data: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            for family, methods in (("NSA", NSA_METHODS), ("Other PyOD", BASELINES)):
                counts = Counter()
                partial, full, zero = 0, 0, 0
                for method in methods:
                    row = data["summary"][(rep, pct, method)]
                    counts.update(json.loads(row["status_counts"]))
                    successes = int(row["success_seed_rows"])
                    full += successes == 85
                    partial += 0 < successes < 85
                    zero += successes == 0
                issues = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())
                                   if k != "success") or "none"
                rows.append([rep, pct, family, f"{counts['success']}/{85*len(methods)}",
                             full, partial, zero, issues])
    return table(["Features", "% retained", "Group", "Successful runs",
                  "Fully observed methods", "Partial", "Zero success", "Other outcomes"], rows)


def inventory_table(data: dict) -> str:
    rows = []
    names = set(data["audit"]["datasets"])
    if {r["dataset"] for r in data["inventory"]} != names:
        raise ValueError("dataset inventory names do not match audited grid")
    for row in sorted(data["inventory"], key=lambda r: r["dataset"]):
        total = int(row["source_rows"])
        normals = int(row["source_normals"])
        anomalies = int(row["source_anomalies"])
        if normals + anomalies != total or int(row["fit_seed_count"]) != 5:
            raise ValueError(f"invalid source inventory: {row['dataset']}")
        fit_cells = []
        parent_max = 600
        for pct in FRACTIONS:
            lo = int(row[f"fit_normals_{pct}_min"])
            hi = int(row[f"fit_normals_{pct}_max"])
            if not 2 <= lo <= hi <= parent_max:
                raise ValueError(f"invalid fit-row range: {row['dataset']}/{pct}")
            fit_cells.append(str(lo) if lo == hi else f"{lo}–{hi}")
        rows.append([row["dataset"], total, row["source_features"], normals,
                     anomalies, *fit_cells])
    return table(["Dataset", "Source rows", "Features", "Normals", "Anomalies",
                  "Fit normals 100%", "75%", "50%", "25%"], rows)


def overview(data: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in REDUCED:
            all_changes = [data["change"][(rep, pct, m)] for m in NSA_METHODS]
            complete = [r for r in all_changes if r["full_coverage"] == "True"]
            gain_auc = sum(num(r["macro_mean_delta_auroc"]) > 0 for r in complete)
            gain_ap = sum(num(r["macro_mean_delta_ap"]) > 0 for r in complete)
            positive_ci = sum(num(r["cluster_bootstrap_95_low_auroc"]) > 0
                              for r in complete)
            rows.append([rep, pct, f"{len(complete)}/13", gain_auc, gain_ap,
                         positive_ci, 13-len(complete)])
    return table(["Features", "% retained", "NSA configs with 85 paired runs",
                  "Positive mean ΔAUROC", "Positive mean ΔAP",
                  "AUROC interval above zero", "Partial or no pairs"], rows)


def improvement_pattern(data: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in REDUCED:
            complete = [data["change"][(rep, pct, method)] for method in NSA_METHODS
                        if data["change"][(rep, pct, method)]["full_coverage"] == "True"]
            gains = [r for r in complete if num(r["macro_mean_delta_auroc"]) > 0]
            declines = [r for r in complete if num(r["macro_mean_delta_auroc"]) < 0]
            gains.sort(key=lambda r: -num(r["macro_mean_delta_auroc"]))
            declines.sort(key=lambda r: num(r["macro_mean_delta_auroc"]))
            show = lambda entries: ", ".join(
                f"{DISPLAY[r['method']]} {fmt(num(r['macro_mean_delta_auroc']), True)}"
                for r in entries) or "none"
            rows.append([rep, pct, show(gains), show(declines),
                         ", ".join(DISPLAY[r["method"]] for r in complete
                                   if num(r["cluster_bootstrap_95_low_auroc"]) > 0) or "none"])
    return table(["Features", "% retained", "Complete NSA configs with positive mean ΔAUROC",
                  "Complete NSA configs with negative mean ΔAUROC",
                  "Unadjusted 95% interval wholly above zero"], rows)


def strongest_baseline_check(data: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            eligible = [data["summary"][(rep, pct, m)] for m in BASELINES
                        if data["summary"][(rep, pct, m)]["full_coverage"] == "True"]
            nsa = [m for m in NSA_METHODS
                   if data["summary"][(rep, pct, m)]["full_coverage"] == "True"]
            if not eligible:
                rows.append([rep, pct, "none", "none", "—", "—"])
                continue
            best_auc = max(eligible, key=lambda r: float(r["macro_auroc_full_coverage"]))
            best_ap = max(eligible, key=lambda r: float(r["macro_ap_full_coverage"]))
            auc_pairs = [data["same"][(rep, pct, m, best_auc["method"])] for m in nsa]
            ap_pairs = [data["same"][(rep, pct, m, best_ap["method"])] for m in nsa]
            auc_point = sum(num(r["macro_mean_delta_auroc"]) > 0 for r in auc_pairs)
            ap_point = sum(num(r["macro_mean_delta_ap"]) > 0 for r in ap_pairs)
            auc_ci = sum(num(r["cluster_bootstrap_95_low_auroc"]) > 0 for r in auc_pairs)
            ap_ci = sum(num(r["cluster_bootstrap_95_low_ap"]) > 0 for r in ap_pairs)
            rows.append([rep, pct,
                         f"{best_auc['method']} {fmt(float(best_auc['macro_auroc_full_coverage']))}",
                         f"{best_ap['method']} {fmt(float(best_ap['macro_ap_full_coverage']))}",
                         f"{auc_point}/{len(nsa)} point; {auc_ci}/{len(nsa)} interval > 0",
                         f"{ap_point}/{len(nsa)} point; {ap_ci}/{len(nsa)} interval > 0"])
    return table(["Features", "% retained", "Highest full-coverage other AUROC",
                  "Highest full-coverage other AP", "NSA above AUROC leader on matched cells",
                  "NSA above AP leader on matched cells"], rows)


def concentration_check(data: dict) -> str:
    rows = []
    for rep in REPS:
        complete = [data["change"][(rep, 25, m)] for m in NSA_METHODS
                    if data["change"][(rep, 25, m)]["full_coverage"] == "True"
                    and num(data["change"][(rep, 25, m)]["macro_mean_delta_auroc"]) > 0]
        if not complete:
            rows.append([rep, "No complete NSA gain", "—", "—", "—", "—"])
            continue
        chosen = max(complete, key=lambda r: num(r["macro_mean_delta_auroc"]))
        method = chosen["method"]
        loo = [r for r in data["loo"] if r["representation"] == rep
               and int(r["retention_pct"]) == 25 and r["method"] == method]
        if len(loo) != 17:
            raise ValueError(f"missing full leave-one-out sweep for {rep}/{method}")
        most_influential = max(loo, key=lambda r: float(r["removed_dataset_influence_auroc"]))
        remain = [float(r["leave_one_out_macro_delta_auroc"]) for r in loo]
        rows.append([rep, DISPLAY[method], fmt(num(chosen["macro_mean_delta_auroc"]), True),
                     f"{most_influential['excluded_dataset']} "
                     f"{fmt(float(most_influential['removed_dataset_influence_auroc']), True)}",
                     f"[{fmt(min(remain), True)}, {fmt(max(remain), True)}]",
                     sum(x > 0 for x in remain)])
    return table(["Features", "Largest fully paired NSA gain at 25%",
                  "Full 17-dataset mean ΔAUROC", "Largest single-dataset influence",
                  "Leave-one-out mean range", "Positive after exclusion (of 17)"], rows)


def ranking_table(data: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            for family in ("NSA", "baseline"):
                eligible = [r for r in data["ranks"] if r["representation"] == rep
                            and int(r["retention_pct"]) == pct
                            and r["metric"] == "auroc" and r["family"] == family]
                eligible.sort(key=lambda r: int(r["rank_family_full_coverage"]))
                top = eligible[:5]
                rows.append([rep, pct, "NSA" if family == "NSA" else "Other PyOD",
                             f"{len(eligible)}/{13 if family == 'NSA' else 43}",
                             ", ".join(f"{DISPLAY.get(r['method'], r['method'])} {fmt(float(r['score']))}"
                                       for r in top) or "none"])
    return table(["Features", "% retained", "Group", "Fully observed",
                  "Top five by AUROC among fully observed"], rows)


def raw_all_success_leaders(data: dict) -> str:
    """Descriptive normal-only performance without measured-time filtering."""
    sensitivity = {(r["representation"], int(r["retention_pct"]), r["method"]): r
                   for r in data["recorded_vs_qualified"]}
    if len(sensitivity) != 448:
        raise ValueError("raw all-success sensitivity grid is incomplete")
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            cells = [rep, pct]
            for methods in (NSA_METHODS, BASELINES):
                full = [sensitivity[(rep, pct, m)] for m in methods
                        if int(sensitivity[(rep, pct, m)]["raw_success_seed_rows"]) == 85]
                if full:
                    best = max(full, key=lambda r: float(r["macro_auroc_all_recorded_successes"]))
                    cells.append(f"{DISPLAY.get(best['method'], best['method'])} "
                                 f"{fmt(num(best['macro_auroc_all_recorded_successes']))} / "
                                 f"{fmt(num(best['macro_ap_all_recorded_successes']))} "
                                 f"({len(full)}/{len(methods)} full)")
                else:
                    cells.append(f"none (0/{len(methods)} full)")
            rows.append(cells)
    return table(["Features", "% retained", "Highest raw-success NSA AUROC / AP",
                  "Highest raw-success other PyOD AUROC / AP"], rows)


def same_size_table(data: dict, rep: str, baseline: str) -> str:
    rows = []
    for nsa in NSA_METHODS:
        cells = [DISPLAY[nsa]]
        for pct in REDUCED:
            row = data["same"][(rep, pct, nsa, baseline)]
            cells.append(f"{delta(row, 'auroc')} / AP {delta(row, 'ap', interval=False)}; "
                         f"AUROC-win datasets {row['first_higher_datasets']}/{row['paired_datasets']}")
        rows.append(cells)
    return table(["NSA configuration", *[f"{pct}% retained: NSA − {baseline} ΔAUROC / ΔAP"
                                        for pct in REDUCED]], rows)


def top_comparison(data: dict) -> str:
    rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            nsa = sorted((data["summary"][(rep, pct, m)] for m in NSA_METHODS
                          if data["summary"][(rep, pct, m)]["full_coverage"] == "True"),
                         key=lambda r: -float(r["macro_auroc_full_coverage"]))
            base = sorted((data["summary"][(rep, pct, m)] for m in BASELINES
                           if data["summary"][(rep, pct, m)]["full_coverage"] == "True"),
                          key=lambda r: -float(r["macro_auroc_full_coverage"]))
            if not nsa or not base:
                rows.append([rep, pct, "none" if not nsa else nsa[0]["method"],
                             "none" if not base else base[0]["method"], "—"])
                continue
            a, b = nsa[0], base[0]
            paired = data["same"][(rep, pct, a["method"], b["method"])]
            rows.append([rep, pct,
                         f"{DISPLAY[a['method']]} {fmt(float(a['macro_auroc_full_coverage']))}",
                         f"{b['method']} {fmt(float(b['macro_auroc_full_coverage']))}",
                         delta(paired, "auroc")])
    return table(["Features", "% retained", "Highest AUROC NSA, full coverage",
                  "Highest AUROC other PyOD, full coverage", "Matched NSA − other AUROC"], rows)


def failure_details(data: dict) -> str:
    rows = []
    for rep in REPS:
        for method in METHODS:
            parts = []
            for pct in FRACTIONS:
                row = data["summary"][(rep, pct, method)]
                if int(row["success_seed_rows"]) == 85:
                    continue
                issues = json.loads(row["status_counts"])
                parts.append(f"{pct}%: " + ", ".join(f"{k} {v}" for k, v in sorted(issues.items())))
            if parts:
                rows.append([rep, DISPLAY.get(method, method), "; ".join(parts)])
    return table(["Features", "Method", "Analysis statuses at sizes with incomplete coverage"], rows)


def measured_budget_section(data: dict) -> str:
    audit = data["audit"]
    excluded = data["overbudget"]
    if len(excluded) != audit["overbudget_success_records"]:
        raise ValueError("overbudget row count does not match audit")
    if not excluded:
        return "No recorded success exceeded the measured 15-second budget."
    longest = max(excluded, key=lambda r: float(r["measured_end_to_end_seconds"]))
    rows = []
    for entry in sorted(data["overbudget_methods"],
                        key=lambda r: (-int(r["overbudget_success_records"]), r["method"])):
        count = int(entry["overbudget_success_records"])
        if count:
            rows.append([entry["method"], count,
                         fmt(num(entry["max_measured_end_to_end_seconds"])),
                         entry["by_representation"], entry["by_retention_pct"]])
    affected = sum(int(r["overbudget_success_records"]) for r in data["overbudget_methods"])
    if affected != len(excluded):
        raise ValueError("overbudget method counts do not match row-level audit")
    score_rows = []
    for row in data["recorded_vs_qualified"]:
        count = int(row["overbudget_success_seed_rows"])
        if not count:
            continue
        raw_n = int(row["raw_success_seed_rows"])
        qualified_n = int(row["qualified_success_seed_rows"])
        if raw_n - qualified_n != count:
            raise ValueError("recorded/qualified sensitivity count mismatch")
        score_rows.append([
            row["representation"], row["retention_pct"], row["method"], count,
            f"{fmt(num(row['macro_auroc_all_recorded_successes']))} / "
            f"{fmt(num(row['macro_ap_all_recorded_successes']))} ({raw_n}/85)",
            f"{fmt(num(row['macro_auroc_time_qualified']))} / "
            f"{fmt(num(row['macro_ap_time_qualified']))} ({qualified_n}/85)",
        ])
    score_rows.sort(key=lambda r: (r[0], -int(r[1]), r[2]))
    return "\n".join([
        f"The raw JSONL has **{audit['raw_success_records']:,}** successes. "
        f"After the strict measured-time rule, **{audit['success_records']:,}** remain "
        f"qualified and **{len(excluded):,}** are marked `overbudget_success` for analysis. "
        "Every excluded row retains its original raw `success` status, AUROC, and AP in the "
        "JSONL. The longest measured successful run was "
        f"**{float(longest['measured_end_to_end_seconds']):.3f}s** "
        f"({longest['method']}, {longest['dataset']}, seed {longest['seed']}, "
        f"{longest['representation']}, {longest['retention_pct']}% retained).",
        "",
        table(["Method", "Raw successes over 15s", "Longest measured seconds",
               "By representation", "By retained %"], rows),
        "",
        "**All-success and strict-budget scores for affected method panels**",
        "",
        table(["Features", "% retained", "Method", "Overbudget raw successes",
               "Raw all-success AUROC / AP", "Strict ≤15s AUROC / AP"], score_rows),
        "",
        "Both score columns are descriptive equal-dataset-weight available-case means, "
        "with successful dataset/seed rows out of 85. The raw column includes valid "
        "normal-only fits that finished after the measured budget; the strict column "
        "qualifies runs by time. These two columns may use different successful cells, "
        "so paired comparisons in the primary tables remain the controlled comparison.",
        "",
        "The measured rule uses the **recorded fit seconds plus score seconds > 15.0**. "
        "The 15-second target was fixed before the sweep; the strict measured-time "
        "qualification was applied after discovering that the nominal Python signal timer "
        "could be ignored in native callbacks. This is a post-hoc enforcement decision, "
        "reported alongside the unfiltered raw-success sensitivity above. "
        "Overbudget scores remain valid normal-only outcomes; they are excluded from every "
        "strict-budget primary method mean, rank, paired change, "
        "same-size comparison, and interaction. The row-level "
        "`full_retention_overbudget_success.csv` records both raw and analysis status and "
        "the preserved raw scores. `full_retention_recorded_vs_qualified.csv` shows the "
        "descriptive available-case mean sensitivity for every method and panel; differing "
        "successful cell sets limit direct interpretation of those mean shifts.",
    ])


def watchdog_section(data: dict) -> str:
    audit = data["watchdog_audit"]
    events = audit["events"]
    if len(events) != audit["event_count"]:
        raise ValueError("watchdog event count mismatch")
    if not events:
        return "No external watchdog intervention was recorded in the completed grid."
    rows = []
    for event in events:
        age = float(event["raw_age_seconds_before_action"])
        if event["method"] != "LOCI" or age < 600:
            raise ValueError("invalid LOCI watchdog event in report")
        rows.append([
            event["intervention_id"],
            event["dataset"], event["seed"], event["representation"],
            event["retention_pct"], f"{age:.1f}", event["action_signal"],
        ])
    return "\n".join([
        f"An external watchdog intervened in **{len(events)}** stalled LOCI cell(s) "
        "only after the raw file had not advanced for at least 600 seconds. Before each "
        "intervention, it saved and hashed the partial raw JSONL and log; the final grid "
        "contains a matching LOCI `timeout` row with null scores and an "
        "`ExternalWatchdogTimeout:` reason. The row is a recorded failed attempt, "
        "not an inferred AUROC or AP. The runner then resumes the remaining methods "
        "without rerunning completed records.",
        "",
        table(["Intervention ID", "Dataset", "Seed", "Features", "% retained",
               "Raw age before action (s)", "Signal"], rows),
        "",
        "`watchdog_intervention_audit.json` verifies unique event IDs, the ≥600-second "
        "age rule, snapshot hashes, the appended LOCI row hash, and matching final raw rows. "
        "The reproducibility bundle includes the watchdog script, intervention JSON, "
        "and before snapshots for review. These events do not contribute a score to "
        "the primary or raw all-success performance tables.",
    ])


def dataset_diagnostics(data: dict) -> str:
    # Give every dataset in the visible report. Each cell is the mean of
    # successful matched seed changes; full data are in dataset_changes CSV.
    rows = []
    names = sorted(data["audit"]["datasets"])
    for rep in REPS:
        for dataset in names:
            cells = [rep, dataset]
            for method in ("NSA/dual", "NSA/variable", "IForest", "KNN"):
                row = data["dataset"][(rep, 25, method, dataset)]
                cells.append(f"{fmt(num(row['mean_delta_auroc']), True)} "
                             f"({row['paired_success_seed_rows']}/5)")
            rows.append(cells)
    return table(["Features", "Dataset", "Dual NSA ΔAUROC", "Variable NSA ΔAUROC",
                  "IForest ΔAUROC", "KNN ΔAUROC"], rows)


def report(data: dict) -> str:
    audit = data["audit"]
    required = ["XGBOD", "DevNet"]
    for method in required:
        for rep in REPS:
            for pct in FRACTIONS:
                row = data["summary"][(rep, pct, method)]
                if json.loads(row["status_counts"]) != {"requires_labeled_outliers": 85}:
                    raise ValueError(f"labeled-outlier exclusion drift: {method}/{rep}/{pct}")
    parts = [
        "# Fresh full-roster PyOD NSA benchmark under normal-only training",
        "",
        "**Review report for the user. Nothing was posted to GitHub.**",
        "",
        "## Scope and evaluation rule",
        "",
        "The experiment reruns all 56 predeclared configurations from the tested PyOD checkout: "
        "13 NSA configurations spanning 11 strategy families, and all 43 other registered tabular "
        "methods. Four training sizes retain 100%, 75%, 50%, or 25% of a capped pool of normal "
        "rows. All four sizes are fresh runs. Every fitted model receives feature rows from the "
        "normal class only. The split uses labels to stratify and discard anomalies from the "
        "training partition; held-out labels are used only to calculate AUROC and average "
        "precision (AP). No anomalous feature row or label array is passed to `model.fit`.",
        "",
        "An unsuccessful or inapplicable configuration is recorded with its status and has no "
        "scored AUROC/AP pair. Raw successes whose measured fit-plus-score duration exceeds 15 seconds "
        "are classified as `overbudget_success` for analysis, while their raw status and scores "
        "remain in JSONL. A method with partial coverage is never ranked by its available-case score. "
        "All size changes and NSA/other-method comparisons below use only matched successful "
        "dataset and seed cells; the matching denominator is always visible.",
        "",
        "## Main observations",
        "",
        overview(data),
        "",
        "A positive mean counts only a configuration with complete paired coverage at that size. "
        "The 95% intervals are descriptive dataset-cluster bootstrap intervals; no multiplicity "
        "adjustment was applied. Intervals are not estimated with fewer than five paired "
        "datasets. A positive interval therefore does not establish a universal "
        "or confirmatory training-size benefit.",
        "",
        "### Which complete NSA configurations improved or declined?",
        "",
        improvement_pattern(data),
        "",
        "Positive and negative are based on paired mean AUROC versus each configuration's "
        "fresh 100% result. Methods with partial or no matched coverage are listed separately "
        "in the failure table and are not counted here. AP changes appear in the full family "
        "tables below.",
        "",
        "### Highest observed methods with complete coverage",
        "",
        top_comparison(data),
        "",
        "Each row selects the highest mean AUROC method **after observing the test results** "
        "within each family and training size. It is a descriptive ranking, not a model-selection "
        "procedure or an independent estimate of a selected method. The last column uses matched "
        "cells for the displayed methods.",
        "",
        "### Raw all-success normal-only performance sensitivity",
        "",
        raw_all_success_leaders(data),
        "",
        "This second view uses the unchanged raw successful scores, including successes that "
        "finished after 15 measured seconds. It ranks only methods with all 85 raw "
        "dataset/seed successes in the panel. The leaders are selected descriptively from "
        "held-out outcomes and need not match the strict-budget leaders above. "
        "`full_retention_recorded_vs_qualified.csv` contains both raw and qualified AUROC/AP "
        "and coverage for **all 56 methods** in all eight panels. The raw scores answer the "
        "normal-only detection-performance question without filtering on runtime; strict "
        "scores answer the controlled 15-second-budget question.",
        "",
        "### Comparison with the strongest fully observed other method",
        "",
        strongest_baseline_check(data),
        "",
        "The AUROC and AP leaders are chosen separately within each feature representation "
        "and training size after observing the test results. Counts use NSA configurations with "
        "complete coverage and matched cells with the displayed leader. A positive point "
        "difference is weaker evidence than a descriptive interval wholly above zero. The "
        "many comparisons and selection of a leader from test outcomes prevent a confirmatory "
        "superiority claim.",
        "",
        "### Dataset concentration of the largest 25% NSA gain",
        "",
        concentration_check(data),
        "",
        "This is an exploratory leave-one-dataset-out sensitivity check. The highlighted "
        "configuration is selected after seeing its gain; all complete method/size combinations "
        "were also processed in `full_retention_leave_one_out.csv`. A strong effect driven by "
        "one dataset should be interpreted narrowly.",
        "",
        "## Protocol and audit",
        "",
        f"- **Grid:** 17 ODDS datasets × 5 split seeds (42–46) × two feature representations "
        f"× four retained-normal fractions × 56 configurations = **{audit['expected_records']:,}** "
        "fresh records. All are present before this report is generated.",
        "- **Split:** 60/40 stratified train/test. Every anomaly in the train partition is "
        "discarded. The original normal pool is capped at 600 rows; the mixed held-out test "
        "set is capped at 2,000, preserving its class mix. Retention uses nested, seed-fixed "
        "normal subsets and keeps each subset in the original parent fit-row order.",
        "- **Features:** original dimensions and two-dimensional PCA. StandardScaler and PCA "
        "are fitted again using the retained normal rows at each size. The test rows and labels "
        "stay fixed within each dataset/seed comparison.",
        "- **Models:** the same fixed constructors and label-free neighbor limits are used at every "
        "size. PyOD's `contamination=0.1` is retained for its score cutoff, but no training "
        "anomalies are inserted; AUROC/AP use `decision_function` scores. A nominal 15-second "
        "Python signal timer begins before lazy model-class import, construction, fitting, "
        "and scoring. Optional torch setup happens before that timer starts. Primary analysis "
        "requires the recorded fit-plus-score elapsed time, which includes torch setup, "
        "to be at most 15 seconds.",
        "- **Audit:** dataset SHA-256 hashes, exact full grid, zero anomalous fit rows, "
        "retained normal row identities and order, fixed held-out row identities, metadata, "
        "status validity, and transform consistency across methods were checked. "
        "The machine-readable audit file records total success and each failure status.",
        "",
        "The source checkout is pinned to commit `0d6c901652f82c813da45875873ef4a8560cf049` "
        "(tree `57ec5c9ea2800b559019adf0457d7e5b858fc153`). The dataset manifest "
        "records each ODDS file's source path and SHA-256. Anomalies in the mixed test set are "
        "necessary to measure detection; they never enter fitting.",
        "",
        "### Dataset inventory and actual training sizes",
        "",
        "Source row, feature, and class counts come from the pinned dataset manifest and were "
        "checked against the hashed source arrays. Fit counts come from the audited fresh "
        "normal-only records; a range shows variation across the five split seeds. The same "
        "counts apply to original features and PCA(2). The 100% pool is capped at 600 normals "
        "before the smaller nested subsets are drawn.",
        "",
        inventory_table(data),
        "",
        "## Coverage-qualified ranking",
        "",
        ranking_table(data),
        "",
        "The complete coverage-qualified AUROC **and AP** rankings are in "
        "`full_retention_coverage_qualified_ranks.csv`. These rankings include only methods "
        "with 85 successful dataset/seed runs for the given feature representation and size. "
        "A batch-dependent detector can still appear in this descriptive ranking; inference "
        "semantics are discussed under limitations.",
        "",
        "## All NSA strategies: AUROC, AP, and change versus fresh 100% training",
        "",
        "Each score is an equal-dataset-weight mean over successful runs, followed by successful "
        "runs out of 85. A dagger (†) marks a partial-coverage mean; it cannot be compared "
        "directly across sizes if the successful cells differ. The paired tables solve that "
        "problem. Binary is one NSA family with three matching rules shown separately.",
        "",
    ]
    for rep, title in (("native", "Original features"), ("pca2", "PCA to two dimensions")):
        parts += [f"### {title}", "", method_score_table(data, rep, NSA_METHODS), "",
                  "**Paired AUROC change versus 100% retained**", "",
                  method_change_table(data, rep, NSA_METHODS, "auroc"), "",
                  "**Paired AP change versus 100% retained**", "",
                  method_change_table(data, rep, NSA_METHODS, "ap"), ""]
    parts += ["## All 43 other PyOD methods", "",
              "The same normal-only subsets, transforms, timeout, and held-out rows are used "
              "for every registered method. The method tables include exclusions and failures "
              "through their 0/85 or partial-coverage denominators.", ""]
    for rep, title in (("native", "Original features"), ("pca2", "PCA to two dimensions")):
        parts += [f"### {title}", "", method_score_table(data, rep, BASELINES), "",
                  "**Paired AUROC change versus 100% retained**", "",
                  method_change_table(data, rep, BASELINES, "auroc"), "",
                  "**Paired AP change versus 100% retained**", "",
                  method_change_table(data, rep, BASELINES, "ap"), ""]
    parts += [
        "## NSA versus other methods at the same training size",
        "",
        "The complete `full_retention_nsa_vs_baseline.csv` contains every 13 × 43 NSA/other "
        "comparison at all four sizes and both feature representations: matched AUROC and AP "
        "mean differences, 95% dataset-cluster intervals, seed-level and dataset-level win counts, "
        "and coverage. Differences within 1e-12 AUROC count as ties. Positive values favor "
        "NSA. This is **4,472 comparisons**; reporting "
        "only the largest positive ones in prose would select on test outcomes. The two fixed "
        "controls below make the comparison readable without changing the full table.",
        "",
    ]
    for rep in REPS:
        for baseline in ("IForest", "KNN"):
            parts += [f"### {rep}: NSA minus {baseline} at 75%, 50%, and 25% retained", "",
                      same_size_table(data, rep, baseline), ""]
    parts += [
        "A positive same-size difference is different from a positive **training-size interaction**. "
        "The latter is `(NSA reduced − NSA 100%) − (other reduced − other 100%)` "
        "and says who changed more, not who has the higher absolute AUROC. All **3,354** "
        "four-fit matched interactions with AUROC/AP and coverage are in "
        "`full_retention_interaction.csv`.",
        "",
        "## Dataset-level diagnostic view",
        "",
        "This table shows paired AUROC change at 25% retained versus the fresh 100% run for "
        "four fixed methods. Each cell is the mean over matched successful seeds within that "
        "dataset, with matched seeds out of five. The complete "
        "`full_retention_dataset_changes.csv` has every method, fraction, dataset, and status.",
        "",
        dataset_diagnostics(data),
        "",
        "## Methods excluded or failing under this protocol",
        "",
        "**XGBOD and DevNet require labeled outliers for their intended supervised training.** "
        "They are included in all 56-method denominators as `requires_labeled_outliers` and "
        "are not assigned scores. Their Python `fit` methods may be callable with all-zero labels; "
        "that is not the intended outlier-discrimination training and is not evidence that they "
        "work under the stated protocol. LUNAR, SO_GAAL, and MO_GAAL may synthesize negatives "
        "from normal data; no real anomalous row is provided to them.",
        "",
        "MAD requires one feature and is inapplicable to both tested feature panels. "
        "NSA Voronoi requires 2 or 3 features; deterministic requires exactly 2. Some "
        "binary matching configurations can fail on two PCA features because the bit "
        "signature is too short for their construction. Actual outcomes appear below.",
        "",
        "### Measured-time qualification of raw successes",
        "",
        measured_budget_section(data),
        "",
        f"One timeout fired between AUROC and AP calculation, leaving an intermediate "
        f"AUROC in its raw JSONL row. It is excluded from every score and comparison; "
        f"`full_retention_partial_metric_timeouts.csv` identifies the "
        f"{data['audit']['partial_metric_timeout_rows']} affected row(s).",
        "",
        "### External watchdog interventions",
        "",
        watchdog_section(data),
        "",
        "### Process restart provenance",
        "",
        "Two 25%-retention PCA(2) cells have two identical logged runner commands: "
        "glass/seed 43 resumed after 36 committed records, and pendigits/seed 43 "
        "resumed after 30. Each later command completed its cell to 56 method records "
        "without rewriting the committed prefix. Separate sidecars in `provenance/` "
        "hash both invocation-log segments, the preserved raw prefix, and the final "
        "raw/log files; `process_restart_audit.json` verifies both. The saved logs do "
        "not establish the first process exit codes or identify which methods caused "
        "the processes to end. No score or failure status was imputed from either restart.",
        "",
        coverage_table(data),
        "",
        failure_details(data),
        "",
        "No error, timeout, geometry exclusion, or overbudget success was replaced with "
        "AUROC/AP zero. A 0/85 method is unranked. A partial-coverage result can be inspected in the full "
        "status and matched-pair tables but cannot be generalized to the omitted cells.",
        "",
        "## Interpretation limits",
        "",
        "- The 600-normal training and 2,000-row test caps make this a controlled CPU screen, "
        "not a full-data replication of the original papers. In small datasets, 25% retained "
        "can leave only a few dozen normals. Five split seeds do not create five independent "
        "datasets; the bootstrap resamples whole datasets and retains the seeds inside each.",
        "- The nominal 15-second Python signal timer includes lazy model-class import and "
        "construction, but optional torch import and seed setup occur before it. The recorded "
        "fit time includes that setup. Native callbacks may not stop when the signal is "
        "raised; some run logs show Numba/llvmlite callbacks ignoring a `FitTimeout` and "
        "later returning a raw success. Those logs do not establish the cause of every "
        "overbudget result. The strict **measured end-to-end ≤15s** "
        "qualification was introduced after observing this behavior and removes such "
        "overbudget successes from primary comparisons; both raw and qualified results are "
        "reported. "
        "A timeout or overbudget success establishes only a budget outcome in this environment, "
        "not that the detector is intrinsically unusable. "
        "Methods run in a fixed roster order within each cell. Lazy import caching, "
        "execution order, and concurrent machine load can affect which fits reach the "
        "15-second timeout. The recorded primary scores and paired comparisons remain "
        "the results for this run.",
        "- Deep-model compute overrides differ from paper defaults: DIF uses 10 ensembles "
        "instead of 50; DeepSVDD 50 epochs instead of 100; AnoGAN 50 training and 5 query "
        "epochs instead of 500 and 20; ALAD 50 instead of 200 epochs; LUNAR 50 instead "
        "of 200 epochs. Fixed constructors are not optimized for each dataset or training size.",
        "- PCA is refitted on the retained normal set at every size. A change in PCA(2) "
        "performance can reflect altered scaling or axes as well as altered detector learning. "
        "Two PCA dimensions can discard anomaly-relevant directions.",
        "- COF, LOCI, SOS, SOD, ECOD, COPOD, RGraph, SUOD, LMDD, DIF, CD, and ROD "
        "have batch-dependent scoring behavior in this evaluation. Their test scores can "
        "depend on other test rows; they are not strictly pointwise novelty predictors "
        "like NSA. The analysis CSV marks these methods. The active model factory metadata "
        "under-flags LMDD, DIF, CD, and ROD as inductive; the analysis-level correction is "
        "recorded in `batch_scoring_audit.json`. The active model factory was not changed.",
        "- AP depends on anomaly prevalence in the held-out set. Interpret AP changes within "
        "matched splits. All intervals are descriptive and unadjusted for thousands of "
        "method, fraction, representation, and metric comparisons. Coverage-qualified "
        "ranks are descriptive and cannot establish a generally best method.",
        "",
        "## Reproducibility files",
        "",
        "The raw JSONL includes success and non-success records for every cell. "
        "`full_retention_analysis_audit.json` records the validation and overall status counts. "
        "`batch_scoring_audit.json` records scoring-semantics metadata corrections. "
        "`watchdog_intervention_audit.json` links any external LOCI intervention to its "
        "snapshots and final timeout row. "
        "`process_restart_audit.json` links the glass and pendigits PCA(2)/seed 43/25% "
        "restart sidecars to their raw and log evidence. "
        "The overbudget row list and recorded-versus-qualified sensitivity tables preserve "
        "the distinction between raw and derived status. "
        "The method summary, matched changes, dataset changes, complete NSA/other pairwise "
        "table, interactions, coverage-qualified ranks, leave-one-dataset-out results, "
        "and 17-dataset source/training-size inventory "
        "are separate CSVs. The dataset "
        "manifest pins every source file; the run manifest and source scripts pin the tested "
        "checkout, constructors, and code. No PR response or other external post was made.",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).parent / "analysis")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "report_draft.md")
    args = parser.parse_args()
    data = load(args.analysis_dir)
    result = report(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    print(f"Wrote {len(result):,} characters to {args.output}")


if __name__ == "__main__":
    main()
