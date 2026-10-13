"""Write a review-only six-size normal-only retention report from audited CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze import FRACTIONS, METHODS, NEW_FRACTIONS, NSA_METHODS, REPS

HERE = Path(__file__).resolve().parent
TABLES = (
    "method_summary", "common_field_ranks", "raw_common_ranks", "panel_ranks", "rank_change",
    "change_vs_100", "nsa_vs_baseline", "relative_change_vs_100", "dataset_nsa_ranks",
    "dataset_inventory", "raw_vs_strict",
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        values = list(csv.DictReader(stream))
    if not values:
        raise ValueError(f"missing report table: {path}")
    return values


def number(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or value == "":
        return "—"
    result = float(value)
    return f"{result:+.{digits}f}" if signed else f"{result:.{digits}f}"


def md_table(header: list[str], data: list[list[object]]) -> str:
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, header)) + " |",
                      "| " + " | ".join("---" for _ in header) + " |",
                      *("| " + " | ".join(map(clean, row)) + " |" for row in data)])


def load(directory: Path) -> dict:
    audit = json.loads((directory / "extreme_retention_analysis_audit.json").read_text())
    recovery = json.loads((directory / "extreme_retention_recovery_audit.json").read_text())
    data = {name: rows(directory / f"extreme_retention_{name}.csv") for name in TABLES}
    paper_path = directory / "extreme_retention_focal_metrics.json"
    paper_bytes = paper_path.read_bytes()
    paper = json.loads(paper_bytes)
    expected = {
        "method_summary": 672,
        "change_vs_100": 560,
        "nsa_vs_baseline": 6708,
        "relative_change_vs_100": 2236,
        "dataset_nsa_ranks": 2652,
        "dataset_inventory": 17,
        "raw_vs_strict": 672,
    }
    if (audit["expected_records"] != 57120
            or audit["expected_cells"] != 1020
            or audit["retention_percent"] != list(FRACTIONS)
            or audit["representations"] != list(REPS)
            or audit["methods"] != METHODS
            or audit["strict_measured_time_budget_seconds"] != 15.0
            or audit["new_recovery"]["new_event_count"] != recovery["new_event_count"]
            or audit["focal_metrics_sha256"] != hashlib.sha256(paper_bytes).hexdigest()
            or audit["focal_metrics_panels"] != 12 or len(paper["panels"]) != 12):
        raise ValueError("six-size audit is inconsistent with the declared protocol")
    for name, count in expected.items():
        if len(data[name]) != count or audit["tables"][name] != count:
            raise ValueError(f"incomplete {name} report table")
    for row in data["method_summary"]:
        qualified = int(row["qualified_success_seed_rows"])
        unsuitable = int(row["unsuitable_cases"])
        if (qualified + unsuitable != 85
                or not row["suitability_label"]
                or not row["unsuitability_causes"]):
            raise ValueError("invalid method suitability or coverage summary")
    if (set(audit["tables"]) != set(TABLES)
            or any(len(data[name]) != audit["tables"][name] for name in TABLES)):
        raise ValueError("analysis table counts differ from the audit")
    data["audit"] = audit
    data["recovery"] = recovery
    data["paper"] = paper
    data["summary_index"] = {(r["representation"], int(r["retention_pct"]), r["method"]): r
                             for r in data["method_summary"]}
    data["common_index"] = {(r["representation"], int(r["retention_pct"]), r["metric"],
                              r["field"], r["method"]): r
                             for r in data["common_field_ranks"]}
    data["panel_index"] = {(r["representation"], int(r["retention_pct"]), r["metric"],
                             r["field"], r["method"]): r
                            for r in data["panel_ranks"]}
    data["raw_common_index"] = {(r["representation"], int(r["retention_pct"]), r["metric"],
                                  r["field"], r["method"]): r
                                 for r in data["raw_common_ranks"]}
    data["change_index"] = {(r["representation"], int(r["retention_pct"]), r["method"]): r
                              for r in data["change_vs_100"]}
    data["pair_index"] = {(r["representation"], int(r["retention_pct"]), r["nsa"], r["baseline"]): r
                            for r in data["nsa_vs_baseline"]}
    data["relative_index"] = {(r["representation"], int(r["retention_pct"]),
                                r["nsa"], r["baseline"]): r
                               for r in data["relative_change_vs_100"]}
    for table, index in (("method_summary", "summary_index"),
                         ("common_field_ranks", "common_index"),
                         ("panel_ranks", "panel_index"),
                         ("raw_common_ranks", "raw_common_index"),
                         ("change_vs_100", "change_index"),
                         ("nsa_vs_baseline", "pair_index"),
                         ("relative_change_vs_100", "relative_index")):
        if len(data[table]) != len(data[index]):
            raise ValueError(f"duplicate key in {table}")
    if ({(p["representation"], int(p["retention_pct"])) for p in paper["panels"]}
            != {(rep, pct) for rep in REPS for pct in FRACTIONS}
            or any(len(p["nsa"]) != len(NSA_METHODS) for p in paper["panels"])):
        raise ValueError("paper metric panels do not cover all representations, sizes, or NSA methods")
    independent_path = directory / "independent_audit.json"
    if independent_path.exists():
        independent = json.loads(independent_path.read_text(encoding="utf-8"))
        if (independent.get("expected_cells") != 340
                or independent.get("expected_records") != 19040
                or independent.get("observed_cells") != 340
                or independent.get("observed_records") != 19040
                or independent.get("complete_cells") != 340
                or independent.get("all_fit_rows_independently_verified_normal") is not True
                or independent.get("watchdog_review", {}).get("completed_interventions_verified")
                    != recovery["new_event_count"]):
            raise ValueError("independent extreme audit conflicts with the six-size report")
        data["independent_audit_sha256"] = hashlib.sha256(
            independent_path.read_bytes()).hexdigest()
    adaptive_dir = HERE / "adaptive_batch" / "analysis"
    adaptive_audit_path = adaptive_dir / "adaptive_batch_audit.json"
    if adaptive_audit_path.exists():
        adaptive_audit = json.loads(adaptive_audit_path.read_text(encoding="utf-8"))
        filenames = ("adaptive_batch_cell_comparison.csv",
                     "adaptive_batch_method_summary.csv",
                     "adaptive_batch_fixed_rank_sensitivity.csv")
        expected_adaptive = {filenames[0]: 740, filenames[1]: 24, filenames[2]: 312}
        if (adaptive_audit.get("sensitivity_protocol") != "adaptive-training-batch-v1"
                or adaptive_audit.get("expected_records") != 740
                or adaptive_audit.get("actual_records") != 740
                or adaptive_audit.get("expected_cells") != 370
                or adaptive_audit.get("actual_cells") != 370
                or adaptive_audit.get("adapter_applied_rows") != 540
                or adaptive_audit.get("primary_analysis_audit_sha256")
                    != hashlib.sha256((directory / "extreme_retention_analysis_audit.json").read_bytes()).hexdigest()
                or adaptive_audit.get("primary_method_summary_sha256")
                    != hashlib.sha256((directory / "extreme_retention_method_summary.csv").read_bytes()).hexdigest()):
            raise ValueError("adaptive-batch audit does not match this six-size analysis")
        adaptive_tables = {}
        for filename in filenames:
            path = adaptive_dir / filename
            if (adaptive_audit["output_rows"].get(filename) != expected_adaptive[filename]
                    or adaptive_audit["output_sha256"].get(filename)
                        != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError(f"adaptive-batch table changed: {filename}")
            adaptive_tables[filename] = rows(path)
            if len(adaptive_tables[filename]) != expected_adaptive[filename]:
                raise ValueError(f"adaptive-batch table incomplete: {filename}")
        adaptive_summary = adaptive_tables["adaptive_batch_method_summary.csv"]
        adaptive_rank = adaptive_tables["adaptive_batch_fixed_rank_sensitivity.csv"]
        summary_index = {(r["representation"], int(r["retention_pct"]), r["method"]): r
                         for r in adaptive_summary}
        rank_index = {(r["representation"], int(r["retention_pct"]), r["metric"], r["nsa"]): r
                      for r in adaptive_rank}
        if len(summary_index) != 24 or len(rank_index) != 312:
            raise ValueError("adaptive-batch panel keys are duplicated")
        overlap_path = HERE / "adaptive_batch" / "provenance" / "startup_timing_overlap.json"
        overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
        first_wave = overlap.get("affected_first_wave_cells", [])
        timed_rows = [row for cell in first_wave
                      for row in cell.get("method_status_and_measured_seconds", [])]
        if (overlap.get("schema_version") != 1
                or overlap.get("scope") != "adaptive-training-batch-v1 first-wave timing observation"
                or len(first_wave) != 8 or len(timed_rows) != 16
                or any(row["status"] != "success"
                       or not 0 <= row["fit_plus_score_seconds"] < 5.1
                       for row in timed_rows)):
            raise ValueError("adaptive startup timing sidecar is inconsistent")
        data["adaptive"] = {"audit": adaptive_audit, "summary": summary_index,
                            "ranks": rank_index,
                            "startup_timing_overlap_sha256": hashlib.sha256(
                                overlap_path.read_bytes()).hexdigest()}
    return data


def fit_inventory(data: dict) -> str:
    output = []
    for r in data["dataset_inventory"]:
        sizes = []
        for pct in FRACTIONS:
            lo, hi = int(r[f"fit_normals_{pct}_min"]), int(r[f"fit_normals_{pct}_max"])
            sizes.append(str(lo) if lo == hi else f"{lo}–{hi}")
        output.append([r["dataset"], r["source_rows"], r["source_features"],
                       r["source_normals"], r["source_anomalies"], *sizes])
    return md_table(["Dataset", "Rows", "Features", "Normal", "Anomaly",
                     "100% fit", "75% fit", "50% fit", "25% fit", "10% fit", "5% fit"], output)


def field_summary(data: dict) -> str:
    audit = data["audit"]
    rows_out = []
    for rep in REPS:
        f = audit["common_fields"][rep]
        raw = audit["raw_common_fields"][rep]
        rows_out.append([rep, len(f["all"]), len(f["nsa"]), len(f["baseline"]),
                         len(f["inductive_all"]), len(f["inductive_baseline"]),
                         len(raw["all"]), len(raw["nsa"])])
    return md_table(["Features", "All common", "NSA common", "Other common",
                     "Inductive common", "Inductive other", "Raw common", "Raw NSA common"], rows_out)


def all_method_new_panel(data: dict, rep: str, pct: int) -> str:
    out = []
    for method in METHODS:
        row = data["summary_index"][(rep, pct, method)]
        common = data["common_index"].get((rep, pct, "auroc", "all_common", method))
        out.append([method, row["family"],
                    f"{row['qualified_success_seed_rows']}/85",
                    f"{row['raw_success_seed_rows']}/85",
                    row["suitability_label"],
                    number(row["macro_auroc_qualified_available"]),
                    number(row["macro_ap_qualified_available"]),
                    f"{common['rank']}/{common['field_size']}" if common else "—",
                    row["unsuitability_causes"]])
    return md_table(["Method", "Family", "≤15 s success", "Raw success",
                     "Suitability under tested configuration", "AUROC available",
                     "AP available", "Six-size all rank", "Non-success causes and counts"], out)


def suitability_overview(data: dict) -> str:
    out = []
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            panel = [data["summary_index"][(rep, pct, method)] for method in METHODS]
            coverage = [int(row["qualified_success_seed_rows"]) for row in panel]
            zero = [row["method"] for row, count in zip(panel, coverage) if count == 0]
            out.append([rep, pct, sum(count == 85 for count in coverage),
                        sum(0 < count < 85 for count in coverage), len(zero),
                        ", ".join(zero) if zero else "none"])
    return md_table(["Features", "Retained %", "85/85 methods", "1–84/85 methods",
                     "0/85 methods", "Methods with 0/85 qualified scores"], out)


def nsa_rank_table(data: dict, rep: str, metric: str) -> str:
    out = []
    for method in NSA_METHODS:
        cells = []
        for pct in FRACTIONS:
            summary = data["summary_index"][(rep, pct, method)]
            key = (rep, pct, metric, "common_baselines_plus_focal_nsa", method)
            r = data["common_index"].get(key)
            if r:
                flag = "six-size ✓" if r["focal_eligible_all_six"] == "True" else "panel only"
                cells.append(f"{r['rank']}/{r['field_size']} · {number(r['score'])} · {flag}")
            else:
                cells.append(f"— · {summary['qualified_success_seed_rows']}/85")
        out.append([method, *cells])
    return md_table(["NSA configuration", *(f"{pct}%" for pct in FRACTIONS)], out)


def top_baselines(data: dict) -> str:
    out = []
    for rep in REPS:
        for pct in FRACTIONS:
            for metric in ("auroc", "ap"):
                fixed = [r for r in data["common_field_ranks"]
                         if r["representation"] == rep and int(r["retention_pct"]) == pct
                         and r["metric"] == metric and r["field"] == "common_baselines"]
                panel = [r for r in data["panel_ranks"]
                         if r["representation"] == rep and int(r["retention_pct"]) == pct
                         and r["metric"] == metric and r["field"] == "panel_baselines"]
                f = max(fixed, key=lambda r: float(r["score"])) if fixed else None
                p = max(panel, key=lambda r: float(r["score"])) if panel else None
                out.append([rep, pct, metric.upper(),
                            f"{f['method']} ({number(f['score'])})" if f else "—",
                            len(fixed),
                            f"{p['method']} ({number(p['score'])})" if p else "—",
                            len(panel)])
    return md_table(["Features", "% fit", "Metric", "Top fixed-field other detector",
                     "Fixed other N", "Top panel-field other detector", "Panel other N"], out)


def paired_nsa(data: dict) -> str:
    out = []
    for rep in REPS:
        for method in NSA_METHODS:
            for pct in (75, 50, 25, 10, 5):
                r = data["change_index"][(rep, pct, method)]
                ci_auc = (f"[{number(r['cluster_bootstrap_95_low_auroc'], 3, True)}, "
                          f"{number(r['cluster_bootstrap_95_high_auroc'], 3, True)}]")
                ci_ap = (f"[{number(r['cluster_bootstrap_95_low_ap'], 3, True)}, "
                         f"{number(r['cluster_bootstrap_95_high_ap'], 3, True)}]")
                if r["cluster_bootstrap_95_low_auroc"] == "":
                    ci_auc = "interval not estimated"
                if r["cluster_bootstrap_95_low_ap"] == "":
                    ci_ap = "interval not estimated"
                out.append([rep, method, pct, f"{r['paired_seed_rows']}/85",
                            r["paired_datasets"], number(r["macro_mean_delta_auroc"], 4, True),
                            ci_auc, number(r["macro_mean_delta_ap"], 4, True), ci_ap])
    return md_table(["Features", "NSA configuration", "% fit", "Matched runs",
                     "Datasets", "ΔAUROC vs 100%", "95% CI", "ΔAP vs 100%", "95% CI"], out)


def rank_changes(data: dict) -> str:
    out = []
    for rep in REPS:
        roster = data["audit"]["common_fields"][rep]["nsa"]
        for method in roster:
            for metric in ("auroc", "ap"):
                for pct in (75, 50, 25, 10, 5):
                    key = (rep, pct, metric, "all_common", method)
                    r = next((r for r in data["rank_change"]
                              if (r["representation"], int(r["retention_pct"]),
                                  r["metric"], r["field"], r["method"]) == key), None)
                    if not r:
                        raise ValueError(f"missing fixed-field rank change: {key}")
                    out.append([rep, method, metric.upper(), pct,
                                f"{r['rank_100']}/{r['field_size']}",
                                f"{r['rank_reduced']}/{r['field_size']}",
                                r["rank_improvement"], number(r["score_change"], 4, True)])
    return md_table(["Features", "NSA", "Metric", "% fit", "100% rank",
                     "Reduced rank", "Rank improvement", "Score change"], out)


def new_size_pair_leaders(data: dict) -> str:
    out = []
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            base = [r for r in data["common_field_ranks"]
                    if r["representation"] == rep and int(r["retention_pct"]) == pct
                    and r["metric"] == "auroc" and r["field"] == "common_baselines"]
            leader = max(base, key=lambda r: float(r["score"])) if base else None
            if not leader:
                out.append([rep, pct, "—", "—", "—", "—"])
                continue
            for nsa in NSA_METHODS:
                pair = data["pair_index"][(rep, pct, nsa, leader["method"])]
                out.append([rep, pct, nsa, leader["method"],
                            f"{pair['paired_seed_rows']}/85",
                            number(pair["macro_mean_delta_auroc"], 4, True)])
    return md_table(["Features", "% fit", "NSA", "Top fixed other", "Matched runs",
                     "NSA − other AUROC"], out)


def top_focal_comparison(data: dict, pcts=FRACTIONS) -> str:
    """Descriptive top-NSA versus top fixed-baseline comparison at every size."""
    out = []
    for rep in REPS:
        for pct in pcts:
            for metric in ("auroc", "ap"):
                eligible = [r for r in data["common_field_ranks"]
                            if r["representation"] == rep and int(r["retention_pct"]) == pct
                            and r["metric"] == metric and r["field"] == "common_baselines"]
                focal = [data["common_index"].get(
                    (rep, pct, metric, "common_baselines_plus_focal_nsa", nsa))
                    for nsa in NSA_METHODS]
                focal = [r for r in focal if r is not None]
                if not eligible or not focal:
                    out.append([rep, pct, metric.upper(), "—", "—", "—", "—", "—", "—"])
                    continue
                best_nsa = max(focal, key=lambda r: float(r["score"]))
                best_other = max(eligible, key=lambda r: float(r["score"]))
                pair = data["pair_index"][(rep, pct, best_nsa["method"], best_other["method"])]
                lo = pair[f"cluster_bootstrap_95_low_{metric}"]
                hi = pair[f"cluster_bootstrap_95_high_{metric}"]
                ci = (f"[{number(lo, 3, True)}, {number(hi, 3, True)}]"
                      if lo != "" and hi != "" else "interval not estimated")
                flag = "six-size" if best_nsa["focal_eligible_all_six"] == "True" else "panel only"
                out.append([rep, pct, metric.upper(),
                            f"{best_nsa['method']} ({number(best_nsa['score'])}; {flag})",
                            f"{best_other['method']} ({number(best_other['score'])})",
                            f"{pair['paired_seed_rows']}/85",
                            number(pair[f"macro_mean_delta_{metric}"], 4, True), ci,
                            (f"{pair['first_higher_datasets']}/{pair['paired_datasets']}"
                             if metric == "auroc" else "—")])
    return md_table(["Features", "% fit", "Metric", "Top panel-complete NSA",
                     "Top fixed-field other", "Matched", "NSA − other",
                     "95% CI", "AUROC-win datasets"], out)


def dual_measurement(data: dict) -> str:
    out = []
    method = "NSA/dual"
    for rep in REPS:
        for pct in FRACTIONS:
            s = data["summary_index"][(rep, pct, method)]
            change = data["change_index"].get((rep, pct, method))
            rankings = [data["common_index"].get(
                (rep, pct, metric, "common_baselines_plus_focal_nsa", method))
                for metric in ("auroc", "ap")]
            out.append([rep, pct, f"{s['qualified_success_seed_rows']}/85",
                        number(s["macro_auroc_qualified_available"]),
                        number(s["macro_ap_qualified_available"]),
                        *[f"{r['rank']}/{r['field_size']}" if r else "—" for r in rankings],
                        (f"{number(change['macro_mean_delta_auroc'], 4, True)} "
                         f"({change['paired_seed_rows']}/85)") if change else "reference",
                        (f"{number(change['macro_mean_delta_ap'], 4, True)} "
                         f"({change['paired_seed_rows']}/85)") if change else "reference"])
    return md_table(["Features", "% fit", "Strict success", "AUROC available",
                     "AP available", "AUROC rank", "AP rank",
                     "Paired ΔAUROC vs 100%", "Paired ΔAP vs 100%"], out)


def nsa_direction_summary(data: dict) -> str:
    out = []
    tol = 1e-12
    for rep in REPS:
        for pct in (75, 50, 25, 10, 5):
            for metric in ("auroc", "ap"):
                eligible = [data["change_index"][(rep, pct, m)] for m in NSA_METHODS
                            if int(data["change_index"][(rep, pct, m)]["paired_seed_rows"]) == 85]
                gains = [r for r in eligible if float(r[f"macro_mean_delta_{metric}"]) > tol]
                declines = [r for r in eligible if float(r[f"macro_mean_delta_{metric}"]) < -tol]
                top_gain = max(gains, key=lambda r: float(r[f"macro_mean_delta_{metric}"])) if gains else None
                top_loss = min(declines, key=lambda r: float(r[f"macro_mean_delta_{metric}"])) if declines else None
                out.append([rep, pct, metric.upper(), f"{len(eligible)}/13",
                            len(gains), len(declines), len(eligible) - len(gains) - len(declines),
                            (f"{top_gain['method']} "
                             f"{number(top_gain[f'macro_mean_delta_{metric}'], 4, True)}")
                            if top_gain else "—",
                            (f"{top_loss['method']} "
                             f"{number(top_loss[f'macro_mean_delta_{metric}'], 4, True)}")
                            if top_loss else "—"])
    return md_table(["Features", "% fit", "Metric", "Fully matched NSA",
                     "Increase", "Decrease", "Tie", "Largest increase", "Largest decrease"], out)


def dual_relative_change(data: dict) -> str:
    """Hold each comparator method fixed at 100%, 10%, and 5%."""
    out = []
    for rep in REPS:
        for metric in ("auroc", "ap"):
            baseline_100 = [r for r in data["common_field_ranks"]
                            if r["representation"] == rep and int(r["retention_pct"]) == 100
                            and r["metric"] == metric and r["field"] == "common_baselines"]
            leader = max(baseline_100, key=lambda r: float(r["score"])) if baseline_100 else None
            references = {"IForest": ["IForest"], "KNN": ["KNN"]}
            if leader:
                references.setdefault(leader["method"], []).append("top fixed other at 100%")
            for pct in NEW_FRACTIONS:
                for baseline, labels in references.items():
                    row = data["relative_index"][(rep, pct, "NSA/dual", baseline)]
                    lo = row[f"cluster_bootstrap_95_low_{metric}"]
                    hi = row[f"cluster_bootstrap_95_high_{metric}"]
                    ci = (f"[{number(lo, 3, True)}, {number(hi, 3, True)}]"
                          if lo != "" and hi != "" else "interval not estimated")
                    out.append([rep, pct, metric.upper(), baseline, ", ".join(labels),
                                f"{row['paired_seed_rows']}/85", row["paired_datasets"],
                                number(row[f"macro_mean_delta_{metric}"], 4, True), ci,
                                f"{row['first_higher_datasets' if metric == 'auroc' else 'ap_gain_datasets']}/{row['paired_datasets']}"])
    return md_table(["Features", "% fit", "Metric", "Fixed comparator",
                     "Reference", "Four-success matches", "Datasets",
                     "Relative change", "95% CI", "Gain datasets"], out)


def dual_dataset_table(data: dict, rep: str, pct: int) -> str:
    subset = [r for r in data["dataset_nsa_ranks"]
              if r["representation"] == rep and int(r["retention_pct"]) == pct
              and r["method"] == "NSA/dual"]
    out = []
    for r in sorted(subset, key=lambda row: row["dataset"]):
        def rank_and_top(metric: str):
            suffix = "" if metric == "auroc" else "_ap"
            value = r[f"rank_fixed_baselines_plus_focal{suffix}"]
            if value == "":
                return "—", "—"
            position, size = int(value), int(r["fixed_baselines_plus_focal_size"])
            return f"{position}/{size}", f"{100 * (size - position + 1) / size:.1f}%"
        auc_rank, auc_top = rank_and_top("auroc")
        ap_rank, ap_top = rank_and_top("ap")
        out.append([r["dataset"], f"{r['qualified_seed_rows']}/5",
                    number(r["mean_auroc_five_seeds"]), number(r["mean_ap_five_seeds"]),
                    auc_rank, auc_top, ap_rank, ap_top, r["status_counts"]])
    if len(out) != 17:
        raise ValueError(f"incomplete NSA/dual dataset panel: {rep}/{pct}")
    return md_table(["Dataset", "Strict success", "AUROC", "AP", "AUROC rank",
                     "AUROC top %", "AP rank", "AP top %", "Statuses"], out)


def dataset_strategy_coverage(data: dict) -> str:
    out = []
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            for nsa in NSA_METHODS:
                subset = [r for r in data["dataset_nsa_ranks"]
                          if r["representation"] == rep and int(r["retention_pct"]) == pct
                          and r["method"] == nsa]
                full = [r for r in subset if r["full_dataset_coverage"] == "True"]
                fixed = [r for r in full if r["rank_fixed_all_common"] != ""]
                out.append([rep, pct, nsa, len(full), len(fixed),
                            sum(int(r["qualified_seed_rows"]) for r in subset)])
    return md_table(["Features", "% fit", "NSA", "Datasets with 5/5",
                     "Datasets in six-size common field", "Successful seed runs / 85"], out)


def raw_sensitivity(data: dict) -> str:
    out = []
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            panel = [r for r in data["raw_vs_strict"]
                     if r["representation"] == rep and int(r["retention_pct"]) == pct]
            for label, count, score in (
                ("raw recorded", "raw_success_seed_rows", "raw_macro_auroc_available"),
                ("strict ≤15 s", "strict_success_seed_rows", "strict_macro_auroc_available"),
            ):
                eligible = [r for r in panel if int(r[count]) == 85]
                for family, allowed in (("NSA", lambda m: m.startswith("NSA/")),
                                        ("Other", lambda m: not m.startswith("NSA/"))):
                    candidates = [r for r in eligible if allowed(r["method"])]
                    top = max(candidates, key=lambda r: float(r[score])) if candidates else None
                    out.append([rep, pct, label, family, len(candidates),
                                top["method"] if top else "—",
                                number(top[score]) if top else "—"])
    return md_table(["Features", "% fit", "Qualification", "Family",
                     "Full coverage methods", "Top AUROC method", "AUROC"], out)


def raw_fixed_field_ranks(data: dict) -> str:
    out = []
    for rep in REPS:
        for pct in NEW_FRACTIONS:
            for nsa in NSA_METHODS:
                raw = data["raw_common_index"].get(
                    (rep, pct, "auroc", "raw_common_baselines_plus_focal_nsa", nsa))
                strict = data["common_index"].get(
                    (rep, pct, "auroc", "common_baselines_plus_focal_nsa", nsa))
                if not raw and not strict:
                    continue
                out.append([rep, pct, nsa,
                            f"{raw['rank']}/{raw['field_size']}" if raw else "—",
                            number(raw["score"]) if raw else "—",
                            f"{strict['rank']}/{strict['field_size']}" if strict else "—",
                            number(strict["score"]) if strict else "—",
                            raw["focal_eligible_all_six"] if raw else "—",
                            strict["focal_eligible_all_six"] if strict else "—"])
    return md_table(["Features", "% fit", "NSA", "Raw rank", "Raw AUROC",
                     "Strict rank", "Strict AUROC", "Raw six-size eligible",
                     "Strict six-size eligible"], out)


def adaptive_batch_section(data: dict) -> str:
    """Render isolated trained-AE/VAE sensitivity only after its full audit exists."""
    if "adaptive" not in data:
        return ""
    adaptive = data["adaptive"]
    audit = adaptive["audit"]
    primary = data["summary_index"]
    fields = audit["fixed_fields"]
    field_rows = []
    for rep in REPS:
        original = fields["primary"][rep]
        revised = fields["adaptive"][rep]
        field_rows.append([rep, len(original), len(revised),
                           ", ".join(m for m in revised if m not in original) or "—",
                           ", ".join(m for m in original if m not in revised) or "—"])

    ae_rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            for method in ("AutoEncoder", "VAE"):
                r = adaptive["summary"][(rep, pct, method)]
                ae_rows.append([rep, pct, method,
                                f"{r['original_qualified_success_seed_rows']}/85",
                                f"{r['adaptive_qualified_success_seed_rows']}/85",
                                r["adaptive_source_seed_rows"],
                                number(r["adaptive_macro_auroc_qualified_available"]),
                                number(r["adaptive_macro_ap_qualified_available"]),
                                r["adaptive_status_counts"]])

    dual_rows = []
    for rep in REPS:
        for pct in FRACTIONS:
            auc = adaptive["ranks"][(rep, pct, "auroc", "NSA/dual")]
            ap = adaptive["ranks"][(rep, pct, "ap", "NSA/dual")]
            dual_rows.append([rep, pct, f"{auc['nsa_qualified_seed_rows']}/85",
                              number(auc["nsa_macro_score"]), number(ap["nsa_macro_score"]),
                              (f"{auc['primary_fixed_baseline_rank']}/{auc['primary_field_size_with_focal']}"
                               if auc["primary_fixed_baseline_rank"] else "—"),
                              (f"{auc['adaptive_fixed_baseline_rank']}/{auc['adaptive_field_size_with_focal']}"
                               if auc["adaptive_fixed_baseline_rank"] else "—"),
                              (f"{ap['primary_fixed_baseline_rank']}/{ap['primary_field_size_with_focal']}"
                               if ap["primary_fixed_baseline_rank"] else "—"),
                              (f"{ap['adaptive_fixed_baseline_rank']}/{ap['adaptive_field_size_with_focal']}"
                               if ap["adaptive_fixed_baseline_rank"] else "—")])

    direction = []
    for rep in REPS:
        eligible = data["audit"]["common_fields"][rep]["nsa"]
        for pct in NEW_FRACTIONS:
            for metric in ("auroc", "ap"):
                changes = []
                for method in eligible:
                    full = adaptive["ranks"][(rep, 100, metric, method)]
                    reduced = adaptive["ranks"][(rep, pct, metric, method)]
                    if not full["adaptive_fixed_baseline_rank"] or not reduced["adaptive_fixed_baseline_rank"]:
                        raise ValueError("six-size eligible NSA lost adaptive rank")
                    changes.append((method, int(full["adaptive_fixed_baseline_rank"])
                                    - int(reduced["adaptive_fixed_baseline_rank"])))
                direction.append([rep, pct, metric.upper(), len(changes),
                                  sum(delta > 0 for _, delta in changes),
                                  sum(delta < 0 for _, delta in changes),
                                  sum(delta == 0 for _, delta in changes),
                                  ", ".join(f"{m} ({delta:+d})" for m, delta in changes if delta > 0) or "—"])

    leader_rows = []
    for rep in REPS:
        baseline_roster = fields["adaptive"][rep]
        for pct in NEW_FRACTIONS:
            for metric in ("auroc", "ap"):
                def baseline_score(method):
                    if method in ("AutoEncoder", "VAE"):
                        value = adaptive["summary"][(rep, pct, method)][
                            f"adaptive_macro_{metric}_qualified_available"]
                    else:
                        value = primary[(rep, pct, method)][f"macro_{metric}_qualified_available"]
                    if value == "" or value is None:
                        raise ValueError("adaptive complete baseline has no macro score")
                    return float(value)
                top_base = max(baseline_roster, key=baseline_score) if baseline_roster else None
                focal = [m for m in NSA_METHODS
                         if int(primary[(rep, pct, m)]["qualified_success_seed_rows"]) == 85]
                top_nsa = max(focal, key=lambda m: float(primary[(rep, pct, m)][
                    f"macro_{metric}_qualified_available"])) if focal else None
                rank_row = adaptive["ranks"][(rep, pct, metric, top_nsa)] if top_nsa else None
                nsa_score = float(primary[(rep, pct, top_nsa)][
                    f"macro_{metric}_qualified_available"]) if top_nsa else None
                base_score = baseline_score(top_base) if top_base else None
                leader_rows.append([rep, pct, metric.upper(), top_nsa or "—",
                                    number(nsa_score),
                                    (f"{rank_row['adaptive_fixed_baseline_rank']}/"
                                     f"{rank_row['adaptive_field_size_with_focal']}")
                                    if rank_row and rank_row["adaptive_fixed_baseline_rank"] else "—",
                                    top_base or "—", number(base_score),
                                    number(nsa_score - base_score, 4, True)
                                    if nsa_score is not None and base_score is not None else "—"])

    return "\n\n".join([
        "## Isolated adaptive-batch sensitivity: AutoEncoder and VAE",
        f"This separate sensitivity contains **{audit['actual_records']} fresh rows** "
        f"in **{audit['actual_cells']} cells**; the training-batch adapter applied "
        f"to **{audit['adapter_applied_rows']} rows**. It changes only the two "
        "declared PyOD training batch sizes to min(32, retained normal rows), "
        "keeps their inference batch size at 32, and records observed training "
        "forward calls. Other methods and all primary scores are reused by "
        "checked hash. The following ranks belong to a newly qualified "
        "six-size baseline field; they never overwrite the primary ranking.",
        "### Fixed baseline field membership",
        md_table(["Features", "Primary other N", "Adaptive other N", "Added", "Removed"],
                 field_rows),
        "### AutoEncoder and VAE coverage and scores",
        "An available-case score with less than 85/85 qualified outcomes is "
        "descriptive and never receives a complete-field rank.",
        md_table(["Features", "% fit", "Method", "Primary strict", "Adaptive strict",
                  "Fresh adaptive rows", "Adaptive AUROC available", "Adaptive AP available",
                  "Adaptive statuses"], ae_rows),
        "### NSA/dual scores and primary versus adaptive AUROC/AP ranks",
        "The NSA/dual scores are identical in both columns of this sensitivity; "
        "rank or field-size changes reflect the separately trained baselines.",
        md_table(["Features", "% fit", "Dual strict", "Dual AUROC", "Dual AP",
                  "Primary AUROC rank", "Adaptive AUROC rank", "Primary AP rank",
                  "Adaptive AP rank"], dual_rows),
        "### All-six-eligible NSA adaptive rank changes versus 100%",
        "Each count uses the same adaptive six-size baseline roster in its "
        "100% and reduced panels. Positive rank change means a better ordinal "
        "position. NSA configurations without full six-size coverage are "
        "excluded from these rank-change counts, though they remain in the "
        "312-row sensitivity CSV.",
        md_table(["Features", "% fit", "Metric", "Eligible NSA", "Better", "Worse",
                  "Same", "Improved methods and positions"], direction),
        "### Best complete NSA against the adaptive fixed-field baseline leader",
        "Leaders are selected by their observed macro score separately per "
        "metric and panel. The gap is a descriptive difference of complete "
        "17-dataset macro means without a paired interval in this sensitivity.",
        md_table(["Features", "% fit", "Metric", "Top NSA", "NSA score",
                  "Adaptive rank", "Top adaptive other", "Other score",
                  "NSA − other"], leader_rows),
        "These ranks remain conditional on normal-only initial fitting, "
        "refitted PCA where applicable, the measured ≤15-second rule, and "
        "query-batch behavior of the 12 audited non-pointwise scorers. Dataset "
        "variation and incomplete outcomes remain visible in the primary tables. "
        "A separate startup-timing sidecar records possible brief overlap between "
        "an independent source/PCA audit and the first eight adaptive worker cells. "
        "Exact per-method start times were not logged; all 16 methods in those "
        "cells succeeded in under 5.1 seconds measured, below the 15-second "
        "qualification threshold. Their runtime values may reflect transient load. "
        f"The sidecar SHA-256 is `{adaptive['startup_timing_overlap_sha256']}`."
    ])


def render(data: dict) -> str:
    a = data["audit"]
    all_fits = [int(r[f"fit_normals_{pct}_min"]) for r in data["dataset_inventory"]
                for pct in NEW_FRACTIONS]
    new_recovery = a["new_recovery"]["new_event_count"]
    adaptive_evidence = (
        "The isolated `adaptive_batch/analysis/` directory additionally contains "
        "the 740-row cell comparison, 24 AE/VAE panel summaries, 312 focal "
        "NSA AUROC/AP rank rows, and its SHA-validated audit."
        if "adaptive" in data else "")
    independent_evidence = (
        "A separate independent audit reconstructed all 340 new cells and "
        "confirmed every initial fit pool contains only normal rows; its "
        f"SHA-256 is `{data['independent_audit_sha256']}`. Raw JSONL retains "
        "aggregate AUROC/AP but not per-test score vectors, so that audit "
        "cannot recompute those metric values independently."
        if "independent_audit_sha256" in data else "")
    return "\n\n".join([
        "# Normal-only PyOD retention benchmark: six training sizes",
        "**Private review draft.** This report combines the frozen 100/75/50/25% runs "
        "with fresh 10/5% runs. It describes the observed implementation and does not "
        "assert general NSA superiority.",
        "## Executive measurements",
        "The following measurements use the strict recorded-success, measured "
        "≤15-second rule. A score marked available can be based on fewer than "
        "85 runs; its coverage is shown, and it receives no full-field rank. "
        "The matched change columns compare each method with its own 100% run "
        "on the same dataset and seed.",
        "### NSA/dual across all six sizes", dual_measurement(data),
        "### Best complete NSA and fixed-field other detectors at 10% and 5%",
        "These leaders are selected by observed outcomes separately for AUROC "
        "and AP. Their paired gaps are exploratory and do not establish general "
        "superiority.",
        top_focal_comparison(data, NEW_FRACTIONS),
        "### Change in NSA/dual's gap to the same comparator",
        "Relative change is (Dual − comparator) at the reduced size minus "
        "(Dual − comparator) at 100%, using only dataset/seed cells where all "
        "four outcomes succeeded. A positive value means Dual gained relative "
        "to that comparator as normal training rows were removed; it does not "
        "necessarily mean Dual won at the reduced size. IForest, KNN, and the "
        "strongest fixed-field other method selected at 100% are held by name "
        "across sizes. Confidence intervals are descriptive and unadjusted; "
        "the full 2,236-row CSV includes every NSA–baseline pairing.",
        dual_relative_change(data),
        "### NSA increases and decreases relative to 100%",
        "Only configurations with all 85 matched dataset/seed pairs enter the "
        "increase/decrease counts. Detailed scores, coverage, and intervals for "
        "every NSA are below and in the CSV. A positive within-method change "
        "does not imply a win over another detector.",
        nsa_direction_summary(data),
        "## Scope and controls",
        f"The exact grid contains **{a['expected_records']:,} records** in "
        f"**{a['expected_cells']:,} dataset/seed/representation/size cells**: "
        "17 ODDS datasets × five seeds (42–46) × native or PCA(2) features × "
        "six retained fractions (100, 75, 50, 25, 10, 5%) × 56 fixed configurations "
        "(13 NSA configurations from 11 strategy families and 43 other PyOD entries). "
        "Dataset labels stratify the split and select normal training rows; "
        "anomalies in the original training partition are discarded. Benchmark "
        "preprocessing and each initial detector.fit call receive retained "
        "normal features only, without a target label. A "
        "separate mixed held-out test set uses labels to calculate AUROC and "
        "average precision (AP). "
        "The same seed-fixed normal parent pool, capped at 600, supplies nested "
        "subsets at all six sizes; an implementation minimum of two rows applies. "
        f"The smallest observed 10/5% fit pool has **{min(all_fits)} normal rows**. "
        "Preprocessing and PCA are fitted again on each retained normal subset, "
        "so representation drift is part of the measured pipeline.",
        independent_evidence,
        "## Actual source and fit counts", fit_inventory(data),
        "## Time qualification and coverage",
        f"Raw successes: **{a['raw_success_records']:,}**; measured ≤15-second "
        f"qualified successes: **{a['strict_success_records']:,}**; successful "
        f"scores beyond that measured duration: **{a['total_overbudget_success_records']:,}** "
        f"(historical {a['historical_overbudget_success_records']}, new "
        f"{a['new_overbudget_success_records']}). Scores beyond 15 seconds remain "
        "valid normal-only outcomes in the raw sensitivity table but do not enter the "
        "strict-budget ranks, means, or paired comparisons. The Python alarm is "
        "nominal: native callbacks may swallow it. Measured fit plus score time "
        "includes optional Torch setup before the alarm starts. Failures, timeouts, "
        "inapplicability, partial AUROC on a timeout, and incomplete seed coverage "
        "are never scored as zero.",
        f"The six-size field requires **85/85 qualified dataset/seed outcomes at "
        f"every size** in a representation. A baseline-only field contains "
        f"the qualifying other detectors; each NSA is inserted individually "
        f"when it has 85/85 success in that panel. Its focal six-size "
        f"eligibility is flagged separately; cross-size rank changes require "
        f"that eligibility. A separate inductive sensitivity "
        f"removes all {len(a['batch_dependent_scoring_methods'])} audited "
        f"batch-dependent scoring methods. Panel-specific fields are shown only "
        f"as sensitivity because their membership changes with size.",
        field_summary(data),
        "## All configurations at 10% and 5%",
        "Every method is shown, including failed, timed-out, inapplicable, and "
        "partially covered configurations. Scores marked available use successful "
        "dataset/seed cells only and should not be compared across differing "
        "coverage. A six-size all-method rank requires 85/85 qualified outcomes "
        "at every size. 'Unsuitable in X/85 tested cases' means X cases did not "
        "produce a measured ≤15-second score with these fit sizes, dimensions, "
        "constructor settings, and runtime. Zero qualified scores are labeled "
        "'unsuitable under tested configuration.' This does not establish that "
        "the detector family cannot train on normal data in another configuration. "
        "The cause column counts the derived non-success statuses: labeled-class "
        "requirement, dimension mismatch, model construction or fit/scoring error, "
        "reported timeout, or measured overbudget success. Partially covered "
        "available-case means receive no coverage-qualified rank. Raw status "
        "counts and scores are preserved in the JSONL and global audit.",
        "### Suitability coverage overview", suitability_overview(data),
        "### Native features, 10% fit", all_method_new_panel(data, "native", 10),
        "### Native features, 5% fit", all_method_new_panel(data, "native", 5),
        "### PCA(2), 10% fit", all_method_new_panel(data, "pca2", 10),
        "### PCA(2), 5% fit", all_method_new_panel(data, "pca2", 5),
        "## Fixed-field NSA rank and score",
        "Each cell gives the NSA rank after insertion into the fixed six-size "
        "eligible baseline field, its absolute score, and whether that NSA also "
        "qualifies at all six sizes. If it lacks 85/85 success in that panel, "
        "the rank is blank and coverage is shown. The all-method common-field "
        "ranks are exported separately. Competition ranks count scores "
        "strictly more than 1e−12 above the focal score; a smaller rank fraction "
        "means a higher position, not an accuracy percentage.",
        "### Native features: AUROC", nsa_rank_table(data, "native", "auroc"),
        "### PCA(2): AUROC", nsa_rank_table(data, "pca2", "auroc"),
        "### Native features: AP", nsa_rank_table(data, "native", "ap"),
        "### PCA(2): AP", nsa_rank_table(data, "pca2", "ap"),
        "## Best other PyOD detectors under fixed and changing fields",
        "The fixed-field leaders are selected descriptively from methods that "
        "qualified at all six sizes. The panel field is a changing-field sensitivity.",
        top_baselines(data),
        "## Top NSA and top fixed-field other method at each size",
        "Leaders are chosen descriptively using each panel's observed score; "
        "the comparator and focal can change between sizes and between AUROC "
        "and AP. Each difference is then calculated on matched dataset/seed "
        "outcomes, with an unadjusted dataset-cluster interval. This selected "
        "leader comparison is exploratory and is not a test of general supremacy.",
        top_focal_comparison(data),
        "## Fixed-field rank changes versus 100%", rank_changes(data),
        "## Paired NSA score changes versus 100%",
        "Differences use only matched successful dataset/seed cells. CIs are "
        "descriptive, unadjusted 95% dataset-cluster bootstrap intervals (5,000 "
        "draws by default); no interval is estimated from fewer than five "
        "paired datasets. Positive Δ means the smaller training pool did better "
        "on that metric. A within-method change alone does not establish "
        "superiority over another detector.",
        paired_nsa(data),
        "## NSA versus the strongest fixed-field other detector at 10% and 5%",
        "The comparator is chosen by its observed AUROC in the common baseline "
        "field at that size. The full CSV contains all 6,708 NSA-versus-other "
        "same-size comparisons, including AP and uncertainty. This concise table "
        "shows AUROC point differences and matched coverage.",
        new_size_pair_leaders(data),
        "## Dataset strategy coverage at the new sizes",
        "All 13 configurations, representing all 11 strategy families, remain "
        "in the dataset-level CSV. A dataset rank is blank if fewer than five "
        "qualified seed outcomes are available. Fixed baseline-plus-focal ranks "
        "hold the baseline comparison field across all six sizes; locally "
        "qualified ranks use a changing field and are explicitly secondary.",
        dataset_strategy_coverage(data),
        "## NSA/dual by dataset at 10% and 5%",
        "Each score is the mean across all five seeds only when the dataset has "
        "5/5 strict successes. The comparator field is the same six-size eligible "
        "baseline roster plus NSA/dual; its size is printed with each rank. "
        "Top % is a rank-derived display value, 100 × (field size − rank + 1) "
        "/ field size. It is not a statistical confidence level. Missing ranks "
        "or scores mean incomplete coverage, not zero performance.",
        "### Native features, 10% fit", dual_dataset_table(data, "native", 10),
        "### Native features, 5% fit", dual_dataset_table(data, "native", 5),
        "### PCA(2), 10% fit", dual_dataset_table(data, "pca2", 10),
        "### PCA(2), 5% fit", dual_dataset_table(data, "pca2", 5),
        "## Raw-success sensitivity at the new sizes",
        "These full-coverage leaders use every recorded successful score, "
        "including successes over the measured 15-second budget. Their field "
        "can differ from the strict field. Available-case scores of incomplete "
        "methods are present in the CSV, with their coverage.",
        raw_sensitivity(data),
        "### Raw versus strict fixed-baseline ranks", raw_fixed_field_ranks(data),
        adaptive_batch_section(data),
        "## Recovery provenance and interpretation limits",
        f"The new 10/5% root has **{new_recovery}** externally recorded "
        "watchdog intervention(s). The active monitor mode recorded in its "
        f"supervision manifest is **{a['new_supervision_manifest']['active_monitor_mode']}**. "
        "A read-only monitor observation never creates an outcome row. Any "
        "intervention counted here requires a new-root snapshot, hash, "
        "≥600-second raw-file age, and matching no-score timeout row; it is "
        "separate from the frozen historical recovery evidence. The historical "
        "results and 118 measured overbudget successes were read from the "
        "unchanged four-size files after checking their published package hashes.",
        "XGBOD and DevNet need labeled anomalies for their intended supervised "
        "training, so they are inapplicable under this protocol; supplying "
        "all-zero labels would not create the missing anomaly class. Other "
        "failures may arise from geometry, tiny samples, implementation errors, "
        "or runtime. MAD is one-dimensional, Voronoi NSA accepts two or three "
        "features, and deterministic NSA needs two; binary NSA can fail under "
        "the PCA(2) constructor. LUNAR and GAAL variants may synthesize "
        "negative examples internally without receiving real anomaly rows.",
        "AutoEncoder and VAE retain the pinned PyOD default training batch "
        "size of 32. Their shared deep-learning loader uses drop_last=True; "
        "when fewer than 32 normal rows remain, it yields no training batch "
        "and the training loop errors before learning. Such errors show a "
        "mini-batch constraint, not a need for labeled outliers. Any separate "
        "adaptive-batch sensitivity is outside these primary fixed-constructor "
        "results and must be labeled separately.",
        "For 12 audited batch-dependent scorers, decision_function can depend "
        "on other rows in the mixed test query batch. The normal-only constraint "
        "describes benchmark preprocessing and the initial fit call, not every "
        "internal calculation during scoring. The inductive-only field removes "
        "these 12 configurations for a stricter query-independent comparison.",
        "The frozen constructors include reduced training epochs or ensembles "
        "for some expensive methods. Query-batch scoring, the 600/2,000 caps, "
        "refitted PCA, one clean-normal selection rule, fixed hyperparameters, "
        "and very small five-percent pools limit generalization. Rankings are "
        "conditional on coverage and field definition. There are many exploratory "
        "comparisons, and the intervals are not multiplicity adjusted. No "
        "detector is declared generally superior by this benchmark alone.",
        "## Machine-readable evidence",
        "The `analysis/` directory contains `extreme_retention_method_summary.csv`, "
        "`extreme_retention_common_field_ranks.csv`, `extreme_retention_panel_ranks.csv`, "
        "`extreme_retention_rank_change.csv`, `extreme_retention_change_vs_100.csv`, "
        "`extreme_retention_nsa_vs_baseline.csv`, "
        "`extreme_retention_relative_change_vs_100.csv`, "
        "`extreme_retention_dataset_nsa_ranks.csv`, "
        "`extreme_retention_dataset_inventory.csv`, `extreme_retention_raw_vs_strict.csv`, "
        "`extreme_retention_raw_common_ranks.csv`, "
        "`extreme_retention_focal_metrics.json`, and the analysis/recovery audit "
        "JSON. The companion `plot.py` renders native and PCA(2) PNG summaries "
        "from these validated tables using only fully covered points. Historical "
        "raw rows and the new "
        "10/5% raw rows remain in separate roots. This report accompanies the "
        "public benchmark evidence.",
        adaptive_evidence,
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=HERE / "analysis")
    parser.add_argument("--output", type=Path, default=HERE / "analysis" / "extreme_retention_report.md")
    args = parser.parse_args()
    data = load(args.analysis_dir)
    report = render(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
