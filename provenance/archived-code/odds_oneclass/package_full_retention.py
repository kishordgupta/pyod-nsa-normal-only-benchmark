"""Package a *completed* fresh 56-method normal-only retention benchmark.

This script refuses an incomplete or mixed-protocol grid. It only reads
production `full_retention/raw/*.jsonl` and `logs/*.log`; isolated smoke files
are never discovered. Final files go to outputs with the distinct
`pyod_nsa_full_retention_` prefix. Do not run until the report and analysis
tables are complete.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "full_retention"
OUTPUTS = HERE.parent.parent / "outputs"
PREFIX = "pyod_nsa_full_retention"
ANALYSIS_AUDIT_NAME = "full_retention_analysis_audit.json"
CORE_SCRIPTS = (
    "runner.py", "model_factory.py", "loader.py", "full_retention_sweep.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def record_name(dataset: str, representation: str, seed: int,
                fraction: float) -> str:
    suffix = "" if fraction == 1.0 else f"_retain{round(fraction * 100)}"
    return f"{dataset}_{representation}_seed{seed}{suffix}"


def valid_production_command(line: str) -> bool:
    tokens = line.split()
    try:
        timeout = float(tokens[tokens.index("--method-timeout") + 1])
        method_filter = tokens[tokens.index("--method-filter") + 1]
    except (ValueError, IndexError):
        return False
    return timeout == 15.0 and method_filter == "all" and any(
        Path(token).name == "runner.py" for token in tokens
    )


def exposed_analysis_name(path: Path) -> str:
    suffix = path.name.removeprefix("full_retention_")
    return f"{PREFIX}_{suffix}"


def check_complete_grid(run_root: Path, manifest: dict) -> tuple[list[Path], list[Path], dict]:
    if manifest.get("protocol_version") != "retention-v2-parent-order":
        raise ValueError("Unexpected split/protocol version")
    if float(manifest.get("method_timeout_seconds", -1)) != 15.0:
        raise ValueError("Production package requires the declared 15 s method budget")
    datasets = [item["name"] for item in manifest["datasets"]]
    seeds = manifest["seeds"]
    fractions = manifest["fractions_retained"]
    representations = manifest["representations"]
    methods = manifest["methods"]
    if (len(datasets), len(seeds), len(fractions), len(representations), len(methods)) != (17, 5, 4, 2, 56):
        raise ValueError("Manifest does not declare the intended 17×5×4×2×56 grid")
    if set(seeds) != set(range(42, 47)) or set(fractions) != {1.0, 0.75, 0.5, 0.25}:
        raise ValueError("Unexpected seeds or retained fractions")
    if set(representations) != {"native", "pca2"} or len(set(methods)) != 56:
        raise ValueError("Unexpected representations or duplicate methods")
    expected_files = {
        f"{record_name(dataset, representation, seed, fraction)}.jsonl"
        for dataset in datasets for seed in seeds for representation in representations
        for fraction in fractions
    }
    raw_dir = run_root / "raw"
    raw_paths = sorted(raw_dir.glob("*.jsonl"))
    actual_files = {path.name for path in raw_paths}
    if actual_files != expected_files:
        raise ValueError(
            f"Production raw grid differs: missing={sorted(expected_files - actual_files)[:10]}, "
            f"extra={sorted(actual_files - expected_files)[:10]}"
        )
    expected_records = len(expected_files) * len(methods)
    if expected_records != manifest.get("expected_records") or expected_records != 38_080:
        raise ValueError("Manifest record count mismatch")
    log_paths = sorted((run_root / "logs").glob("*.log"))
    expected_logs = {name.removesuffix(".jsonl") + ".log" for name in expected_files}
    actual_logs = {path.name for path in log_paths}
    if actual_logs != expected_logs:
        raise ValueError(
            f"Production log grid differs: missing={sorted(expected_logs - actual_logs)[:10]}, "
            f"extra={sorted(actual_logs - expected_logs)[:10]}"
        )
    status_counts: Counter[str] = Counter()
    status_by_panel: dict[str, Counter[str]] = defaultdict(Counter)
    overbudget_success_by_panel: Counter[str] = Counter()
    overbudget_success_records = 0
    parent_hashes: dict[tuple[str, int], str] = {}
    test_hashes: dict[tuple[str, int], str] = {}
    fit_hashes: dict[tuple[str, int, float], str] = {}
    normal_counts: dict[tuple[str, int], int] = {}
    total = 0
    expected_methods = set(methods)
    for dataset in datasets:
        for seed in seeds:
            for representation in representations:
                for fraction in fractions:
                    name = record_name(dataset, representation, seed, fraction)
                    raw_path = raw_dir / f"{name}.jsonl"
                    log_path = run_root / "logs" / f"{name}.log"
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")
                    commands = [line for line in log_text.splitlines() if line.startswith("COMMAND")]
                    if not commands or not all(valid_production_command(line)
                                               for line in commands):
                        raise ValueError(f"Method budget or all-method command absent in {log_path}")
                    cell_methods: set[str] = set()
                    with raw_path.open(encoding="utf-8") as stream:
                        for line_no, line in enumerate(stream, 1):
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(f"Malformed {raw_path}:{line_no}: {exc}") from exc
                            method = row.get("method")
                            if method not in expected_methods or method in cell_methods:
                                raise ValueError(f"Unexpected or duplicate method {raw_path}:{line_no}: {method}")
                            cell_methods.add(method)
                            total += 1
                            checks = {
                                "dataset": dataset,
                                "seed": seed,
                                "representation": representation,
                                "retention_fraction": fraction,
                                "reduction_fraction": 1.0 - fraction,
                                "protocol_version": manifest["protocol_version"],
                                "fit_outliers": 0,
                                "train_cap": manifest["train_cap"],
                                "test_cap": manifest["test_cap"],
                                "family": manifest["method_meta"][method]["family"],
                            }
                            mismatch = [key for key, expected in checks.items()
                                        if row.get(key) != expected]
                            if mismatch:
                                raise ValueError(f"Protocol mismatch {raw_path}:{line_no}: {mismatch}")
                            base = (dataset, seed)
                            parent = row.get("fit_parent_index_sha256")
                            test = row.get("test_index_sha256")
                            fit = row.get("fit_index_sha256")
                            ordered = row.get("fit_index_ordered_sha256")
                            if not all(isinstance(value, str) and len(value) == 64
                                       for value in (parent, test, fit, ordered)):
                                raise ValueError(f"Missing row-order or split hash {raw_path}:{line_no}")
                            if base in parent_hashes and parent_hashes[base] != parent:
                                raise ValueError(f"Parent-fit indices drifted for {base}")
                            if base in test_hashes and test_hashes[base] != test:
                                raise ValueError(f"Held-out test rows drifted for {base}")
                            fit_key = (dataset, seed, fraction)
                            if fit_key in fit_hashes and fit_hashes[fit_key] != fit:
                                raise ValueError(f"Retained fit subset differs across representations for {fit_key}")
                            parent_hashes[base] = parent
                            test_hashes[base] = test
                            fit_hashes[fit_key] = fit
                            original = row.get("original_fit_normals")
                            if not isinstance(original, int) or original < 2:
                                raise ValueError(f"Invalid parent normal count {raw_path}:{line_no}")
                            if base in normal_counts and normal_counts[base] != original:
                                raise ValueError(f"Parent normal count drifted for {base}")
                            normal_counts[base] = original
                            expected_fit = original if fraction == 1.0 else max(2, math.floor(original * fraction))
                            if row.get("fit_normals") != expected_fit:
                                raise ValueError(f"Incorrect retained fit count {raw_path}:{line_no}")
                            status = row.get("status")
                            allowed_statuses = {
                                "success", "error", "timeout", "build_error",
                                "requires_labeled_outliers", "requires_1d",
                                "requires_2d", "requires_2d_or_3d",
                            }
                            if not isinstance(status, str) or status not in allowed_statuses:
                                raise ValueError(f"Unknown or unfinished outcome status {raw_path}:{line_no}: {status}")
                            applicability = manifest["method_meta"][method]["applicability"]
                            if applicability != "applicable" and status != applicability:
                                raise ValueError(f"Declared inapplicability differs from result {raw_path}:{line_no}")
                            if applicability == "applicable" and status == "requires_labeled_outliers":
                                raise ValueError(f"Unexpected labeled-outlier status {raw_path}:{line_no}")
                            if status == "requires_1d" and method != "MAD":
                                raise ValueError(f"Unexpected one-feature status {raw_path}:{line_no}")
                            if status == "requires_2d" and method != "NSA/deterministic":
                                raise ValueError(f"Unexpected two-feature status {raw_path}:{line_no}")
                            if status == "requires_2d_or_3d" and method != "NSA/voronoi":
                                raise ValueError(f"Unexpected Voronoi geometry status {raw_path}:{line_no}")
                            if status == "success":
                                if not all(isinstance(row.get(metric), (int, float))
                                           and math.isfinite(row[metric])
                                           and -1e-12 <= row[metric] <= 1 + 1e-12
                                           for metric in ("auroc", "ap")):
                                    raise ValueError(f"Invalid successful metrics {raw_path}:{line_no}")
                                if not all(isinstance(row.get(metric), (int, float))
                                           and math.isfinite(row[metric])
                                           and row[metric] >= 0
                                           for metric in ("fit_seconds", "score_seconds")):
                                    raise ValueError(f"Invalid successful timing {raw_path}:{line_no}")
                                # Preserve the recorded success status. Analysis
                                # derives an exclusion from the measured budget.
                                if row["fit_seconds"] + row["score_seconds"] > 15.0:
                                    overbudget_success_records += 1
                                    overbudget_success_by_panel[
                                        f"{representation}/retain{round(fraction * 100)}"
                                    ] += 1
                            elif row.get("auroc") is not None or row.get("ap") is not None:
                                # A timeout can interrupt between AUROC and AP;
                                # its intermediate AUROC is not a scored outcome.
                                partial_auc = row.get("auroc")
                                if (status != "timeout" or row.get("ap") is not None
                                        or not isinstance(partial_auc, (int, float))
                                        or not math.isfinite(partial_auc)
                                        or not -1e-12 <= partial_auc <= 1 + 1e-12):
                                    raise ValueError(f"Failure has scored metrics {raw_path}:{line_no}")
                            status_counts[status] += 1
                            status_by_panel[f"{representation}/retain{round(fraction * 100)}"][status] += 1
                    if cell_methods != expected_methods:
                        raise ValueError(f"Incomplete method roster in {raw_path}: {len(cell_methods)}/56")
    if total != 38_080:
        raise ValueError(f"Expected 38,080 production rows, found {total}")
    summary = {
        "production_cells": len(raw_paths),
        "production_records": total,
        "fit_outliers": 0,
        "status_counts": dict(sorted(status_counts.items())),
        "status_by_panel": {key: dict(sorted(value.items()))
                            for key, value in sorted(status_by_panel.items())},
        "overbudget_success_records": overbudget_success_records,
        "overbudget_success_by_panel": dict(sorted(overbudget_success_by_panel.items())),
    }
    return raw_paths, log_paths, summary


def verify_watchdog_interventions(run_root: Path, manifest: dict) -> tuple[list[Path], dict]:
    """Tie every external intervention snapshot to its final raw timeout row."""
    root = run_root / "watchdog" / "interventions"
    if not root.exists():
        return [], {"completed_events": 0, "intervention_ids": []}
    if not root.is_dir():
        raise ValueError(f"Watchdog evidence path is not a directory: {root}")
    evidence_paths: list[Path] = []
    identifiers: list[str] = []
    seen_cells: set[str] = set()
    dataset_names = {entry["name"] for entry in manifest["datasets"]}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            raise ValueError(f"Unexpected watchdog evidence file: {directory}")
        expected_names = {"intervention.json", "raw.before.jsonl", "log.before.log"}
        actual_names = {path.name for path in directory.iterdir()}
        if actual_names != expected_names:
            raise ValueError(f"Incomplete or extra watchdog evidence in {directory}")
        audit_path = directory / "intervention.json"
        event = json.loads(audit_path.read_text(encoding="utf-8"))
        identifier = event.get("intervention_id")
        if (event.get("schema_version") != 1 or event.get("state") != "completed"
                or identifier != directory.name or identifier in identifiers):
            raise ValueError(f"Invalid or unfinished watchdog intervention: {directory}")
        cell = event.get("cell", {})
        dataset, representation = cell.get("dataset"), cell.get("representation")
        seed, fraction = cell.get("seed"), cell.get("retention_fraction")
        if (dataset not in dataset_names or representation not in manifest["representations"]
                or seed not in manifest["seeds"]
                or fraction not in manifest["fractions_retained"]
                or cell.get("method") != "LOCI"):
            raise ValueError(f"Watchdog event is outside the declared grid: {directory}")
        cell_name = record_name(dataset, representation, seed, fraction)
        if cell.get("name") != cell_name or cell_name in seen_cells:
            raise ValueError(f"Duplicate or mismatched watchdog cell: {directory}")
        seen_cells.add(cell_name)
        before = event.get("before", {})
        raw_before = (directory / "raw.before.jsonl").read_bytes()
        log_before = (directory / "log.before.log").read_bytes()
        if (before.get("raw_sha256") != hashlib.sha256(raw_before).hexdigest()
                or before.get("log_sha256") != hashlib.sha256(log_before).hexdigest()
                or before.get("raw_snapshot") != "raw.before.jsonl"
                or before.get("log_snapshot") != "log.before.log"
                or not isinstance(before.get("raw_age_seconds"), (int, float))
                or before["raw_age_seconds"] < 600):
            raise ValueError(f"Watchdog snapshot or age check failed: {directory}")
        prior_rows = [json.loads(line) for line in raw_before.splitlines()]
        if (not raw_before.endswith(b"\n")
                or before.get("raw_record_count") != len(prior_rows)
                or [row.get("method") for row in prior_rows] != manifest["methods"][:len(prior_rows)]
                or len(prior_rows) != manifest["methods"].index("LOCI")
                or prior_rows[-1]["method"] != "CBLOF"):
            raise ValueError(f"Watchdog pre-intervention roster prefix differs: {directory}")
        action = event.get("action", {})
        if (action.get("signal") != "SIGTERM"
                or action.get("termination_confirmed") is not True
                or action.get("primary_method_timeout_seconds") != 15.0):
            raise ValueError(f"Watchdog termination was not confirmed: {directory}")
        if "FitTimeout: per-method wall-time budget exceeded" not in before.get("signal_evidence_excerpt", ""):
            raise ValueError(f"Watchdog audit lacks swallowed signal evidence: {directory}")
        raw_final = (run_root / "raw" / f"{cell_name}.jsonl").read_bytes()
        if not raw_final.startswith(raw_before):
            raise ValueError(f"Final raw does not preserve watchdog snapshot: {directory}")
        suffix_lines = raw_final[len(raw_before):].splitlines(keepends=True)
        if not suffix_lines or not suffix_lines[0].endswith(b"\n"):
            raise ValueError(f"Watchdog row is absent from final raw: {directory}")
        row_bytes = suffix_lines[0]
        row = json.loads(row_bytes)
        if (row.get("method") != "LOCI" or row.get("status") != "timeout"
                or row.get("auroc") is not None or row.get("ap") is not None
                or row.get("fit_seconds") is not None
                or row.get("score_seconds") is not None
                or not str(row.get("error", "")).startswith("ExternalWatchdogTimeout:")
                or row.get("external_watchdog", {}).get("intervention_id") != identifier
                or event.get("appended_row_sha256") != hashlib.sha256(row_bytes).hexdigest()
                or event.get("after", {}).get("raw_sha256")
                   != hashlib.sha256(raw_before + row_bytes).hexdigest()):
            raise ValueError(f"Watchdog row/provenance mismatch: {directory}")
        identifiers.append(identifier)
        evidence_paths.extend(directory / name for name in sorted(expected_names))
    return evidence_paths, {
        "completed_events": len(identifiers), "intervention_ids": identifiers,
    }


def verify_process_restart(run_root: Path) -> tuple[list[Path], dict]:
    """Recompute both saved restart sidecars from the final raw/log files."""
    provenance_dir = run_root / "provenance"
    sidecars = sorted(provenance_dir.glob("*_restart.json"))
    script_path = run_root / "process_restart_audit.py"
    spec = importlib.util.spec_from_file_location("process_restart_audit", script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load process restart validator: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_names = sorted(f"{cell}_restart.json" for cell in module.RESTART_CELLS)
    if [path.name for path in sidecars] != expected_names:
        raise ValueError("Expected exactly the glass and pendigits PCA2/seed43/25% restart sidecars")
    live = module.verify_all(run_root)
    events = []
    for path in sidecars:
        saved = json.loads(path.read_text(encoding="utf-8"))
        cell = saved["cell"]
        if cell not in live or saved != live[cell]:
            raise ValueError(f"Saved restart sidecar differs from final raw/log evidence: {path}")
        events.append({
            "cell": cell,
            "sidecar_sha256": sha256(path),
            "first_completed_record_count": saved["first_invocation"]["completed_record_count"],
            "final_record_count": saved["final"]["record_count"],
        })
    return sidecars, {
        "event_count": len(events),
        "events": events,
        "raw_and_log_hashes_verified": True,
    }


def verify_inputs(manifest: dict, data_manifest_path: Path,
                  data_dir: Path) -> tuple[list[Path], Path, str]:
    for name in CORE_SCRIPTS:
        path = HERE / name
        if sha256(path) != manifest["code_sha256"][name]:
            raise ValueError(f"Benchmark script changed since run: {path}")
    if sha256(data_manifest_path) != manifest["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed since run")
    dataset_manifest = json.loads(data_manifest_path.read_text())
    entries = {entry["name"]: entry for entry in dataset_manifest["datasets"]}
    data_paths = []
    for item in manifest["datasets"]:
        entry = entries[item["name"]]
        path = data_dir / entry["filename"]
        if sha256(path) != item["sha256"] or item["sha256"] != entry["sha256"]:
            raise ValueError(f"Dataset SHA256 mismatch: {path}")
        data_paths.append(path)
    repo = Path(manifest["source_repo"])
    commit = manifest["source_git_commit"]
    tree = git_output(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if tree != manifest["source_git_tree"]:
        raise ValueError("Committed source tree differs from benchmark manifest")
    source_nsa = git_output(repo, "show", f"{commit}:pyod/models/nsa.py")
    if hashlib.sha256(source_nsa).hexdigest() != manifest["source_nsa_sha256"]:
        raise ValueError("NSA source at pinned commit differs from benchmark manifest")
    if manifest.get("source_git_status"):
        raise ValueError("Run used a dirty source checkout; committed archive cannot capture its full state")
    return data_paths, repo, commit


def readme(manifest: dict, summary: dict, analysis_names: list[str],
           retry_included: bool, watchdog_summary: dict,
           restart_summary: dict) -> str:
    text = f"""# Fresh PyOD NSA normal-only retention benchmark evidence

This archive contains the complete production grid: {summary['production_cells']} cells,
{summary['production_records']} method records, and zero anomalous training rows.
It evaluates 17 ODDS datasets, seeds 42–46, original features and PCA(2),
retained normal-training fractions 100%, 75%, 50%, and 25%, and 56 declared
configurations (13 NSA configurations from 11 strategy families and 43 other
PyOD tabular methods). Held-out mixed test data were used for scoring only.

Archive layout:
- `report/`: final Markdown report.
- `analysis/`: final machine-readable tables and figures: {', '.join(analysis_names)}.
- `production/raw/`: exactly 680 per-cell JSONL files. Every row records split
  hashes, ordered fit-row hash, status, AUROC/AP when successful, and fit count.
- `production/logs/`: matching invocation logs with the nominal 15 s
  per-method Python signal timer.
- `protocol/`: run and dataset manifests and package audit.
- `protocol/requirements_pinned.txt`: dependency versions recorded by the run;
  install the PyOD source snapshot itself separately.
- `code/`: benchmark runner, roster, loader, sweep, and available analysis
  scripts. The source snapshot below is the tested PyOD checkout.
- `datasets/`: all 17 byte-verified ODDS `.mat` files.
- `source_snapshot/pyod_source_commit.zip`: Git archive of source commit
  `{manifest['source_git_commit']}`, tree `{manifest['source_git_tree']}`.
"""
    if watchdog_summary["completed_events"]:
        text += (f"- `watchdog/interventions/`: {watchdog_summary['completed_events']} "
                 "externally terminated LOCI cell(s), each with pre-intervention "
                 "raw/log snapshots and an audited no-score timeout row.\n")
    text += (f"- `provenance/process_restarts/`: {restart_summary['event_count']} "
             "audited runner process restart sidecars, including raw-prefix and "
             "invocation-log hashes.\n")
    if retry_included:
        text += "- `supplemental_timeout_retry/`: optional follow-up runs; these are separate from the 38,080-row primary grid.\n"
    text += f"""
The run manifest pins Python, package versions, model parameters, source hashes,
dataset hashes, split design, and method budget. A nominal 15 s Python
`SIGALRM` timer begins immediately before `build_model` and is intended to
cover model-class import, construction, fitting, and scoring. Native code can
delay or swallow that signal, so a raw successful status alone does not prove
completion within 15 s. Optional `torch` import and seed setup happen before
the timer. On a successful run, `fit_seconds` starts before the optional
`torch` work and also includes model import and construction; it is not a
fit-only timing measure. The analysis treats a raw `status=success` row whose
measured `fit_seconds + score_seconds` exceeds 15 s as a derived
`overbudget_success`: {summary['overbudget_success_records']} such rows are
retained with their original status and scores in `production/raw/`, but
excluded from the reported 15 s AUROC/AP summaries and paired comparisons.
Those summaries require measured `fit_seconds + score_seconds <= 15 s`.
The pre-timer setup included in `fit_seconds` can contribute to this measured
overrun. `protocol/package_audit.json` verifies that the analysis audit and
raw grid count the same rows. Methods that
require labeled outliers or cannot handle a representation remain in the raw
grid with explicit status; no score is imputed for them.

An external watchdog was used on {watchdog_summary['completed_events']} LOCI
cell(s) that remained without a new raw row for at least 600 s after the
15 s signal was swallowed in a native callback. Each intervention preserved
the partial raw/log files before terminating only that runner process and
added an unscored LOCI `timeout` row with an intervention ID. The regular
sweep could then resume later methods; the analysis audit and package audit
identify these interventions separately from ordinary timer outcomes.

Two PCA(2)/seed 43/25% cells have two identical logged runner commands each.
The first glass invocation committed 36 method records; the first pendigits
invocation committed 30. The second command completed each cell to 56 records.
Both restart sidecars verify the preserved prefixes and final raw/log hashes.
The saved logs do not establish the first process exit codes or which methods
caused them to end. No method outcome was imputed from either restart.

The analysis CSV marks LMDD, DIF, CD, and ROD as batch-dependent in addition
to the methods marked that way in the original roster metadata. The original
`method_meta.inference_mode` field in the run manifest did not include those
four; use the audited analysis flag when interpreting scoring semantics.

For an independent reproduction, use Python 3.12, extract
`source_snapshot/pyod_source_commit.zip`, and install the pinned dependencies
and that PyOD source checkout. For example, from the extracted bundle:

    unzip source_snapshot/pyod_source_commit.zip -d pyod_source
    python -m pip install -r protocol/requirements_pinned.txt
    python -m pip install -e pyod_source

    python code/full_retention_sweep.py --data-dir datasets --run-root reproduced --workers 8

The source code's `dataset_manifest.json` contains original PyOD links and
SHA-256 values. The packaged `.mat` files remove the need to fetch them.
`protocol/package_audit.json` lists a SHA-256 digest for every included source
file and reports the validated production status counts.
"""
    return text


def package(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    report_path = args.report.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Completed run manifest and final report are required")
    if not report_path.read_text(encoding="utf-8").strip():
        raise ValueError("Final report is empty")
    manifest = json.loads(manifest_path.read_text())
    raw_paths, log_paths, summary = check_complete_grid(run_root, manifest)
    watchdog_paths, watchdog_summary = verify_watchdog_interventions(run_root, manifest)
    restart_paths, restart_summary = verify_process_restart(run_root)
    data_manifest_path = HERE / "dataset_manifest.json"
    data_dir = args.data_dir.resolve()
    data_paths, source_repo, source_commit = verify_inputs(
        manifest, data_manifest_path, data_dir
    )
    analysis_paths = sorted(path for path in analysis_dir.rglob("*")
                            if path.is_file() and path.suffix.lower()
                            in {".csv", ".json", ".png", ".svg", ".pdf"})
    if not any(path.suffix.lower() == ".csv" for path in analysis_paths):
        raise ValueError(f"No final CSV analysis tables found in {analysis_dir}")
    analysis_audit_path = analysis_dir / ANALYSIS_AUDIT_NAME
    if not analysis_audit_path.is_file():
        raise ValueError(f"Analysis audit is required: {analysis_audit_path}")
    analysis_audit = json.loads(analysis_audit_path.read_text(encoding="utf-8"))
    analysis_overbudget = analysis_audit.get("overbudget_success_records")
    if (type(analysis_overbudget) is not int
            or analysis_overbudget != summary["overbudget_success_records"]):
        raise ValueError(
            "Analysis audit overbudget_success_records does not match the "
            f"raw measured >15 s success count: analysis={analysis_overbudget}, "
            f"raw={summary['overbudget_success_records']}"
        )
    for key in ("external_watchdog_intervention_count",
                "external_watchdog_loci_timeout_rows"):
        value = analysis_audit.get(key)
        if type(value) is not int or value != watchdog_summary["completed_events"]:
            raise ValueError(
                f"Analysis audit {key}={value} differs from the "
                f"validated watchdog count {watchdog_summary['completed_events']}"
            )
    if analysis_audit.get("process_restart_event_count") != restart_summary["event_count"]:
        raise ValueError("Analysis audit process restart count differs from verified sidecar")
    analysis_restart_path = analysis_dir / "process_restart_audit.json"
    if (not analysis_restart_path.is_file()
            or json.loads(analysis_restart_path.read_text(encoding="utf-8")) != restart_summary):
        raise ValueError("Analysis process restart audit differs from verified sidecar")
    analysis_names = [str(path.relative_to(analysis_dir)) for path in analysis_paths]
    analysis_scripts = sorted(path for path in run_root.glob("*.py") if path.is_file())
    if not analysis_scripts:
        raise ValueError("Analysis source script is missing from run root")
    retry_paths: list[Path] = []
    if args.retry_root is not None:
        retry_root = args.retry_root.resolve()
        if not retry_root.is_dir():
            raise FileNotFoundError(retry_root)
        retry_manifest_path = retry_root / "run_manifest.json"
        retry_analysis_dir = retry_root / "analysis"
        if not retry_manifest_path.is_file() or not (retry_analysis_dir / "validation.json").is_file():
            raise ValueError("Retry root needs its production manifest and validated analysis")
        retry_manifest = json.loads(retry_manifest_path.read_text())
        if retry_manifest.get("primary_manifest_sha256") != sha256(manifest_path) \
                or retry_manifest.get("retry_method_timeout_seconds") != 60:
            raise ValueError("Retry manifest differs from the primary run or 60 s protocol")
        expected_retry_names = {f"{cell['name']}.jsonl"
                                for cell in retry_manifest["primary_timeout_cells"]}
        retry_raw = sorted((retry_root / "raw").glob("*.jsonl"))
        if {path.name for path in retry_raw} != expected_retry_names:
            raise ValueError("Production retry raw files differ from the pinned timeout cells")
        retry_sidecars = sorted((retry_root / "raw").glob("*.retry.json"))
        expected_sidecars = {name.removesuffix(".jsonl") + ".retry.json"
                             for name in expected_retry_names}
        if {path.name for path in retry_sidecars} != expected_sidecars:
            raise ValueError("Production retry provenance sidecars are incomplete")
        retry_logs = sorted((retry_root / "logs").glob("*.log"))
        expected_logs = {name.removesuffix(".jsonl") + ".log"
                         for name in expected_retry_names}
        if {path.name for path in retry_logs} != expected_logs:
            raise ValueError("Production retry invocation logs are incomplete")
        retry_analysis = sorted(path for path in retry_analysis_dir.rglob("*")
                                if path.is_file() and path.suffix.lower()
                                in {".csv", ".json", ".md", ".png", ".svg"})
        retry_paths = [retry_manifest_path, *retry_raw, *retry_sidecars,
                       *retry_logs, *retry_analysis]
        if not retry_raw:
            raise ValueError("Retry root was specified but has no production retries")
    if args.dry_run:
        print(json.dumps({
            "validated": summary,
            "report": str(report_path),
            "analysis_files": analysis_names,
            "dataset_files": len(data_paths),
            "source_commit": source_commit,
            "retry_files": len(retry_paths),
            "watchdog_events": watchdog_summary["completed_events"],
            "process_restart_events": restart_summary["event_count"],
            "would_write_to": str(output_dir),
        }, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_report = output_dir / f"{PREFIX}_report.md"
    output_zip = output_dir / f"{PREFIX}_reproducibility.zip"
    output_audit = output_dir / f"{PREFIX}_package_audit.json"
    output_manifest = output_dir / f"{PREFIX}_run_manifest.json"
    output_data_manifest = output_dir / f"{PREFIX}_dataset_manifest.json"
    exposed = [output_report, output_zip, output_audit, output_manifest,
               output_data_manifest]
    exposed.extend(output_dir / exposed_analysis_name(path) for path in analysis_paths)
    if len(set(exposed)) != len(exposed):
        raise ValueError("Analysis file basenames collide in outputs")
    if not args.replace and any(path.exists() and path.resolve() != report_path
                                for path in exposed):
        raise FileExistsError("A destination already exists; pass --replace only after review")

    with tempfile.TemporaryDirectory(prefix="full_retention_package_", dir=run_root) as tmp:
        staging = Path(tmp)
        source_zip = staging / "pyod_source_commit.zip"
        with source_zip.open("wb") as stream:
            subprocess.run(["git", "-C", str(source_repo), "archive", "--format=zip",
                            source_commit], stdout=stream, check=True)
        audit = {
            "schema_version": 1,
            "source_commit": source_commit,
            "source_tree": manifest["source_git_tree"],
            "production_validation": summary,
            "watchdog_validation": watchdog_summary,
            "process_restart_validation": restart_summary,
            "analysis_validation": {
                "audit_file": f"analysis/{ANALYSIS_AUDIT_NAME}",
                "overbudget_success_records": analysis_overbudget,
                "matches_raw_overbudget_success_records": True,
                "external_watchdog_intervention_count": watchdog_summary["completed_events"],
                "matches_watchdog_evidence": True,
                "process_restart_event_count": restart_summary["event_count"],
                "matches_process_restart_evidence": True,
            },
            "supplemental_timeout_retry_included": bool(retry_paths),
            "members_sha256": {},
        }
        members: list[tuple[Path, str]] = []
        pinned_requirements = staging / "requirements_pinned.txt"
        pinned_requirements.write_text(
            "".join(f"{name}=={version}\n" for name, version
                    in sorted(manifest["packages"].items())
                    if name != "pyod" and version is not None),
            encoding="utf-8",
        )
        members += [(report_path, "report/final_report.md")]
        members += [(manifest_path, "protocol/run_manifest.json"),
                    (data_manifest_path, "protocol/dataset_manifest.json"),
                    (pinned_requirements, "protocol/requirements_pinned.txt")]
        members += [(path, f"production/raw/{path.name}") for path in raw_paths]
        members += [(path, f"production/logs/{path.name}") for path in log_paths]
        members += [(path, f"watchdog/interventions/{path.parent.name}/{path.name}")
                    for path in watchdog_paths]
        members += [(path, f"provenance/process_restarts/{path.name}")
                    for path in restart_paths]
        members += [(path, f"datasets/{path.name}") for path in data_paths]
        members += [(HERE / name, f"code/{name}") for name in CORE_SCRIPTS]
        members += [(HERE / "package_full_retention.py", "code/package_full_retention.py")]
        members += [(data_manifest_path, "code/dataset_manifest.json")]
        members += [(path, f"code/full_retention/{path.name}") for path in analysis_scripts]
        members += [(path, f"analysis/{path.relative_to(analysis_dir).as_posix()}")
                    for path in analysis_paths]
        members += [(source_zip, "source_snapshot/pyod_source_commit.zip")]
        if retry_paths:
            members += [(HERE / name, f"code/optional_retry/{name}") for name in (
                "timeout_retry_runner.py", "full_retention_timeout_retry.py",
                "full_retention_timeout_retry_analysis.py",
            )]
            members += [(path, f"supplemental_timeout_retry/{path.relative_to(retry_root).as_posix()}")
                        for path in retry_paths]
        archive_names = [name for _, name in members]
        if len(archive_names) != len(set(archive_names)):
            raise ValueError("Duplicate archive member path")
        for path, name in members:
            audit["members_sha256"][name] = sha256(path)
        readme_path = staging / "README.md"
        readme_path.write_text(readme(manifest, summary, analysis_names,
                                      bool(retry_paths), watchdog_summary,
                                      restart_summary),
                               encoding="utf-8")
        audit["members_sha256"]["README.md"] = sha256(readme_path)
        audit_path = staging / "package_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        temp_zip = staging / "bundle.zip"
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            archive.write(readme_path, "README.md")
            archive.write(audit_path, "protocol/package_audit.json")
            for path, name in members:
                archive.write(path, name)
        with zipfile.ZipFile(temp_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive integrity test failed")
            expected_member_count = len(members) + 2
            if len(archive.namelist()) != expected_member_count:
                raise ValueError("Archive member count mismatch")
            for name, expected_sha in audit["members_sha256"].items():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_sha:
                    raise ValueError(f"Archived bytes differ from audited source: {name}")
        audit["archive_sha256"] = sha256(temp_zip)
        audit["archive_bytes"] = temp_zip.stat().st_size
        temp_audit = staging / "package_audit_sidecar.json"
        temp_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        # Expose only a fully built, integrity-checked ZIP. The prerequisite
        # grid/report/source checks have all passed before sidecars are copied.
        if report_path != output_report:
            shutil.copy2(report_path, output_report)
        shutil.copy2(manifest_path, output_manifest)
        shutil.copy2(data_manifest_path, output_data_manifest)
        for path in analysis_paths:
            shutil.copy2(path, output_dir / exposed_analysis_name(path))
        shutil.copy2(temp_audit, output_audit)
        os.replace(temp_zip, output_zip)
    print(json.dumps({
        "report": str(output_report),
        "archive": str(output_zip),
        "archive_sha256": audit["archive_sha256"],
        "audit": str(output_audit),
        "production_records": summary["production_records"],
        "analysis_files": len(analysis_paths),
        "retry_files": len(retry_paths),
        "watchdog_events": watchdog_summary["completed_events"],
        "process_restart_events": restart_summary["event_count"],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--report", type=Path, required=True,
                        help="Reviewed final Markdown report to package and copy to outputs")
    parser.add_argument("--analysis-dir", type=Path,
                        default=RUN_ROOT / "analysis")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(json.loads((HERE / "dataset_manifest.json").read_text())
                                     ["source"]["local_directory"]))
    parser.add_argument("--retry-root", type=Path,
                        help="Optional supplemental timeout-retry files, kept separate from the primary grid")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate complete grid and inputs without writing deliverables")
    parser.add_argument("--replace", action="store_true",
                        help="Replace previously generated files with this prefix")
    args = parser.parse_args()
    package(args)


if __name__ == "__main__":
    main()
