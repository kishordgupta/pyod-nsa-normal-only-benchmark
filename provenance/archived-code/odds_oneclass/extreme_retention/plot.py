"""Render static, coverage-aware NSA/dual six-size benchmark figures.

Each panel uses only 85/85 measured-time-qualified outcomes for an absolute
score or rank. A missing point is annotated with its qualified coverage;
Matplotlib does not connect a line through a missing point. The baseline
envelope selects the highest scoring member of the fixed six-size baseline
cohort separately at each size, so its method identity can change.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import FRACTIONS, REPS

HERE = Path(__file__).resolve().parent
_report_spec = importlib.util.spec_from_file_location(
    "_extreme_retention_report_for_plot", HERE / "report.py")
if _report_spec is None or _report_spec.loader is None:
    raise RuntimeError("cannot load the six-size report validator")
_report_module = importlib.util.module_from_spec(_report_spec)
_report_spec.loader.exec_module(_report_module)
load = _report_module.load


def plot_representation(data: dict, rep: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True,
                             constrained_layout=True)
    x = np.arange(len(FRACTIONS))
    for col, metric in enumerate(("auroc", "ap")):
        dual_scores, baseline_scores, rank_fractions, rank_labels, coverage = [], [], [], [], []
        for pct in FRACTIONS:
            row = data["summary_index"][(rep, pct, "NSA/dual")]
            full = int(row["qualified_success_seed_rows"]) == 85
            coverage.append(int(row["qualified_success_seed_rows"]))
            dual_scores.append(float(row[f"macro_{metric}_qualified_available"]) if full else np.nan)
            ranked = data["common_index"].get(
                (rep, pct, metric, "common_baselines_plus_focal_nsa", "NSA/dual"))
            rank_fractions.append(float(ranked["rank"]) / int(ranked["field_size"])
                                  if ranked and full else np.nan)
            rank_labels.append(f"{ranked['rank']}/{ranked['field_size']}"
                               if ranked and full else "")
            candidates = [r for r in data["common_field_ranks"]
                          if r["representation"] == rep and int(r["retention_pct"]) == pct
                          and r["metric"] == metric and r["field"] == "common_baselines"]
            leader = max(candidates, key=lambda r: float(r["score"])) if candidates else None
            baseline_scores.append(float(leader["score"]) if leader else np.nan)

        score_ax = axes[0, col]
        score_ax.plot(x, dual_scores, "o-", color="#1f77b4", linewidth=2,
                      label="NSA/dual (85/85 strict successes)")
        score_ax.plot(x, baseline_scores, "s--", color="#d55e00", linewidth=1.8,
                      label="Top member of fixed baseline cohort")
        score_ax.set_ylim(0, 1.03)
        score_ax.set_ylabel(metric.upper())
        score_ax.set_title(f"{rep}: {metric.upper()} across retained normal pools")
        score_ax.grid(axis="y", alpha=0.25)
        for xpos, q in zip(x, coverage):
            if q < 85:
                score_ax.text(xpos, 0.03, f"{q}/85", ha="center", va="bottom",
                              color="#555555", fontsize=8)
        if col == 0:
            score_ax.legend(loc="center left", fontsize=8)

        rank_ax = axes[1, col]
        rank_ax.plot(x, rank_fractions, "o-", color="#5e3c99", linewidth=2)
        rank_ax.set_ylim(0, 1.04)
        rank_ax.set_ylabel("NSA/dual rank / field size (lower is better)")
        rank_ax.set_title("Fixed six-size baseline cohort + focal NSA/dual")
        rank_ax.grid(axis="y", alpha=0.25)
        for xpos, value, label, q in zip(x, rank_fractions, rank_labels, coverage):
            if np.isfinite(value):
                rank_ax.annotate(label, (xpos, value), textcoords="offset points",
                                 xytext=(0, 8), ha="center", fontsize=8)
            else:
                rank_ax.text(xpos, 0.03, f"{q}/85", ha="center", va="bottom",
                             color="#555555", fontsize=8)
        rank_ax.set_xticks(x, [f"{pct}%" for pct in FRACTIONS])
        rank_ax.set_xlabel("Retained normal training pool")

    fig.suptitle("NSA/dual versus fixed-cohort PyOD baseline leaders", fontsize=15)
    fig.text(0.5, -0.035,
             "Only full 85/85 measured ≤15 s successes are plotted. Baseline leader can change "
             "with size; missing NSA points are unconnected and show qualified coverage.",
             ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=HERE / "analysis")
    parser.add_argument("--out-dir", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    data = load(args.analysis_dir)
    for rep in REPS:
        path = args.out_dir / f"extreme_retention_dual_{rep}.png"
        plot_representation(data, rep, path)
        print(path)


if __name__ == "__main__":
    main()
