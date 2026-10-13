"""Plot paired AUROC changes from the normal-training retention experiment.

Each cell is the equal-dataset-weight mean of paired AUROC differences between
one reduced normal-only training set and the matched 100% training set. The
test set stays fixed within each dataset/seed/representation pair. The input
table is produced by retention_analysis.py; this script never recomputes a
score or selects a strategy using test labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np


METHODS = (
    "NSA/fixed",
    "NSA/variable",
    "NSA/coverage",
    "NSA/grid",
    "NSA/hierarchical",
    "NSA/voronoi",
    "NSA/deterministic",
    "NSA/suppressed",
    "NSA/dual",
    "NSA/annealed",
    "NSA/binary/hamming",
    "NSA/binary/rcontiguous",
    "NSA/binary/rchunk",
)
LABELS = (
    "Fixed", "Variable", "Coverage", "Grid", "Hierarchical",
    "Voronoi", "Deterministic", "Suppressed", "Dual", "Annealed",
    "Binary · Hamming", "Binary · contiguous", "Binary · chunk",
)
REPRESENTATIONS = ("native", "pca2")
FRACTIONS = (25, 50, 75)
REQUIRED_COLUMNS = {
    "representation", "retention_pct", "method", "expected_pairs",
    "paired_seed_rows", "full_coverage", "macro_mean_delta_auroc",
}


def _fraction_pct(value: str) -> int:
    number = float(value)
    if 0 < number <= 1:
        number *= 100
    rounded = round(number)
    if not np.isclose(number, rounded) or rounded not in FRACTIONS:
        raise ValueError(f"unexpected retained training fraction: {value!r}")
    return rounded


def _boolean(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid full_coverage value: {value!r}")
    return lowered in {"true", "1"}


def load_changes(path: Path) -> dict[tuple[str, str, int], dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("change table has no header")
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"change table is missing columns: {sorted(missing_columns)}")
        indexed: dict[tuple[str, str, int], dict] = {}
        for row in reader:
            method = row["method"]
            if method not in METHODS:
                continue
            rep = row["representation"]
            if rep not in REPRESENTATIONS:
                raise ValueError(f"unexpected representation for {method}: {rep!r}")
            pct = _fraction_pct(row["retention_pct"])
            key = (rep, method, pct)
            if key in indexed:
                raise ValueError(f"duplicate NSA change row: {key}")
            expected = int(row["expected_pairs"])
            paired = int(row["paired_seed_rows"])
            full = _boolean(row["full_coverage"])
            if expected <= 0 or not 0 <= paired <= expected or full != (paired == expected):
                raise ValueError(f"inconsistent paired coverage for {key}")
            raw_delta = row["macro_mean_delta_auroc"].strip()
            if paired == 0:
                if raw_delta:
                    raise ValueError(f"metric shown without successful pairs for {key}")
                delta = None
            else:
                if not raw_delta:
                    raise ValueError(f"missing metric for {paired} pairs at {key}")
                delta = float(raw_delta)
                if not np.isfinite(delta) or not -1 <= delta <= 1:
                    raise ValueError(f"invalid paired AUROC difference at {key}")
            indexed[key] = {"delta": delta, "paired": paired,
                            "expected": expected, "full": full}
    expected_keys = {(rep, method, pct) for rep in REPRESENTATIONS
                     for method in METHODS for pct in FRACTIONS}
    missing = expected_keys - set(indexed)
    if missing:
        raise ValueError(f"incomplete NSA change grid: {len(missing)} missing; "
                         f"examples={sorted(missing)[:4]}")
    return indexed


def draw(changes: dict[tuple[str, str, int], dict], output: Path, dpi: int) -> None:
    finite = [row["delta"] for row in changes.values() if row["delta"] is not None]
    if not finite:
        raise ValueError("no successful paired comparison to plot")
    scale = max(abs(value) for value in finite)
    # A symmetric scale keeps positive and negative differences comparable.
    scale = max(scale, 0.01)
    cmap = LinearSegmentedColormap.from_list(
        "retention_change", ("#aa3b33", "#f8f7f2", "#167a78")
    )
    cmap.set_bad("#d8d8d8")
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.0), dpi=dpi,
                             sharey=True, gridspec_kw={"wspace": 0.11})
    images = []
    for ax, rep, title in zip(
        axes, REPRESENTATIONS,
        ("Original features", "PCA to two dimensions"),
    ):
        values = np.full((len(METHODS), len(FRACTIONS)), np.nan)
        for i, method in enumerate(METHODS):
            for j, pct in enumerate(FRACTIONS):
                value = changes[(rep, method, pct)]["delta"]
                if value is not None:
                    values[i, j] = value
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                          aspect="auto", interpolation="none")
        images.append(image)
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=11)
        ax.set_xticks(range(len(FRACTIONS)),
                      [f"{pct}% retained\n({100-pct}% removed)" for pct in FRACTIONS],
                      fontsize=9)
        ax.set_yticks(range(len(METHODS)), LABELS, fontsize=9)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.set_xticks(np.arange(-0.5, len(FRACTIONS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", length=0)
        ax.axhline(9.5, color="#6c6c6c", linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, method in enumerate(METHODS):
            for j, pct in enumerate(FRACTIONS):
                cell = changes[(rep, method, pct)]
                value = cell["delta"]
                paired, expected = cell["paired"], cell["expected"]
                if value is None:
                    annotation = f"No pairs\n0/{expected}"
                    color = "#3c3c3c"
                elif cell["full"]:
                    annotation = f"{value:+.3f}"
                    color = "white" if abs(value) > 0.55 * scale else "#202020"
                else:
                    annotation = f"{value:+.3f}\n{paired}/{expected} pairs"
                    color = "white" if abs(value) > 0.55 * scale else "#202020"
                    # Amber corner also distinguishes a partial cell when the
                    # figure is viewed too small to read its denominator.
                    ax.add_patch(Polygon(
                        ((j + 0.22, i - 0.48), (j + 0.48, i - 0.48),
                         (j + 0.48, i - 0.22)),
                        closed=True, facecolor="#dd9a2a", edgecolor="none",
                        zorder=3,
                    ))
                ax.text(j, i, annotation, ha="center", va="center",
                        color=color, fontsize=8 if "\n" in annotation else 9,
                        fontweight="medium", linespacing=1.12, zorder=4)
    axes[1].tick_params(labelleft=False)
    fig.suptitle("Effect of reducing normal-only training data on NSA AUROC",
                 fontsize=15, fontweight="semibold", y=0.985)
    fig.subplots_adjust(left=0.21, right=0.985, top=0.90, bottom=0.18)
    colorbar = fig.colorbar(images[0], ax=axes, orientation="horizontal",
                            fraction=0.035, pad=0.07, aspect=45)
    colorbar.set_label("Mean paired AUROC change from 100% training  ·  "
                       "positive means reduced training improved AUROC",
                       fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.text(0.5, 0.048,
             "Each cell compares the same dataset, split, test set, and NSA configuration. "
             "Means weight datasets equally across matched seeds.",
             ha="center", va="center", fontsize=8.5, color="#444444")
    fig.text(0.5, 0.024,
             "Amber corner and n/N: partial paired coverage. Gray: no successful pair. "
             "Binary matching rules are shown separately.",
             ha="center", va="center", fontsize=8.5, color="#444444")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changes", type=Path, required=True,
                        help="retention_change_vs_full.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="output figure path, such as a PNG or PDF")
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    changes = load_changes(args.changes)
    draw(changes, args.output, args.dpi)
    print(f"Plotted {len(changes)} NSA retention cells to {args.output}")


if __name__ == "__main__":
    main()
