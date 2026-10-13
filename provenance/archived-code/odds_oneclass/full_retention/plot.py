"""Static paired-AUROC-change heatmap for the audited full-roster screen.

The NSA strategies and five fixed controls are plotted in predeclared order.
No outcome-selected ranking or test-label model selection enters the figure.
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

NSA = (
    "NSA/fixed", "NSA/variable", "NSA/coverage", "NSA/grid",
    "NSA/hierarchical", "NSA/voronoi", "NSA/deterministic",
    "NSA/suppressed", "NSA/dual", "NSA/annealed",
    "NSA/binary/hamming", "NSA/binary/rcontiguous", "NSA/binary/rchunk",
)
CONTROLS = ("IForest", "KNN", "OCSVM", "ABOD", "ECOD")
METHODS = NSA + CONTROLS
LABELS = (
    "Fixed NSA", "Variable NSA", "Coverage NSA", "Grid NSA",
    "Hierarchical NSA", "Voronoi NSA", "Deterministic NSA",
    "Suppressed NSA", "Dual NSA", "Annealed NSA",
    "Binary · Hamming", "Binary · contiguous", "Binary · chunk",
    "IForest", "KNN", "OCSVM", "ABOD", "ECOD",
)
REPS = ("native", "pca2")
SIZES = (75, 50, 25)


def load(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"representation", "retention_pct", "method", "expected_pairs",
                    "paired_seed_rows", "full_coverage", "macro_mean_delta_auroc"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("missing paired-change columns")
        result = {}
        for row in reader:
            if row["method"] not in METHODS:
                continue
            key = (row["representation"], int(row["retention_pct"]), row["method"])
            if key in result:
                raise ValueError(f"duplicate plot cell: {key}")
            n, expected = int(row["paired_seed_rows"]), int(row["expected_pairs"])
            full = row["full_coverage"].strip().lower() == "true"
            value = (float(row["macro_mean_delta_auroc"])
                     if row["macro_mean_delta_auroc"].strip() else None)
            if expected != 85 or not 0 <= n <= expected or full != (n == expected):
                raise ValueError(f"invalid plot coverage: {key}")
            if (value is None) != (n == 0) or (value is not None and
                                              (not np.isfinite(value) or abs(value) > 1)):
                raise ValueError(f"invalid plot value: {key}")
            result[key] = {"value": value, "n": n, "expected": expected, "full": full}
    expected_keys = {(rep, pct, method) for rep in REPS for pct in SIZES for method in METHODS}
    if set(result) != expected_keys:
        raise ValueError(f"incomplete plot grid: missing={sorted(expected_keys-set(result))[:3]}")
    return result


def draw(cells: dict, output: Path, dpi: int) -> None:
    finite = [r["value"] for r in cells.values() if r["value"] is not None]
    if not finite:
        raise ValueError("no successful pairwise AUROC changes")
    scale = max(0.01, max(abs(x) for x in finite))
    cmap = LinearSegmentedColormap.from_list("retention", ("#a9382d", "#fbfaf7", "#147f73"))
    cmap.set_bad("#d9dce0")
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 9.2), dpi=dpi,
                             sharey=True, gridspec_kw={"wspace": 0.10})
    image = None
    for ax, rep, title in zip(axes, REPS, ("Original features", "PCA to two dimensions")):
        matrix = np.full((len(METHODS), len(SIZES)), np.nan)
        for i, method in enumerate(METHODS):
            for j, pct in enumerate(SIZES):
                value = cells[(rep, pct, method)]["value"]
                if value is not None:
                    matrix[i, j] = value
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm,
                          interpolation="none", aspect="auto")
        ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)
        ax.set_xticks(range(len(SIZES)), [f"{p}% retained\n{100-p}% removed" for p in SIZES],
                      fontsize=10)
        ax.set_yticks(range(len(METHODS)), LABELS, fontsize=9.5)
        ax.set_xticks(np.arange(-0.5, len(SIZES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="both", length=0, pad=5)
        ax.axhline(len(NSA)-0.5, color="#565f65", linewidth=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, method in enumerate(METHODS):
            for j, pct in enumerate(SIZES):
                cell = cells[(rep, pct, method)]
                value, n = cell["value"], cell["n"]
                if value is None:
                    label, color = "No pairs\n0/85", "#394147"
                elif cell["full"]:
                    label = "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"
                    color = "white" if abs(value) > 0.6 * scale else "#1f2c2d"
                else:
                    number = "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"
                    label = f"{number}\n{n}/85"
                    color = "white" if abs(value) > 0.6 * scale else "#1f2c2d"
                    ax.add_patch(Polygon(
                        ((j+0.19, i-0.48), (j+0.48, i-0.48), (j+0.48, i-0.19)),
                        closed=True, facecolor="#d89024", edgecolor="none", zorder=3))
                ax.text(j, i, label, ha="center", va="center", color=color,
                        fontsize=8.5 if "\n" in label else 9.2, zorder=4,
                        fontweight="medium", linespacing=1.04)
    axes[1].tick_params(labelleft=False)
    fig.suptitle("Paired AUROC change as normal-only training data shrinks",
                 fontsize=16, fontweight="semibold", y=0.988)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.925, bottom=0.17)
    bar = fig.colorbar(image, ax=axes, orientation="horizontal",
                       fraction=0.03, pad=0.065, aspect=48)
    bar.set_label("Mean change from fresh 100% training  ·  positive favors the smaller training set",
                  fontsize=10.5)
    bar.ax.tick_params(labelsize=9)
    fig.text(0.5, 0.056,
             "Means weight datasets equally after matching each method on dataset, seed and held-out test rows. "
             "Five split seeds; 17 datasets.",
             ha="center", fontsize=9.0, color="#42484b")
    fig.text(0.5, 0.031,
             "Amber corner and n/85: partial paired coverage. Gray: no successful pair. "
             "NSA and controls have fixed display order.",
             ha="center", fontsize=9.0, color="#42484b")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changes", type=Path, default=Path(__file__).parent / "analysis" /
                        "full_retention_change_vs_full.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "analysis" /
                        "full_retention_auroc_change.png")
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be >= 72")
    cells = load(args.changes)
    draw(cells, args.output, args.dpi)
    print(f"Wrote {len(cells)} validated NSA/control cells to {args.output}")


if __name__ == "__main__":
    main()
