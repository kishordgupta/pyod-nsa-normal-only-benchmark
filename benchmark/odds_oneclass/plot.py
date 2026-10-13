"""Publication-style paired AUROC comparison from the three-split panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["fixed", "variable", "coverage", "grid", "hierarchical",
         "voronoi", "deterministic", "suppressed", "dual", "annealed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.pairs)
    df = df[(df.baseline == "IForest") & (df.full_coverage == True)]
    colors = {"native": "#b44d3d", "pca2": "#31688e"}
    fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=170)
    y = np.arange(len(ORDER))
    for representation, offset in (("native", -0.17), ("pca2", 0.17)):
        sub = df[df.representation == representation].set_index("nsa")
        for i, strategy in enumerate(ORDER):
            name = "NSA/" + strategy
            if name not in sub.index:
                continue
            row = sub.loc[name]
            mean = row.macro_mean_delta_auroc
            lo = row.cluster_bootstrap_95_low_auroc
            hi = row.cluster_bootstrap_95_high_auroc
            ax.plot([lo, hi], [i + offset, i + offset], color=colors[representation],
                    linewidth=2.1, solid_capstyle="round")
            ax.plot(mean, i + offset, marker="o", markersize=5.2,
                    color=colors[representation])
    ax.axvline(0, color="#454545", linewidth=1.15, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired AUROC difference: NSA − IForest")
    ax.set_title("Normal-only training on 17 ODDS datasets, three splits")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], color=colors[rep], marker="o", linewidth=2,
                          label=label)
               for rep, label in (("native", "Original features"),
                                  ("pca2", "PCA to two dimensions"))]
    fig.legend(handles=handles, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.055), ncol=2)
    fig.text(0.5, 0.005,
             "Intervals resample datasets. Missing points indicate method inapplicability or incomplete success.",
             ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
