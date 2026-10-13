"""Three-split sensitivity panel: all NSA variants and fixed common baselines.

The method set is declared before inspecting multi-seed results. Primary full
roster results remain in separate seed-42 files.
"""

import runner


SELECTED_METHODS = [method for method in runner.METHODS
                    if method.startswith("NSA/")] + [
    "IForest", "ECOD", "OCSVM", "KNN", "ABOD"
]


if __name__ == "__main__":
    runner.METHODS = SELECTED_METHODS
    runner.main()
