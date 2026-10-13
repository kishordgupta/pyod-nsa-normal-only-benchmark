# Third-party materials

The PyOD source under `vendor/pyod` is copied from the clean checkout of
`https://github.com/yzhao062/pyod` at commit
`0d6c901652f82c813da45875873ef4a8560cf049` (tree
`57ec5c9ea2800b559019adf0457d7e5b858fc153`). It includes the NSA implementation
from PyOD pull request [#747](https://github.com/yzhao062/pyod/pull/747).
The original BSD 2-Clause license, copyright notices and attribution remain
in the package and in `vendor/pyod/LICENSE`. The repository license is the
same BSD 2-Clause license; it does not supersede third-party notices.

The 17 ODDS `.mat` files are not redistributed here. `scripts/fetch_datasets.py`
retrieves them from their immutable PyOD source commit
`ef6d5e8dc828615f498286c1d6a22cbd02fd4cd2` and verifies the recorded SHA-256.
The individual source URLs, class counts and shapes are retained in the
dataset manifest. See the [ODDS catalog](https://shebuti.com/outlier-detection-datasets-odds/)
for dataset provenance and original data sources. The BSD license covering
code is not a claim about the licenses of the original datasets.

The experiments depend on NumPy, SciPy, scikit-learn, PyTorch, XGBoost and
the other libraries listed in `requirements.txt`; their own licenses apply.
Those libraries and virtual environments are not bundled in this repository.
Upstream dataset test fixtures are omitted; the NSA implementation tests,
which generate synthetic test observations, remain in the source snapshot.
