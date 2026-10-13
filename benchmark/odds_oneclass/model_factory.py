"""Predeclared PyOD model roster for the clean-normal ODDS experiment.

The roster is the 44 shipped tabular entries in this checkout's
``pyod/utils/knowledge/algorithms.json``: NSA is expanded into its 11
strategies, with all three binary matching rules shown separately, and the
other 43 registry entries appear once each. A model whose fit needs labeled
anomalies is listed but is *not* silently trained with test labels.

Every algorithm receives the same feature matrix from the runner. The only
label-free constructor adaptation is limiting a neighbor count to n_train-1.
PyOD's contamination=0.1 is retained because BaseDetector rejects zero; it
sets PyOD's score cutoff and does not introduce training anomalies. Ranking
metrics must use decision_function scores, not this cutoff.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


NSA_STRATEGIES = (
    "fixed", "variable", "coverage", "grid", "hierarchical", "voronoi",
    "deterministic", "suppressed", "dual", "annealed",
)
NSA_BINARY_RULES = ("hamming", "rcontiguous", "rchunk")
NSA_METHODS = [f"NSA/{strategy}" for strategy in NSA_STRATEGIES] + [
    f"NSA/binary/{rule}" for rule in NSA_BINARY_RULES
]

# Registry names and import paths at the tested PyOD checkout. This explicit
# inventory prevents a newly installed PyOD release from silently changing the
# denominator of the comparison.
BASELINE_CLASSES = {
    "ECOD": "pyod.models.ecod.ECOD",
    "ABOD": "pyod.models.abod.ABOD",
    "COPOD": "pyod.models.copod.COPOD",
    "MAD": "pyod.models.mad.MAD",
    "SOS": "pyod.models.sos.SOS",
    "QMCD": "pyod.models.qmcd.QMCD",
    "KDE": "pyod.models.kde.KDE",
    "Sampling": "pyod.models.sampling.Sampling",
    "GMM": "pyod.models.gmm.GMM",
    "PCA": "pyod.models.pca.PCA",
    "KPCA": "pyod.models.kpca.KPCA",
    "MCD": "pyod.models.mcd.MCD",
    "CD": "pyod.models.cd.CD",
    "OCSVM": "pyod.models.ocsvm.OCSVM",
    "LMDD": "pyod.models.lmdd.LMDD",
    "LOF": "pyod.models.lof.LOF",
    "COF": "pyod.models.cof.COF",
    "CBLOF": "pyod.models.cblof.CBLOF",
    "LOCI": "pyod.models.loci.LOCI",
    "HBOS": "pyod.models.hbos.HBOS",
    "HDBSCAN": "pyod.models.hdbscan.HDBSCAN",
    "KNN": "pyod.models.knn.KNN",
    "SOD": "pyod.models.sod.SOD",
    "ROD": "pyod.models.rod.ROD",
    "IForest": "pyod.models.iforest.IForest",
    "INNE": "pyod.models.inne.INNE",
    "DIF": "pyod.models.dif.DIF",
    "FeatureBagging": "pyod.models.feature_bagging.FeatureBagging",
    "LSCP": "pyod.models.lscp.LSCP",
    "LODA": "pyod.models.loda.LODA",
    "SUOD": "pyod.models.suod.SUOD",
    "XGBOD": "pyod.models.xgbod.XGBOD",
    "AutoEncoder": "pyod.models.auto_encoder.AutoEncoder",
    "VAE": "pyod.models.vae.VAE",
    "SO_GAAL": "pyod.models.so_gaal.SO_GAAL",
    "MO_GAAL": "pyod.models.mo_gaal.MO_GAAL",
    "DeepSVDD": "pyod.models.deep_svdd.DeepSVDD",
    "AnoGAN": "pyod.models.anogan.AnoGAN",
    "ALAD": "pyod.models.alad.ALAD",
    "AE1SVM": "pyod.models.ae1svm.AE1SVM",
    "DevNet": "pyod.models.devnet.DevNet",
    "RGraph": "pyod.models.rgraph.RGraph",
    "LUNAR": "pyod.models.lunar.LUNAR",
}

METHODS = NSA_METHODS + list(BASELINE_CLASSES)

# These reductions are specified before seeing outcomes and are intentionally
# visible in METHOD_META. They make a broad CPU screen possible but do not
# reproduce the corresponding papers' default training budgets. The report
# must not use a limited-budget deep baseline to assert NSA superiority.
COMPUTE_OVERRIDES = {
    "DIF": {"n_ensemble": 10},                 # PyOD default: 50
    "DeepSVDD": {"epochs": 50},                # default: 100
    "AnoGAN": {"epochs": 50, "epochs_query": 5},  # defaults: 500, 20
    "ALAD": {"epochs": 50},                    # default: 200
    "LUNAR": {"n_epochs": 50},                # default: 200
}

REQUIRES = {
    "DIF": ["torch"], "AutoEncoder": ["torch"], "VAE": ["torch"],
    "SO_GAAL": ["torch"], "MO_GAAL": ["torch"],
    "DeepSVDD": ["torch"], "AnoGAN": ["torch"], "ALAD": ["torch"],
    "AE1SVM": ["torch"], "DevNet": ["torch"],
    "LUNAR": ["torch"], "SUOD": ["suod", "combo"],
    "FeatureBagging": ["combo"],
    "XGBOD": ["xgboost"],
    "HDBSCAN": ["sklearn.cluster.HDBSCAN"],
}

# These implementations can score a test collection using neighborhoods or
# distances among test rows. They are listed, but are not purely inductive
# pointwise novelty models like NSA. The report should flag this distinction.
BATCH_DEPENDENT = {
    "COF", "LOCI", "SOS", "SOD",  # query-only neighborhoods/distances
    "ECOD", "COPOD",               # append the whole query batch to train
    "RGraph",                      # graph update includes query blocks
    "SUOD",                        # its default ensemble includes COPOD
}

METHOD_META: dict[str, dict[str, Any]] = {}
for _method in NSA_METHODS:
    _strategy = _method.split("/")[1]
    _reason = ""
    if _strategy == "voronoi":
        _reason = "Voronoi implementation accepts only 2 or 3 features."
    elif _strategy == "deterministic":
        _reason = "Deterministic lattice implementation accepts only 2 features."
    elif _strategy == "binary":
        _reason = ("Hamming uses default match_threshold=1 (0 in 1D); "
                   "contiguous/chunk use a fixed, label-free 40%-of-feature "
                   "matching length, matching the upstream example's 12/30 ratio.")
    METHOD_META[_method] = {
        "family": "NSA",
        "class_path": "pyod.models.nsa.NSA",
        "applicability": "applicable",
        "reason": _reason,
        "requires": [],
        "strategy": _strategy,
        "batch_dependent": False,
        "inference_mode": "inductive",
        "compute_overrides": {},
    }

for _method, _path in BASELINE_CLASSES.items():
    _reason = ""
    _applicability = "applicable"
    if _method in {"XGBOD", "DevNet"}:
        _applicability = "requires_labeled_outliers"
        _reason = ("fit(X, y) requires outlier labels and a positive class; "
                   "clean-normal training supplies neither.")
    elif _method == "MAD":
        _reason = "Univariate only; inapplicable to the native and PCA(2) panels."
    elif _method == "HDBSCAN":
        _reason = "Requires sklearn.cluster.HDBSCAN in the installed scikit-learn."
    METHOD_META[_method] = {
        "family": "baseline",
        "class_path": _path,
        "applicability": _applicability,
        "reason": _reason,
        "requires": REQUIRES.get(_method, []),
        "batch_dependent": _method in BATCH_DEPENDENT,
        "inference_mode": ("batch-dependent" if _method in BATCH_DEPENDENT
                           else "inductive"),
        "compute_overrides": COMPUTE_OVERRIDES.get(_method, {}),
    }


@dataclass
class ModelBuildError(Exception):
    """Constructor/import/applicability failure with a machine-readable stage."""

    method: str
    stage: str
    detail: str

    def __str__(self) -> str:
        return f"{self.method} [{self.stage}]: {self.detail}"


def _load_class(class_path: str, method: str):
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as exc:
        raise ModelBuildError(method, "import", f"{type(exc).__name__}: {exc}") from exc


def _bounded_neighbors(default: int, n_train: int) -> int:
    return max(1, min(default, n_train - 1))


def _baseline_kwargs(name: str, seed: int, n_train: int,
                     n_features: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if name in {"Sampling", "GMM", "PCA", "KPCA", "MCD", "LMDD", "CBLOF",
                "IForest", "INNE", "DIF", "FeatureBagging", "LODA",
                "AutoEncoder", "VAE", "DeepSVDD", "LUNAR"}:
        kwargs["random_state"] = seed
    if name in {"ECOD", "ABOD", "COPOD", "LOF", "KNN", "SOD",
                "CBLOF", "HDBSCAN", "IForest", "FeatureBagging", "SUOD",
                "LUNAR"}:
        kwargs["n_jobs"] = 1
    if name in {"ABOD", "LOF", "COF", "SOD"}:
        kwargs["n_neighbors"] = _bounded_neighbors(
            5 if name == "ABOD" else 20, n_train)
    if name == "KNN":
        kwargs["n_neighbors"] = _bounded_neighbors(5, n_train)
    if name == "HDBSCAN":
        kwargs["min_cluster_size"] = min(5, n_train)
    if name == "DeepSVDD":
        kwargs["n_features"] = n_features
        kwargs["verbose"] = 0
    if name in {"AutoEncoder", "VAE"}:
        kwargs["verbose"] = 0
    if name == "RGraph":
        kwargs["verbose"] = False
    if name == "LSCP":
        # LSCP has no default detector_list. Three predeclared, heterogeneous
        # PyOD bases; no test labels are used to select them.
        from pyod.models.lof import LOF
        from pyod.models.knn import KNN
        from pyod.models.hbos import HBOS
        kwargs["detector_list"] = [
            LOF(n_neighbors=_bounded_neighbors(20, n_train), n_jobs=1),
            KNN(n_neighbors=_bounded_neighbors(5, n_train), n_jobs=1),
            HBOS(),
        ]
        kwargs["random_state"] = seed
    kwargs.update(COMPUTE_OVERRIDES.get(name, {}))
    return kwargs


def build_model(name: str, seed: int, n_train: int, n_features: int):
    """Construct one registered PyOD method or raise ``ModelBuildError``.

    Import and constructor failures are kept distinct from method fit and
    score failures, which the benchmark runner records separately.
    """
    if name not in METHOD_META:
        raise ModelBuildError(name, "lookup", "unknown benchmark method")
    if n_train < 2 or n_features < 1:
        raise ModelBuildError(name, "input", "need at least 2 rows and 1 feature")
    meta = METHOD_META[name]
    if meta["applicability"] != "applicable":
        raise ModelBuildError(name, "applicability", meta["reason"])
    if name == "MAD" and n_features != 1:
        raise ModelBuildError(name, "applicability", "MAD requires one feature")
    if name == "NSA/voronoi" and n_features not in (2, 3):
        raise ModelBuildError(name, "applicability", "Voronoi requires 2 or 3 features")
    if name == "NSA/deterministic" and n_features != 2:
        raise ModelBuildError(name, "applicability", "deterministic requires 2 features")

    if name.startswith("NSA/"):
        strategy = name.split("/")[1]
        kwargs = {"strategy": strategy, "random_state": seed}
        if strategy == "binary":
            rule = name.split("/")[2]
            kwargs["binary_match"] = rule
            if rule != "hamming":
                # The upstream examples use a 12-of-30-bit matching window.
                # The same label-free 40% rule is fixed for every dataset.
                kwargs["match_threshold"] = max(1, round(0.4 * n_features))
            elif n_features == 1:
                kwargs["match_threshold"] = 0
        params = kwargs
    else:
        params = _baseline_kwargs(name, seed, n_train, n_features)

    cls = _load_class(meta["class_path"], name)
    try:
        return cls(**params)
    except Exception as exc:
        raise ModelBuildError(name, "constructor", f"{type(exc).__name__}: {exc}") from exc


__all__ = ["METHODS", "METHOD_META", "ModelBuildError", "build_model",
           "BASELINE_CLASSES", "NSA_METHODS", "COMPUTE_OVERRIDES"]
