"""
Export the trained auditor so it can run in a browser, and prove it still agrees.

The audit tool is meant to be usable by anyone, which means it has to run where
they already are rather than behind a server they have to trust. The analysis
itself is far too heavy for that, but the fingerprint is cheap arithmetic and
the model is a small tree ensemble, so both can be evaluated client-side.

The risk in porting a model is silent disagreement: a JavaScript
reimplementation that is subtly different still returns plausible numbers, and
the conformal guarantee, which was measured on the Python version, no longer
applies to what users actually run. This module therefore does two things. It
writes the ensemble out node by node, and it then re-evaluates the exported
form with an independent implementation and checks it against scikit-learn to
machine precision. If the two disagree the export fails rather than shipping.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, datafile  # noqa: F401

import io
import json
import numpy as np

from fingerprint import FEATURES
from generate import top_up
from train import (usable_features, fit, conformal_quantiles, ood_model,
                   ALPHA, ALPHA_DEFAULT)

TOL = 1e-9


def dump_tree(pred):
    """One tree as flat arrays, which is what a browser can walk cheaply."""
    n = pred.nodes
    return dict(
        feature=[int(v) for v in n["feature_idx"]],
        threshold=[float(v) for v in n["num_threshold"]],
        left=[int(v) for v in n["left"]],
        right=[int(v) for v in n["right"]],
        value=[float(v) for v in n["value"]],
        leaf=[int(v) for v in n["is_leaf"]],
        missing_left=[int(v) for v in n["missing_go_to_left"]],
    )


def walk(tree, x):
    """
    The traversal the browser will perform, written here to be checked.

    Missing values are not imputed: scikit-learn sends them down whichever side
    the split learned, and the fingerprint does produce missing values when a
    series is too short or too flat for an axis to be defined.
    """
    i = 0
    while not tree["leaf"][i]:
        f = tree["feature"][i]
        v = x[f]
        if v != v:                                   # NaN
            i = tree["left"][i] if tree["missing_left"][i] else tree["right"][i]
        elif v <= tree["threshold"][i]:
            i = tree["left"][i]
        else:
            i = tree["right"][i]
    return tree["value"][i]


def raw_scores(bundle, x):
    base = np.array(bundle["baseline"], dtype=float)
    out = base.copy()
    for stage in bundle["trees"]:
        for k, tree in enumerate(stage):
            out[k] += walk(tree, x)
    return out


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main(seed=20260819, base_n=90, extra_real=240):
    X, lab = top_up("REAL", extra_real, base_n=base_n, seed=seed)
    keep, names = usable_features(X)
    X = X[:, keep]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_fit = int(X.shape[0] * 0.45)
    n_cal = int(X.shape[0] * 0.35)
    i_fit, i_cal = perm[:n_fit], perm[n_fit:n_fit + n_cal]

    model = fit(X[i_fit], lab[i_fit], seed)
    q = conformal_quantiles(model, X[i_cal], lab[i_cal])
    order = [str(c) for c in model.classes_]

    baseline = np.ravel(model._baseline_prediction).astype(float)
    if baseline.size == 1 and len(order) > 1:
        baseline = np.repeat(baseline, len(order))
    trees = [[dump_tree(p) for p in stage] for stage in model._predictors]

    bundle = dict(
        features=names,
        classes=order,
        alpha={c: ALPHA.get(c, ALPHA_DEFAULT) for c in order},
        quantiles={str(k): float(v) for k, v in q.items()},
        baseline=[float(v) for v in baseline],
        trees=trees,
        ood=ood_model(X[i_fit]),
        # class centroids and spreads, so the interface can show where a claim
        # sits against each mechanism's real signature instead of an invented one
        centroid={c: [None if not np.isfinite(v) else round(float(v), 6)
                      for v in np.nanmean(X[lab == c], axis=0)]
                  for c in order},
        spread={c: [None if not np.isfinite(v) else round(float(v), 6)
                    for v in np.nanstd(X[lab == c], axis=0)]
                for c in order},
        n_train=int(i_fit.size),
        n_cal=int(i_cal.size),
        n_trees=len(trees),
    )

    # ---- the check that decides whether this ships ------------------------
    ref = model.predict_proba(X)
    mine = np.array([softmax(raw_scores(bundle, row)) for row in X])
    err = float(np.max(np.abs(ref - mine)))
    print(f"  classes      {order}")
    print(f"  axes         {len(names)}")
    print(f"  boosting stages {len(trees)}, trees {sum(len(s) for s in trees)}")
    print(f"  max |sklearn - exported| = {err:.3e}")
    if err > TOL:
        raise SystemExit(f"export disagrees with scikit-learn by {err:.3e}; "
                         "not written")

    site = ROOT / "site"
    site.mkdir(exist_ok=True)
    for path in (DATA / "results" / "audit_model.json", site / "audit_model.json"):
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, separators=(",", ":"))
        print(f"  wrote {path.relative_to(ROOT)}  "
              f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
