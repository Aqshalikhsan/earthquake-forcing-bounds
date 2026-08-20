"""
Train the auditor, and calibrate it so that its confidence means something.

The classifier maps a fingerprint to a mechanism. That alone would be a black
box giving a verdict, which is the thing this project spends a paper arguing
against, so two properties are built in on top of it.

CONFORMAL PREDICTION SETS. The output is not one label but the set of labels
consistent with the fingerprint at a stated confidence. If a claim is
ambiguous, the set is large and says so, instead of a point verdict that hides
the ambiguity. The guarantee is distribution-free and comes from a held-out
calibration split, so it holds whatever the classifier does.

ASYMMETRIC COVERAGE. The coverage level is set per class, deliberately. REAL is
given 99% coverage and the artefact classes 90%, which makes the tool reluctant
to rule out a real association and comparatively willing to name an artefact.
The consequence is the behaviour we want: a set containing only artefact labels
is strong evidence, while a set containing REAL is weak evidence of anything.
An auditor should be able to refute much more confidently than it can endorse.

ABSTENTION. A fingerprint far from everything seen in training gets no verdict
at all. Mahalanobis distance in fingerprint space, thresholded at the 99th
percentile of the training distribution, decides that. Without it the classifier
would confidently label inputs of a kind it has never seen, which is exactly
how an audit tool becomes worse than no audit tool.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile  # noqa: F401

import json
import numpy as np

from fingerprint import FEATURES
from generate import build, CLASSES

ALPHA = {"REAL": 0.01}          # reluctant to rule REAL out
ALPHA_DEFAULT = 0.10            # willing to name an artefact
OOD_PCT = 99.0


def usable_features(X):
    """Columns that carry information; null_fpr is absent in cheap mode."""
    keep = [j for j in range(X.shape[1])
            if np.isfinite(X[:, j]).mean() > 0.5 and np.nanstd(X[:, j]) > 0]
    return keep, [FEATURES[j] for j in keep]


def fit(X, lab, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # deliberately small: a saturated model makes conformal calibration
    # degenerate, and the fingerprint is low-dimensional and structured
    m = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=8,
        min_samples_leaf=15, l2_regularization=2.0, random_state=seed)
    m.fit(X, lab)
    return m


def _margin(proba, j):
    """
    Nonconformity as a margin rather than as one minus the probability.

    With a confident model the probabilities saturate, every calibration score
    collapses towards zero, and the quantile with it, so almost everything is
    excluded and the coverage promise silently fails. The margin between the
    true class and the best competitor stays spread out even when the model is
    confident, which keeps the quantile informative.
    """
    p_true = proba[:, j]
    other = np.delete(proba, j, axis=1).max(axis=1)
    return other - p_true


def conformal_quantiles(model, Xc, yc):
    """
    Class-conditional (Mondrian) calibration.

    One quantile per class, computed only from calibration examples of that
    class, so the coverage promise is per class rather than on average. An
    average promise would let a rare class be systematically missed while the
    headline number still looked correct.
    """
    proba = model.predict_proba(Xc)
    order = list(model.classes_)
    q = {}
    for c in order:
        m = (yc == c)
        if m.sum() < 5:
            q[c] = 1.0
            continue
        scores = _margin(proba[m], order.index(c))
        a = ALPHA.get(c, ALPHA_DEFAULT)
        k = int(np.ceil((m.sum() + 1) * (1 - a))) - 1
        k = int(np.clip(k, 0, m.sum() - 1))
        q[c] = float(np.sort(scores)[k])
    return q


def predict_set(model, q, X):
    proba = model.predict_proba(X)
    order = list(model.classes_)
    out = []
    for row in proba:
        r = row[None, :]
        s = {c: float(_margin(r, order.index(c))[0]) for c in order}
        out.append([c for c in order if s[c] <= q[c]])
    return out, proba


def ood_model(X):
    mu = np.nanmean(X, axis=0)
    Z = np.where(np.isfinite(X), X, mu)
    cov = np.cov(Z.T) + np.eye(Z.shape[1]) * 1e-6
    inv = np.linalg.pinv(cov)
    d = np.array([np.sqrt((z - mu) @ inv @ (z - mu)) for z in Z])
    return dict(mu=mu.tolist(), inv=inv.tolist(),
                thresh=float(np.percentile(d, OOD_PCT)))


def main(n_per_class=60, seed=20260819):
    print("building training set ...")
    X, lab = build(n_per_class=n_per_class, seed=seed, cheap=True)
    keep, names = usable_features(X)
    X = X[:, keep]
    print(f"\nfingerprint dimensions kept: {len(keep)}  {names}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_fit = int(X.shape[0] * 0.45)
    n_cal = int(X.shape[0] * 0.35)
    i_fit, i_cal, i_te = perm[:n_fit], perm[n_fit:n_fit + n_cal], \
        perm[n_fit + n_cal:]

    model = fit(X[i_fit], lab[i_fit], seed)
    q = conformal_quantiles(model, X[i_cal], lab[i_cal])
    sets, proba = predict_set(model, q, X[i_te])
    order = list(model.classes_)

    print("\n=== point accuracy on held-out")
    pred = model.predict(X[i_te])
    print(f"  {(pred == lab[i_te]).mean():.3f} over {i_te.size} examples")

    print("\n=== confusion (rows true, columns predicted)")
    print("            " + " ".join(f"{c[:5]:>6}" for c in order))
    for c in order:
        m = lab[i_te] == c
        row = [int(((pred == d) & m).sum()) for d in order]
        print(f"  {c:<10} " + " ".join(f"{v:>6d}" for v in row))

    print("\n=== conformal behaviour")
    print(f"  {'class':<10} {'coverage':>9} {'set size':>9} {'alpha':>7}")
    for c in order:
        m = lab[i_te] == c
        if m.sum() == 0:
            continue
        cov = np.mean([c in s for s, keep_ in zip(sets, m) if keep_])
        size = np.mean([len(s) for s, keep_ in zip(sets, m) if keep_])
        print(f"  {c:<10} {cov:>9.2f} {size:>9.2f} "
              f"{ALPHA.get(c, ALPHA_DEFAULT):>7.2f}")

    print("\n=== the asymmetry that matters")
    art = [c for c in order if c not in ("REAL", "NULL")]
    is_art = np.isin(lab[i_te], art)
    art_only = np.array([len(s) > 0 and all(c in art for c in s) for s in sets])
    real_in = np.array(["REAL" in s for s in sets])
    print(f"  artefact-only verdict, given a true artefact : "
          f"{art_only[is_art].mean():.2f}")
    print(f"  artefact-only verdict, given a true REAL     : "
          f"{art_only[lab[i_te] == 'REAL'].mean():.2f}   (should be near 0)")
    print(f"  REAL kept in the set, given a true REAL      : "
          f"{real_in[lab[i_te] == 'REAL'].mean():.2f}")

    ood = ood_model(X[i_fit])
    out = DATA / "results" / "audit_model.json"
    json.dump(dict(features=names, classes=order, quantiles=q, ood=ood,
                   n_train=int(i_fit.size), n_cal=int(i_cal.size)),
              io_open(out), indent=1)
    print(f"\nsaved {out.name}")
    return model, q, names


def io_open(p):
    import io
    return io.open(p, "w", encoding="utf-8")


if __name__ == "__main__":
    n = int(_sys.argv[1]) if len(_sys.argv) > 1 else 60
    main(n_per_class=n)
