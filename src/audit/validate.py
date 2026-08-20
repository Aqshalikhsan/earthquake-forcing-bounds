"""
Does the conformal guarantee actually hold?

A single train/calibrate/test split gives one coverage estimate per class, from
twenty or so examples, and the standard error on that is about six percentage
points. Reading such a number as a failure or a success is reading noise. An
earlier run of this project did exactly that: two classes came out at 0.72 and
0.78 against a nominal 0.90, which looked like a broken calibration and was in
fact within what twenty-five samples produce.

Fingerprints are expensive and cached, so the honest check costs almost
nothing: hold the fingerprints fixed, repeat the split many times, and read the
mean coverage with a proper interval on it. Anything still short of nominal
after that is a real defect.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile  # noqa: F401

import numpy as np

from generate import build, top_up
from train import (usable_features, fit, conformal_quantiles, predict_set,
                   ALPHA, ALPHA_DEFAULT)


def run(n_per_class=90, seed=20260819, n_rep=60):
    # REAL is topped up because its 99% coverage target needs at least 99
    # calibration examples of that class; see top_up for why.
    X, lab = top_up("REAL", 240, base_n=n_per_class, seed=seed)
    keep, names = usable_features(X)
    X = X[:, keep]
    classes = sorted(set(lab))

    cov = {c: [] for c in classes}
    size = {c: [] for c in classes}
    acc, art_real, art_art, real_real = [], [], [], []

    for r in range(n_rep):
        rng = np.random.default_rng(1000 + r)
        perm = rng.permutation(X.shape[0])
        n_fit = int(X.shape[0] * 0.45)
        n_cal = int(X.shape[0] * 0.35)
        i_f, i_c, i_t = (perm[:n_fit], perm[n_fit:n_fit + n_cal],
                         perm[n_fit + n_cal:])

        model = fit(X[i_f], lab[i_f], seed=r)
        q = conformal_quantiles(model, X[i_c], lab[i_c])
        sets, _ = predict_set(model, q, X[i_t])
        acc.append(float((model.predict(X[i_t]) == lab[i_t]).mean()))

        for c in classes:
            m = lab[i_t] == c
            if m.sum() == 0:
                continue
            cov[c].append(np.mean([c in s for s, k in zip(sets, m) if k]))
            size[c].append(np.mean([len(s) for s, k in zip(sets, m) if k]))

        art = [c for c in classes if c not in ("REAL", "NULL")]
        only = np.array([len(s) > 0 and all(c in art for c in s) for s in sets])
        is_a = np.isin(lab[i_t], art)
        is_r = lab[i_t] == "REAL"
        art_art.append(float(only[is_a].mean()))
        art_real.append(float(only[is_r].mean()) if is_r.any() else np.nan)
        real_real.append(float(np.mean([("REAL" in s) for s, k in
                                        zip(sets, is_r) if k])))
        print(f"    split {r+1}/{n_rep}", end="\r", flush=True)
    print(" " * 30, end="\r")

    print(f"\n{n_rep} splits, {X.shape[0]} fingerprints, "
          f"{len(names)} axes\n")
    print(f"  point accuracy   {np.mean(acc):.3f} "
          f"+/- {np.std(acc):.3f}\n")
    print(f"  {'class':<10} {'nominal':>8} {'coverage':>19} {'set size':>9}")
    print("  " + "-" * 50)
    for c in classes:
        a = 1 - ALPHA.get(c, ALPHA_DEFAULT)
        m = np.mean(cov[c])
        se = np.std(cov[c]) / np.sqrt(len(cov[c]))
        flag = "" if m + 2 * se >= a else "   UNDER"
        print(f"  {c:<10} {a:>8.2f} {m:>13.3f} +/-{2*se:5.3f} "
              f"{np.mean(size[c]):>9.2f}{flag}")

    print(f"""
  artefact-only verdict, given a true artefact : {np.mean(art_art):.3f}
  artefact-only verdict, given a true REAL     : {np.nanmean(art_real):.3f}
  REAL kept in the set, given a true REAL      : {np.mean(real_real):.3f}
""")


if __name__ == "__main__":
    run(n_rep=int(_sys.argv[1]) if len(_sys.argv) > 1 else 60)
