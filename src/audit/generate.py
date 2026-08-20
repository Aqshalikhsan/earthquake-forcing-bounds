"""
Training data for the auditor, generated rather than collected.

A classifier that judges whether a precursor claim is real needs examples of
both, and the real class is empty: no external forcing has ever been shown to
anticipate earthquakes. Collecting labelled claims is therefore impossible, and
a classifier trained on published claims would only learn to say no.

The way out is to make the ground truth instead of finding it. We control the
mechanism, so we can plant a real association of known size, and we can plant
each artefact whose mechanism we have already measured. Every example is built
from the real catalogue and real forcing series, so the noise, the clustering
and the seasonality are genuine; only the mechanism under study is imposed.

    REAL        a modulation of known size is written into the event times
    NULL        no association at all, and a clean pipeline
    CALENDAR    the predictor is a deterministic function of the date
    CLUSTER     aftershocks are left in, so one sequence counts many times
    LOOKAHEAD   the predictor window reaches across the event
    SEARCH      the best of many predictors is reported without correction
    GROWTH      both series ride the growth of the catalogue

The classes are defined by how they RESPOND to the perturbation battery in
fingerprint.py, not by the size of any single p-value. Two of them routinely
produce p below 0.001, which is the entire point.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile  # noqa: F401

import numpy as np

from fingerprint import fingerprint, FEATURES, _deciles, NDEC

CLASSES = ["REAL", "NULL", "CALENDAR", "CLUSTER", "LOOKAHEAD", "SEARCH",
           "GROWTH"]


def load_base():
    """The real daily grid: M>=6 counts, and the growing all-magnitude count."""
    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy")).astype(float)
    y = (cnt > 0).astype(float)
    try:
        gdays = np.load(datafile("daily_days.npy"))
        gcnt = np.load(datafile("daily_counts.npy")).astype(float)
        table = {int(d): i for i, d in enumerate(gdays)}
        idx = np.array([table.get(int(d), -1) for d in days])
        grow = np.where(idx >= 0, gcnt[np.clip(idx, 0, gcnt.size - 1)], np.nan)
    except Exception:
        grow = None
    return days, y, grow


def smooth_noise(n, scale, rng):
    """A smooth random series, the shape most forcing variables actually have."""
    w = rng.normal(size=n)
    k = np.ones(int(scale)) / int(scale)
    s = np.convolve(w, k, mode="same")
    return (s - s.mean()) / (s.std() + 1e-12)


def calendar_series(n, period, rng):
    """A deterministic function of the date, like a planetary angle."""
    t = np.arange(n, dtype=float)
    phase = rng.uniform(0, 2 * np.pi)
    return (np.sin(2 * np.pi * t / period + phase)
            + 0.3 * np.sin(4 * np.pi * t / (period * 1.7) + phase))


def inject_real(x, y, beta, rng):
    """
    Move a fraction beta of event days onto high-decile days of the predictor.

    The event count is preserved and only the timing changes, so the base rate
    is untouched and the modulation is exactly what beta says it is.
    """
    y = y.copy()
    idx = _deciles(x)
    top = np.flatnonzero(idx >= NDEC - 2)
    ev = np.flatnonzero(y > 0)
    if top.size == 0 or ev.size == 0:
        return y
    k = int(round(beta * ev.size))
    if k == 0:
        return y
    move = rng.choice(ev, size=min(k, ev.size), replace=False)
    y[move] = 0.0
    land = rng.choice(top, size=move.size, replace=True)
    y[land] = 1.0
    return y


def add_aftershocks(y, rng, n_seq=60, decay=1.1, span=40):
    """Omori-like bursts after a random subset of events, left undeclustered."""
    y = y.copy()
    ev = np.flatnonzero(y > 0)
    if ev.size == 0:
        return y
    for m in rng.choice(ev, size=min(n_seq, ev.size), replace=False):
        for d in range(1, span):
            if rng.random() < 1.0 / (d ** decay):
                j = m + d
                if j < y.size:
                    y[j] = 1.0
    return y


def contaminate(x, y, rng):
    """
    Add weak versions of the other mechanisms on top of the dominant one.

    A published claim rarely carries exactly one flaw. It has a leading
    mechanism and traces of the others, and a classifier trained on pure
    examples learns a boundary that does not exist in the wild. The label
    stays the dominant mechanism; everything added here is deliberately too
    weak to take it over.
    """
    if rng.random() < 0.35:
        y = add_aftershocks(y, rng, n_seq=int(rng.integers(5, 25)), span=15)
    if rng.random() < 0.30:
        t = np.arange(x.size, dtype=float) / x.size
        x = x + t * rng.uniform(0.1, 0.6) * np.std(x)
    if rng.random() < 0.40:
        x = x + rng.normal(0, rng.uniform(0.2, 0.9) * np.std(x), x.size)
    return x, y


def make_example(cls, days, y0, grow, rng):
    """One labelled example: a predictor, a target, and the class that made it."""
    n = y0.size

    if cls == "REAL":
        x = smooth_noise(n, rng.integers(5, 60), rng)
        # log-uniform from 2%, so most examples sit where the decision is hard
        beta = float(np.exp(rng.uniform(np.log(0.02), np.log(0.45))))
        y = inject_real(x, y0, beta, rng)

    elif cls == "NULL":
        x = smooth_noise(n, rng.integers(5, 60), rng)
        y = np.roll(y0, int(rng.integers(400, n - 400)))

    elif cls == "CALENDAR":
        x = calendar_series(n, rng.uniform(300, 4000), rng)
        x = x + rng.normal(0, rng.uniform(0.05, 0.8), n)   # never a pure sine
        y = y0.copy()

    elif cls == "CLUSTER":
        # The association has to be carried ENTIRELY by the clustered days.
        # Injecting it into the mainshocks and then adding aftershocks behind
        # them does the opposite: the sequences land where the predictor is
        # still high, so they reinforce a real association instead of
        # manufacturing a false one, and declustering leaves it standing.
        # Here the mainshocks are placed at random, and only the ones that
        # happen to sit high in the predictor are given long sequences.
        x = smooth_noise(n, rng.integers(20, 120), rng)
        idx = _deciles(x)
        y = np.zeros(n)
        main = rng.choice(n, size=int(y0.sum() * rng.uniform(0.06, 0.15)),
                          replace=False)
        y[main] = 1.0
        span = int(rng.integers(25, 60))
        for m in main:
            if idx[m] < NDEC - 3:
                continue                      # a short, ordinary sequence
            for d in range(1, span):
                if rng.random() < 1.0 / (d ** rng.uniform(0.6, 0.9)):
                    if m + d < n:
                        y[m + d] = 1.0

    elif cls == "LOOKAHEAD":
        # the predictor window reaches forward across the event, which is the
        # error that turned an apparent p < 0.0001 into p = 0.36 once caught
        k = int(rng.integers(2, 40))
        lead = np.convolve(y0, np.ones(k) / k, mode="full")[k - 1:k - 1 + n]
        x = lead + rng.normal(0, rng.uniform(0.4, 1.2), n)

        y = y0.copy()

    elif cls == "SEARCH":
        # the best of many, reported as if it had been the only one
        best, bx = -1.0, None
        for _ in range(int(rng.integers(5, 80))):
            c = smooth_noise(n, rng.integers(5, 60), rng)
            s = np.abs(np.corrcoef(c, y0)[0, 1])
            if s > best:
                best, bx = s, c
        x, y = bx, y0.copy()

    elif cls == "GROWTH":
        base = grow if grow is not None and np.isfinite(grow).mean() > 0.8 \
            else np.linspace(0.2, 1.0, n) + smooth_noise(n, 200, rng) * 0.05
        t = np.arange(n, dtype=float)
        x = (t / n) * rng.uniform(1.0, 3.0) + smooth_noise(n, 90, rng) * 0.4
        pr = base / (np.nanmax(base) + 1e-12)
        y = (rng.random(n) < np.clip(pr * y0.mean() * 2.0, 0, 1)).astype(float)

    else:
        raise ValueError(cls)

    x, y = contaminate(np.asarray(x, dtype=float),
                       np.asarray(y, dtype=float), rng)
    return x, y


def build(n_per_class=40, seed=20260819, cheap=True, verbose=True,
          cache=True):
    """Fingerprints are expensive and deterministic in the seed, so they are
    cached: every later experiment reuses one build instead of repeating it."""
    path = DATA / "results" / f"audit_fp_{n_per_class}_{seed}.npz"
    if cache and path.exists():
        z = np.load(path, allow_pickle=True)
        if verbose:
            print(f"    reusing {path.name}")
        return z["X"], z["lab"]

    rng = np.random.default_rng(seed)
    days, y0, grow = load_base()
    X, lab = [], []
    for cls in CLASSES:
        for i in range(n_per_class):
            x, y = make_example(cls, days, y0, grow, rng)
            f, _ = fingerprint(x, y, rng=rng, cheap=cheap)
            X.append(f)
            lab.append(cls)
            if verbose:
                print(f"    {cls:<10} {i+1}/{n_per_class}", end="\r", flush=True)
        if verbose:
            print(f"    {cls:<10} {n_per_class} done            ")
    X, lab = np.array(X), np.array(lab)
    if cache:
        np.savez(path, X=X, lab=lab)
    return X, lab


def top_up(cls, n_extra, base_n=90, seed=20260819, verbose=True):
    """
    Add more examples of one class to an existing build.

    Conformal coverage at level 1-alpha cannot exceed n/(n+1) for n calibration
    examples of that class, so a 99% promise for REAL needs at least 99 of them
    in calibration and no amount of tuning substitutes for that. The classes
    are therefore deliberately unbalanced: REAL is the class the tool must be
    slowest to rule out, so it is the class given the most calibration data.

    Regenerating everything to change one class would waste the cached
    fingerprints, so the extra examples are appended to what is already there.
    """
    X, lab = build(n_per_class=base_n, seed=seed, verbose=False)
    path = DATA / "results" / f"audit_fp_{base_n}_{seed}_{cls}{n_extra}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return (np.vstack([X, z["X"]]), np.concatenate([lab, z["lab"]]))

    rng = np.random.default_rng(seed + 977)
    days, y0, grow = load_base()
    Xe = []
    for i in range(n_extra):
        x, y = make_example(cls, days, y0, grow, rng)
        f, _ = fingerprint(x, y, rng=rng, cheap=True)
        Xe.append(f)
        if verbose:
            print(f"    extra {cls} {i+1}/{n_extra}", end="\r", flush=True)
    if verbose:
        print(f"    extra {cls} {n_extra} done            ")
    Xe = np.array(Xe)
    lab_e = np.array([cls] * n_extra)
    np.savez(path, X=Xe, lab=lab_e)
    return np.vstack([X, Xe]), np.concatenate([lab, lab_e])


if __name__ == "__main__":
    X, lab = build(n_per_class=6, cheap=True)
    print("\nshape", X.shape)
    for j, name in enumerate(FEATURES):
        col = X[:, j]
        print(f"  {name:<13} finite {np.isfinite(col).mean():4.0%}  "
              f"range [{np.nanmin(col):8.3f}, {np.nanmax(col):8.3f}]")
