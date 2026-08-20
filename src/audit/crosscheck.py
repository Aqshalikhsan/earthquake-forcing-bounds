"""
Prove the browser computes the same fingerprint as Python.

The conformal coverage this tool promises was measured on the Python
implementation. If the JavaScript port drifts from it, the promise is void and
nothing in the interface would show that: a slightly different fingerprint
still lands somewhere in the model and still produces a confident-looking
verdict.

So the two are compared directly. This script writes a set of test series and
their Python fingerprints, and site/crosscheck.mjs recomputes them in Node and
reports the largest disagreement per axis. Run them in that order.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT  # noqa: F401

import io
import json
import numpy as np

from fingerprint import fingerprint, FEATURES

N = 4000


def cases(rng):
    """Deliberately varied: a smooth predictor, a calendar one, and a
    contaminated one, so the comparison covers every branch."""
    t = np.arange(N, dtype=float)
    smooth = np.convolve(rng.normal(size=N), np.ones(25) / 25, mode="same")
    cal = np.sin(2 * np.pi * t / 900.0) + 0.3 * np.sin(2 * np.pi * t / 311.0)
    trend = t / N * 2.0 + smooth * 0.4

    y = (rng.random(N) < 0.29).astype(float)
    yc = y.copy()
    for i in np.flatnonzero(y > 0)[:300]:
        yc[i + 1:i + 4] = 1.0

    return {"smooth": (smooth, y), "calendar": (cal, y), "trend": (trend, yc)}


def main():
    rng = np.random.default_rng(4242)
    out = {}
    for name, (x, y) in cases(rng).items():
        vec, _ = fingerprint(x, y, cheap=True)
        out[name] = dict(
            x=[float(v) for v in x],
            y=[float(v) for v in y],
            python=[None if not np.isfinite(v) else float(v) for v in vec],
        )
        print(f"  {name:<9} fingerprinted")
    path = ROOT / "site" / "crosscheck.json"
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(features=FEATURES[:15], cases=out), fh)
    print(f"  wrote {path.relative_to(ROOT)}  "
          f"{path.stat().st_size/1024:.0f} KB")
    print("  now run:  node site/crosscheck.mjs")


if __name__ == "__main__":
    main()
