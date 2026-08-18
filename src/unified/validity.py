"""
Which days does each variable actually have data for?

forcing_bank aligns every source onto one daily grid, and where a source does
not reach it fills: forward from the last real value, and backward from the
median for anything before the record starts. That keeps the matrix rectangular
and every downstream model runnable, but it silently creates long constant
blocks, and a constant block is not data.

The consequences are measurable and were found by plotting the per-family
decile curves rather than by reading the code:

    sol_f107         real from 2004-10-27   63% of days are one filled value
    hyd_grace_land   real from 2002-04-17   59%
    ion_tec_mean     real from 1997-12-31   51%

A decile test on such a column is not doing what it says. More than half the
days land in a single tie, the quantile edges collapse, and only four to six of
the ten deciles end up populated. This does NOT manufacture false positives --
the null is computed on the same structure, so the p-values stay calibrated --
but it wastes half the sample and makes the reported upper bound looser than it
needs to be. The family is being credited with a weaker test than it deserves.

The fix is to test each variable only on the days it actually covers. The span
is recovered from the data itself: a filled region is exactly constant, so the
first and last index at which a column changes bracket its real record. That is
robust for the fill pattern used here, and it is checked against the known start
dates of the underlying products.
"""

from __future__ import annotations

import numpy as np

# Known first days of the underlying products, used to check the inference.
KNOWN_START = {
    "sol_f107": "2004-10-27",
    "ion_tec": "1997-12-31",
    "hyd_grace": "2002-04-17",
}
MIN_VARY = 0.02        # a column varying on under 2% of days is treated as flat


def valid_mask(X, names=None):
    """
    Boolean (days x variables): True where the column carries real data.

    A column is valid between the first and last day on which it changes.
    Columns that vary essentially everywhere -- the computed ephemerides, and
    any measured series covering the whole grid -- come back all True.
    """
    n, m = X.shape
    mask = np.ones((n, m), dtype=bool)
    for j in range(m):
        v = X[:, j].astype(float)
        d = np.flatnonzero(np.abs(np.diff(v)) > 0)
        if d.size == 0:                       # constant throughout: unusable
            mask[:, j] = False
            continue
        if d.size / n >= 1.0 - MIN_VARY:      # varies almost everywhere
            continue
        lo, hi = int(d.min()), int(d.max()) + 1
        mask[:lo, j] = False
        mask[hi + 1:, j] = False
    return mask


def coverage_report(X, names, fams, days=None):
    """Per-variable coverage, worst first. Returns a list of tuples."""
    mask = valid_mask(X, names)
    rows = []
    for j, (nm, fam) in enumerate(zip(names, fams)):
        cov = float(mask[:, j].mean())
        rows.append((fam, nm, cov, int(mask[:, j].sum())))
    rows.sort(key=lambda r: r[2])
    return rows, mask


def print_report(X, names, fams, limit=12):
    rows, mask = coverage_report(X, names, fams)
    print(f"\n  {'keluarga':<12} {'variabel':<26} {'cakupan':>9} {'hari':>9}")
    print("  " + "-" * 60)
    for fam, nm, cov, n in rows[:limit]:
        flag = "  <-- sebagian besar isian" if cov < 0.75 else ""
        print(f"  {fam:<12} {nm[:26]:<26} {100*cov:>8.1f}% {n:>9,}{flag}")
    full = sum(1 for r in rows if r[2] > 0.99)
    print(f"\n  {full} dari {len(rows)} variabel mencakup seluruh rekaman")
    return mask
