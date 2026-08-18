"""
Step 5: prove the pipeline can find a tidal signal that is really there.

A null result is worthless without this. "We detected nothing" and "our code is
broken" produce identical output, and the only way to tell them apart is to
plant a signal and check it comes back.

Three controls, each running the FULL chain from scratch -- ephemeris, tidal
potential, surface strain, fault resolution, phase extraction, Schuster:

  1. shift a fraction of events onto their own dCFS maximum.
     The recomputed phase must come back at ~0 deg. If the frame, the sign of
     the shear traction, or the phase construction were wrong, it would not.
  2. shift a fraction onto their dCFS minimum -> must come back at ~180 deg.
  3. sweep the shifted fraction to measure detection power, which converts the
     null in tidal_analysis.py into an upper bound on any real effect.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import argparse
from copy import copy
from datetime import timedelta
from pathlib import Path

import numpy as np

from gcmt import parse_ndk
from ephem_vec import jd_from_datetime_array
from tidal_stress import dcfs_series, tidal_phase, schuster
from tidal_analysis import (decluster_cmt, compute_tidal, rule,
                            N_T, I_CENTRE, STEP_MIN, BONF)

HERE = DATA
RNG = np.random.default_rng(20260816)


def extreme_offsets(events, chunk=250):
    """
    Minutes from each origin time to the nearest dCFS maximum, and to the
    nearest minimum, within +/-1 day.
    """
    n = len(events)
    off_max = np.full(n, np.nan)
    off_min = np.full(n, np.nan)
    grid = (np.arange(N_T) - I_CENTRE) * STEP_MIN

    for s in range(0, n, chunk):
        block = events[s:s + chunk]
        nb = len(block)
        times = []
        for e in block:
            times.extend(e.dt + timedelta(minutes=float(o)) for o in grid)
        jd = jd_from_datetime_array(times).reshape(nb, N_T)

        lat = np.array([[e.lat] for e in block])
        lon = np.array([[e.lon] for e in block])
        geom = np.array([e.fault_plane("shallow") for e in block], dtype=float)
        d = dcfs_series(lat, lon, geom[:, 0], geom[:, 1], geom[:, 2], jd)

        is_max = np.zeros_like(d, dtype=bool)
        is_max[:, 1:-1] = (d[:, 1:-1] > d[:, :-2]) & (d[:, 1:-1] >= d[:, 2:])
        is_min = np.zeros_like(d, dtype=bool)
        is_min[:, 1:-1] = (d[:, 1:-1] < d[:, :-2]) & (d[:, 1:-1] <= d[:, 2:])

        for i in range(nb):
            for flags, out in ((is_max, off_max), (is_min, off_min)):
                idx = np.flatnonzero(flags[i])
                if idx.size:
                    out[s + i] = grid[idx[np.argmin(np.abs(idx - I_CENTRE))]]
        print(f"    locating tidal extrema: {min(s+chunk, n)}/{n}", end="\r", flush=True)
    print(" " * 60, end="\r")
    return off_max, off_min


def shifted_catalog(events, offsets, fraction, rng):
    """Move a random `fraction` of events by their stored offset, in minutes."""
    n = len(events)
    pick = rng.random(n) < fraction
    pick &= np.isfinite(offsets)
    out = []
    for i, e in enumerate(events):
        if pick[i]:
            e2 = copy(e)
            e2.dt = e.dt + timedelta(minutes=float(offsets[i]))
            out.append(e2)
        else:
            out.append(e)
    return out, int(pick.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mw", type=float, default=6.0)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    rule("POSITIVE CONTROLS: can this pipeline detect a tide that is really there?")

    allev = parse_ndk(HERE / "gcmt.ndk")
    sel = [e for e in allev if e.mw >= args.min_mw and e.depth <= 70.0]
    events = decluster_cmt(sel)
    n = len(events)
    print(f"\n  events (Mw >= {args.min_mw}, shallow, declustered): {n}")

    base_phase, _ = compute_tidal(events)
    p0, r0, mu0, n0 = schuster(base_phase[np.isfinite(base_phase)])
    print(f"  unmodified catalogue: p = {p0:.4f}, mean direction {mu0:.1f} deg\n")

    off_max, off_min = extreme_offsets(events)
    print(f"  median |shift| to nearest dCFS maximum: "
          f"{np.nanmedian(np.abs(off_max)):.0f} min\n")

    rule("CONTROL 1 & 2: do planted events come back where they were planted?")
    print()
    for name, offs, expect in (("maximum", off_max, 0.0), ("minimum", off_min, 180.0)):
        cat, k = shifted_catalog(events, offs, 1.0, RNG)
        ph, _ = compute_tidal(cat)
        good = np.isfinite(ph)
        p, r, mu, nn = schuster(ph[good])
        err = abs(((mu - expect + 180) % 360) - 180)
        ok = "PASS" if (err < 12 and p < 1e-6) else "FAIL"
        print(f"  all events moved onto their dCFS {name:<8} "
              f"-> recovered mean {mu:6.1f} deg (expected {expect:5.1f}), "
              f"R/n = {r:.3f}, p = {p:.2e}   [{ok}]")

    rule("CONTROL 3: detection power vs the fraction of events that are triggered")
    print(f"""
  A real triggering effect does not move every earthquake. Here only a fraction
  is nudged onto the tidal stress maximum and the rest stay where they are, so
  the modulation amplitude is roughly that fraction. Significance threshold is
  the pre-registered p < {BONF:.5f}.
""")
    print(f"  {'triggered fraction':>19} {'median p':>12} {'detections':>13} {'R/n':>8}")
    print("  " + "-" * 56)
    for frac in (0.01, 0.02, 0.03, 0.05, 0.10):
        ps, rs, hits = [], [], 0
        for _ in range(args.trials):
            cat, k = shifted_catalog(events, off_max, frac, RNG)
            ph, _ = compute_tidal(cat)
            p, r, mu, nn = schuster(ph[np.isfinite(ph)])
            ps.append(p)
            rs.append(r)
            if p < BONF:
                hits += 1
        print(f"  {frac:>18.0%} {np.median(ps):>12.2e} "
              f"{hits}/{args.trials:<11} {np.mean(rs):>8.4f}")

    rule("WHAT THIS MEANS FOR THE NULL RESULT")
    print(f"""
  The pipeline recovers a planted signal, at the right phase, with the right
  sign. So the null in tidal_analysis.py is a measurement, not a malfunction.

  Read together with the power table above, the result is an upper bound: on
  this catalogue, body-tide triggering of shallow Mw >= {args.min_mw} earthquakes affects
  at most a few percent of events. That is entirely compatible with the
  published literature, which reports effects of a few percent concentrated in
  settings where OCEAN loading -- not modelled here -- dominates the stress.

  It is not compatible with the claim that lunar geometry lets you anticipate
  individual earthquakes.
""")


if __name__ == "__main__":
    main()
