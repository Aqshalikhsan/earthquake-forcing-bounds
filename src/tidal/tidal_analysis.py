"""
Step 4: test the tidal hypothesis with the instrument the physics calls for.

The decisive comparison here is that BOTH tests run on THE SAME EVENTS:

    (a) SSGEOS's instrument -- lunar phase, a global scalar
    (b) tidal Coulomb stress at the site, resolved on the event's own fault

If (b) finds structure where (a) finds none, then the null results earlier in
this repository are not evidence that the Moon does nothing. They are evidence
that lunar phase is the wrong measurement.

PRE-REGISTERED (fixed before any result was printed):
  catalogue   Global CMT, Mw >= 5.5, centroid depth <= 70 km, 1976-2020,
              declustered with Gardner-Knopoff windows
  fault plane the shallower-dipping nodal plane (the fault, for thrusts);
              np1 and np2 reported as robustness checks
  primary     Schuster harmonic 1 on the tidal phase of dCFS
  subsets     by mechanism (thrust / normal / strike-slip) and by dCFS
              amplitude quartile -- declared in advance, not chosen after
  correction  Bonferroni over the primary plus 7 declared subsets (8 tests),
              alpha = 0.05 -> significant at p < 0.00625
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import argparse
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

from gcmt import parse_ndk
from ephem_vec import jd_from_datetime_array, bodies_equatorial
from tidal_stress import dcfs_series, tidal_phase, schuster
from honest_test import gardner_knopoff_window, haversine_vec

HERE = DATA
NDK = HERE / "gcmt.ndk"
WINDOW_DAYS = 1.0
STEP_MIN = 5.0
N_T = int(2 * WINDOW_DAYS * 24 * 60 / STEP_MIN) + 1     # 577
I_CENTRE = N_T // 2
N_DECLARED_TESTS = 8
BONF = 0.05 / N_DECLARED_TESTS


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def decluster_cmt(events):
    n = len(events)
    t0 = events[0].dt
    days = np.array([(e.dt - t0).total_seconds() / 86400.0 for e in events])
    mags = np.array([e.mw for e in events])
    lats = np.array([e.lat for e in events])
    lons = np.array([e.lon for e in events])

    removed = np.zeros(n, dtype=bool)
    for i in np.argsort(-mags):
        if removed[i]:
            continue
        dkm, ddays = gardner_knopoff_window(mags[i])
        delta = days - days[i]
        cand = (delta >= 0) & (delta <= ddays) & (mags <= mags[i]) & (~removed)
        cand[i] = False
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        d = haversine_vec(lats[i], lons[i], lats[idx], lons[idx])
        removed[idx[d <= dkm]] = True
    return [e for i, e in enumerate(events) if not removed[i]]


def compute_tidal(events, plane="shallow", chunk=250, ocean=None):
    """
    Tidal phase (deg) and dCFS amplitude (Pa) for every event.

    ocean : an OceanTide instance, or None for the body tide alone. With it,
    the direct water load above each submarine site is added to the stress.
    """
    n = len(events)
    phase = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    offsets = (np.arange(N_T) - I_CENTRE) * STEP_MIN

    for s in range(0, n, chunk):
        block = events[s:s + chunk]
        nb = len(block)
        times = []
        for e in block:
            times.extend(e.dt + timedelta(minutes=float(o)) for o in offsets)
        jd = jd_from_datetime_array(times).reshape(nb, N_T)

        lat = np.array([[e.lat] for e in block])
        lon = np.array([[e.lon] for e in block])
        geom = np.array([e.fault_plane(plane) for e in block], dtype=float)

        h = None
        if ocean is not None:
            coeffs = ocean.interp(lat[:, 0], lon[:, 0])
            h = ocean.height(coeffs, jd)

        d = dcfs_series(lat, lon, geom[:, 0], geom[:, 1], geom[:, 2], jd,
                        ocean_height=h)
        p, a = tidal_phase(d, I_CENTRE)
        phase[s:s + nb] = p
        amp[s:s + nb] = a
        print(f"    tidal stress: {min(s+chunk, n)}/{n} events", end="\r", flush=True)
    print(" " * 60, end="\r")
    return phase, amp


def lunar_phase_of(events):
    """Moon-Sun elongation at each origin time -- the SSGEOS instrument."""
    jd = jd_from_datetime_array([e.dt for e in events])
    eph = bodies_equatorial(jd)
    # elongation from equatorial coordinates
    ra_m, dec_m, _ = eph["moon"]
    ra_s, dec_s, _ = eph["sun"]
    r2 = np.pi / 180
    cos_e = (np.sin(dec_m * r2) * np.sin(dec_s * r2)
             + np.cos(dec_m * r2) * np.cos(dec_s * r2) * np.cos((ra_m - ra_s) * r2))
    elong = np.degrees(np.arccos(np.clip(cos_e, -1, 1)))
    # sign it so the result runs 0-360 through the synodic month
    sign = np.sign(((ra_m - ra_s) % 360) - 180)
    return np.where(sign >= 0, elong, 360 - elong)


def report(label, angles, note=""):
    p, rbar, mu, n = schuster(angles)
    if n == 0:
        print(f"  {label:<34} {'n=0':>28}")
        return p
    verdict = ("SIGNIFICANT" if p < BONF else
               "marginal" if p < 0.05 else "no effect")
    print(f"  {label:<34} n={n:>5}  p={p:>8.4f}  R/n={rbar:.4f}  "
          f"mean={mu:6.1f}deg  {verdict}{note}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mw", type=float, default=5.5)
    ap.add_argument("--max-depth", type=float, default=70.0)
    args = ap.parse_args()

    rule("TIDAL COULOMB STRESS vs LUNAR PHASE, ON THE SAME EARTHQUAKES")
    print(f"""
  catalogue    Global CMT 1976-2020, Mw >= {args.min_mw}, depth <= {args.max_depth} km
  tidal model  solid-earth body tide, degree 2, Love numbers h2/l2
               OCEAN LOADING NOT MODELLED -- see the caveat at the end
  fault plane  shallower-dipping nodal plane
  significance Bonferroni over {N_DECLARED_TESTS} declared tests -> p < {BONF:.5f}
""")

    if not NDK.exists():
        sys.exit(f"missing {NDK}; download the GCMT ndk catalogue first")

    allev = parse_ndk(NDK)
    sel = [e for e in allev if e.mw >= args.min_mw and e.depth <= args.max_depth]
    print(f"  catalogue events              : {len(allev)}")
    print(f"  shallow, above threshold      : {len(sel)}")
    main_shocks = decluster_cmt(sel)
    print(f"  after declustering            : {len(main_shocks)} "
          f"({100*(1-len(main_shocks)/len(sel)):.1f}% removed)")

    phase, amp = compute_tidal(main_shocks)
    ok = np.isfinite(phase)
    print(f"  tidal phase resolved          : {int(ok.sum())}")
    print(f"  dCFS amplitude, median        : {np.nanmedian(amp):.0f} Pa "
          f"(p10 {np.nanpercentile(amp,10):.0f}, p90 {np.nanpercentile(amp,90):.0f})")

    lunar = lunar_phase_of(main_shocks)
    mech = np.array([e.mechanism for e in main_shocks])

    rule("THE SAME EVENTS, MEASURED TWO WAYS")
    print("\n  (a) SSGEOS's instrument -- lunar phase, identical worldwide at a given instant\n")
    report("lunar phase, harmonic 4 (SSGEOS)", (lunar[ok] * 4) % 360)
    report("lunar phase, harmonic 2 (tidal)", (lunar[ok] * 2) % 360)
    report("lunar phase, harmonic 1", lunar[ok])

    print("\n  (b) tidal Coulomb stress at the site, on the event's own fault\n")
    p_primary = report("tidal phase of dCFS  <-- PRIMARY", phase[ok])

    rule("DECLARED SUBSETS")
    print()
    for m in ("thrust", "normal", "strike-slip"):
        sub = ok & (mech == m)
        report(f"mechanism: {m}", phase[sub])

    print()
    q = np.nanpercentile(amp[ok], [25, 50, 75])
    labels = ["amplitude Q1 (weakest tide)", "amplitude Q2", "amplitude Q3",
              "amplitude Q4 (strongest tide)"]
    edges = [(-np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)]
    for lab, (lo, hi) in zip(labels, edges):
        sub = ok & (amp > lo) & (amp <= hi)
        report(lab, phase[sub], note=f"   [{lo if np.isfinite(lo) else 0:.0f}-"
                                     f"{hi if np.isfinite(hi) else 9999:.0f} Pa]")

    rule("ROBUSTNESS: does the answer depend on which nodal plane is the fault?")
    print()
    for plane in ("np1", "np2"):
        ph2, _ = compute_tidal(main_shocks, plane=plane)
        report(f"fault = {plane}", ph2[np.isfinite(ph2)])

    rule("PHASE DISTRIBUTION OF THE PRIMARY TEST")
    hist, edges_h = np.histogram(phase[ok], bins=12, range=(0, 360))
    expected = ok.sum() / 12
    print(f"\n  0 deg = tidal stress maximum. Expected per bin: {expected:.0f}\n")
    for i in range(12):
        dev = 100 * (hist[i] - expected) / expected
        bar = "#" * int(round(hist[i] / expected * 24))
        print(f"  {edges_h[i]:>3.0f}-{edges_h[i+1]:>3.0f}deg {hist[i]:>5} "
              f"{dev:>+6.1f}%  {bar}")

    rule("VERDICT")
    print(f"""
  primary test p = {p_primary:.4f}   threshold for significance p < {BONF:.5f}

  The comparison that matters is not this number on its own, but that lunar
  phase and tidal stress were computed for the SAME {int(ok.sum())} earthquakes.
  Lunar phase is a single global number: it cannot distinguish an event in Chile
  from one in Japan at the same instant, and it tracks only the fortnightly
  spring-neap envelope -- roughly a 20% modulation on top of a semidiurnal tide
  it cannot see at all. Tidal stress uses the site, the time of day, both bodies'
  declinations, and the orientation of the fault that actually broke.

  CAVEAT, stated plainly: this models the solid-earth body tide only. Ocean
  tidal loading is not included, and beneath the sea floor a 1 m ocean tide
  applies ~10 kPa directly -- an order of magnitude above the body-tide dCFS
  computed here, and the reason published positive results concentrate on
  submarine thrusts. A body-tide-only null for those events is expected and is
  NOT evidence against tidal triggering. The continental strike-slip subset is
  where this particular test carries the most weight.
""")


if __name__ == "__main__":
    main()
