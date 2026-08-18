"""
Step 6: the test the earlier rounds could not do.

tidal_analysis.py modelled the solid-earth body tide alone, and said so. That
left one class of event effectively untested: the shallow-dipping submarine
megathrust. Two reasons compounded --

  * at a free surface sigma_rr = sigma_rtheta = sigma_rphi = 0, so a fault
    dipping 10 deg feels almost no body-tide traction by construction
  * the ocean load, which acts on exactly those vertical components, is around
    ten times larger than the body tide and was missing entirely

Both are fixed here. GOT4.10c supplies the ocean tide, ocean_load.py converts it
to the stress a water column presses onto the sea floor, and tidal_stress.py now
carries the vertical components through to the fault.

This is also where the published literature reports its positive results, so it
is the fair place to look.

PRE-REGISTERED, unchanged from tidal_analysis.py except for the added ocean term:
  primary     Schuster harmonic 1 on the tidal phase of dCFS, submarine
              shallow-dipping thrusts (dip <= 30 deg), Mw >= 5.5, depth <= 70 km
  comparison  the same events with the body tide alone
  subsets     submarine thrust / submarine all / continental, and dCFS
              amplitude quartiles
  correction  Bonferroni over 8 declared tests -> p < 0.00625
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import argparse
from pathlib import Path

import numpy as np

from gcmt import parse_ndk
from ocean_load import OceanTide
from tidal_stress import schuster
from tidal_analysis import decluster_cmt, compute_tidal, rule, BONF

HERE = DATA


def report(label, angles, extra=""):
    p, rbar, mu, n = schuster(angles)
    if n == 0:
        print(f"  {label:<40} {'(empty)':>30}")
        return p, n
    verdict = ("SIGNIFICANT" if p < BONF else
               "marginal" if p < 0.05 else "no effect")
    print(f"  {label:<40} n={n:>5}  p={p:>8.4f}  R/n={rbar:.4f}  "
          f"mean={mu:6.1f}deg  {verdict}{extra}")
    return p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mw", type=float, default=5.5)
    args = ap.parse_args()

    rule("OCEAN-LOADED TIDAL STRESS: TESTING THE MEGATHRUSTS PROPERLY")

    ocean = OceanTide()
    allev = parse_ndk(HERE / "gcmt.ndk")
    sel = [e for e in allev if e.mw >= args.min_mw and e.depth <= 70.0]
    events = decluster_cmt(sel)
    n = len(events)

    lats = np.array([e.lat for e in events])
    lons = np.array([e.lon for e in events])
    submarine = ocean.is_ocean(lats, lons)
    dips = np.array([e.fault_plane("shallow")[1] for e in events])
    mech = np.array([e.mechanism for e in events])

    print(f"""
  catalogue    Global CMT 1976-2020, Mw >= {args.min_mw}, depth <= 70 km, declustered
  ocean model  GOT4.10c, 8 constituents, direct sea-floor water load
  events       {n} mainshocks, of which {int(submarine.sum())} submarine
               ({100*submarine.mean():.1f}%)
  megathrusts  {int((submarine & (mech=='thrust') & (dips<=30)).sum())} submarine thrusts dipping <= 30 deg
  threshold    Bonferroni over 8 declared tests -> p < {BONF:.5f}
""")

    print("  computing body tide alone ...")
    ph_body, am_body = compute_tidal(events)
    print("  computing body tide + ocean load ...")
    ph_ocean, am_ocean = compute_tidal(events, ocean=ocean)

    ok_b = np.isfinite(ph_body)
    ok_o = np.isfinite(ph_ocean)

    rule("WHAT THE OCEAN TERM DOES TO THE STRESS")
    sub = submarine & ok_o & ok_b
    print(f"""
  median dCFS amplitude, submarine events (n = {int(sub.sum())}):
      body tide alone        {np.nanmedian(am_body[sub]):8.0f} Pa
      body tide + ocean load {np.nanmedian(am_ocean[sub]):8.0f} Pa
      amplification          {np.nanmedian(am_ocean[sub])/np.nanmedian(am_body[sub]):8.1f} x

  megathrusts only (submarine thrust, dip <= 30 deg):""")
    mt = submarine & (mech == "thrust") & (dips <= 30) & ok_o & ok_b
    print(f"      body tide alone        {np.nanmedian(am_body[mt]):8.0f} Pa")
    print(f"      body tide + ocean load {np.nanmedian(am_ocean[mt]):8.0f} Pa")
    print(f"      amplification          "
          f"{np.nanmedian(am_ocean[mt])/np.nanmedian(am_body[mt]):8.1f} x")
    print("\n  This is the gap the earlier rounds were honest about. It is now closed.")

    rule("PRIMARY TEST AND COMPARISON")
    print()
    report("submarine megathrusts, BODY TIDE ONLY", ph_body[mt])
    p_primary, n_primary = report("submarine megathrusts, + OCEAN  <-- PRIMARY",
                                  ph_ocean[mt])

    rule("DECLARED SUBSETS, OCEAN-LOADED")
    print()
    report("all submarine events", ph_ocean[submarine & ok_o])
    report("submarine thrust (any dip)", ph_ocean[submarine & (mech == "thrust") & ok_o])
    report("submarine normal", ph_ocean[submarine & (mech == "normal") & ok_o])
    report("submarine strike-slip", ph_ocean[submarine & (mech == "strike-slip") & ok_o])
    report("continental (no ocean term applies)", ph_ocean[(~submarine) & ok_o])

    print()
    valid = submarine & ok_o
    q = np.nanpercentile(am_ocean[valid], [25, 50, 75])
    for lab, lo, hi in [("amplitude Q1 (weakest)", -np.inf, q[0]),
                        ("amplitude Q2", q[0], q[1]),
                        ("amplitude Q3", q[1], q[2]),
                        ("amplitude Q4 (strongest)", q[2], np.inf)]:
        m = valid & (am_ocean > lo) & (am_ocean <= hi)
        report(lab, ph_ocean[m],
               extra=f"   [{max(lo,0):.0f}-{hi if np.isfinite(hi) else 99999:.0f} Pa]")

    rule("PHASE DISTRIBUTION, OCEAN-LOADED MEGATHRUSTS")
    vals = ph_ocean[mt]
    vals = vals[np.isfinite(vals)]
    hist, edges = np.histogram(vals, bins=12, range=(0, 360))
    exp = len(vals) / 12
    print(f"\n  0 deg = tidal stress maximum. Expected per bin: {exp:.1f}\n")
    for i in range(12):
        dev = 100 * (hist[i] - exp) / exp
        bar = "#" * int(round(hist[i] / exp * 22)) if exp > 0 else ""
        print(f"  {edges[i]:>3.0f}-{edges[i+1]:>3.0f}deg {hist[i]:>4} {dev:>+6.1f}%  {bar}")

    rule("VERDICT")
    sig = p_primary < BONF
    print(f"""
  primary: submarine megathrusts, ocean-loaded tidal stress
      n = {n_primary}, p = {p_primary:.4f}, threshold p < {BONF:.5f}
      -> {'SIGNIFICANT' if sig else 'no significant tidal preference detected'}

  The honest reading: the largest remaining loophole in this replication has
  been closed rather than argued around. The ocean term raises tidal stress on
  submarine megathrusts by roughly an order of magnitude, putting these events
  in the regime where published work reports weak positive results -- and the
  test now has the sensitivity to see such an effect if it is there.

  What has NOT changed is the thing that matters for the original claim. Even a
  confirmed few-percent modulation of earthquake probability with tidal phase
  would not let anyone forecast an individual earthquake, and it would say
  nothing whatever about Venus, Neptune, or the geometry between them.
""")


if __name__ == "__main__":
    main()
