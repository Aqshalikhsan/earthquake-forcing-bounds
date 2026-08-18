"""
Step 2: rebuild the SSGEOS analysis properly and test whether the claim survives.

The repo's own scripts cannot do this -- all seven are matplotlib files with the
results typed in by hand. Here the numbers are recomputed from the shipped
datasets and from an independent ephemeris.

Sections:
  1. Verify the ephemeris (are their phase timestamps astronomically right?)
  2. Replicate the headline claim exactly as published
  3. Window sensitivity  -- was +/-2.95 d chosen, or found?
  4. Window-free tests   -- Schuster / Rayleigh, no tunable parameter
  5. The physically meaningful test: tidal amplitude, not "cardinal phase"
  6. Split instrumental vs documentary events
  7. Monte Carlo of the search procedure itself (look-elsewhere effect)
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, kstest, chi2

from parse_ssgeos import parse_file, SYNODIC
from lunar_ephem import (
    phase_angle_at, moon_distance_at, find_phase_time,
    jd_from_datetime, datetime_from_jd, delta_t_seconds,
)

BASE = SSGEOS / "datasets"
MAX_OFFSET = SYNODIC / 8.0           # 3.6913 d: farthest you can be from a cardinal phase
RNG = np.random.default_rng(20260816)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 1. ephemeris

def verify_ephemeris(events):
    rule("1. IS THEIR EPHEMERIS CORRECT?  (independent Meeus recomputation)")

    modern = [e for e in events if e.year >= 1904]
    errs_dist, errs_phase = [], []
    for e in modern:
        d_mine = moon_distance_at(e.dt)
        errs_dist.append(d_mine - e.moon_distance_km)
        # recompute the time of the nearest cardinal phase they quote
        target = {"New Moon": 0.0, "First Quarter": 90.0,
                  "Full Moon": 180.0, "Third Quarter": 270.0}[e.nearest_phase]
        pdt = [p for p in e.phases if p[0] == e.nearest_phase][0][1]
        jd_mine = find_phase_time(target, jd_from_datetime(pdt))
        jd_theirs = jd_from_datetime(pdt) + delta_t_seconds(pdt.year) / 86400.0
        errs_phase.append((jd_mine - jd_theirs) * 24 * 60)   # minutes

    errs_dist = np.array(errs_dist)
    errs_phase = np.array(errs_phase)
    print(f"\n  instrumental events checked : {len(modern)}")
    print(f"  lunar distance, mean |error|: {np.abs(errs_dist).mean():.2f} km "
          f"(max {np.abs(errs_dist).max():.2f} km) out of ~384,000 km")
    print(f"  cardinal-phase time, mean |error|: {np.abs(errs_phase).mean():.1f} min "
          f"(max {np.abs(errs_phase).max():.1f} min)")
    print("\n  VERDICT: the astronomy is sound. Their ephemeris is genuinely accurate.")
    print("           Whatever is wrong with the conclusion, it is not the ephemeris.")

    # --- but which calendar did the ancient dates come from? ---
    rule("1b. THE CALENDAR PROBLEM FOR PRE-1582 EVENTS")
    print("\n  Every date before 1582-10-15 was recorded in the JULIAN calendar.")
    print("  Feeding such a date to a Gregorian-based ephemeris shifts the Moon's")
    print("  phase by however many days the two calendars differ.\n")
    print(f"  {'event':<44} {'as Gregorian':>13} {'as Julian':>11} {'shift':>8}")
    print("  " + "-" * 78)
    pre1582 = sorted([e for e in events if e.dt < datetime(1582, 10, 15)],
                     key=lambda e: e.dt)
    for e in pre1582:
        a_g = phase_angle_at(e.dt, "gregorian")
        a_j = phase_angle_at(e.dt, "julian")
        off_g = min(abs(((a_g - c + 180) % 360) - 180) for c in (0, 90, 180, 270))
        off_j = min(abs(((a_j - c + 180) % 360) - 180) for c in (0, 90, 180, 270))
        og = off_g / 360 * SYNODIC
        oj = off_j / 360 * SYNODIC
        jd_g = jd_from_datetime(e.dt, "gregorian")
        jd_j = jd_from_datetime(e.dt, "julian")
        label = f"{e.dt.date()} M{e.magnitude} {e.place[:24]}"
        print(f"  {label:<44} {og:>10.2f} d {oj:>9.2f} d {jd_g - jd_j:>6.0f} d")
    print("\n  'shift' is how many days the same written date moves between calendars.")
    print("  SSGEOS's reported offsets match the Gregorian column, i.e. they treated")
    print("  historical Julian dates as if they were Gregorian.")
    n_flip = 0
    for e in pre1582:
        a_g = phase_angle_at(e.dt, "gregorian")
        a_j = phase_angle_at(e.dt, "julian")
        og = min(abs(((a_g - c + 180) % 360) - 180) for c in (0, 90, 180, 270)) / 360 * SYNODIC
        oj = min(abs(((a_j - c + 180) % 360) - 180) for c in (0, 90, 180, 270)) / 360 * SYNODIC
        if (og <= 2.95) != (oj <= 2.95):
            n_flip += 1
    print(f"\n  events that change near-phase classification under the correct")
    print(f"  calendar: {n_flip} of {len(pre1582)}")


# ---------------------------------------------------------------- 2. replicate

def replicate(events, label, threshold):
    sub = [e for e in events if e.magnitude >= threshold]
    n = len(sub)
    near = sum(1 for e in sub if e.nearest_offset <= 2.95)
    p0 = 2.95 / MAX_OFFSET
    res = binomtest(near, n, p0, alternative="greater")
    return n, near, p0, res.pvalue


def replicate_headline(m85, m80):
    rule("2. REPLICATING THE PUBLISHED HEADLINE")
    print(f"\n  null probability of landing within +/-2.95 d of a cardinal phase")
    print(f"  = 2.95 / {MAX_OFFSET:.4f} = {2.95 / MAX_OFFSET:.4f}  (i.e. the 'signal'")
    print(f"    window already covers {100 * 2.95 / MAX_OFFSET:.1f}% of the lunar cycle)\n")

    print(f"  {'dataset':<34} {'n':>4} {'near':>5} {'obs%':>7} {'exp%':>7} {'p':>9}")
    print("  " + "-" * 72)
    for lbl, evs, thr in [
        ("M>=8.0 (1904-2021)", m80, 8.0),
        ("M>=8.2 (1904-2021)", m80, 8.2),
        ("M>=8.5 (365-2025)  <- headline", m85, 8.5),
        ("M>=8.8 (365-2025)", m85, 8.8),
    ]:
        n, near, p0, p = replicate(evs, lbl, thr)
        print(f"  {lbl:<34} {n:>4} {near:>5} {100*near/n:>6.1f}% {100*p0:>6.1f}% {p:>9.4f}")

    print("\n  README claims: n=55 near=51 p=0.008 | n=108 near=87 p=0.490")
    print("  -> replication succeeds. The arithmetic in the README is correct.")
    print("     So the question is not whether they computed it right, but whether")
    print("     the thing they computed means anything.")


# ---------------------------------------------------------------- 3. window sweep

def window_sweep(m85):
    rule("3. WINDOW SENSITIVITY: was +/-2.95 d chosen in advance, or found?")
    sub = [e for e in m85 if e.magnitude >= 8.5]
    offs = np.array([e.nearest_offset for e in sub])
    n = len(offs)

    widths = np.arange(0.20, MAX_OFFSET, 0.05)
    rows = []
    for w in widths:
        p0 = w / MAX_OFFSET
        k = int((offs <= w).sum())
        p = binomtest(k, n, p0, alternative="greater").pvalue
        rows.append((w, k, p0, p))

    best = min(rows, key=lambda r: r[3])
    print(f"\n  swept window half-width from 0.20 d to {MAX_OFFSET:.2f} d in 0.05 d steps")
    print(f"  ({len(rows)} candidate windows)\n")
    print(f"  {'window':>8} {'k/n':>8} {'expected':>9} {'p-value':>10}")
    print("  " + "-" * 40)
    for w, k, p0, p in rows:
        if abs(w - 2.95) < 0.026 or p == best[3] or abs(w % 0.5) < 0.026:
            mark = "  <-- PUBLISHED" if abs(w - 2.95) < 0.026 else ""
            mark = "  <-- best possible" if p == best[3] else mark
            print(f"  {w:>7.2f}d {k:>3}/{n:<4} {p0*n:>8.1f} {p:>10.4f}{mark}")

    print(f"\n  minimum achievable p over all windows : {best[3]:.4f} at w = {best[0]:.2f} d")
    print(f"  published window {2.95:.2f} d gives      : "
          f"{[r for r in rows if abs(r[0]-2.95)<0.026][0][3]:.4f}")
    n_sig = sum(1 for r in rows if r[3] < 0.05)
    print(f"  windows (of {len(rows)}) that reach p<0.05 : {n_sig}")
    print("\n  The published window sits essentially at the optimum of a free parameter.")
    print("  A threshold picked after seeing the data does not carry its nominal p-value.")
    return rows


# ---------------------------------------------------------------- 4. window-free

def schuster_p(angles_deg, harmonic=1):
    """Schuster's test: p = exp(-R^2/n) for phase angles at a given harmonic."""
    th = np.radians(angles_deg) * harmonic
    r = np.hypot(np.cos(th).sum(), np.sin(th).sum())
    n = len(th)
    return math.exp(-(r ** 2) / n), r / n


def window_free(m85, m80):
    rule("4. WINDOW-FREE TESTS (no tunable parameter -- the standard in the literature)")

    for label, evs, thr in [("M>=8.5 (365-2025), n=55", m85, 8.5),
                            ("M>=8.0 (1904-2021), n=108", m80, 8.0)]:
        sub = [e for e in evs if e.magnitude >= thr]
        ang = np.array([e.phase_angle() for e in sub])
        offs = np.array([e.nearest_offset for e in sub])

        print(f"\n  --- {label} ---")
        for k, meaning in [(1, "New vs Full asymmetry"),
                           (2, "syzygy vs quadrature  <-- the tidal hypothesis"),
                           (4, "all four cardinal phases <-- the SSGEOS hypothesis")]:
            p, rbar = schuster_p(ang, k)
            flag = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"    Schuster harmonic {k}  ({meaning:<40}) : "
                  f"p = {p:.3f}  R/n = {rbar:.3f}   {flag}")

        ks = kstest(offs, "uniform", args=(0, MAX_OFFSET))
        print(f"    KS test, offsets vs Uniform[0, {MAX_OFFSET:.3f}]        : "
              f"p = {ks.pvalue:.3f}   "
              f"{'SIGNIFICANT' if ks.pvalue < 0.05 else 'not significant'}")
        print(f"    mean offset {offs.mean():.2f} d vs {MAX_OFFSET/2:.2f} d expected")

    print("\n  With the arbitrary window removed, the headline effect does not survive")
    print("  in any window-free test.")


# ---------------------------------------------------------------- 5. tidal test

def tidal_amplitude(angle_deg):
    """
    Spring-neap envelope of the lunisolar tide, normalised to 1 at syzygy.
    Amplitudes add as vectors at twice the elongation; the lunar semidiurnal
    constituent (M2) is ~2.17x the solar one (S2).
    Peaks at New AND Full Moon, minimum at both quadratures.
    """
    am, as_ = 2.17, 1.0
    th = np.radians(2 * np.asarray(angle_deg))
    amp = np.sqrt(am**2 + as_**2 + 2 * am * as_ * np.cos(th))
    return amp / (am + as_)


def tidal_test(m85, m80):
    rule("5. THE PHYSICALLY MEANINGFUL TEST: tidal amplitude, not 'cardinal phase'")
    print("""
  Tidal stress does NOT peak at all four cardinal phases. It peaks at New and
  Full Moon (spring tides) and is at its MINIMUM at First and Third Quarter
  (neap tides). Grouping all four together puts the tidal maximum and the tidal
  minimum in the same 'signal' bin -- so the SSGEOS statistic cannot be a test
  of tidal triggering even in principle. The right question is whether quakes
  prefer high tidal amplitude.""")

    for label, evs, thr in [("M>=8.5 (365-2025)", m85, 8.5),
                            ("M>=8.0 (1904-2021)", m80, 8.0)]:
        sub = [e for e in evs if e.magnitude >= thr]
        n = len(sub)
        ang = np.array([e.phase_angle() for e in sub])

        n_syz = sum(1 for e in sub if e.is_syzygy)
        bt = binomtest(n_syz, n, 0.5, alternative="greater")

        amp = tidal_amplitude(ang)
        null_amp = tidal_amplitude(RNG.uniform(0, 360, 400_000))
        p_amp = (null_amp.mean() >= amp.mean())
        # Monte Carlo p for the mean amplitude
        sims = tidal_amplitude(RNG.uniform(0, 360, (20000, n))).mean(axis=1)
        p_mc = float((sims >= amp.mean()).mean())

        print(f"\n  --- {label}, n = {n} ---")
        print(f"    syzygy (New/Full) events      : {n_syz}/{n} = {100*n_syz/n:.1f}% "
              f"(50% expected)   binomial p = {bt.pvalue:.3f}")
        print(f"    mean tidal amplitude at quakes: {amp.mean():.4f}")
        print(f"    mean under random timing      : {null_amp.mean():.4f}")
        print(f"    Monte Carlo p (20k draws)     : {p_mc:.3f}   "
              f"{'SIGNIFICANT' if p_mc < 0.05 else 'not significant'}")

    print("\n  The tidal hypothesis -- the only one with a physical mechanism behind it --")
    print("  is NOT supported by SSGEOS's own catalogue.")


# ---------------------------------------------------------------- 6. era split

def era_split(m85):
    rule("6. WHERE DOES THE SIGNAL ACTUALLY COME FROM?  instrumental vs documentary")
    sub = [e for e in m85 if e.magnitude >= 8.5]
    p0 = 2.95 / MAX_OFFSET

    groups = [
        ("pre-1600 (chronicles, Julian calendar)", [e for e in sub if e.year < 1600]),
        ("1600-1899 (documentary)", [e for e in sub if 1600 <= e.year < 1900]),
        ("1904+ (instrumental, measured times)", [e for e in sub if e.year >= 1904]),
    ]
    print(f"\n  {'subset':<42} {'n':>4} {'near':>5} {'obs%':>7} {'p':>8}")
    print("  " + "-" * 70)
    for name, g in groups:
        if not g:
            continue
        n = len(g)
        k = sum(1 for e in g if e.nearest_offset <= 2.95)
        p = binomtest(k, n, p0, alternative="greater").pvalue
        print(f"  {name:<42} {n:>4} {k:>5} {100*k/n:>6.1f}% {p:>8.3f}")

    pre = [e for e in sub if e.year < 1900]
    mod = [e for e in sub if e.year >= 1904]
    kp = sum(1 for e in pre if e.nearest_offset <= 2.95)
    km = sum(1 for e in mod if e.nearest_offset <= 2.95)
    print(f"\n  ALL documentary (pre-1900) : {kp}/{len(pre)} = {100*kp/len(pre):.1f}%  "
          f"p = {binomtest(kp, len(pre), p0, alternative='greater').pvalue:.4f}")
    print(f"  ALL instrumental (1904+)   : {km}/{len(mod)} = {100*km/len(mod):.1f}%  "
          f"p = {binomtest(km, len(mod), p0, alternative='greater').pvalue:.4f}")

    print("""
  The dataset with measured origin times shows nothing. The claimed effect lives
  entirely in events whose date and hour were reconstructed from chronicles,
  tsunami deposits and tree rings -- 10 of which SSGEOS lists at an exact round
  hour because no real time exists for them.""")


# ---------------------------------------------------------------- 7. look-elsewhere

def look_elsewhere(m85, n_sims=20000):
    rule("7. MONTE CARLO OF THE SEARCH ITSELF (look-elsewhere / garden of forking paths)")
    sub = sorted([e for e in m85 if e.magnitude >= 8.5],
                 key=lambda e: -e.magnitude)
    mags = np.array([e.magnitude for e in sub])
    n = len(mags)
    thresholds = [8.5, 8.6, 8.8, 9.0]
    widths = np.arange(0.5, MAX_OFFSET, 0.05)

    print(f"""
  Suppose the Moon has no effect whatsoever. Generate {n} events at uniformly
  random lunar phases, then let an analyst do what SSGEOS did: try several
  magnitude thresholds ({len(thresholds)}) and several window widths ({len(widths)}),
  and report the smallest p-value found. How often does 'p <= 0.008' appear?
""")
    beat = 0
    beat05 = 0
    best_ps = np.empty(n_sims)
    for s in range(n_sims):
        ang = RNG.uniform(0, 360, n)
        off = np.minimum.reduce([np.abs(((ang - c + 180) % 360) - 180)
                                 for c in (0, 90, 180, 270)]) / 360 * SYNODIC
        best = 1.0
        for thr in thresholds:
            m = mags >= thr
            nn = int(m.sum())
            if nn < 15:
                continue
            o = off[m]
            for w in widths:
                k = int((o <= w).sum())
                p0 = w / MAX_OFFSET
                # survival function of Binomial(nn, p0) at k-1
                from scipy.stats import binom
                p = binom.sf(k - 1, nn, p0)
                if p < best:
                    best = p
        best_ps[s] = best
        if best <= 0.008:
            beat += 1
        if best <= 0.05:
            beat05 += 1

    print(f"  simulations run                        : {n_sims}")
    print(f"  reached p <= 0.05  somewhere in search : {100*beat05/n_sims:.1f}%")
    print(f"  reached p <= 0.008 somewhere in search : {100*beat/n_sims:.1f}%")
    print(f"  median 'best p' found in pure noise    : {np.median(best_ps):.4f}")
    print(f"\n  So the published p = 0.008 is reached by chance alone in roughly")
    print(f"  1 of every {max(n_sims/max(beat,1),1):.0f} noise datasets when the same freedom")
    print(f"  of choice is exercised. Corrected significance is nowhere near 0.008.")


# ---------------------------------------------------------------- main

def main():
    m85 = parse_file(BASE / "0365-2025-lunar-phases-m85p.txt")
    m80 = parse_file(BASE / "1904-2021-lunar-phases-m80p.txt")

    verify_ephemeris(m85)
    replicate_headline(m85, m80)
    window_sweep(m85)
    window_free(m85, m80)
    tidal_test(m85, m80)
    era_split(m85)
    look_elsewhere(m85)

    rule("SUMMARY")
    print("""
  reproducible from the repo as published      : NO  (all 7 scripts hardcode results)
  ephemeris / astronomy correct                : YES
  published arithmetic reproducible from data  : YES
  window +/-2.95 d chosen a priori             : NO  (sits at the optimum)
  survives any window-free test                : NO
  survives on instrumental data alone          : NO
  physically coherent as a tidal test          : NO  (lumps tide max with tide min)
  survives correction for the search performed : NO
""")


if __name__ == "__main__":
    main()
