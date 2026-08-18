"""
Do the eight forcings modulate AFTERSHOCK rate? A better-posed question.

Everything so far asked whether external forcing can say when a mainshock will
strike anywhere on Earth. The answer was no, family by family. This asks
something narrower and physically more promising: given that a mainshock has
just happened, do the forcings modulate the aftershocks that follow?

There is a real reason to expect more here. An aftershock zone is a patch of
crust that has just been pushed past failure and is left covered in small
faults sitting at the edge of slipping. A stress perturbation of a few hundred
pascals is negligible against the ~10 MPa needed to break intact rock, but it
is not obviously negligible against a fault that is already within a whisker of
going. Tidal modulation is reported more often in aftershock sequences, in
volcanic settings and in induced seismicity than in ordinary background
seismicity, and that pattern is what the critically-stressed-fault picture
predicts.

It is also the question with a use. Aftershock forecasting is already
operational at USGS and BMKG, so a forcing term that sharpened it would be
adopted immediately -- unlike a mainshock precursor, which would have to clear
a far higher bar before anyone acted on it.

    mainshocks   GCMT Mw >= 6.0, shallow, Gardner-Knopoff declustered
    aftershocks  Mw >= 5.0 shallow events inside a mainshock's space-time window
    statistic    standardised mean of each forcing at aftershock times,
                 max-statistic within family
    null         every mainshock time is shifted by ONE shared circular offset
                 and its aftershocks move with it, so time-since-mainshock is
                 preserved exactly and only the calendar alignment is destroyed

THE NULL IS THE WHOLE DESIGN. Aftershock rate collapses along Omori's law, from
many per day to nearly none within weeks. Any forcing that happens to drift on
a similar timescale would look like a trigger under a naive null. Shifting the
mainshocks carries the entire Omori structure along untouched, so the decay
cannot leak into the result.

Tidal phase is tested separately by the Schuster statistic, which is exactly
uniform under the null and is the standard instrument in this literature.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec
from forcing_bank import build_all

MAIN_MW, AFT_MW, MAXDEPTH = 6.0, 5.0, 70.0
N_SHIFT = 300
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    rule("APAKAH KEDELAPAN GAYA MEMODULASI GEMPA SUSULAN?")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk")) if e.depth <= MAXDEPTH]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev])
    lon = np.array([e.lon for e in ev])

    big = np.flatnonzero(mw >= MAIN_MW)
    small = np.flatnonzero(mw >= AFT_MW)
    print(f"""
  katalog     GCMT dangkal (<= {MAXDEPTH:.0f} km)
  induk       Mw >= {MAIN_MW}: {big.size:,}
  kandidat    Mw >= {AFT_MW}: {small.size:,}
""")

    # ---- assign aftershocks to mainshocks, largest mainshock first ----------
    taken = np.zeros(len(ev), dtype=bool)
    is_main = np.zeros(len(ev), dtype=bool)
    aft_day, aft_dt = [], []
    for i in big[np.argsort(-mw[big])]:
        if taken[i]:
            continue                       # already an aftershock of a bigger one
        is_main[i] = True
        dkm, ddays = gardner_knopoff_window(mw[i])
        d = ordn - ordn[i]
        cand = (d > 0) & (d <= ddays) & (mw >= AFT_MW) & (mw < mw[i]) & (~taken)
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            continue
        near = idx[haversine_vec(lat[i], lon[i], lat[idx], lon[idx]) <= dkm]
        taken[near] = True
        for j in near:
            aft_day.append(ordn[j])
            aft_dt.append(d[j])

    aft_day = np.array(aft_day)
    aft_dt = np.array(aft_dt)
    main_idx = np.flatnonzero(is_main)
    print(f"  induk terpakai      : {main_idx.size:,}")
    print(f"  gempa susulan       : {aft_day.size:,}")
    print(f"  jeda induk-susulan  : median {np.median(aft_dt):.0f} hari, "
          f"maks {aft_dt.max():.0f}")
    if aft_day.size < 500:
        print("\n  terlalu sedikit untuk diuji.")
        return

    # ---- forcing bank on the daily grid ------------------------------------
    days = np.load(datafile("m6_days.npy"))
    print("\n  membangun bank variabel:")
    X, names, fams = build_all(days)
    d0, d1 = int(days[0]), int(days[-1])
    span = d1 - d0
    print(f"\n  {X.shape[1]} variabel, {len(set(fams))} keluarga")

    # standardise so every variable is on one scale
    Z = (X - np.nanmean(X, axis=0)) / np.where(np.nanstd(X, axis=0) > 0,
                                               np.nanstd(X, axis=0), 1)
    Z = np.nan_to_num(Z)

    def stat_for(offset):
        """Mean standardised forcing at aftershock times, shifted by offset."""
        pos = ((aft_day - d0 + offset) % span).astype(int)
        return Z[pos].mean(axis=0)

    obs = stat_for(0)
    null = np.empty((N_SHIFT, X.shape[1]))
    for i in range(N_SHIFT):
        null[i] = stat_for(int(RNG.integers(60, span - 60)))
        if (i + 1) % 50 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    rule("HASIL PER KELUARGA")
    print(f"""
  'z maks'   simpangan terbesar di keluarga itu, dibakukan oleh null
  'p'        sudah terkoreksi untuk seluruh variabel dalam keluarga
  Diuji pada {aft_day.size:,} gempa susulan dari {main_idx.size:,} gempa induk.
""")
    print(f"  {'keluarga':<12} {'n var':>6} {'z maks':>8} {'p':>8}  variabel terbaik")
    print("  " + "-" * 64)

    mu, sd = null.mean(axis=0), null.std(axis=0)
    sd = np.where(sd > 0, sd, 1)
    z_obs = np.abs(obs - mu) / sd
    z_null = np.abs(null - mu) / sd

    best_p = 1.0
    for fam in dict.fromkeys(fams.tolist()):
        sel = np.flatnonzero(fams == fam)
        m_obs = float(z_obs[sel].max())
        m_null = z_null[:, sel].max(axis=1)
        p = float((m_null >= m_obs).mean())
        best_p = min(best_p, p)
        top = names[int(sel[np.argmax(z_obs[sel])])]
        print(f"  {fam:<12} {sel.size:>6} {m_obs:>8.2f} {p:>8.3f}  {top[:24]}"
              f"{'  <-- LOLOS' if p < 0.05 else ''}")

    rule("KOREKSI GABUNGAN: seluruh keluarga sekaligus")
    m_obs_all = float(z_obs.max())
    m_null_all = z_null.max(axis=1)
    p_all = float((m_null_all >= m_obs_all).mean())
    print(f"""
    total variabel    : {X.shape[1]}
    terbaik data asli : z = {m_obs_all:.2f}  ({names[int(np.argmax(z_obs))]})
    terbaik pada null : rata-rata z = {m_null_all.mean():.2f}, maks {m_null_all.max():.2f}
    p TERKOREKSI      : {p_all:.3f}
""")

    rule("CATATAN")
    print(f"""
  Null menggeser waktu gempa INDUK dan membawa serta susulannya, sehingga
  peluruhan Omori terjaga persis dan hanya keselarasan kalender yang dirusak.
  Tanpa itu, gaya apa pun yang kebetulan meluruh pada skala waktu serupa akan
  tampak seperti pemicu.

  Resolusi: bank variabel ini harian dan global. Untuk empat keluarga medan
  (ATMOSPHERE, HYDRO, IONOSPHERE, TIDAL) itu berarti 98% variasi lokal dibuang
  -- lihat resolution_test.py. Jadi nol di keempatnya di sini BUKAN batas yang
  sah; yang sah hanya untuk keluarga yang memang global (LUNAR, PLANETARY,
  ROTATION, SOLAR).
""")


if __name__ == "__main__":
    main()
