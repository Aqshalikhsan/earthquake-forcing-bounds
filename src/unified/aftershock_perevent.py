"""
Aftershocks, read at their own location. The last hypothesis left standing.

Two results set this up, and it sits exactly where they cross.

    aftershock_test.py   asked whether the eight forcings modulate aftershock
                         rate, and found nothing -- but it read every forcing as
                         one global number per day.
    resolution_test.py   measured what that costs: the pressure anomaly above an
                         epicentre correlates with the global daily mean at
                         r = 0.13, so a global variable carries under 2% of the
                         local variation. The four field families were being
                         tested nearly blind.

So the null in aftershock_test is a real bound only for the four genuinely
global families. For ATMOSPHERE, HYDRO, IONOSPHERE and TIDAL it decided nothing,
and this file finishes the job: each aftershock is scored against the forcing
directly above ITS OWN epicentre.

That combination -- a critically stressed fault patch, and a forcing resolved
where the fault actually is -- is the one place left in this project where a
positive is still physically plausible.

    mainshocks   GCMT Mw >= 6.0, shallow, Gardner-Knopoff
    aftershocks  Mw >= 5.0 shallow, inside a mainshock's space-time window,
                 each carrying its own latitude and longitude
    sampling     CAUSAL for the measured fields -- pressure 1-3 days before,
                 TEC 1-5 days before, GRACE 30-60 days before, so an
                 aftershock cannot contaminate its own predictor. The tide is
                 exempt because it is computed from ephemerides, not measured,
                 and cannot carry a seismic signature.
    null         one shared circular offset moves every MAINSHOCK time, and its
                 aftershocks travel with it. Omori decay and the spatial layout
                 are preserved exactly; only the calendar alignment breaks.

Nothing here overwrites the earlier tests. The field builders and the
aftershock assignment are imported from perevent_test.py and aftershock_test.py
so that both designs keep using one definition of each.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec
from perevent_test import build_atmosphere, build_ionosphere, build_hydro

MAIN_MW, AFT_MW, MAXDEPTH = 6.0, 5.0, 70.0
N_SHIFT = 300
JITTER = 15
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def collect_aftershocks():
    """Aftershocks with their own coordinates, tagged by parent mainshock."""
    ev = [e for e in parse_ndk(datafile("gcmt.ndk")) if e.depth <= MAXDEPTH]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev])
    lon = np.array([e.lon for e in ev])

    taken = np.zeros(len(ev), dtype=bool)
    is_main = np.zeros(len(ev), dtype=bool)
    rows = []                              # (index, parent index, days since)
    big = np.flatnonzero(mw >= MAIN_MW)
    for i in big[np.argsort(-mw[big])]:
        if taken[i]:
            continue
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
            rows.append((int(j), int(i), float(d[j])))
    return ev, np.array(rows, dtype=float), int(is_main.sum())


def main():
    rule("GEMPA SUSULAN, DIBACA DI LOKASINYA SENDIRI")

    ev, rows, n_main = collect_aftershocks()
    aidx = rows[:, 0].astype(int)
    pidx = rows[:, 1].astype(int)
    dts = rows[:, 2]

    lats = np.array([ev[i].lat for i in aidx])
    lons = np.array([ev[i].lon for i in aidx])
    aft_ord = np.array([ev[i].dt.toordinal() for i in aidx], dtype=float)
    par_ord = np.array([ev[i].dt.toordinal() for i in pidx], dtype=float)

    print(f"""
  induk       Mw >= {MAIN_MW} dangkal, Gardner-Knopoff : {n_main:,}
  susulan     Mw >= {AFT_MW} dangkal, di dalam jendela  : {len(aidx):,}
  jeda        median {np.median(dts):.0f} hari, maks {dts.max():.0f}
  rentang     {datetime.fromordinal(int(aft_ord.min())).date()} .. """
          f"""{datetime.fromordinal(int(aft_ord.max())).date()}
""")

    fams = {}
    for label, builder in [("ATMOSPHERE", build_atmosphere),
                           ("IONOSPHERE", build_ionosphere),
                           ("HYDRO", build_hydro)]:
        try:
            f = builder(lats, lons)
            fams[label] = f
            print(f"    {label:<12} siap  ({f.coverage(aft_ord):,} susulan tercakup)")
        except Exception as exc:
            print(f"    {label:<12} DILEWATI: {type(exc).__name__}: {exc}")

    obs = {k: f.stats(aft_ord) for k, f in fams.items()}

    # The offset moves the MAINSHOCK, and each aftershock keeps its own delay,
    # so Omori decay rides along untouched and only the calendar shifts.
    span = int(par_ord.max() - par_ord.min())
    n_year = max(int(span // 365.25) - 1, 1)
    print(f"\n    null: {n_year} offset tahunan x +/-{JITTER} hari "
          f"= {n_year*(2*JITTER+1):,} pergeseran (musim tetap)")

    null = {k: np.full((N_SHIFT, 3), np.nan) for k in fams}
    base = par_ord.min()
    for i in range(N_SHIFT):
        off = int(round(365.25 * RNG.integers(1, n_year + 1))
                  + RNG.integers(-JITTER, JITTER + 1))
        shifted_parent = base + (par_ord - base + off) % span
        sh = shifted_parent + dts                    # aftershock follows parent
        for k, f in fams.items():
            null[k][i] = f.stats(sh)
        if (i + 1) % 25 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 40, end="\r")

    rule("HASIL PER KELUARGA  (di episentrum susulan, bukan rerata global)")
    print(f"""
  Diuji pada {len(aidx):,} gempa susulan dari {n_main:,} gempa induk.
  'p' sudah terkoreksi untuk seluruh variabel dalam keluarga.
""")
    print(f"  {'keluarga':<12} {'n susulan':>10} {'variabel':<14} {'z maks':>8} {'p':>8}")
    print("  " + "-" * 60)

    pv = {}
    for k, f in fams.items():
        nb = null[k]
        mu, sd = np.nanmean(nb, axis=0), np.nanstd(nb, axis=0)
        good = np.isfinite(sd) & (sd > 0) & np.isfinite(obs[k])
        if not good.any():
            print(f"  {k:<12} {'(tak terukur)':>10}")
            continue
        z_o = np.where(good, np.abs(obs[k] - mu) / np.where(sd > 0, sd, 1), 0)
        z_n = np.abs(nb - mu) / np.where(sd > 0, sd, 1)
        z_n[:, ~good] = 0
        m_o = float(np.nanmax(z_o))
        m_n = np.nanmax(z_n, axis=1)
        p = float((m_n >= m_o).mean())
        pv[k] = (p, m_n, m_o)
        print(f"  {k:<12} {f.coverage(aft_ord):>10,} "
              f"{f.vars[int(np.nanargmax(z_o))]:<14} {m_o:>8.2f} {p:>8.3f}"
              f"{'  <-- LOLOS' if p < 0.05 else ''}")

    # ---- tidal phase on the aftershocks, Schuster (uniform null) -----------
    p_t = np.nan
    try:
        from tidal_analysis import compute_tidal
        sub = [ev[i] for i in aidx]
        phase, _ = compute_tidal(sub, plane="shallow")
        ok = np.isfinite(phase)
        a = np.radians(phase[ok]); n = a.size
        R2 = np.sum(np.cos(a)) ** 2 + np.sum(np.sin(a)) ** 2
        p_t = float(np.exp(-R2 / n))
        print(f"  {'TIDAL':<12} {n:>10,} {'phase(Schuster)':<14} {'--':>8} "
              f"{p_t:>8.3f}{'  <-- LOLOS' if p_t < 0.05 else ''}")
        print(f"  {'':<12} {'':>10} R/n = {np.sqrt(R2)/n:.4f}")
    except Exception as exc:
        print(f"  {'TIDAL':<12} DILEWATI: {type(exc).__name__}: {exc}")

    rule("KOREKSI GABUNGAN: min-p lintas keluarga")
    if pv:
        keys = list(pv)
        obs_min = min(pv[k][0] for k in keys)
        cols = len(keys) + (1 if np.isfinite(p_t) else 0)
        if np.isfinite(p_t):
            obs_min = min(obs_min, p_t)
        pn = np.empty((N_SHIFT, cols))
        for j, k in enumerate(keys):
            _, m_n, _ = pv[k]
            for i in range(N_SHIFT):
                pn[i, j] = (np.sum(m_n >= m_n[i]) - 1) / (N_SHIFT - 1)
        if np.isfinite(p_t):
            pn[:, -1] = RNG.random(N_SHIFT)
        nm = pn.min(axis=1)
        print(f"""
    keluarga diuji    : {cols}
    min-p data asli   : {obs_min:.4f}
    min-p pada null   : median {np.median(nm):.4f}, minimum {nm.min():.4f}
    p TERKOREKSI      : {float((nm <= obs_min).mean()):.3f}
""")

    rule("CATATAN")
    print("""
  Ini gabungan dua syarat yang selama ini tidak pernah dipenuhi bersamaan:
  sesar yang sudah kritis (zona susulan), dan gaya yang dibaca di tempat sesar
  itu berada. Kalau di sini pun nol, tidak tersisa rancangan yang lebih
  menguntungkan bagi hipotesis gaya luar dalam kerangka ini.

  Sampling kausal tetap ditegakkan untuk medan terukur. Pasang surut
  dikecualikan karena dihitung dari ephemeris, bukan diukur, sehingga tidak
  mungkin membawa jejak gempa -- itulah sebabnya uji Schuster memakai fase
  pada waktu kejadian, sebagaimana lazim di literatur pasang surut.
""")


if __name__ == "__main__":
    main()
