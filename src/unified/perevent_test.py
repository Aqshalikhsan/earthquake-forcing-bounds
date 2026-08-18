"""
The four field families, tested at the resolution they actually have.

resolution_test.py measured what the unified bank throws away: the pressure
anomaly above an epicentre correlates with the global daily mean at r = 0.13,
so the bank variable carries 1.8% of the local variation. In the injection
test, a per-event design found a 5% local modulation every time while the
global-daily design needed 40-80% to find it half the time -- a gap of roughly
8-16x in effect size.

So the null results for ATMOSPHERE, HYDRO, IONOSPHERE and TIDAL in
unified_test.py are not bounds on those hypotheses. They are bounds on what a
badly-resolved version of those hypotheses could show. This file replaces them.

Each family is read AT EACH EARTHQUAKE'S OWN LOCATION:

    ATMOSPHERE   NCEP surface pressure anomaly above the epicentre
    HYDRO        GRACE water load and ocean bottom pressure at its cell
    IONOSPHERE   TEC anomaly above it
    TIDAL        Coulomb stress on ITS OWN fault plane, from its CMT solution

Global CMT is the master catalogue: it is the only one carrying focal
mechanisms, which TIDAL needs, and it gives every family the same events.

    catalogue   GCMT, Mw >= 6.0, depth <= 70 km, Gardner-Knopoff declustered
    sampling    CAUSAL ONLY -- days strictly before the origin time. This is
                the discipline that turned GRACE from p < 0.0001 into p = 0.36
                and it is enforced here for every family.
    null        circular shift of the event TIMES, holding LOCATIONS fixed. The
                same offset is applied to every family on each replicate, so
                any real correlation between families survives into the null.
    correction  studentised max-statistic within each family, then min-p across
                families. Every replicate is searched exactly as hard as the
                real data, so the reported p already pays for all the variables.

Families cover different spans -- pressure 1970-2024, GRACE 2002-2024, TEC
2015-2019 -- so each is tested on the events it can actually see, and the event
count is reported per family rather than being cut down to the shortest one.

TIDAL enters through the Schuster test on tidal phase, whose p-value is exactly
uniform under the null, so its null replicates are drawn as uniforms rather
than recomputed -- recomputing the body tide for 300 shifts would cost hours
and buy nothing.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime, timedelta

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec

N_SHIFT = 300
JITTER = 15               # days of jitter around the whole-year offset
MINMW, MAXDEPTH = 6.0, 70.0
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def running_mean(a, win):
    n = a.shape[0]
    half = win // 2
    pad = np.concatenate([np.repeat(a[:1], half, 0), a,
                          np.repeat(a[-1:], win - half, 0)], axis=0)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(pad, axis=0)])
    return a - (cs[win:win + n] - cs[:n]) / win


def decluster_cmt(ev):
    n = len(ev)
    t0 = ev[0].dt
    days = np.array([(e.dt - t0).total_seconds() / 86400.0 for e in ev])
    mw = np.array([e.mw for e in ev])
    la = np.array([e.lat for e in ev])
    lo = np.array([e.lon for e in ev])
    dead = np.zeros(n, dtype=bool)
    for i in np.argsort(-mw):
        if dead[i]:
            continue
        dkm, ddays = gardner_knopoff_window(mw[i])
        d = days - days[i]
        cand = (d >= 0) & (d <= ddays) & (mw <= mw[i]) & (~dead)
        cand[i] = False
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        dead[idx[haversine_vec(la[i], lo[i], la[idx], lo[idx]) <= dkm]] = True
    return [e for i, e in enumerate(ev) if not dead[i]]


# --------------------------------------------------------------- FAMILY: grid
class GridField:
    """A spatial field sampled causally at each event's own cell."""

    def __init__(self, name, days, arr, ilat, ilon, lag, unit, min_vals=2):
        self.name, self.unit = name, unit
        self.days, self.arr = days, arr
        self.ilat, self.ilon = ilat, ilon
        self.lag, self.min_vals = lag, min_vals
        self.index = {int(d): i for i, d in enumerate(days)}
        self.vars = [f"{name}_level", f"{name}_abs", f"{name}_rate"]

    def sample(self, ords):
        lo, hi = self.lag
        n = len(ords)
        out = np.full((n, 3), np.nan)
        for k in range(n):
            if self.ilat[k] < 0:
                continue
            vals = []
            for g in range(lo, hi + 1):
                i = self.index.get(int(ords[k]) - g)
                if i is None:
                    continue
                v = self.arr[i, self.ilat[k], self.ilon[k]]
                if np.isfinite(v):
                    vals.append(float(v))
            if len(vals) >= self.min_vals:
                v = np.array(vals)
                out[k] = (v.mean(), np.abs(v).mean(),
                          np.abs(np.diff(v)).mean() if v.size > 1 else np.nan)
        return out

    def stats(self, ords):
        s = self.sample(ords)
        with np.errstate(invalid="ignore"):
            return np.array([np.nanmean(s[:, j]) if np.isfinite(s[:, j]).sum() > 30
                             else np.nan for j in range(3)])

    def coverage(self, ords):
        return int(np.isfinite(self.sample(ords)[:, 0]).sum())


def build_atmosphere(lats, lons):
    arr = np.load(datafile("pressure_cells.npy")).astype(np.float32)
    days = np.load(datafile("pressure_days.npy"))
    cells = np.load(datafile("pressure_cellid.npy"))
    o = np.argsort(days); days, arr = days[o], arr[o]
    anom = running_mean(arr, 30)
    # stored as (day, cell); present it as (day, cell, 1) so one indexer works
    ilat = np.full(len(lats), -1)
    pos = {int(c): j for j, c in enumerate(cells)}
    il = np.clip(np.round((90.0 - lats) / 2.5).astype(int), 0, 72)
    ilo = np.round((lons % 360.0) / 2.5).astype(int) % 144
    for k, f in enumerate(il * 144 + ilo):
        ilat[k] = pos.get(int(f), -1)
    return GridField("atm", days, anom[:, :, None], ilat,
                     np.zeros(len(lats), dtype=int), (1, 3), "Pa")


def build_ionosphere(lats, lons):
    grid = np.load(datafile("tec_daily.npy"))
    days = np.load(datafile("tec_days.npy"))
    anom = running_mean(grid, 27)
    glat = np.linspace(87.5, -87.5, grid.shape[1])
    glon = np.linspace(-180.0, 180.0, grid.shape[2])
    ilat = np.abs(glat[None, :] - lats[:, None]).argmin(axis=1)
    lo = ((lons + 180.0) % 360.0) - 180.0
    ilon = np.abs(glon[None, :] - lo[:, None]).argmin(axis=1)
    return GridField("ion", days, anom, ilat, ilon, (1, 5), "TECU", min_vals=3)


class MergedField:
    """
    Two grids covering complementary ground, read as one family.

    GRACE land water alone reaches only 15% of M>=6 epicentres, because most of
    them are offshore. Ocean bottom pressure covers the rest, and both are
    converted to Pa so the merged variable is one physical quantity: the water
    load above the hypocentre.
    """

    def __init__(self, name, primary, fallback):
        self.name = name
        self.a, self.b = primary, fallback
        self.vars = primary.vars

    def sample(self, ords):
        sa = self.a.sample(ords)
        sb = self.b.sample(ords)
        miss = ~np.isfinite(sa[:, 0])
        sa[miss] = sb[miss]
        return sa

    stats = GridField.stats
    coverage = GridField.coverage


def build_hydro(lats, lons):
    from grace_hydro2 import load_grid, cell_index

    def grid(fname, prefer, scale):
        t, la, lo, arr, _, _ = load_grid(fname, prefer=prefer)
        d = np.array([x.toordinal() for x in t])
        ila, ilo = cell_index(la, lo, lats, lons)
        #
        # A GRACE solution is a MONTHLY average, and its timestamp is the middle
        # of the month it covers. A solution dated 10 days before an event
        # therefore contains data from about 25 days before to 5 days AFTER it,
        # so reading it would let the earthquake's own co-seismic gravity step
        # into its predictor. That is the exact error that turned an apparent
        # p < 0.0001 into p = 0.36 once it was caught. The window starts at 30
        # days so that the whole month a solution covers ends before the event.
        #
        return GridField("hyd", d, arr * scale, ila, ilo, (30, 60), "Pa",
                         min_vals=1)

    land = grid("grace_tws.nc", ("tws",), 98.07)       # cm air -> Pa
    try:
        ocean = grid("grace_obp.nc", ("resobp",), 100.0)   # hPa -> Pa
    except Exception:
        return land
    return MergedField("hyd", land, ocean)


# -------------------------------------------------------------- FAMILY: tidal
def tidal_schuster_p(ev):
    """Schuster test on tidal phase at each event's own fault. Uniform null."""
    from tidal_analysis import compute_tidal
    phase, amp = compute_tidal(ev, plane="shallow")
    ok = np.isfinite(phase)
    a = np.radians(phase[ok])
    n = a.size
    if n == 0:
        return np.nan, 0, np.nan
    R2 = np.sum(np.cos(a)) ** 2 + np.sum(np.sin(a)) ** 2
    p = float(np.exp(-R2 / n))
    return p, n, float(np.sqrt(R2) / n)


# ---------------------------------------------------------------------- main
def main():
    rule("UJI TERPADU PER-GEMPA: gaya dibaca DI LOKASI GEMPANYA SENDIRI")

    ev = parse_ndk(datafile("gcmt.ndk"))
    sel = [e for e in ev if e.mw >= MINMW and e.depth <= MAXDEPTH]
    ev = decluster_cmt(sel)
    lats = np.array([e.lat for e in ev]); lons = np.array([e.lon for e in ev])
    ords = np.array([e.dt.toordinal() for e in ev])
    span = int(ords.max() - ords.min())
    print(f"""
  katalog   Global CMT, Mw>={MINMW}, kedalaman<={MAXDEPTH} km
  mentah    {len(sel):,} -> declustered {len(ev):,}
  rentang   {datetime.fromordinal(int(ords.min())).date()} .. """
          f"""{datetime.fromordinal(int(ords.max())).date()}
  null      {N_SHIFT} pergeseran melingkar waktu, LOKASI tetap
""")

    fams = {}
    for label, builder in [("ATMOSPHERE", build_atmosphere),
                           ("IONOSPHERE", build_ionosphere),
                           ("HYDRO", build_hydro)]:
        try:
            fams[label] = builder(lats, lons)
            print(f"    {label:<12} siap  ({fams[label].coverage(ords):,} gempa tercakup)")
        except Exception as exc:
            print(f"    {label:<12} DILEWATI: {type(exc).__name__}: {exc}")

    # observed statistics
    obs = {k: f.stats(ords) for k, f in fams.items()}

    # null: one shared offset per replicate, applied to every family
    #
    # SEASON-PRESERVING SHIFTS. A free offset breaks the alignment between the
    # event calendar and the field's annual cycle, so any seasonal coincidence
    # -- winter storms in the pressure field, the annual hydrological cycle in
    # GRACE -- registers as signal. Offsets are therefore whole years, which
    # keeps every event in its own season and leaves only the year-to-year
    # variation to be tested. This is the difference between "earthquakes
    # happen in the season when water load swings fastest" (a calendar fact)
    # and "earthquakes happen when water load swings fastest" (the hypothesis).
    #
    # Whole-year offsets alone give only n_year distinct nulls -- over a 45-year
    # catalogue that is 43, so the finest resolvable p is 1/43 = 0.023 and
    # drawing 300 replicates from 43 values would report a precision the design
    # does not have. A jitter of +/-JITTER days is added: the season is still
    # held to within half a month, but the number of distinct nulls rises to
    # n_year * (2*JITTER+1), which is enough to resolve p properly.
    n_year = max(int(span // 365.25) - 1, 1)
    n_distinct = n_year * (2 * JITTER + 1)
    print(f"    null: {n_year} offset tahunan x +/-{JITTER} hari jitter "
          f"= {n_distinct:,} pergeseran berbeda (musim tetap)")
    null = {k: np.full((N_SHIFT, 3), np.nan) for k in fams}
    for i in range(N_SHIFT):
        off = int(round(365.25 * RNG.integers(1, n_year + 1))
                  + RNG.integers(-JITTER, JITTER + 1))
        sh = ords.min() + (ords - ords.min() + off) % span
        for k, f in fams.items():
            null[k][i] = f.stats(sh)
        if (i + 1) % 25 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 40, end="\r")

    # studentised max-statistic within family
    rule("HASIL PER KELUARGA  (dibaca di episentrum, bukan rerata global)")
    print(f"""
  'z maks'   simpangan terbesar di keluarga itu, dibakukan oleh null
  'p'        sudah terkoreksi untuk seluruh variabel dalam keluarga
""")
    print(f"  {'keluarga':<12} {'n gempa':>8} {'variabel terbaik':<14} "
          f"{'z maks':>8} {'p':>8}")
    print("  " + "-" * 60)

    pvals = {}
    for k, f in fams.items():
        nb = null[k]
        mu = np.nanmean(nb, axis=0)
        sd = np.nanstd(nb, axis=0)
        good = np.isfinite(sd) & (sd > 0) & np.isfinite(obs[k])
        if not good.any():
            print(f"  {k:<12} {'-':>8} {'(tak terukur)':<14}")
            continue
        z_obs = np.where(good, np.abs(obs[k] - mu) / np.where(sd > 0, sd, 1), 0)
        z_null = np.abs(nb - mu) / np.where(sd > 0, sd, 1)
        z_null[:, ~good] = 0
        m_obs = float(np.nanmax(z_obs))
        m_null = np.nanmax(z_null, axis=1)
        p = float((m_null >= m_obs).mean())
        pvals[k] = (p, m_null, m_obs)
        best = f.vars[int(np.nanargmax(z_obs))]
        print(f"  {k:<12} {f.coverage(ords):>8,} {best:<14} "
              f"{m_obs:>8.2f} {p:>8.3f}{'  <-- LOLOS' if p < 0.05 else ''}")

    # tidal, via Schuster
    try:
        p_t, n_t, rbar = tidal_schuster_p(ev)
        print(f"  {'TIDAL':<12} {n_t:>8,} {'phase (Schuster)':<14} "
              f"{'--':>8} {p_t:>8.3f}{'  <-- LOLOS' if p_t < 0.05 else ''}")
        print(f"  {'':<12} {'':>8} R/n = {rbar:.4f}")
    except Exception as exc:
        p_t = np.nan
        print(f"  {'TIDAL':<12} DILEWATI: {type(exc).__name__}: {exc}")

    # joint min-p across families
    rule("KOREKSI GABUNGAN: min-p lintas keluarga")
    if pvals:
        keys = list(pvals)
        obs_minp = min(pvals[k][0] for k in keys)
        if np.isfinite(p_t):
            obs_minp = min(obs_minp, p_t)
        # per replicate, p of each family via leave-one-out rank
        pn = np.empty((N_SHIFT, len(keys) + (1 if np.isfinite(p_t) else 0)))
        for j, k in enumerate(keys):
            _, m_null, _ = pvals[k]
            for i in range(N_SHIFT):
                pn[i, j] = (np.sum(m_null >= m_null[i]) - 1) / (N_SHIFT - 1)
        if np.isfinite(p_t):
            pn[:, -1] = RNG.random(N_SHIFT)      # Schuster p is uniform on H0
        null_minp = pn.min(axis=1)
        p_joint = float((null_minp <= obs_minp).mean())
        print(f"""
    keluarga diuji    : {len(keys) + (1 if np.isfinite(p_t) else 0)}
    min-p data asli   : {obs_minp:.4f}
    min-p pada null   : median {np.median(null_minp):.4f}, minimum {null_minp.min():.4f}
    p TERKOREKSI      : {p_joint:.3f}
""")

    rule("CATATAN")
    print("""
  Ini rancangan yang uji daya bilang sanggup menemukan modulasi lokal 5%.
  Rancangan global-harian di unified_test.py butuh 40-80% untuk hal yang sama,
  jadi angka p di sini menggantikan angka p keempat keluarga itu di sana --
  bukan menambahinya.

  Sampling kausal ditegakkan di semua keluarga: hanya hari SEBELUM waktu asal
  gempa yang dibaca. Tanpa itu, GRACE memberi p<0,0001 yang ternyata gravitasi
  ko-seismik, dan TEC akan menangkap gangguan ionosfer yang DISEBABKAN gempa.

  Yang masih menjadi keterbatasan: HYDRO bercadence bulanan, sehingga jendela
  kausalnya 10-40 hari sebelum gempa dan jauh lebih tumpul daripada
  ATMOSPHERE (1-3 hari) atau IONOSPHERE (1-5 hari). TEC hanya mencakup
  2015-2019, jadi keluarga itu diuji pada beberapa ratus gempa saja.
""")


if __name__ == "__main__":
    main()
