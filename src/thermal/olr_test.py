"""
Thermal precursor: does the ground radiate differently before an earthquake?

This is the last of the three precursor signals cited most often. Radon has no
global archive and cannot be tested at all; TEC was tested and came back null;
OLR is the remaining one, and it has the longest record of any field in this
project -- 48 years, twice the span of GRACE and nearly double TEC.

The claim is that in the days before a large earthquake the surface above it
emits differently, through some combination of warming, gas release and changes
in near-surface humidity. It is a claim about a SYMPTOM, not a force: OLR
pushes on nothing and does not appear on the stress ladder. That is what makes
causal sampling essential rather than tidy. An earthquake and its aftershocks
disturb the ground, dust and cloud above them, so a window that reaches even
one day past the origin time can find an effect the earthquake caused and
report it as one that preceded it. That is precisely the error that turned
GRACE from p < 0.0001 into p = 0.36.

    data       NOAA interpolated OLR, daily 2.5 degree grid, 1974-2022,
               reduced to the 1,900 cells that contain earthquakes
    anomaly    each cell against its own 30-day running mean, which removes the
               seasonal cycle and the enormous latitude gradient -- tropics
               radiate ~280 W/m2, poles ~180, and none of that is a precursor
    sampling   days 1-7 BEFORE the origin time, strictly
    null       whole-year circular offsets with +/-15 days of slack, so each
               event keeps its season and only the year-to-year alignment breaks

Two questions are asked, because they fail differently:

    MAINSHOCK  is the anomaly above an epicentre unusual before it ruptures?
    FORESHOCK  among small earthquakes, does the anomaly separate the ones a
               mainshock follows from the ones nothing follows?

The second is the sharper test. Both groups are earthquakes in seismically
active crust, so climate, terrain and cloud regime are present on both sides
and cancel; what is left is the question actually being claimed.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec
from foreshock_perevent import FastField
from foreshock_test import label_foreshocks, CAND_MW, TARGET_MW, RADIUS, WINDOW

MAXDEPTH = 70.0
MEDIAN_WIN = 30
LAG = (1, 7)
N_SHIFT = 300
JITTER = 15
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
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(
        np.nan_to_num(pad), axis=0)])
    return a - (cs[win:win + n] - cs[:n]) / win


class _Shim:
    """Minimal object with the attributes FastField expects."""

    def __init__(self, days, arr, ilat, ilon, lag, vars_):
        self.days, self.arr = days, arr
        self.ilat, self.ilon, self.lag, self.vars = ilat, ilon, lag, vars_


def build_olr(lats, lons):
    arr = np.load(datafile("olr_cells.npy")).astype(np.float32)
    days = np.load(datafile("olr_days.npy"))
    cells = np.load(datafile("olr_cellid.npy"))
    nlat, nlon = np.load(datafile("olr_gridshape.npy"))
    o = np.argsort(days); days, arr = days[o], arr[o]
    anom = running_mean(arr, MEDIAN_WIN)

    glat = np.linspace(90.0, -90.0, int(nlat))
    ilat_g = np.abs(glat[None, :] - lats[:, None]).argmin(axis=1)
    ilon_g = np.round((lons % 360.0) / (360.0 / int(nlon))).astype(int) % int(nlon)
    pos = {int(c): j for j, c in enumerate(cells)}
    col = np.array([pos.get(int(f), -1) for f in ilat_g * int(nlon) + ilon_g])
    return FastField(_Shim(days, anom[:, :, None], col,
                           np.zeros(lats.size, dtype=int), LAG,
                           ["olr_level", "olr_abs", "olr_rate"]))


def maxstat(obs, null):
    mu, sd = np.nanmean(null, axis=0), np.nanstd(null, axis=0)
    good = np.isfinite(sd) & (sd > 0) & np.isfinite(obs)
    if not good.any():
        return np.nan, np.nan, -1
    z_o = np.where(good, np.abs(obs - mu) / np.where(sd > 0, sd, 1), 0)
    z_n = np.abs(null - mu) / np.where(sd > 0, sd, 1)
    z_n[:, ~good] = 0
    m_o = float(np.nanmax(z_o))
    p = float((np.nanmax(z_n, axis=1) >= m_o).mean())
    return m_o, p, int(np.nanargmax(z_o))


def main():
    rule("ANOMALI TERMAL (OLR) — DIUJI DI ATAS EPISENTRUM")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= MAXDEPTH and e.mw >= CAND_MW]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lats = np.array([e.lat for e in ev]); lons = np.array([e.lon for e in ev])

    f = build_olr(lats, lons)
    s0 = f.sample(ordn)
    cov = np.isfinite(s0[:, 0])
    print(f"""
  OLR        {f.table.size:,} hari terindeks, anomali per sel """
          f"""(rerata berjalan {MEDIAN_WIN} hari)
  kandidat   Mw >= {CAND_MW} dangkal : {mw.size:,}
  tercakup   {int(cov.sum()):,} ({100*cov.mean():.1f}%)
  jendela    {LAG[0]}-{LAG[1]} hari SEBELUM gempa (kausal)
""")

    # mainshocks: declustered, so aftershocks are not counted twice
    big = np.flatnonzero(mw >= TARGET_MW)
    dead = np.zeros(big.size, dtype=bool)
    ob, mb, lb, nb_ = ordn[big], mw[big], lats[big], lons[big]
    for k in np.argsort(-mb):
        if dead[k]:
            continue
        dkm, ddays = gardner_knopoff_window(mb[k])
        d = ob - ob[k]
        c = (d > 0) & (d <= ddays) & (mb <= mb[k]) & (~dead)
        i = np.flatnonzero(c)
        if i.size:
            dead[i[haversine_vec(lb[k], nb_[k], lb[i], nb_[i]) <= dkm]] = True
    main_i = big[~dead]

    y = label_foreshocks(ordn, mw, lats, lons)
    a, b = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    print(f"  gempa induk (declustered) : {main_i.size:,}")
    print(f"  foreshock / bukan         : {a.size:,} / {b.size:,}")

    span = int(ordn.max() - ordn.min())
    n_year = max(int(span // 365.25) - 1, 1)
    base = ordn.min()
    offs = [int(round(365.25 * RNG.integers(1, n_year + 1))
                + RNG.integers(-JITTER, JITTER + 1)) for _ in range(N_SHIFT)]
    print(f"  null                      : {n_year} offset tahunan "
          f"x +/-{JITTER} hari\n")

    def stat_main(ords):
        s = f.sample(ords)
        with np.errstate(invalid="ignore"):
            return np.array([np.nanmean(s[main_i, j]) for j in range(3)])

    def stat_fore(ords):
        s = f.sample(ords)
        with np.errstate(invalid="ignore"):
            return np.array([np.nanmean(s[a, j]) - np.nanmean(s[b, j])
                             for j in range(3)])

    rule("HASIL")
    print(f"\n  {'pertanyaan':<26} {'n':>8} {'variabel':<12} {'z':>7} {'p':>8}")
    print("  " + "-" * 66)
    out = {}
    for label, fn, n in [("gempa induk", stat_main, main_i.size),
                         ("foreshock vs biasa", stat_fore, a.size)]:
        obs = fn(ordn)
        null = np.empty((N_SHIFT, 3))
        for i, off in enumerate(offs):
            null[i] = fn(base + (ordn - base + off) % span)
            if (i + 1) % 30 == 0:
                print(f"    {label} null {i+1}/{N_SHIFT}", end="\r", flush=True)
        print(" " * 46, end="\r")
        z, p, j = maxstat(obs, null)
        out[label] = p
        print(f"  {label:<26} {n:>8,} {f.vars[j]:<12} {z:>7.2f} {p:>8.3f}"
              f"{'  <-- LOLOS' if p < 0.025 else ''}")

    rule("VONIS")
    worst = min(out.values())
    print(f"""
  Bonferroni atas 2 pertanyaan -> ambang p < 0,025
  {'ADA yang lolos -- perlu uji stabilitas separuh katalog.' if worst < 0.025
   else 'TIDAK ADA yang lolos.'}

  Dengan ini ketiga sinyal prekursor yang paling sering dikutip sudah selesai:

      radon      tidak ada arsip global -- tidak bisa diuji sama sekali
      TEC        1998-2024, 1.907 gempa   p = 0,90 (per-gempa)
      termal     1974-2022, {int(cov.sum()):,} kandidat   p = {out['gempa induk']:.2f} / """
          f"""{out['foreshock vs biasa']:.2f}

  Catatan kategori: OLR bukan gaya. Ia gejala, dan gejala punya cara gagal yang
  tidak dimiliki gaya -- efek yang DISEBABKAN gempa terbaca sebagai pendahulu.
  Sampling {LAG[0]}-{LAG[1]} hari sebelum waktu asal adalah yang menutup jalan itu.
""")


if __name__ == "__main__":
    main()
