"""
Ionospheric TEC as an earthquake precursor -- tested per event, at its own place.

TEC anomalies are among the most frequently claimed earthquake precursors, and
the LSTM paper cites that literature approvingly. The claims are usually of the
form "total electron content was anomalously high or low over the epicentre in
the days before the event".

That is testable, and it has the one property the celestial families lacked:
TEC is a FIELD, not a global scalar. Each earthquake can be scored against the
ionosphere directly above it, and the null can shift time while holding place.

    data      UPC-IonSAT UQRG global maps, 2.5 x 5 degrees, daily mean, 2015-2019
    anomaly   TEC minus a 27-day running median at the same cell -- one solar
              rotation, which removes the diurnal, seasonal and solar-cycle
              trends that dwarf any precursor
    window    the 1-5 days before each earthquake, per the usual claim

Pre-registered, mirroring the GRACE tests so the two are directly comparable:

  H1 LEVEL   |TEC anomaly| above the epicentre is elevated before an event
  H2 SIGNED  the anomaly is systematically positive or negative
  H3 RATE    the anomaly is changing rapidly

CAUSAL SAMPLING IS ENFORCED. Only days strictly before the earthquake are read.
This is the same discipline that overturned the GRACE result: there, using the
month spanning the event produced p < 0.0001, which collapsed to p = 0.36 once
only prior months were allowed. An ionospheric disturbance caused BY an
earthquake (or by its tsunami) is well documented, so a non-causal window would
find exactly that and call it a precursor.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime, timedelta

from honest_test import gardner_knopoff_window, haversine_vec

N_SHIFT = 2000
LAG_DAYS = (1, 5)          # "the days before": 1 to 5 days prior
MEDIAN_WIN = 27            # one solar rotation
RNG = np.random.default_rng(20260816)

LATS = np.arange(87.5, -88.0, -2.5)
LONS = np.arange(-180.0, 180.1, 5.0)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_tec():
    grid = np.load(datafile("tec_daily.npy"))
    days = np.load(datafile("tec_days.npy"))
    order = np.argsort(days)
    return days[order], grid[order]


def tec_anomaly(grid, win=MEDIAN_WIN):
    """TEC minus a centred running median per cell, in TECU."""
    n = grid.shape[0]
    out = np.empty_like(grid)
    half = win // 2
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = grid[i] - np.median(grid[a:b], axis=0)
    return out


def load_quakes(minmag=6.0, maxdepth=70.0):
    out = []
    for r in csv.DictReader(io.StringIO((datafile("global_m6.csv")).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"]); lo = float(r["longitude"])
            dep = float(r["depth"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if m < minmag or dep > maxdepth:
            continue
        out.append((dt, m, la, lo))
    out.sort(key=lambda e: e[0])
    return out


def decluster(rows):
    n = len(rows)
    days = np.array([(r[0] - rows[0][0]).total_seconds() / 86400.0 for r in rows])
    mags = np.array([r[1] for r in rows])
    lats = np.array([r[2] for r in rows])
    lons = np.array([r[3] for r in rows])
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
    return [r for i, r in enumerate(rows) if not removed[i]]


def sample(days_t, anom, ilat, ilon, when):
    """
    Anomaly statistics over the LAG_DAYS window strictly BEFORE each time.
    Returns (mean level, mean |level|, mean day-to-day change).
    """
    idx = {int(d): i for i, d in enumerate(days_t)}
    lo, hi = LAG_DAYS
    n = len(when)
    lvl = np.full(n, np.nan)
    absl = np.full(n, np.nan)
    rate = np.full(n, np.nan)
    for k, t in enumerate(when):
        base = t.toordinal()
        vals = []
        for lag in range(lo, hi + 1):
            i = idx.get(base - lag)
            if i is not None:
                v = anom[i, ilat[k], ilon[k]]
                if np.isfinite(v):
                    vals.append(v)
        if len(vals) >= 3:
            v = np.array(vals)
            lvl[k] = v.mean()
            absl[k] = np.abs(v).mean()
            rate[k] = np.abs(np.diff(v)).mean()
    return lvl, absl, rate


def cell_index(lats, lons):
    ilat = np.abs(LATS[None, :] - lats[:, None]).argmin(axis=1)
    lo = ((lons + 180.0) % 360.0) - 180.0
    ilon = np.abs(LONS[None, :] - lo[:, None]).argmin(axis=1)
    return ilat, ilon


def main():
    rule("ANOMALI TEC IONOSFER — DIUJI PER GEMPA DI ATAS EPISENTRUMNYA")

    if not (datafile("tec_daily.npy")).exists():
        print("\n  tec_daily.npy belum ada. Jalankan fetch_tec.py dulu.")
        return

    days_t, grid = load_tec()
    print(f"""
  TEC       {len(days_t)} hari, {datetime.fromordinal(int(days_t[0])).date()} .. """
          f"""{datetime.fromordinal(int(days_t[-1])).date()}
  grid      {grid.shape[1]} x {grid.shape[2]} (2,5 x 5 derajat)
  nilai     median {np.nanmedian(grid):.1f} TECU, maks {np.nanmax(grid):.1f}
""")
    anom = tec_anomaly(grid)
    print(f"  anomali (setelah median berjalan {MEDIAN_WIN} hari): "
          f"sd {np.nanstd(anom):.2f} TECU")

    ev = decluster(load_quakes())
    lo_d, hi_d = int(days_t[0]) + MEDIAN_WIN, int(days_t[-1])
    ev = [e for e in ev if lo_d <= e[0].toordinal() <= hi_d]
    dts = [e[0] for e in ev]
    lats = np.array([e[2] for e in ev])
    lons = np.array([e[3] for e in ev])
    ilat, ilon = cell_index(lats, lons)
    print(f"\n  gempa M>=6,0 dangkal, declustered, dalam rentang TEC: {len(ev):,}")

    lvl, absl, rate = sample(days_t, anom, ilat, ilon, dts)
    ok = np.isfinite(lvl)
    print(f"  tersampel : {int(ok.sum()):,} ({100*ok.mean():.1f}%)")
    print(f"  jendela   : {LAG_DAYS[0]}-{LAG_DAYS[1]} hari SEBELUM gempa (kausal)\n")

    stats = {"H2_level_mean": np.nanmean(lvl[ok]),
             "H1_absLevel_mean": np.nanmean(absl[ok]),
             "H3_rate_mean": np.nanmean(rate[np.isfinite(rate)])}
    null = {k: np.empty(N_SHIFT) for k in stats}

    span = dts[-1].toordinal() - dts[0].toordinal()
    base = np.array([d.toordinal() for d in dts], dtype=float)
    for i in range(N_SHIFT):
        off = RNG.integers(30, max(span - 30, 60))
        sh = [datetime.fromordinal(int(v)) for v in
              (dts[0].toordinal() + (base - base[0] + off) % span)]
        l2, a2, r2 = sample(days_t, anom, ilat, ilon, sh)
        m2 = np.isfinite(l2); r2m = np.isfinite(r2)
        null["H2_level_mean"][i] = np.nanmean(l2[m2]) if m2.any() else np.nan
        null["H1_absLevel_mean"][i] = np.nanmean(a2[m2]) if m2.any() else np.nan
        null["H3_rate_mean"][i] = np.nanmean(r2[r2m]) if r2m.any() else np.nan
        if (i + 1) % 400 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    rule("HASIL")
    alpha = 0.05 / len(stats)
    print(f"\n  {'statistik':<20} {'terukur':>10} {'null rata2':>12} "
          f"{'null sd':>10} {'p':>9}")
    print("  " + "-" * 66)
    hit = False
    for k, v in stats.items():
        nb = null[k][np.isfinite(null[k])]
        if nb.size < 50 or not np.isfinite(v):
            continue
        p = float((np.abs(nb - nb.mean()) >= abs(v - nb.mean())).mean())
        hit |= p < alpha
        print(f"  {k:<20} {v:>10.4f} {nb.mean():>12.4f} {nb.std():>10.4f} "
              f"{p:>9.4f}{'  <-- LOLOS' if p < alpha else ''}")
    print(f"\n  Bonferroni atas {len(stats)} statistik -> ambang p < {alpha:.4f}")

    rule("VONIS")
    print(f"""
  {'ADA anomali TEC prekursor yang lolos koreksi.' if hit
    else 'TIDAK ADA anomali TEC prekursor yang lolos koreksi.'}

  Sampling kausal ditegakkan: hanya hari {LAG_DAYS[0]}-{LAG_DAYS[1]} SEBELUM gempa yang dibaca.
  Gangguan ionosfer yang DISEBABKAN gempa (dan tsunaminya) terdokumentasi baik;
  jendela non-kausal akan menemukan itu lalu menyebutnya prekursor. Itu persis
  kesalahan yang membatalkan hasil GRACE: p < 0,0001 menjadi p = 0,36 setelah
  hanya bulan sebelumnya yang diizinkan.

  Confound yang tersisa dan perlu disebut di naskah: badai geomagnetik menaikkan
  TEC secara global, dan badai itu digerakkan aktivitas matahari -- yang sudah
  diukur nol di keluarga SOLAR (AUC 0,45-0,48). Median berjalan {MEDIAN_WIN} hari
  menghapus tren lambat, tapi tidak menghapus badai tunggal.
""")


if __name__ == "__main__":
    main()
