"""
Atmospheric pressure loading per earthquake, 1970-2024.

Of every non-tectonic surface load, this is the one most often quoted as a scale
reference and least often tested. A passing weather system moves surface
pressure by tens of hectopascals, and 1 hPa is 100 Pa on the crust -- so a deep
low is thousands of pascals, several times the solid-earth body tide of the Moon
and within a factor of a few of ocean tidal loading.

It also has the property the celestial families lack: it is a FIELD. Each
earthquake is scored against the pressure directly above it, and the null shifts
time while holding place.

    data     NCEP/NCAR Reanalysis 1, surface pressure, 2.5 degree grid, daily,
             1970-2024 -- 20,089 days at the 939 cells containing earthquakes

ANOMALY, NOT RAW PRESSURE. Raw surface pressure is dominated by elevation: a
cell in the Andes reads 300 hPa lower than one at sea level, and the standard
deviation across all cells (92 hPa) is mostly topography, not weather. Each cell
is therefore differenced against its own 30-day running median, which removes
elevation and the seasonal cycle and leaves the synoptic variation that actually
loads and unloads the crust.

CAUSAL SAMPLING. Only days 1-3 BEFORE each earthquake are read. Pressure is
strongly autocorrelated day to day, so this loses almost nothing physically, and
it removes any possibility of the event influencing its own predictor -- the
error that produced a spurious p < 0.0001 in the GRACE test.

Pre-registered, mirroring the GRACE and TEC tests:
  H1 LEVEL     signed pressure anomaly above the epicentre
  H2 ABSLEVEL  |anomaly| -- loading OR unloading
  H3 RATE      rate of change, i.e. how fast the load is shifting
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import datafile, DATA, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime

from honest_test import gardner_knopoff_window, haversine_vec

N_SHIFT = 2000
LAG = (1, 3)              # days before the event
MEDIAN_WIN = 30           # per-cell running median: removes elevation + season
MINMAG, MAXDEPTH = 6.0, 70.0
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_pressure():
    arr = np.load(datafile("pressure_cells.npy"))
    days = np.load(datafile("pressure_days.npy"))
    cells = np.load(datafile("pressure_cellid.npy"))
    order = np.argsort(days)
    return days[order], arr[order], cells


def per_cell_anomaly(arr, win=MEDIAN_WIN):
    """Subtract each cell's own running median: kills elevation and season."""
    n = arr.shape[0]
    out = np.empty_like(arr)
    half = win // 2
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = arr[i] - np.median(arr[a:b], axis=0)
    return out


def load_quakes():
    out = []
    for r in csv.DictReader(io.StringIO(datafile("global_m6.csv").read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"]); lo = float(r["longitude"])
            dep = float(r["depth"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if m < MINMAG or dep > MAXDEPTH:
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


def column_of(lats, lons, cells):
    """Map each event to its column in the stored cell array."""
    ilat = np.clip(np.round((90.0 - lats) / 2.5).astype(int), 0, 72)
    ilon = np.round((lons % 360.0) / 2.5).astype(int) % 144
    flat = ilat * 144 + ilon
    pos = {int(c): j for j, c in enumerate(cells)}
    return np.array([pos.get(int(f), -1) for f in flat])


def sample(days_p, anom, col, when):
    """Anomaly statistics over days LAG[0]..LAG[1] before each time."""
    idx = {int(d): i for i, d in enumerate(days_p)}
    lo, hi = LAG
    n = len(when)
    lvl = np.full(n, np.nan)
    absl = np.full(n, np.nan)
    rate = np.full(n, np.nan)
    for k, t in enumerate(when):
        c = col[k]
        if c < 0:
            continue
        base = t.toordinal()
        vals = [anom[idx[base - lag], c] for lag in range(lo, hi + 1)
                if (base - lag) in idx]
        if len(vals) >= 2:
            v = np.array(vals, dtype=float)
            if np.all(np.isfinite(v)):
                lvl[k] = v.mean()
                absl[k] = np.abs(v).mean()
                rate[k] = np.abs(np.diff(v)).mean()
    return lvl, absl, rate


def main():
    rule("BEBAN TEKANAN ATMOSFER — PER GEMPA, DI ATAS EPISENTRUMNYA")

    days_p, arr, cells = load_pressure()
    print(f"""
  NCEP/NCAR  {len(days_p):,} hari, {datetime.fromordinal(int(days_p[0])).date()} .. """
          f"""{datetime.fromordinal(int(days_p[-1])).date()}
  sel        {len(cells):,} (hanya yang berisi gempa)
  mentah     median {np.nanmedian(arr)/100:.0f} hPa, sd {np.nanstd(arr)/100:.1f} hPa
             <- sd ini didominasi KETINGGIAN, bukan cuaca
""")
    anom = per_cell_anomaly(arr)
    sd_pa = np.nanstd(anom)
    print(f"  anomali per sel (median berjalan {MEDIAN_WIN} hari):")
    print(f"    sd {sd_pa/100:.2f} hPa = {sd_pa:.0f} Pa   <- ini beban cuaca sesungguhnya")
    print(f"    bandingkan: body tide Bulan 723 Pa, pasang surut laut ~10.000 Pa")

    ev = decluster(load_quakes())
    ev = [e for e in ev if days_p[0] + MEDIAN_WIN <= e[0].toordinal() <= days_p[-1]]
    dts = [e[0] for e in ev]
    lats = np.array([e[2] for e in ev])
    lons = np.array([e[3] for e in ev])
    col = column_of(lats, lons, cells)
    print(f"\n  gempa M>=6,0 dangkal, declustered, 1970-2024: {len(ev):,}")

    lvl, absl, rate = sample(days_p, anom, col, dts)
    ok = np.isfinite(lvl)
    print(f"  tersampel : {int(ok.sum()):,} ({100*ok.mean():.1f}%)")
    print(f"  jendela   : {LAG[0]}-{LAG[1]} hari SEBELUM gempa (kausal)\n")

    stats = {"H1_level_mean": np.nanmean(lvl[ok]),
             "H2_absLevel_mean": np.nanmean(absl[ok]),
             "H3_rate_mean": np.nanmean(rate[np.isfinite(rate)])}
    null = {k: np.empty(N_SHIFT) for k in stats}

    span = dts[-1].toordinal() - dts[0].toordinal()
    base = np.array([d.toordinal() for d in dts], dtype=float)
    for i in range(N_SHIFT):
        off = RNG.integers(30, max(span - 30, 60))
        sh = [datetime.fromordinal(int(v)) for v in
              (dts[0].toordinal() + (base - base[0] + off) % span)]
        l2, a2, r2 = sample(days_p, anom, col, sh)
        m2 = np.isfinite(l2); rm = np.isfinite(r2)
        null["H1_level_mean"][i] = np.nanmean(l2[m2]) if m2.any() else np.nan
        null["H2_absLevel_mean"][i] = np.nanmean(a2[m2]) if m2.any() else np.nan
        null["H3_rate_mean"][i] = np.nanmean(r2[rm]) if rm.any() else np.nan
        if (i + 1) % 400 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    rule("HASIL")
    alpha = 0.05 / len(stats)
    print(f"\n  {'statistik':<20} {'terukur':>12} {'null rata2':>12} "
          f"{'null sd':>10} {'p':>9}")
    print("  " + "-" * 68)
    hit = False
    for k, v in stats.items():
        nb = null[k][np.isfinite(null[k])]
        if nb.size < 50 or not np.isfinite(v):
            continue
        p = float((np.abs(nb - nb.mean()) >= abs(v - nb.mean())).mean())
        hit |= p < alpha
        print(f"  {k:<20} {v:>9.1f} Pa {nb.mean():>12.1f} {nb.std():>10.1f} "
              f"{p:>9.4f}{'  <-- LOLOS' if p < alpha else ''}")
    print(f"\n  Bonferroni atas {len(stats)} statistik -> ambang p < {alpha:.4f}")

    rule("VONIS")
    print(f"""
  {'ADA hubungan tekanan-gempa yang lolos koreksi.' if hit
    else 'TIDAK ADA yang lolos koreksi.'}

  Ini keluarga dengan beban terukur terbesar setelah pasang surut laut:
  sd {sd_pa:.0f} Pa di lokasi gempa, sekitar {sd_pa/723:.1f}x body tide Bulan, pada
  {len(ev):,} gempa selama 55 tahun -- rentang dua kali lebih panjang dari GRACE.

  Kalau {sd_pa:.0f} Pa pun tidak menghasilkan efek terdeteksi, itu batas yang
  berguna: ia langsung menutup semua kandidat yang lebih lemah, termasuk seluruh
  keluarga langit yang bebannya di bawah 1.000 Pa.
""")


if __name__ == "__main__":
    main()
