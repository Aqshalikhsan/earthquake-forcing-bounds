"""
Hydrological loading per earthquake -- land AND ocean, keeping M >= 6.

grace_hydro.py scored every earthquake against GRACE terrestrial water storage.
That worked, but only 289 of 1,869 events (15.5%) sat on land where the land
product has values. Most large earthquakes are offshore, on subduction zones,
and were simply discarded. A null from 289 events is a weak null.

The obvious fix -- lower the magnitude threshold -- was not wanted. So this uses
the other half of the same mission instead.

    TWS   terrestrial water storage, land       -> continental events
    OBP   ocean bottom pressure, sea floor      -> submarine events

Both come from the same GRACE/GRACE-FO processing at GFZ, on the same monthly
1-degree grid. Together they cover the whole planet, so no earthquake has to be
thrown away and the magnitude threshold stays where it was.

Physically they are the same thing seen in two places: a mass of water sitting
on the crust. One centimetre of equivalent water height is rho*g*h = 98 Pa of
surface load either way.

Pre-registered, unchanged from grace_hydro.py:

  H1 LOAD   earthquakes prefer times of anomalously high or low local load
  H2 RATE   earthquakes prefer times of rapid loading or unloading

Null: circular shift of the earthquake times with locations held fixed. Each
event is re-scored against the load its own region carried at a different time,
so the spatial pattern of seismicity and of water storage both survive intact;
only the timing link is destroyed.
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
PA_PER_CM = 1000.0 * 9.80665 * 0.01
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_grid(fname, prefer=("tws", "obp")):
    """Read a GravIS Level-3 grid, auto-detecting the payload variable."""
    import netCDF4
    d = netCDF4.Dataset(datafile(fname))
    name = None
    for p in prefer:
        if p in d.variables:
            name = p
            break
    if name is None:
        for k, v in d.variables.items():
            if v.ndim == 3:
                name = k
                break
    arr = np.array(d.variables[name][:], dtype=np.float32)
    lat = np.array(d.variables["lat"][:], dtype=float)
    lon = np.array(d.variables["lon"][:], dtype=float)
    units = getattr(d.variables[name], "units", "?")
    tv = np.array(d.variables["time"][:], dtype=float)
    d.close()
    epoch = datetime(2002, 4, 18)
    times = np.array([epoch + timedelta(days=float(v)) for v in tv])
    arr = np.where(np.abs(arr) > 1e4, np.nan, arr)
    return times, lat, lon, arr, name, units


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


def cell_index(lat_g, lon_g, lats, lons):
    ilat = np.abs(lat_g[None, :] - lats[:, None]).argmin(axis=1)
    ilon = np.abs((lon_g % 360.0)[None, :] - (lons % 360.0)[:, None]).argmin(axis=1)
    return ilat, ilon


def sample(times_g, arr, ilat, ilon, when):
    """Load and its monthly rate of change at each event's own cell."""
    tg = np.array([t.toordinal() for t in times_g], dtype=float)
    tq = np.array([t.toordinal() for t in when], dtype=float)
    j = np.clip(np.searchsorted(tg, tq) - 1, 0, len(tg) - 2)
    ok = (tq >= tg[0]) & (tq <= tg[-1]) & (np.abs(tq - tg[j]) <= 45)

    load = np.full(len(when), np.nan)
    rate = np.full(len(when), np.nan)
    for k in np.flatnonzero(ok):
        a = int(j[k]); b = min(a + 1, len(tg) - 1)
        v0 = arr[a, ilat[k], ilon[k]]
        if np.isnan(v0):
            continue
        load[k] = v0
        v1 = arr[b, ilat[k], ilon[k]]
        dt = tg[b] - tg[a]
        if b > a and dt > 0 and not np.isnan(v1):
            rate[k] = (v1 - v0) / dt * 30.0
    return load, rate


def main():
    rule("BEBAN AIR PER GEMPA — DARATAN + LAUT, M TETAP >= 6.0")

    t_l, lat_l, lon_l, tws, n_l, u_l = load_grid("grace_tws.nc", prefer=("tws",))
    print(f"\n  TWS daratan : {len(t_l)} bulan, variabel '{n_l}' [{u_l}]")
    try:
        t_o, lat_o, lon_o, obp, n_o, u_o = load_grid("grace_obp.nc", prefer=("resobp",))
        print(f"  OBP laut    : {len(t_o)} bulan, variabel '{n_o}' [{u_o}]")
        have_obp = True
    except Exception as e:
        print(f"  OBP laut    : belum tersedia ({type(e).__name__})")
        have_obp = False

    ev = decluster(load_quakes())
    ev = [e for e in ev if t_l[0] <= e[0] <= t_l[-1]]
    dts = [e[0] for e in ev]
    lats = np.array([e[2] for e in ev])
    lons = np.array([e[3] for e in ev])
    print(f"\n  gempa M>=6.0 dangkal, declustered, era GRACE: {len(ev):,}")

    # satuan berbeda: TWS dalam cm air, OBP dalam hPa. Samakan ke Pa.
    #   1 cm air = rho*g*h = 98.07 Pa      1 hPa = 100 Pa
    tws = tws * PA_PER_CM
    if have_obp:
        obp = obp * 100.0

    il_l, io_l = cell_index(lat_l, lon_l, lats, lons)
    load_l, rate_l = sample(t_l, tws, il_l, io_l, dts)
    src = np.where(np.isfinite(load_l), "darat", "")

    load = load_l.copy()
    rate = rate_l.copy()
    if have_obp:
        il_o, io_o = cell_index(lat_o, lon_o, lats, lons)
        load_o, rate_o = sample(t_o, obp, il_o, io_o, dts)
        fill = ~np.isfinite(load) & np.isfinite(load_o)
        load[fill] = load_o[fill]
        rate[fill] = rate_o[fill]
        src[fill] = "laut"

    ok = np.isfinite(load)
    n_land = int((src == "darat").sum())
    n_sea = int((src == "laut").sum())
    print(f"""
  tersampel : {int(ok.sum()):,} / {len(ev):,}  ({100*ok.mean():.1f}%)
     darat  : {n_land:,}
     laut   : {n_sea:,}   <- dipulihkan oleh OBP
  sebelumnya (hanya TWS) : {n_land:,}
""")
    if ok.sum() < 100:
        print("  sampel masih terlalu kecil.")
        return

    print(f"  beban di lokasi gempa : rata-rata {np.nanmean(load[ok]):+.0f} Pa, "
          f"sd {np.nanstd(load[ok]):.0f} Pa")
    if n_land and n_sea:
        print(f"     darat sd {np.nanstd(load[src=='darat']):.0f} Pa"
              f"   |   laut sd {np.nanstd(load[src=='laut']):.0f} Pa")

    stats = {"H1_load_mean": np.nanmean(load[ok]),
             "H1_load_absmean": np.nanmean(np.abs(load[ok])),
             "H2_rate_mean": np.nanmean(rate[np.isfinite(rate)]),
             "H2_rate_absmean": np.nanmean(np.abs(rate[np.isfinite(rate)]))}
    null = {k: np.empty(N_SHIFT) for k in stats}

    span = (dts[-1] - dts[0]).days
    base = np.array([d.toordinal() for d in dts], dtype=float)
    for i in range(N_SHIFT):
        off = RNG.integers(60, max(span - 60, 120))
        sh = [datetime.fromordinal(int(v)) for v in
              (dts[0].toordinal() + (base - base[0] + off) % span)]
        l2, r2 = sample(t_l, tws, il_l, io_l, sh)
        if have_obp:
            lo2, ro2 = sample(t_o, obp, il_o, io_o, sh)
            f2 = ~np.isfinite(l2) & np.isfinite(lo2)
            l2[f2] = lo2[f2]; r2[f2] = ro2[f2]
        m2 = np.isfinite(l2); rr = np.isfinite(r2)
        null["H1_load_mean"][i] = np.nanmean(l2[m2]) if m2.any() else np.nan
        null["H1_load_absmean"][i] = np.nanmean(np.abs(l2[m2])) if m2.any() else np.nan
        null["H2_rate_mean"][i] = np.nanmean(r2[rr]) if rr.any() else np.nan
        null["H2_rate_absmean"][i] = np.nanmean(np.abs(r2[rr])) if rr.any() else np.nan
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
        print(f"  {k:<20} {v:>10.3f} {nb.mean():>12.3f} {nb.std():>10.3f} "
              f"{p:>9.4f}{'  <-- LOLOS' if p < alpha else ''}")
    print(f"\n  Bonferroni atas {len(stats)} statistik -> ambang p < {alpha:.4f}")

    rule("VONIS")
    print(f"""
  {'ADA hubungan yang lolos koreksi.' if hit else 'TIDAK ADA yang lolos koreksi.'}

  Sampel naik dari {n_land:,} menjadi {int(ok.sum()):,} tanpa menurunkan ambang
  magnitudo, karena gempa lepas pantai kini dinilai dengan tekanan dasar laut
  alih-alih dibuang. Itu {int(ok.sum())/max(n_land,1):.1f}x lebih banyak event pada M yang sama.

  Satuan disamakan ke pascal: TWS (cm air) x 98,07 dan OBP (hPa) x 100.

  Untuk memperpanjang lagi tanpa menurunkan M, langkah berikutnya adalah
  GRACE-REC (Humphrey & Gudmundsson 2019, CC BY 4.0): rekonstruksi simpanan air
  1901-2014 pada grid 0,5 derajat. Itu menambah rentang dari 22 tahun menjadi
  113 tahun -- sekitar 5x lebih banyak gempa lagi, pada M yang sama.
""")


if __name__ == "__main__":
    main()
