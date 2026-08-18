"""
Hydrological loading from GRACE, tested per earthquake at its own location.

Every forcing tested so far except the tides is a GLOBAL scalar: lunar phase,
planetary aspect and solar activity all take one value for the whole planet at
a given instant, so they cannot say where. That is the structural reason they
carry no spatial information, and it is why they were never going to work.

GRACE is different. It measures terrestrial water storage on a 1-degree grid,
monthly, since 2002. Each earthquake can therefore be scored against the water
load at ITS OWN location and month -- the same design Johnson et al. (2017,
Science) used for California, applied globally.

Physics: TWS is reported as equivalent water height. One centimetre of water is
a surface pressure of rho*g*h = 1000 * 9.81 * 0.01 = 98 Pa, so a 30 cm seasonal
swing in a monsoon basin is about 3 kPa -- the same order as ocean tidal loading,
and roughly four times the solid-earth body tide.

Two quantities are tested, both pre-registered here before running:

  H1  LOAD      earthquakes prefer times when local water storage is anomalously
                high or low  -> mean TWS at events differs from the null
  H2  RATE      earthquakes prefer times of rapid loading or unloading
                -> mean |dTWS/dt| or signed dTWS/dt differs from the null

Null: circular shift of the earthquake times. Locations stay put, so each event
is re-scored against the water load its own region had at a different time.
That destroys the timing relationship and preserves everything else -- the
spatial distribution of seismicity, the spatial pattern of water storage, and
the seasonal cycle of both.
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
PA_PER_CM = 1000.0 * 9.80665 * 0.01     # 98.07 Pa per cm of water
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_grace_grid():
    """GRACE/GRACE-FO terrestrial water storage anomaly, cm of equivalent water."""
    import netCDF4
    d = netCDF4.Dataset(datafile("grace_tws.nc"))
    tws = np.array(d.variables["tws"][:], dtype=np.float32)
    lat = np.array(d.variables["lat"][:], dtype=float)
    lon = np.array(d.variables["lon"][:], dtype=float)
    tvar = d.variables["time"]
    epoch = datetime(2002, 4, 18)
    times = np.array([epoch + timedelta(days=float(v)) for v in tvar[:]])
    d.close()
    tws = np.where(np.abs(tws) > 1e4, np.nan, tws)
    return times, lat, lon, tws


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


def sample_grace(times_g, lat_g, lon_g, tws, lats, lons, when):
    """
    TWS and its rate of change at each event's own cell, for the given times.
    `when` lets the caller pass shifted times without moving the locations.
    """
    ilat = np.clip(np.searchsorted(lat_g, lats) - 0, 0, len(lat_g) - 1)
    if lat_g[0] > lat_g[-1]:
        ilat = np.clip(np.abs(lat_g[None, :] - lats[:, None]).argmin(axis=1),
                       0, len(lat_g) - 1)
    else:
        ilat = np.abs(lat_g[None, :] - lats[:, None]).argmin(axis=1)
    lo = lons % 360.0
    lg = lon_g % 360.0
    ilon = np.abs(lg[None, :] - lo[:, None]).argmin(axis=1)

    tg = np.array([t.toordinal() for t in times_g], dtype=float)
    tq = np.array([t.toordinal() for t in when], dtype=float)
    j = np.clip(np.searchsorted(tg, tq) - 1, 0, len(tg) - 2)
    # only accept events inside the GRACE record with a nearby month
    ok = (tq >= tg[0]) & (tq <= tg[-1]) & (np.abs(tq - tg[j]) <= 45)

    load = np.full(len(lats), np.nan)
    rate = np.full(len(lats), np.nan)
    for k in np.flatnonzero(ok):
        a, b = int(j[k]), int(min(j[k] + 1, len(tg) - 1))
        v0 = tws[a, ilat[k], ilon[k]]
        v1 = tws[b, ilat[k], ilon[k]]
        if np.isnan(v0):
            continue
        load[k] = v0
        dt = tg[b] - tg[a]
        if b > a and dt > 0 and not np.isnan(v1):
            rate[k] = (v1 - v0) / dt * 30.0        # cm per month
    return load, rate


def main():
    rule("BEBAN AIR GRACE, DIUJI PER GEMPA DI LOKASINYA SENDIRI")

    times_g, lat_g, lon_g, tws = load_grace_grid()
    print(f"""
  GRACE     {len(times_g)} bulan, {times_g[0].date()} .. {times_g[-1].date()}
  grid      {len(lat_g)} x {len(lon_g)} (1 derajat), satuan cm air ekuivalen
  konversi  1 cm air = {PA_PER_CM:.1f} Pa tekanan permukaan
""")

    ev = decluster(load_quakes())
    ev = [e for e in ev if times_g[0] <= e[0] <= times_g[-1]]
    dts = [e[0] for e in ev]
    lats = np.array([e[2] for e in ev])
    lons = np.array([e[3] for e in ev])
    print(f"  gempa     M>=6.0, kedalaman <=70 km, declustered, dalam era GRACE: {len(ev):,}")

    load, rate = sample_grace(times_g, lat_g, lon_g, tws, lats, lons, dts)
    ok = np.isfinite(load)
    print(f"  tersampel di darat (GRACE punya nilai): {int(ok.sum()):,} "
          f"({100*ok.mean():.1f}%)\n")
    if ok.sum() < 100:
        print("  terlalu sedikit event di darat untuk diuji.")
        return

    print(f"  simpanan air di lokasi gempa : rata-rata {np.nanmean(load[ok]):+.2f} cm "
          f"({np.nanmean(load[ok])*PA_PER_CM:+.0f} Pa)")
    print(f"  sebaran                      : sd {np.nanstd(load[ok]):.2f} cm "
          f"(= {np.nanstd(load[ok])*PA_PER_CM:.0f} Pa)")

    # ---------------- calibrated null: shift times, keep places ----------------
    span_days = (dts[-1] - dts[0]).days
    base = np.array([d.toordinal() for d in dts], dtype=float)
    stats = {"H1_load_mean": np.nanmean(load[ok]),
             "H1_load_absmean": np.nanmean(np.abs(load[ok])),
             "H2_rate_mean": np.nanmean(rate[np.isfinite(rate)]),
             "H2_rate_absmean": np.nanmean(np.abs(rate[np.isfinite(rate)]))}
    null = {k: np.empty(N_SHIFT) for k in stats}

    for i in range(N_SHIFT):
        off = RNG.integers(60, max(span_days - 60, 120))
        shifted = [datetime.fromordinal(int(v)) for v in
                   (dts[0].toordinal() + (base - base[0] + off) % span_days)]
        l2, r2 = sample_grace(times_g, lat_g, lon_g, tws, lats, lons, shifted)
        m2 = np.isfinite(l2)
        rr = np.isfinite(r2)
        null["H1_load_mean"][i] = np.nanmean(l2[m2]) if m2.any() else np.nan
        null["H1_load_absmean"][i] = np.nanmean(np.abs(l2[m2])) if m2.any() else np.nan
        null["H2_rate_mean"][i] = np.nanmean(r2[rr]) if rr.any() else np.nan
        null["H2_rate_absmean"][i] = np.nanmean(np.abs(r2[rr])) if rr.any() else np.nan
        if (i + 1) % 400 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    rule("HASIL  (null: waktu digeser, lokasi tetap)")
    print(f"""
  {'statistik':<20} {'terukur':>12} {'null rata2':>12} {'null sd':>10} {'p dua sisi':>11}
  """ + "-" * 70)
    alpha = 0.05 / len(stats)
    any_hit = False
    for k, v in stats.items():
        nb = null[k][np.isfinite(null[k])]
        if nb.size < 50 or not np.isfinite(v):
            continue
        p = float((np.abs(nb - nb.mean()) >= abs(v - nb.mean())).mean())
        flag = "  <-- LOLOS" if p < alpha else ""
        any_hit |= p < alpha
        unit = "cm" if "load" in k else "cm/bln"
        print(f"  {k:<20} {v:>9.3f} {unit:<3} {nb.mean():>11.3f} {nb.std():>10.3f} "
              f"{p:>11.4f}{flag}")

    print(f"\n  Bonferroni atas {len(stats)} statistik -> ambang p < {alpha:.4f}")

    rule("VONIS")
    print(f"""
  {'ADA hubungan beban air-gempa yang lolos koreksi.' if any_hit
    else 'TIDAK ADA hubungan yang lolos koreksi.'}

  Yang membuat uji ini lebih kuat daripada semua uji langit sebelumnya:
  GRACE punya struktur spasial, jadi tiap gempa dinilai dengan beban air di
  lokasinya sendiri, bukan satu angka global. Null-nya menggeser waktu tetapi
  menahan lokasi, sehingga pola spasial seismisitas dan pola spasial simpanan
  air keduanya utuh -- yang dihancurkan hanya kaitan waktunya.

  Batas fisik untuk konteks: sd simpanan air di lokasi gempa setara
  {np.nanstd(load[ok])*PA_PER_CM:.0f} Pa. Bandingkan body tide 723 Pa, beban
  pasang surut laut ~10.000 Pa, transfer tegangan antar-gempa 10^5-10^7 Pa.
""")


if __name__ == "__main__":
    main()
