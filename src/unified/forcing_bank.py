"""
One bank of external forcing variables, five families, one interface.

Every hypothesis in this repository was tested with its own script, its own
catalogue slice and its own null. That is how the published literature looks
too: Hough (2018) tested lunar phase, Love & Thomas (2013) tested solar
activity, Romanet (2023) tested planetary conjunctions, Ide et al. (2016)
tested tides, Johnson et al. (2017) tested hydrology -- each with different
data, different periods and different statistics. The results cannot be laid
next to each other.

This module puts them on one footing. Every family produces variables sampled
on the same daily grid, over the same span, so a single test with a single null
can be run across all of them and the answers compared directly.

  LUNAR       phase harmonics 1-4, distance, declination, offset to cardinal
  TIDAL       computed Coulomb stress on a real fault: body tide, ocean load,
              spring-neap envelope
  PLANETARY   pairwise aspects in three frames, alignments, aspect counts
  SOLAR       Kp, ap, sunspot number, F10.7, and OMNI solar-wind proton
              density, speed, ram pressure, Dst, AE
  HYDRO       annual and semi-annual harmonics -- the seasonal water-loading
              signature that the Himalayan and Californian studies detect.
              A GRACE slot is reserved; see load_grace() below.

Only variables that actually vary are returned, so a family silently shrinks
rather than failing when a data file is missing.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime, timedelta

from ephem_vec import jd_from_datetime_array, bodies_equatorial, AU, D2R
from planetary import heliocentric, PLANETS, aspect_distance
from planetary_tide import planet_equatorial, dcfs_from_body, GM
from tidal_stress import dcfs_series
from ocean_load import OceanTide

SYNODIC = 29.530588853
SITE = dict(lat=-9.5, lon=108.0, strike=288.0, dip=12.0, rake=95.0)
GM_MOON_ = 4.9028695e12
GM_SUN_ = 1.32712440018e20


# ------------------------------------------------------------------ helpers

def _ffill(v):
    v = np.asarray(v, dtype=float).copy()
    last = np.nan
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    med = np.nanmedian(v)
    v[np.isnan(v)] = 0.0 if np.isnan(med) else med
    return v


def _daily_from_records(recs, days):
    """Average a {datetime: value} mapping onto the daily grid."""
    acc = {}
    for t, v in recs:
        if np.isnan(v):
            continue
        d = t.toordinal()
        a = acc.get(d)
        acc[d] = (a[0] + v, a[1] + 1) if a else (v, 1)
    out = np.full(len(days), np.nan)
    for i, d in enumerate(days):
        a = acc.get(int(d))
        if a:
            out[i] = a[0] / a[1]
    return _ffill(out)


# ------------------------------------------------------------------ families

def family_lunar(days, jd, eph):
    out = {}
    ra_m, dec_m, dist_m = eph["moon"]
    ra_s, dec_s, dist_s = eph["sun"]
    r = D2R
    ce = (np.sin(dec_m * r) * np.sin(dec_s * r)
          + np.cos(dec_m * r) * np.cos(dec_s * r) * np.cos((ra_m - ra_s) * r))
    el = np.degrees(np.arccos(np.clip(ce, -1, 1)))
    sign = np.sign(((ra_m - ra_s) % 360) - 180)
    phase = np.where(sign >= 0, el, 360 - el)

    out["moon_phase"] = phase
    for k in (1, 2, 3, 4):
        out[f"moon_h{k}_sin"] = np.sin(np.radians(k * phase))
        out[f"moon_h{k}_cos"] = np.cos(np.radians(k * phase))
    out["moon_offset_cardinal"] = np.minimum.reduce(
        [np.abs(((phase - c + 180) % 360) - 180) for c in (0, 90, 180, 270)])
    out["moon_distance"] = dist_m
    out["moon_declination"] = dec_m
    out["sun_distance"] = dist_s
    return out


def family_tidal(days, jd, eph):
    out = {}
    for body, gm in (("moon", GM_MOON_), ("sun", GM_SUN_)):
        ra, dec, dd = eph[body]
        out[f"tide_dcfs_{body}"] = dcfs_from_body(
            SITE["lat"], SITE["lon"], SITE["strike"], SITE["dip"],
            SITE["rake"], jd, ra, dec, dd, gm)
    total = out["tide_dcfs_moon"] + out["tide_dcfs_sun"]
    out["tide_dcfs_lunisolar"] = total
    out["tide_dcfs_abs"] = np.abs(total)

    # spring-neap envelope
    ra_m, dec_m, _ = eph["moon"]
    ra_s, dec_s, _ = eph["sun"]
    r = D2R
    ce = (np.sin(dec_m * r) * np.sin(dec_s * r)
          + np.cos(dec_m * r) * np.cos(dec_s * r) * np.cos((ra_m - ra_s) * r))
    el = np.degrees(np.arccos(np.clip(ce, -1, 1)))
    am, asun = 2.17, 1.0
    out["tide_springneap"] = np.sqrt(am**2 + asun**2
                                     + 2 * am * asun * np.cos(np.radians(2 * el)))

    # ocean loading at the same site
    try:
        ot = OceanTide()
        co = ot.interp(SITE["lat"], SITE["lon"])
        h = ot.height(co, jd[None, :])[0]
        out["ocean_tide_height"] = h
        out["ocean_load_abs"] = np.abs(h)
        d = dcfs_series(np.array([[SITE["lat"]]]), np.array([[SITE["lon"]]]),
                        np.array([SITE["strike"]]), np.array([SITE["dip"]]),
                        np.array([SITE["rake"]]), jd[None, :],
                        ocean_height=h[None, :])
        out["tide_dcfs_with_ocean"] = d.ravel()
    except Exception as e:
        print(f"    (beban laut dilewati: {type(e).__name__})")
    return out


def family_planetary(days, jd, eph):
    out = {}
    ex, ey, _ = heliocentric("Earth", jd)
    sun_lon = (np.degrees(np.arctan2(ey, ex)) + 180) % 360
    geo, helio = {}, {}
    for p in PLANETS:
        px, py, _ = heliocentric(p, jd)
        geo[p] = np.degrees(np.arctan2(py - ey, px - ex)) % 360
        helio[p] = np.degrees(np.arctan2(py, px)) % 360
        out[f"pl_{p}_elong"] = aspect_distance(geo[p], sun_lon)

    pairs = [(PLANETS[i], PLANETS[j])
             for i in range(len(PLANETS)) for j in range(i + 1, len(PLANETS))]
    for a, b in pairs:
        sg = aspect_distance(geo[a], geo[b])
        sh = aspect_distance(helio[a], helio[b])
        out[f"pl_geo_{a}_{b}"] = sg
        out[f"pl_helio_{a}_{b}"] = sh
        out[f"pl_geo_{a}_{b}_prox"] = np.minimum.reduce(
            [np.abs(sg - t) for t in (0, 45, 90, 135, 180)])

    for tol in (2.0, 3.0):
        n_act = np.zeros(len(days))
        for a, b in pairs:
            sep = aspect_distance(geo[a], geo[b])
            near = np.minimum.reduce([np.abs(sep - t) for t in (0, 45, 90, 135, 180)])
            n_act += (near <= tol)
        out[f"pl_aspectcount_tol{tol:g}"] = n_act

    # planetary tidal stress on the same fault as the TIDAL family
    tot = np.zeros(len(days))
    for p in PLANETS:
        ra, dec, dd = planet_equatorial(p, jd)
        d = dcfs_from_body(SITE["lat"], SITE["lon"], SITE["strike"],
                           SITE["dip"], SITE["rake"], jd, ra, dec, dd, GM[p])
        out[f"pl_tide_{p}"] = d
        tot += d
    out["pl_tide_total"] = tot
    return out


def family_solar(days, jd, eph):
    out = {}
    # Kp / ap, 3-hourly
    f = datafile("kp_ap.txt")
    if f.exists():
        recs_k, recs_a = [], []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            g = line.split()
            try:
                t = datetime(int(g[0]), int(g[1]), int(g[2])) + timedelta(hours=int(float(g[3])))
                k, a = float(g[7]), float(g[8])
            except (ValueError, IndexError):
                continue
            if k < 0 or a < 0:
                continue
            recs_k.append((t, k)); recs_a.append((t, a))
        out["sol_kp"] = _daily_from_records(recs_k, days)
        out["sol_ap"] = _daily_from_records(recs_a, days)

    # sunspot number, daily
    f = datafile("sunspot.txt")
    if f.exists():
        recs = []
        for line in f.read_text(encoding="utf-8").splitlines():
            g = line.split()
            if len(g) < 5:
                continue
            try:
                sn = float(g[4])
                if sn < 0:
                    continue
                recs.append((datetime(int(g[0]), int(g[1]), int(g[2])), sn))
            except (ValueError, IndexError):
                continue
        out["sol_sunspot"] = _daily_from_records(recs, days)

    # F10.7, daily
    f = datafile("f107.txt")
    if f.exists():
        recs = []
        for line in f.read_text(encoding="utf-8").splitlines():
            g = line.split()
            if len(g) < 6 or not g[0].isdigit() or len(g[0]) != 8:
                continue
            try:
                recs.append((datetime(int(g[0][:4]), int(g[0][4:6]), int(g[0][6:8])),
                             float(g[5])))
            except (ValueError, IndexError):
                continue
        if recs:
            out["sol_f107"] = _daily_from_records(recs, days)

    # OMNI solar wind, hourly -- includes the proton density the LSTM paper uses
    f = datafile("omni_hourly.csv")
    if f.exists():
        cols = {"omni_sw_n": "sol_proton_density", "omni_sw_v": "sol_sw_speed",
                "omni_sw_ram": "sol_ram_pressure", "omni_dst": "sol_dst",
                "omni_ae": "sol_ae"}
        buf = {k: [] for k in cols}
        for row in csv.DictReader(io.StringIO(f.read_text(encoding="utf-8"))):
            try:
                t = datetime.strptime(row["Time"][:19], "%Y-%m-%dT%H:%M:%S")
            except (ValueError, TypeError, KeyError):
                continue
            for src in cols:
                try:
                    v = float(row.get(src, ""))
                except (TypeError, ValueError):
                    continue
                if abs(v) < 9e4:
                    buf[src].append((t, v))
        for src, name in cols.items():
            if buf[src]:
                out[name] = _daily_from_records(buf[src], days)
    return out


def load_grace(days):
    """
    GRACE / GRACE-FO terrestrial water storage, wired in.

    The file is a monthly 1-degree global grid of equivalent water height (cm).
    For this daily, globally-aggregated framework the natural summaries are the
    land-average storage and its rate of change, interpolated onto the daily
    grid. One centimetre of water is 98 Pa of surface pressure.

    NOTE ON WHICH ARM IS STRONGER. Collapsing GRACE to a global average throws
    away the very thing that makes it better than lunar phase or solar activity:
    spatial structure. The stronger test scores each earthquake against the water
    load at ITS OWN location and month, and lives in grace_hydro.py. What is
    returned here is the weaker, globally-averaged arm, included so that HYDRO
    sits in the same table as the other families.
    """
    f = datafile("grace_tws.nc")
    if not f.exists():
        return {}
    try:
        import netCDF4
    except ImportError:
        return {}
    d = netCDF4.Dataset(f)
    tws = np.array(d.variables["tws"][:], dtype=np.float32)
    tvar = np.array(d.variables["time"][:], dtype=float)
    d.close()
    tws = np.where(np.abs(tws) > 1e4, np.nan, tws)

    epoch = datetime(2002, 4, 18)
    gt = np.array([(epoch + timedelta(days=float(v))).toordinal() for v in tvar],
                  dtype=float)
    land_mean = np.nanmean(tws.reshape(tws.shape[0], -1), axis=1)
    land_abs = np.nanmean(np.abs(tws.reshape(tws.shape[0], -1)), axis=1)
    good = np.isfinite(land_mean)
    if good.sum() < 24:
        return {}

    dd = np.asarray(days, dtype=float)
    out = {}
    out["hyd_grace_land"] = np.interp(dd, gt[good], land_mean[good],
                                      left=np.nan, right=np.nan)
    out["hyd_grace_absload"] = np.interp(dd, gt[good], land_abs[good],
                                         left=np.nan, right=np.nan)
    rate = np.concatenate([[0.0], np.diff(land_mean[good]) /
                           np.maximum(np.diff(gt[good]), 1e-9)])
    out["hyd_grace_rate"] = np.interp(dd, gt[good], rate,
                                      left=np.nan, right=np.nan)
    # outside the GRACE era the variable is undefined; fill so it is dropped
    # only if it carries no variance at all
    for k in list(out):
        v = out[k]
        if np.all(np.isnan(v)):
            del out[k]
        else:
            out[k] = _ffill(v)
    return out


def family_hydro(days, jd, eph):
    out = {}
    dts = [datetime.fromordinal(int(d)) for d in days]
    doy = np.array([(d - datetime(d.year, 1, 1)).days for d in dts], float)
    yearlen = np.array([366.0 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0))
                        else 365.0 for d in dts])
    ang = 2 * np.pi * doy / yearlen
    out["hyd_annual_sin"] = np.sin(ang)
    out["hyd_annual_cos"] = np.cos(ang)
    out["hyd_semiannual_sin"] = np.sin(2 * ang)
    out["hyd_semiannual_cos"] = np.cos(2 * ang)
    out["hyd_dayofyear"] = doy

    out.update(load_grace(days))
    return out


def family_rotation(days, jd, eph):
    """Earth rotation from IERS: length of day and polar motion."""
    out = {}
    f = datafile("iers_eop.txt")
    if not f.exists():
        return out
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        g = line.split()
        if len(g) < 13 or not g[0].isdigit():
            continue
        try:
            d = datetime(int(g[0]), int(g[1]), int(g[2])).toordinal()
            rows.append((d, float(g[12]) * 1000.0,
                         np.hypot(float(g[5]), float(g[6]))))
        except (ValueError, IndexError):
            continue
    if len(rows) < 100:
        return out
    rows.sort()
    a = np.array(rows)
    gd, lod, pm = a[:, 0], a[:, 1], a[:, 2]
    dd = np.asarray(days, dtype=float)
    out["rot_lod"] = np.interp(dd, gd, lod)
    out["rot_lod_abs"] = np.abs(out["rot_lod"] - np.median(out["rot_lod"]))
    # backward difference: np.gradient averages both neighbours and would let
    # a "rate" variable read one day into the future
    out["rot_lod_rate"] = np.concatenate([[0.0], np.diff(out["rot_lod"])])
    smooth = np.convolve(lod, np.ones(5 * 365) / (5 * 365), mode="same")
    out["rot_lod_decadal"] = np.interp(dd, gd, smooth)
    out["rot_polar_motion"] = np.interp(dd, gd, pm)
    return out


def family_atmosphere(days, jd, eph):
    """
    Surface pressure loading, as an anomaly against each cell's own running
    median -- raw pressure is dominated by elevation, not weather.

    The stored array holds only the grid cells containing earthquakes, so the
    global-daily summary here is the mean and spread across those cells.
    """
    out = {}
    fc, fd = datafile("pressure_cells.npy"), datafile("pressure_days.npy")
    if not (fc.exists() and fd.exists()):
        return out
    arr = np.load(fc)
    pdays = np.load(fd)
    order = np.argsort(pdays)
    arr, pdays = arr[order], pdays[order]

    win, half = 30, 15
    anom = np.empty_like(arr)
    for i in range(arr.shape[0]):
        a, b = max(0, i - half), min(arr.shape[0], i + half + 1)
        anom[i] = arr[i] - np.median(arr[a:b], axis=0)

    dd = np.asarray(days, dtype=float)
    gd = pdays.astype(float)
    out["atm_anom_mean"] = np.interp(dd, gd, np.nanmean(anom, axis=1))
    out["atm_anom_absmean"] = np.interp(dd, gd, np.nanmean(np.abs(anom), axis=1))
    out["atm_anom_spread"] = np.interp(dd, gd, np.nanstd(anom, axis=1))
    out["atm_anom_rate"] = np.concatenate([[0.0], np.diff(out["atm_anom_mean"])])
    return out


def family_ionosphere(days, jd, eph):
    """Global TEC anomaly, if the maps have been fetched."""
    out = {}
    fg, fd = datafile("tec_daily.npy"), datafile("tec_days.npy")
    if not (fg.exists() and fd.exists()):
        return out
    grid = np.load(fg)
    tdays = np.load(fd)
    order = np.argsort(tdays)
    grid, tdays = grid[order], tdays[order]
    half = 13
    anom = np.empty_like(grid)
    for i in range(grid.shape[0]):
        a, b = max(0, i - half), min(grid.shape[0], i + half + 1)
        anom[i] = grid[i] - np.median(grid[a:b], axis=0)
    flat = anom.reshape(anom.shape[0], -1)
    dd = np.asarray(days, dtype=float)
    gd = tdays.astype(float)
    out["ion_tec_mean"] = np.interp(dd, gd, np.nanmean(flat, axis=1),
                                    left=np.nan, right=np.nan)
    out["ion_tec_absmean"] = np.interp(dd, gd, np.nanmean(np.abs(flat), axis=1),
                                       left=np.nan, right=np.nan)
    out["ion_tec_spread"] = np.interp(dd, gd, np.nanstd(flat, axis=1),
                                      left=np.nan, right=np.nan)
    for k in list(out):
        if np.all(np.isnan(out[k])):
            del out[k]
        else:
            out[k] = _ffill(out[k])
    return out


FAMILY_BUILDERS = {
    "LUNAR": family_lunar,
    "TIDAL": family_tidal,
    "PLANETARY": family_planetary,
    "SOLAR": family_solar,
    "HYDRO": family_hydro,
    "ROTATION": family_rotation,
    "ATMOSPHERE": family_atmosphere,
    "IONOSPHERE": family_ionosphere,
}


def build_all(days, verbose=True):
    """Every family on one daily grid. Returns (X, names, family_of_column)."""
    dts = [datetime.fromordinal(int(d)) for d in days]
    jd = jd_from_datetime_array(dts)
    eph = bodies_equatorial(jd)

    names, cols, fams = [], [], []
    for fam, fn in FAMILY_BUILDERS.items():
        got = fn(days, jd, eph)
        kept = 0
        for k, v in got.items():
            v = np.asarray(v, dtype=float).ravel()
            if v.size != len(days) or not np.all(np.isfinite(v)) or np.ptp(v) <= 0:
                continue
            names.append(k); cols.append(v); fams.append(fam); kept += 1
        if verbose:
            print(f"    {fam:<10} {kept:>4} variabel")
    return np.column_stack(cols), names, np.array(fams)
