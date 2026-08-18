"""
Surface atmospheric pressure at earthquake locations, 1970-2024.

Atmospheric loading is the surface load everyone quotes as a scale reference and
nobody tests. Weather systems swing surface pressure by +/-30 hPa, which is
+/-3000 Pa on the crust -- four times the solid-earth body tide of the Moon and
within a factor of three of ocean tidal loading. It has spatial structure, a
daily cadence, and an open record going back to 1948.

    source   NCEP/NCAR Reanalysis 1, pres.sfc, 2.5 degree grid, daily
             https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/

Rather than storing the whole global grid for 55 years (about 850 MB in memory),
only the grid cells that actually contain earthquakes are kept. The null shifts
event times while holding their locations fixed, so those cells are the only
ones ever read -- roughly 800 of 10,512, and the stored array drops to tens of
megabytes.

Output: pressure_cells.npy (days x cells), pressure_days.npy, pressure_cellid.npy
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import time
import numpy as np
import requests
from datetime import datetime, timedelta

BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface"
YEARS = range(1970, 2025)
MINMAG, MAXDEPTH = 6.0, 70.0


def quake_cells():
    """Unique NCEP grid cells containing an M>=6 shallow earthquake."""
    lats, lons = [], []
    for r in csv.DictReader(io.StringIO((datafile("global_m6.csv")).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            if float(r["mag"]) < MINMAG or float(r["depth"]) > MAXDEPTH:
                continue
            lats.append(float(r["latitude"]))
            lons.append(float(r["longitude"]))
        except (ValueError, TypeError, KeyError):
            continue
    lats = np.array(lats); lons = np.array(lons)
    # NCEP grid: lat 90..-90 step 2.5 (73), lon 0..357.5 step 2.5 (144)
    ilat = np.clip(np.round((90.0 - lats) / 2.5).astype(int), 0, 72)
    ilon = np.round((lons % 360.0) / 2.5).astype(int) % 144
    flat = ilat * 144 + ilon
    return np.unique(flat)


def main():
    cells = quake_cells()
    print(f"  sel grid berisi gempa: {len(cells):,} dari {73*144:,} "
          f"({100*len(cells)/(73*144):.1f}%)")

    import netCDF4
    days_all, cols_all = [], []
    t0 = time.time()
    for k, y in enumerate(YEARS):
        url = f"{BASE}/pres.sfc.{y}.nc"
        try:
            r = requests.get(url, timeout=300)
            if r.status_code != 200:
                print(f"    {y}: HTTP {r.status_code}")
                continue
            tmp = datafile("_tmp_pres.nc")
            tmp.write_bytes(r.content)
            d = netCDF4.Dataset(tmp)
            p = d.variables["pres"]
            tvar = d.variables["time"]
            # hours since 1800-01-01
            base = datetime(1800, 1, 1)
            times = [base + timedelta(hours=float(v)) for v in tvar[:]]
            arr = np.array(p[:], dtype=np.float32).reshape(len(times), -1)
            cols_all.append(arr[:, cells])
            days_all.extend(t.toordinal() for t in times)
            d.close()
            tmp.unlink()
        except Exception as e:
            print(f"    {y}: {type(e).__name__}")
            continue
        if (k + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(YEARS)} tahun  {el/60:.1f} mnt  "
                  f"sisa ~{(len(YEARS)-k-1)*el/(k+1)/60:.0f} mnt", flush=True)

    if not cols_all:
        print("  tidak ada data terunduh.")
        return
    arr = np.vstack(cols_all)
    days = np.array(days_all)
    order = np.argsort(days)
    np.save(datafile("pressure_cells.npy"), arr[order].astype(np.float32))
    np.save(datafile("pressure_days.npy"), days[order])
    np.save(datafile("pressure_cellid.npy"), cells)
    print(f"\n  selesai: {arr.shape[0]:,} hari x {arr.shape[1]:,} sel "
          f"({arr.nbytes/1e6:.0f} MB)")
    print(f"  tekanan: median {np.nanmedian(arr)/100:.0f} hPa, "
          f"sd {np.nanstd(arr)/100:.1f} hPa")


if __name__ == "__main__":
    main()
