"""
Surface air temperature and precipitable water, at earthquake cells.

olr_test.py tested outgoing longwave radiation and found nothing, but OLR is
measured at the TOP of the atmosphere. Most papers claiming a thermal precursor
describe something else: the ground itself, or the air just above it, warming by
a degree or two in the days before a rupture. Some describe a water-vapour
anomaly instead. Neither is the same quantity as OLR, so neither has actually
been tested here yet.

    air.sig995     air temperature at the sigma 0.995 level -- essentially the
                   near-surface air, which is what "thermal anomaly" usually
                   means in these papers
    pr_wtr.eatm    precipitable water: total column water vapour

    source   NOAA NCEP/NCAR Reanalysis 1, daily, 2.5 degree grid, one file per
             year, from the same host already used for surface pressure

Same storage trick as the pressure and OLR fetchers: the full grids run to
gigabytes, but only the cells containing earthquakes are kept, and each year is
written as its own part so an interrupted run resumes instead of restarting.

These two close the last testable gap. Everything else proposed is either a
duplicate of a family already tested (cosmic ray flux is essentially inverted
solar activity), or measured at too few stations to reach any useful number of
earthquakes (about 30 superconducting gravimeters, about 150 magnetometers),
or has no global archive at all (radon).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import time
import urllib.request as _u
import numpy as np

BASE = ("https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/"
        "Dailies/surface")
VARS = {"air": "air.sig995", "pwat": "pr_wtr.eatm"}
YEARS = range(1970, 2026)
OUT_DIR = DATA / "thermal"
PARTS = OUT_DIR / "_surface_parts"


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fetch_year(stem, year, path):
    url = f"{BASE}/{stem}.{year}.nc"
    try:
        with _u.urlopen(url, timeout=120) as r:
            data = r.read()
    except Exception:
        return None
    if len(data) < 100_000 or data[:4] not in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
        return None
    path.write_bytes(data)
    return path


def cells_for_events():
    """Grid cells (2.5 deg, 73x144) that contain an Mw >= 5 shallow event."""
    from gcmt import parse_ndk
    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= 70.0 and e.mw >= 5.0]
    la = np.array([e.lat for e in ev]); lo = np.array([e.lon for e in ev])
    ilat = np.clip(np.round((90.0 - la) / 2.5).astype(int), 0, 72)
    ilon = np.round((lo % 360.0) / 2.5).astype(int) % 144
    return np.unique(ilat * 144 + ilon)


def main():
    rule("SUHU UDARA PERMUKAAN & UAP AIR — NCEP, DI SEL BERISI GEMPA")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    flat = cells_for_events()
    rows, cols = flat // 144, flat % 144
    print(f"\n  sel berisi gempa: {flat.size:,} dari 10.512 "
          f"({100*flat.size/10512:.1f}%)\n")

    import netCDF4
    tmp = PARTS / "_tmp.nc"
    t0 = time.time()

    for key, stem in VARS.items():
        got_days, got_vals = [], []
        for year in YEARS:
            part = PARTS / f"{key}_{year}.npz"
            if part.exists():
                z = np.load(part)
                got_days.append(z["days"]); got_vals.append(z["vals"])
                continue
            if fetch_year(stem, year, tmp) is None:
                print(f"    {key} {year}: tidak tersedia")
                continue
            d = netCDF4.Dataset(tmp)
            name = next(k for k, v in d.variables.items() if v.ndim == 3)
            tv = d.variables["time"]
            times = netCDF4.num2date(tv[:], tv.units,
                                     only_use_cftime_datetimes=False,
                                     only_use_python_datetimes=True)
            days = np.array([t.toordinal() for t in times])
            arr = np.ma.filled(d.variables[name][:], np.nan).astype(np.float32)
            vals = arr[:, rows, cols]
            d.close()
            np.savez_compressed(part, days=days, vals=vals)
            got_days.append(days); got_vals.append(vals)
            el = (time.time() - t0) / 60
            print(f"    {key} {year}: {days.size} hari   {el:.1f} mnt",
                  flush=True)

        if not got_days:
            print(f"  {key}: tidak ada data.")
            continue
        days = np.concatenate(got_days)
        vals = np.concatenate(got_vals)
        o = np.argsort(days)
        days, vals = days[o], vals[o]
        np.save(OUT_DIR / f"surf_{key}_cells.npy", vals)
        np.save(OUT_DIR / f"surf_{key}_days.npy", days)
        unit = "K" if key == "air" else "kg/m2"
        print(f"""
  {key:<5} tersimpan  {vals.shape}  {vals.nbytes/1e6:.0f} MB
        rentang    {days.min()} .. {days.max()} ({days.size:,} hari)
        nilai      median {np.nanmedian(vals):.1f} {unit}, """
              f"""sd {np.nanstd(vals):.1f}, hilang {100*np.isnan(vals).mean():.2f}%
""")

    np.save(OUT_DIR / "surf_cellid.npy", flat)
    if tmp.exists():
        tmp.unlink()
    print(f"  selesai dalam {(time.time()-t0)/60:.1f} menit")


if __name__ == "__main__":
    main()
