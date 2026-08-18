"""
Fetch daily outgoing longwave radiation and reduce it to earthquake cells.

OLR is the infrared energy the Earth radiates to space. It is the measurement
behind the thermal precursor claim -- that a patch of ground warms, or its
emission changes, in the days before a large earthquake beneath it. Together
with TEC and radon it makes up the trio of precursor signals cited most often,
and of the three it is the only one still untested here: radon has no global
archive at all, and TEC was tested and came back null.

    source   NOAA PSL, interpolated OLR, daily 2.5 degree global grid,
             1974 to present -- the same host already used for surface
             pressure, so the route is known to work

Note what category this belongs to. OLR is not a force: it pushes on nothing
and appears nowhere on the stress ladder. The claim is the reverse of the tidal
one -- that earthquake preparation disturbs the surface, rather than that a
surface load triggers an earthquake. That makes it a symptom, and symptoms have
a failure mode forces do not: an effect CAUSED by the earthquake can be
mistaken for one preceding it. Causal sampling is therefore not optional here,
and the same discipline that turned GRACE from p < 0.0001 into p = 0.36 is
applied when this is tested.

Storage follows fetch_pressure.py: the full grid is 349 MB, but only the cells
that actually contain earthquakes are kept, which is under a tenth of that.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import time
import urllib.request as _u
import numpy as np

URL = ("https://psl.noaa.gov/thredds/fileServer/Datasets/interp_OLR/"
       "olr.day.mean.nc")
OUT_DIR = DATA / "thermal"
RAW = OUT_DIR / "olr.day.mean.nc"


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def _total_size():
    with _u.urlopen(_u.Request(URL, method="HEAD"), timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def download(max_tries=40):
    """
    Resumable byte-range download.

    A single 349 MB stream over this link does not survive: the first attempt
    stopped at 9 MB and urllib reported success, because a dropped connection
    simply looks like end-of-file. The server advertises Accept-Ranges, so the
    file is fetched in resumable pieces instead and every attempt continues
    from whatever is already on disk. The length is checked against the
    server's Content-Length at the end, which is the only way to tell a
    finished download from a truncated one.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = _total_size()
    print(f"  ukuran di server: {total/1e6:.0f} MB")

    t0 = time.time()
    for attempt in range(max_tries):
        have = RAW.stat().st_size if RAW.exists() else 0
        if have >= total > 0:
            print(f"  lengkap: {have/1e6:.0f} MB dalam "
                  f"{(time.time()-t0)/60:.1f} menit")
            return
        req = _u.Request(URL, headers={"Range": f"bytes={have}-"})
        try:
            with _u.urlopen(req, timeout=120) as r, open(RAW, "ab") as f:
                if r.status not in (200, 206):
                    raise OSError(f"status {r.status}")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as exc:
            print(f"    putus ({type(exc).__name__}), menyambung ulang",
                  flush=True)
        now = RAW.stat().st_size if RAW.exists() else 0
        print(f"    {now/1e6:>6.0f}/{total/1e6:.0f} MB  "
              f"{100*now/total:>5.1f}%  percobaan {attempt+1}", flush=True)
        if now <= have:                      # no progress at all: back off
            time.sleep(3)

    have = RAW.stat().st_size if RAW.exists() else 0
    if have < total:
        raise OSError(f"unduhan tidak lengkap: {have:,} dari {total:,} byte")


def main():
    rule("OLR — RADIASI INFRAMERAH HARIAN, NOAA PSL")
    download()

    import netCDF4
    d = netCDF4.Dataset(RAW)
    name = next((k for k in ("olr", "OLR") if k in d.variables), None)
    if name is None:
        name = next(k for k, v in d.variables.items() if v.ndim == 3)
    var = d.variables[name]
    lats = np.array(d.variables["lat"][:], dtype=float)
    lons = np.array(d.variables["lon"][:], dtype=float)
    tvar = d.variables["time"]
    import netCDF4 as nc
    times = nc.num2date(tvar[:], tvar.units,
                        only_use_cftime_datetimes=False,
                        only_use_python_datetimes=True)
    days = np.array([t.toordinal() for t in times])
    print(f"""
  variabel  {name}   {var.shape}
  grid      {lats.size} x {lons.size}  ({abs(lats[1]-lats[0]):.1f} x """
          f"""{abs(lons[1]-lons[0]):.1f} derajat)
  waktu     {times[0].date()} .. {times[-1].date()}  ({days.size:,} hari)
""")

    # keep only cells that contain an earthquake -- same trick as the pressure
    # fetcher, and the reason this ends up under 40 MB instead of 349
    from gcmt import parse_ndk
    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= 70.0 and e.mw >= 5.0]
    ela = np.array([e.lat for e in ev]); elo = np.array([e.lon for e in ev])
    ilat = np.abs(lats[None, :] - ela[:, None]).argmin(axis=1)
    lo360 = elo % 360.0
    glon = lons % 360.0
    ilon = np.abs(glon[None, :] - lo360[:, None]).argmin(axis=1)
    flat = np.unique(ilat * lons.size + ilon)
    print(f"  sel berisi gempa: {flat.size:,} dari {lats.size*lons.size:,} "
          f"({100*flat.size/(lats.size*lons.size):.1f}%)")

    rows, cols = flat // lons.size, flat % lons.size
    out = np.empty((days.size, flat.size), dtype=np.float32)
    step = 500
    t0 = time.time()
    for a in range(0, days.size, step):
        b = min(a + step, days.size)
        # netCDF4 returns a masked array; np.array() would keep the raw
        # missing_value (-9.97e36) as data and poison every later mean and sd.
        # valid_range is 0-500 W/m2, so anything outside it becomes NaN.
        block = np.ma.filled(var[a:b, :, :], np.nan).astype(np.float32)
        block[(block < 0.0) | (block > 500.0)] = np.nan
        out[a:b] = block[:, rows, cols]
        el = (time.time() - t0) / 60
        print(f"    {b:,}/{days.size:,} hari  {el:.1f} mnt", end="\r", flush=True)
    print(" " * 50, end="\r")

    np.save(OUT_DIR / "olr_cells.npy", out)
    np.save(OUT_DIR / "olr_days.npy", days)
    np.save(OUT_DIR / "olr_cellid.npy", flat)
    np.save(OUT_DIR / "olr_gridshape.npy", np.array([lats.size, lons.size]))
    print(f"""
  tersimpan  olr_cells.npy  {out.nbytes/1e6:.0f} MB
             olr_days.npy, olr_cellid.npy, olr_gridshape.npy
  nilai      median {np.nanmedian(out):.1f} W/m2, sd {np.nanstd(out):.1f}
  hilang     {100*np.isnan(out).mean():.2f}%
""")


if __name__ == "__main__":
    main()
