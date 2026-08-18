"""
Fetch global ionospheric TEC maps and reduce them to a compact daily grid.

Source: UPC-IonSAT (Technical University of Catalonia), the UQRG rapid global
ionosphere maps in IONEX 1.0 format. Open, no login. NASA CDDIS holds the same
IGS products but requires an Earthdata account, and CODE Bern was unreachable.

    https://chapman.upc.es/tomion/rapid/YYYY/DDD_YYMMDD.15min/uqrgDDD0.YYi.Z

Each daily file is 97 maps at 15-minute spacing on a 2.5 x 5 degree grid --
1.1 MB compressed, 6.3 MB expanded. Keeping five years of that raw would be
about 2 GB, so each file is reduced to a daily mean per grid cell as it arrives
and the raw is discarded. The result is one float32 array of roughly 38 MB.

Why the daily mean rather than the full 15-minute series: TEC swings by a factor
of five between day and night, and that diurnal cycle is far larger than any
claimed precursor. Averaging it out first is what the precursor literature does
anyway, and it keeps the comparison with GRACE (monthly) sane.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import time
import concurrent.futures as cf
import numpy as np
import requests
from datetime import date, timedelta

BASE = "https://chapman.upc.es/tomion/rapid"
YEARS = range(1998, 2025)              # full UPC archive: 27 years
WORKERS = 5                            # below core count: leave the laptop usable
PARTS = None                           # set in __main__ (needs DATA)
NLAT, NLON = 71, 73                    # 87.5..-87.5 step 2.5 ; -180..180 step 5
OUT_GRID = datafile("tec_daily.npy")
OUT_DAYS = datafile("tec_days.npy")


def parse_ionex(text):
    """Mean TEC per grid cell across all maps in one daily IONEX file, in TECU."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "END OF HEADER" in l) + 1
    except StopIteration:
        return None

    acc = np.zeros((NLAT, NLON))
    cnt = 0
    i = start
    n = len(lines)
    while i < n:
        l = lines[i]
        if "START OF TEC MAP" in l:
            grid = np.full((NLAT, NLON), np.nan)
            row = -1
            i += 1
            while i < n and "END OF TEC MAP" not in lines[i]:
                if "LAT/LON1/LON2/DLON/H" in lines[i]:
                    row += 1
                    vals, j = [], i + 1
                    while j < n and len(vals) < NLON and \
                            "LAT/LON1/LON2/DLON/H" not in lines[j] and \
                            "END OF TEC MAP" not in lines[j]:
                        s = lines[j]
                        vals.extend(int(s[k:k + 5]) for k in range(0, len(s.rstrip()) - 4, 5))
                        j += 1
                    if 0 <= row < NLAT:
                        v = np.array(vals[:NLON], dtype=float)
                        v[v == 9999] = np.nan
                        if v.size == NLON:
                            grid[row] = v / 10.0        # 0.1 TECU -> TECU
                    i = j - 1
                i += 1
            if np.isfinite(grid).any():
                acc += np.nan_to_num(grid)
                cnt += 1
        elif "START OF RMS MAP" in l:
            break                                        # RMS maps follow; stop
        i += 1
    return acc / cnt if cnt else None


def fetch_day(d: date, session):
    doy = d.timetuple().tm_yday
    url = (f"{BASE}/{d.year}/{doy:03d}_{d:%y%m%d}.15min/"
           f"uqrg{doy:03d}0.{d:%y}i.Z")
    r = session.get(url, timeout=120)
    if r.status_code != 200 or len(r.content) < 10_000:
        return None
    import unlzw3
    return parse_ionex(unlzw3.unlzw(r.content).decode("ascii", "replace"))


_SESSION = None


def _worker(ordinal):
    """One day, in its own process. Returns (ordinal, grid) or None."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    try:
        g = fetch_day(date.fromordinal(ordinal), _SESSION)
    except Exception:
        return None
    return None if g is None else (ordinal, g.astype(np.float32))


def main():
    """
    Downloaded a year at a time, in parallel, and each finished year is written
    to its own part file.

    Serially this is 9,862 files at ~2.6 s each -- roughly seven hours, with no
    checkpoint, so any interruption throws all of it away. Both problems are
    fixed here: WORKERS processes run at once, and a year already on disk is
    skipped, so the job can be stopped and resumed at will. Decompression and
    IONEX parsing are pure Python and hold the GIL, so this uses processes
    rather than threads; WORKERS is kept below the core count to leave the
    machine usable.
    """
    PARTS.mkdir(parents=True, exist_ok=True)
    total = sum((date(y, 12, 31) - date(y, 1, 1)).days + 1 for y in YEARS)
    print(f"  {total:,} hari, {WORKERS} proses paralel, disimpan per tahun")
    print(f"  bagian: {PARTS}\n")

    t0 = time.time()
    done = ok = 0
    for y in YEARS:
        part = PARTS / f"tec_{y}.npz"
        ords = [date(y, 1, 1).toordinal() + i
                for i in range((date(y, 12, 31) - date(y, 1, 1)).days + 1)]
        if part.exists():
            done += len(ords)
            ok += int(np.load(part)["days"].size)
            print(f"  {y}  sudah ada, dilewati", flush=True)
            continue

        got = []
        with cf.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for res in pool.map(_worker, ords, chunksize=4):
                done += 1
                if res is not None:
                    got.append(res)
                    ok += 1
        got.sort(key=lambda r: r[0])
        np.savez_compressed(
            part,
            days=np.array([g[0] for g in got]),
            grids=np.stack([g[1] for g in got]) if got
            else np.zeros((0, 71, 73), np.float32))
        el = time.time() - t0
        rate = done / max(el, 1)
        print(f"  {y}  {len(got):>3} hari   total {done:,}/{total:,}  "
              f"{el/60:.0f} mnt  sisa ~{(total-done)/max(rate,1e-9)/60:.0f} mnt",
              flush=True)

    days, grids = [], []
    for y in YEARS:
        part = PARTS / f"tec_{y}.npz"
        if not part.exists():
            continue
        z = np.load(part)
        if z["days"].size:
            days.extend(z["days"].tolist())
            grids.append(z["grids"])
    grids = [g for g in grids if len(g)]

    if not grids:
        print("  tidak ada berkas terunduh.")
        return
    grids = [np.concatenate(grids)]
    arr = grids[0]
    np.save(OUT_GRID, arr)
    np.save(OUT_DAYS, np.array(days))
    print(f"\n  selesai: {ok}/{total} hari")
    print(f"  grid    : {arr.shape}  ({arr.nbytes/1e6:.0f} MB)")
    print(f"  TEC     : median {np.nanmedian(arr):.1f} TECU, "
          f"maks {np.nanmax(arr):.1f}")
    print(f"  tersimpan: {OUT_GRID.name}, {OUT_DAYS.name}")


if __name__ == "__main__":
    PARTS = OUT_GRID.parent / "_parts"
    main()
