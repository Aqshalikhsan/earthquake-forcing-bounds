"""
Is the weak result a real null, or just too little data? Asked family by family.

A null result has two possible causes and they are not the same thing:

    the effect is absent
    the effect is there but the sample cannot see it

Everything reported so far has been the first reading. This file checks the
second, because the four field families differ enormously in what they cover:
pressure runs 55 years, GRACE 22, TEC only 5. A null from 3,000 events and a
null from 348 do not carry the same weight and should not be written down the
same way.

For each family, two numbers:

    COVERAGE    how many M>=6 epicentres the data actually reaches, and how
                many it misses. This separates "not global" from "not long".
    DETECTABLE  the smallest systematic offset the sample could have caught,
                expressed in units of the field's own standard deviation. It
                comes from the standard error of the per-event mean, at the
                conventional 5% level and 80% power, so it reads directly as
                "an effect this big or larger would not have been missed".

A family whose detectable threshold is small has given a real null. A family
whose threshold is large has given no verdict at all, and saying otherwise
would be claiming evidence of absence from an underpowered test -- the mirror
image of the error the precursor literature makes in the other direction.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from perevent_test import (decluster_cmt, build_atmosphere, build_ionosphere,
                           build_hydro, MINMW, MAXDEPTH)

Z_ALPHA, Z_POWER = 1.96, 0.84          # two-sided 5%, 80% power


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    rule("DAYA PER KELUARGA: nol yang sungguhan, atau sekadar data kurang?")

    ev = parse_ndk(datafile("gcmt.ndk"))
    ev = decluster_cmt([e for e in ev if e.mw >= MINMW and e.depth <= MAXDEPTH])
    lats = np.array([e.lat for e in ev]); lons = np.array([e.lon for e in ev])
    ords = np.array([e.dt.toordinal() for e in ev])
    n_all = len(ev)
    print(f"""
  katalog induk  GCMT Mw>={MINMW}, dangkal, declustered: {n_all:,} gempa
                 {datetime.fromordinal(int(ords.min())).date()} .. """
          f"""{datetime.fromordinal(int(ords.max())).date()}
""")

    rows = []
    for label, builder in [("ATMOSPHERE", build_atmosphere),
                           ("IONOSPHERE", build_ionosphere),
                           ("HYDRO", build_hydro)]:
        try:
            f = builder(lats, lons)
        except Exception as exc:
            print(f"  {label} dilewati: {type(exc).__name__}")
            continue
        s = f.sample(ords)
        # the family's most informative variable in the per-event test is the
        # rate column; report the level column too so the units stay legible
        cov = np.isfinite(s[:, 0])
        n = int(cov.sum())
        span_days = (f.a.days if hasattr(f, "a") else f.days)
        span_yr = (span_days.max() - span_days.min()) / 365.25
        # in-window events: those whose date falls inside the family's record
        inwin = int(((ords >= span_days.min()) & (ords <= span_days.max())).sum())
        sd_ev = float(np.nanstd(s[cov, 0]))
        se = sd_ev / np.sqrt(max(n, 1))
        detect = (Z_ALPHA + Z_POWER) * se / sd_ev if sd_ev > 0 else np.nan
        rows.append((label, n, inwin, span_yr, detect))

    print(f"  {'keluarga':<12} {'rentang':>9} {'gempa dlm':>10} {'terpakai':>9} "
          f"{'% katalog':>10} {'terdeteksi':>11}")
    print(f"  {'':<12} {'(tahun)':>9} {'rentang':>10} {'':>9} {'':>10} {'(sd medan)':>11}")
    print("  " + "-" * 68)
    for label, n, inwin, yr, det in rows:
        print(f"  {label:<12} {yr:>9.0f} {inwin:>10,} {n:>9,} "
              f"{100*n/n_all:>9.0f}% {det:>11.3f}")

    rule("MEMBACA TABEL INI")
    print("""
  'gempa dlm rentang'  berapa gempa yang JATUH di dalam periode datanya
  'terpakai'           berapa yang benar-benar tersampel di selnya sendiri
  Kalau kedua angka itu berdekatan, cakupannya GLOBAL dan yang membatasi
  adalah PANJANG REKAMAN, bukan keluasan wilayah.

  'terdeteksi' adalah pergeseran sistematis terkecil yang masih tertangkap,
  dalam satuan simpangan baku medan itu sendiri. Makin kecil, makin kuat
  nolnya. Angka 0,05 berarti: pergeseran sebesar 5% sd tidak akan terlewat,
  jadi nol di situ adalah nol yang sungguhan. Angka 0,15 berarti hanya efek
  besar yang tertangkap, dan nol di situ belum memutuskan apa-apa.
""")


if __name__ == "__main__":
    main()
