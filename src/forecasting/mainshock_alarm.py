"""
The 88%: can a mainshock be warned about at all, if it follows nothing?

window_forecast.py showed that aftershock forecasting works well and reaches
about 12% of large earthquakes. This file goes after the rest -- the mainshocks
that arrive without a large earthquake before them, which is where every real
prediction claim has always been aimed, and where this project has now ruled
out eight external forcings.

One seismological signal remains, and unlike planetary geometry it has genuine
support: FORESHOCKS. Some mainshocks are preceded by smaller earthquakes on the
same structure. Nobody disputes that this happens. What is disputed is whether
it happens often enough, and distinctively enough, to act on -- because small
earthquakes are extremely common, and almost all of them are followed by
nothing.

That trade-off is measured here with the standard alarm framework:

    targets   Mw >= 6.0, shallow, Gardner-Knopoff DECLUSTERED -- mainshocks
              only, so nothing is credited for predicting an aftershock
    alarm     after any event of Mw >= Mt, declare an alarm within R km for
              D days; Mt is swept downwards to buy coverage
    hit rate  fraction of mainshocks that fall inside some alarm
    baseline  the same mainshocks with their times shifted, which measures how
              often an alarm would have caught them by luck alone
    gain      hit rate divided by baseline -- how much better than chance

LOWERING THE TRIGGER IS NOT FREE. Dropping Mt from 6.0 to 5.0 multiplies the
number of alarms roughly tenfold, and the baseline rises with it. A rule that
declares alarms often will catch more earthquakes while knowing less; the gain
column is what strips that illusion out, because it divides by exactly how much
alarm was spent.

Catalogue completeness is the limit on how far Mt can be pushed: Global CMT is
complete to about Mw 5.0-5.5 worldwide, so 4.5 is reported but should be read
as indicative only.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec

MAXDEPTH = 70.0
TARGET_MW = 6.0
TRIGGERS = [4.5, 5.0, 5.5, 6.0]
RADIUS = 100
WINDOWS = [7, 15, 30]
N_SHIFT = 200
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def decluster(ordn, mw, lat, lon):
    """Gardner-Knopoff: keep mainshocks only."""
    dead = np.zeros(mw.size, dtype=bool)
    for i in np.argsort(-mw):
        if dead[i]:
            continue
        dkm, ddays = gardner_knopoff_window(mw[i])
        d = ordn - ordn[i]
        cand = (d > 0) & (d <= ddays) & (mw <= mw[i]) & (~dead)
        idx = np.flatnonzero(cand)
        if idx.size:
            dead[idx[haversine_vec(lat[i], lon[i], lat[idx], lon[idx]) <= dkm]] = True
    return ~dead


def under_alarm(t, la, lo, ordn, lat, lon, mw, Mt, R, D):
    """Was (t, la, lo) inside an alarm opened by an Mw >= Mt event?"""
    d = t - ordn
    sel = (d > 0) & (d <= D) & (mw >= Mt)
    if not sel.any():
        return False
    idx = np.flatnonzero(sel)
    return bool((haversine_vec(la, lo, lat[idx], lon[idx]) <= R).any())


def main():
    rule("BISAKAH GEMPA UTAMA DIPERINGATKAN? (bukan susulan)")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk")) if e.depth <= MAXDEPTH]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev])
    lon = np.array([e.lon for e in ev])

    big = mw >= TARGET_MW
    keep_main = np.zeros(mw.size, dtype=bool)
    idx_big = np.flatnonzero(big)
    sub = decluster(ordn[idx_big], mw[idx_big], lat[idx_big], lon[idx_big])
    keep_main[idx_big[sub]] = True
    tgt = np.flatnonzero(keep_main)
    span = ordn.max() - ordn.min()

    print(f"""
  katalog   GCMT dangkal            : {mw.size:,}
  Mw>={TARGET_MW}                        : {int(big.sum()):,}
  setelah declustering (UTAMA)     : {tgt.size:,}
                                     <- inilah yang harus diperingatkan
  rentang   {datetime.fromordinal(int(ordn.min())).date()} .. """
          f"""{datetime.fromordinal(int(ordn.max())).date()}

  Aturan alarm: setelah gempa Mw>=Mt, siaga {RADIUS} km selama D hari.
""")

    for D in WINDOWS:
        rule(f"JENDELA {D} HARI, RADIUS {RADIUS} KM")
        print(f"\n  {'pemicu':>8} {'n alarm':>10} {'tertangkap':>12} "
              f"{'kebetulan':>11} {'kelipatan':>11}")
        print("  " + "-" * 58)
        for Mt in TRIGGERS:
            n_trig = int((mw >= Mt).sum())
            hit = np.mean([
                under_alarm(ordn[i], lat[i], lon[i], ordn, lat, lon, mw,
                            Mt, RADIUS, D) for i in tgt])
            base = []
            for _ in range(N_SHIFT):
                off = RNG.integers(60, span - 60)
                s = RNG.choice(tgt, size=min(250, tgt.size), replace=False)
                t = ordn.min() + (ordn[s] - ordn.min() + off) % span
                base.append(np.mean([
                    under_alarm(t[k], lat[j], lon[j], ordn, lat, lon, mw,
                                Mt, RADIUS, D) for k, j in enumerate(s)]))
            pb = float(np.mean(base))
            gain = hit / pb if pb > 0 else np.inf
            flag = "   <- di bawah completeness" if Mt < 5.0 else ""
            print(f"  Mw>={Mt:<4.1f} {n_trig:>10,} {100*hit:>11.1f}% "
                  f"{100*pb:>10.1f}% {gain:>10.1f}x{flag}")

    rule("ARTINYA")
    print("""
  Baca kolom 'tertangkap' dan 'kelipatan' BERSAMAAN.

  Menurunkan ambang pemicu menaikkan jumlah gempa utama yang tertangkap --
  tetapi juga menaikkan berapa sering alarm dinyalakan, sehingga kolom
  'kebetulan' ikut naik. Kelipatan adalah satu-satunya kolom yang sudah
  memperhitungkan harga itu: ia membagi hasil dengan seberapa banyak alarm
  yang dihabiskan.

  Foreshock itu NYATA -- sebagian gempa utama memang didahului gempa kecil di
  struktur yang sama. Yang tidak nyata adalah kemampuan membedakannya di muka:
  gempa kecil terjadi ribuan kali setiap tahun, dan hampir semuanya tidak
  diikuti apa-apa. Itulah sebabnya kelipatan tetap sederhana walaupun
  'tertangkap' terlihat besar.

  Bandingkan dengan peramalan susulan (window_forecast.py): di sana kelipatan
  mencapai 14-19x pada irisan 12% gempa. Di sini irisan bisa jauh lebih besar,
  tetapi kelipatannya jatuh -- dan itu pertukaran yang sesungguhnya, bukan
  kegagalan perangkat lunak.
""")


if __name__ == "__main__":
    main()
