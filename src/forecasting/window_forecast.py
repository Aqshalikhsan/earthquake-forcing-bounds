"""
"Region plus a 10-15 day window" -- what that actually delivers today.

A forecast does not have to name the day. Saying "inside this circle, over the
next two weeks, the probability is X%" is a real and useful product, and it is
what USGS and BMKG already publish after a large earthquake. The question worth
answering is not whether that framing is legitimate -- it is -- but what
numbers it currently produces, and for how many earthquakes it can produce them
at all.

Both halves are computed here straight from the catalogue, with no model
fitted, so nothing depends on a parameter choice:

    SKILL     after a mainshock, the observed probability that another
              earthquake follows inside a radius and a window, against the
              probability of the same thing happening at the same places with
              the calendar shifted. The ratio is what the mainshock buys.

    COVERAGE  the fraction of all large earthquakes that follow a previous
              large earthquake closely enough to have been inside such a
              window in the first place. This is the number that decides how
              much of the problem the method can touch, and it is the one
              usually left out.

The two are in tension, and that tension is the honest answer to "is a 10-15
day regional window good enough". Skill is high where it applies. Coverage
decides how often it applies. A method with a large gain over a small slice of
events is worth deploying, but it is not earthquake prediction, and the
difference matters to anyone deciding what to tell the public.

Catalogue: Global CMT, shallow, Mw >= 5.0. Distances are great-circle.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import haversine_vec

MAXDEPTH = 70.0
MAIN_MW = 6.0
RADII = [100, 200]
WINDOWS = [7, 10, 15, 30]
N_SHIFT = 200
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def any_within(ord_m, lat_m, lon_m, ordn, lat, lon, mw, thresh, R, D,
               exclude_self=True):
    """Did an event of Mw >= thresh occur within R km and D days after?"""
    d = ordn - ord_m
    sel = (d > 0) & (d <= D) & (mw >= thresh)
    if not sel.any():
        return False
    idx = np.flatnonzero(sel)
    return bool((haversine_vec(lat_m, lon_m, lat[idx], lon[idx]) <= R).any())


def main():
    rule("PERAMALAN JENDELA: WILAYAH + 10-15 HARI, APA YANG BISA DIBERIKAN")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk")) if e.depth <= MAXDEPTH]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev])
    lon = np.array([e.lon for e in ev])
    keep = mw >= 5.0
    mw, ordn, lat, lon = mw[keep], ordn[keep], lat[keep], lon[keep]

    main_i = np.flatnonzero(mw >= MAIN_MW)
    span = ordn.max() - ordn.min()
    print(f"""
  katalog   GCMT dangkal Mw>=5,0 : {mw.size:,}
  induk     Mw>={MAIN_MW}              : {main_i.size:,}
  rentang   {datetime.fromordinal(int(ordn.min())).date()} .. """
          f"""{datetime.fromordinal(int(ordn.max())).date()}
""")

    for thresh, label in [(5.0, "M>=5,0"), (6.0, "M>=6,0")]:
        rule(f"PELUANG ADA {label} SUSULAN SETELAH GEMPA INDUK M>=6,0")
        print(f"\n  {'radius':>7} {'jendela':>9} {'setelah induk':>15} "
              f"{'hari biasa':>12} {'kelipatan':>11}")
        print("  " + "-" * 60)
        for R in RADII:
            for D in WINDOWS:
                hit = np.array([
                    any_within(ordn[i], lat[i], lon[i], ordn, lat, lon, mw,
                               thresh, R, D)
                    for i in main_i])
                p_after = hit.mean()

                # baseline: same places, calendar shifted -- what a random day
                # at the same location would have given
                base = []
                for _ in range(N_SHIFT):
                    off = RNG.integers(60, span - 60)
                    sub = RNG.choice(main_i, size=min(300, main_i.size),
                                     replace=False)
                    t = ordn.min() + (ordn[sub] - ordn.min() + off) % span
                    base.append(np.mean([
                        any_within(t[k], lat[j], lon[j], ordn, lat, lon, mw,
                                   thresh, R, D)
                        for k, j in enumerate(sub)]))
                p_base = float(np.mean(base))
                gain = p_after / p_base if p_base > 0 else np.inf
                print(f"  {R:>5} km {D:>7} hr {100*p_after:>14.1f}% "
                      f"{100*p_base:>11.1f}% {gain:>10.1f}x")

    # ---------------------------------------------------------------- cakupan
    rule("CAKUPAN: berapa banyak gempa besar yang BISA diperingatkan begini?")
    print("""
  Sebuah gempa hanya bisa masuk jendela seperti ini kalau ia DIDAHULUI gempa
  besar lain di dekatnya. Yang datang tanpa pendahulu tidak tersentuh metode
  ini sama sekali, berapa pun bagus modelnya.
""")
    print(f"  {'radius':>7} {'jendela':>9} {'didahului M>=6,0':>18} {'tidak':>10}")
    print("  " + "-" * 50)
    for R in RADII:
        for D in WINDOWS:
            pre = 0
            for i in main_i:
                d = ordn[i] - ordn
                sel = (d > 0) & (d <= D) & (mw >= MAIN_MW)
                if sel.any():
                    j = np.flatnonzero(sel)
                    if (haversine_vec(lat[i], lon[i], lat[j], lon[j]) <= R).any():
                        pre += 1
            f = pre / main_i.size
            print(f"  {R:>5} km {D:>7} hr {100*f:>17.1f}% {100*(1-f):>9.1f}%")

    rule("ARTINYA")
    print("""
  Dua angka itu harus dibaca bersama, dan keduanya jujur:

    KETERAMPILAN tinggi -- setelah gempa induk, peluang di wilayah itu naik
    berkali-kali lipat dibanding hari biasa. Itu nyata, terukur, dan sudah
    dipakai USGS maupun BMKG. Peringatan "waspada 2 minggu ke depan di radius
    100 km" adalah produk yang sah dan berguna.

    CAKUPAN kecil -- hanya sebagian kecil gempa besar yang didahului gempa
    besar lain di dekatnya. Sisanya datang tanpa pendahulu, dan untuk mereka
    metode ini tidak memberi apa-apa.

  Jadi jawabannya bukan ya atau tidak, melainkan: BISA, untuk gempa yang
  mengikuti gempa lain. Dan itu bukan prediksi gempa -- itu peramalan susulan,
  yang memang sudah berjalan hari ini tanpa perlu bantuan langit.
""")


if __name__ == "__main__":
    main()
