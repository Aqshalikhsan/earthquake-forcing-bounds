"""
Does the b-value drop before a large earthquake? The credible precursor, tested.

This is not a ninth forcing family and it does not belong with them. The eight
families are things outside the earthquake system -- the Moon, the planets,
water, air. The b-value is computed FROM the earthquake catalogue itself, so it
belongs with ETAS, on the reference side of the comparison.

It is here because it is the precursor that mainstream seismology takes most
seriously. The Gutenberg-Richter law says the number of earthquakes falls off
with magnitude as 10^(-b*M). Laboratory rock mechanics finds b falling as
differential stress rises, and several studies report b dropping in a fault
patch in the months before it ruptures. If anything in this project is going to
survive a calibrated test, the prior odds favour this over planetary geometry
by a wide margin.

    b-value    maximum-likelihood Aki-Utsu estimator on the events inside a
               radius, with the completeness magnitude fixed rather than fitted
    candidates shallow Mw >= 6.0 mainshocks, Gardner-Knopoff declustered
    controls   the same places at other times, so geography cancels: a region
               with a permanently low b-value must not be scored as though it
               were warning of something
    null       whole-year circular offsets, which hold the season and the
               catalogue's growth in station coverage fixed

WHY THE CONTROL MATTERS MORE THAN THE STATISTIC. b varies strongly from place
to place -- it is high in volcanic and geothermal crust, low in cold
subduction. Comparing pre-earthquake b against a global average would rediscover
that geography and call it a precursor. Every comparison here is therefore
between the same location before an earthquake and that same location at other
times.

The catalogue also imposes a hard limit that must be stated: Global CMT is
complete to roughly Mw 5.0-5.5, so a b-value from it rests on the handful of
magnitudes between completeness and the mainshock. Regional catalogues reaching
M2-3 would measure b far better. A null here therefore bounds what is visible
in a global moment-tensor catalogue, not what a dense local network might see.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec

MC = 5.2                    # completeness: fixed, not fitted per window
MAXDEPTH = 70.0
TARGET_MW = 6.0
RADII = [150.0, 300.0]
WINDOWS = [90, 180, 365]    # days before, over which b is measured
MIN_EV = 12                 # fewest events for a usable b
N_SHIFT = 300
JITTER = 15
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def b_aki_utsu(mags, mc=MC, dm=0.1):
    """Maximum-likelihood b. dm is the magnitude bin correction."""
    m = mags[mags >= mc]
    if m.size < MIN_EV:
        return np.nan
    mean = m.mean()
    if mean - (mc - dm / 2) <= 1e-6:
        return np.nan
    return float(np.log10(np.e) / (mean - (mc - dm / 2)))


def main():
    rule("NILAI-B SEBELUM GEMPA BESAR: prekursor yang paling kredibel")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk")) if e.depth <= MAXDEPTH]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev]); lon = np.array([e.lon for e in ev])

    # targets: declustered mainshocks, so nothing is scored for an aftershock
    big = np.flatnonzero(mw >= TARGET_MW)
    dead = np.zeros(big.size, dtype=bool)
    ob, mb, lab, lob = ordn[big], mw[big], lat[big], lon[big]
    for k in np.argsort(-mb):
        if dead[k]:
            continue
        dkm, ddays = gardner_knopoff_window(mb[k])
        d = ob - ob[k]
        c = (d > 0) & (d <= ddays) & (mb <= mb[k]) & (~dead)
        i = np.flatnonzero(c)
        if i.size:
            dead[i[haversine_vec(lab[k], lob[k], lab[i], lob[i]) <= dkm]] = True
    tgt = big[~dead]

    usable = mw >= MC
    print(f"""
  katalog     GCMT dangkal              : {mw.size:,}
  di atas Mc={MC}                        : {int(usable.sum()):,}
  gempa induk Mw>={TARGET_MW} (declustered) : {tgt.size:,}
  rentang     {datetime.fromordinal(int(ordn.min())).date()} .. """
          f"""{datetime.fromordinal(int(ordn.max())).date()}

  Mc dipatok {MC}, tidak dicocokkan per jendela -- Mc yang dicocokkan akan
  bergerak bersama jumlah gempa dan menciptakan tren b palsu.
""")

    om, mm, lam, lom = ordn[usable], mw[usable], lat[usable], lon[usable]

    #
    # The neighbourhood of each target is FIXED. Shifting the calendar changes
    # which of its neighbours fall inside the window, never which events are
    # nearby. Recomputing haversine on every null replicate would cost
    # 3,700 targets x 19,000 events x 300 shifts x 6 settings -- of order 1e11
    # distance evaluations, which does not finish. The neighbour lists are
    # therefore built once per radius, sorted by time, and each replicate is
    # reduced to two binary searches per target.
    #
    neigh = {}

    def neighbours(R):
        if R in neigh:
            return neigh[R]
        idx, tim, mag = [], [], []
        for k in range(la_t.size):
            n = np.flatnonzero(haversine_vec(la_t[k], lo_t[k], lam, lom) <= R)
            o = n[np.argsort(om[n])]
            idx.append(o); tim.append(om[o]); mag.append(mm[o])
            if (k + 1) % 500 == 0:
                print(f"    tetangga R={R:.0f} km: {k+1}/{la_t.size}",
                      end="\r", flush=True)
        print(" " * 46, end="\r")
        neigh[R] = (tim, mag)
        return neigh[R]

    def b_before(times, R, W):
        """b-value in the W days before each time, within R km of the target."""
        tim, mag = neighbours(R)
        out = np.full(times.size, np.nan)
        for k in range(times.size):
            t = tim[k]
            if t.size < MIN_EV:
                continue
            lo_i = np.searchsorted(t, times[k] - W, side="left")
            hi_i = np.searchsorted(t, times[k], side="left")
            if hi_i - lo_i >= MIN_EV:
                out[k] = b_aki_utsu(mag[k][lo_i:hi_i])
        return out

    la_t, lo_t, or_t = lat[tgt], lon[tgt], ordn[tgt]
    span = int(ordn.max() - ordn.min())
    n_year = max(int(span // 365.25) - 1, 1)
    base_ord = ordn.min()

    rule("HASIL")
    print(f"""
  'b sebelum'  rata-rata nilai-b di jendela sebelum gempa induk
  'b kontrol'  tempat yang SAMA, waktu lain ({n_year} offset tahunan +/-{JITTER} hari)
  'p'          dua sisi, dari sebaran kontrol
""")
    print(f"  {'radius':>7} {'jendela':>8} {'n':>6} {'b sebelum':>10} "
          f"{'b kontrol':>10} {'selisih':>9} {'p':>8}")
    print("  " + "-" * 66)

    results = []
    for R in RADII:
        for W in WINDOWS:
            obs = b_before(or_t, R, W)
            ok = np.isfinite(obs)
            if ok.sum() < 50:
                print(f"  {R:>5.0f} km {W:>6}d  data tidak cukup")
                continue
            o = float(np.nanmean(obs))

            null = np.empty(N_SHIFT)
            for i in range(N_SHIFT):
                off = int(round(365.25 * RNG.integers(1, n_year + 1))
                          + RNG.integers(-JITTER, JITTER + 1))
                sh = base_ord + (or_t - base_ord + off) % span
                v = b_before(sh, R, W)
                null[i] = np.nanmean(v) if np.isfinite(v).sum() > 20 else np.nan
                if (i + 1) % 60 == 0:
                    print(f"    R={R:.0f} W={W} null {i+1}/{N_SHIFT}",
                          end="\r", flush=True)
            print(" " * 46, end="\r")
            nb = null[np.isfinite(null)]
            if nb.size < 30:
                print(f"  {R:>5.0f} km {W:>6}d  kontrol tidak cukup")
                continue
            c = float(nb.mean())
            p = float((np.abs(nb - c) >= abs(o - c)).mean())
            results.append((R, W, int(ok.sum()), o, c, p))
            print(f"  {R:>5.0f} km {W:>6}d {int(ok.sum()):>6,} {o:>10.3f} "
                  f"{c:>10.3f} {o-c:>+9.3f} {p:>8.3f}"
                  f"{'  <-- LOLOS' if p < 0.05/len(RADII)/len(WINDOWS) else ''}")

    alpha = 0.05 / max(len(RADII) * len(WINDOWS), 1)
    rule("VONIS")
    hit = any(r[5] < alpha for r in results)
    print(f"""
  Bonferroni atas {len(RADII)*len(WINDOWS)} kombinasi radius x jendela -> ambang p < {alpha:.4f}

  {'ADA penurunan nilai-b yang lolos koreksi.' if hit
    else 'TIDAK ADA yang lolos koreksi.'}

  Kalau nilai-b memang turun sebelum gempa, kolom 'selisih' harus NEGATIF dan
  konsisten di seluruh baris. Tanda yang berganti-ganti antar radius atau
  jendela adalah tanda derau, bukan fisika.

  Batas yang berlaku: ini menguji apa yang terlihat di katalog momen-tensor
  GLOBAL dengan Mc={MC}. Jaringan lokal yang lengkap sampai M2-3 mengukur b jauh
  lebih baik, dan nol di sini tidak menutup kemungkinan itu. Yang ditutup
  adalah klaim bahwa penurunan b terlihat secara global tanpa jaringan padat.

  Catatan kategori: nilai-b BUKAN gaya luar. Ia dihitung dari katalog gempa
  itu sendiri, jadi tempatnya di sisi pembanding bersama ETAS -- bukan di
  antara kedelapan keluarga yang diuji.
""")


if __name__ == "__main__":
    main()
