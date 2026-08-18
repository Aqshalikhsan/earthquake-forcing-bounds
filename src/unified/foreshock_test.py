"""
The last gap: can the sky tell a foreshock from an ordinary small earthquake?

Mainshocks and aftershocks have both been tested against all eight families and
both came back null. Foreshocks never were, and the question they raise is not
the same one. It is not "does forcing change how often small earthquakes
happen" -- it is:

    a magnitude 5 has just occurred. Almost always nothing follows. Sometimes a
    magnitude 6+ follows within days. Does the state of the sky separate the
    two cases?

This is the most favourable question left in the project, for three reasons.

    PHYSICS      a foreshock marks a fault that is already loaded and already
                 slipping in small increments. If a few hundred pascals ever
                 decide anything, this is the moment -- not whether a fault
                 breaks, but whether a break that has started runs away.
    OPERATIONS   it needs no new instrument. Foreshocks are detected now, and
                 anything that sharpened the follow-up probability would be
                 usable immediately.
    STATISTICS   the comparison is between two groups of earthquakes, so
                 everything that makes seismicity hard -- clustering, catalogue
                 growth, spatial concentration -- is present on BOTH sides and
                 largely cancels.

    candidates   shallow Mw >= 5.0 from Global CMT
    label        1 if followed by Mw >= 6.0 within RADIUS km and WINDOW days
    statistic    standardised difference in each forcing between the two groups
    null         the forcing series is circularly shifted against a FIXED set
                 of events and labels. Event times, clustering and the label
                 imbalance are all preserved exactly; only the alignment
                 between sky and seismicity is destroyed.

Shuffling the labels instead would be wrong. Foreshocks are not scattered at
random through the catalogue -- they sit next to mainshocks, in the same
regions and the same weeks -- so a label permutation would break that structure
and produce a null that is easier to beat than reality.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import haversine_vec
from forcing_bank import build_all

MAXDEPTH = 70.0
CAND_MW, TARGET_MW = 5.0, 6.0
RADIUS, WINDOW = 100.0, 15
N_SHIFT = 400
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def label_foreshocks(ordn, mw, lat, lon):
    """1 if an Mw >= TARGET_MW follows within RADIUS km and WINDOW days."""
    y = np.zeros(mw.size, dtype=np.int8)
    big = np.flatnonzero(mw >= TARGET_MW)
    ob, lab, lob = ordn[big], lat[big], lon[big]
    for k in range(mw.size):
        d = ob - ordn[k]
        sel = (d > 0) & (d <= WINDOW)
        if not sel.any():
            continue
        j = np.flatnonzero(sel)
        if (haversine_vec(lat[k], lon[k], lab[j], lob[j]) <= RADIUS).any():
            y[k] = 1
    return y


def main():
    rule("BISAKAH LANGIT MEMBEDAKAN FORESHOCK DARI GEMPA KECIL BIASA?")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= MAXDEPTH and e.mw >= CAND_MW]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev])
    lon = np.array([e.lon for e in ev])

    y = label_foreshocks(ordn, mw, lat, lon)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    print(f"""
  kandidat    Mw >= {CAND_MW} dangkal          : {mw.size:,}
  foreshock   disusul Mw >= {TARGET_MW} dalam
              {RADIUS:.0f} km / {WINDOW} hari              : {n1:,}  ({100*n1/mw.size:.1f}%)
  bukan                                : {n0:,}
  rentang     {datetime.fromordinal(int(ordn.min())).date()} .. """
          f"""{datetime.fromordinal(int(ordn.max())).date()}

  Jadi {100*n1/mw.size:.1f}% gempa Mw>={CAND_MW} ternyata foreshock. Pertanyaannya:
  apakah langit tahu yang mana, PADA HARI gempa kecil itu terjadi?
""")
    if n1 < 100:
        print("  terlalu sedikit foreshock untuk diuji.")
        return

    days = np.load(datafile("m6_days.npy"))
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)
    d0, span = int(days[0]), len(days)
    print(f"\n  {X.shape[1]} variabel, {len(set(fams))} keluarga")

    Z = (X - np.nanmean(X, axis=0)) / np.where(np.nanstd(X, axis=0) > 0,
                                               np.nanstd(X, axis=0), 1)
    Z = np.nan_to_num(Z)

    pos = (ordn - d0).astype(int)
    ok = (pos >= 0) & (pos < span)
    pos, yy = pos[ok], y[ok]
    a, b = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    print(f"  di dalam rentang bank: {int(ok.sum()):,} "
          f"({a.size:,} foreshock, {b.size:,} bukan)")

    def contrast(shift):
        """Standardised difference in mean forcing, foreshock minus rest."""
        p = (pos + shift) % span
        v = Z[p]
        return v[a].mean(axis=0) - v[b].mean(axis=0)

    obs = contrast(0)
    null = np.empty((N_SHIFT, X.shape[1]))
    for i in range(N_SHIFT):
        null[i] = contrast(int(RNG.integers(60, span - 60)))
        if (i + 1) % 50 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    mu, sd = null.mean(axis=0), null.std(axis=0)
    sd = np.where(sd > 0, sd, 1)
    z_obs = np.abs(obs - mu) / sd
    z_null = np.abs(null - mu) / sd

    rule("HASIL PER KELUARGA")
    print(f"""
  Membandingkan {a.size:,} foreshock dengan {b.size:,} gempa kecil biasa.
  'p' sudah terkoreksi untuk seluruh variabel dalam keluarga.
""")
    print(f"  {'keluarga':<12} {'n var':>6} {'z maks':>8} {'p':>8}  variabel terbaik")
    print("  " + "-" * 64)
    for fam in dict.fromkeys(fams.tolist()):
        sel = np.flatnonzero(fams == fam)
        m_o = float(z_obs[sel].max())
        m_n = z_null[:, sel].max(axis=1)
        p = float((m_n >= m_o).mean())
        top = names[int(sel[np.argmax(z_obs[sel])])]
        print(f"  {fam:<12} {sel.size:>6} {m_o:>8.2f} {p:>8.3f}  {top[:24]}"
              f"{'  <-- LOLOS' if p < 0.05 else ''}")

    rule("KOREKSI GABUNGAN")
    m_o = float(z_obs.max())
    m_n = z_null.max(axis=1)
    print(f"""
    total variabel    : {X.shape[1]}
    terbaik data asli : z = {m_o:.2f}  ({names[int(np.argmax(z_obs))]})
    terbaik pada null : rata-rata z = {m_n.mean():.2f}, maks {m_n.max():.2f}
    p TERKOREKSI      : {float((m_n >= m_o).mean()):.3f}
""")

    rule("SEBERAPA BESAR EFEK YANG MASIH MUNGKIN BERSEMBUNYI?")
    thr = np.percentile(m_n, 95)
    base = 100 * a.size / (a.size + b.size)
    print(f"""
  Ambang deteksi (persentil 95 null) : z = {thr:.2f}
  Peluang dasar sebuah Mw>={CAND_MW} adalah foreshock : {base:.1f}%

  Efek sebesar ambang itu akan menggeser peluang tersebut sekitar
  {base*thr/np.sqrt(a.size):.2f} poin persen. Apa pun yang lebih besar dari itu
  sudah pasti tertangkap, dan tidak tertangkap.
""")


if __name__ == "__main__":
    main()
