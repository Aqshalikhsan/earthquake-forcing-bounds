"""
Do many forcings peak AT ONCE when an earthquake happens? A different question.

Everything so far has asked one of two things. The decile tests asked, of each
variable separately, whether earthquakes prefer high values of it. The
classifiers asked whether some boundary through the variable space separates
earthquake days from the rest. Neither asks what is probably the most natural
version of the idea:

    on the day an earthquake strikes, is there something UNUSUAL about the
    combined state of everything at once -- several forcings extreme together,
    a coincidence that ordinary days do not show?

That is not the same question, and a classifier does not necessarily answer it.
A gradient-boosted tree or an LSTM looks for a region of space where the
earthquake rate is higher. A coincidence is different: it can appear anywhere in
the space, need not be the same variables each time, and shows up as an
earthquake day being ODD rather than being located somewhere in particular.

Four statistics, each a different meaning of "unusual":

    COINCIDENCE   how many of the 130 variables sit in their extreme decile
                  that day. This is the idea in its plainest form -- several
                  things peaking together.
    FAMILIES      how many of the 8 families have at least one variable
                  extreme, so that 80 planetary columns cannot dominate by
                  weight of numbers alone.
    PEAK          the single largest |z| that day: is an earthquake day the day
                  something hit a record?
    MAHALANOBIS   distance of the whole 130-dimensional state from its usual
                  position, with a shrunk covariance so correlated variables
                  are not counted repeatedly. This catches an unusual
                  COMBINATION even when no single variable is extreme.

    null          circular time shift of the seismicity, which preserves
                  clustering, seasonality and the trend in catalogue growth and
                  destroys only the alignment with the forcings
    correction    the null is searched across all four statistics exactly as
                  hard as the real data, so trying four costs what it should
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from forcing_bank import build_all

N_SHIFT = 2000
EXTREME = 0.10           # top/bottom decile counts as "extreme"
SHRINK = 0.20            # covariance shrinkage toward the diagonal
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    rule("TANDA TANGAN GABUNGAN: apakah banyak metrik memuncak BERSAMAAN?")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y = (cnt > 0).astype(float)
    n = y.size
    print(f"""
  katalog  global M>=6,0, {datetime.fromordinal(int(days[0])).date()} .. """
          f"""{datetime.fromordinal(int(days[-1])).date()}
  hari     {n:,}   laju dasar {100*y.mean():.2f}%
""")
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)
    m = X.shape[1]
    print(f"\n  {m} variabel dari {len(set(fams))} keluarga")

    # standardise by rank, not by mean and sd: many of these variables are
    # cyclic or heavy-tailed, and "extreme" should mean "rare for this
    # variable", not "far from a mean that may not describe it
    R = np.empty_like(X, dtype=np.float64)
    for j in range(m):
        v = X[:, j]
        order = np.argsort(np.argsort(v))
        R[:, j] = (order + 0.5) / n                      # uniform 0..1
    extreme = (R < EXTREME) | (R > 1 - EXTREME)

    coincide = extreme.sum(axis=1).astype(float)
    fam_hit = np.zeros(n)
    for fam in dict.fromkeys(fams.tolist()):
        sel = np.flatnonzero(fams == fam)
        fam_hit += extreme[:, sel].any(axis=1)

    Z = np.sqrt(2) * np.abs(R - 0.5) * 2                 # 0 centre, 1 extreme
    peak = Z.max(axis=1)

    # Mahalanobis on a shrunk covariance. With 130 correlated columns the raw
    # covariance is near-singular and its inverse would amplify noise; shrinking
    # toward the diagonal keeps it invertible and keeps the distance meaningful.
    C = np.cov(Z.T)
    C = (1 - SHRINK) * C + SHRINK * np.diag(np.diag(C))
    C += 1e-6 * np.eye(m)
    Ci = np.linalg.inv(C)
    d = Z - Z.mean(axis=0)
    maha = np.einsum("ij,jk,ik->i", d, Ci, d)

    stats = {"KEBETULAN (n var ekstrem)": coincide,
             "KELUARGA (n keluarga ekstrem)": fam_hit,
             "PUNCAK (|z| terbesar)": peak,
             "MAHALANOBIS (jarak gabungan)": maha}

    print(f"""
  KEBETULAN   rata-rata {coincide.mean():.1f} dari {m} variabel ekstrem per hari
  KELUARGA    rata-rata {fam_hit.mean():.2f} dari {len(set(fams))} keluarga
  MAHALANOBIS median {np.median(maha):.1f}
""")

    def contrast(v, yy):
        """Mean on earthquake days minus mean on quiet days."""
        a, b = yy > 0, yy == 0
        return float(v[a].mean() - v[b].mean())

    obs = np.array([contrast(v, y) for v in stats.values()])
    null = np.empty((N_SHIFT, len(stats)))
    for i in range(N_SHIFT):
        ys = np.roll(y, int(RNG.integers(1, n)))
        null[i] = [contrast(v, ys) for v in stats.values()]
        if (i + 1) % 250 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    mu, sd = null.mean(axis=0), null.std(axis=0)
    sd = np.where(sd > 0, sd, 1)
    z_obs = np.abs(obs - mu) / sd
    z_null = np.abs(null - mu) / sd

    rule("HASIL")
    print(f"""
  'hari gempa' dan 'hari tenang' adalah rata-rata statistik itu di kedua
  kelompok. 'p' sudah terkoreksi untuk keempat statistik sekaligus.
""")
    print(f"  {'statistik':<30} {'hari gempa':>11} {'hari tenang':>12} "
          f"{'z':>6} {'p':>8}")
    print("  " + "-" * 72)
    mn = z_null.max(axis=1)
    for k, (name, v) in enumerate(stats.items()):
        a, b = y > 0, y == 0
        p = float((mn >= z_obs[k]).mean())
        print(f"  {name:<30} {v[a].mean():>11.2f} {v[b].mean():>12.2f} "
              f"{z_obs[k]:>6.2f} {p:>8.3f}"
              f"{'  <-- LOLOS' if p < 0.05 else ''}")

    rule("UJI EKOR: apakah hari 'paling aneh' lebih sering bergempa?")
    print(f"""
  Kalau tanda tangannya nyata, ia paling terlihat di ujung: hari dengan
  paling banyak variabel ekstrem seharusnya paling sering bergempa.
""")
    print(f"  {'persentil':<14} {'ambang':>9} {'n hari':>9} {'laju gempa':>12} "
          f"{'vs dasar':>10}")
    print("  " + "-" * 60)
    base = y.mean()
    for q in (50, 75, 90, 95, 99):
        t = np.percentile(coincide, q)
        sel = coincide >= t
        if sel.sum() < 50:
            continue
        r = y[sel].mean()
        print(f"  >= p{q:<11} {t:>9.0f} {int(sel.sum()):>9,} "
              f"{100*r:>11.2f}% {r/base:>9.3f}x")

    rule("VONIS")
    best = float(np.min([(mn >= z).mean() for z in z_obs]))
    print(f"""
  p terkecil setelah koreksi : {best:.3f}
  {'ADA tanda tangan gabungan yang lolos.' if best < 0.05
   else 'TIDAK ADA tanda tangan gabungan yang lolos.'}

  Uji ini menanyakan hal yang tidak ditanyakan classifier. Pohon keputusan dan
  LSTM mencari WILAYAH di ruang variabel tempat laju gempa lebih tinggi.
  Kebetulan berbeda sifatnya: ia bisa muncul di mana saja di ruang itu, tidak
  harus melibatkan variabel yang sama tiap kali, dan tampak sebagai hari yang
  ANEH -- bukan hari yang berada di suatu tempat tertentu.

  Kolom 'vs dasar' di uji ekor adalah bacaan yang paling langsung: kalau hari
  dengan paling banyak metrik ekstrem tidak lebih sering bergempa daripada hari
  biasa, maka tidak ada kebetulan yang bisa dijadikan dasar peringatan.
""")


if __name__ == "__main__":
    main()
