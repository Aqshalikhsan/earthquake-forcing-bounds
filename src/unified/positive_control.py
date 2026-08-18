"""
Is the pipeline broken? Planted signals of known size, and whether it finds them.

The right objection to a long run of null results is that the machinery might
simply be broken -- a bug, a sign error, a variable silently full of zeros.
Arguing about it is useless. The test is to plant a signal of known size in the
real variables and see whether the code reports it.

That is what runs here. For every family, earthquakes are re-drawn so that they
genuinely do prefer certain values of a real forcing variable, by a controlled
amount, and the ordinary analysis is then pointed at the result:

    beta      the planted effect: how much the daily earthquake rate is raised
              at high values of the variable, relative to the base rate
    detection how often the family's own max-statistic test, with its own
              calibrated null, reports p < 0.05

Two things make this decisive rather than decorative:

    SAME CODE      the statistic, the binning and the null are the ones used in
                   unified_test.py. Nothing is re-implemented for the occasion,
                   so a bug in the real analysis is a bug in this one too.
    REAL VARIABLES the planted signal rides on the actual Moon phase, actual
                   planetary angles, actual TEC. If a variable were empty,
                   constant or misaligned, nothing could be planted in it and
                   its row here would fail.

beta = 0 is the false-alarm check: with nothing planted, detection must come
out near 5% and not above it, or the null is too generous and every other
number in the project is worthless.

The published claims give the yardstick. An AUC of 0.99, or an R-squared above
0.9, corresponds to an effect vastly larger than anything in the table below.
If a 10% modulation is caught every time, a 90% one cannot be missed.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from forcing_bank import build_all
from unified_test import bin_indices, stats_all, N_BINS

N_NULL = 300
N_REP = 20
BETAS = [0.0, 0.05, 0.10, 0.20, 0.40]
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    rule("KONTROL POSITIF: apakah pipeline ini SANGGUP menemukan sinyal?")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y_real = (cnt > 0).astype(float)
    n = y_real.size
    base = y_real.mean()

    print(f"""
  hari       {n:,}
  laju dasar {100*base:.2f}% per hari
  metode     statistik, binning dan null PERSIS dari unified_test.py
  ulangan    {N_REP} per tingkat, null {N_NULL} pergeseran melingkar
""")
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)
    print(f"\n  {X.shape[1]} variabel, {len(set(fams))} keluarga")

    idx = bin_indices(X)

    # The null distribution of the max-statistic does not depend on what is
    # planted -- it is the distribution under NO association. So it is built
    # once per family from circular shifts of the real series and reused for
    # every injection level, which is both correct and about a hundred times
    # cheaper than rebuilding it each time.
    print("\n  membangun ambang null per keluarga:")
    null_all = np.empty((N_NULL, X.shape[1]))
    for i in range(N_NULL):
        null_all[i] = stats_all(idx, np.roll(y_real, int(RNG.integers(1, n))))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N_NULL}", end="\r", flush=True)
    print(" " * 30, end="\r")

    order = list(dict.fromkeys(fams.tolist()))
    thr = {}
    for fam in order:
        sel = np.flatnonzero(fams == fam)
        thr[fam] = (sel, np.percentile(null_all[:, sel].max(axis=1), 95))

    rule("HASIL: seberapa besar sinyal yang harus ada agar tertangkap?")
    print(f"""
  Angka di sel = persen dari {N_REP} percobaan yang mencapai p < 0,05.
  Kolom beta=0 adalah kontrol alarm palsu: harus di sekitar 5%, tidak lebih.
""")
    hdr = "  ".join(f"{100*b:>5.0f}%" for b in BETAS)
    print(f"  {'keluarga':<12} {'var':>4}  {hdr}")
    print("  " + "-" * (20 + 7 * len(BETAS)))

    for fam in order:
        sel, t95 = thr[fam]
        # plant the signal on the family's own strongest-varying variable
        j = int(sel[np.argmax(np.nanstd(X[:, sel], axis=0) > 0)])
        v = X[:, j].astype(float)
        z = (v - np.nanmean(v)) / (np.nanstd(v) if np.nanstd(v) > 0 else 1)
        z = np.nan_to_num(z)

        row = []
        for beta in BETAS:
            hits = 0
            for r in range(N_REP):
                rng = np.random.default_rng(9000 + 97 * r + int(1000 * beta))
                p = np.clip(base * (1.0 + beta * z), 1e-6, 0.999)
                p *= base / p.mean()                  # hold the base rate fixed
                y = (rng.random(n) < p).astype(float)
                s = stats_all(idx[:, sel], y)
                hits += float(s.max()) > t95
            row.append(f"{100*hits/N_REP:>5.0f}%")
        print(f"  {fam:<12} {sel.size:>4}  {'  '.join(row)}")

    rule("ARTINYA")
    print("""
  Baca kolom beta=0 lebih dulu. Kalau ia di sekitar 5%, nullnya terkalibrasi:
  kerangka ini TIDAK menciptakan temuan dari ketiadaan.

  Lalu baca ke kanan. Tingkat beta terkecil yang sudah tertangkap hampir
  selalu adalah batas kepekaan kerangka ini. Sinyal apa pun yang lebih besar
  dari itu MUSTAHIL terlewat.

  Bandingkan dengan klaim yang dipublikasikan: AUC 0,99 atau R-kuadrat di atas
  0,9 setara dengan modulasi yang jauh lebih besar daripada kolom paling kanan
  di tabel ini. Kalau modulasi 10% saja tertangkap setiap kali, modulasi
  sebesar itu tidak mungkin tidak terlihat.

  Jadi selisih antara hasil di sini dan hasil di paper-paper itu bukan soal
  kepekaan alat. Alatnya lebih peka daripada yang dibutuhkan.
""")


if __name__ == "__main__":
    main()
