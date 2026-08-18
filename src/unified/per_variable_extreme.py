"""
Extremeness, variable by variable -- not pooled into one signature.

signature_test.py asked whether MANY variables are extreme together on
earthquake days, and the answer was no. That test pools: it counts across all
130 columns, so a real effect confined to one or two variables is diluted by
the 128 that carry nothing. This file asks the same question one variable at a
time, which is the version with power against a narrow effect.

It is also not a repeat of unified_test.py. That test bins each variable into
deciles and takes the largest single-decile departure from the base rate. Here
the two TAILS are combined:

    ATAS    the variable is in its top decile
    BAWAH   the variable is in its bottom decile
    EKOR    either tail -- extreme in absolute terms, in either direction

The third form is the one with the extra power, and it is the physically
motivated one. A stress that triggers should do so at both ends: peak
compression and peak extension are both departures from the mean state, and a
test that looks only for "high values raise the rate" would split a symmetric
effect across two deciles and find neither convincing. Reporting the one-tailed
forms alongside it shows the direction whenever something does appear.

    statistic  P(extreme | earthquake day) - P(extreme | quiet day)
    ranks      extremeness is defined by rank, not by standard deviations,
               because many of these variables are cyclic or heavy-tailed and
               a mean does not describe them
    null       circular time shift of the seismicity series
    correction max-statistic over all 130 variables x 3 forms, so searching 390
               ways costs exactly what searching 390 ways should cost
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from forcing_bank import build_all

N_SHIFT = 2000
EXTREME = 0.10
CHUNK = 200
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    rule("KEUNIKAN PER VARIABEL: tiap metrik diuji SENDIRI-SENDIRI")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y = (cnt > 0).astype(np.float64)
    n = y.size
    print(f"""
  katalog  global M>=6,0, {datetime.fromordinal(int(days[0])).date()} .. """
          f"""{datetime.fromordinal(int(days[-1])).date()}
  hari     {n:,}   laju dasar {100*y.mean():.2f}%
  ekstrem  desil teratas / terbawah ({100*EXTREME:.0f}%)
""")
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)
    m = X.shape[1]

    R = np.empty_like(X, dtype=np.float64)
    for j in range(m):
        R[:, j] = (np.argsort(np.argsort(X[:, j])) + 0.5) / n

    hi = (R > 1 - EXTREME)
    lo = (R < EXTREME)
    E = np.concatenate([hi, lo, hi | lo], axis=1).astype(np.float64)
    form = np.array(["ATAS"] * m + ["BAWAH"] * m + ["EKOR"] * m)
    vname = np.concatenate([names, names, names])
    vfam = np.concatenate([fams, fams, fams])
    print(f"\n  {m} variabel x 3 bentuk = {E.shape[1]} uji")

    def contrast(Y):
        """P(extreme | quake) - P(extreme | quiet), for every column at once."""
        nq = Y.sum(axis=0)
        a = (E.T @ Y) / nq
        b = (E.T @ (1.0 - Y)) / (n - nq)
        return a - b

    obs = contrast(y[:, None]).ravel()

    null = np.empty((N_SHIFT, E.shape[1]))
    for s in range(0, N_SHIFT, CHUNK):
        k = min(CHUNK, N_SHIFT - s)
        Y = np.empty((n, k))
        for i in range(k):
            Y[:, i] = np.roll(y, int(RNG.integers(1, n)))
        null[s:s + k] = contrast(Y).T
        print(f"    null {s+k}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    mu, sd = null.mean(axis=0), null.std(axis=0)
    sd = np.where(sd > 0, sd, 1)
    z_obs = np.abs(obs - mu) / sd
    z_null = np.abs(null - mu) / sd

    rule("HASIL PER KELUARGA  (dikoreksi untuk variabel x bentuk keluarga itu)")
    print(f"\n  {'keluarga':<12} {'n uji':>6} {'z maks':>8} {'p':>8}  "
          f"{'bentuk':<7} variabel terbaik")
    print("  " + "-" * 72)
    for fam in dict.fromkeys(fams.tolist()):
        sel = np.flatnonzero(vfam == fam)
        b = int(sel[np.argmax(z_obs[sel])])
        mn = z_null[:, sel].max(axis=1)
        p = float((mn >= z_obs[b]).mean())
        print(f"  {fam:<12} {sel.size:>6} {z_obs[b]:>8.2f} {p:>8.3f}  "
              f"{form[b]:<7} {vname[b][:26]}"
              f"{'  <-- LOLOS' if p < 0.05 else ''}")

    rule("SEPULUH VARIABEL TERDEKAT, apa pun keluarganya")
    print(f"""
  'selisih' = peluang variabel itu ekstrem pada hari gempa DIKURANGI peluangnya
  ekstrem pada hari tenang. Nol berarti ekstremnya tidak ada hubungan dengan
  gempa. 'p sendiri' belum dikoreksi -- ia diberikan supaya terlihat bahwa
  angka mentah pun tidak mencolok.
""")
    print(f"  {'variabel':<28} {'bentuk':<7} {'selisih':>9} {'z':>6} "
          f"{'p sendiri':>10}")
    print("  " + "-" * 66)
    for b in np.argsort(-z_obs)[:10]:
        p_alone = float((np.abs(null[:, b] - mu[b]) >= abs(obs[b] - mu[b])).mean())
        print(f"  {vname[b][:28]:<28} {form[b]:<7} {100*obs[b]:>8.2f}% "
              f"{z_obs[b]:>6.2f} {p_alone:>10.3f}")

    rule("KOREKSI GABUNGAN: seluruh 130 variabel x 3 bentuk")
    b = int(np.argmax(z_obs))
    mn = z_null.max(axis=1)
    p_all = float((mn >= z_obs[b]).mean())
    print(f"""
    total uji         : {E.shape[1]}
    terbaik data asli : z = {z_obs[b]:.2f}  ({vname[b]}, {form[b]})
    terbaik pada null : rata-rata z = {mn.mean():.2f}, maks {mn.max():.2f}
    p TERKOREKSI      : {p_all:.3f}
""")

    rule("CATATAN")
    print(f"""
  Bentuk EKOR adalah yang paling berdaya kalau efeknya simetris: tegangan
  puncak tekan dan puncak tarik sama-sama menyimpang dari keadaan rata-rata,
  dan uji yang hanya mencari "nilai tinggi menaikkan laju" akan memecah efek
  simetris ke dua desil lalu menganggap keduanya tidak meyakinkan. Di sini
  keduanya digabung, sehingga kalau ada efek semacam itu, ia terlihat.

  Bandingkan dengan signature_test.py: di sana 130 variabel dijumlahkan, jadi
  efek yang hanya ada di satu-dua variabel akan tenggelam oleh 128 lainnya.
  Di sini tidak ada penjumlahan sama sekali -- tiap variabel berdiri sendiri,
  dan harga pencariannya dibayar lewat null yang menyapu seluruh {E.shape[1]} uji
  pada setiap pergeseran.
""")


if __name__ == "__main__":
    main()
