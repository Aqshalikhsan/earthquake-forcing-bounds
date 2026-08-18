"""
The precursor test proper: does any forcing at time t-k predict earthquakes at t?

Everything so far compared forcings to earthquakes at the same instant, or over
a short causal window chosen in advance. That answers "is there a relationship",
but not the question a precursor claim actually makes, which is "does the signal
come FIRST, and by how long".

This is also the test El Moudden et al. (2024) never really ran. Their features
and their earthquake counts share the same date, so nothing precedes anything.
They did try lagging once -- Section 4.1 reports "the worst results, such as the
RMSE of 15.76 and the negative R-Squared value" -- and then kept "precursor" in
the title.

Here the lag is swept openly for every family:

    lags      0, 1, 2, 3, 5, 7, 10, 14, 21, 30 days
    pairing   forcing at day (t - k) against earthquakes on day t
    statistic largest deviation of any decile's rate from the overall rate
    null      circular time shift of the seismicity series

THE LAG IS A FREE PARAMETER, AND IT IS PAID FOR. Sweeping ten lags across a
hundred variables is a thousand chances to find something. The max-statistic
null searches the whole (variable x lag) grid on every shifted dataset too, so
the reported p already includes the cost of the sweep. Without that, a lag sweep
is simply a machine for manufacturing precursors -- which is roughly what the
precursor literature has been running on.

Nothing wraps around: a lag of k discards the first k days rather than folding
the end of the record onto the start.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from forcing_bank import build_all

LAGS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
N_SHIFT = 300
N_BINS = 10
MIN_BIN = 300
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def bin_column(x, n_bins=N_BINS):
    e = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if e.size < 3:
        return None
    return np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2).astype(np.int16)


def stat(bins, y, n_bins=N_BINS):
    base = y.mean()
    cnt = np.bincount(bins, minlength=n_bins).astype(float)
    s = np.bincount(bins, weights=y, minlength=n_bins)
    ok = cnt >= MIN_BIN
    if not ok.any():
        return 0.0
    rates = np.where(ok, s / np.maximum(cnt, 1), base)
    return float(np.max(np.abs(rates - base)))


def main():
    rule("SAPUAN JEDA WAKTU: apakah ada gaya yang MENDAHULUI gempa?")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y_full = (cnt > 0).astype(float)

    print(f"""
  katalog  global M>=6,0, {datetime.fromordinal(int(days[0])).date()} .. """
          f"""{datetime.fromordinal(int(days[-1])).date()}
  hari     {len(y_full):,}   laju dasar {100*y_full.mean():.2f}%
  jeda     {LAGS} hari
""")
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)
    n_var = X.shape[1]
    print(f"\n  {n_var} variabel x {len(LAGS)} jeda = {n_var*len(LAGS):,} uji")

    # Pre-bin every (variable, lag) pair once. Lag k pairs X[t-k] with y[t],
    # so the first k days are dropped from both.
    maxlag = max(LAGS)
    y = y_full[maxlag:]
    cells = []          # (family, name, lag, bins)
    for k in LAGS:
        lo = maxlag - k
        hi = len(y_full) - k
        for j in range(n_var):
            b = bin_column(X[lo:hi, j])
            if b is not None and len(b) == len(y):
                cells.append((fams[j], names[j], k, b))
    print(f"  pasangan valid: {len(cells):,}   hari dipakai: {len(y):,}")

    obs = np.array([stat(b, y) for _, _, _, b in cells])

    null = np.empty((N_SHIFT, len(cells)))
    for i in range(N_SHIFT):
        ys = np.roll(y, int(RNG.integers(1, len(y))))
        for c, (_, _, _, b) in enumerate(cells):
            null[i, c] = stat(b, ys)
        if (i + 1) % 50 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    fam_arr = np.array([c[0] for c in cells])
    lag_arr = np.array([c[2] for c in cells])
    nm_arr = np.array([c[1] for c in cells])

    rule("HASIL PER KELUARGA  (jeda terbaik dari sapuan penuh)")
    print(f"""
  'jeda'     jeda hari terbaik yang ditemukan di keluarga itu
  'terukur'  modulasi laju pada jeda tersebut
  'p'        sudah terkoreksi untuk SELURUH sapuan variabel x jeda
""")
    print(f"  {'keluarga':<12} {'jeda':>5} {'terukur':>9} {'ambang':>8} {'p':>7}  variabel")
    print("  " + "-" * 78)
    for fam in dict.fromkeys(fam_arr.tolist()):
        sel = np.flatnonzero(fam_arr == fam)
        if sel.size == 0:
            continue
        b = sel[int(np.argmax(obs[sel]))]
        nb = null[:, sel].max(axis=1)
        p = float((nb >= obs[b]).mean())
        thr = np.percentile(nb, 95)
        flag = " <-- LOLOS" if p < 0.05 else ""
        print(f"  {fam:<12} {lag_arr[b]:>4}d {100*obs[b]:>8.2f}% {100*thr:>7.2f}% "
              f"{p:>7.3f}  {nm_arr[b][:24]}{flag}")

    rule("KOREKSI GABUNGAN: seluruh keluarga x seluruh jeda")
    b = int(np.argmax(obs))
    nb = null.max(axis=1)
    p_all = float((nb >= obs[b]).mean())
    print(f"""
    total uji         : {len(cells):,}  ({n_var} variabel x {len(LAGS)} jeda)
    terbaik data asli : {100*obs[b]:.2f}%  ({nm_arr[b]}, jeda {lag_arr[b]} hari)
    terbaik pada null : rata-rata {100*nb.mean():.2f}%, maks {100*nb.max():.2f}%
    p TERKOREKSI      : {p_all:.3f}
""")

    rule("APAKAH JEDA TERTENTU LEBIH MENJANJIKAN?")
    print(f"\n  {'jeda':>6} {'terbaik':>10} {'ambang 95%':>12}  keluarga")
    print("  " + "-" * 56)
    for k in LAGS:
        sel = np.flatnonzero(lag_arr == k)
        if sel.size == 0:
            continue
        b = sel[int(np.argmax(obs[sel]))]
        thr = np.percentile(null[:, sel].max(axis=1), 95)
        print(f"  {k:>5}d {100*obs[b]:>9.2f}% {100*thr:>11.2f}%  {fam_arr[b]}")

    rule("CATATAN")
    print(f"""
  Jeda adalah parameter bebas, dan di sini dibayar penuh: null menyapu seluruh
  {len(cells):,} kombinasi variabel x jeda pada setiap pergeseran, sama kerasnya
  dengan pencarian pada data asli. Tanpa itu, sapuan jeda hanyalah mesin
  pencetak prekursor -- sepuluh jeda memberi sepuluh kesempatan tambahan untuk
  menemukan p<0,05 secara kebetulan.

  Perbandingan langsung: El Moudden dkk. mencoba menggeser jendela sekali,
  mendapat R-squared negatif, melaporkannya dalam satu kalimat, dan
  mempertahankan kata 'precursor' di judul.
""")


if __name__ == "__main__":
    main()
