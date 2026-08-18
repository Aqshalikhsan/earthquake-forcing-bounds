"""
Why the "best lag" column is not a prediction. Shown, not argued.

The lag sweep prints a table that looks like a discovery:

    TIDAL       10d   3.37%
    ROTATION    21d   7.05%
    LUNAR       14d   3.20%

Read quickly, that says tidal stress leads earthquakes by ten days and the
length of day leads them by three weeks. It says nothing of the kind, and the
reason is simple: EVERY family is guaranteed to have a best lag, because a
maximum of ten numbers always exists. The column reports where the largest
value happened to land, not whether that value means anything.

Two demonstrations, both on the real machinery:

    NOISE       the same table, computed on seismicity that has been circularly
                shifted so no real relationship can survive. If the shifted
                table looks just like the real one, the format is producing the
                impression of a result, not the data.

    STABILITY   the catalogue is split into two halves and the sweep is run on
                each. A real lead time is a property of the physics and must
                appear in both halves. A lag picked out of noise will not
                reproduce, because the noise differs between halves.

The second test is the decisive one and it is cheap. Nobody who reports a
precursor lag runs it.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from forcing_bank import build_all
from lag_sweep import LAGS, bin_column, stat, N_BINS

RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def best_per_family(cells, obs, fam_arr, lag_arr, nm_arr):
    out = {}
    for fam in dict.fromkeys(fam_arr.tolist()):
        sel = np.flatnonzero(fam_arr == fam)
        b = sel[int(np.argmax(obs[sel]))]
        out[fam] = (int(lag_arr[b]), float(obs[b]), str(nm_arr[b]))
    return out


def show(title, table, order):
    print(f"\n  {title}")
    print(f"  {'keluarga':<12} {'jeda':>6} {'terukur':>9}  variabel")
    print("  " + "-" * 60)
    for fam in order:
        if fam not in table:
            continue
        lag, val, nm = table[fam]
        print(f"  {fam:<12} {lag:>5}d {100*val:>8.2f}%  {nm[:28]}")


def main():
    rule("KENAPA KOLOM 'JEDA' BUKAN PREDIKSI HARI")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y_full = (cnt > 0).astype(float)
    print(f"""
  katalog  global M>=6,0, {datetime.fromordinal(int(days[0])).date()} .. """
          f"""{datetime.fromordinal(int(days[-1])).date()}
  hari     {len(y_full):,}   laju dasar {100*y_full.mean():.2f}%
""")
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)

    maxlag = max(LAGS)
    y = y_full[maxlag:]
    cells = []
    for k in LAGS:
        lo, hi = maxlag - k, len(y_full) - k
        for j in range(X.shape[1]):
            b = bin_column(X[lo:hi, j])
            if b is not None and len(b) == len(y):
                cells.append((fams[j], names[j], k, b))
    fam_arr = np.array([c[0] for c in cells])
    lag_arr = np.array([c[2] for c in cells])
    nm_arr = np.array([c[1] for c in cells])
    order = list(dict.fromkeys(fam_arr.tolist()))
    print(f"\n  {len(cells):,} pasangan (variabel x jeda)")

    obs = np.array([stat(b, y) for _, _, _, b in cells])
    real = best_per_family(cells, obs, fam_arr, lag_arr, nm_arr)

    rule("UJI 1 — TABEL YANG SAMA, TAPI DARI DERAU MURNI")
    print("""
  Seismisitas digeser melingkar, sehingga tidak mungkin ada hubungan nyata
  yang tersisa. Kalau tabel hasil geseran terlihat sama meyakinkannya,
  berarti formatnya yang menciptakan kesan hasil, bukan datanya.
""")
    show("DATA ASLI", real, order)
    for i in range(3):
        ys = np.roll(y, int(RNG.integers(1000, len(y) - 1000)))
        o = np.array([stat(b, ys) for _, _, _, b in cells])
        show(f"DERAU #{i+1}  (tidak ada sinyal sama sekali)",
             best_per_family(cells, o, fam_arr, lag_arr, nm_arr), order)

    rule("UJI 2 — APAKAH JEDA ITU MUNCUL LAGI DI SEPARUH KATALOG YANG LAIN?")
    print("""
  Jeda yang nyata adalah sifat fisika: ia harus muncul di kedua separuh
  katalog. Jeda yang terambil dari derau tidak akan terulang, karena derau
  di separuh pertama berbeda dari separuh kedua.
""")
    half = len(y) // 2
    tabs = {}
    for lbl, sl in [("separuh awal", slice(0, half)),
                    ("separuh akhir", slice(half, len(y)))]:
        o = np.array([stat(b[sl], y[sl]) for _, _, _, b in cells])
        tabs[lbl] = best_per_family(cells, o, fam_arr, lag_arr, nm_arr)
    show("SEPARUH AWAL", tabs["separuh awal"], order)
    show("SEPARUH AKHIR", tabs["separuh akhir"], order)

    print(f"\n  {'keluarga':<12} {'awal':>7} {'akhir':>7} {'selisih':>9}   cocok?")
    print("  " + "-" * 54)
    agree = 0
    for fam in order:
        a = tabs["separuh awal"][fam][0]
        b = tabs["separuh akhir"][fam][0]
        ok = a == b
        agree += ok
        print(f"  {fam:<12} {a:>6}d {b:>6}d {abs(a-b):>8}d   "
              f"{'YA' if ok else 'tidak'}")
    print(f"\n  cocok pada {agree} dari {len(order)} keluarga")

    rule("KESIMPULAN")
    print(f"""
  Kolom 'jeda' selalu terisi, apa pun datanya. Maksimum dari sepuluh angka
  selalu ada, bahkan ketika kesepuluh angka itu murni derau -- dan tabel
  DERAU di atas membuktikannya: bentuknya sama persis, angkanya sama
  masuk akalnya, padahal di sana dijamin tidak ada apa-apa.

  Yang memisahkan temuan dari kebetulan bukan kolom 'jeda', melainkan
  perbandingan 'terukur' dengan 'ambang' di lag_sweep.py. Di sana kedelapan
  keluarga berada DI BAWAH ambangnya masing-masing, dan yang terbaik pada
  data asli (7,05%) bahkan lebih kecil daripada rata-rata yang ditemukan
  derau (7,77%).

  Uji separuh katalog menutupnya: jeda 'terbaik' hanya cocok di {agree} dari
  {len(order)} keluarga. Jeda yang nyata akan cocok di semuanya.
""")


if __name__ == "__main__":
    main()
