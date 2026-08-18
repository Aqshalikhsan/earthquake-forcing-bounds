"""
Step 19: why the LSTM paper reaches 84% accuracy without any solar-seismic link.

Altaibek et al. (2024, Atmosphere 15, 1290) classify, for each hour, whether an
earthquake (M > 6, depth < 60 km, declustered) will occur anywhere on Earth in
the next 48 hours, using only SOHO proton density. They report 84.47% accuracy,
recall 0.84 and F1 0.75 for the earthquake class, and conclude that "solar
activity, in particular proton density, can anticipate earthquake events".

Two features of their setup make that number reachable with no relationship
whatsoever between the two series:

  1. THE LABEL IS ENORMOUSLY AUTOCORRELATED. A 48-hour forward window means
     hour t and hour t+1 share 47 of their 48 hours, so they almost always
     carry the same label. Consecutive hours are near-duplicates.

  2. THE SPLIT IS NOT TEMPORAL. Section 2.1 says only "the data was divided
     into test and training sets at a ratio of 80% to 20%". With a random
     split -- the default in scikit-learn and Keras -- hour t lands in training
     and hour t+1 in test. The model does not need to learn physics. It needs
     to notice that the test hour looks like the training hour beside it.

This file demonstrates the mechanism directly. Proton density is replaced by a
synthetic series with realistic solar autocorrelation, and earthquakes are
placed at RANDOM times, statistically independent of it by construction. There
is nothing to learn. Then the same two splits are compared.

If the random split reproduces their accuracy on data built to contain no
signal, their accuracy is explained without any solar-seismic connection.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RNG = np.random.default_rng(20260816)

N_YEARS = 27                      # 1996 - 2023, as in the paper
N_HOURS = N_YEARS * 8766
WINDOW = 48                       # hours of history fed to the model
HORIZON = 48                      # "earthquake in the next 48 h"
EQ_PER_YEAR = 65                  # global M>6, shallow, after declustering


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def synthetic_proton_density(n):
    """
    A stand-in for solar-wind proton density: strongly autocorrelated, with the
    27-day solar rotation and the 11-year cycle, log-normal spikes. Its exact
    shape does not matter -- only that it is smooth in time, as the real series
    is. It contains no earthquake information of any kind.
    """
    t = np.arange(n)
    cycle = 1.0 + 0.35 * np.sin(2 * np.pi * t / (11 * 8766))
    rotation = 0.25 * np.sin(2 * np.pi * t / (27 * 24))
    ar = np.zeros(n)
    eps = RNG.normal(0, 1, n)
    for i in range(1, n):                      # AR(1), ~2-day memory
        ar[i] = 0.98 * ar[i - 1] + eps[i]
    ar = ar / ar.std()
    return np.exp(1.3 + 0.45 * (ar + rotation) ) * cycle


def random_earthquakes(n, per_year):
    """Earthquake hours drawn uniformly at random -- independent of everything."""
    k = int(per_year * n / 8766)
    return np.sort(RNG.choice(n, size=k, replace=False))


def make_labels(n, eq_hours, horizon=HORIZON):
    """1 if at least one earthquake falls in the next `horizon` hours."""
    y = np.zeros(n, dtype=int)
    for h in eq_hours:
        y[max(0, h - horizon):h] = 1
    return y


def make_windows(x, window=WINDOW):
    """Sliding windows of the density series, as the LSTM receives."""
    n = len(x)
    idx = np.arange(window)[None, :] + np.arange(n - window)[:, None]
    return x[idx]


def evaluate(name, ytrue, ypred):
    acc = accuracy_score(ytrue, ypred)
    p, r, f, _ = precision_recall_fscore_support(ytrue, ypred, labels=[1],
                                                 zero_division=0)
    print(f"  {name:<34} akurasi={acc:.4f}  presisi={p[0]:.4f}  "
          f"recall={r[0]:.4f}  F1={f[0]:.4f}")
    return acc, p[0], r[0], f[0]


def main():
    rule("APAKAH 84% BISA MUNCUL TANPA HUBUNGAN APA PUN?")

    x = synthetic_proton_density(N_HOURS)
    eq = random_earthquakes(N_HOURS, EQ_PER_YEAR)
    y_all = make_labels(N_HOURS, eq)

    X = make_windows(x)
    y = y_all[WINDOW:]

    print(f"""
  jam total               : {len(y):,}  ({N_YEARS} tahun)
  gempa acak ditanam      : {len(eq):,}  ({EQ_PER_YEAR}/tahun)
  jendela fitur           : {WINDOW} jam kepadatan proton
  label                   : ada gempa dalam {HORIZON} jam ke depan
  laju dasar kelas gempa  : {100*y.mean():.2f}%
  korelasi asli data      : NOL -- gempa ditanam acak, tidak melihat x
""")

    common = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                  min_samples_leaf=50, random_state=0,
                  class_weight="balanced")

    rule("SPLIT ACAK 80/20  (yang tampaknya dipakai paper)")
    print()
    idx = RNG.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    m = HistGradientBoostingClassifier(**common).fit(X[tr], y[tr])
    r_rand = evaluate("acak: jam bersebelahan terpisah", y[te], m.predict(X[te]))

    rule("SPLIT TEMPORAL 80/20  (satu-satunya yang sah untuk deret waktu)")
    print()
    cut = int(0.8 * len(y))
    m2 = HistGradientBoostingClassifier(**common).fit(X[:cut], y[:cut])
    r_temp = evaluate("temporal: latih masa lalu, uji masa depan",
                      y[cut:], m2.predict(X[cut:]))

    print()
    evaluate("tebak 'selalu tidak ada gempa'", y[cut:],
             np.zeros(len(y) - cut, dtype=int))
    evaluate("tebak acak sesuai laju dasar", y[cut:],
             (RNG.random(len(y) - cut) < y.mean()).astype(int))

    rule("VONIS")
    print(f"""
  Paper melaporkan : akurasi 0.8447, presisi 0.68, recall 0.84, F1 0.75
  Split acak       : akurasi {r_rand[0]:.4f}, presisi {r_rand[1]:.4f}, recall {r_rand[2]:.4f}, F1 {r_rand[3]:.4f}
  Split temporal   : akurasi {r_temp[0]:.4f}, presisi {r_temp[1]:.4f}, recall {r_temp[2]:.4f}, F1 {r_temp[3]:.4f}

  Ingat: gempa di sini DITANAM ACAK. Tidak ada apa pun untuk dipelajari.

  Kalau split acak mereproduksi angka paper pada data tanpa sinyal, maka
  angka paper tidak memerlukan hubungan matahari-gempa untuk dijelaskan.
  Yang dipelajari model bukan fisika, melainkan bahwa jam di sebelah jam uji
  ada di data latih, dengan label yang hampir selalu sama.

  Perbaikannya satu baris: split temporal, bukan acak. Semua yang lain --
  arsitektur LSTM, class_weight, dropout, EarlyStopping -- tidak menyentuh
  masalah ini.
""")


if __name__ == "__main__":
    main()
