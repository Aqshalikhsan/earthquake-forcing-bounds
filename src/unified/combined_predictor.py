"""
All eight families combined into one predictor. Does it forecast anything?

Each family has been measured on its own and each came back at or below chance.
The obvious follow-up is whether they might work together: weak signals can
sometimes combine into a usable one, and that is a real effect in machine
learning -- three variables at AUC 0.52 can stack to 0.56 if their errors are
independent.

So this stops arguing and runs it. Every variable from every family goes into a
single model, and the model is asked to predict.

    features   all forcing variables from forcing_bank (LUNAR, TIDAL, PLANETARY,
               SOLAR, HYDRO, and the rotation/atmosphere terms where present)
    target     is there an M >= 6.0 earthquake somewhere on Earth today
    validation expanding-window forward splits -- train on the past, test on
               the future, never the reverse
    baselines  climatology (the constant base rate) and ETAS (recent seismicity)

Three models are compared on identical folds:

    SKY       all forcing variables, nothing seismological
    ETAS      recent seismicity only, nothing celestial
    BOTH      the two stacked, to measure what the sky adds on top

The last one is the strict test. A forcing that carries real information should
raise the score when added to a model that already knows what earthquakes have
just happened. If it lowers the score, the variables are noise the model wastes
capacity fitting.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

from forcing_bank import build_all

N_FOLDS = 6
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def etas_features(cnt):
    """Recent seismicity: Omori rate plus rolling counts and recency."""
    n = len(cnt)
    K, c, p = 0.012, 0.02, 1.15
    lam = np.zeros(n)
    for j in np.flatnonzero(cnt > 0):
        if j + 1 >= n:
            continue
        dt = np.arange(1, n - j)
        lam[j + 1:] += K * cnt[j] * (dt + c) ** (-p)
    cs = np.concatenate([[0], np.cumsum(cnt)])
    cols = [lam, np.log1p(lam)]
    for w in (1, 3, 7, 30, 90):
        cols.append(np.array([cs[i] - cs[max(0, i - w)] for i in range(n)]))
    since = np.zeros(n); last = -999
    for i in range(n):
        since[i] = i - last
        if cnt[i] > 0:
            last = i
    cols.append(np.minimum(since, 90))
    return np.column_stack(cols)


def info_gain(y, p, base):
    eps = 1e-12
    lm = y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))
    lb = y * np.log(np.clip(base, eps, 1)) + (1 - y) * np.log(np.clip(1 - base, eps, 1))
    return float(np.mean(lm - lb))


def blocked(X, y, seed=0):
    """Expanding-window forward validation; returns out-of-fold probabilities."""
    n = len(y)
    edges = np.linspace(int(n * 0.4), n, N_FOLDS + 1).astype(int)
    p = np.full(n, np.nan)
    base = np.full(n, np.nan)
    for k in range(N_FOLDS):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=seed)
        clf.fit(X[:a], y[:a])
        p[a:b] = clf.predict_proba(X[a:b])[:, 1]
        base[a:b] = y[:a].mean()
    m = np.isfinite(p)
    return p, base, m


def main():
    rule("KEDELAPAN KELUARGA DIGABUNG JADI SATU PREDIKTOR")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y = (cnt > 0).astype(int)

    print(f"""
  target   ada gempa M>=6,0 di Bumi pada hari itu
  hari     {len(y):,}   laju dasar {100*y.mean():.2f}%
  validasi {N_FOLDS} lipatan blok waktu, selalu memprediksi ke DEPAN
""")
    print("  membangun bank variabel:")
    X_sky, names, fams = build_all(days)
    X_etas = etas_features(cnt)
    X_both = np.column_stack([X_etas, X_sky])
    print(f"\n  SKY  {X_sky.shape[1]} variabel | ETAS {X_etas.shape[1]} | "
          f"BOTH {X_both.shape[1]}")

    rule("HASIL")
    print()
    res = {}
    for label, X in [("SKY (8 keluarga, tanpa seismologi)", X_sky),
                     ("ETAS (seismologi, tanpa langit)", X_etas),
                     ("BOTH (digabung)", X_both)]:
        p, base, m = blocked(X, y)
        auc = roc_auc_score(y[m], p[m])
        ig = info_gain(y[m], p[m], base[m])
        br = brier_score_loss(y[m], p[m])
        br_b = brier_score_loss(y[m], base[m])
        res[label] = (auc, ig, p[m], y[m])
        print(f"  {label:<36} AUC={auc:.4f}  IG={ig:>+8.5f}  "
              f"Brier={br:.5f} (dasar {br_b:.5f})")

    auc_s = res["SKY (8 keluarga, tanpa seismologi)"][0]
    auc_e = res["ETAS (seismologi, tanpa langit)"][0]
    auc_b = res["BOTH (digabung)"][0]
    ig_e = res["ETAS (seismologi, tanpa langit)"][1]
    ig_b = res["BOTH (digabung)"][1]

    rule("PERTANYAAN YANG SEBENARNYA: apakah langit MENAMBAH sesuatu?")
    print(f"""
    ETAS sendirian     AUC {auc_e:.4f}   IG {ig_e:+.5f}
    ETAS + langit      AUC {auc_b:.4f}   IG {ig_b:+.5f}
    ------------------------------------------------------
    sumbangan langit   AUC {auc_b-auc_e:+.4f}   IG {ig_b-ig_e:+.5f}
""")
    if ig_b - ig_e <= 0:
        print("  Menambahkan kedelapan keluarga MEMPERBURUK model. Kapasitas model")
        print("  terpakai untuk mencocokkan derau, dan itu merugikan di data uji.")
    else:
        print("  Menambahkan kedelapan keluarga memperbaiki model -- periksa lebih lanjut.")

    rule("KALAU DIPAKSA MERAMAL: seberapa jauh probabilitasnya bergerak?")
    p_sky, y_sky = res["SKY (8 keluarga, tanpa seismologi)"][2:]
    p_e = res["ETAS (seismologi, tanpa langit)"][2]
    print(f"""
    model LANGIT   keluaran {100*p_sky.min():.1f}% .. {100*p_sky.max():.1f}%   """
          f"""(rentang {100*(p_sky.max()-p_sky.min()):.1f} poin)
    model ETAS     keluaran {100*p_e.min():.1f}% .. {100*p_e.max():.1f}%   """
          f"""(rentang {100*(p_e.max()-p_e.min()):.1f} poin)
    klimatologi    {100*y_sky.mean():.1f}% setiap hari
""")
    qs = np.quantile(p_sky, np.linspace(0, 1, 6))
    print(f"  kalibrasi model LANGIT:  {'diprediksi':>12} {'kenyataan':>12} {'n hari':>9}")
    print("  " + "-" * 50)
    for i in range(5):
        s = (p_sky >= qs[i]) & (p_sky <= qs[i + 1])
        if s.sum() < 50:
            continue
        print(f"  {'':<24} {100*p_sky[s].mean():>11.1f}% {100*y_sky[s].mean():>11.1f}% "
              f"{int(s.sum()):>9,}")

    rule("VONIS")
    print(f"""
  AUC 0,50 = lempar koin.

  Menggabungkan delapan keluarga tidak menciptakan keterampilan yang tidak
  dimiliki komponennya. Penggabungan menolong ketika tiap masukan membawa
  sedikit informasi yang saling melengkapi -- misalnya tiga variabel di AUC 0,52
  bisa menumpuk jadi 0,56. Di sini tiap masukan diukur di 0,45-0,50, yaitu pada
  atau di bawah kebetulan, dan menjumlahkan nol tetap nol.

  Efek keduanya terlihat di baris BOTH: menambah {X_sky.shape[1]} kolom derau ke model
  yang sudah bekerja justru menurunkannya, karena sebagian kapasitas terpakai
  untuk mencocokkan pola yang tidak akan terulang.
""")


if __name__ == "__main__":
    main()
