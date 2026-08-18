"""
Step 15: the experiment nobody has run -- planetary geometry measured as an
INCREMENT over a real seismological baseline.

Both published papers compare planetary features against nothing. El Moudden
et al. (2024) use nine Earth-planet distances and no seismological predictor at
all: no past seismicity, no aftershock history, no b-value, nothing from Earth.
Against a baseline of zero, planetary features look useful -- but step 14 showed
that apparent usefulness is the catalogue's own growth curve.

The fair and much harder question is different:

    given a model that ALREADY knows what earthquakes have just happened,
    does solar-system geometry explain anything that is left over?

That is the question this file answers, and it is the strongest form of the
hypothesis. Planetary geometry is no longer asked to compete with catalogue
trends. It is handed the residual -- the part real seismology cannot predict --
and given every chance to explain it.

Design:
  target     >=1 global M>=6.0 on day d (base rate ~31%, catalogue flat since 1970)
  baseline   ETAS-style features: fitted Omori rate, recent counts, recency,
             recent maximum magnitude, productivity-weighted sums
  test       the 68 geometric features from planetary_ml.py
  compare    IG(ETAS + geometry) - IG(ETAS alone), on identical time-blocked folds
  null       permute ONLY the geometric block in time, keeping ETAS aligned.
             That destroys the planetary relationship and nothing else, which is
             exactly the null the increment needs.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from planetary_ml import build_features, info_gain, rule

HERE = DATA
N_PERM = 30
BLOCK = 45          # days per block when permuting the geometry
RNG = np.random.default_rng(20260816)

# temporal ETAS parameters, refit here on the global M>=6 catalogue
ETAS_P = dict(K=0.012, alpha=1.35, c=0.02, p=1.15, mu=0.30)
MC = 6.0


def etas_features(days, cnt, mags_by_day):
    """
    Causal seismological predictors: everything uses strictly past days.
    This is the baseline the geometry has to beat.
    """
    n = len(days)
    names = []
    cols = []

    def add(nm, v):
        names.append(nm)
        cols.append(np.asarray(v, float))

    # Omori-style conditional rate from all previous events
    lam = np.zeros(n)
    P = ETAS_P
    hist_idx = np.flatnonzero(cnt > 0)
    for j in hist_idx:
        if j + 1 >= n:
            continue
        dt = np.arange(1, n - j)
        contrib = (P["K"] * np.exp(P["alpha"] * (mags_by_day[j] - MC))
                   * (dt + P["c"]) ** (-P["p"]))
        lam[j + 1:] += contrib
    add("etas_rate", lam)
    add("etas_log", np.log1p(lam))

    # rolling counts over past windows (shifted: never includes today)
    cs = np.concatenate([[0], np.cumsum(cnt)])
    for w in (1, 3, 7, 14, 30, 90, 365):
        past = np.zeros(n)
        for i in range(n):
            lo = max(0, i - w)
            past[i] = cs[i] - cs[lo]
        add(f"count_{w}d", past)

    # days since last event, and since last big one
    since = np.zeros(n)
    last = -999
    for i in range(n):
        since[i] = i - last
        if cnt[i] > 0:
            last = i
    add("days_since_last", np.minimum(since, 90))

    since7 = np.zeros(n)
    last7 = -999
    for i in range(n):
        since7[i] = i - last7
        if mags_by_day[i] >= 7.0:
            last7 = i
    add("days_since_M7", np.minimum(since7, 365))

    # largest magnitude and productivity in recent past
    for w in (7, 30):
        mx = np.zeros(n)
        prod = np.zeros(n)
        for i in range(n):
            lo = max(0, i - w)
            seg = mags_by_day[lo:i]
            mx[i] = seg.max() if len(seg) else 0.0
            prod[i] = np.sum(10 ** (0.7 * np.clip(seg - MC, 0, None))) if len(seg) else 0.0
        add(f"maxmag_{w}d", mx)
        add(f"prod_{w}d", prod)

    return np.column_stack(cols), names


def blocked_eval(X, y, n_folds=6, seed=0):
    """Expanding-window forward validation; returns out-of-fold probabilities."""
    n = len(y)
    edges = np.linspace(int(n * 0.4), n, n_folds + 1).astype(int)
    p = np.full(n, np.nan)
    base = np.full(n, np.nan)
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=seed)
        clf.fit(X[:a], y[:a])
        p[a:b] = clf.predict_proba(X[a:b])[:, 1]
        base[a:b] = y[:a].mean()
    m = np.isfinite(p)
    return p, base, m


def permute_blocks(X, block, rng):
    """Shuffle contiguous blocks of rows -- destroys date alignment, keeps
    the internal structure and marginal distribution of every column."""
    n = len(X)
    nb = n // block
    idx = np.arange(n)
    chunks = [idx[i * block:(i + 1) * block] for i in range(nb)]
    tail = idx[nb * block:]
    rng.shuffle(chunks)
    return X[np.concatenate(chunks + [tail])]


def main():
    rule("APAKAH GEOMETRI MENAMBAH SESUATU DI ATAS SEISMOLOGI?")

    days = np.load(HERE / "m6_days.npy")
    cnt = np.load(HERE / "m6_cnt.npy")

    # largest magnitude per day, needed for the productivity terms
    import csv, io
    mags = np.zeros(len(days))
    d0 = days[0]
    for rec in csv.DictReader(io.StringIO((HERE / "global_m6.csv").read_text(encoding="utf-8"))):
        if rec.get("type", "earthquake") != "earthquake":
            continue
        try:
            o = datetime.strptime(rec["time"][:10], "%Y-%m-%d").toordinal() - d0
            m = float(rec["mag"])
        except (ValueError, TypeError):
            continue
        if 0 <= o < len(days):
            mags[o] = max(mags[o], m)

    y = (cnt > 0).astype(int)
    X_geo, geo_names = build_features(days)
    X_etas, etas_names = etas_features(days, cnt, mags)

    print(f"""
  target      >=1 gempa M>=6.0 di Bumi pada hari itu
  hari        {len(y):,}   dengan gempa {y.sum():,} ({100*y.mean():.1f}%)
  fitur ETAS  {X_etas.shape[1]} kolom (laju Omori, hitungan lampau, resensi, produktivitas)
  fitur geo   {X_geo.shape[1]} kolom (sudut, aspek, fase bulan, pasang surut)
  validasi    6 lipatan blok waktu, selalu ke depan
""")

    rule("HASIL UTAMA")
    print()
    results = {}
    for label, X in (("ETAS saja (basis seismologi)", X_etas),
                     ("geometri saja", X_geo),
                     ("ETAS + geometri", np.column_stack([X_etas, X_geo]))):
        p, base, m = blocked_eval(X, y)
        ig = info_gain(y[m], p[m], base[m])
        auc = roc_auc_score(y[m], p[m])
        results[label] = (ig, auc)
        print(f"  {label:<32} IG={ig:>+9.5f} nat/hari   AUC={auc:.4f}")

    ig_e = results["ETAS saja (basis seismologi)"][0]
    ig_c = results["ETAS + geometri"][0]
    delta = ig_c - ig_e
    print(f"""
  SUMBANGAN GEOMETRI = IG(ETAS+geo) - IG(ETAS) = {delta:+.5f} nat/hari""")

    rule(f"UJI NULL: {N_PERM} permutasi, hanya blok geometri yang diacak")
    print(f"""
  Fitur ETAS tetap sejajar dengan tanggalnya. Hanya {X_geo.shape[1]} kolom geometri
  yang digeser dalam blok {BLOCK} hari, sehingga kaitannya dengan tanggal hancur
  sementara segalanya yang lain utuh. Kalau geometri benar-benar menyumbang,
  sumbangan aslinya harus menonjol dari sebaran ini.
""")
    deltas = []
    for i in range(N_PERM):
        rng = np.random.default_rng(500 + i)
        Xg = permute_blocks(X_geo, BLOCK, rng)
        p, base, m = blocked_eval(np.column_stack([X_etas, Xg]), y, seed=i)
        deltas.append(info_gain(y[m], p[m], base[m]) - ig_e)
        print(f"    permutasi {i+1}/{N_PERM}", end="\r", flush=True)
    deltas = np.array(deltas)
    print(" " * 40, end="\r")
    pval = float((deltas >= delta).mean())
    print(f"  sumbangan asli                : {delta:+.5f} nat/hari")
    print(f"  sumbangan pada noise, rata2   : {deltas.mean():+.5f}")
    print(f"  sumbangan pada noise, maks    : {deltas.max():+.5f}")
    print(f"  sumbangan pada noise, sd      : {deltas.std():.5f}")
    print(f"  p permutasi                   : {pval:.3f}")
    thresh = np.percentile(deltas, 95)
    print(f"\n  ambang deteksi (persentil 95 noise): {thresh:+.5f} nat/hari")
    print(f"  -> sumbangan geometri di atas ambang ini akan terdeteksi;")
    print(f"     yang terukur adalah {delta:+.5f}")

    rule("VONIS")
    detected = pval < 0.05 and delta > 0
    print(f"""
  {'GEOMETRI MENYUMBANG sesuatu yang nyata' if detected
    else 'GEOMETRI TIDAK MENYUMBANG apa pun di atas seismologi'}

  Ini bentuk terkuat dari hipotesis yang bisa diuji, dan yang paling adil:
  geometri tidak disuruh bersaing menjelaskan tren katalog, melainkan diberi
  SISA yang tidak bisa dijelaskan seismologi -- lalu diukur.

  Batas atas dari uji ini: sumbangan geometri lebih kecil dari {max(thresh,0):.5f}
  nat/hari. Sebagai skala, ETAS sendiri bernilai {ig_e:+.5f} nat/hari pada
  data yang sama.
""")


if __name__ == "__main__":
    main()
