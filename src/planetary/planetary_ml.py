"""
Step 14: the planetary/geometric ML model, built to be robust.

This keeps the concept of El Moudden et al. (2024) and SSGEOS -- planetary and
lunar geometry as input, machine learning as the engine -- and repairs every
failure mode found in steps 10-13. The aim is a HONEST probability: given
today's solar-system geometry, what is the chance of a large earthquake?

The four repairs, each targeting a specific way the earlier work went wrong:

  1. FLAT TARGET. El Moudden et al. used a catalogue whose rate tripled from
     1970 to 2024 as detection improved, and features that reconstruct the date
     to R^2 = 0.999. Any smooth function of time then scores well with no
     physics at all. Here the target is M >= 6.0, complete since 1970: the rate
     is 0.30-0.43/day across every decade with no systematic ramp. With nothing
     time-varying to exploit, date leakage buys nothing.

  2. PERIODIC FEATURES ONLY. Angles and aspect proximities, no raw distances to
     outer planets (Uranus and Neptune distances are near-monotonic ramps over
     this window -- they ARE the calendar).

  3. TIME-BLOCKED VALIDATION. A random train/test split leaks: neighbouring days
     share aftershock sequences and near-identical geometry. Folds here are
     contiguous blocks, always predicting forward.

  4. A REAL BASELINE, AND A PERMUTATION NULL. Every number is reported against
     climatology (the constant base rate), because "68% accuracy" means nothing
     without knowing what 0 features achieve. And the whole pipeline is re-run
     on block-shuffled targets to see what it scores on noise.

Output is a calibrated probability, which is what was actually asked for:
a percentage, honestly derived, with its skill measured rather than asserted.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from planetary import heliocentric, PLANETS, aspect_distance
from ephem_vec import jd_from_datetime_array, bodies_equatorial, AU

HERE = DATA
RNG = np.random.default_rng(20260816)
SYNODIC = 29.530588853

KEY_PAIRS = [("Venus", "Neptune"), ("Venus", "Saturn"), ("Mercury", "Mars"),
             ("Mercury", "Jupiter"), ("Jupiter", "Saturn"), ("Mars", "Uranus"),
             ("Venus", "Jupiter"), ("Venus", "Mercury"), ("Mars", "Jupiter"),
             ("Saturn", "Neptune")]


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def build_features(days):
    """Purely periodic solar-system geometry. No raw outer-planet distances."""
    dts = [datetime.fromordinal(int(d)) for d in days]
    jd = jd_from_datetime_array(dts)
    eph = bodies_equatorial(jd)
    names, cols = [], []

    def add(name, v):
        names.append(name)
        cols.append(np.asarray(v, dtype=float))

    # --- lunar phase and its harmonics (harmonic 4 = the SSGEOS claim) ---
    ra_m, dec_m, dist_m = eph["moon"]
    ra_s, dec_s, dist_s = eph["sun"]
    r = np.pi / 180
    ce = (np.sin(dec_m * r) * np.sin(dec_s * r)
          + np.cos(dec_m * r) * np.cos(dec_s * r) * np.cos((ra_m - ra_s) * r))
    el = np.degrees(np.arccos(np.clip(ce, -1, 1)))
    sign = np.sign(((ra_m - ra_s) % 360) - 180)
    phase = np.where(sign >= 0, el, 360 - el)
    for k in (1, 2, 4):
        add(f"moon_phase_sin{k}", np.sin(np.radians(k * phase)))
        add(f"moon_phase_cos{k}", np.cos(np.radians(k * phase)))
    add("moon_offset_days",
        np.minimum.reduce([np.abs(((phase - c + 180) % 360) - 180)
                           for c in (0, 90, 180, 270)]) / 360 * SYNODIC)
    add("moon_dist", dist_m / 1e8)
    add("moon_dec", dec_m)

    # --- spring-neap tidal envelope: the one physically motivated term ---
    am, asun = 2.17, 1.0
    add("tidal_amp", np.sqrt(am**2 + asun**2
                             + 2 * am * asun * np.cos(np.radians(2 * phase))))
    add("sun_dist", dist_s / AU)
    add("sun_dec", dec_s)

    # --- planetary elongations from the Sun, geocentric ---
    ex, ey, ez = heliocentric("Earth", jd)
    sun_lon = (np.degrees(np.arctan2(ey, ex)) + 180) % 360
    geo, helio = {}, {}
    for p in PLANETS:
        px, py, pz = heliocentric(p, jd)
        geo[p] = np.degrees(np.arctan2(py - ey, px - ex)) % 360
        helio[p] = np.degrees(np.arctan2(py, px)) % 360
        elong = aspect_distance(geo[p], sun_lon)
        add(f"{p}_elong_sin", np.sin(np.radians(elong)))
        add(f"{p}_elong_cos", np.cos(np.radians(elong)))

    # --- pairwise aspect geometry: the core SSGEOS construct ---
    for a, b in KEY_PAIRS:
        sep = aspect_distance(geo[a], geo[b])
        add(f"{a}_{b}_sin", np.sin(np.radians(sep)))
        add(f"{a}_{b}_cos", np.cos(np.radians(sep)))
        add(f"{a}_{b}_sin4", np.sin(np.radians(4 * sep)))
        # proximity to the nearest of 0/45/90/135/180
        add(f"{a}_{b}_aspect_prox",
            np.minimum.reduce([np.abs(sep - t) for t in (0, 45, 90, 135, 180)]))

    # --- heliocentric alignment: "Earth between two planets" ---
    for a, b in KEY_PAIRS[:5]:
        ax, ay, _ = heliocentric(a, jd)
        bx, by, _ = heliocentric(b, jd)
        va = np.degrees(np.arctan2(ay - ey, ax - ex)) % 360
        vb = np.degrees(np.arctan2(by - ey, bx - ex)) % 360
        add(f"align_{a}_{b}", aspect_distance(va, vb))

    return np.column_stack(cols), names


def info_gain(y_true, p_model, p_base):
    """Nats per day gained over the baseline. The CSEP-style score."""
    eps = 1e-12
    lm = y_true * np.log(np.clip(p_model, eps, 1)) + (1 - y_true) * np.log(np.clip(1 - p_model, eps, 1))
    lb = y_true * np.log(np.clip(p_base, eps, 1)) + (1 - y_true) * np.log(np.clip(1 - p_base, eps, 1))
    return float(np.mean(lm - lb))


def blocked_cv(X, y, n_folds=6, seed=0):
    """
    Expanding-window forward validation. Fold k trains on everything before a
    cut and tests on the block after it -- never the reverse, never interleaved.
    """
    n = len(y)
    edges = np.linspace(int(n * 0.4), n, n_folds + 1).astype(int)
    oof_p = np.full(n, np.nan)
    oof_base = np.full(n, np.nan)
    for k in range(n_folds):
        tr_end, te_end = edges[k], edges[k + 1]
        if te_end <= tr_end:
            continue
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xte = X[tr_end:te_end]
        base = ytr.mean()
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=seed)
        clf.fit(Xtr, ytr)
        oof_p[tr_end:te_end] = clf.predict_proba(Xte)[:, 1]
        oof_base[tr_end:te_end] = base
    m = np.isfinite(oof_p)
    return oof_p, oof_base, m


def evaluate(y, p, base, label):
    ig = info_gain(y, p, base)
    auc = roc_auc_score(y, p)
    br = brier_score_loss(y, p)
    br_b = brier_score_loss(y, base)
    print(f"  {label:<34} IG={ig:>+8.5f} nat/hari  AUC={auc:.4f}  "
          f"Brier={br:.5f} (baseline {br_b:.5f})")
    return ig, auc, br


def main():
    rule("MODEL PLANET/GEOMETRI YANG DIPERBAIKI  -- probabilitas terkalibrasi")

    days = np.load(HERE / "m6_days.npy")
    cnt = np.load(HERE / "m6_cnt.npy")
    y = (cnt > 0).astype(int)

    X, names = build_features(days)
    print(f"""
  target      ada tidaknya gempa M>=6.0 di Bumi pada hari itu
  periode     {datetime.fromordinal(int(days[0])).date()} .. {datetime.fromordinal(int(days[-1])).date()}
  hari        {len(y):,}   hari dengan gempa: {y.sum():,} ({100*y.mean():.2f}%)
  fitur       {X.shape[1]} kolom geometri tata surya (sudut, aspek, fase, pasang surut)
  validasi    6 lipatan blok waktu, selalu memprediksi ke DEPAN
""")

    rule("HASIL")
    print()
    p, base, m = blocked_cv(X, y)
    ig, auc, br = evaluate(y[m], p[m], base[m], "A. geometri planet + bulan")

    # baseline yang hanya tahu musim
    doy = np.array([datetime.fromordinal(int(d)).timetuple().tm_yday for d in days], float)
    Xs = np.column_stack([np.sin(2 * np.pi * doy / 365.25),
                          np.cos(2 * np.pi * doy / 365.25)])
    ps, bs, ms = blocked_cv(Xs, y)
    evaluate(y[ms], ps[ms], bs[ms], "B. hanya musim (kontrol)")

    # null: acak target dalam blok 30 hari (pertahankan klaster, hancurkan kaitan)
    nb = len(y) // 30
    blocks = np.array_split(y[:nb * 30], nb)
    RNG.shuffle(blocks)
    y_null = np.concatenate(blocks + [y[nb * 30:]])
    pn, bn, mn = blocked_cv(X, y_null)
    evaluate(y_null[mn], pn[mn], bn[mn], "C. NULL: target diacak per blok")

    rule("APAKAH SKILL-NYA NYATA?  uji permutasi")
    print("\n  Ulangi seluruh pipeline pada 20 target acak-blok, lihat sebaran IG:\n")
    igs = []
    for i in range(20):
        blocks = np.array_split(y[:nb * 30], nb)
        rng2 = np.random.default_rng(100 + i)
        rng2.shuffle(blocks)
        yn = np.concatenate(blocks + [y[nb * 30:]])
        pp, bb, mm = blocked_cv(X, yn, seed=i)
        igs.append(info_gain(yn[mm], pp[mm], bb[mm]))
    igs = np.array(igs)
    pval = float((igs >= ig).mean())
    print(f"    IG pada data asli        : {ig:+.5f} nat/hari")
    print(f"    IG pada noise, rata-rata : {igs.mean():+.5f}")
    print(f"    IG pada noise, maksimum  : {igs.max():+.5f}")
    print(f"    p permutasi              : {pval:.3f}")

    rule("KALIBRASI: kalau model bilang X%, apakah benar X%?")
    pm, ym = p[m], y[m]
    qs = np.quantile(pm, np.linspace(0, 1, 11))
    print(f"\n  {'prediksi model':>18} {'kenyataan':>11} {'n hari':>8}")
    print("  " + "-" * 40)
    for i in range(10):
        sel = (pm >= qs[i]) & (pm <= qs[i + 1])
        if sel.sum() < 20:
            continue
        print(f"  {100*pm[sel].mean():>16.1f}% {100*ym[sel].mean():>10.1f}% {sel.sum():>8}")
    spread = pm.max() - pm.min()
    print(f"\n  rentang probabilitas yang dikeluarkan model: "
          f"{100*pm.min():.1f}% .. {100*pm.max():.1f}%  (lebar {100*spread:.1f} poin)")
    print(f"  klimatologi (tebakan konstan)               : {100*ym.mean():.1f}%")

    rule("VONIS")
    verdict = "ADA skill di atas klimatologi" if pval < 0.05 and ig > 0 else \
              "TIDAK ada skill yang bisa dibedakan dari noise"
    print(f"""
  {verdict}

  IG = {ig:+.5f} nat/hari, p permutasi = {pval:.3f}, AUC = {auc:.4f}
  (AUC 0.50 = tebak koin, 1.00 = sempurna)

  Sebagai pembanding dari langkah 8 pada data yang sama jenisnya:
    ETAS spatio-temporal   IG = +1.81 nat per gempa
    model bulan            IG = +0.002 nat per gempa
""")


if __name__ == "__main__":
    main()
