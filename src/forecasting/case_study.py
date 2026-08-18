"""
One real earthquake, day by day: what each model actually said beforehand.

Every result in this project is a p-value or an AUC, and those are the right
instruments for deciding what is true -- but they are the wrong instrument for
seeing what a model would have done. This file replaces the summary statistic
with the thing itself: pick a real earthquake, stand at each day before it, and
print what the model would have told you that morning.

Two models are asked the same question at the same moments:

    LANGIT   the 130 forcing variables -- Moon, planets, Sun, tides, water,
             rotation, air pressure, ionosphere
    ETAS     nothing but recent local seismicity, through Omori's law

    target   is there an M >= 6.0 within RADIUS km of this place, today?

Training uses only days before the case window, with a further gap of one year
left between training and test, so nothing the model saw can overlap the period
it is being asked about.

A quiet control period is run alongside each earthquake. Without it there is no
way to tell a warning from a habit: a model that reads 35% every day is not
warning about anything when it reads 35% the day before a disaster.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime, date

from sklearn.ensemble import HistGradientBoostingClassifier

from gcmt import parse_ndk
from honest_test import haversine_vec
from forcing_bank import build_all

RADIUS = 500.0
GAP_DAYS = 365
BEFORE, AFTER = 20, 5
RNG = np.random.default_rng(20260817)

CASES = [
    ("Palu, Sulawesi Tengah", date(2018, 9, 28), -0.256, 119.846, "M7,5"),
    ("Lombok, NTB", date(2018, 8, 5), -8.29, 116.45, "M6,9"),
]


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def local_target(ordn, lat, lon, mw, days, la, lo, minmag=6.0):
    """Daily 0/1: an M >= minmag within RADIUS km of (la, lo)."""
    sel = mw >= minmag
    d = haversine_vec(la, lo, lat[sel], lon[sel]) <= RADIUS
    hit = ordn[sel][d].astype(int)
    y = np.zeros(days.size, dtype=int)
    idx = np.searchsorted(days, hit)
    idx = idx[(idx >= 0) & (idx < days.size)]
    y[idx] = 1
    return y


def etas_local(ordn, lat, lon, mw, days, la, lo, minmag=4.5):
    """Omori rate from recent nearby seismicity -- strictly causal."""
    sel = mw >= minmag
    near = haversine_vec(la, lo, lat[sel], lon[sel]) <= RADIUS
    t = ordn[sel][near].astype(int)
    m = mw[sel][near]
    K, c, p, alpha = 0.02, 0.05, 1.10, 0.9
    lam = np.zeros(days.size)
    pos = np.searchsorted(days, t)
    for k in range(t.size):
        j = pos[k]
        if j + 1 >= days.size:
            continue
        dt = np.arange(1, days.size - j)
        lam[j + 1:] += K * 10 ** (alpha * (m[k] - minmag)) * (dt + c) ** (-p)
    return lam


def main():
    rule("SATU GEMPA NYATA, HARI PER HARI: apa yang dikatakan tiap model")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk")) if e.depth <= 70.0]
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lat = np.array([e.lat for e in ev]); lon = np.array([e.lon for e in ev])
    mw = np.array([e.mw for e in ev])

    days = np.load(datafile("m6_days.npy")).astype(int)
    print("\n  membangun bank variabel:")
    X, names, fams = build_all(days)
    print(f"\n  LANGIT {X.shape[1]} variabel   |   ETAS 3 variabel "
          f"(laju Omori lokal)")

    for title, when, la, lo, mag in CASES:
        t0 = when.toordinal()
        y = local_target(ordn, lat, lon, mw, days, la, lo)
        lam = etas_local(ordn, lat, lon, mw, days, la, lo)
        # Backward difference, not np.gradient: gradient averages the
        # neighbours on both sides, so on day X it reads day X+1 and the
        # feature would carry the earthquake it is supposed to precede.
        dlam = np.concatenate([[0.0], np.diff(lam)])
        Xe = np.column_stack([lam, np.log1p(lam), dlam])

        train_end = np.searchsorted(days, t0 - BEFORE - GAP_DAYS)
        if train_end < 2000 or y[:train_end].sum() < 25:
            print(f"\n  {title}: data latih tidak cukup, dilewati.")
            continue

        models = {}
        for lbl, M in [("LANGIT", X), ("ETAS", Xe)]:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=60, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15, random_state=0)
            clf.fit(M[:train_end], y[:train_end])
            models[lbl] = clf.predict_proba(M)[:, 1]
        base = y[:train_end].mean()

        rule(f"{title} — {mag}, {when}")
        print(f"""
  target    ada gempa M>=6,0 dalam {RADIUS:.0f} km dari {title.split(',')[0]}
  latih     {datetime.fromordinal(int(days[0])).date()} .. """
              f"""{datetime.fromordinal(int(days[train_end-1])).date()}
            (dihentikan {GAP_DAYS} hari sebelum jendela ini)
  laju dasar {100*base:.1f}% per hari
""")
        print(f"  {'hari':>6} {'tanggal':>12} {'LANGIT':>9} {'ETAS':>9}   kenyataan")
        print("  " + "-" * 58)
        for off in range(-BEFORE, AFTER + 1):
            i = int(np.searchsorted(days, t0 + off))
            if i >= days.size:
                continue
            mark = ""
            if off == 0:
                mark = f"<<< GEMPA {mag}"
            elif y[i]:
                mark = "gempa M>=6"
            tag = f"X{off:+d}" if off else "X"
            print(f"  {tag:>6} {str(datetime.fromordinal(int(days[i])).date()):>12} "
                  f"{100*models['LANGIT'][i]:>8.1f}% {100*models['ETAS'][i]:>8.1f}%   {mark}")

        w = slice(int(np.searchsorted(days, t0 - BEFORE)),
                  int(np.searchsorted(days, t0)))
        q = slice(int(np.searchsorted(days, t0 - 400)),
                  int(np.searchsorted(days, t0 - 200)))
        print(f"""
  Rentang 20 hari sebelum gempa   LANGIT {100*models['LANGIT'][w].min():.1f}-"""
              f"""{100*models['LANGIT'][w].max():.1f}%   ETAS {100*models['ETAS'][w].min():.1f}-"""
              f"""{100*models['ETAS'][w].max():.1f}%
  Periode tenang (X-400..X-200)   LANGIT {100*models['LANGIT'][q].mean():.1f}%"""
              f"""        ETAS {100*models['ETAS'][q].mean():.1f}%

  Kalau angka di dua baris itu mirip, model tidak sedang memperingatkan
  apa pun -- ia hanya mengulang kebiasaannya.""")

    rule("CARA MEMBACANYA")
    print("""
  Kolom LANGIT adalah jawaban atas pertanyaan Anda: "gempa terjadi hari X,
  model bilang apa di hari Y". Lihat pergerakannya menjelang hari X, lalu
  bandingkan dengan periode tenang di bawahnya.

  Yang dicari bukan angka yang tinggi, melainkan angka yang NAIK tepat sebelum
  gempa dan TIDAK naik di waktu lain. Angka tinggi setiap hari bukan
  peringatan; itu cuma kebiasaan model, dan kalau dipakai akan menyalakan
  alarm sepanjang tahun.

  Kolom ETAS memperlihatkan bentuk yang sebaliknya: datar sebelum gempa, lalu
  melonjak SETELAHNYA. Itu jujur -- ia tidak mengaku tahu lebih dulu, ia hanya
  tahu bahwa kerak yang baru patah akan patah lagi.
""")


if __name__ == "__main__":
    main()
