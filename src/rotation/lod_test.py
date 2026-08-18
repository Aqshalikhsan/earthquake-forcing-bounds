"""
Earth rotation (length of day) as an earthquake trigger.

Two reasons this family belongs in the table.

First, it is the mechanism Awadh (2021) proposed: planetary attraction changes
Earth's rotation rate, which stresses plates. That paper's own equation turned
out to be S = V cos(latitude) -- the schoolroom formula for surface speed by
latitude, containing no planetary term at all, and implying a 7% swing in
rotation that would lengthen the day by 99 minutes. Here the claim is tested
with the quantity actually measured: IERS length-of-day, accurate to
microseconds, which varies by 1-4 milliseconds.

Second, there is a specific published claim to check: Bendick & Bilham (2017,
GRL) reported a 5-6 year correlation between decadal LOD variation and the rate
of M >= 7 earthquakes. It has been contested and never settled.

    data     IERS EOP 20 C04, daily, 1962-2026 (23,574 days)
             columns: x, y polar motion; UT1-UTC; LOD(s)

LOD is a GLOBAL SCALAR, like lunar phase and solar activity: one value for the
whole planet on a given day. It therefore carries no spatial information and
cannot say where -- the same structural limit that defeated those families. It
is tested here because a specific claim exists, not because it could ever
support a location-specific forecast.

Pre-registered:
  H1 LEVEL     earthquakes prefer days of anomalous LOD
  H2 RATE      earthquakes prefer days of rapid LOD change (rotation
               accelerating or decelerating)
  H3 DECADAL   the Bendick & Bilham form: LOD smoothed over 5-6 years
  H4 POLAR     polar motion amplitude, included since it shares the mechanism

Null: circular time shift of the earthquake series, as everywhere else.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime

from honest_test import gardner_knopoff_window, haversine_vec

N_SHIFT = 2000
N_BINS = 10
MIN_BIN = 200
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_iers():
    """IERS EOP C04: returns day-ordinal, LOD (ms), polar motion amplitude."""
    rows = []
    for line in (datafile("iers_eop.txt")).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 13 or not f[0].isdigit():
            continue
        try:
            d = datetime(int(f[0]), int(f[1]), int(f[2])).toordinal()
            x, y = float(f[5]), float(f[6])
            lod = float(f[12]) * 1000.0            # s -> ms
        except (ValueError, IndexError):
            continue
        rows.append((d, lod, np.hypot(x, y)))
    rows.sort()
    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1], a[:, 2]


def smooth(v, win):
    """Centred running mean, used for the decadal form."""
    k = np.ones(win) / win
    return np.convolve(v, k, mode="same")


def load_quakes(minmag, maxdepth=70.0):
    out = []
    for r in csv.DictReader(io.StringIO((datafile("global_m6.csv")).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"]); lo = float(r["longitude"])
            dep = float(r["depth"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if m < minmag or dep > maxdepth:
            continue
        out.append((dt, m, la, lo))
    out.sort(key=lambda e: e[0])
    return out


def decluster(rows):
    n = len(rows)
    days = np.array([(r[0] - rows[0][0]).total_seconds() / 86400.0 for r in rows])
    mags = np.array([r[1] for r in rows])
    lats = np.array([r[2] for r in rows])
    lons = np.array([r[3] for r in rows])
    removed = np.zeros(n, dtype=bool)
    for i in np.argsort(-mags):
        if removed[i]:
            continue
        dkm, ddays = gardner_knopoff_window(mags[i])
        delta = days - days[i]
        cand = (delta >= 0) & (delta <= ddays) & (mags <= mags[i]) & (~removed)
        cand[i] = False
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        d = haversine_vec(lats[i], lons[i], lats[idx], lons[idx])
        removed[idx[d <= dkm]] = True
    return [r for i, r in enumerate(rows) if not removed[i]]


def daily_counts(events, days):
    idx = {int(d): i for i, d in enumerate(days)}
    y = np.zeros(len(days))
    for dt, *_ in events:
        i = idx.get(dt.toordinal())
        if i is not None:
            y[i] = 1
    return y


def stat_max_dev(x, y, n_bins=N_BINS):
    e = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if e.size < 3:
        return 0.0
    b = np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2)
    base = y.mean()
    cnt = np.bincount(b).astype(float)
    s = np.bincount(b, weights=y)
    ok = cnt >= MIN_BIN
    if not ok.any():
        return 0.0
    rates = np.where(ok, s / np.maximum(cnt, 1), base)
    return float(np.max(np.abs(rates - base)))


def run_family(name, variables, y, days):
    obs = {k: stat_max_dev(v, y) for k, v in variables.items()}
    null = {k: np.empty(N_SHIFT) for k in variables}
    for i in range(N_SHIFT):
        ys = np.roll(y, int(RNG.integers(1, len(y))))
        for k, v in variables.items():
            null[k][i] = stat_max_dev(v, ys)
    best_k = max(obs, key=obs.get)
    nb = np.max(np.column_stack([null[k] for k in variables]), axis=1)
    p_fam = float((nb >= obs[best_k]).mean())

    print(f"\n  {name}")
    print(f"  {'variabel':<24} {'terukur':>10} {'ambang 95%':>12} {'p sendiri':>11}")
    print("  " + "-" * 62)
    for k in variables:
        thr = np.percentile(null[k], 95)
        p_own = float((null[k] >= obs[k]).mean())
        print(f"  {k:<24} {100*obs[k]:>9.2f}% {100*thr:>11.2f}% {p_own:>11.3f}")
    print(f"\n  terbaik keluarga: {best_k}  ->  p terkoreksi = {p_fam:.3f}")
    print(f"  batas atas (persentil 95 null max): {100*np.percentile(nb,95):.2f}% "
          f"= {100*np.percentile(nb,95)/y.mean():.1f}% relatif")
    return p_fam


def main():
    rule("ROTASI BUMI (LOD) SEBAGAI PEMICU GEMPA")

    d_iers, lod, pm = load_iers()
    print(f"""
  IERS EOP 20 C04
    hari       {len(d_iers):,}  ({datetime.fromordinal(int(d_iers[0])).date()} .. """
          f"""{datetime.fromordinal(int(d_iers[-1])).date()})
    LOD        {lod.min():+.3f} .. {lod.max():+.3f} ms  (sd {lod.std():.3f} ms)
    polar mot. {pm.min():.3f} .. {pm.max():.3f} arcsec
""")
    print("  Klaim Awadh (2021) menyiratkan rotasi berubah ~7%, yaitu panjang hari")
    print(f"  bergeser ~99 menit. Yang terukur: {lod.max()-lod.min():.3f} milidetik.")
    print(f"  Selisihnya sekitar {99*60*1000/(lod.max()-lod.min()):,.0f} kali lipat.\n")

    variables = {
        "lod": lod,
        "lod_abs": np.abs(lod - np.median(lod)),
        "lod_rate": np.gradient(lod),
        "lod_decadal_5y": smooth(lod, 5 * 365),
        "lod_decadal_rate": np.gradient(smooth(lod, 5 * 365)),
        "polar_motion": pm,
    }

    for minmag, label in [(6.0, "M>=6.0"), (7.0, "M>=7.0  (bentuk Bendick & Bilham)")]:
        ev = decluster(load_quakes(minmag))
        ev = [e for e in ev if d_iers[0] <= e[0].toordinal() <= d_iers[-1]]
        y = daily_counts(ev, d_iers)
        rule(f"TARGET {label}   n = {int(y.sum()):,} gempa, laju dasar {100*y.mean():.2f}%")
        run_family(f"keluarga ROTASI ({len(variables)} variabel)", variables, y, d_iers)

    rule("CATATAN")
    print("""
  LOD adalah skalar global: satu nilai untuk seluruh Bumi pada hari tertentu.
  Seperti fase bulan dan aktivitas matahari, ia tidak membawa informasi lokasi
  dan karena itu tidak bisa menopang ramalan yang menyebut tempat -- terlepas
  dari hasil uji di atas.

  Bentuk dekadal (rata-rata berjalan 5 tahun) diuji karena itu yang diklaim
  Bendick & Bilham (2017). Perlu dicatat: penghalusan 5 tahun membuat variabel
  ini nyaris monoton dalam rentang data, sehingga membagi hari menurutnya
  mendekati membagi hari menurut era -- perangkap yang sama seperti jarak
  Uranus/Neptunus. Null pergeseran melingkar menanganinya, tapi ambangnya
  otomatis jadi longgar, dan itu terlihat pada kolom 'ambang 95%'.
""")


if __name__ == "__main__":
    main()
