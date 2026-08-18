"""
Step 16: back to basics -- one variable at a time, no machine learning.

The combined model in step 15 was the wrong place to start. It stacked 71
features at once, and its permutation null was flawed: block-shuffling changes
how SMOOTH a feature is, and a tree overfits a smooth feature more easily than a
jumpy one, so the real geometry and the shuffled geometry were not being judged
on equal terms.

So this file does the simplest honest thing instead. For each geometric variable,
one at a time:

    sort the days into 10 bins by that variable,
    and print the fraction of days in each bin that had an M >= 6.0 earthquake.

No model, no fitting, nothing to overfit. If solar-system geometry influences
when earthquakes happen, the rate must differ between bins. That is the whole
claim, reduced to something you can read off a table.

The null is a circular time shift of the geometry -- the whole series slid by a
random offset and wrapped around. That preserves smoothness, periodicity and
marginal distribution exactly, and destroys only the alignment with real dates.
It is the matched null the previous step should have used.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np
from datetime import datetime
from pathlib import Path

from planetary import heliocentric, PLANETS, aspect_distance
from ephem_vec import jd_from_datetime_array, bodies_equatorial, AU

HERE = DATA
N_SHIFT = 2000
N_BINS = 10
SYNODIC = 29.530588853
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def variables(days):
    """One dictionary of candidate geometric variables. Plain and readable."""
    dts = [datetime.fromordinal(int(d)) for d in days]
    jd = jd_from_datetime_array(dts)
    eph = bodies_equatorial(jd)
    out = {}

    ra_m, dec_m, dist_m = eph["moon"]
    ra_s, dec_s, dist_s = eph["sun"]
    r = np.pi / 180
    ce = (np.sin(dec_m * r) * np.sin(dec_s * r)
          + np.cos(dec_m * r) * np.cos(dec_s * r) * np.cos((ra_m - ra_s) * r))
    el = np.degrees(np.arccos(np.clip(ce, -1, 1)))
    sign = np.sign(((ra_m - ra_s) % 360) - 180)
    phase = np.where(sign >= 0, el, 360 - el)

    out["fase bulan (0-360)"] = phase
    out["jarak ke fase kardinal (hari)"] = np.minimum.reduce(
        [np.abs(((phase - c + 180) % 360) - 180) for c in (0, 90, 180, 270)]) / 360 * SYNODIC
    out["jarak Bumi-Bulan (km)"] = dist_m / 1000
    am, asun = 2.17, 1.0
    out["amplitudo pasang surut"] = np.sqrt(
        am**2 + asun**2 + 2 * am * asun * np.cos(np.radians(2 * phase)))
    out["deklinasi Bulan (deg)"] = dec_m
    out["jarak Bumi-Matahari (AU)"] = dist_s / AU

    ex, ey, _ = heliocentric("Earth", jd)
    sun_lon = (np.degrees(np.arctan2(ey, ex)) + 180) % 360
    geo = {}
    for p in PLANETS:
        px, py, _ = heliocentric(p, jd)
        geo[p] = np.degrees(np.arctan2(py - ey, px - ex)) % 360

    for a, b in [("Venus", "Neptune"), ("Jupiter", "Saturn"), ("Mercury", "Mars"),
                 ("Venus", "Jupiter")]:
        sep = aspect_distance(geo[a], geo[b])
        out[f"sudut {a}-{b} (deg)"] = sep
        out[f"kedekatan aspek {a}-{b}"] = np.minimum.reduce(
            [np.abs(sep - t) for t in (0, 45, 90, 135, 180)])

    # an SSGEOS-style index: how many planet pairs sit near an aspect today
    n_active = np.zeros(len(days))
    for i in range(len(PLANETS)):
        for j in range(i + 1, len(PLANETS)):
            sep = aspect_distance(geo[PLANETS[i]], geo[PLANETS[j]])
            near = np.minimum.reduce([np.abs(sep - t) for t in (0, 45, 90, 135, 180)])
            n_active += (near <= 2.0)
    out["jumlah aspek aktif (+/-2 deg)"] = n_active
    return out


def bin_rates(x, y, n_bins=N_BINS):
    """Fraction of days with an earthquake, per quantile bin of x."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    rates, counts, centres = [], [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() < 50:
            continue
        rates.append(y[m].mean())
        counts.append(int(m.sum()))
        centres.append(0.5 * (edges[i] + edges[i + 1]))
    return np.array(rates), np.array(counts), np.array(centres)


def spread_stat(x, y, n_bins=N_BINS):
    """How much does the rate vary across bins? Weighted standard deviation."""
    rates, counts, _ = bin_rates(x, y, n_bins)
    if len(rates) < 3:
        return 0.0
    w = counts / counts.sum()
    mean = np.sum(w * rates)
    return float(np.sqrt(np.sum(w * (rates - mean) ** 2)))


def circular_shift_null(x, y, n_shift=N_SHIFT, n_bins=N_BINS):
    """Slide the geometry in time; smoothness and periodicity survive intact."""
    n = len(x)
    obs = spread_stat(x, y, n_bins)
    null = np.empty(n_shift)
    offsets = RNG.integers(1, n, n_shift)
    for k, o in enumerate(offsets):
        null[k] = spread_stat(np.roll(x, int(o)), y, n_bins)
    return obs, null, float((null >= obs).mean())


def main():
    days = np.load(HERE / "m6_days.npy")
    cnt = np.load(HERE / "m6_cnt.npy")
    y = (cnt > 0).astype(float)

    rule("LANGKAH 1 -- SATU VARIABEL, SATU TABEL, TANPA MODEL")
    print(f"""
  {len(y):,} hari, 1970-2024. Rata-rata {100*y.mean():.2f}% hari ada gempa M>=6.0.
  Kalau geometri berpengaruh, laju harus BERBEDA antar bin. Sesederhana itu.
""")

    v = variables(days)
    print(f"  {'variabel':<34} {'laju terendah':>14} {'tertinggi':>11} "
          f"{'sebaran':>9} {'p':>7}")
    print("  " + "-" * 80)

    results = []
    for name, x in v.items():
        obs, null, p = circular_shift_null(x, y)
        rates, counts, _ = bin_rates(x, y)
        results.append((name, rates.min(), rates.max(), obs, p))
        flag = "  <--" if p < 0.05 else ""
        print(f"  {name:<34} {100*rates.min():>13.1f}% {100*rates.max():>10.1f}% "
              f"{100*obs:>8.2f}% {p:>7.3f}{flag}")

    alpha = 0.05 / len(results)
    print(f"\n  Bonferroni atas {len(results)} variabel -> ambang p < {alpha:.4f}")
    best = min(results, key=lambda r: r[4])
    print(f"  p terkecil: {best[4]:.3f} ({best[0]})")

    rule("LANGKAH 2 -- LIHAT TABEL LENGKAP DUA KLAIM UTAMA SSGEOS")
    for name in ("jarak ke fase kardinal (hari)", "jumlah aspek aktif (+/-2 deg)"):
        x = v[name]
        rates, counts, centres = bin_rates(x, y)
        obs, null, p = circular_shift_null(x, y)
        print(f"\n  {name}")
        print(f"  {'nilai variabel':>18} {'n hari':>9} {'% ada gempa M6+':>18}")
        print("  " + "-" * 50)
        for c, n_, r_ in zip(centres, counts, rates):
            bar = "#" * int(round(r_ * 100))
            print(f"  {c:>18.2f} {n_:>9,} {100*r_:>17.1f}%  {bar}")
        print(f"  {'rata-rata':>18} {counts.sum():>9,} {100*y.mean():>17.1f}%")
        print(f"  sebaran antar bin {100*obs:.2f} poin, p = {p:.3f}")

    rule("LANGKAH 3 -- SEBERAPA BESAR EFEK YANG BISA TERDETEKSI?")
    print(f"""
  Dengan {len(y):,} hari dan laju dasar {100*y.mean():.1f}%, sebaran antar bin yang
  muncul murni karena kebetulan (persentil 95 dari 2000 pergeseran):
""")
    for name in ("jarak ke fase kardinal (hari)", "amplitudo pasang surut",
                 "jumlah aspek aktif (+/-2 deg)"):
        obs, null, p = circular_shift_null(v[name], y)
        print(f"    {name:<34} ambang {100*np.percentile(null,95):>5.2f}%  "
              f"terukur {100*obs:>5.2f}%")
    print("""
  Artinya: efek nyata yang lebih besar dari ambang itu PASTI terlihat di tabel
  di atas sebagai selisih antar baris. Tidak ada yang tersembunyi di balik
  model -- angkanya dihitung langsung dari data mentah.
""")


if __name__ == "__main__":
    main()
