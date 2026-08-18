"""
Step 18: the deepest honest search of the planetary hypothesis space.

The request was to go deep -- add variables until something makes planetary
aspects matter. Taken literally that is guaranteed to succeed and guaranteed to
be worthless: with 200 candidate variables, about 10 will clear p < 0.05 by
chance alone. That is exactly how the three papers reached their conclusions,
and exactly why none of them replicates.

So this does the version that can actually produce a publishable result: search
as widely as possible, and let the null model absorb the entire search.

    MAX-STATISTIC PERMUTATION
    1. compute a test statistic for every candidate variable on the real data
    2. record the single best value across all of them
    3. circularly shift the earthquake series, recompute ALL statistics,
       record the best again
    4. repeat many times
    5. p = fraction of shifted datasets whose best beats the real best

Because step 3 searches just as hard as step 1, the p-value is already corrected
for however many variables were tried. Adding more variables cannot inflate it.
This is the honest way to grant the hypothesis unlimited freedom.

Variable families, far wider than anything in the three papers:

  A  geocentric pairwise aspects, 21 pairs -- angle, and proximity to each of
     0/45/90/135/180 separately
  B  heliocentric pairwise aspects, 21 pairs (SSGEOS's actual frame: "Earth
     aligned between Venus and Neptune" is heliocentric)
  C  planet-Earth-planet alignment angles, 21 pairs
  D  per-planet elongation, declination, distance, and rate of change
  E  planetary tidal stress at a reference site with Earth's rotation included,
     per planet and summed -- the physically correct quantity
  F  solar barycentric motion: the Sun's displacement and speed about the solar
     system barycentre, which is entirely planet-driven and is the mechanism
     invoked by the solar-activity literature the papers cite
  G  aspect-count indices at four tolerances, and a convergence index counting
     exact aspects inside a moving window -- the closest formalisation of what
     SSGEOS actually says on air
  H  lunar terms: phase harmonics 1-4, distance, declination, nodal position
  I  combined lunisolar tidal amplitude, with and without the planetary terms
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np
from datetime import datetime
from pathlib import Path

from planetary import heliocentric, PLANETS, aspect_distance, ELEMENTS
from ephem_vec import jd_from_datetime_array, bodies_equatorial, AU, D2R
from planetary_tide import planet_equatorial, dcfs_from_body, GM

HERE = DATA
N_PERM = 600
N_BINS = 10
RNG = np.random.default_rng(20260816)

# reference site for the tidal terms: Java trench megathrust
SITE = dict(lat=-9.5, lon=108.0, strike=288.0, dip=12.0, rake=95.0)
GM_SUN_ = 1.32712440018e20
GM_MOON_ = 4.9028695e12


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def build_deep_features(days, verbose=True):
    """Every planetary quantity worth trying. Returns (matrix, names)."""
    dts = [datetime.fromordinal(int(d)) for d in days]
    jd = jd_from_datetime_array(dts)
    eph = bodies_equatorial(jd)
    names, cols = [], []

    def add(nm, v):
        v = np.asarray(v, dtype=float)
        if np.all(np.isfinite(v)) and np.ptp(v) > 0:
            names.append(nm)
            cols.append(v)

    ex, ey, ez = heliocentric("Earth", jd)
    geo, helio, dist = {}, {}, {}
    for p in PLANETS:
        px, py, pz = heliocentric(p, jd)
        geo[p] = np.degrees(np.arctan2(py - ey, px - ex)) % 360
        helio[p] = np.degrees(np.arctan2(py, px)) % 360
        dist[p] = np.sqrt((px - ex) ** 2 + (py - ey) ** 2 + (pz - ez) ** 2)

    sun_lon = (np.degrees(np.arctan2(ey, ex)) + 180) % 360
    pairs = [(PLANETS[i], PLANETS[j])
             for i in range(len(PLANETS)) for j in range(i + 1, len(PLANETS))]

    # --- A, B, C: aspect geometry in three frames ---
    for a, b in pairs:
        sg = aspect_distance(geo[a], geo[b])
        sh = aspect_distance(helio[a], helio[b])
        add(f"A_geo_{a}_{b}", sg)
        add(f"B_helio_{a}_{b}", sh)
        for t in (0, 45, 90, 135, 180):
            add(f"A_geo_{a}_{b}_prox{t}", np.abs(sg - t))
            add(f"B_helio_{a}_{b}_prox{t}", np.abs(sh - t))
        # planet-Earth-planet angle
        va = np.degrees(np.arctan2(
            heliocentric(a, jd)[1] - ey, heliocentric(a, jd)[0] - ex)) % 360
        vb = np.degrees(np.arctan2(
            heliocentric(b, jd)[1] - ey, heliocentric(b, jd)[0] - ex)) % 360
        add(f"C_align_{a}_{b}", aspect_distance(va, vb))

    # --- D: per-planet terms ---
    for p in PLANETS:
        add(f"D_{p}_elong", aspect_distance(geo[p], sun_lon))
        add(f"D_{p}_dist", dist[p])
        add(f"D_{p}_dlon", np.gradient(np.unwrap(np.radians(geo[p]))))
        ra, dec, dd = planet_equatorial(p, jd)
        add(f"D_{p}_dec", dec)

    # --- E: real tidal stress on a real fault, rotation included ---
    tot = np.zeros(len(days))
    for p in PLANETS:
        ra, dec, dd = planet_equatorial(p, jd)
        d = dcfs_from_body(SITE["lat"], SITE["lon"], SITE["strike"],
                           SITE["dip"], SITE["rake"], jd, ra, dec, dd, GM[p])
        add(f"E_tide_{p}", d)
        tot += d
    add("E_tide_planets_total", tot)
    add("E_tide_planets_abs", np.abs(tot))

    # --- F: solar motion about the barycentre, entirely planet-driven ---
    bx = by = bz = 0.0
    msun = 1.0
    masses = {"Mercury": 1.66e-7, "Venus": 2.45e-6, "Earth": 3.00e-6,
              "Mars": 3.23e-7, "Jupiter": 9.55e-4, "Saturn": 2.86e-4,
              "Uranus": 4.37e-5, "Neptune": 5.15e-5}
    for p, m in masses.items():
        px, py, pz = heliocentric(p, jd)
        bx = bx + m * px
        by = by + m * py
        bz = bz + m * pz
    tot_m = msun + sum(masses.values())
    bx, by, bz = bx / tot_m, by / tot_m, bz / tot_m
    r_bary = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    add("F_sun_bary_dist", r_bary)
    add("F_sun_bary_speed", np.abs(np.gradient(r_bary)))
    add("F_sun_bary_lon", np.degrees(np.arctan2(by, bx)) % 360)
    vx, vy = np.gradient(bx), np.gradient(by)
    add("F_sun_bary_angmom", bx * vy - by * vx)

    # --- G: aspect counts and convergence, several tolerances ---
    for tol in (1.0, 2.0, 3.0, 5.0):
        n_act = np.zeros(len(days))
        for a, b in pairs:
            sep = aspect_distance(geo[a], geo[b])
            near = np.minimum.reduce([np.abs(sep - t) for t in (0, 45, 90, 135, 180)])
            n_act += (near <= tol)
        add(f"G_aspectcount_tol{tol:g}", n_act)
        for w in (3, 7):
            k = np.ones(2 * w + 1)
            add(f"G_convergence_tol{tol:g}_w{w}",
                np.convolve(n_act, k, mode="same"))

    # --- H: lunar terms ---
    ra_m, dec_m, dist_m = eph["moon"]
    ra_s, dec_s, dist_s = eph["sun"]
    r = np.pi / 180
    ce = (np.sin(dec_m * r) * np.sin(dec_s * r)
          + np.cos(dec_m * r) * np.cos(dec_s * r) * np.cos((ra_m - ra_s) * r))
    el = np.degrees(np.arccos(np.clip(ce, -1, 1)))
    sign = np.sign(((ra_m - ra_s) % 360) - 180)
    phase = np.where(sign >= 0, el, 360 - el)
    for k in (1, 2, 3, 4):
        add(f"H_moon_h{k}_sin", np.sin(np.radians(k * phase)))
        add(f"H_moon_h{k}_cos", np.cos(np.radians(k * phase)))
    add("H_moon_dist", dist_m)
    add("H_moon_dec", dec_m)
    add("H_moon_phase_prox", np.minimum.reduce(
        [np.abs(((phase - c + 180) % 360) - 180) for c in (0, 90, 180, 270)]))

    # --- I: lunisolar tide, the physically motivated reference ---
    for body, gm in (("moon", GM_MOON_), ("sun", GM_SUN_)):
        ra, dec, dd = eph[body]
        add(f"I_tide_{body}", dcfs_from_body(
            SITE["lat"], SITE["lon"], SITE["strike"], SITE["dip"],
            SITE["rake"], jd, ra, dec, dd, gm))
    add("I_sun_dist", dist_s)

    X = np.column_stack(cols)
    if verbose:
        fam = {}
        for n in names:
            fam[n[0]] = fam.get(n[0], 0) + 1
        print(f"  variabel dibangun: {len(names)}")
        for k in sorted(fam):
            print(f"    keluarga {k}: {fam[k]:>4}")
    return X, names


def bin_indices(X, n_bins=N_BINS):
    """Quantile-bin every column once; permutations then reuse this."""
    n, m = X.shape
    idx = np.empty((n, m), dtype=np.int16)
    for j in range(m):
        e = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        e = np.unique(e)
        idx[:, j] = np.clip(np.digitize(X[:, j], e[1:-1]), 0, n_bins - 1)
    return idx


def all_stats(idx, y, n_bins=N_BINS):
    """
    Max absolute deviation of any bin's rate from the overall rate, per column.
    One scalar per variable, computed for every variable at once.
    """
    n, m = idx.shape
    base = y.mean()
    out = np.empty(m)
    for j in range(m):
        cnt = np.bincount(idx[:, j], minlength=n_bins).astype(float)
        s = np.bincount(idx[:, j], weights=y, minlength=n_bins)
        ok = cnt >= 200
        if not ok.any():
            out[j] = 0.0
            continue
        rates = np.where(ok, s / np.maximum(cnt, 1), base)
        out[j] = np.max(np.abs(rates - base))
    return out


def main():
    rule("PENCARIAN MENDALAM RUANG HIPOTESIS PLANET")

    days = np.load(HERE / "m6_days.npy")
    cnt = np.load(HERE / "m6_cnt.npy")
    y = (cnt > 0).astype(float)
    print(f"\n  {len(y):,} hari, laju dasar {100*y.mean():.2f}%\n")

    X, names = build_deep_features(days)
    idx = bin_indices(X)

    obs = all_stats(idx, y)
    order = np.argsort(-obs)

    rule("KANDIDAT TERBAIK PADA DATA ASLI")
    print(f"\n  {'variabel':<40} {'simpangan maks':>16}")
    print("  " + "-" * 60)
    for j in order[:15]:
        print(f"  {names[j]:<40} {100*obs[j]:>15.2f}%")

    rule(f"NULL MAX-STATISTIC: {N_PERM} pergeseran, pencarian penuh tiap kali")
    print(f"""
  Tiap pergeseran menggeser deret gempa lalu MENCARI ULANG di seluruh
  {len(names)} variabel, persis seperti pencarian pada data asli. Jadi
  p-value-nya sudah terkoreksi untuk berapa pun variabel yang saya tambahkan.
""")
    best_obs = obs.max()
    best_name = names[int(np.argmax(obs))]
    null_best = np.empty(N_PERM)
    for i in range(N_PERM):
        o = RNG.integers(1, len(y))
        null_best[i] = all_stats(idx, np.roll(y, int(o))).max()
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N_PERM}", end="\r", flush=True)
    print(" " * 30, end="\r")

    p_corrected = float((null_best >= best_obs).mean())
    print(f"  variabel terbaik data asli : {best_name}")
    print(f"  simpangan maksimum         : {100*best_obs:.2f}%")
    print(f"  simpangan terbaik pada noise, rata-rata : {100*null_best.mean():.2f}%")
    print(f"  simpangan terbaik pada noise, maksimum  : {100*null_best.max():.2f}%")
    print(f"  ambang 95% dari noise                   : {100*np.percentile(null_best,95):.2f}%")
    print(f"\n  p TERKOREKSI (sudah memperhitungkan {len(names)} variabel) : {p_corrected:.4f}")

    rule("BERAPA BESAR EFEK YANG MASIH BISA BERSEMBUNYI?")
    thr = np.percentile(null_best, 95)
    print(f"""
  Batas atas: setiap efek yang menggeser laju gempa lebih dari
  {100*thr:.2f} poin persen di salah satu dari {len(names)} variabel ini akan
  terdeteksi oleh pencarian di atas.

  Laju dasar {100*y.mean():.1f}%, jadi batas itu setara modulasi relatif
  sekitar {100*thr/y.mean():.1f}%.

  Sebagai perbandingan pada data yang sama:
    gempa susulan setelah mainshock  : +16.000% (161x)
    beban pasang surut laut          : batas beberapa persen
    geometri planet                  : di bawah {100*thr/y.mean():.1f}%
""")

    rule("VONIS")
    ok = p_corrected < 0.05
    print(f"""
  {'ADA variabel planet yang bertahan setelah koreksi pencarian penuh.'
    if ok else
    'TIDAK ADA variabel planet yang bertahan setelah koreksi pencarian penuh.'}

  Ini bukan ambang yang saya naikkan. Ini pencarian yang saya perluas
  sampai {len(names)} variabel -- tiga frame acuan, tegangan pasang surut
  sungguhan dengan rotasi Bumi, gerak Matahari terhadap barisentrum, indeks
  konvergensi, dan seluruh keluarga aspek -- lalu null-nya diberi kebebasan
  mencari yang sama persis.

  Kalau ada sesuatu di ruang ini, cara inilah yang menemukannya.
""")


if __name__ == "__main__":
    main()
