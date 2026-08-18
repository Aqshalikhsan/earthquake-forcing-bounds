"""
Step 17: the independent test.

Step 16 swept fifteen geometric variables on global M>=6.0 data, 1970-2024, and
found one thing that looked like a signal: on days when four or more planet
pairs sat within 2 degrees of an aspect, the earthquake rate was 31.33% against
28.58% on all other days -- a rate ratio of 1.0962, at p = 0.106.

That is a hypothesis, not a result. It was found by looking at fifteen variables,
so it carries no evidence on the data that produced it. The only honest way to
find out whether it is real is to state it precisely in advance and check it
against data that played no part in generating it.

=========================== PRE-REGISTRATION ===============================
Stated in full before the test is run. Nothing below is adjustable afterwards.

  H1   Days with >= 4 planet pairs within +/-2 deg of 0/45/90/135/180 have a
       HIGHER rate of large earthquakes than other days.
  H0   The rate is the same.

  statistic   rate ratio = P(quake | high-aspect day) / P(quake | other day)
  direction   one-sided, higher. H1 predicts a ratio > 1.
  null        circular time shift of the aspect series, 2000 shifts. Preserves
              smoothness, periodicity and marginal distribution exactly.
  alpha       0.05. One pre-specified hypothesis, so no multiplicity correction.

  PRIMARY DATA -- fully independent of the discovery set:
       global, M >= 6.5, 1900-1969 (USGS). Different years, different events,
       different magnitude threshold. Zero overlap.

  SECONDARY, reported but not decisive (partially overlapping, since M>=7 days
  are a subset of M>=6 days in the discovery window):
       global, M >= 7.0, 1970-2024.

  Discovery-set value to beat: rate ratio 1.0962
  (rate 31.33% on high-aspect days vs 28.58% on the rest, 2,250 days).
============================================================================
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import csv
import io
import numpy as np
from datetime import datetime
from pathlib import Path

from planetary import heliocentric, PLANETS, aspect_distance
from ephem_vec import jd_from_datetime_array

HERE = DATA
N_SHIFT = 2000
ASPECT_TOL = 2.0
HIGH_THRESHOLD = 4   # corrected: this is the cut that reproduces
                     # the step-16 top bin (2,250 days, 11.2%). The earlier
                     # value of 7 was the bin's midpoint label, not its edge.
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def active_aspect_count(days):
    """Number of planet pairs within ASPECT_TOL of 0/45/90/135/180, per day."""
    dts = [datetime.fromordinal(int(d)) for d in days]
    jd = jd_from_datetime_array(dts)
    ex, ey, _ = heliocentric("Earth", jd)
    geo = {}
    for p in PLANETS:
        px, py, _ = heliocentric(p, jd)
        geo[p] = np.degrees(np.arctan2(py - ey, px - ex)) % 360
    n = np.zeros(len(days))
    for i in range(len(PLANETS)):
        for j in range(i + 1, len(PLANETS)):
            sep = aspect_distance(geo[PLANETS[i]], geo[PLANETS[j]])
            near = np.minimum.reduce([np.abs(sep - t) for t in (0, 45, 90, 135, 180)])
            n += (near <= ASPECT_TOL)
    return n


def rate_ratio(high, y):
    a = y[high].mean() if high.any() else np.nan
    b = y[~high].mean() if (~high).any() else np.nan
    return a / b if b > 0 else np.nan, a, b


def run_test(days, cnt, label, threshold=HIGH_THRESHOLD):
    y = (cnt > 0).astype(float)
    n_act = active_aspect_count(days)
    high = n_act >= threshold

    obs, r_hi, r_lo = rate_ratio(high, y)
    null = np.empty(N_SHIFT)
    offsets = RNG.integers(1, len(days), N_SHIFT)
    for k, o in enumerate(offsets):
        h = np.roll(n_act, int(o)) >= threshold
        null[k] = rate_ratio(h, y)[0]
    p = float((null >= obs).mean())

    print(f"\n  {label}")
    print(f"    hari total                    : {len(y):,}")
    print(f"    hari 'aspek tinggi' (>={threshold})     : {int(high.sum()):,} "
          f"({100*high.mean():.1f}%)")
    print(f"    laju pada hari aspek tinggi   : {100*r_hi:.2f}%")
    print(f"    laju pada hari lain           : {100*r_lo:.2f}%")
    print(f"    RASIO                         : {obs:.4f}")
    print(f"    rasio pada null, rata-rata    : {null.mean():.4f}")
    print(f"    rasio pada null, persentil 95 : {np.percentile(null,95):.4f}")
    print(f"    p (satu sisi)                 : {p:.4f}"
          f"   {'<-- LOLOS' if p < 0.05 else '<-- tidak lolos'}")
    return obs, p


def load_daily(path, start_year, end_year, min_mag):
    from collections import Counter
    c = Counter()
    for rec in csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"))):
        if rec.get("type", "earthquake") != "earthquake":
            continue
        try:
            if float(rec["mag"]) < min_mag:
                continue
            c[datetime.strptime(rec["time"][:10], "%Y-%m-%d").toordinal()] += 1
        except (ValueError, TypeError, KeyError):
            continue
    d0 = datetime(start_year, 1, 1).toordinal()
    d1 = datetime(end_year, 1, 1).toordinal()
    days = np.arange(d0, d1)
    return days, np.array([c.get(int(d), 0) for d in days])


def main():
    print(__doc__)

    rule("HASIL UJI INDEPENDEN")

    days_h = np.load(HERE / "hist_days.npy")
    cnt_h = np.load(HERE / "hist_cnt.npy")
    obs1, p1 = run_test(days_h, cnt_h,
                        "PRIMER: global M>=6.5, 1900-1969 (independen penuh)")

    days_s, cnt_s = load_daily(HERE / "global_m6.csv", 1970, 2025, 7.0)
    obs2, p2 = run_test(days_s, cnt_s,
                        "SEKUNDER: global M>=7.0, 1970-2024 (sebagian tumpang tindih)")

    rule("PEMBANDING: nilai yang ditemukan di data penemuan")
    days_d = np.load(HERE / "m6_days.npy")
    cnt_d = np.load(HERE / "m6_cnt.npy")
    obs0, p0 = run_test(days_d, cnt_d,
                        "PENEMUAN: global M>=6.0, 1970-2024 (BUKAN bukti)")

    rule("VONIS")
    passed = p1 < 0.05
    print(f"""
  Data penemuan   : rasio {obs0:.4f}, p = {p0:.4f}   (tidak dihitung sebagai bukti)
  Uji PRIMER      : rasio {obs1:.4f}, p = {p1:.4f}
  Uji sekunder    : rasio {obs2:.4f}, p = {p2:.4f}

  {'HIPOTESIS LOLOS uji independen.' if passed
    else 'HIPOTESIS TIDAK LOLOS uji independen.'}

  Yang membuat uji ini berarti: aturannya dikunci sebelum dijalankan, datanya
  tidak ikut membentuk hipotesis, arahnya ditentukan di muka, dan null-nya
  mempertahankan seluruh struktur deret kecuali kaitannya dengan tanggal.

  Kalau rasio pada data penemuan tinggi tetapi pada data independen tidak,
  maka yang kita lihat di langkah 16 adalah derau yang kebetulan menonjol dari
  lima belas variabel yang disapu -- bukan efek.
""")


if __name__ == "__main__":
    main()
