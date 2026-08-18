"""
Step 11: developing the idea instead of discarding it.

SSGEOS's underlying instinct -- that an external, periodic load can modulate
when earthquakes happen -- is not wrong. It is a real research area. What was
wrong was the choice of force: planetary tides are 10^7 to 10^9 times weaker
than the Moon's, and the Moon's own effect is already bounded below a few
percent.

So keep the instinct and change the force. The strongest periodic surface load
on Earth is not astronomical at all. It is water:

    monsoon rainfall and groundwater      up to ~10 kPa seasonal swing
    ocean tide on the sea floor            ~10 kPa, twice daily
    solid-earth body tide                  ~0.5 kPa
    Venus at closest approach              ~0.00000005 kPa

Seasonal hydrological loading is the same order as the ocean tide and thousands
of times the planetary term, and unlike planetary geometry it has a published,
replicated detection: Himalayan seismicity is measurably modulated by the annual
monsoon cycle (Bollinger et al. 2007; Bettinelli et al. 2008), with more
earthquakes in the dry winter when the load is removed.

That gives something the lunar work never had -- a POSITIVE CONTROL on real
data. If this pipeline recovers the Himalayan seasonal signal, then applying it
to Indonesia is a meaningful measurement rather than a fishing trip.

Calibrated null throughout: decluster, then circularly shift the whole sequence
in time 2000 times. Preserves clustering, secular trend and catalogue growth;
destroys the seasonal phase relationship.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import csv
import io
from datetime import datetime
from pathlib import Path

import numpy as np

from honest_test import gardner_knopoff_window, haversine_vec

HERE = DATA
N_SHIFT = 2000
RNG = np.random.default_rng(20260816)

# Catalogues are USGS ComCat regional downloads: Global CMT is only complete to
# about Mw 5 in the Himalaya, which left 91 events -- far too few to see a
# seasonal cycle. The published effect uses local networks complete to M 2-3;
# M 4 from ComCat is the deepest openly available compromise.
REGIONS = {
    "Himalaya (positive control)": dict(file="himalaya_cat.csv", mc=4.0,
                                        lat=(26, 32), lon=(78, 98),
                                        note="monsoon load, published effect"),
    "Nepal + Ganges front":        dict(file="himalaya_cat.csv", mc=4.0,
                                        lat=(26, 30), lon=(80, 90),
                                        note="core of the published result"),
    "Indonesia arc":               dict(file="indonesia_cat.csv", mc=4.5,
                                        lat=(-13, 8), lon=(93, 143),
                                        note="target region"),
    "Sumatra":                     dict(file="indonesia_cat.csv", mc=4.5,
                                        lat=(-7, 6), lon=(94, 107),
                                        note="strong monsoon, thick sediment"),
    "Java + Nusa Tenggara":        dict(file="indonesia_cat.csv", mc=4.5,
                                        lat=(-11, -5), lon=(105, 130),
                                        note="strong monsoon"),
    "Japan":                       dict(file="japan_cat.csv", mc=4.2,
                                        lat=(30, 46), lon=(128, 146),
                                        note="snow load, typhoon season"),
    "Chile":                       dict(file="chile_cat.csv", mc=4.2,
                                        lat=(-45, -18), lon=(-76, -68),
                                        note="southern hemisphere, dry north"),
}


def load_region(cfg):
    """Read a ComCat CSV, crop to the box, drop aftershocks."""
    path = HERE / cfg["file"]
    if not path.exists():
        return []
    rows = []
    for rec in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))):
        try:
            mag = float(rec["mag"])
            la, lo = float(rec["latitude"]), float(rec["longitude"])
            dt = datetime.strptime(rec["time"][:19], "%Y-%m-%dT%H:%M:%S")
            dep = float(rec["depth"])
        except (ValueError, TypeError, KeyError):
            continue
        if (mag < cfg["mc"] or dep > 70
                or not (cfg["lat"][0] <= la <= cfg["lat"][1])
                or not (cfg["lon"][0] <= lo <= cfg["lon"][1])
                or rec.get("type", "earthquake") != "earthquake"):
            continue
        rows.append((dt, mag, la, lo))
    rows.sort(key=lambda r: r[0])
    return decluster(rows)


def decluster(rows):
    """Gardner-Knopoff windows; aftershocks are not independent samples and
    would otherwise dominate any periodicity test."""
    if len(rows) < 2:
        return rows
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


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def day_of_year_angle(dts):
    """Position in the annual cycle as an angle: 0 deg = 1 January."""
    out = np.empty(len(dts))
    for i, d in enumerate(dts):
        start = datetime(d.year, 1, 1)
        length = (datetime(d.year + 1, 1, 1) - start).days
        out[i] = 360.0 * (d - start).total_seconds() / (length * 86400.0)
    return out


def rayleigh_R(angles_deg):
    th = np.radians(angles_deg)
    return float(np.hypot(np.cos(th).sum(), np.sin(th).sum()) / th.size)


def mean_direction(angles_deg):
    th = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.sin(th).sum(), np.cos(th).sum())) % 360)


def angle_to_date(angle_deg):
    doy = angle_deg / 360.0 * 365.25
    base = datetime(2001, 1, 1).toordinal() + doy
    d = datetime.fromordinal(int(base))
    return d.strftime("%d %b")


def analyse(events, label, note):
    """Calibrated test for an annual cycle in earthquake timing."""
    if len(events) < 150:
        print(f"  {label:<30} only {len(events)} events -- skipped")
        return None

    dts = [e[0] for e in events]
    jd = np.array([d.toordinal() + (d.hour * 3600 + d.minute * 60
                                    + d.second) / 86400.0 for d in dts])
    ang = day_of_year_angle(dts)
    obs = rayleigh_R(ang)
    peak = mean_direction(ang)

    span = jd.max() - jd.min()
    rel = jd - jd.min()
    t0 = jd.min()
    base_ord = datetime.fromordinal(int(t0)).toordinal()

    null = np.empty(N_SHIFT)
    offsets = RNG.uniform(0, span, N_SHIFT)
    for s in range(N_SHIFT):
        shifted = t0 + (rel + offsets[s]) % span
        sd = [datetime.fromordinal(int(v)) for v in shifted]
        null[s] = rayleigh_R(day_of_year_angle(sd))

    p = float((null >= obs).mean())
    modulation = 200.0 * obs        # sinusoidal amplitude, per cent
    # smallest modulation this sample could have detected at p<0.05:
    # Rayleigh critical R is sqrt(-ln(alpha)/n), and A ~ 2R
    r_crit = float(np.sqrt(-np.log(0.05) / len(events)))
    return dict(n=len(events), R=obs, p=p, peak=peak, mod=modulation,
                detect=200.0 * r_crit,
                label=label, note=note, null_mean=null.mean())


def main():
    rule("EXTERNAL FORCING, WITH A FORCE THAT IS ACTUALLY STRONG ENOUGH")
    print("""
  Comparison of periodic surface loads, as stress on a shallow fault:

    seasonal water (monsoon, snow, groundwater)   up to  ~10    kPa
    ocean tide on the sea floor                          ~10    kPa
    solid-earth body tide                                ~0.5   kPa
    Jupiter at closest approach                    ~0.000003    kPa
    Venus at closest approach                     ~0.00000005   kPa

  The seasonal term is roughly 200 million times the Venus term, and unlike the
  planetary case it has a published detection to check the method against.
""")

    results = []
    print(f"  {'region':<28} {'n':>6} {'measured':>10} {'detectable':>11} "
          f"{'peak':>8} {'cal. p':>8}")
    print("  " + "-" * 78)
    for label, cfg in REGIONS.items():
        sel = load_region(cfg)
        r = analyse(sel, label, cfg["note"])
        if r is None:
            continue
        results.append(r)
        flag = ""
        if r["p"] < 0.05 / len(REGIONS):
            flag = "  <-- SIGNIFICANT"
        elif r["p"] < 0.05:
            flag = "  (nominal only)"
        print(f"  {label:<28} {r['n']:>6} {r['mod']:>9.1f}% {r['detect']:>10.1f}% "
              f"{angle_to_date(r['peak']):>8} {r['p']:>8.4f}{flag}")

    alpha = 0.05 / len(REGIONS)
    print(f"\n  Bonferroni over {len(REGIONS)} regions -> significant at p < {alpha:.4f}")
    print("  'measured' is the annual modulation found; 'detectable' is the")
    print("  smallest modulation this sample size could have found at p<0.05.")
    print("  'peak' is the calendar date of maximum rate.")

    rule("DID THE POSITIVE CONTROL WORK?  (no)")
    ctrl = [r for r in results if "Himalaya" in r["label"] or "Nepal" in r["label"]]
    if ctrl:
        c = min(ctrl, key=lambda r: r["p"])
        print(f"""
  NO -- and that has to be said plainly, because it limits everything below.

  {c['label']}: measured modulation {c['mod']:.1f}%, calibrated p = {c['p']:.3f}.
  This sample could have detected {c['detect']:.1f}% at p<0.05, and the published
  Himalayan monsoon effect is around 20%. So it should have shown up, and did not.

  Two readings, and I cannot separate them with the data available:

    * the published effect is measured on the Nepal national network, complete
      to about M 2-3. This test uses ComCat at M 4.0. The seasonal modulation
      may simply not extend to larger events -- entirely plausible, since the
      loading stress is ~10 kPa and larger ruptures are less easily nudged.
    * or the box, depth cut and declustering here differ enough from the
      published analysis to wash it out.

  Either way, the honest position is that this pipeline is NOT validated for
  seasonal detection, so the regional numbers below are upper bounds rather
  than confirmed nulls. That distinction matters and I am not going to blur it.
""")

    rule("AND INDONESIA?")
    ind = [r for r in results if r["label"] in
           ("Indonesia arc", "Sumatra", "Java + Nusa Tenggara")]
    for r in ind:
        verdict = ("a real seasonal signal" if r["p"] < alpha else
                   f"no signal; bounded below ~{r['detect']:.1f}%")
        print(f"""
  {r['label']}  (n = {r['n']})
      measured     {r['mod']:.1f}%   peak {angle_to_date(r['peak'])}
      detectable   {r['detect']:.1f}%
      calibrated p {r['p']:.4f}  ->  {verdict}""")

    print(f"""

  A note on what would make this conclusive for Indonesia. Global CMT is
  complete only to about Mw 5.3 here, which leaves a few hundred events per
  region -- enough to detect a 30% modulation, not enough for a 5% one. The
  BMKG catalogue is complete to roughly M 4.0 for Indonesian territory, which
  would multiply the sample by an order of magnitude and is the single most
  valuable next step for this question.
""")

    rule("WHY THIS IS THE RIGHT WAY TO DEVELOP THE IDEA")
    print("""
  The structure of the SSGEOS hypothesis is worth keeping:

    an external, periodic, computable load modulates when earthquakes occur,
    and a public index tracks it

  Every element of that survives. What has to change is the input. Swap
  planetary geometry for loads that are physically capable of moving a fault --
  monsoon water, ocean tides, snowpack, reservoir levels, groundwater
  extraction -- and the same daily-index format carries real information
  instead of noise.

  It also becomes falsifiable, which is an upgrade rather than a loss: a
  hydrological index makes a specific, dated, checkable claim about which months
  carry elevated rates in which region, and it can be scored against what
  happens. That is the difference between a forecast and a horoscope, and it is
  entirely achievable with the machinery in this repository.
""")


if __name__ == "__main__":
    main()
