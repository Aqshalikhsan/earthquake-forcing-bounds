"""
Step 12: keep the product, replace the engine.

SSGEOS has something genuinely scarce -- a regularly updated public index that
people actually check, ending with "have an earthquake plan". The format works.
The audience is real. What fails is the input: planetary geometry carries no
information, as measured across steps 1-11.

So this keeps the concept and swaps the engine. Same idea -- a daily index, by
region, that says where seismic rates are currently elevated and by how much --
driven by the model that scored 863:1 against the lunar one on the same data.

Honest about its own limits, in the output, every time it runs:

  * this is NOT a prediction of a coming earthquake
  * elevated rate means aftershocks are likely where shaking already happened
  * a quiet index does NOT mean a region is safe -- the largest earthquakes
    arrive with no warning at all, and most of Indonesia's deadliest ruptures
    began on a quiet day

Run it daily. It fetches the recent catalogue itself.

    python daily_index.py                # last 30 days, forecast next 7
    python daily_index.py --days 14 --window 1
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import argparse
import csv
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = DATA

# ETAS parameters fitted on Indonesia 1990-2015 in forecast.py (spatio-temporal)
ETAS = dict(mu_tot=0.506, K=0.0153, alpha=1.49, c=0.0273, p=1.226,
            D=8.2, gamma=0.52, q=2.16)
MC = 5.0
KM_PER_DEG = 111.19
T_CUT = 500.0

ZONES = [
    ("Aceh - North Sumatra",      (1.0, 6.5),   (94.0, 99.5)),
    ("West Sumatra - Mentawai",   (-3.5, 1.0),  (97.0, 102.0)),
    ("South Sumatra - Lampung",   (-7.0, -3.5), (100.0, 106.0)),
    ("West Java - Banten",        (-9.0, -5.5), (105.0, 109.0)),
    ("Central & East Java",       (-10.0, -6.0), (109.0, 115.0)),
    ("Bali - West Nusa Tenggara", (-11.0, -7.0), (114.0, 119.5)),
    ("East Nusa Tenggara - Flores", (-11.5, -7.0), (119.5, 127.0)),
    ("Sulawesi",                  (-6.5, 3.0),  (118.0, 126.0)),
    ("Maluku - Banda Sea",        (-8.5, 3.0),  (125.0, 132.0)),
    ("Papua",                     (-9.5, 1.0),  (130.0, 141.5)),
]

# Long-term background rate of M>=5 per zone per 7 days, from 1990-2026 ComCat.
# Computed once by calibrate(); stored so the daily run needs no history fetch.
BASELINE = {
    "Aceh - North Sumatra": 0.36, "West Sumatra - Mentawai": 0.33,
    "South Sumatra - Lampung": 0.20, "West Java - Banten": 0.16,
    "Central & East Java": 0.17, "Bali - West Nusa Tenggara": 0.16,
    "East Nusa Tenggara - Flores": 0.30, "Sulawesi": 0.42,
    "Maluku - Banda Sea": 0.62, "Papua": 0.53,
}


def xy_km(lat, lon):
    return lon * KM_PER_DEG * np.cos(np.radians(-2.0)), lat * KM_PER_DEG


def fetch_recent(days):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + T_CUT)
    r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",
                     params=dict(format="csv", orderby="time-asc",
                                 starttime=start.strftime("%Y-%m-%d"),
                                 endtime=end.strftime("%Y-%m-%d"),
                                 minmagnitude=MC, minlatitude=-13,
                                 maxlatitude=8, minlongitude=93,
                                 maxlongitude=143), timeout=120)
    r.raise_for_status()
    out = []
    for rec in csv.DictReader(io.StringIO(r.text)):
        try:
            m = float(rec["mag"])
            dt = datetime.strptime(rec["time"][:19], "%Y-%m-%dT%H:%M:%S")
            la, lo = float(rec["latitude"]), float(rec["longitude"])
        except (ValueError, TypeError, KeyError):
            continue
        if m >= MC and rec.get("type", "earthquake") == "earthquake":
            out.append((dt, m, la, lo, rec.get("place", "")))
    out.sort(key=lambda e: e[0])
    return out, end


def zone_rate(events, now, window_days, lat_rng, lon_rng, nsample=400):
    """
    Expected number of M>=MC in the zone over the next `window_days`, from the
    ETAS triggering of everything that has already happened. The background is
    handled separately by BASELINE, so this returns only the triggered part.
    """
    if not events:
        return 0.0, []
    P = ETAS
    t_now = 0.0
    days_ago = np.array([(now.replace(tzinfo=None) - e[0]).total_seconds() / 86400.0
                         for e in events])
    mags = np.array([e[1] for e in events])
    lats = np.array([e[2] for e in events])
    lons = np.array([e[3] for e in events])

    keep = (days_ago >= 0) & (days_ago < T_CUT)
    if not keep.any():
        return 0.0, []

    # Monte Carlo the zone area to integrate the spatial kernel
    rng = np.random.default_rng(7)
    qlat = rng.uniform(lat_rng[0], lat_rng[1], nsample)
    qlon = rng.uniform(lon_rng[0], lon_rng[1], nsample)
    qx, qy = xy_km(qlat, qlon)
    area = ((lat_rng[1] - lat_rng[0]) * KM_PER_DEG
            * (lon_rng[1] - lon_rng[0]) * KM_PER_DEG
            * np.cos(np.radians(-2.0)))

    ex, ey = xy_km(lats[keep], lons[keep])
    dm = mags[keep] - MC
    t_i = -days_ago[keep]                        # negative = in the past

    omori = (P["K"] * np.exp(P["alpha"] * dm)
             * ((window_days - t_i + P["c"]) ** (1 - P["p"])
                - (0.0 - t_i + P["c"]) ** (1 - P["p"])) / (1 - P["p"]))

    d2 = P["D"] ** 2 * 10 ** (P["gamma"] * dm)
    total = 0.0
    contrib = np.zeros(dm.size)
    for k in range(dm.size):
        r2 = (qx - ex[k]) ** 2 + (qy - ey[k]) ** 2
        f = (P["q"] - 1) / (np.pi * d2[k]) * (1 + r2 / d2[k]) ** (-P["q"])
        share = f.mean() * area
        contrib[k] = omori[k] * share
        total += contrib[k]

    idx = np.argsort(-contrib)[:3]
    drivers = [(events[np.flatnonzero(keep)[i]], contrib[i])
               for i in idx if contrib[i] > 0.02]
    return float(total), drivers


def status_for(ratio):
    if ratio >= 5.0:
        return "ELEVATED - active sequence"
    if ratio >= 2.0:
        return "above normal"
    if ratio >= 1.3:
        return "slightly above normal"
    return "normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="how much recent activity to summarise")
    ap.add_argument("--window", type=float, default=7.0,
                    help="forecast window in days")
    args = ap.parse_args()

    events, now = fetch_recent(args.days)
    recent = [e for e in events
              if (now.replace(tzinfo=None) - e[0]).days <= args.days]

    print("=" * 78)
    print(f"  INDONESIA SEISMIC ACTIVITY INDEX".center(78))
    print(f"  {now.strftime('%d %B %Y, %H:%M UTC')}".center(78))
    print("=" * 78)
    print(f"""
  Forecast window   next {args.window:.0f} days, M >= {MC:.0f}
  Engine            spatio-temporal ETAS (Ogata 1988), fitted on 1990-2015
  Catalogue         USGS ComCat, live
  Events M>={MC:.0f} in the last {args.days} days: {len(recent)}
""")

    rows = []
    for name, lat_rng, lon_rng in ZONES:
        trig, drivers = zone_rate(events, now, args.window, lat_rng, lon_rng)
        base = BASELINE[name] * (args.window / 7.0)
        total = base + trig
        ratio = total / base if base > 0 else 1.0
        rows.append((name, total, base, ratio, drivers))

    rows.sort(key=lambda r: -r[3])

    print(f"  {'zone':<30} {'expected':>9} {'normal':>8} {'ratio':>7}  status")
    print("  " + "-" * 76)
    for name, total, base, ratio, _ in rows:
        print(f"  {name:<30} {total:>9.2f} {base:>8.2f} {ratio:>6.1f}x  "
              f"{status_for(ratio)}")

    active = [r for r in rows if r[3] >= 2.0]
    if active:
        print("\n  WHAT IS DRIVING THE ELEVATED ZONES")
        for name, _, _, ratio, drivers in active:
            print(f"\n    {name}  ({ratio:.1f}x normal)")
            for (dt, mag, la, lo, place), c in drivers:
                ago = (now.replace(tzinfo=None) - dt).total_seconds() / 86400.0
                print(f"      M{mag:.1f}  {ago:5.1f} d ago  {place[:44]}")
                print(f"            contributes {c:.2f} expected aftershocks")
    else:
        print("\n  No zone is significantly elevated. No active sequence is driving")
        print("  aftershock rates above their long-term background.")

    biggest = max(recent, key=lambda e: e[1]) if recent else None
    if biggest:
        dt, mag, la, lo, place = biggest
        ago = (now.replace(tzinfo=None) - dt).total_seconds() / 86400.0
        print(f"\n  Largest event in the period: M{mag:.1f}, {ago:.1f} days ago")
        print(f"    {place}")

    print(f"""
{'=' * 78}
  HOW TO READ THIS -- and it matters more than the numbers above
{'=' * 78}

  This is NOT a prediction that an earthquake is coming.

  "2x normal" means that if a hundred weeks looked like this one, about twice
  as many M{MC:.0f}+ earthquakes would occur in that zone as in an average week. In a
  zone whose normal rate is 0.2 per week, twice normal is still only 0.4. The
  ratio is real and it is measured, but it is a rate, not an event.

  Elevated zones are elevated because an earthquake ALREADY happened there.
  What follows is aftershocks, near the same place, decaying by Omori's law.
  That is genuinely useful -- it is why engineers delay re-entry into damaged
  buildings and why search teams work with a known risk rather than a guessed
  one.

  A quiet index does NOT mean a region is safe. Every one of Indonesia's
  deadliest earthquakes -- Yogyakarta 2006, Palu 2018, Cianjur 2022 -- began on
  an ordinary day with no elevated rate anywhere. There is no known precursor.
  This index cannot see them coming and neither can anything else.

  What actually protects you does not depend on this page at all: a house built
  or retrofitted to code, heavy furniture anchored, an agreed meeting point, a
  route to high ground if you are near the coast, and water and a torch you can
  find in the dark.

  Have an earthquake plan. Not because a forecast says so -- because you live
  on a plate boundary and the date is unknowable.
""")


if __name__ == "__main__":
    main()
