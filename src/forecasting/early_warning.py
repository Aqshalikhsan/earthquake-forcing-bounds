"""
Step 9: the method that actually saves lives, and its hard limit.

Earthquake early warning does not predict anything. It detects an earthquake
that has already started and races a message ahead of the damaging waves. The
P wave travels at ~6.5 km/s and does little harm; the S wave and surface waves
travel at ~3.7 km/s and do the damage. That speed difference is the entire
budget, and it is spent before it arrives:

    warning = D/v_S  -  ( d_station/v_P + t_detect + t_process )

Everything below follows from that one line. It is why EEW works well for
distant subduction earthquakes and gives nothing at all for a shallow fault
directly beneath a city -- which is the case that kills the most people in
Indonesia.

Tsunami warning is the same idea on a longer clock: a tsunami in deep water
travels at sqrt(g*h), around 700 km/h, so a trench 200 km offshore buys about
17 minutes. A landslide-driven tsunami in a narrow bay buys three.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import math

V_P = 6.5          # km/s, crustal P
V_S = 3.7          # km/s, damaging S / surface waves
T_DETECT = 3.0     # s, waveform needed before a trigger is trustworthy
T_PROCESS = 2.0    # s, association, location, magnitude, message dispatch
G = 9.81


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def warning_seconds(dist_km, station_km):
    """Seconds of warning at `dist_km` from the source."""
    alert_at = station_km / V_P + T_DETECT + T_PROCESS
    s_arrival = dist_km / V_S
    return s_arrival - alert_at


def blind_radius(station_km):
    """Radius inside which warning arrives after the shaking."""
    return V_S * (station_km / V_P + T_DETECT + T_PROCESS)


def tsunami_minutes(dist_km, depth_m):
    return dist_km / (math.sqrt(G * depth_m) * 3.6) * 60


# (label, distance from source to city, typical nearest-station distance, note)
SCENARIOS = [
    ("Jakarta  <- Sunda megathrust, S of Java", 290, 60,
     "offshore source, dense island network"),
    ("Padang  <- Mentawai segment", 160, 50, "offshore, well instrumented"),
    ("Banda Aceh  <- Sunda trench", 170, 60, "offshore"),
    ("Denpasar  <- Java-Bali trench", 180, 60, "offshore"),
    ("Bengkulu  <- Sunda trench", 130, 50, "offshore"),
    ("Yogyakarta  <- Opak fault (2006)", 20, 25,
     "shallow crustal fault under the city"),
    ("Palu  <- Palu-Koro fault (2018)", 8, 25, "fault runs through the city"),
    ("Bandung  <- Lembang fault", 12, 25, "fault on the city's northern edge"),
    ("Mataram  <- Flores back-arc thrust (2018)", 45, 35, "near-field"),
]

TSUNAMI = [
    ("Padang from Mentawai trench", 200, 4000),
    ("Banda Aceh from Sunda trench", 150, 4000),
    ("Cilacap from Java trench", 220, 5000),
    ("Palu Bay, 2018 (local, landslide-driven)", 25, 300),
    ("Ende/Flores from back-arc (2026)", 60, 1500),
]


def main():
    rule("EARTHQUAKE EARLY WARNING: HOW MANY SECONDS, REALISTICALLY")
    print(f"""
  assumptions   v_P = {V_P} km/s, v_S = {V_S} km/s
                {T_DETECT:.0f} s of waveform to trigger, {T_PROCESS:.0f} s to locate and dispatch
  These are optimistic-but-defensible values for a modern network. Nothing here
  requires predicting anything: the earthquake has already happened.
""")
    print(f"  {'scenario':<42} {'dist':>6} {'warning':>9}   note")
    print("  " + "-" * 92)
    for label, dist, sta, note in SCENARIOS:
        w = warning_seconds(dist, sta)
        if w <= 0:
            txt = "NONE"
        else:
            txt = f"{w:5.0f} s"
        print(f"  {label:<42} {dist:>4.0f}km {txt:>9}   {note}")

    rule("THE BLIND ZONE")
    print()
    print(f"  {'nearest station':>17} {'blind radius':>14}")
    print("  " + "-" * 34)
    for sta in (15, 25, 40, 60, 80):
        print(f"  {sta:>14.0f} km {blind_radius(sta):>11.0f} km")
    print(f"""
  Inside the blind radius the warning arrives after the shaking does. With a
  good network the radius is around {blind_radius(25):.0f} km, and no amount of engineering
  removes it -- it is the S wave's head start turned into geometry.

  Now put the deadliest Indonesian earthquakes against that number:

    Yogyakarta 2006, M6.3   epicentre ~20 km from the city   ~5,700 dead
    Palu 2018, M7.5         fault runs through the city      ~4,300 dead
    Cianjur 2022, M5.6      shallow, directly beneath        ~600 dead

  All three are inside the blind zone. Early warning would have delivered
  nothing to the people who died. This is not a shortcoming of the technology;
  it is the reason building codes, not forecasts, are where the leverage is.
""")

    rule("WHERE EARLY WARNING DOES PAY")
    print("""
  It works when the source is far from the population: subduction megathrusts
  offshore. Those are the M8-9 events, and the warning is tens of seconds --
  enough to stop trains, close valves, halt surgery, open lift doors, and get
  children under desks. Japan, Mexico and the US west coast run exactly this.
""")
    good = [(l, d, s, n) for l, d, s, n in SCENARIOS if warning_seconds(d, s) > 10]
    for label, dist, sta, _ in good:
        w = warning_seconds(dist, sta)
        print(f"    {label:<42} {w:5.0f} s")

    rule("TSUNAMI WARNING: THE SAME LOGIC ON A LONGER CLOCK")
    print(f"""
  Deep-water tsunami speed is sqrt(g*h): about {math.sqrt(G*4000)*3.6:.0f} km/h over a 4 km ocean,
  slowing in shallow water. That leaves real time to act -- unless the source is
  in the bay with you.
""")
    print(f"  {'path':<44} {'depth':>7} {'travel time':>13}")
    print("  " + "-" * 68)
    for label, dist, depth in TSUNAMI:
        print(f"  {label:<44} {depth:>5.0f} m {tsunami_minutes(dist, depth):>10.0f} min")
    print("""
  The 2018 Palu tsunami arrived in roughly three minutes, generated partly by
  submarine landslides that no seismic magnitude estimate would have flagged in
  time. Aceh 2004 gave the near coast about 20 minutes -- and the warning system
  that would have used them did not exist yet. It does now.
""")

    rule("WHERE THE LEVERAGE ACTUALLY IS")
    print("""
  Ranked by lives saved per unit of effort, on the evidence:

    1  BUILDING CODES AND RETROFIT. Earthquakes do not kill people; collapsing
       buildings do. This dominates everything else and works regardless of
       whether anyone knows when the earthquake is coming.
    2  TSUNAMI WARNING AND EVACUATION ROUTES. Minutes of warning, and the
       physics is reliable. Signage, drills, and vertical evacuation structures
       convert those minutes into survival.
    3  EARLY WARNING. Seconds to tens of seconds, valuable for distant sources
       and automated systems, useless inside the blind zone.
    4  AFTERSHOCK FORECASTING (forecast.py). Hours to weeks, for managing the
       response after the event rather than before it.
    5  LONG-TERM HAZARD MAPS. Decades. Feeds item 1, which is the point.

  Notice what is absent from the list. Not because anyone gave up on predicting
  the date, but because a century of trying produced nothing that verifies --
  including the claim this repository set out to test.

  For Indonesia specifically the ordering is stark: the three deadliest recent
  earthquakes were shallow crustal ruptures beneath cities, in the blind zone,
  with no foreshocks. Every one of them was survivable in a building built to
  code. That is the whole answer.
""")


if __name__ == "__main__":
    main()
