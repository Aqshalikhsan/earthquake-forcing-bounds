"""
Step 13: the planetary tide computed properly, with Earth's rotation included.

A fair objection to the earlier work: lunar PHASE is a single global number, but
tidal STRESS is not. Earth rotates, so the face turned toward a planet changes
through the day, and the tidal stress a given site feels depends on that planet's
local hour angle. Location-specific planetary forcing therefore does exist.

That objection is correct, and it was never tested here -- the tidal chapters
computed the Moon and Sun with rotation included but never the planets.

So this does it. Same machinery: degree-2 tide-generating potential, Love-number
surface strain, traction resolved on a real fault, evaluated minute by minute so
the rotational (diurnal and semidiurnal) components are fully present.

The question is not whether a location-specific planetary tide exists. It does.
The question is how big it is next to the things already known to be too small.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np
from datetime import datetime, timedelta

from ephem_vec import jd_from_datetime_array, obliquity, bodies_equatorial, D2R
from tidal_stress import (_surface_stress, fault_vectors, MU_FRICTION,
                          dcfs_series)
from planetary import heliocentric

# Standard gravitational parameters, m^3 s^-2
GM = {
    "Mercury": 2.2032e13, "Venus": 3.24859e14, "Mars": 4.282837e13,
    "Jupiter": 1.26686534e17, "Saturn": 3.7931187e16,
    "Uranus": 5.793939e15, "Neptune": 6.836529e15,
}
GM_MOON = 4.9028695e12
GM_SUN = 1.32712440018e20
AU = 1.495978707e11


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def planet_equatorial(name, jd):
    """Geocentric right ascension, declination (deg) and distance (m)."""
    px, py, pz = heliocentric(name, jd)
    ex, ey, ez = heliocentric("Earth", jd)
    x, y, z = (px - ex) * AU, (py - ey) * AU, (pz - ez) * AU
    eps = obliquity(jd) * D2R
    # ecliptic -> equatorial
    xe = x
    ye = y * np.cos(eps) - z * np.sin(eps)
    ze = y * np.sin(eps) + z * np.cos(eps)
    dist = np.sqrt(xe**2 + ye**2 + ze**2)
    ra = np.degrees(np.arctan2(ye, xe)) % 360.0
    dec = np.degrees(np.arcsin(ze / dist))
    return ra, dec, dist


def dcfs_from_body(lat, lon, strike, dip, rake, jd, ra, dec, dist, gm):
    """Coulomb stress change on a fault from one perturbing body, in Pa."""
    colat = np.array([[(90.0 - lat) * D2R]])
    lon_e = np.array([[lon * D2R]])
    gmst = bodies_equatorial(jd)["gmst"]
    s_tt, s_pp, s_tp = _surface_stress(colat, lon_e, ra, dec, dist, gmst, gm)
    s_nn, s_ee, s_ne = s_tt, s_pp, -s_tp

    n, u = fault_vectors(np.array([strike]), np.array([dip]), np.array([rake]))
    n_n, n_e = n[..., 0][..., None], n[..., 1][..., None]
    u_n, u_e = u[..., 0][..., None], u[..., 1][..., None]
    t_n = s_nn * n_n + s_ne * n_e
    t_e = s_ne * n_n + s_ee * n_e
    sigma_n = t_n * n_n + t_e * n_e
    tau = t_n * u_n + t_e * u_e
    return (tau + MU_FRICTION * sigma_n).ravel()


def main():
    rule("DOES EARTH'S ROTATION MAKE THE PLANETARY TIDE LOCAL?  Yes. How big is it?")

    # a real site and a real fault: the Java trench megathrust
    lat, lon = -9.5, 108.0
    strike, dip, rake = 288.0, 12.0, 95.0

    # one month, one-minute sampling: fully resolves the diurnal and
    # semidiurnal components that rotation creates
    t0 = datetime(2026, 8, 1)
    times = [t0 + timedelta(minutes=k) for k in range(0, 43200, 2)]
    jd = jd_from_datetime_array(times)

    print(f"""
  site        Java trench, {abs(lat):.1f}S {lon:.1f}E
  fault       megathrust, strike {strike:.0f} dip {dip:.0f} rake {rake:.0f}
  sampling    2-minute steps over 30 days, so every rotational component is
              present -- if rotation created a large effect, it would show here
""")

    eph = bodies_equatorial(jd)
    results = []
    for label, (ra, dec, dist), gm in (
        ("Moon", eph["moon"], GM_MOON),
        ("Sun", eph["sun"], GM_SUN),
    ):
        d = dcfs_from_body(lat, lon, strike, dip, rake, jd, ra, dec, dist, gm)
        results.append((label, d.max() - d.min()))

    for name in ("Venus", "Jupiter", "Mars", "Mercury", "Saturn", "Uranus", "Neptune"):
        ra, dec, dist = planet_equatorial(name, jd)
        d = dcfs_from_body(lat, lon, strike, dip, rake, jd, ra, dec, dist, GM[name])
        results.append((name, d.max() - d.min()))

    moon_amp = results[0][1]
    print(f"  {'body':<10} {'peak-to-peak dCFS':>22} {'vs Moon':>14}")
    print("  " + "-" * 50)
    for label, amp in results:
        if amp >= 1.0:
            shown = f"{amp:,.1f} Pa"
        elif amp >= 1e-3:
            shown = f"{amp*1e3:,.2f} mPa"
        elif amp >= 1e-6:
            shown = f"{amp*1e6:,.2f} uPa"
        else:
            shown = f"{amp*1e9:,.2f} nPa"
        print(f"  {label:<10} {shown:>22} {amp/moon_amp:>13.2e}")

    rule("WHAT THOSE NUMBERS ARE UP AGAINST")
    venus = dict(results)["Venus"]
    jup = dict(results)["Jupiter"]
    print(f"""
  Everyday stresses on the same patch of crust, for scale:

    ocean tide loading (1 m of water)          ~10,000     Pa
    solid-earth body tide (Moon + Sun)         ~{moon_amp:,.0f}       Pa
    a passing weather front (10 hPa)            ~1,000     Pa
    monsoon groundwater, seasonal               ~10,000    Pa
    standing in one place (your own weight)       ~5,000   Pa (directly beneath)

    Jupiter, rotation included                  {jup*1e6:>9,.2f} uPa
    Venus, rotation included                    {venus*1e6:>9,.2f} uPa

  Venus moves this fault by about {venus*1e6:,.2f} micropascals. Atmospheric pressure
  changes by roughly {1000/venus:,.0f} times that amount every time the weather shifts,
  and nobody forecasts earthquakes from the barometer.
""")

    rule("BUT DOES ROTATION AMPLIFY ANYTHING? (the resonance question)")
    print("""
  A reasonable follow-up: rotation sets the forcing frequency, so could the
  Earth resonate and amplify a tiny planetary push?

  For the solid Earth, no. The body-tide response is essentially static -- the
  Love numbers (h2 = 0.61, l2 = 0.085) are frequency-independent to well under a
  percent across the tidal band. Earth's free oscillation periods are under an
  hour; the tidal band is 12-24 hours. Nowhere near resonance.

  The oceans DO have near-resonant basins, which is why real ocean tides run
  several times the equilibrium value. But that amplification applies to
  whatever drives the ocean, and it is a factor of a few -- nowhere near the
  five orders of magnitude needed here. Amplifying Venus tenfold leaves it
  about 20,000 times too weak instead of 200,000.
""")

    rule("THE HONEST SUMMARY")
    print(f"""
  You were right about the physics: Earth's rotation does make planetary tidal
  stress location-specific and time-varying. It is a real, computable field, and
  it was not tested in the earlier chapters. Now it is.

  What that changes: the planetary claim is no longer "has no spatial
  information". It does have spatial information.

  What it does not change: the amplitude. Venus reaches {venus*1e6:,.2f} uPa on this
  fault -- {moon_amp/venus:,.0f} times weaker than the Moon, whose own effect is already
  measured to be undetectable across 9,000 earthquakes.

  A signal can be perfectly well-defined, perfectly local, perfectly computable,
  and still far too small to do anything. That is the situation here.
""")


if __name__ == "__main__":
    main()
