"""
Lunisolar tidal Coulomb failure stress at an earthquake's location and fault.

This is what the tidal-triggering literature actually tests, and what a lunar
phase angle cannot represent. The distinction matters more than it looks:

  * Lunar phase is a GLOBAL quantity. It is identical for an earthquake in
    Chile and one in Japan at the same instant, so it carries no information
    about where the event was.
  * Tidal stress is LOCAL. It depends on latitude, longitude, the Moon's and
    Sun's declination and hour angle, and the orientation of the fault.
  * Lunar phase only tracks the fortnightly spring-neap envelope, which is a
    ~20% modulation. The dominant semidiurnal (M2) and diurnal (O1, K1)
    constituents -- the bulk of the amplitude -- are invisible to it.

Method (standard, e.g. Melchior 1983; Tanaka et al. 2002; Cochran et al. 2004):

  1. degree-2 tide-generating potential of Moon and Sun at the site
  2. surface strain from the potential via Love numbers h2, l2
  3. stress from Hooke's law under the free-surface condition
  4. traction resolved on the fault plane -> dCFS = dtau + mu' * dsigma_n
  5. tidal phase of dCFS at the origin time, 0 deg = stress maximum

Approximations, stated rather than buried:
  * solid-earth body tide only; OCEAN TIDAL LOADING IS NOT MODELLED, and near
    coasts it can rival or exceed the body tide. This is the main limitation.
  * free-surface elastic half-space, so results are meaningful for shallow
    events and are not used below 70 km.
  * homogeneous elastic constants.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np

from ephem_vec import bodies_equatorial, GM_MOON, GM_SUN, D2R

# ------------------------------------------------------------------ constants
A_EARTH = 6.371e6      # m
G_SURF = 9.80665       # m s^-2
H2 = 0.6078            # Love number, vertical displacement
L2 = 0.0847            # Shida number, horizontal displacement
MU = 3.0e10            # shear modulus, Pa
NU = 0.25              # Poisson's ratio
MU_FRICTION = 0.4      # apparent friction coefficient for dCFS

_H = 1e-4              # finite-difference step, radians


def _potential(colat, lon_east, ra, dec, dist, gmst, gm):
    """
    Degree-2 tide-generating potential per unit mass, m^2 s^-2.

        W2 = GM * a^2 / d^3 * P2(cos psi)

    colat, lon_east in radians; ra, dec, gmst in degrees; dist in metres.
    Arrays broadcast together.
    """
    hour_angle = (gmst * D2R) + lon_east - (ra * D2R)
    cospsi = (np.cos(colat) * np.sin(dec * D2R)
              + np.sin(colat) * np.cos(dec * D2R) * np.cos(hour_angle))
    return gm * A_EARTH**2 / dist**3 * (3.0 * cospsi**2 - 1.0) / 2.0


def _surface_stress(colat, lon_east, ra, dec, dist, gmst, gm):
    """
    Horizontal stress components (sigma_tt, sigma_pp, sigma_tp) in Pa from one
    body, at the free surface. Derivatives of the potential are taken
    numerically -- the potential is analytic and smooth in position, so central
    differences are exact to far better than the physics warrants, and avoid a
    page of hand-derived spherical-harmonic algebra that is easy to get wrong.
    """
    def W(dt_, dp_):
        return _potential(colat + dt_, lon_east + dp_, ra, dec, dist, gmst, gm)

    h = _H
    w0 = W(0.0, 0.0)
    wtp_, wtm_ = W(h, 0.0), W(-h, 0.0)
    wpp_, wpm_ = W(0.0, h), W(0.0, -h)

    w_t = (wtp_ - wtm_) / (2 * h)
    w_tt = (wtp_ - 2 * w0 + wtm_) / (h * h)
    w_p = (wpp_ - wpm_) / (2 * h)
    w_pp = (wpp_ - 2 * w0 + wpm_) / (h * h)
    w_tp = (W(h, h) - W(h, -h) - W(-h, h) + W(-h, -h)) / (4 * h * h)

    sin_t = np.sin(colat)
    cot_t = np.cos(colat) / sin_t
    k = 1.0 / (A_EARTH * G_SURF)

    eps_tt = k * (L2 * w_tt + H2 * w0)
    eps_pp = k * (L2 * (w_pp / sin_t**2 + cot_t * w_t) + H2 * w0)
    eps_tp = k * L2 * (w_tp / sin_t - w_p * np.cos(colat) / sin_t**2)

    c = 2.0 * MU / (1.0 - NU)
    s_tt = c * (eps_tt + NU * eps_pp)
    s_pp = c * (eps_pp + NU * eps_tt)
    s_tp = 2.0 * MU * eps_tp
    return s_tt, s_pp, s_tp


def fault_vectors(strike_deg, dip_deg, rake_deg):
    """
    Fault normal and slip unit vectors in a North-East-Down frame
    (Aki & Richards convention).
    """
    phi = np.asarray(strike_deg, dtype=float) * D2R
    dlt = np.asarray(dip_deg, dtype=float) * D2R
    lam = np.asarray(rake_deg, dtype=float) * D2R

    n = np.stack([-np.sin(dlt) * np.sin(phi),
                  np.sin(dlt) * np.cos(phi),
                  -np.cos(dlt)], axis=-1)
    u = np.stack([np.cos(lam) * np.cos(phi) + np.cos(dlt) * np.sin(lam) * np.sin(phi),
                  np.cos(lam) * np.sin(phi) - np.cos(dlt) * np.sin(lam) * np.cos(phi),
                  -np.sin(lam) * np.sin(dlt)], axis=-1)
    return n, u


def dcfs_series(lat_deg, lon_deg, strike, dip, rake, jd_ut, ocean_height=None):
    """
    Tidal Coulomb failure stress change, Pa, at one site and fault for an array
    of Julian Days. Positive dCFS promotes failure.

    lat/lon/strike/dip/rake broadcast against jd_ut's trailing axis, so a whole
    chunk of events can be evaluated at once: pass shapes (n, 1) and (n, nt).

    ocean_height : optional array of ocean tide height in metres, same shape as
    jd_ut. When supplied, the direct water load is added to the body tide. This
    matters enormously for shallow-dipping faults: the body tide contributes
    nothing to the vertical stress components at a free surface, so a megathrust
    is nearly blind to it, while the ocean load acts on exactly those
    components at roughly ten times the amplitude.
    """
    colat = (90.0 - np.asarray(lat_deg, dtype=float)) * D2R
    lon_e = np.asarray(lon_deg, dtype=float) * D2R

    eph = bodies_equatorial(jd_ut)
    gmst = eph["gmst"]

    s_tt = s_pp = s_tp = 0.0
    for body, gm in (("moon", GM_MOON), ("sun", GM_SUN)):
        ra, dec, dist = eph[body]
        a, b, c = _surface_stress(colat, lon_e, ra, dec, dist, gmst, gm)
        s_tt = s_tt + a
        s_pp = s_pp + b
        s_tp = s_tp + c

    # (theta, phi, r) = (South, East, Up)  ->  (North, East, Down)
    s_nn, s_ee, s_ne = s_tt, s_pp, -s_tp
    s_dd = np.zeros_like(s_nn)

    if ocean_height is not None:
        from ocean_load import load_stress_components
        o_nn, o_ee, o_dd = load_stress_components(ocean_height)
        o_nn = np.nan_to_num(o_nn)
        o_ee = np.nan_to_num(o_ee)
        o_dd = np.nan_to_num(o_dd)
        s_nn = s_nn + o_nn
        s_ee = s_ee + o_ee
        s_dd = s_dd + o_dd

    n, u = fault_vectors(strike, dip, rake)
    n_n, n_e, n_d = (n[..., k][..., None] for k in range(3))
    u_n, u_e, u_d = (u[..., k][..., None] for k in range(3))

    # traction = sigma . n; sigma_nd and sigma_ed vanish for both the body tide
    # at a free surface and a broad uniform surface load
    t_n = s_nn * n_n + s_ne * n_e
    t_e = s_ne * n_n + s_ee * n_e
    t_d = s_dd * n_d

    sigma_n = t_n * n_n + t_e * n_e + t_d * n_d    # tension positive -> unclamping
    tau = t_n * u_n + t_e * u_e + t_d * u_d        # shear along the slip direction
    return tau + MU_FRICTION * sigma_n


def tidal_phase(dcfs, i_centre):
    """
    Phase of the event within the tidal cycle, degrees, with 0 = a maximum of
    dCFS. Locates the maxima bracketing the origin time and interpolates
    linearly between them, the standard construction in this literature.

    dcfs : (n, nt) array;  i_centre : index of the origin time along axis 1.
    Returns (phase_deg, amplitude_pa); phase is NaN where no bracket exists.
    """
    n, nt = dcfs.shape
    phase = np.full(n, np.nan)
    amp = np.full(n, np.nan)

    is_max = np.zeros_like(dcfs, dtype=bool)
    is_max[:, 1:-1] = (dcfs[:, 1:-1] > dcfs[:, :-2]) & (dcfs[:, 1:-1] >= dcfs[:, 2:])

    for i in range(n):
        idx = np.flatnonzero(is_max[i])
        if idx.size < 2:
            continue
        before = idx[idx <= i_centre]
        after = idx[idx > i_centre]
        if before.size == 0 or after.size == 0:
            continue
        t0, t1 = before[-1], after[0]
        if t1 <= t0:
            continue
        phase[i] = 360.0 * (i_centre - t0) / (t1 - t0)
        seg = dcfs[i, t0:t1 + 1]
        amp[i] = (seg.max() - seg.min()) / 2.0
    return phase, amp


def schuster(angles_deg):
    """Schuster's test on circular data: p = exp(-R^2/n)."""
    th = np.radians(np.asarray(angles_deg, dtype=float))
    th = th[np.isfinite(th)]
    n = th.size
    if n == 0:
        return 1.0, 0.0, np.nan, 0
    r = np.hypot(np.cos(th).sum(), np.sin(th).sum())
    mean_dir = np.degrees(np.arctan2(np.sin(th).sum(), np.cos(th).sum())) % 360.0
    return float(np.exp(-(r ** 2) / n)), float(r / n), float(mean_dir), n


if __name__ == "__main__":
    from datetime import datetime, timedelta
    from ephem_vec import jd_from_datetime_array

    # Tohoku 2011: shallow megathrust, strike 203 dip 10 rake 88
    t0 = datetime(2011, 3, 11, 5, 46, 23)
    times = [t0 + timedelta(minutes=4 * (k - 360)) for k in range(721)]
    jd = jd_from_datetime_array(times)[None, :]

    d = dcfs_series(np.array([[38.30]]), np.array([[142.50]]),
                    np.array([203.0]), np.array([10.0]), np.array([88.0]), jd)
    ph, am = tidal_phase(d, 360)
    print(f"Tohoku 2011 megathrust")
    print(f"  dCFS peak-to-peak amplitude : {2*am[0]:.0f} Pa  ({2*am[0]/1000:.2f} kPa)")
    print(f"  dCFS at origin time         : {d[0, 360]:.0f} Pa")
    print(f"  tidal phase at origin       : {ph[0]:.1f} deg  (0 = stress maximum)")
    print(f"\n  expected order for a body tide: 1-5 kPa peak-to-peak")
