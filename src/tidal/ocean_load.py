"""
Ocean tide heights from the GOT4.10c model, and the stress they load onto a
fault beneath the sea floor.

This closes the largest gap in tidal_analysis.py. The solid-earth body tide
gives ~0.5 kPa of Coulomb stress, and at the free surface it gives almost none
at all to a shallow-dipping fault, because sigma_rr vanishes there. That is why
the megathrust subset was tested weakly. The ocean is the missing term: a 1 m
tide presses on the sea floor with rho*g*h ~ 10 kPa, an order of magnitude more,
and it acts on exactly the vertical components the body tide cannot supply.

Model
-----
GOT4.10c (Ray, NASA/GSFC), 0.5 deg global grids of amplitude and Greenwich
phase lag for eight major constituents. Open access, no registration.

    h(t) = sum_j  f_j * A_j * cos( V_j(t) + u_j - G_j )

with f, u the 18.6-year nodal corrections and V the equilibrium argument.

Load stress
-----------
Ocean tide wavelengths are hundreds to thousands of kilometres; the faults are
10-40 km down. Load width therefore vastly exceeds depth, which is the
laterally-confined limit of an elastic half-space:

    sigma_DD = -P,      sigma_NN = sigma_EE = -P * nu/(1-nu),      P = rho*g*h

(tension positive, so a positive tide is compressive). Shear terms vanish for a
broad uniform load.

Scope, stated rather than buried: this is the DIRECT load of the water column
above the site. It applies where the event is beneath ocean. It does not
include the elastic response to the *distant* ocean load, which is what a
Green's-function convolution (Farrell 1972) would add and which is what matters
for land sites near a coast. Continental events therefore get no ocean term
here, and are reported separately.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import datafile, DATA, ROOT, SSGEOS  # noqa: F401
import re
from pathlib import Path

import numpy as np

RHO_W = 1025.0        # kg m^-3, sea water
G_ACC = 9.80665
NU = 0.25             # must match tidal_stress.NU

GOT_DIR = datafile("got410") / "GOT4.10c" / "grids_oceantide"

# Doodson combination on [tau, s, h, p] plus a constant, degrees.
# tau = 15*UT_hours + h - s + 180  (mean lunar time)
CONSTITUENTS = {
    "m2": ((2, 0, 0, 0), 0.0),
    "s2": ((2, 2, -2, 0), 0.0),
    "n2": ((2, -1, 0, 1), 0.0),
    "k2": ((2, 2, 0, 0), 0.0),
    "k1": ((1, 1, 0, 0), -90.0),
    "o1": ((1, -1, 0, 0), 90.0),
    "p1": ((1, 1, -2, 0), 90.0),
    "q1": ((1, -2, 0, 1), 90.0),
}


def read_got_grid(path: Path):
    """
    Read one GOT '.d' file: a header, an amplitude grid (cm), a second header,
    then a phase-lag grid (deg). Returns a complex grid A*exp(-i*G) with NaN
    over land, plus the latitude and longitude axes.
    """
    text = path.read_text(encoding="latin-1")
    lines = text.splitlines()

    fmt_idx = [i for i, l in enumerate(lines) if re.match(r"\s*\(\s*\d+F", l)]
    if len(fmt_idx) != 2:
        raise ValueError(f"{path.name}: expected 2 data blocks, found {len(fmt_idx)}")

    nlat, nlon = (int(x) for x in lines[fmt_idx[0] - 4].split())
    lat0, lat1 = (float(x) for x in lines[fmt_idx[0] - 3].split())
    lon0, lon1 = (float(x) for x in lines[fmt_idx[0] - 2].split())
    undef = float(lines[fmt_idx[0] - 1].split()[0])

    def block(start, count):
        vals = []
        for line in lines[start:]:
            vals.extend(line.split())
            if len(vals) >= count:
                break
        return np.array(vals[:count], dtype=float).reshape(nlat, nlon)

    amp = block(fmt_idx[0] + 1, nlat * nlon)
    pha = block(fmt_idx[1] + 1, nlat * nlon)

    bad = (amp >= undef - 1) | (pha >= undef - 1)
    amp = np.where(bad, np.nan, amp) / 100.0            # cm -> m
    pha = np.where(bad, np.nan, pha)

    z = amp * np.exp(-1j * np.radians(pha))
    lats = np.linspace(lat0, lat1, nlat)
    lons = np.linspace(lon0, lon1, nlon)
    return z, lats, lons


class OceanTide:
    """GOT4.10c constituent grids with bilinear interpolation to a site."""

    def __init__(self, directory: Path = GOT_DIR):
        self.grids = {}
        self.lats = self.lons = None
        for name in CONSTITUENTS:
            f = Path(directory) / f"{name}.d"
            if not f.exists():
                raise FileNotFoundError(f"missing GOT grid {f}")
            z, lats, lons = read_got_grid(f)
            self.grids[name] = z
            self.lats, self.lons = lats, lons
        self.dlat = self.lats[1] - self.lats[0]
        self.dlon = self.lons[1] - self.lons[0]

    def interp(self, lat, lon):
        """
        Complex constituent amplitudes at the given sites.
        Bilinear over the four surrounding nodes, using only the wet ones and
        renormalising their weights -- interpolating amplitude and phase
        separately would corrupt every phase wrap, so this works on the complex
        value throughout. Returns {name: complex array}, NaN where dry.
        """
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float)) % 360.0

        fi = (lat - self.lats[0]) / self.dlat
        fj = (lon - self.lons[0]) / self.dlon
        i0 = np.clip(np.floor(fi).astype(int), 0, len(self.lats) - 2)
        j0 = np.floor(fj).astype(int) % len(self.lons)
        j1 = (j0 + 1) % len(self.lons)
        wi = fi - i0
        wj = fj - j0

        corners = [(i0, j0, (1 - wi) * (1 - wj)), (i0, j1, (1 - wi) * wj),
                   (i0 + 1, j0, wi * (1 - wj)), (i0 + 1, j1, wi * wj)]

        out = {}
        for name, z in self.grids.items():
            acc = np.zeros(lat.shape, dtype=complex)
            wsum = np.zeros(lat.shape)
            for ii, jj, w in corners:
                v = z[ii, jj]
                good = np.isfinite(v.real) & np.isfinite(v.imag) & (w > 0)
                acc = acc + np.where(good, v * w, 0)
                wsum = wsum + np.where(good, w, 0)
            out[name] = np.where(wsum > 1e-6, acc / np.where(wsum > 0, wsum, 1), np.nan)
        return out

    def is_ocean(self, lat, lon):
        z = self.interp(lat, lon)["m2"]
        return np.isfinite(z.real)

    def height(self, coeffs, jd_ut):
        """
        Tide height in metres. `coeffs` comes from interp() for n sites;
        jd_ut has shape (n, nt) or (nt,). Returns the matching shape.
        """
        jd = np.asarray(jd_ut, dtype=float)
        args, fac = astronomical_arguments(jd)
        total = np.zeros(jd.shape)
        for name, z in coeffs.items():
            zz = np.asarray(z)
            if zz.ndim and jd.ndim > zz.ndim:
                zz = zz.reshape(zz.shape + (1,) * (jd.ndim - zz.ndim))
            f, u = fac[name]
            phase = np.radians(args[name] + u)
            total = total + f * (zz.real * np.cos(phase) - zz.imag * np.sin(phase))
        return total


def astronomical_arguments(jd_ut):
    """
    Equilibrium arguments V (deg) and nodal corrections (f, u) for each
    constituent, following Schureman. jd_ut may be any shape.
    """
    jd = np.asarray(jd_ut, dtype=float)
    t = (jd - 2451545.0) / 36525.0

    s = 218.3164477 + 481267.88123421 * t          # Moon mean longitude
    h = 280.46646 + 36000.76983 * t                # Sun mean longitude
    p = 83.3532465 + 4069.0137287 * t              # lunar perigee
    n = 125.0445479 - 1934.1362891 * t             # ascending node

    hours = (jd + 0.5 - np.floor(jd + 0.5)) * 24.0
    tau = 15.0 * hours + h - s + 180.0

    nr = np.radians(n)
    cN, s1N = np.cos(nr), np.sin(nr)
    c2N, s2N = np.cos(2 * nr), np.sin(2 * nr)
    c3N, s3N = np.cos(3 * nr), np.sin(3 * nr)

    f_m2 = 1.0004 - 0.0373 * cN + 0.0002 * c2N
    u_m2 = -2.14 * s1N
    f_o1 = 1.0089 + 0.1871 * cN - 0.0147 * c2N + 0.0014 * c3N
    u_o1 = 10.80 * s1N - 1.34 * s2N + 0.19 * s3N
    f_k1 = 1.0060 + 0.1150 * cN - 0.0088 * c2N + 0.0006 * c3N
    u_k1 = -8.86 * s1N + 0.68 * s2N - 0.07 * s3N
    f_k2 = 1.0241 + 0.2863 * cN + 0.0083 * c2N - 0.0015 * c3N
    u_k2 = -17.74 * s1N + 0.68 * s2N - 0.04 * s3N
    one, zero = np.ones_like(t), np.zeros_like(t)

    nodal = {"m2": (f_m2, u_m2), "n2": (f_m2, u_m2), "s2": (one, zero),
             "k2": (f_k2, u_k2), "k1": (f_k1, u_k1), "o1": (f_o1, u_o1),
             "q1": (f_o1, u_o1), "p1": (one, zero)}

    args = {}
    for name, ((a, b, c, d), const) in CONSTITUENTS.items():
        args[name] = (a * tau + b * s + c * h + d * p + const) % 360.0
    return args, nodal


def load_stress_components(height_m):
    """
    Stress from the direct water load, tension positive, in the NED frame.
    Returns (sigma_NN, sigma_EE, sigma_DD); shear components are zero.
    """
    p = RHO_W * G_ACC * np.asarray(height_m, dtype=float)
    s_dd = -p
    s_h = -p * NU / (1.0 - NU)
    return s_h, s_h, s_dd


if __name__ == "__main__":
    from datetime import datetime, timedelta
    from ephem_vec import jd_from_datetime_array

    ot = OceanTide()
    print(f"loaded {len(ot.grids)} constituents on a "
          f"{len(ot.lats)}x{len(ot.lons)} grid\n")

    # constituent periods, as a check on the astronomical arguments
    print("  constituent periods implied by the arguments (hours):")
    jd0 = 2451545.0
    a1, _ = astronomical_arguments(np.array([jd0]))
    a2, _ = astronomical_arguments(np.array([jd0 + 1e-4]))
    known = {"m2": 12.4206, "s2": 12.0000, "n2": 12.6583, "k2": 11.9672,
             "k1": 23.9345, "o1": 25.8193, "p1": 24.0659, "q1": 26.8684}
    for k in CONSTITUENTS:
        rate = ((a2[k][0] - a1[k][0] + 180) % 360 - 180) / (1e-4 * 24)   # deg/hour
        per = 360.0 / abs(rate)
        flag = "ok" if abs(per - known[k]) < 0.01 else "MISMATCH"
        print(f"    {k.upper():>3}  computed {per:8.4f}   known {known[k]:8.4f}   {flag}")

    print("\n  M2 amplitude at some well-known places:")
    sites = [("open central Pacific", 0.0, -140.0), ("Japan trench", 38.3, 143.5),
             ("Chile trench", -33.0, -73.5), ("Sumatra trench", 3.0, 94.0),
             ("Kamchatka trench", 52.0, 161.0), ("central Asia (land)", 40.0, 75.0)]
    for name, la, lo in sites:
        z = ot.interp(la, lo)["m2"][0]
        if np.isfinite(z.real):
            print(f"    {name:<22} {abs(z)*100:6.1f} cm   phase {np.degrees(-np.angle(z))%360:6.1f} deg")
        else:
            print(f"    {name:<22}   land (no ocean tide)")

    print("\n  tide height and load stress over 24 h, Japan trench (38.3N 143.5E):")
    t0 = datetime(2011, 3, 11)
    times = [t0 + timedelta(hours=k) for k in range(25)]
    jd = jd_from_datetime_array(times)
    co = ot.interp(38.3, 143.5)
    hgt = ot.height(co, jd[None, :])[0]
    print(f"    tide range        : {hgt.min():+.2f} to {hgt.max():+.2f} m")
    nn, ee, dd = load_stress_components(hgt)
    print(f"    vertical load     : {dd.min()/1000:+.2f} to {dd.max()/1000:+.2f} kPa")
    print(f"    horizontal load   : {nn.min()/1000:+.2f} to {nn.max()/1000:+.2f} kPa")
    print(f"\n    for comparison, body-tide dCFS was ~0.5 kPa peak-to-peak")
