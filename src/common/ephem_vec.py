"""
Vectorised solar/lunar ephemeris for tidal-stress work.

lunar_ephem.py is written for clarity and audits one date at a time. Computing
tidal stress needs the Moon's and Sun's position at millions of instants, so
this module carries the same Meeus series evaluated over numpy arrays, and adds
what the tidal potential requires and phase work does not:

  * the Moon's ecliptic latitude (Meeus table 47.B) -- without it there is no
    declination, and the tide is dominated by declination effects
  * the Sun's distance, which varies 3.3% over the year
  * apparent sidereal time, to get the local hour angle

Returns geocentric equatorial coordinates: right ascension, declination,
distance.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import numpy as np

D2R = np.pi / 180.0

GM_MOON = 4.9028695e12      # m^3 s^-2
GM_SUN = 1.32712440018e20   # m^3 s^-2
AU = 1.495978707e11         # m

# Meeus table 47.A -- (D, M, M', F, sigma_l [1e-6 deg], sigma_r [1e-3 km])
_TERMS_LR = np.array([
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968), (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158), (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733), (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620), (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0), (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636), (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824), (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675), (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445), (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403), (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0), (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322), (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751), (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950), (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0), (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0), (3, 0, -1, 0, -892, 3258),
    (2, 1, 1, 0, -810, 2616), (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117), (2, 2, -1, 0, -700, 2354),
    (2, 1, -2, 0, 691, 0), (2, -1, 0, -2, 596, 0),
    (4, 0, 1, 0, 549, -1423), (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571), (1, 0, -2, 0, -487, -1739),
    (2, 1, 0, -2, -399, 0), (0, 0, 2, -2, -381, -4421),
    (1, 1, 1, 0, 351, 0), (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0), (2, -1, 2, 0, 327, 0),
    (0, 2, 1, 0, -323, 1165), (1, 1, -1, 0, 299, 0),
    (2, 0, 3, 0, 294, 0), (2, 0, -1, -2, 0, 8752),
], dtype=float)

# Meeus table 47.B -- (D, M, M', F, sigma_b [1e-6 deg])
_TERMS_B = np.array([
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237), (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198), (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822), (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211), (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828), (0, 1, 0, 1, -1794), (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565), (1, 0, 0, 1, -1491), (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410), (0, 1, 0, -1, -1344), (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107), (4, 0, 0, -1, 1021), (4, 0, -1, 1, 833),
    (0, 0, 1, -3, 777), (4, 0, -2, 1, 671), (2, 0, 0, -3, 607),
    (2, 0, 2, -1, 596), (2, -1, 1, -1, 491), (2, 0, -2, 1, -451),
    (0, 0, 3, -1, 439), (2, 0, 2, 1, 422), (2, 0, -3, -1, 421),
    (2, 1, -1, 1, -366), (2, 1, 0, 1, -351), (4, 0, 0, 1, 331),
    (2, -1, 1, 1, 315), (2, -2, 0, -1, 302), (0, 0, 1, 3, -283),
    (2, 1, 1, -1, -229), (1, 1, 0, -1, 223), (1, 1, 0, 1, 223),
    (0, 1, -2, -1, -220), (2, 1, -1, -1, -220), (1, 0, 1, 1, -185),
    (2, -1, -2, -1, 181), (0, 1, 2, 1, -177), (4, 0, -2, -1, 176),
    (4, -1, -1, -1, 166), (1, 0, 1, -1, -164), (4, 0, 1, -1, 132),
    (1, 0, -1, -1, -119), (4, -1, 0, -1, 115), (2, -2, 0, 1, 107),
], dtype=float)


def _fundamental(t):
    lp = (218.3164477 + 481267.88123421 * t - 0.0015786 * t**2
          + t**3 / 538841 - t**4 / 65194000) % 360
    d = (297.8501921 + 445267.1114034 * t - 0.0018819 * t**2
         + t**3 / 545868 - t**4 / 113065000) % 360
    m = (357.5291092 + 35999.0502909 * t - 0.0001536 * t**2 + t**3 / 24490000) % 360
    mp = (134.9633964 + 477198.8675055 * t + 0.0087414 * t**2
          + t**3 / 69699 - t**4 / 14712000) % 360
    f = (93.2720950 + 483202.0175233 * t - 0.0036539 * t**2
         - t**3 / 3526000 + t**4 / 863310000) % 360
    return lp, d, m, mp, f


def moon_ecliptic(jde):
    """Geocentric apparent ecliptic longitude (deg), latitude (deg), distance (m)."""
    jde = np.asarray(jde, dtype=float)
    t = (jde - 2451545.0) / 36525.0
    lp, d, m, mp, f = _fundamental(t)
    e = 1 - 0.002516 * t - 0.0000074 * t**2

    def accumulate(table, cols):
        out = [np.zeros_like(jde) for _ in cols]
        for row in table:
            cd, cm, cmp_, cf = row[0], row[1], row[2], row[3]
            arg = (cd * d + cm * m + cmp_ * mp + cf * f) * D2R
            ecc = e ** abs(cm)
            for k, ci in enumerate(cols):
                c = row[ci]
                if c == 0:
                    continue
                out[k] += c * (np.sin(arg) if ci != 5 else np.cos(arg)) * ecc
        return out

    sum_l, sum_r = accumulate(_TERMS_LR, (4, 5))
    (sum_b,) = accumulate(_TERMS_B, (4,))

    a1 = (119.75 + 131.849 * t) % 360
    a2 = (53.09 + 479264.290 * t) % 360
    a3 = (313.45 + 481266.484 * t) % 360

    sum_l += (3958 * np.sin(a1 * D2R) + 1962 * np.sin((lp - f) * D2R)
              + 318 * np.sin(a2 * D2R))
    sum_b += (-2235 * np.sin(lp * D2R) + 382 * np.sin(a3 * D2R)
              + 175 * np.sin((a1 - f) * D2R) + 175 * np.sin((a1 + f) * D2R)
              + 127 * np.sin((lp - mp) * D2R) - 115 * np.sin((lp + mp) * D2R))

    lon = (lp + sum_l / 1e6) % 360.0
    lat = sum_b / 1e6
    dist = (385000.56 + sum_r / 1000.0) * 1000.0     # m
    return lon, lat, dist


def sun_ecliptic(jde):
    """Geocentric apparent ecliptic longitude (deg) and distance (m)."""
    jde = np.asarray(jde, dtype=float)
    t = (jde - 2451545.0) / 36525.0
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t**2
    m = (357.52911 + 35999.05029 * t - 0.0001537 * t**2)
    mr = m * D2R
    c = ((1.914602 - 0.004817 * t - 0.000014 * t**2) * np.sin(mr)
         + (0.019993 - 0.000101 * t) * np.sin(2 * mr)
         + 0.000289 * np.sin(3 * mr))
    true_long = l0 + c
    v = m + c
    ecc = 0.016708634 - 0.000042037 * t - 0.0000001267 * t**2
    r = 1.000001018 * (1 - ecc**2) / (1 + ecc * np.cos(v * D2R)) * AU
    omega = 125.04 - 1934.136 * t
    lon = (true_long - 0.00569 - 0.00478 * np.sin(omega * D2R)) % 360.0
    return lon, r


def obliquity(jde):
    """Mean obliquity of the ecliptic, degrees (Meeus 22.2)."""
    t = (np.asarray(jde, dtype=float) - 2451545.0) / 36525.0
    return (23.0 + 26.0 / 60 + 21.448 / 3600
            - (46.8150 * t + 0.00059 * t**2 - 0.001813 * t**3) / 3600)


def ecliptic_to_equatorial(lon, lat, eps):
    """Ecliptic (deg) -> right ascension and declination (deg)."""
    lo, la, e = lon * D2R, lat * D2R, eps * D2R
    ra = np.arctan2(np.sin(lo) * np.cos(e) - np.tan(la) * np.sin(e), np.cos(lo))
    dec = np.arcsin(np.sin(la) * np.cos(e) + np.cos(la) * np.sin(e) * np.sin(lo))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def gmst_deg(jd_ut):
    """Greenwich mean sidereal time in degrees (Meeus 12.4)."""
    jd_ut = np.asarray(jd_ut, dtype=float)
    t = (jd_ut - 2451545.0) / 36525.0
    return (280.46061837 + 360.98564736629 * (jd_ut - 2451545.0)
            + 0.000387933 * t**2 - t**3 / 38710000.0) % 360.0


def bodies_equatorial(jd_ut, delta_t_sec=69.0):
    """
    Moon and Sun in geocentric equatorial coordinates plus GMST, for an array
    of UT Julian Days. Delta-T is a constant here: over 1976-2020 it runs
    47-70 s, and a 20 s error moves the Moon by 0.01 deg -- negligible against
    a tidal signal with a 12-hour period.
    """
    jd_ut = np.asarray(jd_ut, dtype=float)
    jde = jd_ut + delta_t_sec / 86400.0
    eps = obliquity(jde)

    ml, mb, md = moon_ecliptic(jde)
    sl, sd = sun_ecliptic(jde)

    m_ra, m_dec = ecliptic_to_equatorial(ml, mb, eps)
    s_ra, s_dec = ecliptic_to_equatorial(sl, np.zeros_like(sl), eps)

    return dict(
        moon=(m_ra, m_dec, md),
        sun=(s_ra, s_dec, sd),
        gmst=gmst_deg(jd_ut),
    )


def jd_from_datetime_array(dts):
    """UT Julian Day for a sequence of datetimes (proleptic Gregorian)."""
    out = np.empty(len(dts), dtype=float)
    for i, dt in enumerate(dts):
        y, m = dt.year, dt.month
        day = (dt.day + (dt.hour + dt.minute / 60
                         + (dt.second + dt.microsecond / 1e6) / 3600) / 24.0)
        if m <= 2:
            y, m = y - 1, m + 12
        a = y // 100
        b = 2 - a + a // 4
        out[i] = (np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1))
                  + day + b - 1524.5)
    return out
