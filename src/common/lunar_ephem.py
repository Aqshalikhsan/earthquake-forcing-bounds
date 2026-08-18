"""
Independent lunar ephemeris, so SSGEOS's phase timestamps can be checked
rather than assumed.

Implements Meeus, *Astronomical Algorithms* 2nd ed.:
  - ch. 25  : solar apparent longitude
  - ch. 47  : lunar longitude and distance (truncated ELP2000-82 series)
  - ch. 10  : Delta-T (Espenak & Meeus polynomial fits)

Accuracy of the truncated series is ~0.02 deg in elongation, i.e. ~0.002 days
of phase timing -- far finer than anything that matters here.

Phase angle convention: elongation = moon_longitude - sun_longitude, so
  0 deg = New Moon, 90 = First Quarter, 180 = Full Moon, 270 = Third Quarter.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import math
from datetime import datetime, timedelta

D2R = math.pi / 180.0

# ---------------------------------------------------------------- calendars

def jd_from_gregorian(y: int, m: int, d: float) -> float:
    """Julian Day from a proleptic *Gregorian* calendar date."""
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + d + b - 1524.5)


def jd_from_julian_cal(y: int, m: int, d: float) -> float:
    """Julian Day from a *Julian* calendar date (the calendar actually in use
    for every event in this catalogue before 1582)."""
    if m <= 2:
        y, m = y - 1, m + 12
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + d - 1524.5)


def jd_from_datetime(dt: datetime, calendar: str = "gregorian") -> float:
    day = dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24.0
    if calendar == "gregorian":
        return jd_from_gregorian(dt.year, dt.month, day)
    return jd_from_julian_cal(dt.year, dt.month, day)


def datetime_from_jd(jd: float) -> datetime:
    """Proleptic Gregorian datetime from a Julian Day."""
    jd += 0.5
    z = math.floor(jd)
    f = jd - z
    alpha = math.floor((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    di = int(day)
    frac = (day - di) * 24
    hh = int(frac)
    mm_f = (frac - hh) * 60
    mm = int(mm_f)
    ss = int(round((mm_f - mm) * 60))
    if ss == 60:
        ss = 0
        mm += 1
    if mm == 60:
        mm = 0
        hh += 1
    return datetime(year, month, di) + timedelta(hours=hh, minutes=mm, seconds=ss)


def delta_t_seconds(year_frac: float) -> float:
    """TD - UT in seconds (Espenak & Meeus 2006 polynomial fits)."""
    y = year_frac
    if y < -500:
        u = (y - 1820) / 100
        return -20 + 32 * u * u
    if y < 500:
        u = y / 100
        return (10583.6 - 1014.41 * u + 33.78311 * u**2 - 5.952053 * u**3
                - 0.1798452 * u**4 + 0.022174192 * u**5 + 0.0090316521 * u**6)
    if y < 1600:
        u = (y - 1000) / 100
        return (1574.2 - 556.01 * u + 71.23472 * u**2 + 0.319781 * u**3
                - 0.8503463 * u**4 - 0.005050998 * u**5 + 0.0083572073 * u**6)
    if y < 1700:
        t = y - 1600
        return 120 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129
    if y < 1800:
        t = y - 1700
        return 8.83 + 0.1603 * t - 0.0059285 * t**2 + 0.00013336 * t**3 - t**4 / 1174000
    if y < 1860:
        t = y - 1800
        return (13.72 - 0.332447 * t + 0.0068612 * t**2 + 0.0041116 * t**3
                - 0.00037436 * t**4 + 0.0000121272 * t**5
                - 0.0000001699 * t**6 + 0.000000000875 * t**7)
    if y < 1900:
        t = y - 1860
        return (7.62 + 0.5737 * t - 0.251754 * t**2 + 0.01680668 * t**3
                - 0.0004473624 * t**4 + t**5 / 233174)
    if y < 1920:
        t = y - 1900
        return -2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3 - 0.000197 * t**4
    if y < 1941:
        t = y - 1920
        return 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3
    if y < 1961:
        t = y - 1950
        return 29.07 + 0.407 * t - t**2 / 233 + t**3 / 2547
    if y < 1986:
        t = y - 1975
        return 45.45 + 1.067 * t - t**2 / 260 - t**3 / 718
    if y < 2005:
        t = y - 2000
        return (63.86 + 0.3345 * t - 0.060374 * t**2 + 0.0017275 * t**3
                + 0.000651814 * t**4 + 0.00002373599 * t**5)
    if y < 2050:
        t = y - 2000
        return 62.92 + 0.32217 * t + 0.005589 * t**2
    u = (y - 1820) / 100
    return -20 + 32 * u * u


# ---------------------------------------------------------------- solar

def sun_longitude(jde: float) -> float:
    """Apparent geocentric ecliptic longitude of the Sun, degrees."""
    t = (jde - 2451545.0) / 36525.0
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    m = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    mr = m * D2R
    c = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mr)
         + (0.019993 - 0.000101 * t) * math.sin(2 * mr)
         + 0.000289 * math.sin(3 * mr))
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    return (true_long - 0.00569 - 0.00478 * math.sin(omega * D2R)) % 360.0


# ---------------------------------------------------------------- lunar

# Meeus Table 47.A: (D, M, M', F, coeff_longitude[1e-6 deg], coeff_distance[1e-3 km])
_TERMS = [
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
]


def moon_longitude_distance(jde: float) -> tuple[float, float]:
    """Apparent geocentric ecliptic longitude (deg) and distance (km)."""
    t = (jde - 2451545.0) / 36525.0
    t2, t3, t4 = t * t, t ** 3, t ** 4

    lp = (218.3164477 + 481267.88123421 * t - 0.0015786 * t2
          + t3 / 538841 - t4 / 65194000) % 360
    d = (297.8501921 + 445267.1114034 * t - 0.0018819 * t2
         + t3 / 545868 - t4 / 113065000) % 360
    m = (357.5291092 + 35999.0502909 * t - 0.0001536 * t2 + t3 / 24490000) % 360
    mp = (134.9633964 + 477198.8675055 * t + 0.0087414 * t2
          + t3 / 69699 - t4 / 14712000) % 360
    f = (93.2720950 + 483202.0175233 * t - 0.0036539 * t2
         - t3 / 3526000 + t4 / 863310000) % 360

    e = 1 - 0.002516 * t - 0.0000074 * t2

    sum_l = 0.0
    sum_r = 0.0
    for cd, cm, cmp_, cf, cl, cr in _TERMS:
        arg = (cd * d + cm * m + cmp_ * mp + cf * f) * D2R
        ecc = e ** abs(cm)
        sum_l += cl * math.sin(arg) * ecc
        sum_r += cr * math.cos(arg) * ecc

    a1 = (119.75 + 131.849 * t) % 360
    a2 = (53.09 + 479264.290 * t) % 360
    sum_l += (3958 * math.sin(a1 * D2R)
              + 1962 * math.sin((lp - f) * D2R)
              + 318 * math.sin(a2 * D2R))

    lon = (lp + sum_l / 1_000_000.0) % 360.0
    dist = 385000.56 + sum_r / 1000.0
    return lon, dist


def elongation(jde: float) -> float:
    """Moon-Sun elongation in degrees, 0-360 (0 = new, 180 = full)."""
    lon, _ = moon_longitude_distance(jde)
    return (lon - sun_longitude(jde)) % 360.0


def phase_angle_at(dt: datetime, calendar: str = "gregorian",
                   apply_delta_t: bool = True) -> float:
    """Elongation at a UT civil datetime given in the named calendar."""
    jd = jd_from_datetime(dt, calendar)
    jde = jd + (delta_t_seconds(dt.year) / 86400.0 if apply_delta_t else 0.0)
    return elongation(jde)


def moon_distance_at(dt: datetime, calendar: str = "gregorian",
                     apply_delta_t: bool = True) -> float:
    jd = jd_from_datetime(dt, calendar)
    jde = jd + (delta_t_seconds(dt.year) / 86400.0 if apply_delta_t else 0.0)
    return moon_longitude_distance(jde)[1]


def find_phase_time(target_deg: float, jd_guess: float) -> float:
    """Julian Day (TD) of the nearest instant where elongation == target_deg."""
    def resid(jd):
        return (elongation(jd) - target_deg + 180.0) % 360.0 - 180.0

    jd = jd_guess
    for _ in range(60):
        r = resid(jd)
        h = 0.01
        deriv = (resid(jd + h) - resid(jd - h)) / (2 * h)
        if abs(deriv) < 1e-9:
            break
        step = r / deriv
        step = max(-2.0, min(2.0, step))
        jd -= step
        if abs(step) < 1e-8:
            break
    return jd


if __name__ == "__main__":
    # smoke test: the 2011 Tohoku earthquake, 2011-03-11 05:46:23 UT
    dt = datetime(2011, 3, 11, 5, 46, 23)
    ang = phase_angle_at(dt)
    dist = moon_distance_at(dt)
    print(f"Tohoku 2011: elongation = {ang:.3f} deg, moon distance = {dist:.1f} km")
    print("SSGEOS file says: 1.75 d before First Quarter, distance 396491.59 km")
    age_days = ang / 360.0 * 29.530588853
    print(f"-> implied lunar age {age_days:.2f} d; "
          f"days to First Quarter (90 deg) = {(90 - ang) / 360 * 29.530588853:.2f}")
