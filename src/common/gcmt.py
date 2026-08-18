"""
Parser for the Global CMT catalogue in NDK format.

GCMT is the right catalogue for a tidal-triggering test: homogeneous moment
magnitudes, globally complete for Mw >= 5.5 since 1976, and -- crucially -- it
carries a focal mechanism for every event. Tidal stress cannot be resolved onto
a fault without one, which is why lunar phase gets used as a proxy instead.

NDK is a fixed five-line record:
  1  hypocentre: catalogue, date, time, lat, lon, depth, mb, MS, region
  2  event name and inversion settings
  3  CENTROID: time shift, centroid lat/lon/depth
  4  moment tensor exponent and six components
  5  eigen-decomposition, scalar moment, and the two nodal planes
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class CMTEvent:
    dt: datetime          # centroid time (hypocentre time + centroid time shift)
    lat: float            # centroid latitude, degrees north
    lon: float            # centroid longitude, degrees east
    depth: float          # centroid depth, km
    mw: float
    strike1: float
    dip1: float
    rake1: float
    strike2: float
    dip2: float
    rake2: float
    region: str

    @property
    def mechanism(self) -> str:
        """Classify by rake of the shallower nodal plane (Frohlich-style, simplified)."""
        rake = self.rake1 if self.dip1 <= self.dip2 else self.rake2
        r = ((rake + 180) % 360) - 180
        if 45 <= r <= 135:
            return "thrust"
        if -135 <= r <= -45:
            return "normal"
        return "strike-slip"

    def fault_plane(self, which: str = "shallow"):
        """
        Return (strike, dip, rake). The nodal-plane ambiguity is real: only one
        of the two is the fault. For subduction thrusts the fault is the
        shallow-dipping plane, so 'shallow' is a physical rule applied
        uniformly rather than a choice made per event. 'np1'/'np2' are
        available as robustness checks.
        """
        if which == "np1":
            return self.strike1, self.dip1, self.rake1
        if which == "np2":
            return self.strike2, self.dip2, self.rake2
        if self.dip1 <= self.dip2:
            return self.strike1, self.dip1, self.rake1
        return self.strike2, self.dip2, self.rake2


def _f(s: str, default=float("nan")) -> float:
    try:
        return float(s)
    except ValueError:
        return default


def parse_ndk(path: str | Path) -> list[CMTEvent]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[CMTEvent] = []

    for i in range(0, len(lines) - 4, 5):
        l1, _l2, l3, l4, l5 = lines[i:i + 5]
        if not l3.startswith("CENTROID"):
            continue
        try:
            date_s = l1[5:15]
            time_s = l1[16:26]
            y, mo, d = (int(x) for x in date_s.split("/"))
            hh, mm = int(time_s[:2]), int(time_s[3:5])
            sec = _f(time_s[6:])
            if not math.isfinite(sec):
                continue
            base = datetime(y, mo, d, hh, mm) + timedelta(seconds=sec)

            parts3 = l3.replace("CENTROID:", "").split()
            tshift = _f(parts3[0])
            lat = _f(parts3[2])
            lon = _f(parts3[4])
            depth = _f(parts3[6])

            expo = int(l4[:2])
            parts5 = l5[3:].split()
            # eigen triples (3 x 3 numbers) then scalar moment then 6 plane angles
            m0 = _f(parts5[9]) * (10 ** expo)          # dyne-cm
            s1, d1, r1 = (_f(parts5[10]), _f(parts5[11]), _f(parts5[12]))
            s2, d2, r2 = (_f(parts5[13]), _f(parts5[14]), _f(parts5[15]))

            if not (math.isfinite(m0) and m0 > 0):
                continue
            mw = (2.0 / 3.0) * (math.log10(m0) - 16.1)

            vals = [tshift, lat, lon, depth, s1, d1, r1, s2, d2, r2]
            if not all(math.isfinite(v) for v in vals):
                continue

            events.append(CMTEvent(
                dt=base + timedelta(seconds=tshift),
                lat=lat, lon=lon, depth=depth, mw=mw,
                strike1=s1, dip1=d1, rake1=r1,
                strike2=s2, dip2=d2, rake2=r2,
                region=l1[56:].strip(),
            ))
        except (ValueError, IndexError):
            continue

    events.sort(key=lambda e: e.dt)
    return events


if __name__ == "__main__":
    import collections

    evs = parse_ndk(datafile("gcmt.ndk"))
    print(f"parsed events        : {len(evs)}")
    print(f"time span            : {evs[0].dt.date()} to {evs[-1].dt.date()}")
    print(f"Mw range             : {min(e.mw for e in evs):.2f} - {max(e.mw for e in evs):.2f}")
    shallow = [e for e in evs if e.depth <= 70]
    print(f"depth <= 70 km       : {len(shallow)}")
    for thr in (5.5, 6.0, 6.5, 7.0):
        sub = [e for e in shallow if e.mw >= thr]
        c = collections.Counter(e.mechanism for e in sub)
        print(f"  shallow Mw>={thr}   : {len(sub):>6}   "
              + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
