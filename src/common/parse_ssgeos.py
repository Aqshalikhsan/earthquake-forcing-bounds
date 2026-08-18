"""
Parser for the SSGEOS lunar-phase dataset files.

The repo ships three .txt files whose records look like this:

    2025-07-29, 23:24:52, M 8.8 OFF EAST COAST OF KAMCHATKA
    First Quarter: 2025-08-01, 12:41:55; New Moon: 2025-07-24, 19:11:44;
    ~ 2.55 days before First Quarter; ~ 5.18 days after New Moon; Moon distance: 399021.95 km

None of the seven scripts in the repo actually read these files -- every figure
script has the numbers pasted in by hand. This module reads them for real, so
the published claims can be recomputed instead of taken on trust.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

SYNODIC = 29.530588853

# Phase angle (degrees of Moon-Sun elongation) at each cardinal phase.
PHASE_ANGLE = {
    "New Moon": 0.0,
    "First Quarter": 90.0,
    "Full Moon": 180.0,
    "Third Quarter": 270.0,
}

_EVENT_RE = re.compile(
    r"^(?P<date>-?\d{4}-\d{2}-\d{2}),\s*(?P<time>\d{2}:\d{2}:\d{2}),\s*M\s*(?P<mag>\d+\.\d+)\s+(?P<place>.+?)\s*$"
)
_PHASE_RE = re.compile(
    r"(?P<name>New Moon|First Quarter|Full Moon|Third Quarter)"
    r"(?:\s*\((?P<eclipse>[^)]*eclipse)\))?:\s*"
    r"(?P<date>-?\d{4}-\d{2}-\d{2}),\s*(?P<time>\d{2}:\d{2}:\d{2})"
)
_OFFSET_RE = re.compile(
    r"~\s*(?P<days>\d+\.\d+)\s+days\s+(?P<dir>before|after)\s+"
    r"(?P<name>New Moon|First Quarter|Full Moon|Third Quarter)"
)
_DIST_RE = re.compile(r"Moon distance:\s*(?P<km>\d+\.\d+)\s*km")


@dataclass
class Event:
    dt: datetime
    magnitude: float
    place: str
    # the two bracketing cardinal phases, as (name, datetime, eclipse_or_None)
    phases: list
    # offsets as reported by SSGEOS: (days, 'before'|'after', phase_name)
    offsets: list
    moon_distance_km: float
    source_file: str
    line_no: int

    @property
    def year(self) -> int:
        return self.dt.year

    @property
    def nearest_offset(self) -> float:
        """Days to the closest cardinal phase, as SSGEOS reports it."""
        return min(o[0] for o in self.offsets)

    @property
    def nearest_phase(self) -> str:
        return min(self.offsets, key=lambda o: o[0])[2]

    @property
    def is_syzygy(self) -> bool:
        return self.nearest_phase in ("New Moon", "Full Moon")

    @property
    def eclipse(self):
        """Eclipse tag attached to whichever phase is nearest, if any."""
        near = self.nearest_phase
        for name, _dt, ecl in self.phases:
            if name == near:
                return ecl
        return None

    def phase_angle(self) -> float:
        """
        Reconstruct the Moon's phase angle (0-360 deg) at the moment of the
        quake by interpolating between the two bracketing cardinal phases that
        SSGEOS itself lists. This turns their own numbers into circular data,
        which is what a window-free test needs.
        """
        ordered = sorted(self.phases, key=lambda p: p[1])
        (n0, t0, _), (n1, t1, _) = ordered[0], ordered[1]
        span = (t1 - t0).total_seconds()
        if span <= 0:
            raise ValueError(f"non-monotonic phases for {self.place}")
        frac = (self.dt - t0).total_seconds() / span
        a0 = PHASE_ANGLE[n0]
        a1 = PHASE_ANGLE[n1]
        # walk forward from a0 to a1 the short way round the circle
        delta = (a1 - a0) % 360.0
        return (a0 + frac * delta) % 360.0

    def offset_consistency_error(self) -> float:
        """
        Largest disagreement (in days) between the offsets SSGEOS printed and
        the offsets implied by the phase timestamps they printed on the same
        line. An internal cross-check of their own file.
        """
        worst = 0.0
        for days, direction, name in self.offsets:
            for pname, pdt, _ in self.phases:
                if pname != name:
                    continue
                implied = (self.dt - pdt).total_seconds() / 86400.0
                signed = days if direction == "after" else -days
                worst = max(worst, abs(implied - signed))
        return worst


def _parse_dt(date_s: str, time_s: str) -> datetime:
    y, m, d = (int(x) for x in date_s.split("-"))
    hh, mm, ss = (int(x) for x in time_s.split(":"))
    # datetime supports year >= 1; all records here are year 365+
    return datetime(y, m, d, hh, mm, ss)


def parse_file(path: str | Path) -> list[Event]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    events: list[Event] = []
    i = 0
    while i < len(lines):
        m_ev = _EVENT_RE.match(lines[i].strip())
        if not m_ev:
            i += 1
            continue

        # the record is this line plus the following non-blank lines
        block = [lines[i]]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not _EVENT_RE.match(lines[j].strip()):
            block.append(lines[j])
            j += 1
        blob = " ".join(block)

        phases = [
            (m.group("name"), _parse_dt(m.group("date"), m.group("time")), m.group("eclipse"))
            for m in _PHASE_RE.finditer(blob)
        ]
        offsets = [
            (float(m.group("days")), m.group("dir"), m.group("name"))
            for m in _OFFSET_RE.finditer(blob)
        ]
        m_dist = _DIST_RE.search(blob)

        if len(phases) >= 2 and len(offsets) >= 2 and m_dist:
            events.append(
                Event(
                    dt=_parse_dt(m_ev.group("date"), m_ev.group("time")),
                    magnitude=float(m_ev.group("mag")),
                    place=m_ev.group("place"),
                    phases=phases[:2],
                    offsets=offsets[:2],
                    moon_distance_km=float(m_dist.group("km")),
                    source_file=path.name,
                    line_no=i + 1,
                )
            )
        i = j

    return events


if __name__ == "__main__":
    import sys

    base = SSGEOS / "datasets"
    for f in sorted(base.glob("*.txt")):
        evs = parse_file(f)
        print(f"{f.name}: {len(evs)} events, "
              f"M {min(e.magnitude for e in evs):.1f}-{max(e.magnitude for e in evs):.1f}, "
              f"years {min(e.year for e in evs)}-{max(e.year for e in evs)}")
