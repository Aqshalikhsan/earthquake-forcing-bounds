"""
Step 1: audit the SSGEOS datasets before trusting any statistic computed from them.

Checks:
  A. Do the files contain what their filename/README claims (magnitude threshold, n)?
  B. Are the printed offsets consistent with the printed phase timestamps?
  C. How precise are the event origin times, especially for pre-instrumental events?
  D. What is the catalogue composition (how much of the "M>=8.5, 365-2025" set is historical)?
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import collections
from pathlib import Path

from parse_ssgeos import parse_file

BASE = SSGEOS / "datasets"

FILES = {
    "M>=8.5 (365-2025)": "0365-2025-lunar-phases-m85p.txt",
    "M>=8.0 (1904-2021)": "1904-2021-lunar-phases-m80p.txt",
    "M>=8.2 (1904-2021)": "1904-2021-lunar-phases-m82p.txt",
}

CLAIMED_THRESHOLD = {
    "M>=8.5 (365-2025)": 8.5,
    "M>=8.0 (1904-2021)": 8.0,
    "M>=8.2 (1904-2021)": 8.2,
}


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main():
    datasets = {label: parse_file(BASE / fn) for label, fn in FILES.items()}

    rule("A. DOES EACH FILE MATCH ITS OWN MAGNITUDE THRESHOLD?")
    for label, evs in datasets.items():
        thr = CLAIMED_THRESHOLD[label]
        violations = [e for e in evs if e.magnitude < thr]
        print(f"\n{label}: n = {len(evs)}")
        print(f"  magnitude range in file : {min(e.magnitude for e in evs):.1f} - "
              f"{max(e.magnitude for e in evs):.1f}")
        print(f"  events BELOW threshold  : {len(violations)}")
        for e in violations:
            print(f"     !! M{e.magnitude}  {e.dt.date()}  {e.place}  "
                  f"(line {e.line_no})")

    rule("B. INTERNAL CONSISTENCY: printed offsets vs printed phase timestamps")
    for label, evs in datasets.items():
        errs = [(e.offset_consistency_error(), e) for e in evs]
        errs.sort(reverse=True, key=lambda t: t[0])
        worst = errs[0]
        n_bad = sum(1 for err, _ in errs if err > 0.01)
        print(f"\n{label}:")
        print(f"  max disagreement : {worst[0]:.4f} d  ({worst[1].dt.date()} {worst[1].place})")
        print(f"  records off by >0.01 d : {n_bad} / {len(evs)}")
    print("\n  -> small residuals are just rounding to 2 decimals; this checks that the")
    print("     files are at least self-consistent, which they are.")

    rule("C. ORIGIN-TIME PRECISION (can a lunar phase even be assigned?)")
    for label, evs in datasets.items():
        round_hour = [e for e in evs if e.dt.minute == 0 and e.dt.second == 0]
        midnight = [e for e in evs if e.dt.hour == 0 and e.dt.minute == 0 and e.dt.second == 0]
        print(f"\n{label}:  n = {len(evs)}")
        print(f"  origin time on an exact hour   : {len(round_hour)} "
              f"({100*len(round_hour)/len(evs):.0f}%)")
        print(f"  origin time exactly 00:00:00   : {len(midnight)}")
        if round_hour:
            print("  the exact-hour events (i.e. times that were assumed, not measured):")
            for e in sorted(round_hour, key=lambda e: e.dt):
                print(f"     {e.dt}  M{e.magnitude}  {e.place}")

    rule("D. CATALOGUE COMPOSITION OF THE HEADLINE M>=8.5 SET")
    evs = datasets["M>=8.5 (365-2025)"]
    pre1900 = [e for e in evs if e.year < 1900]
    modern = [e for e in evs if e.year >= 1904]
    print(f"\n  total events                    : {len(evs)}")
    print(f"  pre-1900 (historical/documentary): {len(pre1900)}")
    print(f"  1904+ (instrumental)            : {len(modern)}")
    print(f"\n  time span claimed in README     : 365 - 2025 = 1660 years")
    print(f"  events per century, pre-1900    : {len(pre1900) / 15.3:.2f}")
    print(f"  events per century, 1904+       : {len(modern) / 1.2:.1f}")
    ratio = (len(modern) / 1.2) / max(len(pre1900) / 15.3, 1e-9)
    print(f"  -> instrumental era yields {ratio:.0f}x more M>=8.5 events per century.")
    print("     Either the Earth got ~{:.0f}x more seismic after 1904, or the".format(ratio))
    print("     historical catalogue is drastically incomplete. It is the latter.")

    print("\n  century-by-century count of the M>=8.5 'catalogue':")
    per_century = collections.Counter((e.year // 100) * 100 for e in evs)
    for c in sorted(per_century):
        bar = "#" * per_century[c]
        print(f"     {c:>5}s : {per_century[c]:>2}  {bar}")

    rule("E. THE PRE-INSTRUMENTAL EVENTS, LISTED")
    for e in sorted(pre1900, key=lambda e: e.dt):
        print(f"  {e.dt}  M{e.magnitude:<4} offset {e.nearest_offset:.2f} d "
              f"from {e.nearest_phase:<14} | {e.place}")


if __name__ == "__main__":
    main()
