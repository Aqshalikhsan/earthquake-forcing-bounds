"""
Path setup shared by every script.

Importing this makes all topic folders under src/ importable regardless of
which folder the running script lives in, and exposes the canonical locations
of the data and paper directories. It is the only thing that knows the layout.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DATA = ROOT / "data"
PAPERS = ROOT / "papers"
SSGEOS = ROOT / "ssgeos_original"

for _d in sorted(SRC.iterdir()):
    if _d.is_dir() and not _d.name.startswith(("_", ".")):
        _s = str(_d)
        if _s not in sys.path:
            sys.path.append(_s)


_DATA_INDEX = None


def datafile(name):
    """
    Locate a data file wherever it sits under data/.

    data/ is organised into topic subfolders (catalogs, solar, hydro, ...), but
    scripts should not need to know which one holds what -- and downloads in
    progress land in data/ root before being filed. This searches the tree once,
    caches the result, and falls back to data/<name> so the same call also works
    as a destination for writing something that does not exist yet.
    """
    global _DATA_INDEX
    if _DATA_INDEX is None:
        _DATA_INDEX = {}
        if DATA.exists():
            for p in DATA.rglob("*"):
                _DATA_INDEX.setdefault(p.name, p)
    hit = _DATA_INDEX.get(name)
    return hit if hit is not None else DATA / name
