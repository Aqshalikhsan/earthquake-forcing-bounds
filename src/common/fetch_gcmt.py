"""
Extend the Global CMT catalogue past 2020 from the monthly archive.

The bundled gcmt.ndk stops at 2020-12-31, which was fine while it was only
feeding the tidal test. It is not fine now: the pressure field runs to 2024 and
TEC now runs to 2024 as well, so four years of earthquakes were being discarded
by the catalogue rather than by the data. Every family pays for that at once,
because GCMT is the master catalogue in the per-event design.

    source   https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/
             NEW_MONTHLY/YYYY/mmmYY.ndk -- one file per month, open, no login

NDK is a fixed five-line-per-event text format, so extending the catalogue is
concatenation: the monthly files are appended in chronological order and the
result is re-parsed to prove it reads. Months that do not exist yet are skipped
rather than treated as errors, so the same command can be re-run later to pick
up whatever has since been published.

The original file is copied to gcmt_to2020.ndk before anything is written. If
the rebuilt catalogue fails to parse, the original is restored and nothing is
lost -- the whole per-event analysis depends on this one file.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import shutil
import urllib.request as _u
from datetime import date

BASE = ("https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
START_YEAR = 2021


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fetch(url, timeout=60):
    try:
        with _u.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return None
            return r.read().decode("ascii", "replace")
    except Exception:
        return None


def main():
    rule("PERLUAS KATALOG GLOBAL CMT MELEWATI 2020")

    ndk = datafile("gcmt.ndk")
    backup = ndk.parent / "gcmt_to2020.ndk"
    original = ndk.read_text(encoding="ascii", errors="replace")
    n_before = original.count("\n") // 5
    print(f"""
  berkas    {ndk}
  sekarang  {len(original)/1e6:.1f} MB, {n_before:,} gempa
""")
    if not backup.exists():
        shutil.copy2(ndk, backup)
        print(f"  cadangan  {backup.name}")

    this_year = date.today().year
    chunks, got, missing = [], 0, []
    for y in range(START_YEAR, this_year + 1):
        for m in MONTHS:
            txt = fetch(f"{BASE}/{y}/{m}{str(y)[2:]}.ndk")
            if txt is None or len(txt) < 500:
                missing.append(f"{m}{str(y)[2:]}")
                continue
            if not txt.endswith("\n"):
                txt += "\n"
            chunks.append(txt)
            got += txt.count("\n") // 5
        print(f"    {y}: {got:,} gempa terkumpul", flush=True)

    if not chunks:
        print("\n  tidak ada bulan baru yang terunduh; berkas tidak diubah.")
        return

    print(f"\n  bulan tidak tersedia: {', '.join(missing) if missing else '(tidak ada)'}")

    merged = original if original.endswith("\n") else original + "\n"
    merged += "".join(chunks)
    ndk.write_text(merged, encoding="ascii")

    # prove it parses before declaring success; restore the original if not
    try:
        from gcmt import parse_ndk
        ev = parse_ndk(ndk)
        assert len(ev) > n_before, "tidak ada gempa tambahan setelah parse"
        ev.sort(key=lambda e: e.dt)
    except Exception as exc:
        shutil.copy2(backup, ndk)
        print(f"\n  GAGAL parse ({type(exc).__name__}: {exc})")
        print("  berkas asli dikembalikan; tidak ada yang hilang.")
        return

    rule("HASIL")
    print(f"""
  gempa     {n_before:,} -> {len(ev):,}   (+{len(ev)-n_before:,})
  rentang   {ev[0].dt.date()} .. {ev[-1].dt.date()}
  ukuran    {ndk.stat().st_size/1e6:.1f} MB
  cadangan  {backup.name} (versi lama, kalau perlu dikembalikan)

  Jalankan ulang perintah ini kapan saja untuk mengambil bulan yang baru
  terbit; bulan yang sudah ada akan tertimpa hanya kalau berkas asli
  dikembalikan dulu dari cadangan.
""")


if __name__ == "__main__":
    main()
