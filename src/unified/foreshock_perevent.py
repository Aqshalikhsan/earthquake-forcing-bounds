"""
Foreshocks at their own location: closing the last formal gap.

foreshock_test.py asked whether the sky separates a foreshock from an ordinary
small earthquake, and found nothing -- but it read every forcing as one global
number per day, which resolution_test.py showed discards about 98% of the local
variation. For LUNAR, PLANETARY, ROTATION and SOLAR that costs nothing, because
those forcings really are global. For the four field families it means the
question was never properly put.

Mainshocks and aftershocks have both already been re-tested at full resolution
(perevent_test.py, aftershock_perevent.py) and neither answer moved. Foreshocks
were the one class still untested this way, so this file finishes the set:
every candidate earthquake is scored against the field directly above its own
epicentre.

    candidates   shallow Mw >= 5.0, Global CMT
    label        1 if followed by Mw >= 6.0 within RADIUS km and WINDOW days
    statistic    difference in mean local forcing, foreshock minus the rest
    null         whole-year circular offsets with +/- JITTER days of slack, so
                 the season each event sits in is preserved and only the
                 year-to-year alignment is destroyed
    tidal        Schuster on the foreshocks' own fault planes; the statistic is
                 exactly uniform under the null, so no shifting is needed

Sampling stays causal for everything measured -- pressure 1-3 days before, TEC
1-5 days before, GRACE a full month before -- so a foreshock cannot contaminate
its own predictor, and neither can the mainshock that follows it.

The sampler here is vectorised rather than the per-event loop used elsewhere.
With 42,000 candidates and hundreds of null replicates the loop version would
take hours; an ordinal-to-row lookup table gives the same numbers in minutes.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from gcmt import parse_ndk
from honest_test import haversine_vec
from perevent_test import build_atmosphere, build_ionosphere, build_hydro
from foreshock_test import label_foreshocks, CAND_MW, TARGET_MW, RADIUS, WINDOW

MAXDEPTH = 70.0
N_SHIFT = 200
JITTER = 15
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


class FastField:
    """Vectorised causal sampler over a (day, lat, lon) field."""

    def __init__(self, gf):
        self.arr = gf.arr
        self.ilat = np.asarray(gf.ilat)
        self.ilon = np.asarray(gf.ilon)
        self.lag = gf.lag
        self.vars = gf.vars
        d = np.asarray(gf.days, dtype=np.int64)
        self.lo_ord = int(d.min())
        table = np.full(int(d.max()) - self.lo_ord + 1, -1, dtype=np.int64)
        table[d - self.lo_ord] = np.arange(d.size)
        self.table = table
        self.valid = self.ilat >= 0

    def _rows(self, ords, lag):
        """Row index of (ord - lag) for every event, or -1 if outside."""
        k = (ords.astype(np.int64) - lag) - self.lo_ord
        good = (k >= 0) & (k < self.table.size)
        out = np.full(ords.size, -1, dtype=np.int64)
        out[good] = self.table[k[good]]
        return out

    def sample(self, ords):
        """Mean, mean-abs and mean-|delta| over the causal window."""
        lo, hi = self.lag
        n = ords.size
        vals = np.full((hi - lo + 1, n), np.nan)
        for j, g in enumerate(range(lo, hi + 1)):
            r = self._rows(ords, g)
            ok = (r >= 0) & self.valid
            if ok.any():
                idx = np.flatnonzero(ok)
                vals[j, idx] = self.arr[r[idx], self.ilat[idx], self.ilon[idx]]
        with np.errstate(invalid="ignore"):
            cnt = np.isfinite(vals).sum(axis=0)
            lvl = np.nanmean(vals, axis=0)
            ab = np.nanmean(np.abs(vals), axis=0)
            rate = (np.nanmean(np.abs(np.diff(vals, axis=0)), axis=0)
                    if vals.shape[0] > 1 else np.full(n, np.nan))
        bad = cnt < 1
        lvl[bad] = ab[bad] = rate[bad] = np.nan
        return np.column_stack([lvl, ab, rate])


class MergedFast:
    """Primary grid, falling back to a second one where it has no data."""

    def __init__(self, a, b):
        self.a, self.b, self.vars = a, b, a.vars

    def sample(self, ords):
        sa, sb = self.a.sample(ords), self.b.sample(ords)
        miss = ~np.isfinite(sa[:, 0])
        sa[miss] = sb[miss]
        return sa


def main():
    rule("FORESHOCK DI LOKASINYA SENDIRI — LUBANG FORMAL TERAKHIR")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= MAXDEPTH and e.mw >= CAND_MW]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lats = np.array([e.lat for e in ev])
    lons = np.array([e.lon for e in ev])
    y = label_foreshocks(ordn, mw, lats, lons)
    a, b = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    print(f"""
  kandidat    Mw >= {CAND_MW} dangkal : {mw.size:,}
  foreshock   disusul Mw >= {TARGET_MW} dalam {RADIUS:.0f} km / {WINDOW} hari : {a.size:,}
  bukan                                                : {b.size:,}
  rentang     {datetime.fromordinal(int(ordn.min())).date()} .. """
          f"""{datetime.fromordinal(int(ordn.max())).date()}
""")

    fams = {}
    for label, builder in [("ATMOSPHERE", build_atmosphere),
                           ("IONOSPHERE", build_ionosphere),
                           ("HYDRO", build_hydro)]:
        try:
            gf = builder(lats, lons)
            if hasattr(gf, "a"):
                # GRACE: land water reaches ~15% of epicentres, ocean bottom
                # pressure covers the rest. Keep both or the family loses 90%
                # of its events to a detail of which grid they land on.
                fams[label] = MergedFast(FastField(gf.a), FastField(gf.b))
            else:
                fams[label] = FastField(gf)
            s = fams[label].sample(ordn)
            print(f"    {label:<12} siap  "
                  f"({int(np.isfinite(s[:, 0]).sum()):,} kandidat tercakup)")
        except Exception as exc:
            print(f"    {label:<12} DILEWATI: {type(exc).__name__}: {exc}")

    def contrast(f, ords):
        s = f.sample(ords)
        with np.errstate(invalid="ignore"):
            out = np.full(3, np.nan)
            for j in range(3):
                va, vb = s[a, j], s[b, j]
                if np.isfinite(va).sum() > 50 and np.isfinite(vb).sum() > 50:
                    out[j] = np.nanmean(va) - np.nanmean(vb)
        return out

    obs = {k: contrast(f, ordn) for k, f in fams.items()}

    span = int(ordn.max() - ordn.min())
    n_year = max(int(span // 365.25) - 1, 1)
    print(f"\n    null: {n_year} offset tahunan x +/-{JITTER} hari "
          f"= {n_year*(2*JITTER+1):,} pergeseran (musim tetap)")
    null = {k: np.full((N_SHIFT, 3), np.nan) for k in fams}
    base = ordn.min()
    for i in range(N_SHIFT):
        off = int(round(365.25 * RNG.integers(1, n_year + 1))
                  + RNG.integers(-JITTER, JITTER + 1))
        sh = base + (ordn - base + off) % span
        for k, f in fams.items():
            null[k][i] = contrast(f, sh)
        if (i + 1) % 20 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 40, end="\r")

    rule("HASIL PER KELUARGA  (dibaca di episentrum kandidat)")
    print(f"""
  Membandingkan {a.size:,} foreshock dengan {b.size:,} gempa kecil biasa,
  masing-masing dibaca di lokasinya sendiri.
""")
    print(f"  {'keluarga':<12} {'variabel':<14} {'z maks':>8} {'p':>8}")
    print("  " + "-" * 48)
    pv = {}
    for k, f in fams.items():
        nb = null[k]
        mu, sd = np.nanmean(nb, axis=0), np.nanstd(nb, axis=0)
        good = np.isfinite(sd) & (sd > 0) & np.isfinite(obs[k])
        if not good.any():
            print(f"  {k:<12} (tak terukur)")
            continue
        z_o = np.where(good, np.abs(obs[k] - mu) / np.where(sd > 0, sd, 1), 0)
        z_n = np.abs(nb - mu) / np.where(sd > 0, sd, 1)
        z_n[:, ~good] = 0
        m_o = float(np.nanmax(z_o))
        m_n = np.nanmax(z_n, axis=1)
        p = float((m_n >= m_o).mean())
        pv[k] = p
        print(f"  {k:<12} {f.vars[int(np.nanargmax(z_o))]:<14} {m_o:>8.2f} "
              f"{p:>8.3f}{'  <-- LOLOS' if p < 0.05 else ''}")

    # ---- tidal: Schuster on the foreshocks' own fault planes ---------------
    try:
        from tidal_analysis import compute_tidal
        phase, _ = compute_tidal([ev[i] for i in a], plane="shallow")
        ok = np.isfinite(phase)
        ang = np.radians(phase[ok]); n = ang.size
        R2 = np.sum(np.cos(ang)) ** 2 + np.sum(np.sin(ang)) ** 2
        p_t = float(np.exp(-R2 / n))
        pv["TIDAL"] = p_t
        print(f"  {'TIDAL':<12} {'phase(Schuster)':<14} {'--':>8} {p_t:>8.3f}"
              f"{'  <-- LOLOS' if p_t < 0.05 else ''}")
        print(f"  {'':<12} n = {n:,}   R/n = {np.sqrt(R2)/n:.4f}")
    except Exception as exc:
        print(f"  {'TIDAL':<12} DILEWATI: {type(exc).__name__}: {exc}")

    # ---- joint correction: a within-family p is not the final p ------------
    #
    # Four families were searched, so the smallest of four p-values is itself a
    # selected quantity and has to be paid for. The null min-p is built from the
    # same replicates, each family's null statistics ranked against its own
    # siblings, so the correction costs nothing extra to compute and cannot be
    # skipped by accident.
    #
    rule("KOREKSI GABUNGAN: min-p lintas keluarga")
    keys = [k for k in fams if k in pv]
    if keys:
        cols = len(keys) + (1 if "TIDAL" in pv else 0)
        pn = np.empty((N_SHIFT, cols))
        for j, k in enumerate(keys):
            nb = null[k]
            mu, sd = np.nanmean(nb, axis=0), np.nanstd(nb, axis=0)
            good = np.isfinite(sd) & (sd > 0)
            z_n = np.abs(nb - mu) / np.where(sd > 0, sd, 1)
            z_n[:, ~good] = 0
            m_n = np.nanmax(z_n, axis=1)
            for i in range(N_SHIFT):
                pn[i, j] = (np.sum(m_n >= m_n[i]) - 1) / (N_SHIFT - 1)
        if "TIDAL" in pv:
            pn[:, -1] = RNG.random(N_SHIFT)      # Schuster p is uniform on H0
        nm = pn.min(axis=1)
        obs_min = min(pv.values())
        p_joint = float((nm <= obs_min).mean())
        print(f"""
    keluarga diuji    : {cols}
    min-p data asli   : {obs_min:.4f}
    min-p pada null   : median {np.median(nm):.4f}, minimum {nm.min():.4f}
    p TERKOREKSI      : {p_joint:.3f}
""")
    else:
        p_joint = np.nan

    # ---- stability: does the best family survive in BOTH halves? -----------
    #
    # This is the check that dissolved the lag-sweep results, and it is the one
    # a marginal p-value most often fails. A real effect is a property of the
    # physics and must appear in the first half of the catalogue and the second
    # half alike. An effect assembled out of noise will not reproduce, because
    # the noise differs between halves.
    #
    best = min(pv, key=pv.get) if pv else None
    if best in fams:
        rule(f"UJI STABILITAS: apakah {best} bertahan di KEDUA separuh katalog?")
        cut = np.median(ordn)
        f = fams[best]
        for lbl, m in [("separuh awal", ordn <= cut), ("separuh akhir", ordn > cut)]:
            aa = np.flatnonzero((y == 1) & m)
            bb = np.flatnonzero((y == 0) & m)
            s = f.sample(ordn)
            with np.errstate(invalid="ignore"):
                o = np.array([np.nanmean(s[aa, j]) - np.nanmean(s[bb, j])
                              for j in range(3)])
            nb = np.full((N_SHIFT, 3), np.nan)
            for i in range(N_SHIFT):
                off = int(round(365.25 * RNG.integers(1, n_year + 1))
                          + RNG.integers(-JITTER, JITTER + 1))
                s2 = f.sample(base + (ordn - base + off) % span)
                with np.errstate(invalid="ignore"):
                    nb[i] = [np.nanmean(s2[aa, j]) - np.nanmean(s2[bb, j])
                             for j in range(3)]
            mu, sd = np.nanmean(nb, axis=0), np.nanstd(nb, axis=0)
            good = np.isfinite(sd) & (sd > 0) & np.isfinite(o)
            z_o = np.where(good, np.abs(o - mu) / np.where(sd > 0, sd, 1), 0)
            z_n = np.abs(nb - mu) / np.where(sd > 0, sd, 1)
            z_n[:, ~good] = 0
            m_o = float(np.nanmax(z_o))
            p = float((np.nanmax(z_n, axis=1) >= m_o).mean())
            print(f"  {lbl:<15} n foreshock {aa.size:>5,}   "
                  f"variabel {f.vars[int(np.nanargmax(z_o))]:<12} "
                  f"z {m_o:>5.2f}   p {p:>6.3f}")

    rule("VONIS")
    if pv:
        worst = min(pv.values())
        print(f"""
  p terkecil dalam keluarga           : {worst:.3f}
  p setelah koreksi gabungan          : {p_joint:.3f}
  {'ADA yang lolos koreksi gabungan.' if p_joint < 0.05
   else 'TIDAK ADA yang lolos setelah koreksi gabungan.'}

  Dengan ini ketiga kelas gempa sudah diuji pada resolusi penuh:

      foreshock      perevent  <- berkas ini
      gempa induk    perevent_test.py          p = 0,187
      gempa susulan  aftershock_perevent.py    p = 0,273

  Tidak ada lubang formal yang tersisa: setiap keluarga sudah diuji pada
  resolusi yang dimilikinya, pada ketiga kelas gempa, dengan null yang
  menjaga musim dan sampling yang menjaga arah sebab-akibat.
""")


if __name__ == "__main__":
    main()
