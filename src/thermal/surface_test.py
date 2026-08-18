"""
Near-surface temperature and water vapour before earthquakes. The last test.

olr_test.py measured the infrared leaving the top of the atmosphere. The
thermal precursor literature usually means something nearer the ground: air a
degree or two warmer above the fault in the days before it goes, or a
water-vapour anomaly in the same place. Those are different quantities from
OLR, so this closes the claim in the form it is actually made.

Everything follows the design already used for pressure, TEC and OLR, so the
numbers land on the same scale and can be read straight across:

    anomaly    each cell against its own 30-day running mean, which removes the
               seasonal cycle and the climate of the place -- Sumatra is warm
               every day of every year, and that is not a precursor
    sampling   days 1-7 BEFORE the origin time. These are symptoms, not forces,
               so an effect the earthquake itself caused could otherwise be
               read as one preceding it
    null       whole-year circular offsets, +/-15 days of slack, holding each
               event's season fixed
    questions  mainshocks against their own locations at other times, and
               foreshocks against small earthquakes that led to nothing

The second question is the one that matters. Both groups sit in seismically
active crust with the same climate and terrain, so those cancel, and what
remains is the claim itself: given a small earthquake, does the ground above it
behave differently when a large one is coming?
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np

from gcmt import parse_ndk
from honest_test import gardner_knopoff_window, haversine_vec
from foreshock_perevent import FastField
from foreshock_test import label_foreshocks, CAND_MW, TARGET_MW
from olr_test import running_mean, _Shim, maxstat, MEDIAN_WIN, LAG

MAXDEPTH = 70.0
N_SHIFT = 300
JITTER = 15
RNG = np.random.default_rng(20260817)

FIELDS = {"air": ("suhu udara permukaan", "K"),
          "pwat": ("uap air kolom", "kg/m2")}


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def build_surface(key, lats, lons):
    vals = np.load(datafile(f"surf_{key}_cells.npy")).astype(np.float32)
    days = np.load(datafile(f"surf_{key}_days.npy"))
    cells = np.load(datafile("surf_cellid.npy"))
    o = np.argsort(days); days, vals = days[o], vals[o]
    anom = running_mean(vals, MEDIAN_WIN)

    ilat = np.clip(np.round((90.0 - lats) / 2.5).astype(int), 0, 72)
    ilon = np.round((lons % 360.0) / 2.5).astype(int) % 144
    pos = {int(c): j for j, c in enumerate(cells)}
    col = np.array([pos.get(int(f), -1) for f in ilat * 144 + ilon])
    return FastField(_Shim(days, anom[:, :, None], col,
                           np.zeros(lats.size, dtype=int), LAG,
                           [f"{key}_level", f"{key}_abs", f"{key}_rate"]))


def main():
    rule("SUHU PERMUKAAN & UAP AIR — DIUJI DI ATAS EPISENTRUM")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= MAXDEPTH and e.mw >= CAND_MW]
    ev.sort(key=lambda e: e.dt)
    mw = np.array([e.mw for e in ev])
    ordn = np.array([e.dt.toordinal() for e in ev], dtype=float)
    lats = np.array([e.lat for e in ev]); lons = np.array([e.lon for e in ev])

    big = np.flatnonzero(mw >= TARGET_MW)
    dead = np.zeros(big.size, dtype=bool)
    ob, mb, lb, nb_ = ordn[big], mw[big], lats[big], lons[big]
    for k in np.argsort(-mb):
        if dead[k]:
            continue
        dkm, ddays = gardner_knopoff_window(mb[k])
        d = ob - ob[k]
        c = (d > 0) & (d <= ddays) & (mb <= mb[k]) & (~dead)
        i = np.flatnonzero(c)
        if i.size:
            dead[i[haversine_vec(lb[k], nb_[k], lb[i], nb_[i]) <= dkm]] = True
    main_i = big[~dead]

    y = label_foreshocks(ordn, mw, lats, lons)
    a, b = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    span = int(ordn.max() - ordn.min())
    n_year = max(int(span // 365.25) - 1, 1)
    base = ordn.min()
    offs = [int(round(365.25 * RNG.integers(1, n_year + 1))
                + RNG.integers(-JITTER, JITTER + 1)) for _ in range(N_SHIFT)]

    print(f"""
  kandidat    Mw >= {CAND_MW} dangkal        : {mw.size:,}
  gempa induk (declustered)      : {main_i.size:,}
  foreshock / bukan              : {a.size:,} / {b.size:,}
  jendela     {LAG[0]}-{LAG[1]} hari SEBELUM gempa (kausal)
  null        {n_year} offset tahunan x +/-{JITTER} hari
""")

    rule("HASIL")
    print(f"\n  {'medan':<22} {'pertanyaan':<20} {'variabel':<12} "
          f"{'z':>6} {'p':>8}")
    print("  " + "-" * 72)

    allp = {}
    for key, (label, unit) in FIELDS.items():
        try:
            f = build_surface(key, lats, lons)
        except FileNotFoundError:
            print(f"  {label:<22} BELUM ADA DATA")
            continue
        s0 = f.sample(ordn)
        cov = float(np.isfinite(s0[:, 0]).mean())

        def stat_main(o_):
            s = f.sample(o_)
            with np.errstate(invalid="ignore"):
                return np.array([np.nanmean(s[main_i, j]) for j in range(3)])

        def stat_fore(o_):
            s = f.sample(o_)
            with np.errstate(invalid="ignore"):
                return np.array([np.nanmean(s[a, j]) - np.nanmean(s[b, j])
                                 for j in range(3)])

        for qlabel, fn in [("gempa induk", stat_main),
                           ("foreshock vs biasa", stat_fore)]:
            obs = fn(ordn)
            null = np.empty((N_SHIFT, 3))
            for i, off in enumerate(offs):
                null[i] = fn(base + (ordn - base + off) % span)
                if (i + 1) % 40 == 0:
                    print(f"    {key}/{qlabel} {i+1}/{N_SHIFT}",
                          end="\r", flush=True)
            print(" " * 50, end="\r")
            z, p, j = maxstat(obs, null)
            allp[f"{key}/{qlabel}"] = p
            print(f"  {label:<22} {qlabel:<20} {f.vars[j]:<12} "
                  f"{z:>6.2f} {p:>8.3f}"
                  f"{'  <-- LOLOS' if p < 0.0125 else ''}")
        print(f"  {'':<22} (tercakup {100*cov:.1f}% kandidat)")

    rule("VONIS")
    if allp:
        worst = min(allp.values())
        print(f"""
  Bonferroni atas {len(allp)} uji -> ambang p < {0.05/len(allp):.4f}
  p terkecil : {worst:.3f}
  {'ADA yang lolos -- perlu uji stabilitas separuh katalog.' if worst < 0.05/len(allp)
   else 'TIDAK ADA yang lolos.'}

  Klaim termal kini tertutup dalam ketiga bentuknya:

      OLR              puncak atmosfer, 1974-2022
      suhu permukaan   udara dekat tanah, 1970-2025
      uap air          kolom uap air, 1970-2025

  Ketiganya diuji di atas episentrum, dengan sampling kausal dan null yang
  menjaga musim. Bentuk permukaan inilah yang sebenarnya dimaksud sebagian
  besar paper prekursor termal, dan sampai uji ini bentuk itu belum tersentuh.
""")


if __name__ == "__main__":
    main()
