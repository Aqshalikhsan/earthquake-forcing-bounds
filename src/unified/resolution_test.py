"""
How much signal does global averaging destroy? Measured, not assumed.

Every family in forcing_bank is a single number per day. For LUNAR, PLANETARY
and ROTATION that is honest -- those forcings really are global, the Moon's
phase is the same for everyone. But three families are FIELDS:

    ATMOSPHERE   pressure anomaly, 939 cells
    HYDRO        GRACE water load, global grid
    IONOSPHERE   TEC, 5,183 cells

and they enter the bank as np.nanmean(field, axis=1). A physical trigger is
local: the crust under Sumatra does not care what the pressure is over Chile.
Averaging 939 cells to make one variable puts the one cell that matters in a
bucket with 938 that do not.

This file measures the cost of that in two ways.

    RETENTION   correlation between the anomaly directly above each epicentre
                and the global daily mean on the same day. This is the fraction
                of local variation the bank variable actually carries.

    POWER       a semi-synthetic injection. A triggering effect of known size is
                written into the real pressure field, earthquake times are
                redrawn to follow it, and both designs are then asked to find
                it: the per-event design that reads the anomaly above each
                epicentre, and the global-daily design used by the bank. The
                smallest effect each one can detect is the answer.

The injection preserves the real spatial distribution of earthquakes and the
real pressure field; only the timing is redrawn. Both designs face the same
circular-shift null and the same 0.05 threshold, so the comparison is fair.

If the per-event design detects effects the global design misses, then the
null results in the unified bank are partly an artefact of resolution, and the
fix is to raise resolution rather than to add more variables.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime

from honest_test import gardner_knopoff_window, haversine_vec

MEDIAN_WIN = 30
LAG = 2                       # days before the event
MINMAG, MAXDEPTH = 6.0, 70.0
N_NULL = 300
N_SIM = 40
BETAS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]
N_BINS = 10
RNG = np.random.default_rng(20260817)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def running_mean(a, win):
    """Running mean along axis 0 via cumsum. Removes elevation and season."""
    n = a.shape[0]
    half = win // 2
    pad = np.concatenate([np.repeat(a[:1], half, 0), a,
                          np.repeat(a[-1:], win - half, 0)], axis=0)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(pad, axis=0)])
    return a - (cs[win:win + n] - cs[:n]) / win


def load_quakes():
    out = []
    for r in csv.DictReader(io.StringIO(
            datafile("global_m6.csv").read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"])
            lo = float(r["longitude"]); dep = float(r["depth"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if m < MINMAG or dep > MAXDEPTH:
            continue
        out.append((dt, m, la, lo))
    out.sort(key=lambda e: e[0])
    return out


def decluster(rows):
    n = len(rows)
    days = np.array([(r[0] - rows[0][0]).total_seconds() / 86400.0 for r in rows])
    mags = np.array([r[1] for r in rows])
    lats = np.array([r[2] for r in rows])
    lons = np.array([r[3] for r in rows])
    removed = np.zeros(n, dtype=bool)
    for i in np.argsort(-mags):
        if removed[i]:
            continue
        dkm, ddays = gardner_knopoff_window(mags[i])
        delta = days - days[i]
        cand = (delta >= 0) & (delta <= ddays) & (mags <= mags[i]) & (~removed)
        cand[i] = False
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        d = haversine_vec(lats[i], lons[i], lats[idx], lons[idx])
        removed[idx[d <= dkm]] = True
    return [r for i, r in enumerate(rows) if not removed[i]]


def column_of(lats, lons, cells):
    ilat = np.clip(np.round((90.0 - lats) / 2.5).astype(int), 0, 72)
    ilon = np.round((lons % 360.0) / 2.5).astype(int) % 144
    flat = ilat * 144 + ilon
    pos = {int(c): j for j, c in enumerate(cells)}
    return np.array([pos.get(int(f), -1) for f in flat])


def decile_stat(x, y, n_bins=N_BINS, min_bin=300):
    """Largest deviation of any decile's rate from the overall rate."""
    e = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if e.size < 3:
        return 0.0
    b = np.clip(np.digitize(x, e[1:-1]), 0, e.size - 2)
    base = y.mean()
    cnt = np.bincount(b, minlength=e.size - 1).astype(float)
    s = np.bincount(b, weights=y, minlength=e.size - 1)
    ok = cnt >= min_bin
    if not ok.any():
        return 0.0
    rates = np.where(ok, s / np.maximum(cnt, 1), base)
    return float(np.max(np.abs(rates - base)))


def main():
    rule("UJI RESOLUSI: berapa banyak sinyal yang dibunuh perataan global?")

    arr = np.load(datafile("pressure_cells.npy")).astype(np.float32)
    days_p = np.load(datafile("pressure_days.npy"))
    cells = np.load(datafile("pressure_cellid.npy"))
    order = np.argsort(days_p)
    days_p, arr = days_p[order], arr[order]

    print(f"""
  medan     tekanan NCEP, {arr.shape[0]:,} hari x {arr.shape[1]:,} sel
  detrend   rerata berjalan {MEDIAN_WIN} hari per sel (buang ketinggian+musim)
""")
    anom = running_mean(arr, MEDIAN_WIN)
    z = (anom - np.nanmean(anom)) / np.nanstd(anom)      # baku, satuan sd
    gmean = np.nanmean(z, axis=1)                        # variabel bank
    print(f"  lokal     sd {np.nanstd(anom):>7.1f} Pa")
    print(f"  global    sd {np.nanstd(np.nanmean(anom, axis=1)):>7.1f} Pa"
          f"   <- setelah dirata-rata {arr.shape[1]} sel")

    ev = decluster(load_quakes())
    ev = [e for e in ev if days_p[0] + MEDIAN_WIN <= e[0].toordinal() <= days_p[-1]]
    lats = np.array([e[2] for e in ev]); lons = np.array([e[3] for e in ev])
    col = column_of(lats, lons, cells)
    row = np.searchsorted(days_p, np.array([e[0].toordinal() for e in ev]) - LAG)
    keep = (col >= 0) & (row >= 0) & (row < len(days_p))
    col, row = col[keep], row[keep]
    n_ev = len(col)
    print(f"\n  gempa     M>=6,0 dangkal, declustered: {n_ev:,}")

    # ---------------------------------------------------------------- RETENSI
    rule("RETENSI: seberapa banyak variasi lokal yang dibawa variabel global?")
    loc = z[row, col]
    glo = gmean[row]
    ok = np.isfinite(loc) & np.isfinite(glo)
    r = float(np.corrcoef(loc[ok], glo[ok])[0, 1])
    print(f"""
    anomali di atas episentrum  vs  rerata global hari yang sama

      korelasi        r = {r:+.4f}
      variasi terbawa r^2 = {100*r**2:.2f}%
      terbuang            {100*(1-r**2):.2f}%
""")
    print(f"  Artinya efek lokal sebesar E muncul di variabel bank sebagai"
          f" ~{abs(r):.3f} E.")
    print(f"  Untuk mendeteksinya butuh sekitar {1/max(r**2,1e-9):,.0f}x lebih banyak data.")

    # ------------------------------------------------------------------ DAYA
    rule("DAYA: efek sekecil apa yang bisa ditemukan tiap rancangan?")
    print(f"""
  Efek pemicu berukuran diketahui ditulis ke medan tekanan NYATA, waktu gempa
  diundi ulang mengikutinya, lalu kedua rancangan disuruh mencarinya.
  Lokasi gempa dan medannya tetap asli; hanya waktunya yang diundi.

  beta   = kekuatan pemicu (kenaikan laju per 1 sd anomali lokal)
  deteksi= persen dari {N_SIM} simulasi yang mencapai p<0,05
""")
    nd = len(days_p)
    zc = np.nan_to_num(z[:, col])                 # (hari x gempa) di sel gempa
    print(f"  {'beta':>6} {'modulasi lokal':>16} {'PER-EVENT':>11} {'GLOBAL-HARIAN':>15}")
    print("  " + "-" * 54)

    for beta in BETAS:
        det_ev = det_gl = 0
        for s in range(N_SIM):
            rng = np.random.default_rng(1000 + s)
            w = np.clip(1.0 + beta * zc, 0.01, None)      # bobot waktu per gempa
            w /= w.sum(axis=0, keepdims=True)
            pick = np.array([rng.choice(nd, p=w[:, k]) for k in range(n_ev)])

            # --- rancangan A: per-event, baca anomali di atas episentrum ---
            obs_a = float(np.nanmean(z[pick, col]))
            nb = np.empty(N_NULL)
            for i in range(N_NULL):
                sh = (pick + rng.integers(30, nd - 30)) % nd
                nb[i] = float(np.nanmean(z[sh, col]))
            p_a = float((np.abs(nb - nb.mean()) >= abs(obs_a - nb.mean())).mean())

            # --- rancangan B: global-harian, seperti forcing_bank ---
            y = np.zeros(nd); y[pick] = 1.0
            obs_b = decile_stat(gmean, y)
            nbg = np.empty(N_NULL)
            for i in range(N_NULL):
                nbg[i] = decile_stat(gmean, np.roll(y, int(rng.integers(1, nd))))
            p_b = float((nbg >= obs_b).mean())

            det_ev += p_a < 0.05
            det_gl += p_b < 0.05
        print(f"  {beta:>6.1f} {100*beta*0.5:>15.0f}% {100*det_ev/N_SIM:>10.0f}% "
              f"{100*det_gl/N_SIM:>14.0f}%")

    rule("ARTINYA")
    print(f"""
  Variabel bank hanya membawa {100*r**2:.1f}% variasi lokal. Sisanya dibuang
  sebelum uji apa pun dijalankan.

  EMPAT keluarga terkena ini, karena forcingnya bersifat lokal:
    ATMOSPHERE  939 sel   -> 1 angka/hari
    HYDRO       grid GRACE-> 1 angka/hari
    IONOSPHERE  5.183 sel -> 1 angka/hari
    TIDAL       dihitung di SATU situs acuan (palung Jawa), bukan di episentrum
  Hasil nol keempatnya di uji terpadu TIDAK boleh dibaca sebagai "tidak ada
  efek" -- rancangannya memang hampir buta terhadap efek lokal.

  EMPAT keluarga lain memang global, jadi perataan tidak membuang apa pun dan
  batas atasnya sah apa adanya:
    LUNAR (fase, deklinasi, jarak), PLANETARY (sudut), ROTATION (LOD, kutub),
    SOLAR (Kp, F10.7, bintik) -- semuanya sama untuk seluruh Bumi pada hari
    yang sama.

  Untuk keempat keluarga medan, pendalaman yang berarti bukan menambah variabel
  melainkan menaikkan resolusi: uji per-gempa di lokasinya sendiri, seperti
  yang sudah dilakukan pressure_test.py, grace_hydro2.py, tidal_analysis.py
  dan tec_test.py -- tetapi keempatnya belum pernah disatukan dalam satu null
  dan satu koreksi seperti unified_test.py melakukannya untuk variabel global.
""")


if __name__ == "__main__":
    main()
