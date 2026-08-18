"""
One test, one null, one correction -- applied to every external forcing at once.

The point of this file is comparability. Each hypothesis below has been tested
before, but always separately: different catalogue, different period, different
statistic, different level of rigour. Nobody can lay the results side by side
and ask which forcing matters more, or how large an effect each one is now
ruled out at.

Here every family meets the same conditions:

  catalogue     global M >= 6.0, 1970-2024, USGS -- flat rate, no growth trend
  statistic     largest deviation of any decile's earthquake rate from the
                overall rate, which reads directly as "+/- X% modulation"
  null          circular time shift of the seismicity series. Preserves
                clustering, secular trend and seasonality; destroys only the
                alignment with the forcing.
  correction    max-statistic. Each shifted dataset is searched just as hard as
                the real one, so the p-value already accounts for however many
                variables a family contains. Reported per family and jointly.
  reference     ETAS on the same catalogue, to put every bound on a scale that
                means something.

The output is an upper bound per family: any modulation larger than the stated
figure would have been detected. That is the deliverable -- not a verdict of
"significant" or "not", but a number each hypothesis is now constrained by.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import numpy as np
from datetime import datetime

from forcing_bank import build_all
from validity import valid_mask, print_report

N_SHIFT = 400
N_BINS = 10
MIN_BIN = 300
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def bin_indices(X, n_bins=N_BINS, mask=None):
    """
    Decile index per day per variable, computed on real data only.

    Where a source does not span the whole grid, forcing_bank fills -- forward
    from the last value, and with the median before the record begins. Twelve of
    the 130 variables are majority fill: sol_f107 has real data only from 2004,
    GRACE from 2002, TEC from 1998. Binning those columns over the whole grid
    puts more than half the days into a single tie, collapses the quantile edges
    and leaves only four to six deciles populated. The edges are therefore taken
    over the valid days alone, and filled days are marked -1 and dropped.
    """
    n, m = X.shape
    idx = np.empty((n, m), dtype=np.int16)
    for j in range(m):
        v = X[:, j]
        keep = np.ones(n, dtype=bool) if mask is None else mask[:, j]
        if keep.sum() < 2 * n_bins:
            idx[:, j] = -1
            continue
        e = np.unique(np.quantile(v[keep], np.linspace(0, 1, n_bins + 1)))
        idx[:, j] = np.clip(np.digitize(v, e[1:-1]), 0, max(len(e) - 2, 0))
        idx[~keep, j] = -1
    return idx


def stats_all(idx, y, n_bins=N_BINS):
    """
    Max |decile rate - base rate| for every column.

    The base rate is taken over the same days the column is measured on, so a
    variable covering only part of the record is compared against its own
    period rather than against a different one.
    """
    m = idx.shape[1]
    out = np.empty(m)
    for j in range(m):
        col = idx[:, j]
        keep = col >= 0
        if keep.sum() < 2 * n_bins:
            out[j] = 0.0
            continue
        b = col[keep]
        yy = y[keep]
        base = yy.mean()
        cnt = np.bincount(b, minlength=n_bins).astype(float)
        s = np.bincount(b, weights=yy, minlength=n_bins)
        ok = cnt >= MIN_BIN
        if not ok.any():
            out[j] = 0.0
            continue
        rates = np.where(ok, s / np.maximum(cnt, 1), base)
        out[j] = np.max(np.abs(rates - base))
    return out


def etas_reference(days, cnt):
    """
    Same statistic, but the variable is recent seismicity rather than the sky.
    Gives the scale everything else is measured against.
    """
    n = len(cnt)
    K, alpha, c, p, MC = 0.012, 1.35, 0.02, 1.15, 6.0
    lam = np.zeros(n)
    for j in np.flatnonzero(cnt > 0):
        if j + 1 >= n:
            continue
        dt = np.arange(1, n - j)
        lam[j + 1:] += K * cnt[j] * (dt + c) ** (-p)
    feats = {"etas_rate": lam}
    for w in (1, 7, 30):
        past = np.zeros(n)
        cs = np.concatenate([[0], np.cumsum(cnt)])
        for i in range(n):
            past[i] = cs[i] - cs[max(0, i - w)]
        feats[f"etas_count_{w}d"] = past
    return feats


def main():
    rule("UJI TERPADU: LIMA KELUARGA GAYA LUAR, SATU KERANGKA")

    days = np.load(datafile("m6_days.npy"))
    cnt = np.load(datafile("m6_cnt.npy"))
    y = (cnt > 0).astype(float)

    print(f"""
  katalog   global M>=6.0, {datetime.fromordinal(int(days[0])).date()} .. """
          f"""{datetime.fromordinal(int(days[-1])).date()}
  hari      {len(y):,}   laju dasar {100*y.mean():.2f}%
  null      {N_SHIFT} pergeseran melingkar, pencarian penuh tiap kali
""")
    print("  membangun bank variabel:")
    X, names, fams = build_all(days)
    print(f"\n  total {X.shape[1]} variabel dari {len(set(fams))} keluarga")

    print("\n  cakupan data per variabel (hari isian tidak diuji):")
    mask = print_report(X, names, fams, limit=8)
    idx = bin_indices(X, mask=mask)
    obs = stats_all(idx, y)

    # null distribution, shared across families: one shift, all statistics
    null = np.empty((N_SHIFT, X.shape[1]))
    for i in range(N_SHIFT):
        null[i] = stats_all(idx, np.roll(y, int(RNG.integers(1, len(y)))))
        if (i + 1) % 100 == 0:
            print(f"    null {i+1}/{N_SHIFT}", end="\r", flush=True)
    print(" " * 30, end="\r")

    # ----- ETAS reference on the same statistic -----
    ref = etas_reference(days, cnt)
    Xe = np.column_stack([v for v in ref.values()])
    idxe = bin_indices(Xe)
    obs_e = stats_all(idxe, y)
    null_e = np.empty((N_SHIFT, Xe.shape[1]))
    for i in range(N_SHIFT):
        null_e[i] = stats_all(idxe, np.roll(y, int(RNG.integers(1, len(y)))))

    rule("HASIL PER KELUARGA")
    print(f"""
  'terukur'  modulasi laju terbesar yang ditemukan di keluarga itu
  'ambang'   modulasi yang akan terdeteksi (persentil 95 dari null)
  'p'        sudah terkoreksi untuk jumlah variabel dalam keluarga
""")
    print(f"  {'keluarga':<12} {'n var':>6} {'terukur':>9} {'ambang':>8} "
          f"{'p':>7}  variabel terbaik")
    print("  " + "-" * 78)

    rows = []
    fam_order = [f for f in dict.fromkeys(fams)]
    for fam in fam_order:
        sel = np.flatnonzero(fams == fam)
        if sel.size == 0:
            continue
        best = obs[sel].max()
        nb = null[:, sel].max(axis=1)
        p = float((nb >= best).mean())
        thr = np.percentile(nb, 95)
        top = names[int(sel[np.argmax(obs[sel])])]
        rows.append((fam, sel.size, best, thr, p, top))
        flag = " <-- LOLOS" if p < 0.05 else ""
        print(f"  {fam:<12} {sel.size:>6} {100*best:>8.2f}% {100*thr:>7.2f}% "
              f"{p:>7.3f}  {top[:26]}{flag}")

    # ETAS row
    best_e = obs_e.max()
    nb_e = null_e.max(axis=1)
    p_e = float((nb_e >= best_e).mean())
    thr_e = np.percentile(nb_e, 95)
    print("  " + "-" * 78)
    print(f"  {'ETAS (acuan)':<12} {Xe.shape[1]:>6} {100*best_e:>8.2f}% "
          f"{100*thr_e:>7.2f}% {p_e:>7.3f}  "
          f"{list(ref)[int(np.argmax(obs_e))][:26]}"
          f"{' <-- LOLOS' if p_e < 0.05 else ''}")

    rule("KOREKSI GABUNGAN: seluruh keluarga sekaligus")
    sky = np.arange(len(fams))
    best_all = obs[sky].max()
    nb_all = null[:, sky].max(axis=1)
    p_all = float((nb_all >= best_all).mean())
    print(f"""
  Menggabungkan SELURUH {len(fam_order)} keluarga ({sky.size} variabel) dan
  memberi null kebebasan mencari di seluruhnya:

    terbaik data asli : {100*best_all:.2f}%  ({names[int(sky[np.argmax(obs[sky])])]})
    terbaik pada null : rata-rata {100*nb_all.mean():.2f}%, maks {100*nb_all.max():.2f}%
    p terkoreksi      : {p_all:.3f}
""")

    rule("TABEL BATAS ATAS  (produk utama)")
    print(f"""
  Modulasi laju gempa yang masih mungkin bersembunyi di tiap keluarga,
  relatif terhadap laju dasar {100*y.mean():.1f}%:
""")
    print(f"  {'keluarga':<14} {'batas modulasi':>16} {'setara relatif':>16}")
    print("  " + "-" * 50)
    base = y.mean()
    for fam, n, best, thr, p, top in rows:
        print(f"  {fam:<14} {100*thr:>15.2f}% {100*thr/base:>15.1f}%")
    print(f"  {'ETAS (acuan)':<14} {100*best_e:>15.2f}% {100*best_e/base:>15.1f}%"
          f"   <- terukur, bukan batas")

    rule("CATATAN")
    print("""
  HYDRO di sini memakai harmonik tahunan dan semi-tahunan sebagai proksi beban
  air musiman -- tanda tangan yang dideteksi studi Himalaya dan California.
  Slot GRACE tersedia di forcing_bank.load_grace() dan belum diisi; mengisinya
  akan mengubah HYDRO dari proksi menjadi beban air terukur, dan itu satu-satunya
  keluarga yang masih punya peluang hasil positif.

  Semua variabel diuji pada satu situs acuan untuk keluarga TIDAL dan komponen
  pasang surut PLANETARY (palung Jawa). Untuk target global ini itu pilihan
  konservatif: tegangan pasang surut bersifat lokal, sehingga satu situs
  meremehkan efek yang mungkin ada di tempat lain. Versi per-event ada di
  src/tidal/tidal_analysis.py dan tetap null.
""")


if __name__ == "__main__":
    main()
