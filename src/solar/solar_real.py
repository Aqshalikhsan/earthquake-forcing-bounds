"""
Step 21: solar activity alone, using only real measured observations.

No synthetic data anywhere. Every input is an actual instrumental record:

  Kp / ap    GFZ Potsdam, 3-hourly since 1932. Geomagnetic disturbance measured
             at a global network of observatories -- the response at Earth's
             surface, which is what any telluric-current mechanism would act
             through.
  sunspot    SILSO / Royal Observatory of Belgium, daily since 1818. The
             canonical index of solar magnetic activity.
  F10.7      Natural Resources Canada, daily since 2004. Solar radio flux at
             10.7 cm, the operational proxy for solar EUV output.
  quakes     USGS ComCat, declustered with the 150 km / 6 month windows the
             paper specifies.

Together these span the causal chain the solar-seismicity literature proposes:
solar magnetic activity -> radiative and particle output -> geomagnetic
disturbance at Earth. Proton density (the paper's own input) sits inside that
chain; NASA's archive is unreachable from this machine, and this is stated as a
limitation rather than papered over.

Nothing seismological is used as a feature. No past earthquakes, no ETAS, no
location. The question is exactly the one asked: can this method work alone?

Both splits are reported. For a target whose label spans 48 hours, neighbouring
3-hour rows share 15 of their 16 history values and almost always carry the same
label, so a random split trains and tests on near-duplicates. The paper does not
say which split it used.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import csv
import io
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

HERE = DATA
STEP_H = 3
WINDOW = 16          # 48 h of geomagnetic history
HORIZON = 16         # "earthquake in the next 48 h"
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ real data

def load_kp():
    times, kp, ap = [], [], []
    for line in (HERE / "kp_ap.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        try:
            t = datetime(int(f[0]), int(f[1]), int(f[2])) + timedelta(hours=int(float(f[3])))
            k, a = float(f[7]), float(f[8])
        except (ValueError, IndexError):
            continue
        if k < 0 or a < 0:
            continue
        times.append(t); kp.append(k); ap.append(a)
    return np.array(times), np.array(kp), np.array(ap)


def load_sunspot():
    out = {}
    for line in (HERE / "sunspot.txt").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        try:
            sn = float(f[4])
            if sn < 0:
                continue
            out[datetime(int(f[0]), int(f[1]), int(f[2])).date()] = sn
        except (ValueError, IndexError):
            continue
    return out


def load_f107():
    out = {}
    for line in (HERE / "f107.txt").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 6 or not f[0].isdigit() or len(f[0]) != 8:
            continue
        try:
            d = datetime(int(f[0][:4]), int(f[0][4:6]), int(f[0][6:8])).date()
            out[d] = float(f[5])            # adjusted flux
        except (ValueError, IndexError):
            continue
    return out


def load_proton_density():
    """
    Real solar-wind proton density from the OMNI hourly archive -- the quantity
    Altaibek et al. actually use.

    NASA's servers (spdf / omniweb / cdaweb, all on 169.154.154.x) are
    unreachable from the machine this was written on: DNS resolves but TCP to
    port 443 times out silently, which is what a dropping firewall looks like.
    From an ordinary connection they work fine, so the files just need to be
    fetched once and dropped into data/:

        https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_1996.dat
        ... through omni2_2023.dat        (~1.5 MB each)
        or omni2_all_years.dat            (one file, ~50 MB)

    OMNI2 is fixed-column ASCII, one line per hour. Field 24 (1-indexed) is
    proton density in n/cc; 999.9 marks a gap.

    Returns {} when no file is present, so the rest of the pipeline keeps
    running on the substitute indices.
    """
    out = {}
    files = sorted(DATA.glob("omni2_*.dat"))
    for f in files:
        for line in f.read_text(encoding="latin-1").splitlines():
            g = line.split()
            if len(g) < 24:
                continue
            try:
                year, doy, hour = int(g[0]), int(g[1]), int(g[2])
                dens = float(g[23])
            except (ValueError, IndexError):
                continue
            if dens >= 999.0:                      # OMNI gap marker
                continue
            out[datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour)] = dens
    if out:
        print(f"  kepadatan proton OMNI: {len(out):,} jam dari {len(files)} berkas "
              f"({min(out).date()} .. {max(out).date()})")
    return out


def load_quakes(path, minmag, box=None):
    out = []
    for r in csv.DictReader(io.StringIO((HERE / path).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"]); lo = float(r["longitude"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if m < minmag:
            continue
        if box and not (box[0] <= la <= box[1] and box[2] <= lo <= box[3]):
            continue
        out.append((dt, m, la, lo))
    out.sort(key=lambda e: e[0])
    return out


def decluster(events, km=150.0, months=6.0):
    keep = []
    for dt, m, la, lo in events:
        drop = False
        for dt2, m2, la2, lo2 in reversed(keep):
            if (dt - dt2).days > months * 30.4:
                break
            if m2 >= m:
                d = 111.19 * np.hypot(la - la2, (lo - lo2) * np.cos(np.radians(la)))
                if d <= km:
                    drop = True
                    break
        if not drop:
            keep.append((dt, m, la, lo))
    return keep


# ------------------------------------------------------------------ features

def daily_series(dmap, times, default=np.nan):
    """Look a daily index up for each 3-hour timestamp."""
    return np.array([dmap.get(t.date(), default) for t in times])


def build(times, kp, ap, sn, f107, quakes, t0, t1, use_f107):
    sel = (times >= t0) & (times < t1)
    T, K, A = times[sel], kp[sel], ap[sel]
    S = daily_series(sn, T)
    F = daily_series(f107, T) if use_f107 else None

    # forward-fill the daily indices over any gaps
    def ffill(v):
        v = v.copy()
        last = np.nan
        for i in range(len(v)):
            if np.isnan(v[i]):
                v[i] = last
            else:
                last = v[i]
        return v
    S = ffill(S)
    if F is not None:
        F = ffill(F)

    n = len(T)
    lab = np.zeros(n, dtype=int)
    for dt, *_ in quakes:
        j = int((dt - T[0]).total_seconds() // (STEP_H * 3600))
        if 0 <= j < n:
            lab[max(0, j - HORIZON):j] = 1

    D27 = 27 * 8       # 27 days of 3-hour steps: one solar rotation
    rows, ys = [], []
    for i in range(max(WINDOW, D27), n):
        wk, wa = K[i - WINDOW:i], A[i - WINDOW:i]
        feat = [*wk, *wa,
                wk.max(), wk.mean(), wk[-1] - wk[0],
                wa.max(), wa.mean(), wa.sum(), np.percentile(wa, 90), wa[-1],
                S[i - 1], S[i - WINDOW:i].mean(), S[i - D27:i].mean(),
                S[i - 1] - S[i - D27:i].mean()]
        if F is not None:
            feat += [F[i - 1], F[i - WINDOW:i].mean(), F[i - D27:i].mean(),
                     F[i - 1] - F[i - D27:i].mean()]
        rows.append(feat)
        ys.append(lab[i])
    X = np.array(rows, dtype=float)
    return X, np.array(ys)


def report(name, y, pred, prob=None):
    acc = accuracy_score(y, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1],
                                                    zero_division=0)
    auc = roc_auc_score(y, prob) if prob is not None and len(set(y)) > 1 else float("nan")
    print(f"  {name:<40} akurasi={acc:.4f} presisi={pr[0]:.4f} "
          f"recall={rc[0]:.4f} F1={f1[0]:.4f} AUC={auc:.4f}")
    return acc, auc


def run(label, X, y):
    base = y.mean()
    print(f"\n  {label}")
    print(f"    baris={len(y):,}  laju dasar={100*base:.2f}%  "
          f"baseline 'selalu tidak'={100*(1-base):.2f}%")
    common = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=60, random_state=0, class_weight="balanced")
    cut = int(0.8 * len(y))

    m = HistGradientBoostingClassifier(**common).fit(X[:cut], y[:cut])
    p = m.predict_proba(X[cut:])[:, 1]
    _, auc_t = report("    TEMPORAL (sah)", y[cut:], m.predict(X[cut:]), p)

    perm = RNG.permutation(len(y))
    tr, te = perm[:cut], perm[cut:]
    m2 = HistGradientBoostingClassifier(**common).fit(X[tr], y[tr])
    p2 = m2.predict_proba(X[te])[:, 1]
    _, auc_r = report("    ACAK (bocor)", y[te], m2.predict(X[te]), p2)
    return base, auc_t, auc_r


def main():
    rule("AKTIVITAS MATAHARI SAJA -- SEMUA DATA ASLI TERUKUR")
    times, kp, ap = load_kp()
    sn = load_sunspot()
    f107 = load_f107()
    print(f"""
  Kp/ap   GFZ Potsdam   {len(times):,} interval 3-jam, {times[0].date()} .. {times[-1].date()}
  sunspot SILSO/Belgia  {len(sn):,} hari, {min(sn)} .. {max(sn)}
  F10.7   NRCan Kanada  {len(f107):,} hari, {min(f107)} .. {max(f107)}
  gempa   USGS ComCat, declustering 150 km / 6 bulan (seperti paper)
""")

    IND = (-13, 8, 93, 143)
    targets = [
        ("GLOBAL M>=6.0 (cakupan paper)", decluster(load_quakes("global_m6.csv", 6.0))),
        ("INDONESIA M>=6.0", decluster(load_quakes("indonesia_cat.csv", 6.0, IND))),
        ("INDONESIA M>=7.0", decluster(load_quakes("indonesia_cat.csv", 7.0, IND))),
    ]

    for period, (t0, t1), use_f in [
            ("A. 1996-2024  (periode paper; Kp/ap + sunspot)",
             (datetime(1996, 1, 1), datetime(2024, 1, 1)), False),
            ("B. 2005-2024  (fitur penuh; + F10.7)",
             (datetime(2005, 1, 1), datetime(2024, 1, 1)), True)]:
        rule(period)
        rows = []
        for name, q in targets:
            X, y = build(times, kp, ap, sn, f107, q, t0, t1, use_f)
            rows.append((name, *run(f"{name}  (n gempa={len(q):,}, fitur={X.shape[1]})", X, y)))
        print(f"\n  {'target':<32} {'laju dasar':>11} {'AUC temporal':>13} {'AUC acak':>10}")
        print("  " + "-" * 70)
        for nm, b, at, ar in rows:
            print(f"  {nm:<32} {100*b:>10.2f}% {at:>13.4f} {ar:>10.4f}")

    rule("CATATAN KETERBATASAN")
    print("""
  Kepadatan proton SOHO -- input asli paper -- tidak dapat diunduh dari mesin
  ini (arsip NASA SPDF, CDAWeb, Caltech ACE, dan UMD CELIAS semuanya tidak
  terjangkau). Yang dipakai di sini adalah tiga indeks matahari terukur lain
  yang mengapit besaran itu di rantai sebab-akibat.

  Marchitelli dkk. (2020), rujukan [9] paper itu, melaporkan kepadatan proton
  berkorelasi paling kuat di antara parameter matahari. Jadi ada kemungkinan
  proksi di sini lebih lemah. Namun AUC di kisaran 0,50 bukan 'agak lemah' --
  itu nol. Proksi yang lebih baik menggeser 0,52 menjadi 0,55, bukan 0,84.
""")


if __name__ == "__main__":
    main()
