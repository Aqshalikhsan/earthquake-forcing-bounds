"""
Step 22: the Altaibek et al. (2024) test, with their own variable.

Earlier steps used Kp/ap, sunspot number and F10.7 as substitutes because NASA's
archives were unreachable. That gap is now closed: the OMNI hourly merged solar
wind dataset was retrieved through the AMDA/IRAP HAPI server in France
(amda.irap.omp.eu), which mirrors it. NASA's own spdf/omniweb/cdaweb hosts are
unreachable from both this machine and the user's -- DNS resolves, TCP to 443
times out, ping fails -- so the mirror is the route that works.

    omni_sw_n    solar wind proton density, cm^-3, hourly, 1996-2024
                 -- the paper's input variable
    omni_sw_v    bulk speed, km/s
    omni_sw_ram  ram pressure, nPa
    omni_dst     Dst index, nT
    omni_ae      AE index, nT

Setup follows the paper: hourly cadence, a 48-hour history window, binary label
"is there an earthquake in the next 48 hours", earthquakes declustered with
150 km / 6 month windows, class weighting for the imbalance.

One thing is added that the paper does not state: the split. Section 2.1 says
only "divided into test and training sets at a ratio of 80% to 20%". For a
label that spans 48 hours, neighbouring rows share 47 of their 48 hours and
nearly always carry the same value, so a random split trains and tests on
near-duplicates. Both are reported side by side, and the gap between them is
the finding.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime, timedelta

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

WINDOW = 48          # hours of solar-wind history, as in the paper
HORIZON = 48         # "earthquake in the next 48 h"
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_omni():
    """Real OMNI hourly solar wind. Empty fields are gaps."""
    times, cols = [], {k: [] for k in
                       ("n", "v", "ram", "dst", "ae")}
    txt = (datafile("omni_hourly.csv")).read_text(encoding="utf-8")
    for row in csv.DictReader(io.StringIO(txt)):
        try:
            t = datetime.strptime(row["Time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        def g(k):
            v = row.get(k, "")
            try:
                x = float(v)
            except (TypeError, ValueError):
                return np.nan
            return np.nan if abs(x) > 9e4 else x
        times.append(t)
        cols["n"].append(g("omni_sw_n"))
        cols["v"].append(g("omni_sw_v"))
        cols["ram"].append(g("omni_sw_ram"))
        cols["dst"].append(g("omni_dst"))
        cols["ae"].append(g("omni_ae"))
    times = np.array(times)
    out = {k: np.array(v, dtype=float) for k, v in cols.items()}
    order = np.argsort(times)
    return times[order], {k: v[order] for k, v in out.items()}


def ffill(v):
    v = v.copy()
    last = np.nan
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    med = np.nanmedian(v)
    v[np.isnan(v)] = med
    return v


def load_quakes(path, minmag, box=None):
    out = []
    for r in csv.DictReader(io.StringIO((DATA / path).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"]); lo = float(r["longitude"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
            dep = float(r["depth"])
        except (ValueError, TypeError, KeyError):
            continue
        if m < minmag or dep > 60:
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


def build(times, sw, quakes, proton_only):
    n = len(times)
    lab = np.zeros(n, dtype=int)
    for dt, *_ in quakes:
        j = int((dt - times[0]).total_seconds() // 3600)
        if 0 <= j < n:
            lab[max(0, j - HORIZON):j] = 1

    keys = ["n"] if proton_only else ["n", "v", "ram", "dst", "ae"]
    S = {k: ffill(sw[k]) for k in keys}

    rows, ys = [], []
    for i in range(WINDOW, n):
        f = []
        for k in keys:
            w = S[k][i - WINDOW:i]
            f.extend(w)                                   # the raw 48-h sequence
            f.extend([w.max(), w.mean(), w.std(), w[-1], w[-1] - w[0]])
        rows.append(f)
        ys.append(lab[i])
    return np.array(rows, dtype=float), np.array(ys)


def report(name, y, pred, prob):
    acc = accuracy_score(y, pred)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1],
                                                 zero_division=0)
    auc = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
    print(f"    {name:<34} akurasi={acc:.4f} presisi={p[0]:.4f} "
          f"recall={r[0]:.4f} F1={f[0]:.4f} AUC={auc:.4f}")
    return acc, auc


def run(label, X, y):
    base = y.mean()
    print(f"\n  {label}")
    print(f"    baris={len(y):,} fitur={X.shape[1]}  laju dasar={100*base:.2f}%  "
          f"baseline 'selalu tidak'={100*(1-base):.2f}%")
    cfg = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
               min_samples_leaf=60, random_state=0, class_weight="balanced")
    cut = int(0.8 * len(y))

    m = HistGradientBoostingClassifier(**cfg).fit(X[:cut], y[:cut])
    _, auc_t = report("TEMPORAL (sah)", y[cut:], m.predict(X[cut:]),
                      m.predict_proba(X[cut:])[:, 1])

    perm = RNG.permutation(len(y))
    tr, te = perm[:cut], perm[cut:]
    m2 = HistGradientBoostingClassifier(**cfg).fit(X[tr], y[tr])
    _, auc_r = report("ACAK (bocor)", y[te], m2.predict(X[te]),
                      m2.predict_proba(X[te])[:, 1])
    return base, auc_t, auc_r


def main():
    rule("UJI DENGAN KEPADATAN PROTON ASLI (variabel paper)")
    times, sw = load_omni()
    good = ~np.isnan(sw["n"])
    print(f"""
  sumber   OMNI hourly via AMDA/IRAP HAPI (mirror Prancis)
  rentang  {times[0].date()} .. {times[-1].date()}
  baris    {len(times):,} jam, kepadatan proton terisi {100*good.mean():.1f}%
  proton   min {np.nanmin(sw['n']):.1f}  median {np.nanmedian(sw['n']):.1f}  """
          f"""maks {np.nanmax(sw['n']):.1f} cm^-3
""")

    IND = (-13, 8, 93, 143)
    targets = [
        ("GLOBAL M>=6.0 (cakupan paper)", decluster(load_quakes("global_m6.csv", 6.0))),
        ("INDONESIA M>=6.0", decluster(load_quakes("indonesia_cat.csv", 6.0, IND))),
        ("INDONESIA M>=7.0", decluster(load_quakes("indonesia_cat.csv", 7.0, IND))),
    ]

    for tag, proton_only in [("A. HANYA kepadatan proton (persis paper)", True),
                             ("B. + kecepatan, tekanan ram, Dst, AE", False)]:
        rule(tag)
        rows = []
        for name, q in targets:
            X, y = build(times, sw, q, proton_only)
            rows.append((name, *run(f"{name}  (n gempa={len(q):,})", X, y)))
        print(f"\n  {'target':<32} {'laju dasar':>11} {'AUC temporal':>13} {'AUC acak':>10}")
        print("  " + "-" * 70)
        for nm, b, at, ar in rows:
            print(f"  {nm:<32} {100*b:>10.2f}% {at:>13.4f} {ar:>10.4f}")

    rule("PEMBANDING: yang dilaporkan paper")
    print("""
  Altaibek dkk. (2024): akurasi 0.8447, presisi 0.6807, recall 0.8368, F1 0.7507
  Laju dasar tersirat dari ketiga angka itu: 27.9%
  Baseline 'selalu tebak tidak ada gempa' pada laju itu: 72.1%
""")


if __name__ == "__main__":
    main()
