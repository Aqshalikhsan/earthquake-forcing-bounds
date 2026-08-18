"""
Step 20: can solar activity alone forecast earthquakes? No ETAS, no seismology.

This is the Altaibek et al. (2024) design, rebuilt with the one thing their paper
does not specify: a temporal split. Section 2.1 says only "the data was divided
into test and training sets at a ratio of 80% to 20%". For a time series whose
label is autocorrelated across 48 hours, that single unstated choice decides
whether the result means anything, so both splits are run here side by side.

Input: geomagnetic activity only. The paper uses SOHO proton density; NASA's
archive was unreachable from here, so this uses the GFZ Potsdam Kp/ap index,
3-hourly since 1932, open licence. That is arguably a closer match to the
mechanism the paper invokes -- telluric currents in the crust -- because Kp
measures the geomagnetic disturbance that actually reaches Earth's surface,
rather than the solar wind property upstream of it.

Nothing seismological enters the features. No past earthquakes, no ETAS, no
location. Exactly the question asked: can this method work on its own?

Three targets, because the base rate is what decides whether a forecast is
usable at all:
    global M>=6.0     -- the paper's scope
    Indonesia M>=6.0  -- regional
    Indonesia M>=7.0  -- rare enough that an alarm would mean something
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
STEP_H = 3                     # Kp cadence
WINDOW = 16                    # 16 x 3h = 48 h of history
HORIZON = 16                   # "earthquake in the next 48 h"
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_kp():
    """GFZ Kp/ap, one row per 3-hour interval."""
    times, kp, ap = [], [], []
    for line in (HERE / "kp_ap.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        try:
            y, mo, d = int(f[0]), int(f[1]), int(f[2])
            hh = int(float(f[3]))
            k, a = float(f[7]), float(f[8])
        except (ValueError, IndexError):
            continue
        if k < 0 or a < 0:
            continue
        times.append(datetime(y, mo, d) + timedelta(hours=hh))
        kp.append(k)
        ap.append(a)
    return np.array(times), np.array(kp), np.array(ap)


def load_quakes(path, minmag, box=None):
    out = []
    for r in csv.DictReader(io.StringIO((HERE / path).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"])
            la, lo = float(r["latitude"]), float(r["longitude"])
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
    """Marchitelli-style window declustering, as the paper describes."""
    if not events:
        return events
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


def build(times, kp, ap, quakes, t0, t1):
    """Feature windows of geomagnetic history, and the forward-looking label."""
    sel = (times >= t0) & (times < t1)
    T, K, A = times[sel], kp[sel], ap[sel]
    n = len(T)

    lab = np.zeros(n, dtype=int)
    idx = {}
    for i, t in enumerate(T):
        idx[t.replace(minute=0, second=0)] = i
    for dt, *_ in quakes:
        j = int((dt - T[0]).total_seconds() // (STEP_H * 3600))
        if 0 <= j < n:
            lab[max(0, j - HORIZON):j] = 1

    rows, ys = [], []
    for i in range(WINDOW, n):
        w_k = K[i - WINDOW:i]
        w_a = A[i - WINDOW:i]
        rows.append(np.concatenate([
            w_k, w_a,
            [w_k.max(), w_k.mean(), w_k[-1] - w_k[0],
             w_a.max(), w_a.mean(), w_a.sum(),
             np.percentile(w_a, 90), w_a[-1]]]))
        ys.append(lab[i])
    return np.array(rows), np.array(ys), T[WINDOW:]


def report(name, y, p, prob=None):
    acc = accuracy_score(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[1],
                                                    zero_division=0)
    auc = roc_auc_score(y, prob) if prob is not None and len(set(y)) > 1 else np.nan
    print(f"  {name:<38} akurasi={acc:.4f} presisi={pr[0]:.4f} "
          f"recall={rc[0]:.4f} F1={f1[0]:.4f} AUC={auc:.4f}")
    return acc, auc


def run_target(label, quakes, times, kp, ap, t0, t1):
    X, y, T = build(times, kp, ap, quakes, t0, t1)
    base = y.mean()
    rule(f"TARGET: {label}")
    print(f"""
  interval 3-jam   : {len(y):,}
  gempa (declustered): {len(quakes):,}
  laju dasar kelas 'ada gempa dalam 48 jam' : {100*base:.2f}%
  baseline 'selalu tebak TIDAK ada'         : {100*(1-base):.2f}%
""")
    common = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=60, random_state=0, class_weight="balanced")

    cut = int(0.8 * len(y))
    m = HistGradientBoostingClassifier(**common).fit(X[:cut], y[:cut])
    pr = m.predict_proba(X[cut:])[:, 1]
    a_t, auc_t = report("SPLIT TEMPORAL (sah)", y[cut:], m.predict(X[cut:]), pr)

    perm = RNG.permutation(len(y))
    tr, te = perm[:cut], perm[cut:]
    m2 = HistGradientBoostingClassifier(**common).fit(X[tr], y[tr])
    pr2 = m2.predict_proba(X[te])[:, 1]
    a_r, auc_r = report("SPLIT ACAK (tidak sah utk deret waktu)", y[te],
                        m2.predict(X[te]), pr2)

    report("selalu tebak 'TIDAK ada gempa'", y[cut:],
           np.zeros(len(y) - cut, dtype=int))
    return dict(label=label, base=base, acc_t=a_t, auc_t=auc_t,
                acc_r=a_r, auc_r=auc_r)


def main():
    rule("BISAKAH AKTIVITAS MATAHARI SAJA? (tanpa ETAS, tanpa seismologi)")
    times, kp, ap = load_kp()
    print(f"\n  Kp/ap GFZ: {len(times):,} interval, "
          f"{times[0].date()} .. {times[-1].date()}")

    t0, t1 = datetime(1996, 1, 1), datetime(2024, 1, 1)
    IND = (-13, 8, 93, 143)

    results = []
    g = decluster(load_quakes("global_m6.csv", 6.0))
    results.append(run_target("GLOBAL M>=6.0  (cakupan paper)", g, times, kp, ap, t0, t1))

    i6 = decluster(load_quakes("indonesia_cat.csv", 6.0, IND))
    results.append(run_target("INDONESIA M>=6.0", i6, times, kp, ap, t0, t1))

    i7 = decluster(load_quakes("indonesia_cat.csv", 7.0, IND))
    results.append(run_target("INDONESIA M>=7.0", i7, times, kp, ap, t0, t1))

    rule("RINGKASAN")
    print(f"\n  {'target':<26} {'laju dasar':>11} {'AUC temporal':>13} "
          f"{'AUC acak':>10}")
    print("  " + "-" * 64)
    for r in results:
        print(f"  {r['label'][:26]:<26} {100*r['base']:>10.2f}% "
              f"{r['auc_t']:>13.4f} {r['auc_r']:>10.4f}")
    print(f"""
  AUC 0,50 = lempar koin.

  Kolom terakhir menunjukkan mengapa metode split itu menentukan: split acak
  menaruh interval bertetangga -- yang berbagi 15 dari 16 jam riwayat dan
  hampir selalu berlabel sama -- di data latih DAN data uji sekaligus.

  Paper Altaibek dkk. tidak menyebutkan yang mana yang mereka pakai.
""")


if __name__ == "__main__":
    main()
