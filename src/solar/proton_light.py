"""
The Altaibek et al. (2024) test with real proton density -- light enough to run
on a laptop.

proton_test.py fed the model all 48 raw hourly values of every solar-wind
variable, which makes a 254,000 x 265 design matrix (about 540 MB) and twelve
gradient-boosting fits on top of it. That was my mistake: for a tree model the
raw sequence carries essentially nothing beyond its summary statistics, so the
cost bought no accuracy.

This version keeps the design and drops the waste:

  * 3-hourly grid instead of hourly (the 48-hour window still spans 16 steps)
  * seven summary statistics per variable instead of 48 raw values
  * rolling statistics computed with sliding windows rather than a Python loop
  * the paper's own configuration first, extras second

Same catalogue, same declustering (150 km / 6 months), same 48-hour label, same
class weighting -- and both splits, since the paper does not say which it used.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile, ROOT, SSGEOS  # noqa: F401

import csv
import io
import numpy as np
from datetime import datetime, timedelta
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

STEP_H = 3            # 3-hourly grid
WINDOW = 16           # 16 x 3 h = 48 h of history
HORIZON = 16          # earthquake within the next 48 h
RNG = np.random.default_rng(20260816)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_omni_3h():
    """OMNI hourly -> 3-hourly, keeping the maximum in each block."""
    keys = ["omni_sw_n", "omni_sw_v", "omni_sw_ram", "omni_dst", "omni_ae"]
    times, vals = [], {k: [] for k in keys}
    for row in csv.DictReader(io.StringIO((datafile("omni_hourly.csv")).read_text(encoding="utf-8"))):
        try:
            t = datetime.strptime(row["Time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        times.append(t)
        for k in keys:
            try:
                v = float(row.get(k, ""))
            except (TypeError, ValueError):
                v = np.nan
            vals[k].append(np.nan if abs(v) > 9e4 else v)
    times = np.array(times)
    order = np.argsort(times)
    times = times[order]

    t0 = times[0].replace(hour=0)
    slot = np.array([int((t - t0).total_seconds() // (STEP_H * 3600)) for t in times])
    n = slot.max() + 1
    grid = np.array([t0 + timedelta(hours=STEP_H * i) for i in range(n)])
    out = {}
    for k in keys:
        v = np.array(vals[k], dtype=float)[order]
        # np.maximum propagates NaN, so seed with -inf and convert back
        agg = np.full(n, -np.inf)
        good = np.isfinite(v)
        np.maximum.at(agg, slot[good], v[good])
        agg[np.isneginf(agg)] = np.nan
        out[k] = _ffill(agg)
    return grid, out


def _ffill(v):
    v = v.copy()
    last = np.nan
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    med = np.nanmedian(v)
    v[np.isnan(v)] = 0.0 if np.isnan(med) else med
    return v


def window_features(v, w=WINDOW):
    """Seven summaries per variable, computed on all windows at once."""
    sw = sliding_window_view(v, w)[:-1]          # window ends before the label
    return np.column_stack([
        sw.max(axis=1), sw.mean(axis=1), sw.std(axis=1),
        sw[:, -1], sw[:, -1] - sw[:, 0],
        np.percentile(sw, 90, axis=1), sw.min(axis=1)])


def load_quakes(path, minmag, box=None, maxdepth=60.0):
    out = []
    for r in csv.DictReader(io.StringIO((DATA / path).read_text(encoding="utf-8"))):
        if r.get("type", "earthquake") != "earthquake":
            continue
        try:
            m = float(r["mag"]); la = float(r["latitude"]); lo = float(r["longitude"])
            dep = float(r["depth"])
            dt = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if m < minmag or dep > maxdepth:
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


def labels(grid, quakes):
    n = len(grid)
    lab = np.zeros(n, dtype=int)
    for dt, *_ in quakes:
        j = int((dt - grid[0]).total_seconds() // (STEP_H * 3600))
        if 0 <= j < n:
            lab[max(0, j - HORIZON):j] = 1
    return lab


def report(tag, y, pred, prob):
    acc = accuracy_score(y, pred)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    auc = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
    print(f"    {tag:<22} akurasi={acc:.4f} presisi={p[0]:.4f} "
          f"recall={r[0]:.4f} F1={f[0]:.4f} AUC={auc:.4f}")
    return auc


def run(name, X, y):
    base = y.mean()
    print(f"\n  {name}")
    print(f"    baris={len(y):,} fitur={X.shape[1]}  laju dasar={100*base:.2f}%  "
          f"baseline 'selalu tidak'={100*(1-base):.2f}%")
    cfg = dict(max_iter=200, learning_rate=0.07, max_leaf_nodes=31,
               min_samples_leaf=60, random_state=0, class_weight="balanced")
    cut = int(0.8 * len(y))
    m = HistGradientBoostingClassifier(**cfg).fit(X[:cut], y[:cut])
    a_t = report("TEMPORAL (sah)", y[cut:], m.predict(X[cut:]),
                 m.predict_proba(X[cut:])[:, 1])
    perm = RNG.permutation(len(y))
    tr, te = perm[:cut], perm[cut:]
    m2 = HistGradientBoostingClassifier(**cfg).fit(X[tr], y[tr])
    a_r = report("ACAK (bocor)", y[te], m2.predict(X[te]),
                 m2.predict_proba(X[te])[:, 1])
    return base, a_t, a_r


def main():
    rule("UJI PROTON — VERSI RINGAN")
    grid, sw = load_omni_3h()
    print(f"""
  OMNI 3-jaman  {len(grid):,} langkah, {grid[0].date()} .. {grid[-1].date()}
  proton        median {np.median(sw['omni_sw_n']):.1f} cm^-3, """
          f"""maks {sw['omni_sw_n'].max():.1f}
""")

    feats_proton = window_features(sw["omni_sw_n"])
    feats_all = np.column_stack([window_features(sw[k]) for k in
                                 ["omni_sw_n", "omni_sw_v", "omni_sw_ram",
                                  "omni_dst", "omni_ae"]])
    print(f"  matriks: proton-saja {feats_proton.shape}, semua {feats_all.shape}"
          f"  (sebelumnya 254.261 x 265)")

    IND = (-13, 8, 93, 143)
    targets = [("GLOBAL M>=6.0 (cakupan paper)", "global_m6.csv", 6.0, None),
               ("INDONESIA M>=6.0", "indonesia_cat.csv", 6.0, IND)]

    for tag, X in [("A. HANYA kepadatan proton (persis paper)", feats_proton),
                   ("B. + kecepatan, ram, Dst, AE", feats_all)]:
        rule(tag)
        rows = []
        for name, path, mm, box in targets:
            q = decluster(load_quakes(path, mm, box))
            lab = labels(grid, q)[WINDOW:]
            n = min(len(lab), len(X))
            rows.append((name, *run(f"{name}  (n gempa={len(q):,})", X[:n], lab[:n])))
        print(f"\n  {'target':<32} {'laju dasar':>11} {'AUC temporal':>13} {'AUC acak':>10}")
        print("  " + "-" * 70)
        for nm, b, at, ar in rows:
            print(f"  {nm:<32} {100*b:>10.2f}% {at:>13.4f} {ar:>10.4f}")

    rule("PEMBANDING")
    print("""
  Altaibek dkk. (2024): akurasi 0.8447, presisi 0.6807, recall 0.8368, F1 0.7507
  Laju dasar tersirat dari ketiga angka itu: 27.9%
  Baseline 'selalu tebak tidak ada gempa' pada laju itu: 72.1%
""")


if __name__ == "__main__":
    main()
