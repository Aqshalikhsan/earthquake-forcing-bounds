"""
Step 8: what earthquake forecasting can actually do, measured against what it
cannot.

Everything up to here concerned a claim that does not work. This asks the
constructive question: given that deterministic prediction is out of reach, how
much forecasting skill genuinely exists, where does it come from, and what is it
good for? The answer is not the Moon. It is the earthquakes themselves.

Four models compete on identical data, scored as a point process the way
operational forecasts are scored (CSEP: information gain per earthquake):

  POISSON   constant rate in time, smoothed background in space. The honest
            null -- earthquakes happen where they have happened before.
  LUNAR     the same background, modulated in time by lunar phase. Fitted on
            the TEST data with amplitude and phase both free, which is more
            help than any real forecast would ever get.
  ETAS-T    temporal Epidemic-Type Aftershock Sequence: every earthquake raises
            the regional rate, decaying by Omori's law.
  ETAS-ST   spatio-temporal ETAS -- the operational form. The rate rises where
            the earthquake was, not across the whole country.

Scoring is pseudo-prospective: at every instant the models see only what had
already happened.

Likelihood for a point process on [T0,T1] x region:

    logL = sum_j log lambda(t_j, x_j, y_j)  -  integral of lambda

The spatial triggering kernel integrates to one, so the integral term reduces to
the temporal one -- which makes the spatial model no harder to score than the
temporal one, and is why this fits in a single file.
"""

from __future__ import annotations


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, ROOT, SSGEOS  # noqa: F401
import csv
import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ephem_vec import jd_from_datetime_array, bodies_equatorial

HERE = DATA
CATALOG = HERE / "indonesia_cat.csv"

MC = 5.0
TRAIN_END = datetime(2016, 1, 1)
T_CUT = 500.0            # days of triggering history retained
KM_PER_DEG = 111.19


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- data

def load_catalog():
    rows = []
    for rec in csv.DictReader(io.StringIO(CATALOG.read_text(encoding="utf-8"))):
        try:
            m = float(rec["mag"])
            dt = datetime.strptime(rec["time"][:19], "%Y-%m-%dT%H:%M:%S")
            la, lo = float(rec["latitude"]), float(rec["longitude"])
        except (ValueError, TypeError, KeyError):
            continue
        if m < MC or rec.get("type", "earthquake") != "earthquake":
            continue
        rows.append((dt, m, la, lo))
    rows.sort(key=lambda r: r[0])
    return rows


def xy_km(lat, lon, lat0):
    """Local equal-ish-area projection, kilometres."""
    x = (lon - 0.0) * KM_PER_DEG * np.cos(np.radians(lat0))
    y = lat * KM_PER_DEG
    return x, y


# ---------------------------------------------------------------- background

def background_density(x_ev, y_ev, x_q, y_q, n_neighbour=8, h_min=15.0):
    """
    Adaptive-bandwidth Gaussian kernel estimate of the background density,
    normalised to integrate to 1 over the plane. Bandwidth at each training
    point is its distance to the n-th nearest neighbour, so active zones get
    fine resolution and quiet ones get broad smoothing.
    """
    n = x_ev.size
    h = np.empty(n)
    block = 512
    for s in range(0, n, block):
        e = min(s + block, n)
        d = np.hypot(x_ev[s:e, None] - x_ev[None, :], y_ev[s:e, None] - y_ev[None, :])
        d.sort(axis=1)
        h[s:e] = np.maximum(d[:, min(n_neighbour, n - 1)], h_min)

    dens = np.zeros(x_q.size)
    for s in range(0, n, block):
        e = min(s + block, n)
        hh = h[s:e][None, :]
        d2 = ((x_q[:, None] - x_ev[None, s:e]) ** 2
              + (y_q[:, None] - y_ev[None, s:e]) ** 2)
        dens += np.sum(np.exp(-d2 / (2 * hh ** 2)) / (2 * np.pi * hh ** 2), axis=1)
    return dens / n


# ---------------------------------------------------------------- ETAS

def build_pairs(t, m, x, y, t_lo, t_hi):
    """
    All (target j, ancestor i) pairs with 0 < t_j - t_i < T_CUT, for targets in
    [t_lo, t_hi). Precomputing once turns every later likelihood evaluation into
    one vectorised pass.
    """
    tgt = np.flatnonzero((t >= t_lo) & (t < t_hi))
    js, iss = [], []
    lo = 0
    for j in tgt:
        while t[lo] < t[j] - T_CUT:
            lo += 1
        if j > lo:
            idx = np.arange(lo, j)
            js.append(np.full(idx.size, j))
            iss.append(idx)
    if not js:
        return tgt, np.array([], int), np.array([]), np.array([]), np.array([])
    js = np.concatenate(js)
    iss = np.concatenate(iss)
    dt = t[js] - t[iss]
    dm = m[iss] - MC
    dr = np.hypot(x[js] - x[iss], y[js] - y[iss])
    return tgt, js, dt, dm, dr


def omori_integral(t, m, t0, t1, K, alpha, c, p):
    """Expected triggered count in [t0,t1] from all events before t1."""
    sel = (t < t1) & (t > t0 - T_CUT)
    if not sel.any():
        return 0.0
    ti = t[sel]
    lo = np.maximum(t0, ti)
    return float(np.sum(K * np.exp(alpha * (m[sel] - MC))
                        * ((t1 - ti + c) ** (1 - p) - (lo - ti + c) ** (1 - p))
                        / (1 - p)))


def spatial_kernel(r, dm, D, gamma, q):
    """Normalised so that its integral over the plane is 1."""
    d2 = D * D * 10.0 ** (gamma * dm)
    return (q - 1.0) / (np.pi * d2) * (1.0 + r * r / d2) ** (-q)


def fit_etas(t, m, x, y, bg_at_events, t0, t1, spatial: bool):
    tgt, js, dt, dm, dr = build_pairs(t, m, x, y, t0, t1)
    n_tgt = tgt.size
    pos = np.searchsorted(tgt, js)

    def neg_ll(theta):
        par = np.exp(theta)
        if spatial:
            mu_tot, K, alpha, c, p, D, gamma, q = par
        else:
            mu_tot, K, alpha, c, p = par
            D = gamma = 0.0
            q = 2.0
        p = min(p, 2.6)
        q = min(max(q, 1.05), 4.0)
        if p <= 1.01:
            return 1e12

        trig = K * np.exp(alpha * dm) * (dt + c) ** (-p)
        if spatial:
            trig = trig * spatial_kernel(dr, dm, D, gamma, q)
        summed = np.bincount(pos, weights=trig, minlength=n_tgt)

        base = mu_tot * bg_at_events[tgt] if spatial else np.full(n_tgt, mu_tot)
        lam = base + summed
        if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
            return 1e12
        integ = mu_tot * (t1 - t0) + omori_integral(t, m, t0, t1, K, alpha, c, p)
        val = -(np.sum(np.log(lam)) - integ)
        return val if np.isfinite(val) else 1e12

    if spatial:
        x0 = np.log([0.45, 0.02, 1.4, 0.05, 1.25, 12.0, 0.5, 1.6])
    else:
        x0 = np.log([0.45, 0.02, 1.4, 0.05, 1.25])

    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options=dict(maxiter=6000, maxfev=6000, xatol=1e-4, fatol=1e-3))
    par = np.exp(res.x)
    keys = (["mu_tot", "K", "alpha", "c", "p", "D", "gamma", "q"] if spatial
            else ["mu_tot", "K", "alpha", "c", "p"])
    out = dict(zip(keys, par))
    out["p"] = min(out["p"], 2.6)
    if spatial:
        out["q"] = min(max(out["q"], 1.05), 4.0)
    return out, -res.fun


def score_pointprocess(t, m, x, y, bg_at_events, params, t0, t1, spatial):
    """
    Pseudo-prospective log-likelihood over the test window.

    Both variants are densities in space AND time, so the scores are directly
    comparable. The temporal model spreads its triggering over the background
    density -- it can only say "the region is busier today". The spatio-temporal
    model puts the triggering where the parent earthquake was. That difference
    is the whole point of the comparison, and it only shows up if both are
    normalised the same way.
    """
    tgt, js, dt, dm, dr = build_pairs(t, m, x, y, t0, t1)
    pos = np.searchsorted(tgt, js)
    P = params
    trig = P["K"] * np.exp(P["alpha"] * dm) * (dt + P["c"]) ** (-P["p"])
    if spatial:
        trig = trig * spatial_kernel(dr, dm, P["D"], P["gamma"], P["q"])
    summed = np.bincount(pos, weights=trig, minlength=tgt.size)

    if spatial:
        lam = P["mu_tot"] * bg_at_events[tgt] + summed
    else:
        lam = (P["mu_tot"] + summed) * bg_at_events[tgt]
    lam = np.maximum(lam, 1e-300)
    integ = P["mu_tot"] * (t1 - t0) + omori_integral(t, m, t0, t1,
                                                    P["K"], P["alpha"],
                                                    P["c"], P["p"])
    return float(np.sum(np.log(lam)) - integ), lam, tgt


def lambda_at(qt, qx, qy, qbg, t, m, x, y, params, spatial):
    """
    Forecast rate density at arbitrary space-time query points, using only the
    history strictly before each query time. Needed to sample the space-time
    volume for a Molchan diagram.
    """
    P = params
    out = np.empty(qt.size)
    order = np.argsort(qt)
    for k in order:
        hi = np.searchsorted(t, qt[k], side="left")
        lo = np.searchsorted(t, qt[k] - T_CUT, side="left")
        if hi <= lo:
            out[k] = (P["mu_tot"] * qbg[k] if spatial else P["mu_tot"] * qbg[k])
            continue
        dt = qt[k] - t[lo:hi]
        dm = m[lo:hi] - MC
        trig = P["K"] * np.exp(P["alpha"] * dm) * (dt + P["c"]) ** (-P["p"])
        if spatial:
            dr = np.hypot(qx[k] - x[lo:hi], qy[k] - y[lo:hi])
            out[k] = P["mu_tot"] * qbg[k] + np.sum(
                trig * spatial_kernel(dr, dm, P["D"], P["gamma"], P["q"]))
        else:
            out[k] = (P["mu_tot"] + np.sum(trig)) * qbg[k]
    return np.maximum(out, 1e-300)


# ---------------------------------------------------------------- lunar

def lunar_phase(dts):
    eph = bodies_equatorial(jd_from_datetime_array(dts))
    ra_m, dec_m, _ = eph["moon"]
    ra_s, dec_s, _ = eph["sun"]
    r = np.pi / 180
    ce = (np.sin(dec_m * r) * np.sin(dec_s * r)
          + np.cos(dec_m * r) * np.cos(dec_s * r) * np.cos((ra_m - ra_s) * r))
    el = np.degrees(np.arccos(np.clip(ce, -1, 1)))
    sign = np.sign(((ra_m - ra_s) % 360) - 180)
    return np.where(sign >= 0, el, 360 - el)


def score_lunar(phase_ev, phase_grid, bg_at_events, mu_tot, span, k):
    """
    lambda = mu_tot * bg(x,y) * (1 + A cos(k*theta - phi)).
    A and phi are fitted on the test data itself -- deliberately generous.
    """
    th_e = np.radians(k * phase_ev)
    th_g = np.radians(k * phase_grid)

    def neg_ll(par):
        a, phi = par
        if abs(a) >= 0.98:
            return 1e12
        se = 1 + a * np.cos(th_e - phi)
        if np.any(se <= 0):
            return 1e12
        integ = mu_tot * span * np.mean(1 + a * np.cos(th_g - phi))
        return -(np.sum(np.log(mu_tot * bg_at_events * se)) - integ)

    best, bv = None, np.inf
    for phi0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        r = minimize(neg_ll, [0.05, phi0], method="Nelder-Mead",
                     options=dict(maxiter=1500))
        if r.fun < bv:
            bv, best = r.fun, r.x
    return -bv, best[0]


# ---------------------------------------------------------------- main

def main():
    rule("HOW MUCH EARTHQUAKE FORECASTING SKILL ACTUALLY EXISTS?")

    cat = load_catalog()
    t0_dt = cat[0][0]
    t = np.array([(e[0] - t0_dt).total_seconds() / 86400.0 for e in cat])
    m = np.array([e[1] for e in cat])
    lat = np.array([e[2] for e in cat])
    lon = np.array([e[3] for e in cat])
    x, y = xy_km(lat, lon, lat0=-2.0)

    t_split = (TRAIN_END - t0_dt).total_seconds() / 86400.0
    t_end = t[-1]
    train = t < t_split
    test = (t >= t_split)

    print(f"""
  region     Indonesia and surrounding arc (13S-8N, 93E-143E)
  catalogue  USGS ComCat, M >= {MC}, {cat[0][0].date()} to {cat[-1][0].date()}
  events     {len(cat)}   train {int(train.sum())}   test {int(test.sum())}
  test span  {TRAIN_END.date()} onward, {t_end - t_split:.0f} days
""")

    print("  estimating background seismicity density (adaptive kernel) ...")
    bg = background_density(x[train], y[train], x, y)
    bg = np.maximum(bg, 1e-12)

    print("  fitting temporal ETAS ...")
    par_t, _ = fit_etas(t, m, x, y, bg, t[0] + 30, t_split, spatial=False)
    print("  fitting spatio-temporal ETAS ...")
    par_st, _ = fit_etas(t, m, x, y, bg, t[0] + 30, t_split, spatial=True)

    print(f"""
    temporal   mu={par_t['mu_tot']:.3f}/d  K={par_t['K']:.4f}  a={par_t['alpha']:.2f}"""
          f"""  c={par_t['c']:.4f}  p={par_t['p']:.3f}
    spatial    mu={par_st['mu_tot']:.3f}/d  K={par_st['K']:.4f}  a={par_st['alpha']:.2f}"""
          f"""  c={par_st['c']:.4f}  p={par_st['p']:.3f}
               D={par_st['D']:.1f} km  gamma={par_st['gamma']:.2f}  q={par_st['q']:.2f}""")

    span = t_end - t_split
    n_test = int(((t >= t_split) & (t < t_end)).sum())

    # ---- baselines and models, all as point processes ----
    tgt_idx = np.flatnonzero((t >= t_split) & (t < t_end))
    mu_pois = n_test / span
    ll_pois = float(np.sum(np.log(mu_pois * bg[tgt_idx])) - mu_pois * span)

    ll_t, lam_t, _ = score_pointprocess(t, m, x, y, bg, par_t, t_split, t_end, False)
    ll_st, lam_st, tgt = score_pointprocess(t, m, x, y, bg, par_st, t_split, t_end, True)

    mid = [t0_dt + timedelta(days=float(d) + 0.5)
           for d in np.arange(int(t_split), int(t_end))]
    ph_grid = lunar_phase(mid)
    ph_ev = lunar_phase([t0_dt + timedelta(days=float(v)) for v in t[tgt_idx]])

    lunar_scores = {}
    for name, k in (("LUNAR h4 (SSGEOS form)", 4), ("LUNAR h2 (tidal form)", 2),
                    ("LUNAR h1", 1)):
        ll_l, amp = score_lunar(ph_ev, ph_grid, bg[tgt_idx], mu_pois, span, k)
        lunar_scores[name] = (ll_l, amp)

    rule("INFORMATION GAIN PER EARTHQUAKE (CSEP standard)")
    print("""
  How many times more likely does each model make the sequence that actually
  happened, per earthquake? A gain of G nats means exp(G) times more likely.
  All models share the same spatial background, so this isolates the question
  of what predicts WHEN, and for ETAS-ST also WHERE.
""")
    print(f"  {'model':<26} {'log-likelihood':>15} {'gain/event':>12} {'x more likely':>14}")
    print("  " + "-" * 72)
    print(f"  {'POISSON (benchmark)':<26} {ll_pois:>15.1f} {0.0:>12.4f} {1.00:>14.2f}")
    best_lun = 0.0
    for name, (ll_l, amp) in lunar_scores.items():
        g = (ll_l - ll_pois) / n_test
        best_lun = max(best_lun, g)
        print(f"  {name:<26} {ll_l:>15.1f} {g:>12.4f} {np.exp(g):>14.2f}"
              f"  [A={amp:+.3f}]")
    g_t = (ll_t - ll_pois) / n_test
    g_st = (ll_st - ll_pois) / n_test
    print(f"  {'ETAS temporal':<26} {ll_t:>15.1f} {g_t:>12.4f} {np.exp(g_t):>14.2f}")
    print(f"  {'ETAS spatio-temporal':<26} {ll_st:>15.1f} {g_st:>12.4f} {np.exp(g_st):>14.2f}")

    print(f"""
  ETAS-ST is worth {g_st:.2f} nats per earthquake: each observed earthquake is
  {np.exp(g_st):.0f}x more probable under it than under the time-independent benchmark.
  The best lunar model -- fitted on the test data with amplitude and phase both
  free -- is worth {best_lun:.4f} nats, i.e. {np.exp(best_lun):.3f}x.

  Skill ratio: about {g_st/max(best_lun,1e-9):.0f} to 1.""")

    rule("THE OPERATIONAL NUMBER: AFTERSHOCK FORECASTS, VERIFIED")
    lun_amp = max(abs(v[1]) for v in lunar_scores.values())
    P = par_st
    R_KM, WIN_D = 50.0, 7.0

    big = np.flatnonzero((m >= 6.5) & (t >= t_split) & (t < t_end - WIN_D))
    exp_etas, exp_bg, obs = [], [], []
    for i in big:
        dm_i = m[i] - MC
        d2 = P["D"] ** 2 * 10 ** (P["gamma"] * dm_i)
        frac_disc = 1.0 - (1.0 + R_KM ** 2 / d2) ** (1 - P["q"])
        omori = (P["K"] * np.exp(P["alpha"] * dm_i)
                 * ((WIN_D + P["c"]) ** (1 - P["p"]) - P["c"] ** (1 - P["p"]))
                 / (1 - P["p"]))
        exp_etas.append(omori * frac_disc)
        near = np.hypot(x - x[i], y - y[i]) <= R_KM
        bg_mass = bg[near].sum() / max(bg.sum(), 1e-12)
        exp_bg.append(P["mu_tot"] * bg_mass * WIN_D)
        obs.append(int(np.sum(near & (t > t[i]) & (t <= t[i] + WIN_D))))

    exp_etas = np.array(exp_etas)
    exp_bg = np.array(exp_bg)
    obs = np.array(obs)
    tot_e, tot_b, tot_o = exp_etas.sum(), exp_bg.sum(), obs.sum()

    print(f"""
  The number an operational centre actually publishes: after a large event, how
  many M>={MC:.0f} earthquakes to expect within {R_KM:.0f} km over the next {WIN_D:.0f} days.
  Every M>=6.5 mainshock in the test period, forecast from parameters fitted
  only on pre-2016 data:

    mainshocks tested            {len(big)}
    forecast by ETAS-ST          {tot_e:8.0f} aftershocks
    forecast by background alone {tot_b:8.1f} aftershocks
    OBSERVED                     {tot_o:8.0f} aftershocks

    ETAS-ST is off by a factor of {max(tot_e,tot_o)/max(min(tot_e,tot_o),1e-9):.2f}
    the time-independent baseline is off by a factor of {tot_o/max(tot_b,1e-9):.0f}

  So the elevated risk after a mainshock is not a vague warning -- it is a
  number, and it verifies. The background model underestimates what follows a
  mainshock by roughly {tot_o/max(tot_b,1e-9):.0f}x, which is exactly the margin that matters
  when deciding whether to send people back into a damaged building.

  Set against that, the lunar model's best case is a rate change of
  {100*lun_amp:.1f}%, i.e. x{1+lun_amp:.2f}. It moves a 1-in-1000 day to 1-in-{1000/(1+lun_amp):.0f}.
  There is no decision anywhere that turns on that difference.
""")

    rule("ALARM PERFORMANCE (MOLCHAN DIAGRAM)")
    print("""
  Issue an alarm wherever and whenever the forecast rate exceeds a threshold.
  What fraction of the coming earthquakes does that catch, for what fraction of
  the space-time volume kept under alert? The space-time volume is sampled by
  Monte Carlo, since a full grid over 3880 days would be unnecessary.
  Random guessing lies on the diagonal.
""")
    n_mc = 20000
    rng = np.random.default_rng(20260816)
    qt = rng.uniform(t_split, t_end, n_mc)
    qlat = rng.uniform(-13, 8, n_mc)
    qlon = rng.uniform(93, 143, n_mc)
    qx, qy = xy_km(qlat, qlon, lat0=-2.0)
    qbg = np.maximum(background_density(x[train], y[train], qx, qy), 1e-12)

    lam_vol_st = lambda_at(qt, qx, qy, qbg, t, m, x, y, par_st, True)
    ph_vol = lunar_phase([t0_dt + timedelta(days=float(v)) for v in qt])
    kbest = max(lunar_scores, key=lambda kk: lunar_scores[kk][0])
    kk = {"LUNAR h4 (SSGEOS form)": 4, "LUNAR h2 (tidal form)": 2,
          "LUNAR h1": 1}[kbest]
    _, amp = lunar_scores[kbest]
    lam_vol_lun = mu_pois * qbg * (1 + amp * np.cos(np.radians(kk * ph_vol)))
    lam_ev_lun = mu_pois * bg[tgt_idx] * (1 + amp * np.cos(np.radians(kk * ph_ev)))

    lam_vol_pois = mu_pois * qbg
    lam_ev_pois = mu_pois * bg[tgt_idx]

    print(f"  {'volume on alert':>16} {'ETAS-ST':>10} {'LUNAR':>9} "
          f"{'background':>12} {'random':>9}")
    print("  " + "-" * 64)
    for frac in (0.01, 0.05, 0.10, 0.20, 0.50):
        hit_st = np.mean(lam_st >= np.quantile(lam_vol_st, 1 - frac))
        hit_lu = np.mean(lam_ev_lun >= np.quantile(lam_vol_lun, 1 - frac))
        hit_po = np.mean(lam_ev_pois >= np.quantile(lam_vol_pois, 1 - frac))
        print(f"  {frac:>15.0%} {hit_st:>9.1%} {hit_lu:>9.1%} "
              f"{hit_po:>11.1%} {frac:>8.0%}")
    print("""
  Read the last three columns together. All of them, lunar included, share the
  same map of where earthquakes happen -- that is where most of the apparent
  skill comes from, and it is just history, not prediction. The lunar column
  sits on top of the background column because a few-percent modulation adds
  nothing to it. ETAS-ST is the only one that pulls away, and it does so by
  reacting to earthquakes that have already occurred.""")

    rule("WHAT THIS BUYS, AND WHAT IT DOES NOT")
    print(f"""
  REAL, and operational today:
    * after a large earthquake, a quantified and verifiable expected number of
      aftershocks at a specific place -- {tot_o/max(tot_b,1e-9):.0f}x what a time-independent model
      would say, and accurate to a factor of {max(tot_e,tot_o)/max(min(tot_e,tot_o),1e-9):.1f} on real test data.
      This is what USGS, INGV, GNS and BMKG issue.
    * timescales of hours to weeks, which is enough for operational decisions:
      shelter occupancy, re-entry, crane and scaffolding work, school closure.

  NOT AVAILABLE, from this or any other method:
    * the date of the next large earthquake on a quiet fault
    * magnitude and location in advance
    * any warning whatsoever before the FIRST event of a sequence -- and that
      first event is usually the one that kills people.

  The gap between those two lists is not a software problem. It is why the
  effort that actually saves lives in Indonesia goes into building codes,
  early warning (seconds, from the P wave), tsunami warning (minutes), and
  preparedness -- not into forecasting the date.
""")


if __name__ == "__main__":
    main()
