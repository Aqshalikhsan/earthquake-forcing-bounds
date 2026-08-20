"""
The diagnostic fingerprint of a precursor claim.

A single p-value cannot tell a real association from an artefact, because
several very different mechanisms produce the same p. What separates them is
how the result RESPONDS when the analysis is perturbed in a controlled way, and
that response differs by mechanism:

    a real association        barely moves when the validation split changes,
                              survives declustering, and does not depend on the
                              predictor being a smooth function of time
    a calendar leak           collapses under a chronological split, and the
                              predictor reconstructs the date almost exactly
    aftershock clustering     collapses under declustering
    a lookahead window        collapses when sampling is made strictly causal
    a search artefact         collapses under a max-statistic correction
    catalogue growth          rides a trend present in both series

Each perturbation is therefore one axis, and the vector of responses is a
fingerprint. This module computes that vector from the two things every claim
of this kind has: a daily predictor series and a daily earthquake indicator.

The reference scales come from measurement rather than assumption. In the study
this module was built for, the validation split moved a real seismicity signal
by 0.008 AUC and a calendar-driven forcing bank by 0.208; declustering moved a
tidal harmonic from p = 0.015 to p = 0.596; making a GRACE window causal moved
it from p = 0.0000 to p = 0.36. Those constants are what let the axes be read
on a common scale instead of judged by eye.

Nothing here is specific to earthquakes except the defaults. Any daily
occurrence series and any daily candidate predictor can be fingerprinted.
"""

from __future__ import annotations

import numpy as np

NDEC = 10
N_SHIFT = 200
MIN_SHIFT = 30          # days; a shift shorter than this leaves the series
                        # almost aligned with itself
RNG_DEFAULT = 20260819


class XorShift32:
    """
    A 32-bit xorshift, chosen because it ports exactly.

    Only exclusive-or and shifts are used, and both languages apply those to
    the same 32-bit pattern, so a browser and this module walk the identical
    sequence. Anything richer, including numpy's own generator, would make the
    fingerprint unreproducible outside Python and quietly void the calibration.
    """

    __slots__ = ("x",)

    def __init__(self, seed=RNG_DEFAULT):
        self.x = (seed & 0xFFFFFFFF) or 0x9E3779B9

    def next_u32(self):
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.x = x & 0xFFFFFFFF
        return self.x

    def uniform(self):
        return self.next_u32() / 4294967296.0

    def below(self, n):
        """An integer in [0, n), by the same arithmetic on both sides."""
        return int(self.uniform() * n)

    def permutation(self, n):
        """Fisher-Yates, walked in the same order in both languages."""
        idx = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            idx[i], idx[j] = idx[j], idx[i]
        return np.array(idx)


def _deciles(x):
    """Quantile bin index, with ties pushed into the same bin."""
    edges = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, NDEC + 1))
    edges[-1] += 1e-12
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, NDEC - 1)


def decile_statistic(x, y):
    """
    Largest deviation of any decile's rate from the overall rate, relative.

    Reads directly as "plus or minus X% modulation", which is what makes it
    comparable between an angle, a stress and a solar index.
    """
    idx = _deciles(x)
    base = y.mean()
    if base <= 0:
        return np.nan
    rates = np.array([y[idx == k].mean() if (idx == k).any() else np.nan
                      for k in range(NDEC)])
    return float(np.nanmax(np.abs(rates - base)) / base)


def circular_shift_null(x, y, n=N_SHIFT, rng=None, seed=RNG_DEFAULT):
    """
    Null distribution of the decile statistic under a circular time shift.

    Shifting preserves clustering, secular trend, seasonality and the marginal
    distribution of both series, and destroys only their alignment. A plain
    permutation would also destroy the smoothness of the predictor, which
    biases any comparison in favour of the real data.
    """
    r = XorShift32(seed)
    n_t = y.size
    out = np.empty(n)
    for i in range(n):
        k = MIN_SHIFT + r.below(n_t - 2 * MIN_SHIFT)
        out[i] = decile_statistic(x, np.roll(y, k))
    return out


def shift_p(x, y, n=N_SHIFT, rng=None, seed=RNG_DEFAULT):
    obs = decile_statistic(x, y)
    null = circular_shift_null(x, y, n, seed=seed)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    p = (np.sum(null >= obs) + 1) / (null.size + 1)
    detectable = float(np.percentile(null, 95))
    return float(p), obs, detectable


def _auc(score, y):
    """Rank-based AUC, no sklearn dependency in the hot loop."""
    ok = np.isfinite(score)
    s, t = score[ok], y[ok]
    if t.sum() == 0 or t.sum() == t.size:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    n_pos, n_neg = t.sum(), t.size - t.sum()
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def _logit_fit(X, y, iters=200, lr=0.5):
    """
    Ridge-penalised logistic regression by plain gradient descent.

    A dependency-free fit keeps the whole fingerprint portable: the same
    arithmetic runs here and in the browser build of the audit tool, so a
    fingerprint computed in either place is the same vector.
    """
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) / y.size + 0.01 * np.r_[0.0, w[1:]]
        w -= lr * g
    return w


def _logit_score(X, w):
    return np.column_stack([np.ones(X.shape[0]), X]) @ w


def split_sensitivity(x, y, rng=None, seed=RNG_DEFAULT):
    """
    AUC under a chronological split minus AUC under a random split.

    A real association is nearly indifferent to how the data are cut; an
    association carried by the calendar is not, because a random split lets the
    model interpolate a test day between training days on either side of it.

    This axis was expected to dominate the battery and does not. Permutation
    importance puts it near zero for the seven-way attribution, because the
    predictor's autocorrelation and spectral concentration identify a calendar
    variable more directly, and once those are present this adds little. It
    remains the right quantity to report on a single claim, where the question
    is how much the answer moves rather than which mechanism produced it.
    """
    r = XorShift32(seed)
    X = np.column_stack([x, np.gradient(x)])
    ok = np.isfinite(X).all(axis=1)
    X, yy = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    cut = int(X.shape[0] * 0.7)
    tr_c, te_c = np.arange(cut), np.arange(cut, X.shape[0])
    perm = r.permutation(X.shape[0])
    tr_r, te_r = perm[:cut], perm[cut:]

    a_c = _auc(_logit_score(X[te_c], _logit_fit(X[tr_c], yy[tr_c])), yy[te_c])
    a_r = _auc(_logit_score(X[te_r], _logit_fit(X[tr_r], yy[tr_r])), yy[te_r])
    return a_c, a_r


def date_recoverability(x):
    """
    How much of the calendar the predictor alone reconstructs.

    A planetary angle or a lunar phase is a deterministic function of the date,
    so a model given it can locate a day in time. That is the mechanism behind
    the split sensitivity above, and measuring it directly separates the cause
    from the symptom.
    """
    ok = np.isfinite(x)
    t = np.arange(x.size, dtype=float)[ok]
    xx = x[ok]
    if xx.size < 20 or np.std(xx) == 0:
        return np.nan
    # a short Fourier basis on the predictor, which is what a flexible learner
    # would extract from it anyway
    B = [xx, xx ** 2]
    for k in (1, 2, 3):
        B += [np.sin(k * xx / (np.std(xx) + 1e-12)),
              np.cos(k * xx / (np.std(xx) + 1e-12))]
    B = np.column_stack(B + [np.ones(xx.size)])
    coef, *_ = np.linalg.lstsq(B, t, rcond=None)
    pred = B @ coef
    ss_res = float(np.sum((t - pred) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def half_stability(x, y):
    """
    Agreement of the statistic between the two halves of the record.

    A real association of a given size should appear at roughly that size in
    both halves. An artefact of search or of a trend usually does not, and the
    variable that leads in one half is often not the one that leads in the
    other.
    """
    h = x.size // 2
    s1 = decile_statistic(x[:h], y[:h])
    s2 = decile_statistic(x[h:], y[h:])
    if not np.isfinite(s1) or not np.isfinite(s2) or (s1 + s2) == 0:
        return np.nan
    return float(abs(s1 - s2) / (s1 + s2))


def null_calibration(x, y, n=60, rng=None):
    """
    False-positive rate of the null on data with the association removed.

    Runs the whole test on shifted labels, where nothing can be there, and
    counts how often it still reports p below 0.05. A number far from 0.05
    means the null is broken, and every other axis becomes unreadable.
    """
    r = XorShift32(RNG_DEFAULT + 1)
    hits = 0
    for _i in range(n):
        k = MIN_SHIFT + r.below(y.size - 2 * MIN_SHIFT)
        p, _, _ = shift_p(x, np.roll(y, k), n=60, seed=RNG_DEFAULT + 7 + _i)
        hits += (p is not None and np.isfinite(p) and p < 0.05)
    return hits / n


def autocorrelation(x, lag=1):
    ok = np.isfinite(x)
    xx = x[ok] - np.mean(x[ok])
    if xx.size <= lag or np.std(xx) == 0:
        return np.nan
    return float(np.dot(xx[:-lag], xx[lag:]) / np.dot(xx, xx))


def trend_strength(v):
    """R-squared of a straight line in time. Both series riding one trend is
    the signature of catalogue growth rather than of a forcing."""
    ok = np.isfinite(v)
    t = np.arange(v.size, dtype=float)[ok]
    vv = v[ok]
    if vv.size < 20 or np.std(vv) == 0:
        return 0.0
    A = np.column_stack([t, np.ones(t.size)])
    coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
    res = vv - A @ coef
    return float(1.0 - np.sum(res ** 2) / np.sum((vv - vv.mean()) ** 2))


def decluster_temporal(y, window=30):
    """
    Keep only event days that do not follow another within `window` days.

    The Gardner-Knopoff windows used elsewhere in this project need magnitudes
    and locations, which a daily series does not carry. Single-link thinning on
    the time axis alone is cruder, but it removes the same thing: the repeated
    counting of one sequence as many independent samples.
    """
    ev = np.flatnonzero(y > 0)
    out = np.zeros_like(y)
    last = -(10 ** 9)
    for i in ev:
        if i - last > window:
            out[i] = 1.0
            last = i
    return out


def decluster_sensitivity(x, y, window=5, n=100, rng=None,
                          seed=RNG_DEFAULT + 2):
    """
    How much of the association is carried by clustered days.

    A real association should survive thinning at roughly its original
    strength. An apparent one produced by aftershocks collapses, which is what
    moved a tidal harmonic from p = 0.015 to p = 0.596 in the study this was
    built for.

    Two details decide whether this axis measures anything. The window has to
    be short: at a base rate near 30% a day, single-link thinning at 30 days
    keeps only a tenth of the events, and the statistic afterwards is small
    sample noise rather than a response to declustering. And the comparison has
    to be made against each series' own null, because thinning changes the
    sample size and therefore the width of the null band. Comparing raw
    statistics measures the thinning, not the clustering.
    """
    yd = decluster_temporal(y, window)
    if yd.sum() < 100:
        return np.nan

    def z(yy):
        null = circular_shift_null(x, yy, n, seed=seed)
        null = null[np.isfinite(null)]
        obs = decile_statistic(x, yy)
        if null.size < 10 or not np.isfinite(obs) or np.std(null) == 0:
            return np.nan
        return (obs - np.mean(null)) / np.std(null)

    # Both standardised values are returned raw rather than combined into a
    # ratio. Three hand-built contrasts were tried first and each mixed the
    # effect size with the sample-size shock of thinning, which removes 60% of
    # the days at this base rate. The pair carries strictly more information
    # than any scalar made from it, and a tree can find the boundary without
    # being told in advance what shape it has.
    return float(z(y)), float(z(yd))


def spectral_concentration(x, top=5):
    """
    Fraction of the predictor's power carried by its strongest few frequencies.

    A calendar variable is a sum of a few periodic terms, so its power sits in
    a handful of lines. A smooth random series selected for correlation has a
    broad spectrum instead. The confusion matrix showed those two classes being
    mistaken for one another, and this is the quantity that tells them apart.
    """
    xx = x[np.isfinite(x)]
    if xx.size < 64:
        return np.nan
    # The largest power-of-two prefix. The browser port evaluates this with a
    # radix-2 transform, which needs that length; truncating on both sides
    # keeps the two implementations exact, where zero-padding on one of them
    # would change the frequency grid and quietly change the value.
    n = 1
    while n * 2 <= xx.size:
        n *= 2
    xx = xx[:n]
    xx = xx - xx.mean()
    p = np.abs(np.fft.rfft(xx)) ** 2
    p = p[1:]                       # the mean is not information here
    tot = p.sum()
    if tot <= 0:
        return np.nan
    return float(np.sort(p)[-top:].sum() / tot)


FEATURES = ["stat", "p_shift", "log_p", "detect_ratio", "auc_chrono",
            "auc_random", "split_delta", "date_r2", "half_gap", "ac1",
            "trend_x", "trend_y", "z_full", "z_dec", "spectral",
            "null_fpr"]


def fingerprint(x, y, rng=None, cheap=False):
    """
    The full response vector.

    Set cheap=True to skip the null-calibration axis, which costs as much as
    everything else together and is only needed when the null itself is in
    question.
    """
    # rng is accepted and ignored: the fingerprint is a pure function of the
    # data now, so that a browser computes the same vector from the same input
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p, stat, detect = shift_p(x, y)
    _zf, _zd = decluster_sensitivity(x, y)
    a_c, a_r = split_sensitivity(x, y)
    ysm = np.convolve(y, np.ones(365) / 365.0, mode="same")

    f = {
        "stat": stat,
        "p_shift": p,
        "log_p": -np.log10(max(p, 1e-4)) if np.isfinite(p) else np.nan,
        "detect_ratio": stat / detect if detect and detect > 0 else np.nan,
        "auc_chrono": a_c,
        "auc_random": a_r,
        "split_delta": (a_r - a_c) if np.isfinite(a_r) and np.isfinite(a_c)
                        else np.nan,
        "date_r2": date_recoverability(x),
        "half_gap": half_stability(x, y),
        "ac1": autocorrelation(x),
        "trend_x": trend_strength(x),
        "trend_y": trend_strength(ysm),
        "z_full": _zf,
        "z_dec": _zd,
        "spectral": spectral_concentration(x),
        "null_fpr": np.nan if cheap else null_calibration(x, y, rng=rng),
    }
    return np.array([f[k] for k in FEATURES]), f
