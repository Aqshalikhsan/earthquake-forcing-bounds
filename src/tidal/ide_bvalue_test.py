"""
The one tidal claim a Schuster test on phase cannot reach.

perevent_test.py asks whether the tidal PHASE of large earthquakes is
distributed non-uniformly, and answers no (p = 0.788 on each event's own fault
plane). Ide, Yabe and Tanaka (2016, Nature Geoscience 9, 834-837) claim
something structurally different: not that earthquakes prefer a phase, but that
the SIZE-FREQUENCY DISTRIBUTION shifts with tidal stress AMPLITUDE, so that
when the tide is strong a larger share of the events that do occur are big
ones. In b-value terms, b should fall as amplitude rises. A test on phase is
blind to that whatever its power, so this file supplies the missing test.

    statistic   b estimated by Aki-Utsu (1965) within deciles of tidal dCFS
                amplitude on each event's own fault plane, summarised as
                b(low decile) - b(high decile) and as the Spearman correlation
                of b against decile. The fraction of large events per decile is
                reported alongside because it is closer to the form the claim
                is usually stated in and needs no completeness assumption
                beyond Mc.

    null        The tidal amplitudes are permuted AMONG EVENTS, which breaks
                the pairing between an event's magnitude and its own tidal
                amplitude while leaving both marginal distributions untouched.
                That is exactly H0. The permutation is done WITHIN 10-degree
                cells, because tidal amplitude is largely a matter of where you
                are -- ocean loading dominates it -- and so is b, which varies
                with tectonic setting. A global permutation would let that
                shared geography masquerade as a result; permuting inside a
                cell removes it, since every replicate keeps each cell's own
                amplitudes and each cell's own magnitudes.

    catalogue   GCMT, Mw >= Mc, depth <= 70 km. Reported twice: with the full
                catalogue, which is what a size-frequency claim is normally
                evaluated on, and declustered, because an aftershock sequence
                contributes many correlated magnitudes at nearly one amplitude
                and can move b on its own.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _bootstrap import DATA, datafile  # noqa: F401

import numpy as np

from gcmt import parse_ndk
from tidal_analysis import compute_tidal

MC = 5.2                    # completeness of GCMT, same value as bvalue_test
DM = 0.1                    # magnitude bin correction
MAXDEPTH = 70.0
BIG = 6.5                   # "large" for the fraction statistic
NDEC = 10
N_PERM = 2000
CELL = 10.0                 # degrees, the block permutation is done inside
MIN_EV = 150                # per decile, below which b is not estimated
RNG = np.random.default_rng(20260818)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def b_aki_utsu(mags, mc=MC, dm=DM):
    """Maximum-likelihood b. Same estimator as src/seismicity/bvalue_test.py."""
    m = mags[mags >= mc]
    if m.size < MIN_EV:
        return np.nan
    mean = m.mean()
    if mean - (mc - dm / 2) <= 1e-6:
        return np.nan
    return float(np.log10(np.e) / (mean - (mc - dm / 2)))


def decile_stats(mw, amp):
    """b and the large-event fraction in each decile of tidal amplitude."""
    edges = np.quantile(amp, np.linspace(0, 1, NDEC + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, amp, side="right") - 1, 0, NDEC - 1)
    b = np.array([b_aki_utsu(mw[idx == k]) for k in range(NDEC)])
    frac = np.array([np.mean(mw[idx == k] >= BIG) if (idx == k).sum() else np.nan
                     for k in range(NDEC)])
    return b, frac, idx


def spearman(y):
    """Rank correlation of y against its own index, ignoring NaN."""
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    x = np.arange(y.size)[ok].astype(float)
    r = np.argsort(np.argsort(y[ok])).astype(float)
    xr = np.argsort(np.argsort(x)).astype(float)
    xc, rc = xr - xr.mean(), r - r.mean()
    d = np.sqrt((xc ** 2).sum() * (rc ** 2).sum())
    return float((xc * rc).sum() / d) if d > 0 else np.nan


def statistics(mw, amp):
    b, frac, _ = decile_stats(mw, amp)
    lo = np.nanmean(b[:NDEC // 2])
    hi = np.nanmean(b[NDEC // 2:])
    return dict(b=b, frac=frac, drop=lo - hi, rho=spearman(b),
                fdiff=np.nanmean(frac[NDEC // 2:]) - np.nanmean(frac[:NDEC // 2]))


def block_ids(lat, lon):
    ilat = np.floor((lat + 90.0) / CELL).astype(int)
    ilon = np.floor((lon % 360.0) / CELL).astype(int)
    return ilat * 1000 + ilon


def permute_within(amp_sorted, block_sorted, rng):
    """
    Shuffle amplitudes inside each block, in one vectorised pass.

    Looping over blocks and calling permutation on each costs a Python round
    trip per block per replicate, which is where the runtime went. Sorting once
    by block outside the loop turns the whole permutation into a single
    lexsort: ordering by block first and by a fresh random key second is a
    random permutation within every block simultaneously.
    """
    key = rng.random(amp_sorted.size)
    return amp_sorted[np.lexsort((key, block_sorted))]


def run(tag, mw, amp, lat, lon):
    obs = statistics(mw, amp)
    order = np.argsort(block_ids(lat, lon), kind="stable")
    block_sorted = block_ids(lat, lon)[order]
    amp_sorted, mw_sorted = amp[order], mw[order]
    null = {k: np.empty(N_PERM) for k in ("drop", "rho", "fdiff")}
    for i in range(N_PERM):
        st = statistics(mw_sorted,
                        permute_within(amp_sorted, block_sorted, RNG))
        for k in null:
            null[k][i] = st[k]

    print(f"\n  {tag}: {mw.size:,} gempa Mw>={MC}, "
          f"{int((mw >= BIG).sum()):,} di antaranya Mw>={BIG}")
    print(f"\n  {'desil amplitudo':<18} {'b':>7} {'frac Mw>=6.5':>14} "
          f"{'n':>8}")
    print("  " + "-" * 50)
    _, _, idx = decile_stats(mw, amp)
    for k in range(NDEC):
        n = int((idx == k).sum())
        print(f"  {k+1:>2} {'(terlemah)' if k == 0 else ('(terkuat)' if k == NDEC-1 else ''):<15}"
              f"{obs['b'][k]:>7.3f} {100*obs['frac'][k]:>13.2f}% {n:>8,}")

    print(f"\n  {'statistik':<34} {'teramati':>10} {'null 95%':>18} {'p':>8}")
    print("  " + "-" * 74)
    names = {"drop": "b(lemah) - b(kuat), harus > 0",
             "rho": "korelasi b vs desil, harus < 0",
             "fdiff": "kenaikan fraksi Mw>=6.5"}
    tail = {"drop": "hi", "rho": "lo", "fdiff": "hi"}
    out = {}
    for k, nm in names.items():
        d = null[k][np.isfinite(null[k])]
        if tail[k] == "hi":
            p = (np.sum(d >= obs[k]) + 1) / (d.size + 1)
        else:
            p = (np.sum(d <= obs[k]) + 1) / (d.size + 1)
        lo95, hi95 = np.percentile(d, [2.5, 97.5])
        out[k] = (obs[k], p)
        print(f"  {nm:<34} {obs[k]:>10.4f} "
              f"{'[%.4f, %.4f]' % (lo95, hi95):>18} {p:>8.3f}")
    return obs, out


def main():
    rule("UJI IDE dkk. 2016: apakah nilai-b bergeser dengan AMPLITUDO pasang surut")
    print("""
  Uji fase (Schuster) tidak bisa menjawab klaim ini, seberapa pun besar
  dayanya, karena klaimnya bukan tentang jam berapa gempa terjadi melainkan
  tentang ukuran gempa yang terjadi ketika pasang surut sedang kuat.
""")

    ev = [e for e in parse_ndk(datafile("gcmt.ndk"))
          if e.depth <= MAXDEPTH and e.mw >= MC]
    ev.sort(key=lambda e: e.dt)
    print(f"  menghitung tegangan pasang surut di bidang sesar "
          f"{len(ev):,} gempa ...")
    phase, amp = compute_tidal(ev, plane="shallow")

    mw = np.array([e.mw for e in ev])
    lat = np.array([e.lat for e in ev])
    lon = np.array([e.lon for e in ev])
    ok = np.isfinite(amp)
    print(f"  tercakup {100*ok.mean():.1f}%")
    mw, amp, lat, lon = mw[ok], amp[ok], lat[ok], lon[ok]

    rule("KATALOG PENUH")
    obs_full, res_full = run("penuh", mw, amp, lat, lon)

    rule("SETELAH DECLUSTERING")
    from perevent_test import decluster_cmt
    keep = decluster_cmt([e for e, o in zip(ev, ok) if o])
    kid = {id(e) for e in keep}
    m = np.array([id(e) in kid for e, o in zip(ev, ok) if o])
    obs_dec, res_dec = run("declustered", mw[m], amp[m], lat[m], lon[m])

    np.savez(DATA / "results" / "ide_bvalue.npz",
             b_full=obs_full["b"], frac_full=obs_full["frac"],
             b_dec=obs_dec["b"], frac_dec=obs_dec["frac"],
             p_full=np.array([res_full[k][1] for k in ("drop", "rho", "fdiff")]),
             p_dec=np.array([res_dec[k][1] for k in ("drop", "rho", "fdiff")]),
             n_full=int(mw.size), n_dec=int(m.sum()))

    rule("VONIS")
    pf = min(res_full[k][1] for k in res_full)
    pd_ = min(res_dec[k][1] for k in res_dec)
    print(f"""
  Arah yang diramalkan Ide dkk.: b turun ketika amplitudo pasang surut naik,
  sehingga selisih b(lemah) - b(kuat) positif dan korelasinya negatif.

  p terkecil dari tiga statistik, katalog penuh   : {pf:.3f}
  p terkecil dari tiga statistik, declustered     : {pd_:.3f}

  Null di sini mengacak amplitudo DI DALAM sel 10 derajat, jadi setiap
  replikat mempertahankan amplitudo khas tiap wilayah dan magnitudo khas tiap
  wilayah. Yang dirusak hanya pasangan antara satu gempa dan amplitudonya
  sendiri, yang memang isi hipotesisnya.
""")


if __name__ == "__main__":
    main()
