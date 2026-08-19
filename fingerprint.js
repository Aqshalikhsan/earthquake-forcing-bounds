/*
 * The diagnostic fingerprint, in the browser.
 *
 * This is a line-for-line port of src/audit/fingerprint.py. It has to be one,
 * because the conformal coverage that the tool promises was measured on the
 * Python version: if this file computed a slightly different vector, the
 * guarantee would not apply to what anyone actually runs.
 *
 * Two things make an exact port possible. The random draws come from a 32-bit
 * xorshift, which uses only exclusive-or and shifts and therefore walks the
 * same sequence in both languages. And the fingerprint takes no seed: it is a
 * pure function of the predictor series and the event series.
 *
 * check() at the bottom reproduces the first values of the generator so a
 * mismatch shows up immediately rather than as a wrong verdict later.
 */

const NDEC = 10;
const N_SHIFT = 200;
const MIN_SHIFT = 30;
const SEED = 20260819;

export const FEATURES = [
  "stat", "p_shift", "log_p", "detect_ratio", "auc_chrono", "auc_random",
  "split_delta", "date_r2", "half_gap", "ac1", "trend_x", "trend_y",
  "z_full", "z_dec", "spectral"
];

/* ------------------------------------------------------- random numbers */
class XorShift32 {
  constructor(seed = SEED) {
    this.x = (seed >>> 0) || 0x9e3779b9;
  }
  nextU32() {
    let x = this.x;
    x ^= (x << 13); x >>>= 0;
    x ^= (x >>> 17);
    x ^= (x << 5); x >>>= 0;
    this.x = x;
    return x;
  }
  uniform() { return this.nextU32() / 4294967296; }
  below(n) { return Math.floor(this.uniform() * n); }
  permutation(n) {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = this.below(i + 1);
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }
}

/* --------------------------------------------------------------- helpers */
const finite = (a) => a.filter(Number.isFinite);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

function std(a) {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length);
}

function quantiles(sorted, n) {
  // numpy's default linear interpolation, matched exactly
  const out = [];
  for (let i = 0; i <= n; i++) {
    const pos = (i / n) * (sorted.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    out.push(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo));
  }
  return out;
}

function decileIndex(x) {
  const s = finite(x).slice().sort((a, b) => a - b);
  const e = quantiles(s, NDEC);
  e[NDEC] += 1e-12;
  return x.map((v) => {
    let k = 0;
    while (k < NDEC && v > e[k + 1]) k++;
    return Math.min(Math.max(k, 0), NDEC - 1);
  });
}

function decileStatistic(x, y, idx) {
  idx = idx || decileIndex(x);
  const base = mean(y);
  if (base <= 0) return NaN;
  let best = NaN;
  for (let k = 0; k < NDEC; k++) {
    let s = 0, c = 0;
    for (let i = 0; i < y.length; i++) if (idx[i] === k) { s += y[i]; c++; }
    if (!c) continue;
    const d = Math.abs(s / c - base) / base;
    if (!(best >= d)) best = d;
  }
  return best;
}

function roll(y, k) {
  const n = y.length, out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = y[(((i - k) % n) + n) % n];
  return out;
}

/* The statistic for a shifted series, without building the shifted series.
 * Rolling allocated an array per replicate and then cost one pass per decile;
 * this reads through the offset in a single pass and gives the same number. */
function shiftedStatistic(idx, y, k, base) {
  const n = y.length;
  const sum = new Float64Array(NDEC), cnt = new Float64Array(NDEC);
  for (let i = 0; i < n; i++) {
    const b = idx[i];
    let j = i - k;
    if (j < 0) j += n; else if (j >= n) j -= n;
    sum[b] += y[j];
    cnt[b]++;
  }
  let best = NaN;
  for (let b = 0; b < NDEC; b++) {
    if (!cnt[b]) continue;
    const d = Math.abs(sum[b] / cnt[b] - base) / base;
    if (!(best >= d)) best = d;
  }
  return best;
}

function shiftNull(x, y, n, seed) {
  const r = new XorShift32(seed);
  const idx = decileIndex(x);
  const base = mean(y);
  const out = [];
  for (let i = 0; i < n; i++) {
    const k = MIN_SHIFT + r.below(y.length - 2 * MIN_SHIFT);
    out.push(base > 0 ? shiftedStatistic(idx, y, k, base) : NaN);
  }
  return out;
}

function percentile(a, p) {
  const s = a.slice().sort((u, v) => u - v);
  const pos = (p / 100) * (s.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

function shiftP(x, y, n = N_SHIFT, seed = SEED) {
  const obs = decileStatistic(x, y);
  const nul = finite(shiftNull(x, y, n, seed));
  if (!nul.length || !Number.isFinite(obs)) return [NaN, NaN, NaN];
  const p = (nul.filter((v) => v >= obs).length + 1) / (nul.length + 1);
  return [p, obs, percentile(nul, 95)];
}

/* --------------------------------------------------------- the fifteen axes */
function auc(score, y) {
  const pairs = [];
  for (let i = 0; i < y.length; i++)
    if (Number.isFinite(score[i])) pairs.push([score[i], y[i]]);
  const nPos = pairs.filter((q) => q[1] === 1).length;
  if (!nPos || nPos === pairs.length) return NaN;
  pairs.sort((a, b) => a[0] - b[0]);
  let rankSum = 0;
  for (let i = 0; i < pairs.length; i++) if (pairs[i][1] === 1) rankSum += i + 1;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * (pairs.length - nPos));
}

function logitFit(X, y, iters = 200, lr = 0.5) {
  const d = X[0].length + 1;
  const w = new Array(d).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = new Array(d).fill(0);
    for (let i = 0; i < X.length; i++) {
      let z = w[0];
      for (let j = 0; j < X[i].length; j++) z += w[j + 1] * X[i][j];
      const p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
      const e = p - y[i];
      g[0] += e;
      for (let j = 0; j < X[i].length; j++) g[j + 1] += e * X[i][j];
    }
    for (let j = 0; j < d; j++) {
      let gj = g[j] / y.length;
      if (j > 0) gj += 0.01 * w[j];
      w[j] -= lr * gj;
    }
  }
  return w;
}

function gradient(x) {
  const n = x.length, g = new Array(n);
  g[0] = x[1] - x[0];
  g[n - 1] = x[n - 1] - x[n - 2];
  for (let i = 1; i < n - 1; i++) g[i] = (x[i + 1] - x[i - 1]) / 2;
  return g;
}

function splitSensitivity(x, y) {
  const r = new XorShift32(SEED);
  const gx = gradient(x);
  const rows = [], yy = [];
  for (let i = 0; i < x.length; i++)
    if (Number.isFinite(x[i]) && Number.isFinite(gx[i])) {
      rows.push([x[i], gx[i]]); yy.push(y[i]);
    }
  for (let j = 0; j < 2; j++) {
    const col = rows.map((v) => v[j]);
    const m = mean(col); let s = std(col); if (s === 0) s = 1;
    rows.forEach((v) => { v[j] = (v[j] - m) / s; });
  }
  const cut = Math.floor(rows.length * 0.7);
  const chronoTr = [], chronoTe = [];
  for (let i = 0; i < rows.length; i++) (i < cut ? chronoTr : chronoTe).push(i);
  const perm = r.permutation(rows.length);
  const randTr = perm.slice(0, cut), randTe = perm.slice(cut);

  const run = (tr, te) => {
    const w = logitFit(tr.map((i) => rows[i]), tr.map((i) => yy[i]));
    const sc = te.map((i) => {
      let z = w[0];
      for (let j = 0; j < 2; j++) z += w[j + 1] * rows[i][j];
      return z;
    });
    return auc(sc, te.map((i) => yy[i]));
  };
  return [run(chronoTr, chronoTe), run(randTr, randTe)];
}

function lstsq(B, t) {
  // normal equations with a small ridge, enough for these tiny designs
  const d = B[0].length;
  const A = Array.from({ length: d }, () => new Array(d).fill(0));
  const b = new Array(d).fill(0);
  for (let i = 0; i < B.length; i++) {
    for (let j = 0; j < d; j++) {
      b[j] += B[i][j] * t[i];
      for (let k = 0; k < d; k++) A[j][k] += B[i][j] * B[i][k];
    }
  }
  for (let j = 0; j < d; j++) A[j][j] += 1e-8;
  for (let c = 0; c < d; c++) {
    let piv = c;
    for (let r2 = c + 1; r2 < d; r2++) if (Math.abs(A[r2][c]) > Math.abs(A[piv][c])) piv = r2;
    [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
    const p = A[c][c] || 1e-12;
    for (let r2 = 0; r2 < d; r2++) {
      if (r2 === c) continue;
      const f = A[r2][c] / p;
      for (let k = c; k < d; k++) A[r2][k] -= f * A[c][k];
      b[r2] -= f * b[c];
    }
  }
  return b.map((v, j) => v / (A[j][j] || 1e-12));
}

function dateRecoverability(x) {
  const ok = [], t = [];
  for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i])) { ok.push(x[i]); t.push(i); }
  if (ok.length < 20) return NaN;
  const s = std(ok) + 1e-12;
  if (std(ok) === 0) return NaN;
  const B = ok.map((v) => {
    const row = [v, v * v];
    for (const k of [1, 2, 3]) { row.push(Math.sin(k * v / s), Math.cos(k * v / s)); }
    row.push(1);
    return row;
  });
  const coef = lstsq(B, t);
  const pred = B.map((row) => row.reduce((a, v, j) => a + v * coef[j], 0));
  const tm = mean(t);
  const ssRes = t.reduce((a, v, i) => a + (v - pred[i]) ** 2, 0);
  const ssTot = t.reduce((a, v) => a + (v - tm) ** 2, 0);
  return ssTot > 0 ? 1 - ssRes / ssTot : NaN;
}

function trendStrength(v) {
  const ok = [], t = [];
  for (let i = 0; i < v.length; i++) if (Number.isFinite(v[i])) { ok.push(v[i]); t.push(i); }
  if (ok.length < 20 || std(ok) === 0) return 0;
  const B = t.map((u) => [u, 1]);
  const coef = lstsq(B, ok);
  const m = mean(ok);
  const res = ok.reduce((a, u, i) => a + (u - (coef[0] * t[i] + coef[1])) ** 2, 0);
  const tot = ok.reduce((a, u) => a + (u - m) ** 2, 0);
  return 1 - res / tot;
}

function autocorrelation(x, lag = 1) {
  const xx = finite(x);
  if (xx.length <= lag || std(xx) === 0) return NaN;
  const m = mean(xx);
  let num = 0, den = 0;
  for (let i = 0; i < xx.length - lag; i++) num += (xx[i] - m) * (xx[i + lag] - m);
  for (let i = 0; i < xx.length; i++) den += (xx[i] - m) ** 2;
  return num / den;
}

function declusterTemporal(y, window = 5) {
  const out = new Array(y.length).fill(0);
  let last = -1e9;
  for (let i = 0; i < y.length; i++)
    if (y[i] > 0 && i - last > window) { out[i] = 1; last = i; }
  return out;
}

function declusterPair(x, y, window = 5, n = 100) {
  const yd = declusterTemporal(y, window);
  if (yd.reduce((a, v) => a + v, 0) < 100) return [NaN, NaN];
  const z = (yy) => {
    const nul = finite(shiftNull(x, yy, n, SEED + 2));
    const obs = decileStatistic(x, yy);
    if (nul.length < 10 || !Number.isFinite(obs) || std(nul) === 0) return NaN;
    return (obs - mean(nul)) / std(nul);
  };
  return [z(y), z(yd)];
}

/* In-place iterative radix-2 transform. The naive form is quadratic, which on
 * twenty thousand days is two hundred million trigonometric evaluations and
 * enough to freeze the page on its own. */
function fftPower(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

function spectralConcentration(x, top = 5) {
  const xx = finite(x);
  if (xx.length < 64) return NaN;
  // The largest power-of-two prefix, so a radix-2 transform is exact here and
  // the same truncation is applied in Python. Zero-padding instead would
  // change the frequency grid and with it the value, and the model was trained
  // on the Python number.
  let n = 1;
  while (n * 2 <= xx.length) n *= 2;
  const seg = xx.slice(0, n);
  const m = mean(seg);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = seg[i] - m;
  fftPower(re, im);
  const half = n >> 1;
  const p = new Array(half);
  for (let k = 1; k <= half; k++) p[k - 1] = re[k] * re[k] + im[k] * im[k];
  const tot = p.reduce((a, v) => a + v, 0);
  if (tot <= 0) return NaN;
  const s = p.slice().sort((a, b) => b - a).slice(0, top)
    .reduce((a, v) => a + v, 0);
  return s / tot;
}

/* ---------------------------------------------------------------- entry */
export function fingerprint(x, y) {
  const [p, stat, detect] = shiftP(x, y);
  const [zFull, zDec] = declusterPair(x, y);
  const [aC, aR] = splitSensitivity(x, y);

  const win = 365, ysm = y.map((_, i) => {
    let s = 0, c = 0;
    for (let j = i - Math.floor(win / 2); j < i + Math.ceil(win / 2); j++)
      if (j >= 0 && j < y.length) { s += y[j]; c++; }
    return c ? s / win : 0;
  });

  const h = Math.floor(x.length / 2);
  const s1 = decileStatistic(x.slice(0, h), y.slice(0, h));
  const s2 = decileStatistic(x.slice(h), y.slice(h));
  const halfGap = (Number.isFinite(s1) && Number.isFinite(s2) && s1 + s2 !== 0)
    ? Math.abs(s1 - s2) / (s1 + s2) : NaN;

  const f = {
    stat,
    p_shift: p,
    log_p: Number.isFinite(p) ? -Math.log10(Math.max(p, 1e-4)) : NaN,
    detect_ratio: detect > 0 ? stat / detect : NaN,
    auc_chrono: aC,
    auc_random: aR,
    split_delta: (Number.isFinite(aR) && Number.isFinite(aC)) ? aR - aC : NaN,
    date_r2: dateRecoverability(x),
    half_gap: halfGap,
    ac1: autocorrelation(x),
    trend_x: trendStrength(x),
    trend_y: trendStrength(ysm),
    z_full: zFull,
    z_dec: zDec,
    spectral: spectralConcentration(x)
  };
  return { vector: FEATURES.map((k) => f[k]), detail: f };
}

/* The generator is the one thing that must match bit for bit, so it is
 * checked against the Python stream rather than assumed. */
export function checkRandom() {
  const r = new XorShift32(SEED);
  const got = [r.nextU32(), r.nextU32(), r.nextU32(), r.nextU32(), r.nextU32()];
  const want = [472757172, 3778524601, 3885191513, 985581349, 2836977416];
  return { ok: got.every((v, i) => v === want[i]), got, want };
}
