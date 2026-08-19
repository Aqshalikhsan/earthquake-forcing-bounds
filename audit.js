/*
 * The audit tool: fingerprint a claim, then read the classifier's answer as a
 * conformal prediction set rather than as a verdict.
 *
 * Everything runs here, in the page. Nothing is uploaded, because nothing has
 * to be: the fingerprint is cheap arithmetic and the model is a tree ensemble
 * exported from Python, verified to agree with scikit-learn exactly.
 */

import { fingerprint, FEATURES, checkRandom } from "./fingerprint.js";
import { axisScale } from "./battery.js";
import { realEvents } from "./events.js";

const COLOUR = {
  REAL: "#2f6b4a", NULL: "#8a8a8a", CALENDAR: "#b8862f",
  CLUSTER: "#b02a2a", LOOKAHEAD: "#8e4a99", SEARCH: "#2a5d9e",
  GROWTH: "#2f7d7d"
};

/* The seven explanations, listed on the page so a reader knows what the tool
 * is choosing between before they run anything. */
export function mountMechs(el) {
  if (!el) return;
  el.innerHTML = Object.entries(CLASS_TEXT).map(([c, [name, why]]) => `
    <div class="mech">
      <h4><span class="dot" style="background:${COLOUR[c]}"></span>${name}</h4>
      <p>${why}</p>
    </div>`).join("");
}

const CLASS_TEXT = {
  REAL: ["Consistent with a real association",
    "The evidence survives the perturbations that dissolve the known artefacts. This is the weakest of the possible answers, by design: the tool is calibrated to keep this label whenever it is even plausible."],
  NULL: ["Consistent with no association",
    "Nothing beyond what a shifted copy of the same series produces."],
  CALENDAR: ["Calendar leakage",
    "The predictor reconstructs the date, so a randomly split model can locate a test day in time and recall the local rate without using the predictor's physics at all. Refit under a chronological split."],
  CLUSTER: ["Aftershock clustering",
    "The association is carried by days that follow other events, so one sequence is being counted many times. Decluster the catalogue and retest."],
  LOOKAHEAD: ["Window reaches across the event",
    "The predictor at a given day contains information from days at or after the event. Make the sampling strictly causal and retest."],
  SEARCH: ["Selection over many candidates",
    "The result looks like the best of a search rather than a single test. Apply a max-statistic correction over everything that was tried, including lags and window widths."],
  GROWTH: ["Shared trend with catalogue growth",
    "Both series ride the same secular trend, which instrumentation produces without any physics. Restrict to a magnitude range where the catalogue rate is flat."]
};

const AXIS_TEXT = {
  stat: ["Modulation", "largest decile deviation from the base rate"],
  p_shift: ["p, circular-shift null", "uncorrected"],
  log_p: ["−log₁₀ p", ""],
  detect_ratio: ["Statistic / detectable", "above 1 clears the null band"],
  auc_chrono: ["AUC, chronological split", ""],
  auc_random: ["AUC, random split", ""],
  split_delta: ["Split sensitivity", "0.008 for real signal, 0.208 for a calendar artefact"],
  date_r2: ["Date recoverable from predictor", "R²; 0.99 in the forcing bank"],
  half_gap: ["Disagreement between halves", "0 is perfect stability"],
  ac1: ["Predictor autocorrelation", "lag one day"],
  trend_x: ["Trend in predictor", "R² against time"],
  trend_y: ["Trend in event rate", "R² against time"],
  z_full: ["Evidence, full series", "z against its own null"],
  z_dec: ["Evidence, after thinning", "z against its own null"],
  spectral: ["Spectral concentration", "0.6 and above marks a calendar function"]
};

let MODEL = null;

const $ = (id) => document.getElementById(id);

async function model() {
  if (!MODEL) {
    const r = await fetch("audit_model.json");
    MODEL = await r.json();
  }
  return MODEL;
}

/* ------------------------------------------------------ model evaluation */
function walk(tree, x) {
  let i = 0;
  while (!tree.leaf[i]) {
    const v = x[tree.feature[i]];
    if (!Number.isFinite(v)) i = tree.missing_left[i] ? tree.left[i] : tree.right[i];
    else i = v <= tree.threshold[i] ? tree.left[i] : tree.right[i];
  }
  return tree.value[i];
}

function proba(m, x) {
  const z = m.baseline.slice();
  for (const stage of m.trees)
    for (let k = 0; k < stage.length; k++) z[k] += walk(stage[k], x);
  const mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, v) => a + v, 0);
  return e.map((v) => v / s);
}

function predictionSet(m, p) {
  const out = [];
  for (let j = 0; j < m.classes.length; j++) {
    const other = Math.max(...p.filter((_, k) => k !== j));
    if (other - p[j] <= m.quantiles[m.classes[j]]) out.push(m.classes[j]);
  }
  return out;
}

function mahalanobis(m, x) {
  const mu = m.ood.mu, inv = m.ood.inv;
  const d = x.map((v, j) => (Number.isFinite(v) ? v : mu[j]) - mu[j]);
  let s = 0;
  for (let i = 0; i < d.length; i++)
    for (let j = 0; j < d.length; j++) s += d[i] * inv[i][j] * d[j];
  return Math.sqrt(Math.max(s, 0));
}

/* --------------------------------------------------------------- parsing */
function parse(text) {
  const xs = [], ys = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /[a-df-zA-DF-Z]/.test(line[0])) continue;   // skip a header
    const parts = line.split(/[,;\t ]+/).filter(Boolean);
    if (parts.length < 2) continue;
    const a = Number(parts[0]), b = Number(parts[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    xs.push(a); ys.push(b > 0 ? 1 : 0);
  }
  return [xs, ys];
}

/* ------------------------------------------------------------- examples */
/*
 * Built on the real catalogue, not on a random target.
 *
 * The first version of these used a synthetic event series, and every example
 * landed outside the range the model was calibrated on: the tool refused to
 * answer, which looks like a fault rather than the correct behaviour it is.
 * The generators below mirror src/audit/generate.py, so an example lands where
 * the training examples of that class do.
 */
function xorshift(seed) {
  let x = seed >>> 0;
  return () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
                 return x / 4294967296; };
}

function smooth(n, scale, r) {
  const w = Array.from({ length: n }, () => r() * 2 - 1);
  const h = Math.floor(scale / 2);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let j = i - h; j <= i + h; j++) if (j >= 0 && j < n) { s += w[j]; c++; }
    out[i] = s / c;
  }
  const m = out.reduce((a, v) => a + v, 0) / n;
  const sd = Math.sqrt(out.reduce((a, v) => a + (v - m) ** 2, 0) / n) || 1;
  return out.map((v) => (v - m) / sd);
}

function topDecileIdx(x, frac) {
  return x.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
    .slice(0, Math.floor(x.length * frac)).map((q) => q[1]);
}

export function example(kind) {
  const y0 = realEvents();
  const n = y0.length;
  const r = xorshift(20260819);
  let x, y = y0.slice();

  if (kind === "real") {
    // a modulation of known size written into the event times, exactly as
    // inject_real does: the count is preserved, only the timing moves
    x = smooth(n, 30, r);
    const top = topDecileIdx(x, 0.2);
    const ev = [];
    for (let i = 0; i < n; i++) if (y[i]) ev.push(i);
    const k = Math.floor(ev.length * 0.30);
    for (let i = 0; i < k; i++) {
      y[ev[Math.floor(r() * ev.length)]] = 0;
      y[top[Math.floor(r() * top.length)]] = 1;
    }

  } else if (kind === "null") {
    x = smooth(n, 30, r);
    const k = 3000 + Math.floor(r() * 4000);
    y = y0.map((_, i) => y0[(((i - k) % n) + n) % n]);

  } else if (kind === "calendar") {
    // a deterministic function of the date, against the untouched catalogue
    const period = 700 + r() * 2000, phase = r() * 6.283;
    x = Array.from({ length: n }, (_, i) =>
      Math.sin((2 * Math.PI * i) / period + phase)
      + 0.3 * Math.sin((4 * Math.PI * i) / (period * 1.7) + phase)
      + (r() - 0.5) * 0.3);

  } else if (kind === "lookahead") {
    // the predictor window reaches forward across the event
    const k = 6 + Math.floor(r() * 12);
    x = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = i; j < Math.min(i + k, n); j++) s += y0[j];
      x[i] = s / k + (r() - 0.5) * 1.4;
    }

  } else {                                    // cluster
    x = smooth(n, 60, r);
    const idx = topDecileIdx(x, 0.3);
    const hot = new Set(idx);
    y = new Array(n).fill(0);
    const main = [];
    for (let i = 0; i < n; i++) if (r() < 0.02) { y[i] = 1; main.push(i); }
    for (const mIdx of main) {
      if (!hot.has(mIdx)) continue;
      for (let d = 1; d < 40; d++)
        if (r() < 1 / Math.pow(d, 0.75) && mIdx + d < n) y[mIdx + d] = 1;
    }
  }
  return x.map((v, i) => `${v.toFixed(5)},${y[i]}`).join("\n");
}

/* ---------------------------------------------------------------- render */
function render(m, vec, detail) {
  const p = proba(m, vec);
  const set = predictionSet(m, p);
  const d = mahalanobis(m, vec);
  const ood = d > m.ood.thresh;

  const artefacts = set.filter((c) => c !== "REAL" && c !== "NULL");
  let cls = "unclear", head = "No confident answer", body = "";

  if (ood) {
    head = "Outside the calibrated range";
    body = `This fingerprint sits further from everything the model was trained on than 99% of the training data (Mahalanobis ${d.toFixed(1)} against a threshold of ${m.ood.thresh.toFixed(1)}). No verdict is given, because a classifier asked about an input of a kind it has never seen will answer confidently and wrongly.`;
  } else if (set.length === 0) {
    body = "No class is consistent with this fingerprint at the stated confidence. Treat that as a warning about the input rather than as a result.";
  } else if (artefacts.length && !set.includes("REAL")) {
    cls = "artefact";
    head = artefacts.length === 1
      ? `Artefact: ${CLASS_TEXT[artefacts[0]][0].toLowerCase()}`
      : "Artefact, mechanism not resolved";
    body = artefacts.map((c) => `<strong>${CLASS_TEXT[c][0]}.</strong> ${CLASS_TEXT[c][1]}`).join("<br><br>");
  } else if (set.length === 1 && set[0] === "NULL") {
    cls = "clean";
    head = "Consistent with no association";
    body = CLASS_TEXT.NULL[1];
  } else if (set.length === 1 && set[0] === "REAL") {
    cls = "clean";
    head = "Survives the battery";
    body = CLASS_TEXT.REAL[1];
  } else {
    body = "More than one explanation remains consistent with the evidence: "
      + set.map((c) => `<strong>${CLASS_TEXT[c][0]}</strong>`).join(", ")
      + ". The tool is deliberately reluctant to rule out a real association, so a set of this kind means the data do not settle the question, not that something is wrong.";
  }

  const order = m.classes.map((c, j) => [c, p[j]]).sort((a, b) => b[1] - a[1]);

  const probs = order.map(([c, v]) => `
    <div class="prob${set.includes(c) ? " inset" : ""}">
      <div><span class="dot" style="display:inline-block;width:9px;height:9px;
        border-radius:50%;margin-right:7px;background:${COLOUR[c]}"></span>${CLASS_TEXT[c][0]}</div>
      <div class="prob-bar"><i data-w="${(100 * v).toFixed(1)}"></i></div>
      <div class="prob-val">${(100 * v).toFixed(1)}%</div>
    </div>`).join("");

  // each axis as a position between the mechanisms rather than a bare number,
  // so the reader can see which signature the claim is nearest on that axis
  const axes = FEATURES.map((k, j) => {
    const v = detail[k];
    const sc = axisScale(m, j, v);
    const marks = sc ? sc.marks.map((q) =>
      `<span class="mk" style="left:${q.pct.toFixed(1)}%;background:${COLOUR[q.c]}"
         title="${CLASS_TEXT[q.c][0]}"></span>`).join("") : "";
    return `<div class="axis">
      <div class="axis-head"><b>${AXIS_TEXT[k][0]}</b>
        <span class="v">${Number.isFinite(v) ? v.toFixed(4) : "n/a"}</span></div>
      ${sc ? `<div class="axis-line"><span class="rail"></span>${marks}
        <span class="me" style="left:${sc.pct.toFixed(1)}%"></span></div>` : ""}
      <div class="axis-note">${AXIS_TEXT[k][1] || "&nbsp;"}</div>
    </div>`;
  }).join("");

  $("out").innerHTML = `
    <div class="verdict ${cls}"><h4>${head}</h4><p>${body}</p></div>

    <h3>Consistent labels at the stated confidence</h3>
    <p class="note">Green bars are the labels inside the conformal set. REAL is
    held to 99% coverage and every artefact class to 90%, on purpose: the tool
    can refute far more confidently than it can endorse, so a set of artefact
    labels alone is strong evidence while a set containing REAL is weak
    evidence of anything.</p>
    ${probs}

    <h3>The fingerprint, against each mechanism's own signature</h3>
    <p class="note">The blue mark is your claim. The coloured ticks are where
    each mechanism sits on that axis, taken from the training centroids.</p>
    <div class="panel">${axes}</div>`;

  requestAnimationFrame(() => $("out").querySelectorAll(".prob-bar > i")
    .forEach((b) => { b.style.width = b.dataset.w + "%"; }));
}

/* ------------------------------------------------------------------ wire */
async function run() {
  const [x, y] = parse($("data").value);
  if (x.length < 400) {
    $("out").innerHTML = `<div class="verdict unclear"><h4>Not enough data</h4>
      <p>Read ${x.length} usable rows. The battery needs at least 400 days, and
      several axes only become meaningful well above that.</p></div>`;
    return;
  }
  const ev = y.reduce((a, v) => a + v, 0);
  if (ev < 100 || ev === y.length) {
    $("out").innerHTML = `<div class="verdict unclear"><h4>Event column unusable</h4>
      <p>Found ${ev} event days in ${y.length} rows. The second column must be
      1 on days an event occurred and 0 otherwise, with at least 100 of them.</p></div>`;
    return;
  }
  $("run").disabled = true;
  $("out").innerHTML = `<p class="note">Fingerprinting ${x.length} days across
    15 axes. This runs in your browser and takes a few seconds.</p>`;
  await new Promise((r) => setTimeout(r, 30));
  try {
    const m = await model();
    const { vector, detail } = fingerprint(x, y);
    render(m, vector, detail);
  } catch (err) {
    // Without this the page kept the "fingerprinting" notice forever and gave
    // the reader nothing to act on. A failure has to say what failed.
    $("out").innerHTML = `<div class="verdict artefact">
      <h4>The audit could not finish</h4>
      <p>${String(err && err.message ? err.message : err)}</p>
      <p class="note">If this followed an update, a hard reload clears a stale
      copy of the model bundle. Otherwise the browser console carries the full
      trace.</p></div>`;
    console.error(err);
  } finally {
    $("run").disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("run").addEventListener("click", run);
  document.querySelectorAll("[data-example]").forEach((b) =>
    b.addEventListener("click", () => {
      $("data").value = example(b.dataset.example);
      $("out").innerHTML = "";
    }));
  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) $("data").value = await f.text();
  });
  const c = checkRandom();
  if (!c.ok) $("out").innerHTML =
    `<div class="verdict artefact"><h4>Build problem</h4><p>The random stream in
     this browser does not match the one the model was calibrated against, so
     results would not be trustworthy.</p></div>`;
});
