/*
 * The audit tool: fingerprint a claim, then read the classifier's answer as a
 * conformal prediction set rather than as a verdict.
 *
 * Everything runs here, in the page. Nothing is uploaded, because nothing has
 * to be: the fingerprint is cheap arithmetic and the model is a tree ensemble
 * exported from Python, verified to agree with scikit-learn exactly.
 */

import { fingerprint, FEATURES, checkRandom } from "./fingerprint.js";

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
function xorshift(seed) {
  let x = seed >>> 0;
  return () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; };
}

function example(kind) {
  const n = 6000, r = xorshift(99), x = [], y = [];
  const w = [];
  for (let i = 0; i < n; i++) w.push(r() * 2 - 1);
  const smooth = w.map((_, i) => {
    let s = 0, c = 0;
    for (let j = i - 12; j <= i + 12; j++) if (j >= 0 && j < n) { s += w[j]; c++; }
    return s / c;
  });
  for (let i = 0; i < n; i++) {
    x.push(kind === "calendar"
      ? Math.sin((2 * Math.PI * i) / 733) + 0.3 * Math.sin((2 * Math.PI * i) / 297)
      : smooth[i] * 6);
    y.push(r() < 0.29 ? 1 : 0);
  }
  if (kind === "real") {                      // move a quarter of events high
    const order = x.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const top = order.slice(0, Math.floor(n * 0.2)).map((q) => q[1]);
    const ev = [];
    for (let i = 0; i < n; i++) if (y[i]) ev.push(i);
    for (let k = 0; k < Math.floor(ev.length * 0.25); k++) {
      y[ev[k]] = 0;
      y[top[Math.floor(r() * top.length)]] = 1;
    }
  }
  if (kind === "lookahead") {                 // predictor peeks forward
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = i; j < Math.min(i + 8, n); j++) s += y[j];
      x[i] = s / 8 + (r() - 0.5) * 0.8;
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

  $("out").innerHTML = `
    <div class="verdict ${cls}"><h4>${head}</h4><p>${body}</p></div>

    <h3>Consistent labels at the stated confidence</h3>
    <p class="note">REAL is held to 99% coverage and every artefact class to
    90%, on purpose. The tool can refute far more confidently than it can
    endorse, so a set containing only artefact labels is strong evidence while
    a set containing REAL is weak evidence of anything.</p>
    <div class="scroll"><table>
      <thead><tr><th>Class</th><th class="n">Model probability</th><th>In the set</th></tr></thead>
      <tbody>${order.map(([c, v]) => `
        <tr><td>${CLASS_TEXT[c][0]}</td><td class="n">${(100 * v).toFixed(1)}%</td>
        <td>${set.includes(c) ? "<span class=\"tag null\">yes</span>" : ""}</td></tr>`).join("")}
      </tbody></table></div>

    <h3>The fingerprint</h3>
    <div class="scroll"><table>
      <thead><tr><th>Axis</th><th class="n">Value</th><th>Reference scale</th></tr></thead>
      <tbody>${FEATURES.map((k) => {
        const v = detail[k];
        return `<tr><td>${AXIS_TEXT[k][0]}</td>
          <td class="n">${Number.isFinite(v) ? v.toFixed(4) : "n/a"}</td>
          <td class="note">${AXIS_TEXT[k][1]}</td></tr>`;
      }).join("")}</tbody></table></div>`;
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
