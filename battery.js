/*
 * The hero for the audit page: seven mechanisms, seven signatures.
 *
 * The point of the tool is that a p-value cannot tell these apart but the
 * response to a battery of perturbations can. So the hero shows exactly that:
 * the fifteen axes as a profile, morphing between one mechanism's signature
 * and the next, labelled as it goes.
 *
 * The profiles are the real class centroids from the training data, shipped in
 * audit_model.json, not shapes invented to look convincing. Each axis is
 * standardised across classes before drawing, otherwise an axis measured in
 * z-scores would swamp one measured in probabilities and the picture would
 * show the units rather than the mechanisms.
 */

const LABEL = {
  REAL: "A real association",
  NULL: "No association",
  CALENDAR: "Calendar leakage",
  CLUSTER: "Aftershock clustering",
  LOOKAHEAD: "Window reaches across the event",
  SEARCH: "Selection over many candidates",
  GROWTH: "Shared trend with catalogue growth"
};
const COLOUR = {
  REAL: "#63b189", NULL: "#8c9ab4", CALENDAR: "#e0a95a",
  CLUSTER: "#e0736d", LOOKAHEAD: "#c46fc4", SEARCH: "#6f9fd8",
  GROWTH: "#5aa8a8"
};
const ORDER = ["REAL", "CALENDAR", "CLUSTER", "LOOKAHEAD", "SEARCH",
               "GROWTH", "NULL"];

export function mountBattery(canvas, label, model) {
  const ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const names = model.features;
  const n = names.length;

  // standardise each axis across the seven centroids, so the drawing compares
  // mechanisms rather than units
  const cols = names.map((_, j) => ORDER.map((c) => model.centroid[c][j])
    .filter((v) => v !== null && Number.isFinite(v)));
  const mu = cols.map((v) => v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);
  const sd = cols.map((v, j) => {
    if (!v.length) return 1;
    const s = Math.sqrt(v.reduce((a, b) => a + (b - mu[j]) ** 2, 0) / v.length);
    return s || 1;
  });
  const profile = (c) => model.centroid[c].map((v, j) =>
    v === null || !Number.isFinite(v) ? 0
      : Math.max(-2.4, Math.min(2.4, (v - mu[j]) / sd[j])));

  const P = Object.fromEntries(ORDER.map((c) => [c, profile(c)]));
  let W = 0, H = 0, i = 0, t = 0, cur = P[ORDER[0]].slice(), raf = 0;

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const pad = 22, mid = H / 2;
    const step = (W - pad * 2) / (n - 1);
    const amp = (H / 2 - 26) / 2.4;

    ctx.strokeStyle = "rgba(255,255,255,.10)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, mid); ctx.lineTo(W - pad, mid); ctx.stroke();

    const from = ORDER[i], to = ORDER[(i + 1) % ORDER.length];
    const k = reduce ? 0 : Math.max(0, Math.min(1, (t - 0.62) / 0.38));
    const e = k * k * (3 - 2 * k);
    const col = k < 0.5 ? COLOUR[from] : COLOUR[to];

    for (let j = 0; j < n; j++) {
      cur[j] = P[from][j] * (1 - e) + P[to][j] * e;
      const x = pad + j * step, y = mid - cur[j] * amp;
      ctx.strokeStyle = "rgba(255,255,255,.07)";
      ctx.beginPath(); ctx.moveTo(x, 22); ctx.lineTo(x, H - 22); ctx.stroke();
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.30;
      ctx.fillRect(x - 3, Math.min(mid, y), 6, Math.abs(mid - y));
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(x, y, 3.1, 0, 7); ctx.fill();
    }

    ctx.strokeStyle = col; ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let j = 0; j < n; j++) {
      const x = pad + j * step, y = mid - cur[j] * amp;
      j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();

    const shown = k < 0.5 ? from : to;
    if (label.dataset.cls !== shown) {
      label.dataset.cls = shown;
      label.innerHTML = `<span class="sig-dot" style="background:${COLOUR[shown]}"></span>
        <strong>${LABEL[shown]}</strong>
        <span class="sky-note">its signature across the fifteen axes</span>`;
    }
  }

  function frame() {
    if (!reduce) {
      t += 0.0075;
      if (t >= 1) { t = 0; i = (i + 1) % ORDER.length; }
    }
    draw();
    raf = requestAnimationFrame(frame);
  }

  addEventListener("resize", () => { resize(); draw(); });
  const io = new IntersectionObserver(([e2]) => {
    cancelAnimationFrame(raf);
    if (e2.isIntersecting) raf = requestAnimationFrame(frame);
  }, { threshold: 0 });
  resize(); draw(); io.observe(canvas);
}

/* Where one measured value sits between the mechanisms, as a position on a
 * line rather than a number the reader has to rank in their head. */
export function axisScale(model, j, value) {
  const vals = model.classes
    .map((c) => ({ c, v: model.centroid[c][j] }))
    .filter((q) => q.v !== null && Number.isFinite(q.v));
  if (!vals.length || !Number.isFinite(value)) return null;
  const lo = Math.min(value, ...vals.map((q) => q.v));
  const hi = Math.max(value, ...vals.map((q) => q.v));
  const span = hi - lo || 1;
  return {
    marks: vals.map((q) => ({ ...q, pct: (100 * (q.v - lo)) / span })),
    pct: (100 * (value - lo)) / span
  };
}
