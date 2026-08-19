/*
 * The hero: the suspects, drawn to the scale that actually matters.
 *
 * Every body here is sized and placed by the stress it exerts on a fault, not
 * by its mass or its distance. That is the whole argument of the study in one
 * picture: the Moon is worth 723 Pa, Venus 0.0039 Pa, and an earthquake next
 * door 10^7 Pa. Drawing planets for decoration would make this look like the
 * astrology it examines; drawing them with their pascals attached makes the
 * decoration carry the finding.
 *
 * Orbital radius is log-scaled on stress, so the ordering is honest while
 * still fitting on a screen. Nothing is animated when the visitor has asked
 * for reduced motion, and the loop stops when the canvas leaves the viewport.
 */

const BODIES = [
  { name: "Neighbouring earthquake", pa: 1e6, r: 13, colour: "#c85a4a",
    note: "stress transfer from a nearby rupture", period: 41 },
  { name: "Ocean tidal loading", pa: 1e4, r: 10, colour: "#4f86c6",
    note: "sea-floor load, the largest periodic term", period: 29 },
  { name: "Water load (GRACE)", pa: 1130, r: 8.5, colour: "#5aa0a8",
    note: "measured at earthquake locations", period: 53 },
  { name: "Moon, body tide", pa: 723, r: 8, colour: "#d8d4c8",
    note: "the oldest of the claims", period: 19 },
  { name: "Atmospheric pressure", pa: 563, r: 7.5, colour: "#9aa7b8",
    note: "the best-powered test in the study", period: 34 },
  { name: "Venus", pa: 3.9e-3, r: 5.5, colour: "#d9b878",
    note: "the strongest planetary tide", period: 62 },
  { name: "Jupiter", pa: 1.1e-3, r: 6.5, colour: "#c99a6a",
    note: "largest planet, still negligible", period: 77 },
  { name: "Neptune", pa: 5.9e-7, r: 4.5, colour: "#7d95c4",
    note: "a billion times below the Moon", period: 95 }
];

function fmt(pa) {
  if (pa >= 1e5) return `10^${Math.round(Math.log10(pa))} Pa`;
  if (pa >= 1) return `${pa.toLocaleString("en")} Pa`;
  return `${pa.toExponential(1).replace("e-", " × 10⁻")} Pa`;
}

export function mountSky(canvas, readout) {
  const ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, dpr = 1, stars = [], t = 0, raf = 0, hover = null;

  // radius on screen is log-scaled in stress: the weakest bodies sit furthest
  // out, which is the correct ordering and also the readable one
  const lo = Math.log10(5.9e-7), hi = Math.log10(1e6);
  const orbitOf = (pa) => 0.20 + 0.80 * (1 - (Math.log10(pa) - lo) / (hi - lo));

  function resize() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    stars = Array.from({ length: Math.round((W * H) / 5200) }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      r: Math.random() * 1.05 + 0.25,
      a: Math.random() * 0.5 + 0.16,
      s: Math.random() * 0.9 + 0.25
    }));
  }

  function positions() {
    const cx = W / 2, cy = H / 2;
    const span = Math.min(W, H) * 0.46;
    return BODIES.map((b, i) => {
      const ang = (t / b.period) + (i * 2.399);       // golden angle, no clumps
      const rad = orbitOf(b.pa) * span;
      return { ...b, x: cx + Math.cos(ang) * rad * 1.55,
               y: cy + Math.sin(ang) * rad * 0.72, rad };
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2;

    for (const s of stars) {
      const tw = reduce ? 1 : 0.72 + 0.28 * Math.sin(t * s.s + s.x);
      ctx.globalAlpha = s.a * tw;
      ctx.fillStyle = "#dfe6f2";
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;

    const pts = positions();
    ctx.lineWidth = 1;
    for (const p of pts) {
      ctx.strokeStyle = p === hover ? "rgba(255,255,255,.22)"
                                    : "rgba(255,255,255,.07)";
      ctx.beginPath();
      ctx.ellipse(cx, cy, p.rad * 1.55, p.rad * 0.72, 0, 0, 7);
      ctx.stroke();
    }

    // Earth
    const g = ctx.createRadialGradient(cx - 3, cy - 3, 1, cx, cy, 15);
    g.addColorStop(0, "#4f7fbe"); g.addColorStop(1, "#1d3557");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, 13, 0, 7); ctx.fill();

    for (const p of pts) {
      if (p === hover) {
        ctx.fillStyle = "rgba(255,255,255,.10)";
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r + 9, 0, 7); ctx.fill();
      }
      ctx.fillStyle = p.colour;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function frame() {
    if (!reduce) t += 0.0055;
    draw();
    raf = requestAnimationFrame(frame);
  }

  function pick(ev) {
    const b = canvas.getBoundingClientRect();
    const mx = ev.clientX - b.left, my = ev.clientY - b.top;
    let best = null, bd = 26;
    for (const p of positions()) {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bd) { bd = d; best = p; }
    }
    hover = best ? BODIES.find((q) => q.name === best.name) : null;
    canvas.style.cursor = hover ? "pointer" : "default";
    show(hover);
  }

  function show(b) {
    if (!b) {
      readout.innerHTML = `<span class="sky-hint">Hover a body. Each one is
        placed by the stress it puts on a fault, not by its size or its
        distance.</span>`;
      return;
    }
    const vsMoon = 723 / b.pa;
    const rel = b.pa >= 723
      ? `${(b.pa / 723).toLocaleString("en", { maximumSignificantDigits: 2 })}× the lunar body tide`
      : `${vsMoon.toLocaleString("en", { maximumSignificantDigits: 2 })}× weaker than the lunar body tide`;
    readout.innerHTML =
      `<strong>${b.name}</strong><span class="sky-pa">${fmt(b.pa)}</span>
       <span class="sky-note">${b.note} · ${rel}</span>`;
  }

  const io = new IntersectionObserver(([e]) => {
    cancelAnimationFrame(raf);
    if (e.isIntersecting) raf = requestAnimationFrame(frame);
  }, { threshold: 0 });

  addEventListener("resize", () => { resize(); draw(); });
  canvas.addEventListener("mousemove", pick);
  canvas.addEventListener("mouseleave", () => { hover = null; show(null); });
  resize(); show(null); io.observe(canvas);
}
