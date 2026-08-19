/*
 * The thirteen candidates as cards, and the stress ladder underneath them.
 *
 * The glyphs are drawn rather than listed because the reader recognises a
 * crescent or a droplet faster than they read "lunar phase". They stay small
 * and monochrome so they label the row instead of competing with the number,
 * which is what the card actually exists to show.
 *
 * The ladder animates on first sight only, and not at all when the visitor has
 * asked for reduced motion. Bar length is log-scaled, because the range spans
 * thirteen orders of magnitude and a linear axis would show one bar.
 */

const G = {
  moon: '<path d="M15.6 3.6a8.4 8.4 0 1 0 4.8 12.9A9 9 0 0 1 15.6 3.6z"/>',
  tide: '<path d="M2 15c3 0 3-3 6-3s3 3 6 3 3-3 6-3M2 20c3 0 3-3 6-3s3 3 6 3 3-3 6-3" fill="none" stroke="currentColor" stroke-width="1.7"/><circle cx="18" cy="6" r="3.2"/>',
  bvalue: '<path d="M3 20h18M5 20V9m5 11V4m5 16v-8m5 8V7" fill="none" stroke="currentColor" stroke-width="1.9"/>',
  planet: '<circle cx="12" cy="12" r="5.4"/><ellipse cx="12" cy="12" rx="10.6" ry="3.6" fill="none" stroke="currentColor" stroke-width="1.5" transform="rotate(-22 12 12)"/>',
  sun: '<circle cx="12" cy="12" r="4.6"/><g stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2.1 2.1M16.9 16.9L19 19M19 5l-2.1 2.1M7.1 16.9L5 19"/></g>',
  water: '<path d="M12 2.5S5 10.4 5 14.6a7 7 0 0 0 14 0C19 10.4 12 2.5 12 2.5z"/>',
  rotation: '<circle cx="12" cy="12" r="8.4" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M3.6 12h16.8M12 3.6c2.6 2.4 2.6 14.4 0 16.8-2.6-2.4-2.6-14.4 0-16.8z" fill="none" stroke="currentColor" stroke-width="1.4"/>',
  air: '<path d="M3 8h11a3 3 0 1 0-3-3M3 13h15a3 3 0 1 1-3 3M3 18h8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  iono: '<path d="M12 18a10 10 0 0 1 0-12M12 18a15 15 0 0 0 0-12" fill="none" stroke="currentColor" stroke-width="1.6"/><ellipse cx="12" cy="12" rx="10" ry="4.2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="2.1"/>',
  olr: '<circle cx="12" cy="7" r="3.4"/><path d="M4 15h16M6 19h12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  temp: '<path d="M13.6 13.4V4.6a1.9 1.9 0 1 0-3.8 0v8.8a4.2 4.2 0 1 0 3.8 0z" fill="none" stroke="currentColor" stroke-width="1.7"/><circle cx="11.7" cy="17.4" r="1.9"/>',
  vapour: '<path d="M6 16a4 4 0 0 1 .6-8 5.4 5.4 0 0 1 10.3 1.4A3.4 3.4 0 0 1 17 16z" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M8 20h.01M12 20h.01M16 20h.01" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
  radon: '<circle cx="12" cy="12" r="2"/><g fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="12" rx="10" ry="4"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)"/></g>',
  quake: '<path d="M2 12h3.4l2.4-7 3.4 14 3-10.4 2.2 3.4H22" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/>'
};

const CARDS = [
  ["moon", "Lunar phase and geometry", "0.615", "13 variables · 2.57% measured"],
  ["tide", "Tidal stress, phase", "0.788", "on each event's own fault plane"],
  ["bvalue", "Tidal stress, size-frequency", "0.676", "does b shift with amplitude?"],
  ["planet", "Planetary geometry", "0.590", "80 variables · widest threshold"],
  ["sun", "Solar activity", "0.843", "Kp, ap, sunspot, F10.7, solar wind"],
  ["water", "Hydrological loading", "0.057", "GRACE, 90% event coverage"],
  ["rotation", "Earth rotation", "0.460", "length of day, polar motion"],
  ["air", "Atmospheric loading", "0.107", "best-powered test in the study"],
  ["iono", "Ionospheric TEC", "0.903", "1,907 mainshocks, causal sampling"],
  ["olr", "Outgoing longwave radiation", "0.173", "1974 to 2022, 1,900 cells"],
  ["temp", "Near-surface air temperature", "0.253", "the surface form of the claim"],
  ["vapour", "Precipitable water", "0.890", "99.4% event coverage"],
  ["radon", "Soil-gas radon", null, "no global archive exists"],
  ["quake", "Recent seismicity (ETAS)", "0.005", "9.19 points of modulation", true]
];

const LADDER = [
  ["Neighbouring earthquake", 1e6, "10⁵–10⁷ Pa", true],
  ["Ocean tidal loading", 1e4, "~10⁴ Pa", true],
  ["Water load at epicentres", 1130, "1,130 Pa"],
  ["Moon, body tide", 723, "723 Pa"],
  ["Atmospheric pressure anomaly", 563, "563 Pa"],
  ["Venus, with Earth rotation", 3.9e-3, "0.0039 Pa"],
  ["Jupiter, with Earth rotation", 1.1e-3, "0.0011 Pa"],
  ["Neptune", 5.9e-7, "5.9 × 10⁻⁷ Pa"]
];

function icon(k) {
  return `<svg class="glyph" width="22" height="22" viewBox="0 0 24 24"
    fill="currentColor" aria-hidden="true">${G[k]}</svg>`;
}

export function mountCards(el) {
  el.innerHTML = CARDS.map(([g, name, p, sub, ref]) => `
    <div class="card${ref ? " ref" : ""}">
      ${icon(g)}
      <h4>${name}</h4>
      <div class="p${ref ? " hit" : ""}">${p === null ? "not testable" : "p = " + p}</div>
      <div class="sub">${sub}</div>
    </div>`).join("");
}

export function mountLadder(el) {
  const lo = Math.log10(5.9e-7), hi = Math.log10(1e6);
  el.innerHTML = LADDER.map(([name, pa, label, warm]) => `
    <div class="ladder-row${warm ? " warm" : ""}">
      <div>${name}</div>
      <div class="ladder-bar"><i data-w="${
        (100 * (Math.log10(pa) - lo) / (hi - lo)).toFixed(1)}"></i></div>
      <div class="ladder-val">${label}</div>
    </div>`).join("");

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fill = () => el.querySelectorAll(".ladder-bar > i")
    .forEach((b) => { b.style.width = b.dataset.w + "%"; });

  if (reduce) { fill(); return; }
  new IntersectionObserver((entries, obs) => {
    if (entries[0].isIntersecting) { fill(); obs.disconnect(); }
  }, { threshold: 0.25 }).observe(el);
}
