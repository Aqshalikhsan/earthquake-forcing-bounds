/*
 * Recompute the Python fingerprints here and report the largest disagreement.
 *
 * Run src/audit/crosscheck.py first; it writes crosscheck.json with the test
 * series and the reference vectors. Anything above a few times machine epsilon
 * on a bounded axis means the port has drifted and the calibration measured in
 * Python no longer describes what the browser does.
 */

import { readFileSync } from "node:fs";
import { fingerprint, checkRandom, FEATURES } from "./fingerprint.js";

const rnd = checkRandom();
console.log(`xorshift stream matches Python : ${rnd.ok}`);
if (!rnd.ok) process.exit(1);

const data = JSON.parse(readFileSync(new URL("./crosscheck.json", import.meta.url)));
const worst = new Map(FEATURES.map((f) => [f, 0]));
let worstOverall = 0;

for (const [name, c] of Object.entries(data.cases)) {
  const { vector } = fingerprint(c.x, c.y);
  const line = [];
  for (let j = 0; j < FEATURES.length; j++) {
    const py = c.python[j];
    const js = vector[j];
    if (py === null || js === null || !Number.isFinite(js)) {
      line.push(`${FEATURES[j]}:both-nan`);
      continue;
    }
    const d = Math.abs(py - js);
    if (d > worst.get(FEATURES[j])) worst.set(FEATURES[j], d);
    if (d > worstOverall) worstOverall = d;
  }
  console.log(`  ${name.padEnd(9)} compared ${FEATURES.length} axes`);
}

console.log("\n  axis            max |python - js|");
console.log("  " + "-".repeat(36));
for (const [f, d] of worst) {
  const flag = d > 1e-6 ? "   DRIFT" : "";
  console.log(`  ${f.padEnd(15)} ${d.toExponential(3)}${flag}`);
}
console.log(`\n  worst overall ${worstOverall.toExponential(3)}`);
process.exit(worstOverall > 1e-6 ? 1 : 0);
