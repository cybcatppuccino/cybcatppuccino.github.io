import fs from 'node:fs';
import path from 'node:path';
import { identifyCubicJS } from './ec_core.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
if (app.includes('curves_compact.json')) throw new Error('app.js must not depend on curves_compact.json');
if (!app.includes('pointIdsByTile')) throw new Error('tile-index render path missing');
if (!app.includes('drawFastDot')) throw new Error('fast interaction render path missing');
if (!app.includes('89.7 * RAD')) throw new Error('zenith-centered initial camera missing');

const tests = [
  ['[0,0,1,3,3]', '110592/233'],
  ['y^2 + y = x^3 + 3*x + 3', '110592/233'],
  ['u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0', '110592/233'],
  ['[3,3]', '6912/13'],
];
for (const [input, expected] of tests) {
  const got = identifyCubicJS(input);
  if (!got.ok || got.j !== expected) throw new Error(`${input}: expected ${expected}, got ${got.j}`);
  console.log(input, got.j, got.weierstrass || 'j-only');
}
console.log('v21 smoke checks passed');
