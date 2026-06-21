import { identifyCubicJS } from './ec_core.js';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve('..');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
if (app.includes('curves_compact.json')) throw new Error('app.js must not depend on curves_compact.json');
for (const needle of ['function drawSkyGrid(ctx, fast = false)', 'function conductorSanity', 'function updateSelectionMarker', 'selection-marker']) {
  if (!app.includes(needle)) throw new Error(`missing ${needle}`);
}

const cases = [
  ['[0,0,1,3,3]', '110592/233'],
  ['y^2 + y = x^3 + 3*x + 3', '110592/233'],
  ['u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0', '110592/233'],
  ['[3,3]', '6912/13'],
];
for (const [input, expectedJ] of cases) {
  const res = identifyCubicJS(input);
  if (!res?.ok) throw new Error(`failed to identify ${input}: ${res?.error}`);
  if (String(res.j) !== expectedJ) throw new Error(`bad j for ${input}: ${res.j} !== ${expectedJ}`);
  console.log(input, res.j, res.weierstrass || '');
}

const tau = JSON.parse(fs.readFileSync(path.join(root, 'data/tau_index.json'), 'utf8'));
const cols = tau.columns;
const rows = tau.rows.map(r => Object.fromEntries(cols.map((c, i) => [c, r[i]])));
const curve272 = rows.find(r => r.label === '272.b1');
if (!curve272) throw new Error('missing tau row for 272.b1');
for (const label of ['272.b2', '272.b3', '272.b4']) {
  if (!rows.find(r => r.label === label && r.iso === curve272.iso)) throw new Error(`missing same-class row ${label}`);
}
console.log('v22 smoke checks passed');
