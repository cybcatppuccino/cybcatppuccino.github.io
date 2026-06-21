import fs from 'node:fs';
import { identifyCubicJS } from './ec_core.js';

const tests = [
  { input: '[3,3]', j: '6912/13', coeffs: '0,0,0,3,3', disc: '-5616' },
  { input: '[0,0,1,3,3]', j: '110592/233', coeffs: '0,0,1,3,3', disc: '-6291' },
  { input: 'y^2 + y = x^3 + 3*x + 3', j: '110592/233', coeffs: '0,0,1,3,3', disc: '-6291' },
  { input: 'x^3 + x^2*y + y^3 + y^2 - 2*x + 1 = 0', j: '110592/233', coeffs: '0,0,1,3,3', disc: '-6291' },
  { input: 'u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0', j: '110592/233', coeffs: '0,0,1,3,3', disc: '-6291' },
  { input: 'X^3 + 2Y^3 + 1 = 15XY', j: '350402625/137842', coeffs: '1,-1,1,-1190,9235', disc: '73254890322' }
];
let ok = true;
for (const t of tests) {
  const r = identifyCubicJS(t.input);
  const gotCoeffs = Array.isArray(r.coeffs_int) ? r.coeffs_int.map(String).join(',') : '';
  const pass = r.ok && r.j === t.j && gotCoeffs === t.coeffs && String(r.discriminant) === t.disc;
  console.log(`${pass ? 'PASS' : 'FAIL'} ${t.input}`);
  if (!pass) { console.log(r); ok = false; }
}
const data = JSON.parse(fs.readFileSync(new URL('../data/curves_by_j.json', import.meta.url), 'utf8'));
for (const j of ['110592/233', '6912/13', '350402625/137842']) {
  const rows = data.by_j[j] || [];
  const pass = rows.length > 0;
  console.log(`${pass ? 'PASS' : 'FAIL'} curve data lookup ${j}: ${rows.length}`);
  if (!pass) ok = false;
}
if (!ok) process.exit(1);
