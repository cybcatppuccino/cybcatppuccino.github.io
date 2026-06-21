import { identifyCubicJS } from './ec_core.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
function readJSON(rel){ return JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8')); }
const meta = readJSON('data/plot_meta.json');
const top = readJSON('data/top_points.json');
const sampleTile = readJSON('data/tiles/a00_q00.json');
const curves = readJSON('data/curves_compact.json');
console.log('meta tiles', meta.tiles.length, 'top', top.points.length, 'tile', sampleTile.points.length, 'curves', curves.rows.length);
if (meta.tiles.length !== 360) throw new Error('expected 360 tiles');
if (!top.points.length || !sampleTile.points.length) throw new Error('point data missing');
if (curves.rows.length !== 64687) throw new Error('curve db count mismatch');
const columns = curves.columns;
const rows = curves.rows.map(arr => Object.fromEntries(columns.map((c,i)=>[c,arr[i]])));
function findByLabel(label){ return rows.find(r => r.label === label); }
function findByJ(j){ return rows.filter(r => r.j_str === j).slice(0, 5); }
const r6291 = findByLabel('6291.d1');
if (!r6291) throw new Error('6291.d1 missing');
if (r6291.j_str !== '110592/233') throw new Error('6291.d1 bad j '+r6291.j_str);
const cubic = identifyCubicJS('x^3 + x^2*y + y^3 + y^2 - 2*x + 1 = 0');
console.log(cubic);
if (!cubic.ok || cubic.j !== '110592/233') throw new Error('cubic recognizer failed');
if (!findByJ(cubic.j).some(r => r.label === '6291.d1')) throw new Error('same-j lookup failed');
for (const q of ['[3,3]','[0,0,1,3,3]','y^2 + y = x^3 + 3*x + 3','u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0']) {
  const got = identifyCubicJS(q);
  if (!got.ok) throw new Error('recognizer failed: '+q+' '+got.error);
  console.log(q, got.j, got.coeffs_int);
}
