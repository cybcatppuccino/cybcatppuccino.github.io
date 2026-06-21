import fs from 'node:fs';
import path from 'node:path';
import { identifyCubicJS } from './ec_core.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const readJSON = rel => JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8'));

function assert(cond, msg) { if (!cond) throw new Error(msg); }

assert(app.includes("const EC_ATLAS_VERSION = 'v33';"), 'runtime version should be v33');
assert(html.includes('>v33<'), 'HTML version badge should be v33');
assert(!app.includes("import { identifyCubicJS }"), 'cubic recognizer should be lazy-loaded');
assert(app.includes("import('./js/ec_core.js')"), 'lazy cubic recognizer import missing');
assert(app.includes('JSON_STREAM_PROGRESS_THRESHOLD'), 'small JSON native-loading path missing');
assert(app.includes('rowMatchesNeedle'), 'cached conductor search matching missing');
assert(app.includes('relationMatricesAsync'), 'async C-isogeny matrix generation missing');
assert(app.includes('INTEGRAL_SEARCH_TIME_MULTIPLIER = 2.25'), '2.25x integral/S-integral time budget missing');
assert(app.includes('pointSearches: new Map()'), 'resumable point-search cache missing');
assert(app.includes('makeCubicSquareSieve'), 'modular square sieve missing');
assert(app.includes('basis_rank_guess'), 'heuristic Mordell-Weil basis reporting missing');
assert(app.includes('createGroupSearch'), 'group-generated integral/S-integral search missing');
assert(app.includes('ecMulPoint'), 'elliptic-curve scalar multiplication missing');
assert(!app.includes('curves_compact.json'), 'obsolete compact curve DB dependency returned');

const cubic = identifyCubicJS('u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0');
assert(cubic.ok && cubic.j === '110592/233', 'cubic recognizer regression');

const d0 = readJSON('data/detail_shards/d000.json');
assert(d0.columns.includes('weierstrass_equation') && d0.rows.length > 0, 'detail shards unavailable');
console.log('v33 smoke checks passed');
