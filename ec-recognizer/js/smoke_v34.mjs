import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { identifyCubicJS } from './ec_core.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const readJSON = rel => JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8'));

function assert(cond, msg) { if (!cond) throw new Error(msg); }

assert(app.includes("const EC_ATLAS_VERSION = 'v34';"), 'runtime version should be v34');
assert(html.includes('>v34<'), 'HTML version badge should be v34');
assert(!app.includes("import { identifyCubicJS }"), 'cubic recognizer should be lazy-loaded');
assert(app.includes("import('./js/ec_core.js')"), 'lazy cubic recognizer import missing');
assert(app.includes('JSON_STREAM_PROGRESS_THRESHOLD'), 'small JSON native-loading path missing');
assert(app.includes('searchResultCache: new Map()'), 'search cache missing');
assert(app.includes('parseConductorListingQuery'), 'conductor-prefix search parser missing');
assert(app.includes('loadConductorRowsForN'), 'cached complete-conductor row loader missing');
assert(app.includes('onBatch'), 'incremental search result callback missing');
assert(app.includes('INTEGRAL_SEARCH_TIME_MULTIPLIER = 2.25'), '2.25x integral/S-integral time budget missing');
assert(app.includes('pointSearches: new Map()'), 'resumable point-search cache missing');
assert(app.includes('makeCubicSquareSieve'), 'modular square sieve missing');
assert(app.includes('createRationalSeedState'), 'low-height rational seed search missing');
assert(app.includes('pumpRationalSeedSearch'), 'low-height rational search pump missing');
assert(app.includes('basis_rank_guess'), 'heuristic Mordell-Weil basis reporting missing');
assert(app.includes('createGroupSearch'), 'group-generated integral/S-integral search missing');
assert(app.includes('ecMulPoint'), 'elliptic-curve scalar multiplication missing');
assert(app.includes('searched:-1'), 'integral search must not skip x=0');
assert(!app.includes('curves_compact.json'), 'obsolete compact curve DB dependency returned');

const cubic = identifyCubicJS('u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0');
assert(cubic.ok && cubic.j === '110592/233', 'cubic recognizer regression');

const d0 = readJSON('data/detail_shards/d000.json');
assert(d0.columns.includes('weierstrass_equation') && d0.rows.length > 0, 'detail shards unavailable');

const coreStart = app.indexOf('function normalizeSearchText');
const coreEnd = app.indexOf('async function apiIntegralPoints');
assert(coreStart > 0 && coreEnd > coreStart, 'could not extract arithmetic core');
const core = app.slice(coreStart, coreEnd);
const sandbox = {
  console,
  performance: { now: () => Date.now() },
  API_CACHE: { squareSieveCache: new Map() },
};
vm.runInNewContext(`
const API_CACHE = globalThis.API_CACHE;
function idleYield() { return Promise.resolve(); }
${core}
globalThis.__v34 = { parseConductorHint, parseConductorListingQuery, createIntegralSearchState, runIntegralSearchState };
`, sandbox);

assert(sandbox.__v34.parseConductorHint('6552.') === 6552, 'trailing-dot conductor hint failed');
assert(sandbox.__v34.parseConductorListingQuery('6552.').N === 6552, 'full conductor listing parser failed');
assert(sandbox.__v34.parseConductorListingQuery('6552.a').prefix === '6552.a', 'conductor class-prefix parser failed');

const curve9747d2 = {
  id: 62931,
  label: '9747.d2',
  N: 9747,
  rank: 2,
  pts: 6,
  a1: '0', a2: '0', a3: '1', a4: '0', a6: '90'
};
const st = sandbox.__v34.createIntegralSearchState(curve9747d2);
await sandbox.__v34.runIntegralSearchState(st, 700, null);
const keys = new Set(st.found.map(p => `${p.x},${p.y}`));
assert(keys.has('0,9') && keys.has('0,-10'), '9747.d2 x=0 integral points should be found');
assert((st.rationalSeed?.checked || 0) > 0, 'low-height rational seed search did not run');

console.log('v34 smoke checks passed');
