import fs from 'node:fs';
import path from 'node:path';
import { identifyCubicJS } from './ec_core.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const readJSON = rel => JSON.parse(fs.readFileSync(path.join(root, rel), 'utf8'));

function assert(cond, msg) { if (!cond) throw new Error(msg); }

assert(app.includes("const EC_ATLAS_VERSION = 'v32';"), 'runtime version should be v32');
assert(html.includes('>v32<'), 'HTML version badge should be v32');
assert(!app.includes("import { identifyCubicJS }"), 'cubic recognizer should be lazy-loaded');
assert(app.includes("import('./js/ec_core.js')"), 'lazy cubic recognizer import missing');
assert(app.includes('JSON_STREAM_PROGRESS_THRESHOLD'), 'small JSON native-loading path missing');
assert(app.includes('rowMatchesNeedle'), 'cached conductor search matching missing');
assert(app.includes('relationMatricesAsync'), 'async C-isogeny matrix generation missing');
assert(app.includes('INTEGRAL_SEARCH_TIME_MULTIPLIER = 1.5'), '1.5x integral/S-integral time budget missing');
assert(app.includes('createGroupSearch'), 'group-generated integral/S-integral search missing');
assert(app.includes('ecAddPoints'), 'elliptic-curve group law missing');
assert(!app.includes('curves_compact.json'), 'obsolete compact curve DB dependency returned');

const cubic = identifyCubicJS('u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0');
assert(cubic.ok && cubic.j === '110592/233', 'cubic recognizer regression');

const d0 = readJSON('data/detail_shards/d000.json');
assert(d0.columns.includes('weierstrass_equation') && d0.rows.length > 0, 'detail shards unavailable');
console.log('v32 smoke checks passed');
