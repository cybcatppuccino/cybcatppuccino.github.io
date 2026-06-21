import { identifyCubicJS } from './ec_core.js';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const here = path.dirname(new URL(import.meta.url).pathname);
const root = path.resolve(here, '..');
const app = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
if (!app.includes("EC_ATLAS_VERSION = 'v23'")) throw new Error('version constant missing');
if (!app.includes('detail_shards/d')) throw new Error('v23 detail shards not used');
if (!app.includes('search/exact/e')) throw new Error('v23 exact search index not used');
if (app.includes('curves_compact.json')) throw new Error('curves_compact dependency returned');

const cubic = identifyCubicJS('u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0');
if (!cubic.ok || cubic.j !== '110592/233') throw new Error('cubic recognizer regression');

const detailDir = path.join(root, 'data', 'detail_shards');
const exactDir = path.join(root, 'data', 'search', 'exact');
const nDir = path.join(root, 'data', 'search', 'N');
if (fs.existsSync(detailDir) && fs.readdirSync(detailDir).filter(x => x.endsWith('.json')).length < 512) throw new Error('detail shard count too small');
if (fs.existsSync(exactDir) && fs.readdirSync(exactDir).filter(x => x.endsWith('.json')).length < 256) throw new Error('exact search shard count too small');
if (fs.existsSync(nDir) && fs.readdirSync(nDir).filter(x => x.endsWith('.json')).length < 64) throw new Error('conductor search shard count too small');
console.log('v23 smoke ok', cubic.j);
