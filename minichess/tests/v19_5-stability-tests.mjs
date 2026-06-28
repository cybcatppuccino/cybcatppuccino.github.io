import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { AnalysisCache } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION } from '../js/engine/engine.js';
import {
  RESULT_KIND,
  isSolvedResult,
  isTrustedExactTablebaseResult,
  shouldCacheResult,
  withResultQuality
} from '../js/engine/result-quality.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

const ordinaryWithFutureTbHint = withResultQuality({
  engine: ENGINE_VERSION,
  depth: 12,
  completed: true,
  multiPvVerified: true,
  lines: [{
    move: 'a2a3', score: 220, scoreText: '+2.20',
    pv: ['a2a3', 'a5a4', 'b2b3', 'b5b4', 'c2c3', 'c5c4', 'd2d3', 'd5d4'],
    rootScoreExact: true,
    tablebaseHint: { conditional: true, entersAtPly: 6, wdl: 1, dtmPly: 12 }
  }],
  tablebaseDtmHint: true
});
assert.equal(ordinaryWithFutureTbHint.resultKind, RESULT_KIND.TABLEBASE_HINT);
assert.equal(isSolvedResult(ordinaryWithFutureTbHint), false, 'A future PV tablebase hit is not a root proof.');
assert.equal(shouldCacheResult(ordinaryWithFutureTbHint), false, 'A conditional hint must not enter a resume cache.');

const exactRootTb = withResultQuality({
  engine: ENGINE_VERSION,
  terminal: true,
  completed: true,
  tablebase: true,
  tablebaseScope: 'root-exact',
  multiPvVerified: true,
  lines: [{
    move: 'a2a3', score: 29992, scoreText: '#4', pv: ['a2a3'],
    tablebase: true, tablebaseScope: 'root-exact', tablebaseExactDtm: true,
    dtm: 8, rootScoreExact: true
  }]
});
assert.equal(exactRootTb.resultKind, RESULT_KIND.EXACT_TABLEBASE);
assert.equal(isTrustedExactTablebaseResult(exactRootTb), true);
assert.equal(isSolvedResult(exactRootTb), true);
assert.equal(shouldCacheResult(exactRootTb), true);

const cache = new AnalysisCache(new MemoryStorage());
assert.equal(cache.set('ordinary', ordinaryWithFutureTbHint), null);
assert.ok(cache.set('exact', exactRootTb));
assert.equal(cache.get('ordinary'), null);
assert.equal(cache.get('exact')?.tablebaseScope, 'root-exact');

const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const playWorker = readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const engine = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const tablebase = readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

assert.equal(ENGINE_VERSION, 'Orion JS 19.7');
assert.match(app, /const ANALYSIS_PAINT_INTERVAL_MS = 500;/, 'UI throttle remains 500 ms.');
assert.doesNotMatch(app, /cachePrincipalVariationChildren/, 'PV child corridors must not reseed normal analysis.');
assert.doesNotMatch(worker, /positionCache|bestResume|mergeStableLine|annotateResultWithDtmBounds/, 'Analysis worker must not merge/cross-root resume ordinary results.');
assert.doesNotMatch(playWorker, /resultCache|bestResume|annotateResultWithDtmBounds/, 'Play worker must not reuse ordinary results or promote future TB hints.');
assert.doesNotMatch(engine, /rootDeadDraw/, 'Low-progress status must not hard-flatten a root to draw.');
assert.match(engine, /result\.rootScoreExact = true/, 'Every stable MultiPV line is marked after its final root window.');
assert.match(tablebase, /tablebaseHint/, 'Future tablebase observations are exposed only as hints.');
assert.match(html, /boardStyleSelect/, 'Existing board palettes are selectable from the UI.');
assert.match(html, /Gardner MiniChess Lab v19\.7/);

console.log('v19.7 stability, proof-scope and board-style tests passed.');
