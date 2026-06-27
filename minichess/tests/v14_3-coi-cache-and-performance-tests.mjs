import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { AnalysisCache, buildAnalysisKey } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION, EnginePosition, GardnerSearcher } from '../js/engine/engine.js';
import { Position } from '../js/core/position.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

const registerUrl = new URL('../coi-serviceworker-register.js', import.meta.url);
if (!existsSync(registerUrl)) {
  console.log('v18.1 COI cleanup test: skipped (legacy COI helper files absent from this package)');
} else {
const register = readFileSync(registerUrl, 'utf8');
assert.match(register, /legacy COI helper cleanup/);
assert.doesNotMatch(register, /serviceWorker\.register/);
const fairyWorker = readFileSync(new URL('../vendor/fairy-stockfish/fairy-uci-worker.js', import.meta.url), 'utf8');
assert.ok(!fairyWorker.includes("throw new Error('SharedArrayBuffer is unavailable"), 'SharedArrayBuffer startup path must not throw an uncaught worker error');
assert.match(fairyWorker, /Orion JS will be used/);
}

const storage = new MemoryStorage();
const position = Position.initial();
const key = buildAnalysisKey(position, []);
const migratedResult = {
  engine: 'Orion JS 14.2',
  depth: 11,
  nodes: 1234,
  nps: 5000,
  elapsed: 100,
  completed: true,
  lines: [{ move: 'a2a3', score: 3, scoreText: '+0.03', pv: ['a2a3', 'a4a3'] }]
};
storage.setItem('gardner-analysis-cache-v14_2', JSON.stringify([{ key: `${key}|Kfairy-stockfish`, updatedAt: 1, result: migratedResult }]));
const cache = new AnalysisCache(storage);
assert.equal(cache.get(key), null, 'v18.1 should start fresh instead of migrating v14.2 shared Orion cache');
assert.equal(storage.getItem('gardner-analysis-cache-v14_2'), null, 'v18.1 should retire the stale v14.2 storage key');

for (let i = 0; i < 620; i += 1) {
  cache.set(`${key}|extra-${i}`, {
    engine: ENGINE_VERSION,
    depth: i % 20,
    nodes: i,
    nps: 1,
    elapsed: 1,
    completed: true,
    lines: [{ move: 'a2a3', score: 0, scoreText: '+0.00', pv: ['a2a3'] }]
  });
}
cache.flush();
const payload = JSON.parse(storage.getItem('gardner-analysis-cache-v18.1') || '[]');
assert.ok(payload.length > 192, 'v18.1 cache should be larger than the old 192-entry budget');
assert.ok(payload.length <= 576, 'v18.1 cache should keep the 576-entry budget');

const searcher = new GardnerSearcher({ hashEntries: 16_384 });
assert.equal(searcher.evalScore.length, 524288, 'v15.1 eval cache should keep the expanded budget');
const root = EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
const first = searcher.staticEvaluate(root);
const second = searcher.staticEvaluate(root);
assert.equal(first, second, 'expanded eval cache must preserve static evaluation semantics');

console.log('v18.1 cleanup, cache and conservative performance tests passed.');
