import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { AnalysisCache, buildAnalysisKey } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION } from '../js/engine/engine.js';
import { Position } from '../js/core/position.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

const register = readFileSync(new URL('../coi-serviceworker-register.js', import.meta.url), 'utf8');
assert.match(register, /coi-serviceworker\.js\?\$\{VERSION\}/, 'COI helper should register a versioned service worker URL');
assert.match(register, /updateViaCache:\s*'none'/, 'COI helper should bypass cached old workers');
assert.match(register, /MAX_RELOADS\s*=\s*3/, 'COI helper should use bounded reload attempts');
assert.match(register, /__gardnerRequestCoiReload/, 'UI should be able to explicitly request COI preparation when Fairy is selected');

const sw = readFileSync(new URL('../coi-serviceworker.js', import.meta.url), 'utf8');
assert.match(sw, /X-Gardner-COI-ServiceWorker/);
assert.match(sw, /Cross-Origin-Embedder-Policy/);
assert.match(sw, /require-corp/);

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
assert.match(app, /scheduleAnalysisPaint/, 'analysis UI should coalesce streamed result rendering');
assert.match(app, /fairySharedMemoryReady/, 'app should not start Fairy directly before SharedArrayBuffer is available');
assert.match(app, /__gardnerRequestCoiReload/, 'selecting Fairy should actively request isolated reload preparation');
assert.doesNotMatch(app, /Orion JS 14\.3/);

const storage = new MemoryStorage();
const key = buildAnalysisKey(Position.initial(), []);
storage.setItem('gardner-analysis-cache-v14_3', JSON.stringify([{
  key,
  updatedAt: 10,
  result: {
    engine: 'Orion JS 14.3',
    depth: 9,
    nodes: 100,
    nps: 1000,
    elapsed: 10,
    completed: true,
    lines: [{ move: 'a2a3', score: 1, scoreText: '+0.01', pv: ['a2a3'] }]
  }
}]));
const cache = new AnalysisCache(storage);
assert.equal(cache.get(key)?.engine, ENGINE_VERSION, 'v14.3 Orion entries should migrate to v15');
assert.equal(cache.get(key)?.depth, 9);
assert.ok(storage.getItem('gardner-analysis-cache-v15')?.includes('a2a3'), 'v15 cache should persist migrated entries');

for (let i = 0; i < 700; i += 1) {
  cache.set(`${key}|${i}`, {
    engine: ENGINE_VERSION,
    depth: 1,
    nodes: 1,
    nps: 1,
    elapsed: 1,
    completed: true,
    lines: [{ move: 'a2a3', score: 0, scoreText: '0.00', pv: ['a2a3'] }]
  });
}
cache.flush();
const payload = JSON.parse(storage.getItem('gardner-analysis-cache-v15') || '[]');
assert.equal(payload.length, 576, 'v15 should keep the v14.3 persistent cache size unchanged');

console.log('v15 Stockfish COI, UI throttling and cache migration tests passed.');
