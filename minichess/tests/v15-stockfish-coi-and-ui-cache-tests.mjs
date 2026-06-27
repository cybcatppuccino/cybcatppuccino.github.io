import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { AnalysisCache, buildAnalysisKey } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION } from '../js/engine/engine.js';
import { Position } from '../js/core/position.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

const registerUrl = new URL('../coi-serviceworker-register.js', import.meta.url);
const swUrl = new URL('../coi-serviceworker.js', import.meta.url);
if (!existsSync(registerUrl) || !existsSync(swUrl)) {
  console.log('v18.1 Stockfish COI cleanup test: skipped (legacy COI helper files absent from this package)');
} else {
const register = readFileSync(registerUrl, 'utf8');
assert.match(register, /legacy COI helper cleanup/, 'v15.1 should clean up old COI service workers instead of trying to fake isolation');
assert.match(register, /unregister/, 'cleanup helper should unregister old COI workers');
assert.doesNotMatch(register, /serviceWorker\.register/, 'cleanup helper must not register a new COI service worker');
assert.doesNotMatch(register, /__gardnerRequestCoiReload/, 'app should not be forced into COI-preparing fallback mode');

const sw = readFileSync(swUrl, 'utf8');
assert.match(sw, /legacy file/);
assert.match(sw, /unregister/);
assert.doesNotMatch(sw, /Cross-Origin-Embedder-Policy/, 'v15.1 no longer injects COEP in a service worker');
}

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
assert.match(app, /scheduleAnalysisPaint/, 'analysis UI should coalesce streamed result rendering');
assert.match(app, /kernel:\s*engineKernel/, 'Fairy should be attempted by the provider, not preemptively replaced in the UI');
assert.doesNotMatch(app, /Stockfish preparing COI/);
assert.doesNotMatch(app, /__gardnerRequestCoiReload/);
assert.doesNotMatch(app, /Orion JS 14\.3/);

const storage = new MemoryStorage();
const key = buildAnalysisKey(Position.initial(), []);
storage.setItem('gardner-analysis-cache-v15', JSON.stringify([{
  key,
  updatedAt: 10,
  result: {
    engine: 'Orion JS 15',
    depth: 9,
    nodes: 100,
    nps: 1000,
    elapsed: 10,
    completed: true,
    lines: [{ move: 'a2a3', score: 1, scoreText: '+0.01', pv: ['a2a3'] }]
  }
}]));
const cache = new AnalysisCache(storage);
assert.equal(cache.get(key), null, 'v18.1 should start fresh instead of migrating v15 Orion entries');
assert.equal(storage.getItem('gardner-analysis-cache-v15'), null, 'v18.1 should retire stale v15 storage');

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
const payload = JSON.parse(storage.getItem('gardner-analysis-cache-v18.1') || '[]');
assert.equal(payload.length, 576, 'v18.1 should keep the v14.3/v15 persistent cache size unchanged');

console.log('v18.1 Stockfish startup simplification, UI throttling and cache reset tests passed.');
