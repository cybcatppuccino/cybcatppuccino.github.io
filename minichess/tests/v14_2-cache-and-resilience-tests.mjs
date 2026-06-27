import assert from 'node:assert/strict';
import { AnalysisCache, buildAnalysisKey } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION } from '../js/engine/engine.js';
import { Position } from '../js/core/position.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

const position = Position.initial();
const key = buildAnalysisKey(position, []);
const oldKernelKey = `${key}|Korion-js`;
const oldResult = {
  engine: 'Orion JS 14.1',
  depth: 12,
  selDepth: 18,
  nodes: 1000,
  nps: 5000,
  elapsed: 200,
  completed: true,
  lines: [{ move: 'a2a3', score: 0, scoreText: '+0.00', pv: ['a2a3', 'a4a3'], mateVerified: false }]
};
const storage = new MemoryStorage();
storage.setItem('gardner-analysis-cache-v14_1', JSON.stringify([{ key: oldKernelKey, updatedAt: 42, result: oldResult }]));
const cache = new AnalysisCache(storage);
const migrated = cache.get(key);
assert.equal(migrated, null, 'v18.1 should start a fresh cache instead of migrating old v14.1 Orion entries');
assert.equal(cache.get(oldKernelKey), null, 'legacy kernel-suffixed v14.1 keys should not resolve after v18.1 cache reset');
assert.equal(storage.getItem('gardner-analysis-cache-v14_1'), null, 'v18.1 cache load should retire stale v14.1 storage');
assert.ok(storage.getItem('gardner-analysis-cache-v18.1'), 'v18.1 cache should persist into the current v18.1 storage bucket');

const externalRejected = cache.set(`${key}-fairy`, { ...oldResult, engine: 'Fairy-Stockfish wasm 1.1.11' });
assert.equal(externalRejected, null, 'external kernel results must not overwrite the shared Orion cache');

const solved = {
  ...oldResult,
  engine: ENGINE_VERSION,
  depth: 6,
  lines: [{ move: 'a2a3', score: 29995, scoreText: '#3', pv: ['a2a3', 'a4a3'], mateVerified: true, dtm: 5 }]
};
cache.set(`${key}-mate`, solved);
cache.set(`${key}-mate`, { ...oldResult, engine: ENGINE_VERSION, depth: 20 });
assert.equal(cache.get(`${key}-mate`)?.lines[0].mateVerified, true, 'solved mate cache should not be replaced by deeper non-solved output');

console.log('v18.1 shared cache reset and protection tests passed.');
