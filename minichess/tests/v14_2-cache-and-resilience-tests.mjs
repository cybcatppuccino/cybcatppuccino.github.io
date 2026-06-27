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
assert.equal(migrated?.engine, ENGINE_VERSION, 'v14.1 Orion cache should migrate to the current engine identity');
assert.equal(migrated?.depth, 12, 'migrated cache should preserve depth');
assert.equal(cache.get(oldKernelKey)?.depth, 12, 'legacy kernel-suffixed keys should still resolve');
assert.ok(storage.getItem('gardner-analysis-cache-v15')?.includes('a2a3'), 'migrated cache should be persisted into the current v15 storage');

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

console.log('v14.2/v15 shared cache migration and protection tests passed.');
