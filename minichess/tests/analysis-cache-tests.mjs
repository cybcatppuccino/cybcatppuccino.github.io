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

const storage = new MemoryStorage();
const cache = new AnalysisCache(storage);
const position = Position.initial();
const key = buildAnalysisKey(position, []);
const result = {
  engine: ENGINE_VERSION, depth: 7, selDepth: 11, nodes: 1200, nps: 9000, elapsed: 133,
  hashfull: 120, completed: true, terminal: false, rootTurn: -1,
  lines: [{ move: 'a2a3', score: 12, scoreText: '+0.12', pv: ['a2a3', 'a4a3'] }]
};
cache.set(key, result);
assert.equal(cache.get(key).depth, 7);
const reloaded = new AnalysisCache(storage);
assert.equal(reloaded.get(key).lines[0].move, 'a2a3');
assert.equal(reloaded.get(key).rootTurn, -1, 'v19 should preserve the root side for perspective-aware analysis display');
reloaded.set(key, { ...result, depth: 4 });
assert.equal(reloaded.get(key).depth, 7, 'A shallow transient result must not overwrite a deeper cached result');
const reversiblePosition = position.clone();
reversiblePosition.halfmove = 1;
assert.notEqual(buildAnalysisKey(reversiblePosition, ['rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1']), buildAnalysisKey(reversiblePosition, []), 'Reversible history context must participate in the cache key');

const staleKey = `${key}-stale`;
assert.equal(cache.set(staleKey, { ...result, engine: 'Orion JS 3.0' }), null, 'Results from older engines must be rejected');
assert.equal(cache.get(staleKey), null);
console.log('Persistent analysis-cache tests passed.');
