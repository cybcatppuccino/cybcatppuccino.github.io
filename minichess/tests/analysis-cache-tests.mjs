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
const ordinary = {
  engine: ENGINE_VERSION, depth: 7, selDepth: 11, nodes: 1200, nps: 9000, elapsed: 133,
  hashfull: 120, completed: true, terminal: false, rootTurn: -1,
  multiPvVerified: true,
  lines: [{ move: 'a2a3', score: 12, scoreText: '+0.12', pv: ['a2a3', 'a4a3'], rootScoreExact: true }]
};
assert.equal(cache.set(key, ordinary), null, 'Ordinary cp/PV results must not persist or resume.');
assert.equal(cache.get(key), null);

const exact = {
  engine: ENGINE_VERSION,
  engineLabel: `${ENGINE_VERSION} + GTB`,
  depth: 0,
  completed: true,
  terminal: true,
  tablebase: true,
  tablebaseScope: 'root-exact',
  tablebaseWdl: 1,
  rootTurn: -1,
  multiPvVerified: true,
  lines: [{
    move: 'a2a3', score: 29998, scoreText: '#1', pv: ['a2a3'],
    tablebase: true, tablebaseScope: 'root-exact', tablebaseWdl: 1,
    tablebaseExactDtm: true, dtm: 2, rootScoreExact: true
  }]
};
cache.set(key, exact);
assert.equal(cache.get(key).tablebaseScope, 'root-exact');
assert.equal(cache.get(key).lines[0].move, 'a2a3');
assert.equal(cache.get(key).rootTurn, -1, 'v19.5 preserves the root side for exact display.');
const reloaded = new AnalysisCache(storage);
assert.equal(reloaded.get(key).tablebaseScope, 'root-exact');

const reversiblePosition = position.clone();
reversiblePosition.halfmove = 1;
assert.notEqual(buildAnalysisKey(reversiblePosition, ['rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1']), buildAnalysisKey(reversiblePosition, []), 'Reversible history context must participate in the cache key');

const staleKey = `${key}-stale`;
assert.equal(cache.set(staleKey, { ...exact, engine: 'Orion JS 3.0' }), null, 'Results from older engines must be rejected');
assert.equal(cache.get(staleKey), null);
console.log('Persistent proof-only analysis-cache tests passed.');
