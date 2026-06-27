import assert from 'node:assert/strict';

import { AnalysisCache } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION, EnginePosition, EngineInternals, uciToMove } from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';

function fakeStorageWith(entries = {}) {
  const storage = new Map(Object.entries(entries));
  return {
    storage,
    api: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, String(value)),
      removeItem: key => storage.delete(key)
    }
  };
}

{
  const stalePayload = JSON.stringify([{ key: 'old', updatedAt: 1, result: { engine: 'Orion JS 17.3', depth: 20, completed: true, tablebase: true, lines: [{ move: 'a1a2', score: 29980, pv: ['a1a2'] }] } }]);
  const { storage, api } = fakeStorageWith({
    'gardner-analysis-cache-v17.3': stalePayload,
    'gardner-analysis-cache-v17.2': stalePayload
  });
  const cache = new AnalysisCache(api);
  assert.equal(storage.has('gardner-analysis-cache-v17.3'), false, 'v17.4 should remove stale v17.3 cache bucket');
  assert.equal(storage.has('gardner-analysis-cache-v17.2'), false, 'v17.4 should remove stale v17.2 cache bucket');
  assert.equal(cache.get('old'), null, 'old exact tablebase cache entries should not be migrated');
}

{
  const { api } = fakeStorageWith();
  const cache = new AnalysisCache(api);
  const key = 'missing-previous-edge-case';
  cache.entries.set(key, undefined);
  const result = {
    engine: ENGINE_VERSION,
    depth: 10,
    scoreDepth: 10,
    completed: true,
    pvComplete: true,
    lines: [{ move: 'a1a2', score: 12, pv: ['a1a2', 'a5a4', 'a2a3', 'a4a3', 'b1b2', 'b5b4', 'b2b3', 'b4b3'] }]
  };
  assert.doesNotThrow(() => cache.set(key, result));
  assert.equal(cache.get(key)?.lines?.[0]?.move, 'a1a2');
}

{
  const { api } = fakeStorageWith();
  const cache = new AnalysisCache(api);
  const incomplete = {
    engine: ENGINE_VERSION,
    depth: 8,
    scoreDepth: 8,
    pvComplete: false,
    completed: false,
    lines: [{ move: 'b1b2', score: 40, pv: ['b1b2', 'b5b4'] }]
  };
  const returned = cache.set('first-incomplete-v17.4', incomplete);
  assert.equal(returned?.lines?.[0]?.move, 'b1b2', 'first live incomplete result can still be returned for UI display');
  assert.equal(cache.get('first-incomplete-v17.4'), null, 'first live incomplete result must not persist as resume cache');
}

// Optional exact tablebase subset smoke. This runs when TB_BASE points at a local
// served copy of the uploaded pawn database subset; otherwise it skips cleanly.
if (process.env.TB_BASE) {
  const tablebase = new GardnerTablebase({ baseUrl: process.env.TB_BASE });
  assert.equal(await tablebase.init(), true);
  if (tablebase.exactManifest.tables.KPPvKP) {
    const root = EnginePosition.fromFEN('5/2k2/1p3/1K1PP/5 w - - 0 4');
    const before = await tablebase.analyze(root, { multipv: 5, maxPvPly: 30 });
    assert.equal(before?.tablebase, true);
    assert.ok(before.lines.some(line => line.move === 'd2d3'), 'non-first d3+ candidate should be present from exact 5-piece tablebase');

    const after = EnginePosition.fromFEN('5/2k2/1p3/1K1PP/5 w - - 0 4');
    EngineInternals.makeMove(after, uciToMove(after, 'd2d3'));
    const afterAnalysis = await tablebase.analyze(after, { multipv: 5, maxPvPly: 30 });
    assert.equal(afterAnalysis?.tablebase, true);
    assert.ok(afterAnalysis.lines[0]?.dtm < 19, 'after d3+ should be re-probed as its own 5-piece tablebase position, not reuse the parent #10');
  }
}

console.log('v17.4 cache and tablebase stability tests passed.');
