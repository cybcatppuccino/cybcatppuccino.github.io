import assert from 'node:assert/strict';

import { AnalysisCache } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION } from '../js/engine/engine.js';
import {
  compareAnalysisResults,
  isTrustedExactTablebaseResult,
  RESULT_KIND,
  withResultQuality
} from '../js/engine/result-quality.js';

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
  const liveComplete = withResultQuality({
    engine: ENGINE_VERSION,
    depth: 11,
    completed: true,
    pvComplete: true,
    lines: [{ move: 'a1a2', score: 20, pv: ['a1a2','a5a4','a2a3','a4a3','b1b2','b5b4','b2b3','b4b3','c1c2'] }]
  });
  const tbBound = withResultQuality({
    engine: ENGINE_VERSION,
    depth: 8,
    completed: true,
    tablebaseDtmBound: true,
    lines: [{ move: 'b1b2', score: 29982, scoreText: '≤#9 · TB bound', tablebase: true, tablebaseBound: true, dtmUpperBound: true, pv: ['b1b2'] }]
  });
  assert.equal(tbBound.resultKind, RESULT_KIND.TABLEBASE_BOUND);
  assert.equal(compareAnalysisResults(liveComplete, tbBound), tbBound, 'TB-bound enrichment should be allowed to update a live search result');
}

{
  const exactTb = withResultQuality({
    engine: ENGINE_VERSION,
    tablebase: true,
    tablebaseSource: 'exact-core',
    tablebaseWdl: 1,
    completed: true,
    lines: [{ move: 'd2d3', score: 29981, tablebase: true, tablebaseExactDtm: true, tablebaseWdl: 1, pv: ['d2d3'] }]
  });
  const boundedTb = withResultQuality({
    ...exactTb,
    tablebaseDtmBound: true,
    lines: [{ ...exactTb.lines[0], tablebaseExactDtm: false, tablebaseBound: true, dtmUpperBound: true }]
  });
  assert.equal(isTrustedExactTablebaseResult(exactTb), true, 'exact current-version tablebase result should be trusted as a resume');
  assert.equal(isTrustedExactTablebaseResult(boundedTb), false, 'DTM-bound tablebase result should be refreshed instead of terminating analysis');
  assert.equal(compareAnalysisResults(boundedTb, exactTb), exactTb, 'exact TB must outrank bound TB');
}

{
  const stale = JSON.stringify([{ key: 'old', updatedAt: 1, result: { engine: 'Orion JS 17.4', tablebase: true, completed: true, lines: [{ move: 'a1a2', score: 29980, pv: ['a1a2'] }] } }]);
  const { storage, api } = fakeStorageWith({ 'gardner-analysis-cache-v17.4': stale });
  const cache = new AnalysisCache(api);
  assert.equal(storage.has('gardner-analysis-cache-v17.4'), false, 'v18.2 should remove stale v17.4 cache bucket');
  cache.set('exact-tb', {
    engine: ENGINE_VERSION,
    tablebase: true,
    tablebaseSource: 'exact-core',
    tablebaseWdl: 0,
    completed: true,
    lines: [{ move: 'a1a2', score: 0, tablebase: true, tablebaseWdl: 0, pv: ['a1a2'] }]
  });
  assert.equal(cache.get('exact-tb')?.resultKind, RESULT_KIND.EXACT_TABLEBASE);
}

console.log('v18.2 result-quality and analysis-cache tests passed.');
