import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { ENGINE_VERSION, EnginePosition } from '../js/engine/engine.js';
import { AnalysisCache } from '../js/engine/analysis-cache.js';

{
  const oldFetch = globalThis.fetch;
  let manifestFetched = false;
  globalThis.fetch = async urlLike => {
    const url = String(urlLike);
    if (url.includes('/manifest.json')) {
      manifestFetched = true;
      return Response.json({ tables: { KvK: { path: 'KvK/metadata.json', pieceCount: 2 } } });
    }
    return new Response('missing', { status: 404 });
  };
  const { GardnerTablebase } = await import(`../js/engine/tablebase.js?embedManifest=${Date.now()}`);
  const tablebase = new GardnerTablebase({ baseUrl: 'https://tb.test/tables/' });
  assert.equal(await tablebase.init(), true);
  assert.equal(manifestFetched, true);
  assert.equal(Object.keys(tablebase.exactManifest.tables).length, 111);
  assert.equal(Object.values(tablebase.exactManifest.tables).filter(entry => entry.pieceCount === 5).length, 75);
  assert.ok(tablebase.exactManifest.tables.KQvKBB);
  assert.ok(tablebase.exactManifest.tables.KRvKNP);
  globalThis.fetch = oldFetch;
}

{
  const storage = new Map();
  const fakeStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key)
  };
  const cache = new AnalysisCache(fakeStorage);
  const key = 'cache-completeness-test';
  const complete = {
    engine: ENGINE_VERSION,
    depth: 12,
    scoreDepth: 12,
    pvDepth: 12,
    pvComplete: true,
    completed: true,
    lines: [{ move: 'a1a2', score: 30, pv: ['a1a2', 'a5a4', 'a2a3', 'a4a3', 'b1b2', 'b5b4', 'b2b3', 'b4b3', 'c1c2', 'c5c4'] }]
  };
  const thin = {
    engine: ENGINE_VERSION,
    depth: 14,
    scoreDepth: 14,
    pvDepth: 4,
    pvComplete: false,
    completed: false,
    pvIncomplete: true,
    lines: [{ move: 'b1b2', score: 80, pv: ['b1b2', 'b5b4'] }]
  };
  assert.ok(cache.set(key, complete));
  assert.equal(cache.set(key, thin)?.lines?.[0]?.move, 'a1a2');
  assert.equal(cache.get(key)?.pvComplete, true);
}

console.log('v18.1 tablebase manifest and cache-completeness tests passed.');
