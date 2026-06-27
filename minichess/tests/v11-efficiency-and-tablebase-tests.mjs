import assert from 'node:assert/strict';

import {
  EngineInternals,
  EnginePosition,
  GardnerSearcher,
  ENGINE_VERSION,
  generateLegalMoves
} from '../js/engine/engine.js';
import { AnalysisCache } from '../js/engine/analysis-cache.js';

const { makeMove, undoMove } = EngineInternals;

function scanMasks(position) {
  let pieceCount = 0;
  for (let sq = 0; sq < 25; sq += 1) if (position.board[sq]) pieceCount += 1;
  return { pieceCount };
}


function assertIncrementalFields(position) {
  const scanned = scanMasks(position);
  assert.equal(position.pieceCount, scanned.pieceCount, 'pieceCount must stay incremental-correct');
}


{
  const position = EnginePosition.fromFEN('8/8/3k4/1p1pb3/1P2p3/2P1Pr2/1R1BK3/8 b - - 37 26');
  const original = position.clone();
  assertIncrementalFields(position);
  const moves = generateLegalMoves(position, false).slice(0, 8);
  assert.ok(moves.length > 1, 'test position should have several legal moves');
  for (const move of moves) {
    const state = makeMove(position, move);
    assertIncrementalFields(position);
    const replies = generateLegalMoves(position, false).slice(0, 5);
    for (const reply of replies) {
      const childState = makeMove(position, reply);
      assertIncrementalFields(position);
      undoMove(position, reply, childState);
      assertIncrementalFields(position);
    }
    undoMove(position, move, state);
    assertIncrementalFields(position);
    assert.deepEqual([...position.board], [...original.board], 'make/undo must restore board');
    assert.equal(position.hashA, original.hashA, 'make/undo must restore hashA');
    assert.equal(position.hashB, original.hashB, 'make/undo must restore hashB');
  }
}

{
  const fen = '8/8/8/1p3p2/1k3P2/8/3K4/8 w - - 0 23';
  const result = new GardnerSearcher({ hashEntries: 131072 }).analyze(EnginePosition.fromFEN(fen), {
    timeMs: 650,
    maxDepth: 9,
    multipv: 2,
    mateProbeMs: 120,
    mateMaxPlies: 25
  });
  assert.equal(result.engine, ENGINE_VERSION);
  assert.ok(result.lines.length >= 1);
}

{
  const oldFetch = globalThis.fetch;
  let fetches = 0;
  const wdl = new Uint8Array(512);
  // Packed exact WDL code 1 means draw according to EXACT_MAP.
  wdl.fill(0x55);
  const dtm = new Uint8Array(4096);
  globalThis.fetch = async urlLike => {
    fetches += 1;
    const url = String(urlLike);
    if (url.endsWith('/manifest.json')) {
      return Response.json({ tables: { KvK: { path: 'KvK/metadata.json' } } });
    }
    if (url.endsWith('/practical-manifest.json')) {
      return Response.json({ tables: {} });
    }
    if (url.endsWith('/KvK/metadata.json')) {
      return Response.json({ blockSize: 2048, blocks: [{ wdl: 'wdl.bin', dtm: 'dtm.bin', count: 2048 }] });
    }
    if (url.endsWith('/KvK/wdl.bin')) return new Response(wdl);
    if (url.endsWith('/KvK/dtm.bin')) return new Response(dtm);
    return new Response('missing', { status: 404 });
  };

  const { GardnerTablebase } = await import(`../js/engine/tablebase.js?cacheTest=${Date.now()}`);
  const tablebase = new GardnerTablebase({ baseUrl: 'https://tb.test/tables/' });
  const position = EnginePosition.fromFEN('4k/5/5/5/K4 w - - 0 1');

  const probe1 = await tablebase.probe(position);
  assert.equal(probe1?.wdl, 0, 'mock exact tablebase should return draw');
  const afterFirstProbe = fetches;
  const probe2 = await tablebase.probe(position);
  assert.equal(probe2?.wdl, 0);
  assert.equal(fetches, afterFirstProbe, 'second probe of same position should be served from probe cache');

  const analysis1 = await tablebase.analyze(position, { multipv: 2, maxPvPly: 4 });
  assert.equal(analysis1?.tablebase, true, 'mock exact tablebase should produce an analysis result');
  const afterFirstAnalysis = fetches;
  const analysis2 = await tablebase.analyze(position, { multipv: 2, maxPvPly: 4 });
  assert.equal(analysis2?.tablebase, true);
  assert.equal(fetches, afterFirstAnalysis, 'second identical tablebase analysis should be served from analysis cache');

  globalThis.fetch = oldFetch;
}

{
  const storage = new Map();
  globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => { storage.set(key, String(value)); },
    removeItem: key => { storage.delete(key); }
  };
  const key = 'v14-test-position-key';
  const result = {
    engine: ENGINE_VERSION,
    lines: [{ move: 'a1a2', score: 0, pv: ['a1a2'] }],
    completed: true
  };
  const cache = new AnalysisCache(globalThis.localStorage);
  cache.set(key, result);
  const cached = new AnalysisCache(globalThis.localStorage).get(key);
  assert.equal(cached?.engine, ENGINE_VERSION, 'v15.1 analysis cache should round-trip current engine identity');
  assert.ok([...storage.keys()].some(key => key.includes('v15')), 'persistent cache key should be versioned to v15.1');
}

console.log('v15.1 efficiency and tablebase-cache tests passed.');
