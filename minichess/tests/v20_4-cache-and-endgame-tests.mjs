import assert from 'node:assert/strict';
import { AnalysisCache } from '../js/engine/analysis-cache.js';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.has(key) ? this.map.get(key) : null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

{
  const cache = new AnalysisCache(new MemoryStorage());
  const key = 'ordinary-root';
  const ordinary = {
    engine: 'Orion JS 20.4',
    depth: 6,
    scoreDepth: 6,
    completed: true,
    multiPvVerified: true,
    pvComplete: true,
    lines: [{ move: 'a1a2', score: 37, scoreText: '+0.37', pv: ['a1a2'], rootScoreExact: true }]
  };
  assert.equal(cache.set(key, ordinary), null, 'ordinary numeric analysis must not enter or return from the durable cache');
  assert.equal(cache.get(key), null, 'ordinary numeric analysis must not be reused after a played move');
}

{
  const searcher = new GardnerSearcher({ hashEntries: 32768 });
  const first = EnginePosition.fromFEN('3k1/3pp/5/2PPP/2K2 b - - 3 2');
  const result = searcher.analyze(first, {
    timeMs: 500,
    maxDepth: 5,
    multipv: 2,
    endgameProbeMs: 50,
    fortressProbeMs: 0,
    mateProbeMs: 120
  });
  assert.ok(result.lines.length >= 1);
  assert.ok(searcher.ttOccupied > 0, 'the first root should fill the ordinary TT');
  searcher.mateProofMisses.set('synthetic-miss', 100);
  searcher.endgameProofMisses.set('synthetic-endgame-miss', 100);
  searcher.beginPosition();
  assert.equal(searcher.ttOccupied, 0, 'new root analysis must not inherit ordinary TT entries');
  assert.equal(searcher.mateProofMisses.size, 0, 'new root analysis must not inherit mate-proof misses');
  assert.equal(searcher.endgameProofMisses.size, 0, 'new root analysis must not inherit endgame-proof misses');
}

{
  const fen = '3k1/K2p1/3Pp/2P1P/5 w - - 4 7';
  const searcher = new GardnerSearcher({ hashEntries: 65536 });
  const result = searcher.analyze(EnginePosition.fromFEN(fen), {
    timeMs: 750,
    maxDepth: 1,
    startDepth: 1,
    multipv: 5,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 520,
    mateMaxPlies: 15
  });
  assert.equal(result.lines[0]?.mateVerified, true, 'the near-terminal pawn ending must still be proven as mate');
  assert.equal(result.lines[0]?.scoreText, '#5');
}

{
  const fen = '3k1/3pp/5/2PPP/2K2 b - - 3 2';
  const searcher = new GardnerSearcher({ hashEntries: 65536 });
  const result = searcher.analyze(EnginePosition.fromFEN(fen), {
    timeMs: 1300,
    maxDepth: 6,
    startDepth: 1,
    multipv: 3,
    endgameProbeMs: 100,
    fortressProbeMs: 0,
    mateProbeMs: 220,
    mateMaxPlies: 15
  });
  assert.ok(result.lines.length >= 1);
  assert.equal(result.lines[0].scoreNumeric, true, 'no unproven mate/draw label should replace a real numeric score');
  assert.ok(Number.isFinite(Number(result.lines[0].score)));
  assert.ok(result.lines.some(line => line.move === 'd5c4'), 'the Stockfish-matching king defence should remain visible in the candidate set');
}

console.log('v20.4 cache policy and endgame proof tests passed.');
