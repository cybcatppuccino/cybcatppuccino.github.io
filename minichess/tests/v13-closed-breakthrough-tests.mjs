import assert from 'node:assert/strict';
import { EnginePosition, EngineInternals, GardnerSearcher, analyzeOnce, generateLegalMoves } from '../js/engine/engine.js';

// v13 regression target supplied during the closed-position investigation:
// 1.c3 d3 2.a3 b3 3.e3 Bxa3 4.Rxa3 Nxa3 5.Nxa3 ...
// The important property is not one memorised continuation, but that the
// engine no longer treats the first reversible cycle as a hard 0.00 proof.
const BEFORE_NXA3 = 'r2qk/p1p1p/npPpP/1P1P1/1NBQK w - - 0 5';
const AFTER_NXA3 = 'r2qk/p1p1p/NpPpP/1P1P1/2BQK b - - 0 5';

{
  const pos = EnginePosition.fromFEN(AFTER_NXA3);
  assert.equal(
    EngineInternals.lowProgressSearchNode(pos, generateLegalMoves(pos)),
    true,
    'Queenful locked 5×5 positions should still receive closed-search safeguards'
  );
}

{
  const pos = EnginePosition.fromFEN(BEFORE_NXA3);
  const searcher = new GardnerSearcher({ hashEntries: 16_384 });
  searcher.hashStackA[0] = pos.hashA;
  searcher.hashStackB[0] = pos.hashB;
  searcher.rootRepetition.set(`${pos.hashA}:${pos.hashB}`, 1);
  const cycleScore = searcher.searchCycleScore(pos, 2);
  assert.notEqual(cycleScore, null, 'A second in-search occurrence should be detected');
  assert.notEqual(cycleScore, 0, 'Twofold search cycles should no longer be flattened to an unconditional 0.00');
}

{
  const result = analyzeOnce(BEFORE_NXA3, {
    timeMs: 60_000,
    maxDepth: 10,
    multipv: 5,
    fortressProbeMs: 0,
    endgameProbeMs: 0,
    mateProbeMs: 0
  });
  const nxa3 = result.lines.find(line => line.move === 'b1a3');
  assert.ok(nxa3, '5.Nxa3 should be searched as a legal candidate');
  assert.ok(nxa3.score < 0, `5.Nxa3 should not remain an exact 0.00 repetition artifact; got ${nxa3.score}`);
}

{
  const result = analyzeOnce(AFTER_NXA3, {
    timeMs: 60_000,
    maxDepth: 14,
    multipv: 5,
    fortressProbeMs: 0,
    endgameProbeMs: 0,
    mateProbeMs: 0
  });
  assert.ok(result.lines.length >= 3);
  assert.ok(
    result.lines[0].score < 0,
    `After 5.Nxa3, Black should not be flattened to an exact repetition draw; got ${result.lines[0].move} ${result.lines[0].score}`
  );
}

console.log('v13 closed breakthrough regression tests passed.');
