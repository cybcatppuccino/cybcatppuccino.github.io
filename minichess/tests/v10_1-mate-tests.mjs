import assert from 'node:assert/strict';
import {
  EnginePosition,
  GardnerSearcher,
  EngineInternals,
  uciToMove
} from '../js/engine/engine.js';

const { MATE } = EngineInternals;

// v10.1: a transient, not-yet-replayable mate score must not overwrite the
// previous stable centipawn score with a tiny static fallback.  This removes
// the visible "large edge -> small edge -> verified mate" jump seen just before
// a mate PV becomes complete.
{
  const position = EnginePosition.fromFEN('4k/5/3Q1/2K2/5 w - - 0 1');
  const move = uciToMove(position, 'e4e3');
  const searcher = new GardnerSearcher({ hashEntries: 32768 });
  searcher.previousRootScores.set(move, 1040);
  searcher.previousPVScore[0] = 1015;
  const clean = searcher.sanitizeRootLines(position, [{
    move,
    score: MATE - 5,
    pv: [move]
  }]);
  assert.equal(clean[0].mateVerified, false);
  assert.equal(clean[0].matePending, true);
  assert.ok(clean[0].score >= 1000, `expected stable large edge, got ${clean[0].score}`);
}

// The dedicated forced-mate proof probe can find a mate even when the normal
// depth-limited alpha-beta pass is intentionally too shallow to prove it.
{
  const position = EnginePosition.fromFEN('4k/5/2Q2/2K2/5 w - - 0 1');
  const result = new GardnerSearcher({ hashEntries: 131072 }).analyze(position, {
    timeMs: 80,
    maxDepth: 1,
    multipv: 2,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 500,
    mateMaxPlies: 9
  });
  assert.equal(result.lines[0].mateVerified, true);
  assert.equal(result.lines[0].mateProof, true);
  assert.ok(result.lines[0].dtm > 0 && result.lines[0].dtm <= 9);
  assert.match(result.lines[0].scoreText, /^#/);
}

console.log('v10.1 mate stability and proof-search tests passed.');
