import assert from 'node:assert/strict';
import {
  EnginePosition,
  GardnerSearcher,
  analyzePositionActivity,
  analyzeOnce
} from '../js/engine/engine.js';
import { FAIRY_STOCKFISH_LABEL, validateExternalAnalysisResult } from '../js/engine/external-engine.js';

// v14 regression: a high-material but fully locked queen/rook position.  The
// engine must not preserve a large static material bonus when neither side has
// captures, checks, pawn breaks, promotions, king entries, or enabling moves.
const QUEENFUL_DEADLOCK = 'rq2k/p1p1p/PpPpP/1B1P1/R1Q1K w - - 5 8';

{
  const pos = EnginePosition.fromFEN(QUEENFUL_DEADLOCK);
  const profile = analyzePositionActivity(pos);
  assert.equal(profile.legal !== null, true, 'Queenful locked deadlocks should run the legal low-progress classifier');
  assert.equal(profile.lowProgressDraw, true, 'The no-progress queenful deadlock should be compressed as a practical draw');
  assert.ok(profile.scale <= 0.02, `Expected near-zero draw scale, got ${profile.scale}`);
}

{
  const result = analyzeOnce(QUEENFUL_DEADLOCK, {
    timeMs: 4_000,
    maxDepth: 8,
    multipv: 3,
    fortressProbeMs: 120,
    endgameProbeMs: 0,
    mateProbeMs: 0
  });
  assert.ok(result.lines.length >= 1);
  assert.ok(Math.abs(result.lines[0].score) <= 10, `Deadlocked position should remain near 0.00, got ${result.lines[0].score}`);
}

// Keep the v13 breakthrough class distinct: contact captures around the knight
// prevent the queenful deadlock compression from hiding ...Qb5-style resources.
{
  const tactical = EnginePosition.fromFEN('r2qk/p1p1p/NpPpP/1P1P1/2BQK b - - 0 5');
  const profile = analyzePositionActivity(tactical);
  assert.equal(profile.legal, null, 'Tactical contact in a closed position should not be classified as a deadlock');
  assert.ok(profile.rawContacts > 0, 'The tactical position should expose contact attacks that block deadlock compression');
}

// External UCI providers are optional and never trusted blindly.  The adapter
// must keep legal Fairy-Stockfish PVs and reject illegal ones before the UI or
// AI play layer can consume them.
{
  const root = EnginePosition.fromFEN(QUEENFUL_DEADLOCK);
  const legal = validateExternalAnalysisResult(root, {
    engine: FAIRY_STOCKFISH_LABEL,
    depth: 6,
    nodes: 343,
    nps: 9800,
    elapsed: 35,
    lines: [{ move: 'a1b1', score: 143, scoreText: '+1.43', pv: ['a1b1', 'b5c5', 'c1d1'] }]
  });
  assert.ok(legal);
  assert.equal(legal.lines[0].move, 'a1b1');
  assert.equal(legal.lines[0].source, 'fairy-stockfish');

  const illegal = validateExternalAnalysisResult(root, {
    engine: FAIRY_STOCKFISH_LABEL,
    depth: 6,
    lines: [{ move: 'a1a5', score: 0, pv: ['a1a5'] }]
  });
  assert.equal(illegal, null, 'Illegal external PVs must be discarded so callers can fall back to Orion JS');
}

console.log('v14 closed deadlock and external-engine validation tests passed.');
