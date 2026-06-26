import assert from 'node:assert/strict';
import {
  EnginePosition,
  GardnerSearcher,
  analyzePositionActivity,
  evaluate
} from '../js/engine/engine.js';

// v10: reversible shuffling in a locked rook+bishop ending must not be valued
// as a real advantage.  The position has many legal black rook/bishop moves,
// but none creates a non-losing breakthrough, sound capture, passer, promotion
// race, or king-entry resource.
{
  const position = EnginePosition.fromFEN('8/8/3k4/1p1pb3/1P2p3/2P1Pr2/1R1BK3/8 b - - 37 26');
  const activity = analyzePositionActivity(position);
  assert.equal(activity.lowProgressDraw, true);
  assert.equal(activity.scale, 0);
  assert.equal(activity.pawnBreaks, 0);
  assert.equal(activity.legal.realPawnBreaks, 0);
  assert.equal(activity.legal.soundCaptures, 0);
  assert.equal(activity.legal.progressMoves, 0);
  assert.equal(evaluate(position), 0);

  const result = new GardnerSearcher({ hashEntries: 262144 }).analyze(position, {
    timeMs: 700,
    maxDepth: 8,
    multipv: 5,
    fortressProbeMs: 0,
    endgameProbeMs: 0
  });
  assert.ok(result.lines.length >= 3);
  assert.ok(result.lines.every(line => line.score === 0 && line.scoreText === '+0.00'));
}

// Open and tactically alive positions must not be compressed merely because
// their static score is modest.
{
  const initial = EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
  const activity = analyzePositionActivity(initial);
  assert.equal(activity.lowProgressDraw, false);
  assert.equal(activity.scale, 1);
}

// A near-promotion pawn is a genuine progress resource, not a fortress signal.
{
  const passer = EnginePosition.fromFEN('4k/3P1/5/5/K4 w - - 0 1');
  const activity = analyzePositionActivity(passer);
  assert.equal(activity.lowProgressDraw, false);
  assert.notEqual(activity.scale, 0);
}

console.log('v10 effective low-progress draw tests passed.');
