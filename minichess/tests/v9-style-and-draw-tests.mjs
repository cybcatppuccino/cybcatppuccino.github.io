import assert from 'node:assert/strict';
import {
  EnginePosition,
  GardnerSearcher,
  analyzePositionActivity,
  evaluate
} from '../js/engine/engine.js';

// The supplied compact fortress-like position must remain close to equal even
// though Black owns an extra bishop. This assertion is heuristic, not a claim
// that every structurally similar position is mathematically drawn.
{
  const position = EnginePosition.fromFEN('5/3b1/p1k1p/P3P/1K3 b - - 0 1');
  const activity = analyzePositionActivity(position);
  assert.ok(activity.closure > 0.75);
  assert.equal(activity.pawnBreaks, 0);
  assert.ok(activity.bothConstrained);
  assert.ok(Math.abs(evaluate(position)) <= 30);
  const result = new GardnerSearcher({ hashEntries: 262144 }).analyze(position, {
    timeMs: 700,
    maxDepth: 22,
    multipv: 3,
    fortressProbeMs: 120,
    endgameProbeMs: 0
  });
  assert.ok(result.lines.length >= 2);
  assert.ok(result.lines.every(line => Math.abs(line.score) < 180));
}

// An open initial position must not be compressed toward a draw merely because
// its current static score is near zero.
{
  const position = EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
  const activity = analyzePositionActivity(position);
  assert.ok(activity.closure < 0.25);
  assert.equal(activity.scale, 1);
  assert.ok(activity.pawnBreaks >= 8);
}

console.log('v9 generic low-progress and closed-position tests passed.');
