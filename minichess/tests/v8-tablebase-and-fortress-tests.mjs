import assert from 'node:assert/strict';
import {
  EnginePosition,
  GardnerSearcher,
  evaluate
} from '../js/engine/engine.js';

// Two locked wing pawns plus an opposite-colour bishop is a classic compact-
// board horizon trap. Material must not be reported as a decisive advantage
// before search proves an actual pawn breakthrough.
{
  const fen = '5/3b1/p1k1p/P3P/1K3 b - - 0 1';
  const position = EnginePosition.fromFEN(fen);
  assert.ok(Math.abs(evaluate(position)) <= 30, 'Locked wrong-bishop structure should be evaluated as drawish');
  const result = new GardnerSearcher({ hashEntries: 262_144 }).analyze(position, {
    timeMs: 650,
    maxDepth: 24,
    multipv: 3,
    endgameProbeMs: 0,
    fortressProbeMs: 0
  });
  assert.ok(result.lines.length >= 2);
  assert.ok(result.lines.every(line => Math.abs(line.score) < 180), 'No line should retain a false bishop-sized advantage');
}

console.log('v8 locked-position and draw-stability tests passed.');
