import assert from 'node:assert/strict';
import {
  EnginePosition,
  EngineInternals,
  GardnerSearcher,
  generateLegalMoves,
  isInCheck,
  moveToUci,
  uciToMove
} from '../js/engine/engine.js';

// Regression for 1.a3 bxa3 2.Rxa3. In v4 an all-pruned non-PV node could
// return -INF; negation at the root turned that sentinel into a false mate for
// 2...Bc5. Bc5 is not even check because the e3 pawn blocks the diagonal.
{
  const fen = '8/8/1rnbqk2/1p1ppp2/1R6/2PPPP2/2NBQK2/8 b - - 0 2';
  const position = EnginePosition.fromFEN(fen);
  const bc5 = uciToMove(position, 'c5b4');
  assert.ok(bc5, 'Bc5 must be legal in the regression position');
  assert.equal(EngineInternals.givesCheck(position, bc5), false, 'Bc5 is not check');
  const state = EngineInternals.makeMove(position, bc5);
  assert.equal(isInCheck(position), false);
  assert.ok(generateLegalMoves(position).length >= 8, 'White has many legal replies after Bc5');
  EngineInternals.undoMove(position, bc5, state);

  const result = new GardnerSearcher({ hashEntries: 65_536 }).analyze(
    EnginePosition.fromFEN(fen),
    { timeMs: 420, maxDepth: 9, multipv: 3 }
  );
  assert.equal(result.rejectedMateClaims, 0, 'The fixed search must not produce an INF/mate sentinel');
  assert.ok(result.lines.every(line => !line.scoreText.includes('#')), 'No move in this position is mate in one');
  const bc5Line = result.lines.find(line => line.move === 'c5b4');
  if (bc5Line) assert.equal(bc5Line.mateVerified, false);
}

// The low-material proof search returns an exact DTM line and its PV must end
// in actual checkmate.
{
  const fen = '4k/5/3Q1/2K2/5 w - - 0 1';
  const root = EnginePosition.fromFEN(fen);
  const result = new GardnerSearcher({ hashEntries: 32_768 }).analyze(root, {
    timeMs: 120,
    maxDepth: 5,
    multipv: 2,
    endgameProbeMs: 300
  });
  assert.equal(result.endgameProof, true);
  const best = result.lines[0];
  assert.equal(best.endgameProof, true);
  assert.equal(best.mateVerified, true);
  assert.ok(best.dtm > 0 && best.dtm <= 11);
  const replay = EnginePosition.fromFEN(fen);
  for (const uci of best.pv) {
    const move = uciToMove(replay, uci);
    assert.ok(move, `Proof PV contains illegal move ${uci}`);
    EngineInternals.makeMove(replay, move);
  }
  assert.equal(generateLegalMoves(replay).length, 0);
  assert.equal(isInCheck(replay), true);
}

// Analysis results no longer expose heuristic W/D/L percentages.
{
  const result = new GardnerSearcher({ hashEntries: 16_384 }).analyze(
    EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1'),
    { timeMs: 100, maxDepth: 3, multipv: 1 }
  );
  assert.ok(result.lines.length);
  assert.equal('wdl' in result.lines[0], false);
}

console.log('Orion regression and endgame-proof tests passed.');
