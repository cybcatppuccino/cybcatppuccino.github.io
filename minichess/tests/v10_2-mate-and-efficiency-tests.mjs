import assert from 'node:assert/strict';
import {
  EnginePosition,
  GardnerSearcher,
  EngineInternals,
  generateLegalMoves,
  moveToUci,
  uciToMove
} from '../js/engine/engine.js';

// v10.2 regression for the reported K+P race.  When a mate-like score appears
// before the PV is replay-verifiable, the UI-safe score must retain the latest
// stable large advantage instead of falling back to a tiny one-ply static eval.
{
  const fen = '8/8/8/1p3p2/1k3P2/8/3K4/8 w - - 0 23';
  const searcher = new GardnerSearcher({ hashEntries: 131072 });
  let previousBestAbs = 0;
  for (let depth = 1; depth <= 21; depth += 1) {
    const result = searcher.analyze(EnginePosition.fromFEN(fen), {
      timeMs: 100000,
      maxDepth: depth,
      startDepth: depth,
      multipv: 4,
      mateProbeMs: 0,
      newPosition: depth === 1
    });
    const cpLines = result.lines.filter(line => !line.mateVerified).map(line => Math.abs(line.score));
    const bestAbs = cpLines.length ? Math.max(...cpLines) : previousBestAbs;
    if (depth >= 17) {
      assert.ok(bestAbs >= 900, `depth ${depth} should keep a stable large score, got ${bestAbs}`);
    }
    previousBestAbs = bestAbs;
  }
}

// The exact low-material mate prover should solve a compact king/pawn mate
// without relying on the normal alpha-beta horizon.
{
  const fen = '8/8/8/1p3p2/1k3P2/8/4K3/8 b - - 1 23';
  const searcher = new GardnerSearcher({ hashEntries: 131072 });
  const proof = searcher.proveForcedMate(EnginePosition.fromFEN(fen), {
    timeMs: 1500,
    maxPlies: 13
  });
  assert.ok(proof, 'expected an exact low-material forced mate proof');
  assert.equal(proof.mateVerified, true);
  assert.equal(proof.exactLowMaterialProof, true);
  assert.ok(proof.dtm > 0 && proof.dtm <= 13, `unexpected DTM ${proof.dtm}`);
}

// Incremental king-square maintenance must survive make/undo and match legal
// move generation after a forcing sequence.
{
  const pos = EnginePosition.fromFEN('8/8/8/1p3p2/1k3P2/8/4K3/8 b - - 1 23');
  const initialWhiteKing = EngineInternals.kingSquare(pos, EngineInternals.WHITE);
  const initialBlackKing = EngineInternals.kingSquare(pos, EngineInternals.BLACK);
  const states = [];
  for (const uci of ['b4c3', 'e2f2', 'c3d3']) {
    const move = uciToMove(pos, uci);
    assert.ok(move, `${uci} should be legal`);
    states.push([move, EngineInternals.makeMove(pos, move)]);
  }
  assert.equal(EngineInternals.kingSquare(pos, EngineInternals.BLACK), EngineInternals.square(2, 1));
  while (states.length) {
    const [move, state] = states.pop();
    EngineInternals.undoMove(pos, move, state);
  }
  assert.equal(EngineInternals.kingSquare(pos, EngineInternals.WHITE), initialWhiteKing);
  assert.equal(EngineInternals.kingSquare(pos, EngineInternals.BLACK), initialBlackKing);
  assert.deepEqual(generateLegalMoves(pos).map(moveToUci).sort(), ['b4b3', 'b4c3', 'b4c4', 'b4c5'].sort());
}

console.log('v10.2 mate stability and efficiency tests passed.');
