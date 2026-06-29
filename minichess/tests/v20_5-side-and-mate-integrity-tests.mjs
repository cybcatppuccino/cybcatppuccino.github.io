import assert from 'node:assert/strict';
import { analyzeOnce, EnginePosition, uciToMove, EngineInternals } from '../js/engine/engine.js';

function utility(line, side) {
  return (side === 1 ? 1 : -1) * Number(line?.score || 0);
}

const reported = 'r1k2/P1p1p/1pP1P/1B3/R2K1 b - - 12 15';
{
  const root = EnginePosition.fromFEN(reported);
  const result = analyzeOnce(reported, {
    timeMs: 8_000,
    maxDepth: 9,
    multipv: 5,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 0
  });
  assert.ok(result.lines.length >= 3, 'reported root should expose multiple legal candidates');
  assert.equal(result.lines[0].move, 'c5d5', 'black must keep the strongest defence first, not the losing Rb5/Ra4 mate lines');
  assert.notEqual(result.lines[0].move, 'a5b5', 'black must not choose Rb5 when a stronger defence exists');
  const bestUtility = utility(result.lines[0], root.turn);
  for (const line of result.lines.slice(1)) {
    assert.ok(bestUtility >= utility(line, root.turn), `line ${line.move} is ordered above a better black utility`);
  }
  const rb5 = result.lines.find(line => line.move === 'a5b5');
  assert.ok(rb5?.mateVerified, 'Rb5 may be shown only as a verified losing mate line, not as black best');
  assert.ok(utility(rb5, root.turn) < bestUtility, 'Rb5 must be worse for black than the selected defence');
}

{
  const result = analyzeOnce('3k1/K2p1/3Pp/2P1P/5 w - - 4 7', {
    timeMs: 6_000,
    maxDepth: 9,
    multipv: 3,
    endgameProbeMs: 200,
    fortressProbeMs: 0,
    mateProbeMs: 900,
    mateMaxPlies: 17
  });
  assert.ok(result.lines[0].scoreNumeric !== false, 'visible best line must remain either a score or a verified mate score');
  if (result.lines[0].mateVerified) {
    assert.ok(Math.abs(result.lines[0].score) >= EngineInternals.MATE_BOUND, 'verified mate must carry a mate score');
    assert.match(result.lines[0].scoreText, /^#\d+|-#\d+$/, 'verified mate must display #N, not a label/fallback');
  } else {
    assert.ok(Number.isFinite(Number(result.lines[0].score)), 'non-mate best line must display a real numeric score');
    assert.match(result.lines[0].scoreText, /^[+-]?\d+\.\d{2}$/, 'non-mate best line must display a numeric score');
  }
}

{
  const root = EnginePosition.fromFEN(reported);
  const move = uciToMove(root, 'c5d5');
  assert.ok(move, 'reported best move should be legal');
  EngineInternals.makeMove(root, move);
  const child = analyzeOnce('r4/P1pkp/1pP1P/1B3/R2K1 w - - 13 16', {
    timeMs: 5_000,
    maxDepth: 8,
    multipv: 3,
    endgameProbeMs: 80,
    fortressProbeMs: 80,
    mateProbeMs: 240,
    mateMaxPlies: 21
  });
  assert.ok(child.lines.length, 'follow-up search after the recommended defence should still produce a result');
  assert.ok(child.lines[0].scoreNumeric !== false, 'follow-up result must be a real score or verified mate');
}

console.log('v20.5 side-to-move and mate display integrity tests passed.');
