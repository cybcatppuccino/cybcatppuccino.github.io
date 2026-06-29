import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { EnginePosition, GardnerSearcher, EngineInternals, generateLegalMoves, moveToUci } from '../js/engine/engine.js';
const { makeMove, isInCheck } = EngineInternals;

function applyPv(fen, pv) {
  const pos = EnginePosition.fromFEN(fen);
  for (const uci of pv) {
    const move = generateLegalMoves(pos, false).find(candidate => moveToUci(candidate) === uci);
    assert.ok(move, `PV move ${uci} must be legal`);
    makeMove(pos, move);
  }
  return pos;
}

{
  const fen = '3k1/K2p1/3Pp/2P1P/5 w - - 4 7';
  const pos = EnginePosition.fromFEN(fen);
  const searcher = new GardnerSearcher({ hashEntries: 1 << 16 });
  const result = searcher.analyze(pos, {
    timeMs: 700,
    maxDepth: 1,
    startDepth: 1,
    multipv: 5,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 500,
    mateMaxPlies: 15
  });
  const best = result.lines[0];
  assert.equal(best.mateVerified, true, 'The 7-piece pawn ending must be proven as a full-width mate, not left as +1.xx.');
  assert.equal(best.scoreText, '#5');
  assert.ok(best.mateProof, 'The proof must come from the verified mate prover, not a tablebase label or fallback score.');
  assert.equal(best.pv.length, 9, 'Mate in 5 on the UI is 9 plies from the root.');
  const terminal = applyPv(fen, best.pv);
  assert.equal(generateLegalMoves(terminal, false).length, 0, 'The proof PV must terminate in no legal replies.');
  assert.equal(isInCheck(terminal), true, 'The terminal node must be checkmate, not stalemate or a fallback label.');
}

{
  const fen = '3k1/3pp/5/2PPP/2K2 b - - 3 2';
  const pos = EnginePosition.fromFEN(fen);
  const searcher = new GardnerSearcher({ hashEntries: 1 << 16 });
  const result = searcher.analyze(pos, {
    timeMs: 1200,
    maxDepth: 6,
    startDepth: 6,
    multipv: 5,
    endgameProbeMs: 80,
    fortressProbeMs: 0,
    mateProbeMs: 160,
    mateMaxPlies: 15
  });
  assert.ok(result.lines.length >= 3);
  assert.notEqual(result.lines[0].mateVerified, true, 'No short forced mate should be fabricated for this black-to-move root.');
  assert.notEqual(result.lines[0].scoreNumeric, false, 'When no mate is proven, the best line must remain a real numeric score.');
  assert.ok(Number.isFinite(Number(result.lines[0].score)), 'The black defence must publish a real white-centric score.');
  assert.ok(result.lines.every(line => line.scoreNumeric !== false && Number.isFinite(Number(line.score))), 'All visible black defences must remain numeric ordinary scores when no mate is proven.');
  assert.ok(result.lines.some(line => /^d5[ce][45]$/.test(line.move)), 'King defences must stay in the candidate set instead of being displaced by proof artefacts.');
  assert.ok(Number(result.lines[0].score) > 0, 'The score is white-centric: black is worse even after the best defence.');
}

{
  const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
  assert.doesNotMatch(worker, /queueWideTablebaseBridgeProof\(token, 0, current\.lastResult, \{ delay: 0 \}\)/, 'The broad bridge prover must not start before ordinary analysis has a baseline.');
  assert.match(worker, /lineHasPublishableScore/, 'Stable publication must reject unverified WDL or mate-pending blank scores.');
  assert.match(worker, /mateBudget = stableMate/, 'Mate proof budget must be explicitly bounded relative to ordinary search.');
  assert.match(worker, /TABLEBASE_WIDE_BRIDGE_TIME_SLICES/, 'Wide bridge work must use progressive slices instead of one blocking 2600ms task.');
}

console.log('v20.3 mate scheduler tests passed.');
