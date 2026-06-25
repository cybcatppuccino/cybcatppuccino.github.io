import assert from 'node:assert/strict';
import {
  EnginePosition,
  EngineInternals,
  GardnerSearcher,
  generateLegalMoves,
  isInsufficientMaterial,
  moveToUci,
  uciToMove
} from '../js/engine/engine.js';
import { Position } from '../js/core/position.js';
import { Rules } from '../js/core/rules.js';
import { moveToUci as coreMoveToUci } from '../js/core/notation.js';

const INITIAL = 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1';

assert.throws(() => EnginePosition.fromFEN('k7/8/1rnbqk2/1ppppp2/8/1PPPPP2/1RNBQK2/8 w - - 0 1'), /outside/);
assert.throws(() => EnginePosition.fromFEN('5/5/5/5/K4 w - - 0 1'), /one king per side/);

function deterministicRandom(seed = 0x5a17c9e3) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function engineMoves(fen) {
  return generateLegalMoves(EnginePosition.fromFEN(fen)).map(moveToUci).sort();
}

function coreMoves(position) {
  return Rules.legalMoves(position).map(coreMoveToUci).sort();
}

// Differential move-generation test: the independent UI rules and engine rules
// must agree throughout a deterministic legal random walk.
const random = deterministicRandom();
for (let game = 0; game < 24; game += 1) {
  let position = Position.initial();
  for (let ply = 0; ply < 28; ply += 1) {
    const expected = coreMoves(position);
    const actual = engineMoves(position.toCompactFEN());
    assert.deepEqual(actual, expected, `Move-generator mismatch in game ${game}, ply ${ply}: ${position.toCompactFEN()}`);
    if (!expected.length) break;
    const legal = Rules.legalMoves(position);
    position = position.makeMove(legal[Math.floor(random() * legal.length)]);
  }
}

// Incremental Zobrist and make/unmake must round-trip exactly.
{
  const position = EnginePosition.fromFEN(INITIAL);
  const originalBoard = Array.from(position.board);
  const original = { hashA: position.hashA, hashB: position.hashB, turn: position.turn, halfmove: position.halfmove, fullmove: position.fullmove };
  const move = uciToMove(position, 'c2d4');
  assert.ok(move, 'Expected c2d4 to be legal');
  const state = EngineInternals.makeMove(position, move);
  const recomputed = position.clone();
  recomputed.recomputeHash();
  assert.equal(position.hashA, recomputed.hashA);
  assert.equal(position.hashB, recomputed.hashB);
  EngineInternals.undoMove(position, move, state);
  assert.deepEqual(Array.from(position.board), originalBoard);
  assert.deepEqual(
    { hashA: position.hashA, hashB: position.hashB, turn: position.turn, halfmove: position.halfmove, fullmove: position.fullmove },
    original
  );
}

// Two occurrences are not a threefold draw; the third occurrence is.
{
  const position = EnginePosition.fromFEN(INITIAL);
  const searcher = new GardnerSearcher({ hashEntries: 16_384 });
  searcher.rootHistory = [{ a: position.hashA, b: position.hashB }];
  searcher.rootRepetition.set(`${position.hashA}:${position.hashB}`, 1);
  searcher.hashStackA[0] = position.hashA;
  searcher.hashStackB[0] = position.hashB;
  assert.equal(searcher.repetitionCount(position, 0), 2);
  assert.equal(searcher.isRepetition(position, 0), false);
  searcher.rootHistory.push({ a: position.hashA, b: position.hashB });
  searcher.rootRepetition.set(`${position.hashA}:${position.hashB}`, 2);
  assert.equal(searcher.repetitionCount(position, 0), 3);
  assert.equal(searcher.isRepetition(position, 0), true);
}

// TT identity includes the halfmove clock so positions near the fifty-move
// horizon cannot reuse an otherwise identical fresh-position score.
{
  const fresh = EnginePosition.fromFEN(INITIAL);
  const nearDraw = EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 99 1');
  assert.equal(fresh.hashA, nearDraw.hashA, 'Repetition hashes should ignore the halfmove clock');
  assert.equal(fresh.hashB, nearDraw.hashB, 'Repetition hashes should ignore the halfmove clock');
  const searcher = new GardnerSearcher({ hashEntries: 16_384 });
  searcher.storeTT(fresh, 5, 123, 0, 0, 17, 0);
  assert.equal(searcher.probeTT(fresh, 0)?.score, 123);
  assert.equal(searcher.probeTT(nearDraw, 0), null);
}

// Quiescence called in check must produce a finite score, even at its depth cap.
{
  const checked = EnginePosition.fromFEN('4k/5/2Q2/5/K4 b - - 0 1');
  const searcher = new GardnerSearcher({ hashEntries: 16_384 });
  searcher.deadline = Infinity;
  searcher.hashStackA[0] = checked.hashA;
  searcher.hashStackB[0] = checked.hashB;
  const score = searcher.qsearch(checked, -32_000, 32_000, 0, 16, 0);
  assert.ok(Number.isFinite(score));
  assert.ok(score > -32_000 && score < 32_000);
}

// Dead-material and elementary winning endings receive the correct terminal treatment.
{
  const bareKings = EnginePosition.fromFEN('4k/5/5/5/K4 w - - 0 1');
  assert.equal(isInsufficientMaterial(bareKings), true);
  const terminal = new GardnerSearcher().analyze(bareKings, { timeMs: 80, maxDepth: 3 });
  assert.equal(terminal.terminal, true);

  const queenEnding = EnginePosition.fromFEN('4k/5/2Q2/5/K4 w - - 0 1');
  assert.equal(isInsufficientMaterial(queenEnding), false);
  const result = new GardnerSearcher().analyze(queenEnding, { timeMs: 600, maxDepth: 7, multipv: 2 });
  assert.ok(result.lines.length > 0);
}

// Every published PV move must remain legal when replayed from the root.
{
  const position = EnginePosition.fromFEN(INITIAL);
  const result = new GardnerSearcher({ hashEntries: 65_536 }).analyze(position, { timeMs: 900, maxDepth: 7, multipv: 3 });
  for (const line of result.lines) {
    const replay = EnginePosition.fromFEN(INITIAL);
    for (const uci of line.pv) {
      const move = uciToMove(replay, uci);
      assert.ok(move, `Illegal PV move ${uci} in ${line.pv.join(' ')}`);
      EngineInternals.makeMove(replay, move);
    }
  }
}

console.log('All Gardner MiniChess engine regression tests passed.');
