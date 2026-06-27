import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  COORD_SYSTEMS,
  legacyStudyCoord,
  parseLegacyStudyCoord,
  parseStandardCoord,
  square,
  standardCoord
} from '../js/core/constants.js';
import { Position } from '../js/core/position.js';
import { gameStatus, legalMoves } from '../js/core/rules.js';
import { findMoveBySAN, moveToSAN, moveToUci } from '../js/core/notation.js';
import { flattenTree, parsePGN } from '../js/core/pgn.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

for (let rank = 0; rank < 5; rank += 1) {
  for (let file = 0; file < 5; file += 1) {
    const sq = square(file, rank);
    assert.equal(parseStandardCoord(standardCoord(sq)), sq, 'standard A1–E5 coordinates must round-trip');
    assert.equal(parseLegacyStudyCoord(legacyStudyCoord(sq)), sq, 'legacy b2–f6 coordinates must round-trip');
  }
}
assert.equal(standardCoord(0), 'a1');
assert.equal(legacyStudyCoord(0), 'b2');
assert.equal(standardCoord(24), 'e5');
assert.equal(legacyStudyCoord(24), 'f6');

const initial = Position.initial();
assert.equal(initial.toCompactFEN(), 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
assert.equal(initial.toStandardFEN(), 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
assert.equal(initial.toLegacyStudyFEN(), '8/8/1rnbqk2/1ppppp2/8/1PPPPP2/1RNBQK2/8 w - - 0 1');
assert.equal(Position.fromFEN(initial.toLegacyStudyFEN()).toStandardFEN(), initial.toStandardFEN(), 'legacy study FEN must still import');
assert.equal(Position.fromFEN('8/8/8/rnbqk3/ppppp3/8/PPPPP3/RNBQK3 w - - 0 1').toStandardFEN(), initial.toStandardFEN(), 'standard padded FEN must import');
assert.equal(legalMoves(initial).length, 7);
assert.deepEqual(legalMoves(initial).map(move => moveToSAN(initial, move)).sort(), ['Na3', 'Nc3', 'a3', 'b3', 'c3', 'd3', 'e3'].sort());
assert.deepEqual(legalMoves(initial).map(move => moveToUci(move)).sort(), ['a2a3', 'b1a3', 'b1c3', 'b2b3', 'c2c3', 'd2d3', 'e2e3'].sort());

const a3 = findMoveBySAN(initial, 'a3');
assert.ok(a3);
const legacyB4 = findMoveBySAN(initial, 'b4', { coordSystem: COORD_SYSTEMS.LEGACY_STUDY });
assert.deepEqual(legacyB4, a3, 'legacy SAN b4 must resolve to standard a3');
const afterA3 = initial.makeMove(a3);
assert.equal(afterA3.turn, 'b');
assert.equal(afterA3.toStandardFEN().split(' ')[0], 'rnbqk/ppppp/P4/1PPPP/RNBQK');
assert.equal(moveToSAN(initial, a3, { coordSystem: COORD_SYSTEMS.LEGACY_STUDY }), 'b4');

const promotion = Position.fromFEN('4k/P4/5/5/4K w - - 0 1');
const promotionMoves = legalMoves(promotion).filter(move => move.from === 15);
assert.equal(promotionMoves.length, 4);
assert.deepEqual(new Set(promotionMoves.map(move => move.promotion)), new Set(['q', 'r', 'b', 'n']));

const bareKings = Position.fromFEN('4k/5/5/5/K4 w - - 0 1');
assert.equal(gameStatus(bareKings).state, 'draw-insufficient');

const mate = Position.fromFEN('4k/3Q1/2K2/5/5 b - - 0 1');
assert.equal(gameStatus(mate).state, 'checkmate');

const expectedArchiveErrors = new Map([
  ['Gardneranalysis.pgn', 17],
  ['Gardnerblackoracle_whitemovesb4.pgn', 92]
]);
for (const file of fs.readdirSync(path.join(root, 'data/pgn')).filter(name => name.endsWith('.pgn'))) {
  const text = fs.readFileSync(path.join(root, 'data/pgn', file), 'utf8');
  const parsed = parsePGN(text, file, { coordSystem: COORD_SYSTEMS.LEGACY_STUDY });
  assert.ok(flattenTree(parsed.root).length > 1, `${file} should contain parsed nodes`);
  assert.equal(parsed.errors.length, expectedArchiveErrors.get(file) || 0, `${file} should preserve the known legacy parse behavior`);
}

console.log('All Gardner MiniChess core tests passed.');
