import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Position } from '../js/core/position.js';
import { gameStatus, legalMoves } from '../js/core/rules.js';
import { findMoveBySAN, moveToSAN } from '../js/core/notation.js';
import { flattenTree, parsePGN } from '../js/core/pgn.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

const initial = Position.initial();
assert.equal(initial.toCompactFEN(), 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
assert.equal(initial.toStudyFEN(), '8/8/1rnbqk2/1ppppp2/8/1PPPPP2/1RNBQK2/8 w - - 0 1');
assert.equal(legalMoves(initial).length, 7);
assert.deepEqual(legalMoves(initial).map(move => moveToSAN(initial, move)).sort(), ['Nb4', 'Nd4', 'b4', 'c4', 'd4', 'e4', 'f4'].sort());

const b4 = findMoveBySAN(initial, 'b4');
assert.ok(b4);
const afterB4 = initial.makeMove(b4);
assert.equal(afterB4.turn, 'b');
assert.equal(afterB4.toStudyFEN().split(' ')[0], '8/8/1rnbqk2/1ppppp2/1P6/2PPPP2/1RNBQK2/8');

const promotion = Position.fromFEN('4k/P4/5/5/4K w - - 0 1');
const promotionMoves = legalMoves(promotion).filter(move => move.from === 15);
assert.equal(promotionMoves.length, 4);
assert.deepEqual(new Set(promotionMoves.map(move => move.promotion)), new Set(['q', 'r', 'b', 'n']));

const bareKings = Position.fromFEN('4k/5/5/5/K4 w - - 0 1');
assert.equal(gameStatus(bareKings).state, 'draw-insufficient');

const mate = Position.fromFEN('4k/3Q1/2K2/5/5 b - - 0 1');
assert.equal(gameStatus(mate).state, 'checkmate');

for (const file of fs.readdirSync(path.join(root, 'data/pgn')).filter(name => name.endsWith('.pgn'))) {
  const text = fs.readFileSync(path.join(root, 'data/pgn', file), 'utf8');
  const parsed = parsePGN(text, file);
  assert.ok(flattenTree(parsed.root).length > 1, `${file} should contain parsed nodes`);
}

console.log('All Gardner MiniChess core tests passed.');
