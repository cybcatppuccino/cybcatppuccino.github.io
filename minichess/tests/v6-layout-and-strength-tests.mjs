import assert from 'node:assert/strict';
import fs from 'node:fs';
import { START_LAYOUTS, createStartPosition, layoutDefinition } from '../js/core/start-positions.js';
import { parsePGN, StudyLibrary } from '../js/core/pgn.js';
import { AI_STYLES, selectLineForStyle } from '../js/engine/difficulty.js';

function lcg(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

assert.equal(START_LAYOUTS.length, 6);
assert.equal(createStartPosition('standard').position.toCompactFEN(), 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
assert.equal(createStartPosition('central').position.toCompactFEN(), 'kqbnr/ppppp/5/PPPPP/RNBQK w - - 0 1');
assert.equal(createStartPosition('mallet').position.toCompactFEN(), 'rbkqb/ppppp/5/PPPPP/RNKQN w - - 0 1');

const mirror = layoutDefinition('mini60', lcg(9));
assert.deepEqual(mirror.black, mirror.white);
const central = layoutDefinition('mini60-central', lcg(9));
assert.deepEqual(central.black, [...central.white].reverse());
const independent = layoutDefinition('random', lcg(22));
assert.deepEqual([...independent.white].sort(), ['b','k','n','q','r']);
assert.deepEqual([...independent.black].sort(), ['b','k','n','q','r']);

const malletText = fs.readFileSync(new URL('../data/pgn/MalletM25.pgn', import.meta.url), 'utf8');
const malletStudy = parsePGN(malletText, 'mallet-test');
assert.equal(malletStudy.errors.length, 0);
assert.ok(malletStudy.parsedMoves > 1800);
assert.equal(malletStudy.root.position.toCompactFEN(), 'rbkqb/ppppp/5/PPPPP/RNKQN w - - 0 1');
const malletLibrary = new StudyLibrary();
malletLibrary.addStudy(malletStudy);
assert.ok(malletLibrary.bookMoves(createStartPosition('mallet').position).length > 0);

assert.deepEqual(AI_STYLES.map(style => style.id), ['balanced', 'aggressive', 'conservative', 'cunning', 'pressing']);
assert.ok(AI_STYLES.every(style => style.maxDepth >= 36 && style.timeMs >= 3600));

const base = {
  capture: false, promotion: false, check: false, quiet: true, materialExchange: 0,
  sacrifice: 0, opensPosition: 0, closesPosition: 0, pressureEdge: 0, volatility: 35,
  opponentForcing: 1, restriction: 0, opponentSound: 5, opponentLegal: 6,
  spaceEdge: 0, replyGap: 0, goodReplyCount: 3, bestReplyQuiet: false,
  bestReplyForcing: false
};
const lines = [
  { move: 'best', score: 100, styleProfile: { ...base } },
  { move: 'attack', score: 82, styleProfile: { ...base, check: true, capture: true, quiet: false, materialExchange: 1, sacrifice: 30, opensPosition: .5, pressureEdge: 3, volatility: 110 } },
  { move: 'safe', score: 89, styleProfile: { ...base, volatility: 8, opponentForcing: 0, opponentSound: 3 } },
  { move: 'trap', score: 84, styleProfile: { ...base, replyGap: 95, goodReplyCount: 1, bestReplyQuiet: true, opponentLegal: 9 } },
  { move: 'press', score: 86, styleProfile: { ...base, restriction: 4, opponentSound: 1, spaceEdge: 4, pressureEdge: 4 } },
  { move: 'blunder', score: -300, styleProfile: { ...base, check: true, pressureEdge: 10 } }
];

assert.equal(selectLineForStyle(lines, 'balanced', 'w').move, 'best');
assert.equal(selectLineForStyle(lines, 'aggressive', 'w').move, 'attack');
assert.equal(selectLineForStyle(lines, 'conservative', 'w').move, 'safe');
assert.equal(selectLineForStyle(lines, 'cunning', 'w').move, 'trap');
assert.equal(selectLineForStyle(lines, 'pressing', 'w').move, 'press');
for (const style of AI_STYLES) assert.notEqual(selectLineForStyle(lines, style, 'w').move, 'blunder');

console.log('v9 start-layout, Mallett archive and maximum-strength style tests passed.');
