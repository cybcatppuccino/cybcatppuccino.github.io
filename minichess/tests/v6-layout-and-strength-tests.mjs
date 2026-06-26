import assert from 'node:assert/strict';
import fs from 'node:fs';
import { START_LAYOUTS, createStartPosition, layoutDefinition } from '../js/core/start-positions.js';
import { parsePGN, StudyLibrary } from '../js/core/pgn.js';
import { AI_LEVELS, selectLineForLevel } from '../js/engine/difficulty.js';

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
assert.equal(independent.white.length, 5);
assert.equal(independent.black.length, 5);
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

const lines = [
  { move: 'a', score: 120 },
  { move: 'b', score: 70 },
  { move: 'c', score: 10 },
  { move: 'd', score: -100 },
  { move: 'e', score: -320 },
  { move: 'f', score: -700 },
  { move: 'g', score: -1200 },
  { move: 'h', score: -1800 },
  { move: 'i', score: -2400 },
  { move: 'j', score: -3000 }
];

const averageRank = level => {
  const config = AI_LEVELS[level - 1];
  const rng = lcg(1000 + level);
  let total = 0;
  const trials = 2500;
  for (let i = 0; i < trials; i += 1) {
    const selected = selectLineForLevel(lines, config, 'w', rng);
    total += lines.indexOf(selected);
  }
  return total / trials;
};

const ranks = [1, 3, 5, 7, 9, 10].map(level => [level, averageRank(level)]);
assert.ok(ranks[0][1] > ranks[1][1], JSON.stringify(ranks));
assert.ok(ranks[1][1] > ranks[2][1], JSON.stringify(ranks));
assert.ok(ranks[2][1] > ranks[3][1], JSON.stringify(ranks));
assert.ok(ranks[3][1] > ranks[4][1], JSON.stringify(ranks));
assert.equal(ranks.at(-1)[1], 0);

console.log('v6 start-layout, Mallett archive and AI-strength-gradient tests passed.', ranks);
