import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  EnginePosition,
  GardnerSearcher,
  analyzeOnce,
  generateLegalMoves,
  moveToUci,
  staticExchangeEval
} from '../js/engine/engine.js';
import { COORD_SYSTEMS } from '../js/core/constants.js';
import { Position } from '../js/core/position.js';
import { StudyLibrary, parsePGN } from '../js/core/pgn.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const initialFen = 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1';

const initial = EnginePosition.fromFEN(initialFen);
const legal = generateLegalMoves(initial).map(moveToUci).sort();
assert.deepEqual(legal, ['a2a3', 'b1a3', 'b1c3', 'b2b3', 'c2c3', 'd2d3', 'e2e3'].sort());

const promotion = EnginePosition.fromFEN('4k/P4/5/5/4K w - - 0 1');
assert.equal(generateLegalMoves(promotion).filter(move => moveToUci(move).startsWith('a4a5')).length, 4);

const capture = EnginePosition.fromFEN('4k/5/2p2/1P3/4K w - - 0 1');
const captureMove = generateLegalMoves(capture).find(move => moveToUci(move) === 'b2c3');
assert.ok(captureMove);
assert.ok(Number.isFinite(staticExchangeEval(capture, captureMove)));

const mate = analyzeOnce('4k/3Q1/2K2/5/5 b - - 0 1', { timeMs: 80, maxDepth: 4 });
assert.equal(mate.terminal, true);
assert.equal(mate.lines.length, 0);

const searcher = new GardnerSearcher({ hashEntries: 20000 });
const result = searcher.analyze(EnginePosition.fromFEN(initialFen), { timeMs: 220, maxDepth: 4, multipv: 3 });
assert.ok(result.depth >= 1);
assert.ok(result.lines.length >= 1);
for (let i = 1; i < result.lines.length; i += 1) {
  assert.ok(result.lines[i - 1].score >= result.lines[i].score, 'MultiPV lines must be score-sorted');
}
for (const line of result.lines) assert.ok(legal.includes(line.move), `${line.move} must be legal`);

const library = new StudyLibrary();
const whiteOracle = fs.readFileSync(path.join(root, 'data/pgn/Gardnerwhiteoracle.pgn'), 'utf8');
library.addStudy(parsePGN(whiteOracle, 'white-oracle', { coordSystem: COORD_SYSTEMS.LEGACY_STUDY }));
assert.ok(library.bookMoves(Position.initial()).length > 0, 'The start position should expose archive book moves');

console.log('All Gardner MiniChess engine tests passed.');
