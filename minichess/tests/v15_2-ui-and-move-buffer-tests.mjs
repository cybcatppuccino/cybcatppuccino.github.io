import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { EnginePosition, GardnerSearcher, generateLegalMoves, moveToUci } from '../js/engine/engine.js';

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const engine = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');

assert.doesNotMatch(html, /engineKernelSelect|engineKernelField|Fairy-Stockfish|Stockfish/, 'v15.2 UI must not expose kernel or Stockfish choices');
assert.doesNotMatch(app, /ENGINE_KERNELS|FAIRY_STOCKFISH_LABEL|Fairy-Stockfish|Stockfish/, 'v15.2 app UI path should not expose Stockfish labels');
assert.match(app, /const engineKernel = 'orion-js'/, 'UI dispatch should be pinned to Orion JS');
assert.match(engine, /createMoveList\(\)/, 'engine should define reusable move buffers');
assert.match(engine, /MOVE_FLAG_CHECK/, 'engine should store gives-check metadata on generated legal moves');
assert.match(engine, /generateLegalMovesInto/, 'engine should use buffer-based legal move generation internally');

const root = EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
assert.deepEqual(
  generateLegalMoves(root).map(moveToUci).sort(),
  ['a2a3', 'b1a3', 'b1c3', 'b2b3', 'c2c3', 'd2d3', 'e2e3'].sort(),
  'public legal move generation remains API-compatible'
);

const searcher = new GardnerSearcher({ hashEntries: 16_384 });
const result = searcher.analyze(root, {
  timeMs: 100000,
  maxDepth: 3,
  startDepth: 3,
  multipv: 2,
  mateProbeMs: 0,
  endgameProbeMs: 0,
  fortressProbeMs: 0
});
assert.equal(result.engine, 'Orion JS 17.3');
assert.ok(result.lines.length >= 1, 'search should still return at least one line');

console.log('v17.3 UI and move-buffer optimization compatibility tests passed.');
