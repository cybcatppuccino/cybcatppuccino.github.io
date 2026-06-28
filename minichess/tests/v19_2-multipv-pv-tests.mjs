import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  EngineInternals,
  EnginePosition,
  ENGINE_VERSION,
  GardnerSearcher,
  generateLegalMoves
} from '../js/engine/engine.js';

const INITIAL = 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1';
const root = EnginePosition.fromFEN(INITIAL);
const searcher = new GardnerSearcher({ hashEntries: 16_384 });

// Build one legal non-cyclic line and place a TT move at every child. The
// extension function must reconstruct the same repetition-aware TT context
// even after a different branch has deliberately poisoned the live salts.
const line = [];
const seen = new Set([`${root.hashA}:${root.hashB}:${root.turn}`]);
let cursor = root.clone();
for (let ply = 0; ply < 7; ply += 1) {
  const legal = generateLegalMoves(cursor);
  assert.ok(legal.length, `Expected a legal move at ply ${ply}`);
  let selected = 0;
  for (let offset = 0; offset < legal.length; offset += 1) {
    const candidate = legal[(ply * 3 + offset) % legal.length];
    const probe = cursor.clone();
    EngineInternals.makeMove(probe, candidate);
    const identity = `${probe.hashA}:${probe.hashB}:${probe.turn}`;
    if (!seen.has(identity)) {
      selected = candidate;
      seen.add(identity);
      break;
    }
  }
  if (!selected) selected = legal[0];
  line.push(selected);
  EngineInternals.makeMove(cursor, selected);
}

cursor = root.clone();
searcher.recordSearchPath(0, cursor);
for (let ply = 0; ply < line.length; ply += 1) {
  EngineInternals.makeMove(cursor, line[ply]);
  searcher.recordSearchPath(ply + 1, cursor);
  if (ply + 1 < line.length) searcher.storeTT(cursor, 8 - ply, 0, 0, line[ply + 1], 0, ply + 1);
}

// Poison the live path arrays with a different root continuation. v19.1's PV
// tail probe read these stale salts and therefore missed the correct TT entry.
const poison = root.clone();
const poisonMove = generateLegalMoves(poison).find(move => move !== line[0]) || line[0];
EngineInternals.makeMove(poison, poisonMove);
searcher.recordSearchPath(1, poison);
searcher.ttPathSaltA[2] ^= 0x73a2c41d;
searcher.ttPathSaltB[2] ^= 0x1f123bb5;

const extended = searcher.extendPvWithTt(root, { move: line[0], pv: [line[0]] }, line.length);
assert.equal(extended.pv.length, line.length, 'TT continuation should rebuild every currently known PV ply');
assert.deepEqual(extended.pv, line, 'TT continuation must follow the correct repetition-aware branch');

// Live PVS updates may carry only a root/short prefix. Preserve an older tail
// only for a matching prefix; divergent variation branches must not be spliced.
const longLine = { move: line[0], score: 12, pv: line.slice(0, 6) };
const shortPrefix = { move: line[0], score: 15, pv: line.slice(0, 2), liveUpdate: true };
const mergedPrefix = searcher.mergeRootLinePv(longLine, shortPrefix);
assert.equal(mergedPrefix.score, 15);
assert.deepEqual(mergedPrefix.pv, longLine.pv, 'A short matching-prefix update must not erase a longer PV');
const divergentSecond = generateLegalMoves(root).find(move => move !== line[0]);
const divergent = searcher.mergeRootLinePv(longLine, { move: line[0], score: 20, pv: [line[0], divergentSecond] });
assert.deepEqual(divergent.pv, [line[0], divergentSecond], 'Divergent PVs must remain distinct rather than being stitched together');

// Result completion cannot be based on the first candidate alone. A short
// second/third candidate keeps a high-depth MultiPV result live for repair.
const synthetic = [
  { move: line[0], score: 0, pv: line.slice(0, 6) },
  { move: line[0] + 1, score: -1, pv: [line[0] + 1] },
  { move: line[0] + 2, score: -2, pv: line.slice(0, 6) }
];
assert.equal(searcher.hasThinPrincipalVariation(root, synthetic, 8), true);

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const cache = readFileSync(new URL('../js/engine/analysis-cache.js', import.meta.url), 'utf8');
assert.equal(ENGINE_VERSION, 'Orion JS 19.7');
assert.match(html, /Gardner MiniChess Lab v19\.7/);
assert.match(html, /Orion JS 19\.7/);
assert.match(html, /app\.js\?v=19\.7/);
assert.match(app, /gardner-current-game-v19\.7/);
assert.match(app, /gardner-current-game-v19\.2/);
assert.match(app, /gardner-current-game-v19\.1/);
assert.match(cache, /gardner-analysis-cache-v19\.7/);
assert.match(cache, /gardner-analysis-cache-v19\.2/);
assert.match(cache, /gardner-analysis-cache-v19\.1/);

console.log('v19.7 MultiPV PV reconstruction and versioning tests passed.');
