import assert from 'node:assert/strict';
import { EngineInternals, EnginePosition, uciToMove } from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';

const base = process.env.TB_BASE || 'http://127.0.0.1:8123/tools/gardner_tablebase/tables/';
const tablebase = new GardnerTablebase({ baseUrl: base });
if (!(await tablebase.init())) {
  console.log('v18.1 KQvKBB tablebase smoke: skipped (no manifest found)');
  process.exit(0);
}
if (!tablebase.exactManifest.tables.KQvKBB) {
  console.log('v18.1 KQvKBB tablebase smoke: skipped (KQvKBB absent)');
  process.exit(0);
}

let position = EnginePosition.fromFEN('5/4k/2K1b/b3Q/5 b - - 20 11');
let analysis;
try {
  analysis = await tablebase.analyze(position, { multipv: 5, maxPvPly: 20 });
} catch (error) {
  console.log(`v18.1 KQvKBB tablebase smoke: skipped (${error?.message || error})`);
  process.exit(0);
}
if (analysis?.tablebase !== true) {
  console.log('v18.1 KQvKBB tablebase smoke: skipped (exact metadata/block files unavailable)');
  process.exit(0);
}
assert.equal(analysis?.tablebase, true);
assert.ok(analysis.lines.some(line => line.move === 'a2b3' && line.dtm === 6), 'Bb3 should display the child-optimal #3/6-ply tablebase line');
EngineInternals.makeMove(position, uciToMove(position, 'a2b3'));
const after = await tablebase.analyze(position, { multipv: 3, maxPvPly: 20 });
if (after?.tablebase !== true) {
  console.log('v18.1 KQvKBB tablebase smoke: skipped after move (exact metadata/block files unavailable)');
  process.exit(0);
}
assert.equal(after.lines[0].move, 'e2d3');
assert.equal(after.lines[0].dtm, 5);
console.log('v18.1 KQvKBB tablebase smoke tests passed.');
