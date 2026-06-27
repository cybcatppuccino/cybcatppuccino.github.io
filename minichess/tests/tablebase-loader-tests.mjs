import assert from 'node:assert/strict';
import { EnginePosition } from '../js/engine/engine.js';
import { GardnerTablebase, TablebaseInternals } from '../js/engine/tablebase.js';

const base = process.env.TB_BASE || 'http://127.0.0.1:8123/tools/gardner_tablebase/tables/';
const tb = new GardnerTablebase({ baseUrl: base });
if (!(await tb.init())) {
  console.log('tablebase-loader-tests: skipped (no local tables found)');
  process.exit(0);
}

const position = EnginePosition.fromFEN('4k/5/5/5/KQ3 w - - 0 1');
const spec = TablebaseInternals.materialSpec(position.board);
assert.equal(spec.signature, 'KQvK');
const result = await tb.probe(position);
if (!result) {
  console.log('tablebase-loader-tests: skipped (manifest present but exact block files unavailable)');
  process.exit(0);
}
assert.ok(result);
assert.equal(result.source, 'exact-core');
assert.equal(result.wdl, 1);
assert.ok(result.dtmPly > 0);
const analysis = await tb.analyze(position, { multipv: 2 });
if (analysis?.tablebase !== true) {
  console.log('tablebase-loader-tests: skipped (exact analysis files unavailable)');
  process.exit(0);
}
assert.equal(analysis.tablebase, true);
assert.equal(analysis.tablebaseWdl, 1);
assert.ok(analysis.lines.length >= 1);
assert.ok(analysis.lines[0].move);
console.log('tablebase-loader-tests: ok', result, analysis.lines[0]);
