import assert from 'node:assert/strict';

import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';

const base = process.env.TB_BASE || 'http://127.0.0.1:8123/tools/gardner_tablebase/tables/';
const tablebase = new GardnerTablebase({ baseUrl: base });
if (!(await tablebase.init())) {
  console.log('v12.1 tablebase DTM-bound tests: skipped (no local tables found)');
  process.exit(0);
}

{
  const position = EnginePosition.fromFEN('4k/5/5/5/KQ3 w - - 0 1');
  const result = await tablebase.analyze(position, { multipv: 1, maxPvPly: 1 });
  assert.equal(result?.tablebase, true);
  assert.ok(result.lines[0].scoreText.includes('#'), 'exact DTM should be displayed as a mate distance');
  assert.ok(!/TB win|TB loss/.test(result.lines[0].scoreText), 'exact DTM must not fall back to generic TB win/loss text');
}

{
  await tablebase.warmExactWdl({ pieceLimit: 4 });
  const fen = '8/8/1k6/2p5/8/3PP3/3K4/8 b - - 0 1';
  const position = EnginePosition.fromFEN(fen);
  const searcher = new GardnerSearcher({ hashEntries: 131072 });
  searcher.setTablebaseProbe(pos => tablebase.probeWdlSync(pos));
  const raw = searcher.analyze(position.clone(), {
    timeMs: 650,
    maxDepth: 8,
    multipv: 3,
    mateProbeMs: 80,
    mateMaxPlies: 45
  });
  assert.ok(raw.lines.some(line => Math.abs(Number(line.score || 0)) === 22000), 'raw WDL search should expose the old TB score');
  const annotated = await tablebase.annotateResultWithDtmBounds(position.clone(), raw, { maxLines: 3, maxProbePly: 24 });
  assert.ok(annotated.tablebaseDtmBound, 'normal search result should be annotated with a TB DTM bound');
  assert.ok(annotated.lines.some(line => line.tablebaseBound && line.scoreText.includes('#')), 'annotated line should display a mate-distance bound');
}

console.log('v12.1 tablebase DTM-bound tests passed.');
