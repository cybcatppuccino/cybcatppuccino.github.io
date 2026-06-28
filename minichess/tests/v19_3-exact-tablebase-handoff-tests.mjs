import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { ENGINE_VERSION } from '../js/engine/engine.js';
import { compareAnalysisResults, RESULT_KIND, withResultQuality } from '../js/engine/result-quality.js';

const exactTb = withResultQuality({
  engine: ENGINE_VERSION,
  tablebase: true,
  tablebaseSource: 'exact-core',
  tablebaseWdl: 1,
  completed: true,
  terminal: true,
  lines: [{ move: 'a1a2', score: 29990, scoreText: '#5', tablebase: true, tablebaseExactDtm: true, tablebaseWdl: 1, dtm: 10, pv: ['a1a2'] }]
});
const localMate = withResultQuality({
  engine: ENGINE_VERSION,
  completed: true,
  lines: [{ move: 'a1a2', score: 29994, scoreText: '#3', mateVerified: true, dtm: 6, pv: ['a1a2'] }]
});
const pseudoTb = withResultQuality({
  engine: ENGINE_VERSION,
  depth: 11,
  completed: false,
  lines: [{ move: 'a1a2', score: 22000, scoreText: '+220.00', pv: ['a1a2'] }]
});

assert.equal(exactTb.resultKind, RESULT_KIND.EXACT_TABLEBASE);
assert.equal(compareAnalysisResults(localMate, exactTb), exactTb, 'Exact GTB must outrank a local mate proof for display/caching.');
assert.equal(compareAnalysisResults(exactTb, pseudoTb), exactTb, 'A +/-220 synchronous WDL sentinel must never replace Exact TB.');

const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
assert.equal(ENGINE_VERSION, 'Orion JS 19.3');
assert.match(worker, /function queueExactTablebasePromotion\(/, 'Worker must retry a root exact TB probe after WDL warming.');
assert.match(worker, /tablebasePromotionPending/, 'Worker must track a pending root TB promotion.');
assert.match(worker, /shouldDeferTablebasePseudoDisplay/, 'Worker must defer +/-220 display while exact GTB is being promoted.');
assert.match(worker, /flushDeferredTablebaseDisplay/, 'Worker must release the normal fallback only after bounded retry failure.');
assert.match(app, /function preserveDatabaseDisplay\(/, 'UI must preserve database mate PV/DTM against streamed live snapshots.');
assert.match(app, /TABLEBASE_PSEUDO_SCORE = 22000/, 'UI must recognize the internal WDL sentinel.');
assert.match(html, /Gardner MiniChess Lab v19\.3/);

console.log('v19.3 exact-tablebase handoff and display-priority tests passed.');
