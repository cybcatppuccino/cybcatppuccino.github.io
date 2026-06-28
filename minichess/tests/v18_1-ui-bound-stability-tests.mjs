import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { compareAnalysisResults } from '../js/engine/result-quality.js';

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');

assert.match(app, /const ANALYSIS_PAINT_INTERVAL_MS = 500;/, 'v18.2 should cap streamed analysis paints at 500 ms');
assert.doesNotMatch(app, /const important = Boolean\(/, 'important analysis results should not bypass the 500 ms paint throttle');
assert.match(app, /setTimeout\(flushAnalysisPaint, ANALYSIS_PAINT_INTERVAL_MS\)/, 'pending analysis should render only on the next fixed 500 ms paint tick');
assert.doesNotMatch(app, /elapsed >= ANALYSIS_PAINT_INTERVAL_MS/, 'streamed analysis should not flush immediately after an idle gap');
assert.match(app, /function chooseVisibleAnalysisResult\(/, 'UI should apply shared result-quality selection before painting streamed analysis');

assert.match(worker, /function stableLineRank\(/, 'worker should rank individual root lines before merging streamed results');
assert.match(worker, /previousRank > nextRank/, 'worker should preserve stronger mate\/tablebase-bound line data over live estimates');
assert.match(worker, /liveScoreSuppressed: true/, 'worker should mark suppressed live centipawn replacements for debugging');
assert.match(worker, /const previousStable = current\.lastResult;/, 'worker should retain the last durable result before comparing a new cumulative search slice');
assert.match(worker, /const chosen = compareAnalysisResults\(previousStable, cumulative\)/, 'worker should compare cumulative live results against the last stable result before posting');
assert.match(worker, /const visibleResult = publishLiveProgress/, 'v18.4 should publish fresh node/PV progress without weakening the durable cache choice');
assert.match(worker, /function needsDtmAnnotation\(/, 'worker should skip redundant DTM annotation when visible lines are already bound or exact');

const bound = {
  engine: 'Orion JS 18.2',
  depth: 6,
  completed: true,
  tablebaseDtmBound: true,
  lines: [{ move: 'd3d4', score: 29981, scoreText: '≤#10 · TB bound', pv: ['d3d4'], tablebase: true, tablebaseBound: true, dtmUpperBound: true }]
};
const live = {
  engine: 'Orion JS 18.2',
  depth: 14,
  completed: true,
  lines: [{ move: 'd3d4', score: 220, scoreText: '+2.20', pv: ['d3d4', 'b3b2'] }]
};
assert.equal(compareAnalysisResults(bound, live), bound, 'tablebase-bound mate display should outrank a later live centipawn estimate');

console.log('v18.4 UI throttle, live-progress and bound-stability tests passed.');
