import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { Position } from '../js/core/position.js';
import { gameStatus } from '../js/core/rules.js';
import { buildAnalysisKey } from '../js/engine/analysis-cache.js';
import {
  RESULT_KIND,
  isSolvedResult,
  isTrustedExactTablebaseResult,
  withResultQuality
} from '../js/engine/result-quality.js';

// v18.3 uses the same no-50-move convention as the fixed GTB corpus.
{
  const position = Position.initial();
  position.halfmove = 100;
  assert.notEqual(gameStatus(position, 1).state, 'draw-50');
}

// Same board + same latest ten plies is insufficient for threefold context.
{
  const position = Position.initial();
  position.halfmove = 12;
  const tail = Array.from({ length: 10 }, (_, index) => `tail${index} w - - 0 1`);
  const left = ['older-left w - - 0 1', 'older-left-2 b - - 0 1', ...tail];
  const right = ['older-right w - - 0 1', 'older-right-2 b - - 0 1', ...tail];
  assert.notEqual(buildAnalysisKey(position, left), buildAnalysisKey(position, right));
}

// A direct GTB WDL result is solved even where DTM is only a display bound.
{
  const boundedGtb = withResultQuality({
    engine: 'Orion JS 18.3',
    tablebase: true,
    tablebaseDtmBound: true,
    tablebaseWdl: 0,
    completed: true,
    lines: [{
      move: 'a2a3', score: 0, pv: ['a2a3'], tablebase: true,
      tablebaseWdl: 0, tablebaseBound: true, dtmUpperBound: true
    }]
  });
  assert.equal(isTrustedExactTablebaseResult(boundedGtb), true);
  assert.equal(isSolvedResult(boundedGtb), true);
  assert.equal(boundedGtb.resultKind, RESULT_KIND.EXACT_TABLEBASE);
}

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const engine = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const playWorker = readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
const tablebase = readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
const cache = readFileSync(new URL('../js/engine/analysis-cache.js', import.meta.url), 'utf8');

assert.match(app, /Intentional product behavior: a browser refresh starts a clean AI session/);
assert.match(app, /function ensureAnalysisClient\(/);
assert.match(app, /function ensurePlayClient\(/);
assert.match(app, /function releaseAnalysisClient\(/);
assert.match(app, /function releasePlayClient\(/);
assert.doesNotMatch(app, /const analysisClient = new AnalysisClient/);
assert.doesNotMatch(app, /const playClient = new PlayEngineClient/);
assert.match(engine, /repetitionContextFingerprint/);
assert.match(engine, /ttHistorySaltA/);
assert.doesNotMatch(engine, /halfmove >= 100/);
assert.match(playWorker, /from '\.\/result-quality\.js'/);
assert.match(playWorker, /compareAnalysisResults/);
assert.match(tablebase, /maxConcurrentRequests/);
assert.match(tablebase, /enqueueRequest/);
assert.doesNotMatch(tablebase, /practicalManifest/);
assert.match(cache, /requestIdleCallback/);
assert.match(cache, /this\.dirty/);

console.log('v18.3 rules, repetition, tablebase and worker lifecycle tests passed.');
