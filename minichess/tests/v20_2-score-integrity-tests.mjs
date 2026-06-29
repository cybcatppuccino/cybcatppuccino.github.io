import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';
import { RESULT_KIND, classifyResult, withResultQuality } from '../js/engine/result-quality.js';

{
  const root = EnginePosition.fromFEN('5/k4/p1p2/2P1P/2K2 w - - 0 3');
  const searcher = new GardnerSearcher({ hashEntries: 65536 });
  searcher.setTablebaseProbe(position => position.pieceCount <= 5 ? { wdl: 1, source: 'stub-wdl' } : null);
  const result = searcher.analyze(root, {
    timeMs: 140,
    maxDepth: 4,
    startDepth: 1,
    multipv: 3,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 0
  });
  const wdlLine = result.lines.find(line => line.tablebaseBridgeCandidate);
  assert.ok(wdlLine, 'Expected at least one interior tablebase WDL proof seed.');
  assert.equal(wdlLine.scoreNumeric, false, 'Interior WDL hits must not publish a numeric fallback score.');
  assert.equal(wdlLine.score, 0, 'The internal marker score is neutral and non-publishable.');
  assert.equal(wdlLine.scoreText, '', 'No TB win/loss/draw label may be displayed.');
  assert.equal(wdlLine.tablebaseDisplayFallback, false);
  assert.notEqual(Math.abs(wdlLine.score), 22000, 'The internal sentinel must not be displayed.');
  assert.equal(result.completed, false, 'A WDL-only proof seed is not a publishable completed result.');
}

{
  const quality = classifyResult(withResultQuality({
    engine: 'Orion JS 20.3',
    depth: 6,
    scoreDepth: 6,
    completed: false,
    multiPvVerified: false,
    pvComplete: false,
    lines: [{
      move: 'a1a2',
      score: 0,
      scoreText: '',
      scoreKind: 'tablebase-wdl-proof-seed',
      scoreNumeric: false,
      tablebaseBridgeCandidate: true,
      tablebaseRawScore: 22000,
      pv: ['a1a2'],
      rootScoreExact: false,
      pvComplete: false
    }]
  }));
  assert.equal(quality.kind, RESULT_KIND.TABLEBASE_HINT);
  assert.equal(quality.solved, false);
  assert.equal(quality.exact, false);
}

{
  const engine = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
  const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
  const tablebase = readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
  assert.doesNotMatch(engine, /tablebaseDisplayFallbackScore/);
  assert.doesNotMatch(engine, /tablebaseUnscoredLabel/);
  assert.doesNotMatch(engine, /TB win|TB loss|TB draw/);
  assert.doesNotMatch(worker, /queueTablebaseMixedBoundAudit/);
  assert.doesNotMatch(worker, /tablebaseMixedAudits/);
  assert.match(worker, /queueWideTablebaseBridgeProof/);
  assert.match(worker, /TABLEBASE_WIDE_BRIDGE_MAX_BLOCKS = 128/);
  assert.match(worker, /raw !== 0/);
  assert.match(worker, /rootWdl === 0 && !bridgeLine\.tablebaseTail/);
  assert.match(worker, /tablebaseBridgeAttemptsByKey: new Map\(\)/);
  assert.match(tablebase, /maxCachedBlocks = 128/);
  assert.match(tablebase, /first non-file candidate/);
  assert.match(tablebase, /wdlWarmSignatures/);
}

console.log('v20.3 score integrity tests passed.');
