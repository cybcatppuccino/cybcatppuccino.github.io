import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { readFile, stat } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
  ENGINE_VERSION,
  EngineInternals,
  EnginePosition,
  GardnerSearcher,
  uciToMove
} from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';
import {
  RESULT_KIND,
  classifyResult,
  compareAnalysisResults,
  isSolvedResult,
  shouldCacheResult,
  withResultQuality
} from '../js/engine/result-quality.js';

const TABLES_ROOT = path.resolve(fileURLToPath(new URL('../tools/gardner_tablebase/tables/', import.meta.url)));
const { TB_WIN_SCORE } = EngineInternals;
const FINITE_WDL_CAP = 1_000;

function startTablebaseServer() {
  const server = http.createServer(async (request, response) => {
    try {
      const requested = decodeURIComponent(new URL(request.url || '/', 'http://127.0.0.1').pathname)
        .replace(/^\/+/, '');
      if (!requested || requested.includes('..') || path.isAbsolute(requested)) {
        response.writeHead(404).end();
        return;
      }
      const file = path.resolve(TABLES_ROOT, requested);
      if (!file.startsWith(`${TABLES_ROOT}${path.sep}`)) {
        response.writeHead(403).end();
        return;
      }
      const info = await stat(file);
      if (!info.isFile()) {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, {
        'Content-Type': file.endsWith('.json') ? 'application/json; charset=utf-8' : 'application/octet-stream',
        'Cache-Control': 'no-store'
      });
      response.end(await readFile(file));
    } catch {
      response.writeHead(404).end();
    }
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

function cappedAudit(tablebase, root, rootUci, { maxDepth = 6 } = {}) {
  const rootMove = uciToMove(root, rootUci);
  assert.ok(rootMove, `Expected legal root move ${rootUci}`);
  const searcher = new GardnerSearcher({
    hashEntries: 65_536,
    tablebaseProbe: position => tablebase.probeExactSync(position)
  });
  return searcher.analyze(root.clone(), {
    timeMs: 1_200,
    maxDepth,
    multipv: 1,
    newPosition: true,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 0,
    tablebaseScoreCap: FINITE_WDL_CAP,
    restrictedRootMoves: [rootMove]
  });
}

const server = await startTablebaseServer();
const address = server.address();

try {
  const tablebase = new GardnerTablebase({
    baseUrl: `http://127.0.0.1:${address.port}/`,
    maxCachedBlocks: 128,
    maxCachedWdlBlocks: 128,
    maxConcurrentRequests: 6
  });
  assert.equal(await tablebase.init(), true);

  // This is the v19.7 bridge example. Normal alpha-beta may use its exact WDL
  // leaves to order/cut the tree, but the presentation must not leak +220.00
  // or the former "TB bridge · verifying" placeholder before an AND/OR mate
  // certificate is built.
  const whiteRoot = EnginePosition.fromFEN('5/k4/p1p2/2P1P/2K2 w - - 0 3');
  await tablebase.warmExactBridgeTables(whiteRoot, {
    maxPly: 4,
    maxStates: 320,
    maxBlocks: 36,
    priority: 0
  });
  const normal = new GardnerSearcher({
    hashEntries: 65_536,
    tablebaseProbe: position => tablebase.probeExactSync(position)
  }).analyze(whiteRoot.clone(), {
    timeMs: 1_500,
    maxDepth: 6,
    multipv: 1,
    newPosition: true,
    endgameProbeMs: 0,
    fortressProbeMs: 0,
    mateProbeMs: 0
  });
  assert.equal(normal.lines[0].move, 'c1b1');
  assert.equal(normal.lines[0].tablebaseBridgeCandidate, true);
  assert.notEqual(normal.lines[0].scoreText, 'TB bridge · verifying');
  assert.ok(Math.abs(normal.lines[0].score) < TB_WIN_SCORE, 'Internal WDL sentinel must never be published as a normal score.');

  const whiteAudit = cappedAudit(tablebase, whiteRoot, 'c1b1');
  assert.equal(whiteAudit.completed, true);
  assert.equal(whiteAudit.tablebaseBoundedSearch, true);
  assert.ok(whiteAudit.tablebaseBoundedProbeHits > 0);
  assert.equal(whiteAudit.tablebaseScoreCap, FINITE_WDL_CAP);
  assert.equal(whiteAudit.lines[0].move, 'c1b1');
  assert.equal(whiteAudit.lines[0].score, FINITE_WDL_CAP);
  assert.equal(whiteAudit.lines[0].scoreText, '+10.00');
  assert.equal(whiteAudit.lines[0].tablebaseBridgeCandidate, false, 'The capped search result is numeric, not a raw tablebase sentinel.');
  assert.equal(whiteAudit.lines[0].mateVerified, false, 'A finite WDL audit must not fabricate mate.');

  const mixed = withResultQuality({
    ...normal,
    lines: [{
      ...whiteAudit.lines[0],
      tablebaseMixedAudit: true,
      tablebaseMixedAuditSaturated: true,
      tablebaseMixedAuditCap: FINITE_WDL_CAP,
      tablebaseScope: 'mixed-wdl-audit'
    }],
    tablebaseMixedAudit: true,
    tablebaseMixedAuditCap: FINITE_WDL_CAP,
    tablebaseMixedAuditDepth: whiteAudit.depth,
    completed: true,
    multiPvVerified: true,
    pvComplete: true,
    cached: false,
    solved: false
  });
  assert.equal(classifyResult(mixed).kind, RESULT_KIND.TABLEBASE_MIXED_BOUND);
  assert.equal(isSolvedResult(mixed), false, 'Finite WDL evidence is not a mate/draw proof.');
  assert.equal(shouldCacheResult(mixed), false, 'Mixed audits are display-stable only and never cross-root cacheable.');

  const ordinaryDeeper = withResultQuality({
    engine: ENGINE_VERSION,
    depth: 20,
    scoreDepth: 20,
    completed: true,
    multiPvVerified: true,
    pvComplete: true,
    lines: [{
      move: 'c1b1', score: 750, scoreText: '+7.50',
      pv: ['c1b1', 'a4b4', 'b1a2'], rootScoreExact: true
    }]
  });
  assert.equal(compareAnalysisResults(mixed, ordinaryDeeper), mixed,
    'A completed WDL-aware worst-defence audit remains visible until a proof or a newer audit replaces it.');

  // Symmetry check: a Black-controlled bridge is displayed from White's
  // perspective as a finite negative score, never as -220.00.
  const blackRoot = EnginePosition.fromFEN('3k1/K2PP/4P/1p3/5 b - - 0 1');
  await tablebase.warmExactBridgeTables(blackRoot, {
    maxPly: 4,
    maxStates: 320,
    maxBlocks: 36,
    priority: 0
  });
  const blackAudit = cappedAudit(tablebase, blackRoot, 'd5e4', { maxDepth: 5 });
  assert.equal(blackAudit.completed, true);
  assert.equal(blackAudit.lines[0].score, -FINITE_WDL_CAP);
  assert.equal(blackAudit.lines[0].scoreText, '-10.00');
  assert.equal(blackAudit.lines[0].mateVerified, false);

  const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
  const engine = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.equal(ENGINE_VERSION, 'Orion JS 19.8');
  assert.match(worker, /queueTablebaseMixedBoundAudit/);
  assert.match(worker, /tablebaseScoreCap: TABLEBASE_MIXED_AUDIT_SCORE_CAP/);
  assert.doesNotMatch(worker, /TB bridge · verifying/);
  assert.match(engine, /tablebaseDisplayFallbackScore/);
  assert.match(engine, /tablebaseBoundedSearch/);
  assert.match(app, /const ANALYSIS_PAINT_INTERVAL_MS = 500;/, 'The UI remains on the existing 500 ms paint throttle.');
  assert.match(html, /Gardner MiniChess Lab v19\.8/);

  console.log('v19.8 mixed WDL-bound audit tests passed.', {
    whiteAudit: whiteAudit.lines[0].scoreText,
    whiteProbeHits: whiteAudit.tablebaseBoundedProbeHits,
    blackAudit: blackAudit.lines[0].scoreText,
    blackProbeHits: blackAudit.tablebaseBoundedProbeHits
  });
} finally {
  await new Promise(resolve => server.close(resolve));
}
