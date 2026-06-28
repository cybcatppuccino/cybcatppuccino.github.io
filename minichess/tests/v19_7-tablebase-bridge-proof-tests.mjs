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
  generateLegalMoves,
  isInCheck,
  uciToMove
} from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';
import { RESULT_KIND, classifyResult, shouldCacheResult, withResultQuality } from '../js/engine/result-quality.js';

const TABLES_ROOT = path.resolve(fileURLToPath(new URL('../tools/gardner_tablebase/tables/', import.meta.url)));
const { makeMove, WHITE } = EngineInternals;

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

function applyUci(position, moves) {
  for (const text of moves) {
    const move = uciToMove(position, text);
    assert.ok(move, `Expected legal bridge PV move ${text}`);
    makeMove(position, move);
  }
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

  // The six-piece example is intentionally outside the exact database domain.
  // A finite mate upper bound is admissible only after every Black reply after
  // the chosen Kb1 policy has a tablebase/terminal certificate.
  const root = EnginePosition.fromFEN('5/k4/p1p2/2P1P/2K2 w - - 0 3');
  const warm = await tablebase.warmExactBridgeTables(root, {
    maxPly: 4,
    maxStates: 320,
    maxBlocks: 36,
    priority: 0
  });
  assert.ok(warm.warmed, 'Bridge proof requires resident exact GTB blocks.');
  assert.ok(warm.signatures.includes('KPPvKP'));

  // Integration check: at modest depth normal alpha-beta already selects Kb1,
  // and the existing tail hydrator sees a legal exact-tablebase entry. The
  // subsequent proof still treats that line only as a candidate policy hint.
  const discoverySearcher = new GardnerSearcher({
    hashEntries: 65_536,
    tablebaseProbe: position => tablebase.probeExactSync(position)
  });
  const depthSix = discoverySearcher.analyze(root.clone(), {
    timeMs: 1_500,
    maxDepth: 6,
    multipv: 1,
    newPosition: true
  });
  assert.equal(depthSix.lines[0].move, 'c1b1');
  const hydrated = await tablebase.extendResultWithExactTablebaseTails(root, depthSix, {
    maxLines: 1,
    maxProbePly: 36,
    maxTailPly: 96
  });
  assert.equal(hydrated.tablebaseTailHydrated, true);
  assert.equal(hydrated.lines[0].tablebaseTail.terminal, true);

  const proof = await tablebase.proveExactBridgeOutcome(root, {
    controller: WHITE,
    outcome: 'win',
    // This is an ordinary completed-analysis policy hint. It is not trusted as
    // proof: every opponent reply remains explicitly enumerated below.
    preferredMoves: hydrated.lines[0].pv,
    maxPlies: 41,
    maxNodes: 48_000,
    timeMs: 1_400,
    controllerMoveLimit: 4,
    priority: 0
  });
  assert.ok(proof?.tablebaseBridgeProof, 'Expected an AND/OR bridge certificate.');
  assert.equal(proof.exactDtm, false, '41 is an upper bound, not shortest DTM.');
  assert.equal(proof.upperBound, true);
  assert.equal(proof.dtmPly, 41);
  assert.equal(proof.pv.length, 41);
  assert.equal(proof.pv[0], 'c1b1');
  assert.equal(proof.proof.kind, 'choice');
  assert.equal(proof.proof.child.kind, 'all', 'After Kb1 the resisting side is exhaustive.');

  const afterKb1 = root.clone();
  applyUci(afterKb1, ['c1b1']);
  const blackReplies = generateLegalMoves(afterKb1, false);
  assert.equal(proof.proof.child.children.length, blackReplies.length);
  assert.equal(blackReplies.length, 4, 'Kb1 should have exactly four legal Black replies in the example.');

  const terminal = root.clone();
  applyUci(terminal, proof.pv);
  assert.equal(generateLegalMoves(terminal, false).length, 0);
  assert.equal(isInCheck(terminal), true, 'The selected worst branch must end in actual checkmate.');

  // A 6-piece forced capture enters a KPPvKP WDL draw. Both colors have the
  // same forced move, so this is a compact regression for the dual-controller
  // draw rule: no one-sided "I can draw" route may be promoted to 0.00.
  const drawRoot = EnginePosition.fromFEN('k4/pPP2/5/P4/K4 b - - 0 1');
  const drawWarm = await tablebase.warmExactBridgeTables(drawRoot, {
    maxPly: 4,
    maxStates: 320,
    maxBlocks: 36,
    priority: 0
  });
  assert.ok(drawWarm.warmed);
  const drawHydrated = await tablebase.extendLineWithExactTablebaseTail(drawRoot, {
    move: 'a5b4',
    score: 0,
    scoreText: '0.00',
    pv: ['a5b4']
  }, {
    maxProbePly: 8,
    maxTailPly: 32
  });
  assert.ok(drawHydrated?.tablebaseTail?.bridgeable, 'Exact WDL draw entry should trigger bridge proof consideration.');
  assert.equal(drawHydrated.tablebaseTail.wdl, 0);
  assert.equal(drawHydrated.tablebaseTail.draw, true);
  assert.equal(drawHydrated.tablebaseTail.terminal, false, 'A draw entry is not a fake finite mate tail.');

  const draw = await tablebase.proveExactBridgeDraw(drawRoot, {
    maxPlies: 8,
    maxNodes: 10_000,
    timeMs: 1_000,
    controllerMoveLimit: 4,
    priority: 0
  });
  assert.ok(draw?.tablebaseBridgeDraw, 'Expected a dual-controller exact-tablebase draw certificate.');
  assert.equal(draw.wdl, 0);
  assert.deepEqual(draw.pv, ['a5b4']);

  // Direction check: a six-piece Black-to-move position can choose Kxe4,
  // immediately enter an exact KPPvKP loss for White and obtain a signed
  // negative-from-White bridge bound. This proves the mechanism is not tied to
  // White wins or to the illustrative Kb1 pawn ending.
  const blackRoot = EnginePosition.fromFEN('3k1/K2PP/4P/1p3/5 b - - 0 1');
  const blackWarm = await tablebase.warmExactBridgeTables(blackRoot, {
    maxPly: 4,
    maxStates: 320,
    maxBlocks: 36,
    priority: 0
  });
  assert.ok(blackWarm.warmed);
  const blackProof = await tablebase.proveExactBridgeOutcome(blackRoot, {
    controller: -WHITE,
    outcome: 'win',
    preferredMoves: ['d5e4'],
    maxPlies: 17,
    maxNodes: 10_000,
    timeMs: 1_000,
    controllerMoveLimit: 4,
    priority: 0
  });
  assert.ok(blackProof?.tablebaseBridgeProof, 'Expected a Black bridge win certificate.');
  assert.equal(blackProof.dtmPly, 17);
  assert.equal(blackProof.pv[0], 'd5e4');
  assert.equal(blackProof.wdl, 1, 'WDL is positive from the Black-to-move root perspective.');

  const bridgeResult = withResultQuality({
    engine: ENGINE_VERSION,
    completed: true,
    multiPvVerified: true,
    lines: [{
      move: proof.pv[0],
      score: 30_000 - proof.dtmPly,
      scoreText: '≤#21',
      pv: proof.pv,
      mateVerified: true,
      mateUpperBound: true,
      tablebaseBridgeProof: true,
      rootScoreExact: true,
      pvComplete: true
    }],
    tablebaseBridgeProof: true,
    tablebaseBridgeDtm: proof.dtmPly
  });
  assert.equal(classifyResult(bridgeResult).kind, RESULT_KIND.TABLEBASE_BRIDGE_MATE);
  assert.equal(shouldCacheResult(bridgeResult), false, 'An upper bound is display-stable but never cross-root cacheable.');

  const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
  const tablebaseSource = readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
  assert.equal(ENGINE_VERSION, 'Orion JS 19.7');
  assert.match(worker, /queueTablebaseBridgeProof/);
  assert.match(worker, /tail\?\.bridgeable/);
  assert.match(tablebaseSource, /proveExactBridgeOutcome\(position/);
  assert.match(tablebaseSource, /proveExactBridgeDraw\(position/);
  assert.match(tablebaseSource, /Exact-tablebase territory is a proof boundary/);
  assert.match(app, /const ANALYSIS_PAINT_INTERVAL_MS = 500;/);
  assert.match(html, /Gardner MiniChess Lab v19\.7/);

  console.log('v19.7 AND/OR exact-tablebase bridge proof tests passed.', {
    winUpperBoundPly: proof.dtmPly,
    winProofNodes: proof.proofNodes,
    winProofLeaves: proof.proofLeaves,
    drawProofNodes: draw.proofNodes,
    drawProofLeaves: draw.proofLeaves
  });
} finally {
  await new Promise(resolve => server.close(resolve));
}
