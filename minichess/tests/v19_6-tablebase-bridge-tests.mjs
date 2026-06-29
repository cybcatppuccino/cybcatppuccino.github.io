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
  generateLegalMoves,
  isInCheck,
  moveToUci,
  uciToMove
} from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';

const TABLES_ROOT = path.resolve(fileURLToPath(new URL('../tools/gardner_tablebase/tables/', import.meta.url)));
const { makeMove } = EngineInternals;

function contentType(file) {
  return file.endsWith('.json') ? 'application/json; charset=utf-8' : 'application/octet-stream';
}

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
      response.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'no-store' });
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
    assert.ok(move, `Expected a legal move for ${text}`);
    makeMove(position, move);
  }
}

const server = await startTablebaseServer();
const address = server.address();
const baseUrl = `http://127.0.0.1:${address.port}/`;

try {
  const tablebase = new GardnerTablebase({ baseUrl });
  assert.equal(await tablebase.init(), true, 'Supplied GTB manifest should load.');

  const rootFen = '5/k4/p1p2/2P1P/2K2 w - - 0 3';
  const root = EnginePosition.fromFEN(rootFen);
  assert.equal(root.pieceCount, 6, 'The example root is outside the <=5-piece GTB domain.');

  // Conservative bridging preloads only the first <=5-piece leaves; it does not
  // claim that Kb1 is forced mate merely because one displayed PV reaches GTB.
  const frontier = await tablebase.warmExactFrontier(root, { maxPly: 4, maxStates: 320, priority: 1 });
  assert.ok(frontier.targets >= 1 && frontier.warmed >= 1, 'Expected at least one exact GTB frontier leaf.');

  const tablebaseEntry = root.clone();
  applyUci(tablebaseEntry, ['c1b1', 'a3a2', 'b1a2']);
  const exact = tablebase.probeExactSync(tablebaseEntry);
  assert.ok(exact, 'The warmed KPPvKP leaf must become synchronously searchable.');
  assert.equal(exact.signature, 'KPPvKP');
  assert.equal(exact.wdl, -1, 'At Kb1 ...a2 Kxa2, the side to move (Black) is tablebase-lost.');
  assert.equal(exact.dtmPly, 24, 'The supplied GTB gives an exact 24-ply mating tail from the entry node.');
  assert.equal(exact.exactDtm, true);

  const bridgeCandidate = {
    move: 'c1b1',
    score: 22000,
    scoreText: '+220.00',
    pv: ['c1b1', 'a3a2', 'b1a2'],
    mateVerified: false,
    rootScoreExact: true
  };
  const enriched = await tablebase.extendResultWithExactTablebaseTails(root, {
    engine: ENGINE_VERSION,
    depth: 16,
    completed: true,
    multiPvVerified: true,
    lines: [bridgeCandidate]
  }, { maxLines: 1, maxProbePly: 36, maxTailPly: 96 });

  const line = enriched.lines[0];
  assert.equal(enriched.tablebaseTailHydrated, true);
  assert.equal(line.score, 22000, 'A GTB tail must not overwrite the root evaluation with a mate score.');
  assert.equal(line.mateVerified, false, 'One PV entering GTB is not an AND/OR proof of root mate.');
  assert.equal(line.tablebaseTail.entersAtPly, 3);
  assert.equal(line.tablebaseTail.dtmPly, 24);
  assert.equal(line.tablebaseTail.terminal, true);
  assert.equal(line.pv.length, 27, 'PV should contain Kb1, the bridge moves, and the 24-ply GTB tail.');

  const terminal = root.clone();
  applyUci(terminal, line.pv);
  assert.equal(generateLegalMoves(terminal).length, 0, 'Hydrated GTB tail should terminate the line.');
  assert.equal(isInCheck(terminal), true, 'The exact GTB tail must terminate in checkmate, not stalemate.');

  const worker = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
  const playWorker = readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
  const tablebaseSource = readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');

  assert.equal(ENGINE_VERSION, 'Orion JS 19.8');
  assert.match(tablebaseSource, /probeExactSync\(position\)/);
  assert.match(tablebaseSource, /warmExactFrontier\(position/);
  assert.match(tablebaseSource, /extendResultWithExactTablebaseTails\(position/);
  assert.match(worker, /queueTablebaseFrontierWarmup/);
  assert.match(worker, /queueTablebaseTailHydration/);
  assert.match(worker, /tablebase\.probeSync\(position\)/);
  assert.match(playWorker, /warmExactFrontier\(position\.clone\(\)/);
  assert.match(app, /const ANALYSIS_PAINT_INTERVAL_MS = 500;/);
  assert.match(html, /Gardner MiniChess Lab v19\.8/);

  console.log('v19.8 exact-tablebase bridge, non-mate-root and versioning tests passed.', {
    frontier,
    entry: exact,
    pv: line.pv.map(move => typeof move === 'string' ? move : moveToUci(move))
  });
} finally {
  await new Promise(resolve => server.close(resolve));
}
