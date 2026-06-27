import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  EnginePosition,
  analyzeOnce,
  analyzePositionActivity,
  EngineInternals
} from '../js/engine/engine.js';
import { validateExternalAnalysisResult, FAIRY_STOCKFISH_LABEL } from '../js/engine/external-engine.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const qb1Deadlock = 'rq2k/p1p1p/PpPpP/1B1P1/RQ2K b - - 6 8';

{
  const profile = analyzePositionActivity(EnginePosition.fromFEN(qb1Deadlock));
  assert.ok(profile.legal, 'The legal low-progress classifier should run even with unsound pseudo-contact.');
  assert.equal(profile.lowProgressDraw, true);
  assert.equal(profile.scale, 0);
  assert.equal(profile.quietProgressThreats, 0);
  const result = analyzeOnce(qb1Deadlock, {
    timeMs: 4_000,
    maxDepth: 10,
    multipv: 3,
    fortressProbeMs: 120,
    endgameProbeMs: 0,
    mateProbeMs: 0
  });
  assert.ok(Math.abs(result.lines[0].score) <= 10, `Expected deadlocked Qb1 line near 0.00, got ${result.lines[0].score}`);
}

{
  const tactical = EnginePosition.fromFEN('r2qk/p1p1p/NpPpP/1P1P1/2BQK b - - 0 5');
  const profile = analyzePositionActivity(tactical);
  assert.equal(profile.lowProgressDraw, false);
  assert.ok(profile.quietProgressThreats > 0, 'Concrete heavy-offer breakthroughs should block hard draw compression.');
}

{
  const workerSource = fs.readFileSync(path.join(root, 'vendor/fairy-stockfish/fairy-uci-worker.js'), 'utf8');
  assert.match(workerSource, /SharedArrayBuffer/);
  assert.match(workerSource, /readyok/);
  assert.match(workerSource, /recoverable/);
  const serveSource = fs.readFileSync(path.join(root, 'tools/serve-coi.py'), 'utf8');
  assert.match(serveSource, /Cross-Origin-Embedder-Policy/);
  assert.match(serveSource, /application\/wasm/);
  const coiSource = fs.readFileSync(path.join(root, 'coi-serviceworker.js'), 'utf8');
  assert.match(coiSource, /legacy file/);
  assert.match(coiSource, /unregister/);
  assert.doesNotMatch(coiSource, /Cross-Origin-Embedder-Policy/);
}

{
  const rootPosition = EnginePosition.fromFEN(qb1Deadlock);
  const legal = validateExternalAnalysisResult(rootPosition, {
    engine: FAIRY_STOCKFISH_LABEL,
    depth: 6,
    lines: [{ move: 'b5c5', score: 150, pv: ['b5c5', 'b1d1'] }]
  });
  assert.ok(legal, 'Legal external PVs should be accepted before display/use.');
  const illegal = validateExternalAnalysisResult(rootPosition, {
    engine: FAIRY_STOCKFISH_LABEL,
    depth: 6,
    lines: [{ move: 'b5b1', score: 0, pv: ['b5b1'] }]
  });
  assert.equal(illegal, null, 'Illegal external PVs must be rejected for Orion fallback.');
}

assert.ok(EngineInternals.quietHeavyOfferBreakthroughThreat, 'v14.1 exposes the concrete heavy-offer verifier for regression tests.');
console.log('v15.1 Stockfish startup compatibility and queenful deadlock tests passed.');
