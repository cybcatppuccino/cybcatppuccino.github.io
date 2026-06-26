import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Worker } from 'node:worker_threads';
import {
  EngineInternals,
  EnginePosition,
  GardnerSearcher,
  validateMateResult,
  uciToMove
} from '../js/engine/engine.js';
import {
  AnalysisCache,
  buildAnalysisKey,
  rebaseVerifiedMateLine
} from '../js/engine/analysis-cache.js';
import { Position } from '../js/core/position.js';

class MemoryStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

const fen = '4k/5/3Q1/2K2/5 w - - 0 1';
const root = EnginePosition.fromFEN(fen);
const solved = new GardnerSearcher({ hashEntries: 32_768 }).analyze(root, {
  timeMs: 100,
  maxDepth: 4,
  multipv: 1,
  endgameProbeMs: 320
});
assert.equal(solved.lines[0].mateVerified, true, 'Fixture must produce a verified mate');
solved.solved = true;

// A solved cache entry survives localStorage reload and cannot be replaced by a
// later non-solved update.
{
  const storage = new MemoryStorage();
  const cache = new AnalysisCache(storage);
  const key = buildAnalysisKey(Position.fromFEN(fen), []);
  cache.set(key, solved);
  cache.set(key, {
    ...solved,
    solved: false,
    endgameProof: false,
    depth: solved.depth + 5,
    lines: [{ ...solved.lines[0], mateVerified: false, score: 25, pv: solved.lines[0].pv.slice(0, 1) }]
  });
  const restored = new AnalysisCache(storage).get(key);
  assert.equal(restored.solved, true);
  assert.equal(restored.lines[0].mateVerified, true);
}

// Following a verified PV rebases both the encoded mate distance and DTM. The
// child result must still validate against the exact child board.
{
  const firstUci = solved.lines[0].pv[0];
  const firstMove = uciToMove(root, firstUci);
  assert.ok(firstMove);
  const child = root.clone();
  EngineInternals.makeMove(child, firstMove);
  const childLine = rebaseVerifiedMateLine(solved.lines[0], 1);
  assert.ok(childLine);
  assert.equal(childLine.dtm, solved.lines[0].dtm - 1);
  assert.equal(validateMateResult(child, childLine), true, 'Rebased mate cache must replay from child position');
}

// The analysis worker must publish the solved cached result and finish without
// launching another proof/search cycle.
{
  const worker = new Worker(new URL('./worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
  const messages = [];
  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Solved resume timeout: ${JSON.stringify(messages)}`)), 6000);
    worker.on('error', reject);
    worker.on('message', message => {
      messages.push(message);
      if (message.type === 'ready') {
        worker.postMessage({
          type: 'start', token: 71, fen, cacheKey: 'solved-fixture',
          historyFens: [], effortMs: 550, multipv: 1, resumeResult: solved
        });
      }
      if (message.type === 'state' && message.token === 71 && message.state === 'complete') {
        setTimeout(() => { clearTimeout(timer); resolve(); }, 120);
      }
    });
  });
  try {
    await done;
    const infos = messages.filter(message => message.type === 'info' && message.token === 71);
    assert.equal(infos.length, 1, 'Solved resume should emit only the cached result');
    assert.equal(infos[0].result.cached, true);
    assert.equal(infos[0].result.lines[0].mateVerified, true);
  } finally {
    await worker.terminate();
  }
}

// Static guards for the demanded lazy UI behavior.
{
  const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const app = fs.readFileSync(new URL('../app.js', import.meta.url), 'utf8');
  assert.match(html, /<details class="rules-card">/);
  assert.doesNotMatch(html, /<details class="rules-card"\s+open/);
  assert.match(html, /<details id="gameTreePanel" class="panel study-panel">/);
  assert.doesNotMatch(html, /id="gameTreePanel"[^>]*\sopen/);
  assert.doesNotMatch(app, /\nloadLibrary\(\);\s*$/m, 'PGNs must not load unconditionally at startup');
  assert.match(app, /ensureLibraryLoaded\(\)/);
}

console.log('v7 solved-cache, PV-rebase, worker-resume and lazy-load tests passed.');
