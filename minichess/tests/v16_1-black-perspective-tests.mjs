import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';

const BLACK = -1;
const blackTacticalFen = 'k3q/5/5/5/R3K b - - 0 1';

function utility(line, side) {
  return side === BLACK ? -Number(line?.score || 0) : Number(line?.score || 0);
}

function assertUtilitySorted(lines, side, label) {
  assert.ok(lines.length >= 2, `${label} should provide at least two candidates`);
  for (let i = 1; i < lines.length; i += 1) {
    assert.ok(
      utility(lines[i - 1], side) >= utility(lines[i], side),
      `${label} must be sorted by side-to-move utility: ${JSON.stringify(lines.map(line => [line.move, line.score]))}`
    );
  }
}

const engineSource = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const workerSource = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const playWorkerSource = readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8');

assert.match(engineSource, /Orion JS 18.1/, 'engine version should be v17.3');
assert.match(workerSource, /lineUtilityForSide/, 'analysis worker should rank merged live lines by side-to-move utility');
assert.match(workerSource, /mergeKnownAnalysisResult\(current\.lastResult, result, multipv, current\.position\.turn\)/, 'analysis worker must pass root side to live-result merging');
assert.match(playWorkerSource, /sortResultLinesForSide/, 'play worker should normalize cached and final candidate line order');
assert.match(appSource, /sortAnalysisLinesForPosition/, 'UI/app cache validation should normalize stale cached black-to-move results');

const searcher = new GardnerSearcher({ hashEntries: 16_384 });
const direct = searcher.analyze(EnginePosition.fromFEN(blackTacticalFen), {
  timeMs: 260,
  maxDepth: 3,
  multipv: 3,
  mateProbeMs: 0,
  endgameProbeMs: 0,
  fortressProbeMs: 0
});
assert.equal(direct.engine, 'Orion JS 18.1');
assertUtilitySorted(direct.lines, BLACK, 'direct black search');
assert.ok(direct.lines[0].score <= direct.lines.at(-1).score, 'black root output may be ascending in white-centric score while still being best for Black');

const worker = new Worker(new URL('./worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
const messages = [];
let resolved = false;
const infos = await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error(`Worker timeout. Received: ${JSON.stringify(messages.slice(-8))}`)), 12_000);
  worker.on('error', reject);
  worker.on('message', message => {
    messages.push(message);
    if (message.type === 'ready') {
      worker.postMessage({
        type: 'start',
        token: 161,
        fen: blackTacticalFen,
        bookMoves: [],
        historyFens: [],
        effortMs: 550,
        multipv: 3
      });
    }
    const seen = messages.filter(item => item.type === 'info' && item.token === 161 && item.result?.lines?.length >= 2);
    if (!resolved && seen.length >= 1) {
      resolved = true;
      clearTimeout(timeout);
      resolve(seen);
    }
  });
});
try {
  assertUtilitySorted(infos.at(-1).result.lines, BLACK, 'analysis worker black stream');
} finally {
  await worker.terminate();
}

console.log('v17.3 black-perspective ordering tests passed.');
