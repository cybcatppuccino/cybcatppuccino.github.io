import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { readFile, stat } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';

// v20.3: the Kb1 six-piece bridge is proven proactively, not only after an
// ordinary PV happens to touch an exact table.  Displayed outcomes are either a
// real numeric score or a verified mate bound; no TB WDL label/capped sentinel
// may reach the UI.
const TABLES_ROOT = path.resolve(fileURLToPath(new URL('../tools/gardner_tablebase/tables/', import.meta.url)));

function startTablebaseServer() {
  const server = http.createServer(async (request, response) => {
    try {
      const relative = decodeURIComponent(new URL(request.url || '/', 'http://127.0.0.1').pathname)
        .replace(/^\/tools\/gardner_tablebase\/tables\//, '');
      const file = path.resolve(TABLES_ROOT, relative);
      if (!file.startsWith(`${TABLES_ROOT}${path.sep}`)) throw new Error('outside table root');
      const info = await stat(file);
      if (!info.isFile()) throw new Error('not a file');
      response.writeHead(200, { 'Content-Type': file.endsWith('.json') ? 'application/json' : 'application/octet-stream' });
      response.end(await readFile(file));
    } catch {
      response.writeHead(404).end();
    }
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

function legalDisplay(line) {
  if (line?.mateVerified) return /^≤?#\d+|^-?#\d+/.test(String(line.scoreText || ''));
  return /^[-+]?\d+\.\d{2}$/.test(String(line?.scoreText || ''));
}

function startWorkerSearch(worker, token, fen, cacheKey, { timeoutMs = 18_000 } = {}) {
  return new Promise((resolve, reject) => {
    const infos = [];
    const timer = setTimeout(() => finish(new Error(`Timed out for ${cacheKey}`)), timeoutMs);
    const finish = error => {
      clearTimeout(timer);
      worker.off('message', onMessage);
      if (error) reject(error); else resolve(infos);
    };
    const onMessage = message => {
      if (message.token !== token) return;
      if (message.type === 'error') return finish(new Error(message.message));
      if (message.type !== 'info' || !message.result?.lines?.length) return;
      infos.push(message.result);
      const line = message.result.lines[0];
      assert.ok(legalDisplay(line), `Illegal v20.3 display text: ${line?.scoreText || '<empty>'}`);
      assert.notEqual(line.scoreText, 'TB win');
      assert.notEqual(line.scoreText, 'TB loss');
      assert.notEqual(line.scoreText, 'TB draw');
      if (message.result.tablebaseBridgeProof) finish();
    };
    worker.on('message', onMessage);
    worker.postMessage({
      type: 'start', token, fen, cacheKey, historyFens: [], bookMoves: [], effortMs: 2400, multipv: 1
    });
  });
}

const server = await startTablebaseServer();
const worker = new Worker(new URL('./worker-node-tablebase-wrapper.mjs', import.meta.url), {
  type: 'module',
  workerData: { baseUrl: `http://127.0.0.1:${server.address().port}/` }
});

try {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Worker did not become ready.')), 2_000);
    worker.on('message', message => {
      if (message.type === 'ready') {
        clearTimeout(timer);
        resolve();
      }
    });
  });

  const rootFen = '5/k4/p1p2/2P1P/2K2 w - - 0 3';
  const rootInfos = await startWorkerSearch(worker, 201, rootFen, 'v20-kb1-root');
  const rootBridge = rootInfos.at(-1);
  const rootLine = rootBridge.lines[0];
  assert.equal(rootLine.move, 'c1b1');
  assert.equal(rootLine.scoreText, '≤#21');
  assert.equal(rootLine.tablebaseBridgeDtm, 41);
  assert.equal(rootLine.mateVerified, true);
  assert.equal(rootLine.mateUpperBound, true);
  assert.equal(rootLine.rootScoreExact, false, 'The 41-ply bridge is an upper bound, not exact DTM.');

  let priorNodes = 0;
  let priorTarget = 0;
  for (const result of rootInfos) {
    const nodes = Number(result.nodes || 0);
    const target = Number(result.nodeTarget || 0);
    assert.ok(nodes >= priorNodes, `nodes regressed: ${nodes} < ${priorNodes}`);
    assert.ok(target >= priorTarget, `node target regressed: ${target} < ${priorTarget}`);
    priorNodes = nodes;
    priorTarget = target;
  }

  const variants = [
    ['...a2', '5/k4/2p2/p1P1P/1K3 w - - 0 4', '≤#13', 25],
    ['...Kb4', '5/1k3/p1p2/2P1P/1K3 w - - 2 4', '≤#20', 39],
    ['...Ka5', 'k4/5/p1p2/2P1P/1K3 w - - 2 4', '≤#20', 39],
    ['...Kb5', '1k3/5/p1p2/2P1P/1K3 w - - 2 4', '≤#20', 39]
  ];
  let token = 202;
  for (const [label, fen, expectedText, expectedDtm] of variants) {
    const infos = await startWorkerSearch(worker, token, fen, `v20-kb1-${token}`, { timeoutMs: 18_000 });
    token += 1;
    const firstBridge = infos.find(result => result.tablebaseBridgeProof) || infos.at(-1);
    const line = firstBridge.lines[0];
    assert.equal(firstBridge.tablebaseBridgeProof, true, `${label} should prove a mate bridge.`);
    assert.equal(line.scoreText, expectedText, `${label} wrong bound.`);
    assert.equal(line.tablebaseBridgeDtm, expectedDtm, `${label} wrong proof distance.`);
    assert.ok(Number(line.score || 0) > 29_000, `${label} must not leak an opposite ordinary score.`);
  }

  console.log('v20.3 Kb1 proactive bridge / all-defence display test passed.', {
    root: rootLine.scoreText,
    rootNodes: priorNodes,
    rootNodeTarget: priorTarget,
    variations: variants.map(([label]) => label)
  });
} finally {
  worker.postMessage({ type: 'stop', token: 299 });
  await worker.terminate();
  await new Promise(resolve => server.close(resolve));
}
