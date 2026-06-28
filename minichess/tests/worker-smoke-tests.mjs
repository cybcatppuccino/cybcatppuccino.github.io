import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

const worker = new Worker(new URL('./worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
const messages = [];
let ready = false;
let resolved = false;

const completion = new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error(`Worker timeout. Received: ${JSON.stringify(messages.slice(-8))}`)), 12_000);
  worker.on('error', reject);
  worker.on('message', message => {
    messages.push(message);
    if (message.type === 'ready' && !ready) {
      ready = true;
      worker.postMessage({
        type: 'start',
        token: 17,
        fen: 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1',
        bookMoves: [],
        historyFens: [],
        effortMs: 550,
        multipv: 3
      });
    }
    const infos = messages.filter(item => item.type === 'info' && item.token === 17);
    if (!resolved && infos.length >= 3 && infos.at(-1).result.lines.length >= 1) {
      resolved = true;
      clearTimeout(timeout);
      resolve(infos);
    }
  });
});

try {
  const infos = await completion;
  assert.ok(infos[0].result.depth >= 1);
  assert.ok(infos.at(-1).result.depth >= infos[0].result.depth);
  assert.ok(infos.at(-1).result.nodes > 0);
  assert.ok(infos.at(-1).result.nps > 0);
  assert.ok(infos.at(-1).result.lines.length >= 1);
  assert.ok(infos.some(info => info.result.searchDepth > info.result.depth), 'Expected the worker to announce its next depth');
  assert.ok(infos.some(info => Number(info.result.nodeTarget || 0) > Number(info.result.nodes || 0)), 'Expected an adaptive node estimate for the next depth');
  for (let index = 1; index < infos.length; index += 1) {
    assert.ok(Number(infos[index].result.nodes || 0) >= Number(infos[index - 1].result.nodes || 0), 'Live NODES should never move backwards');
  }
  console.log('Gardner MiniChess worker streamed live iterative-deepening updates.');
  console.table(infos.slice(0, 5).map(info => ({
    depth: info.result.depth,
    selective: info.result.selDepth,
    searching: info.result.searchDepth,
    nodes: info.result.nodes,
    nps: info.result.nps,
    best: info.result.lines[0]?.move
  })));
} finally {
  await worker.terminate();
}
