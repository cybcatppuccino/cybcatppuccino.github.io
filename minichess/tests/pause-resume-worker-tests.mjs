import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

const worker = new Worker(new URL('./worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
const infos = [];
let pausedDepth = 0;
const done = new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('Pause/resume worker timeout')), 15_000);
  worker.on('error', reject);
  worker.on('message', message => {
    if (message.type === 'ready') {
      worker.postMessage({
        type: 'start', token: 31,
        fen: 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1',
        historyFens: [], effortMs: 550, multipv: 3, cacheKey: 'pause-test'
      });
    }
    if (message.type === 'info' && message.token === 31) {
      infos.push(message.result);
      if (infos.length === 2 && !pausedDepth) {
        pausedDepth = message.result.depth;
        worker.postMessage({ type: 'pause', token: 31 });
      }
      if (pausedDepth && message.result.depth > pausedDepth) {
        clearTimeout(timeout);
        resolve();
      }
    }
    if (message.type === 'state' && message.token === 31 && message.state === 'paused') {
      setTimeout(() => worker.postMessage({ type: 'resume', token: 31 }), 50);
    }
  });
});
try {
  await done;
  assert.ok(infos.at(-1).depth > pausedDepth, 'Resume must continue to a deeper completed iteration');
  console.log('Analysis worker pause/resume test passed.');
} finally {
  await worker.terminate();
}
