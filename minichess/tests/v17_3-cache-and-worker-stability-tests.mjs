import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

import { AnalysisCache } from '../js/engine/analysis-cache.js';
import { ENGINE_VERSION } from '../js/engine/engine.js';

{
  const storage = new Map();
  const fakeStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key)
  };
  const cache = new AnalysisCache(fakeStorage);
  const incomplete = {
    engine: ENGINE_VERSION,
    depth: 8,
    scoreDepth: 8,
    pvComplete: false,
    completed: false,
    lines: [{ move: 'a1a2', score: 15, pv: ['a1a2', 'a5a4', 'a2a3'] }]
  };
  const returned = cache.set('first-incomplete-result', incomplete);
  assert.equal(returned?.pvComplete, false, 'first streamed incomplete result may be returned to the caller');
  assert.equal(cache.get('first-incomplete-result'), null, 'first incomplete result must not be persisted as a resume artifact');
}

{
  const worker = new Worker(new URL('./worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
  const messages = [];
  const errors = [];
  const fen = 'nrbkq/ppppp/5/PPPPP/QKRBN w - - 0 1';
  const done = new Promise((resolve, reject) => {
    const timeout = setTimeout(resolve, 8000);
    worker.on('error', reject);
    worker.on('message', message => {
      messages.push(message);
      if (message.type === 'ready') {
        worker.postMessage({
          type: 'start',
          token: 173,
          fen,
          bookMoves: [],
          historyFens: [],
          effortMs: 700,
          multipv: 3
        });
      }
      if (message.type === 'error') {
        errors.push(message.message || 'worker error');
        clearTimeout(timeout);
        resolve();
      }
      if (messages.filter(item => item.type === 'info' && item.token === 173).length >= 3) {
        clearTimeout(timeout);
        resolve();
      }
    });
  });
  await done;
  const infos = messages.filter(item => item.type === 'info' && item.token === 173);
  assert.equal(errors.length, 0, `analysis worker should not throw on the reported start position: ${errors.join('\n')}`);
  assert.ok(infos.length >= 1, 'analysis worker should stream at least one result');
  assert.ok(infos.at(-1).result.lines?.[0]?.move, 'latest streamed result should contain a legal best move');
  await worker.terminate();
}

console.log('v17.3 cache and worker stability tests passed.');
