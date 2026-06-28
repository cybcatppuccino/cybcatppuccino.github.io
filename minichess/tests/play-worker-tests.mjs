import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { Position } from '../js/core/position.js';
import { legalMoves } from '../js/core/rules.js';
import { moveToUci } from '../js/core/notation.js';

const worker = new Worker(new URL('./play-worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
const fen = Position.initial().toCompactFEN();
const cacheKey = 'play-worker-v9-resume-test';

function waitForResult(token, payload) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`Play worker timeout for token ${token}`)), 12_000);
    const onError = error => {
      clearTimeout(timeout);
      reject(error);
    };
    const onMessage = message => {
      if (message.type === 'error' && message.token === token) {
        cleanup();
        reject(new Error(message.message));
      }
      if (message.type === 'result' && message.token === token) {
        cleanup();
        resolve(message.result);
      }
    };
    const cleanup = () => {
      clearTimeout(timeout);
      worker.off('error', onError);
      worker.off('message', onMessage);
    };
    worker.on('error', onError);
    worker.on('message', onMessage);
    worker.postMessage({ type: 'search', token, fen, historyFens: [], cacheKey, ...payload });
  });
}

await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('Play worker ready timeout')), 5_000);
  worker.on('error', reject);
  worker.on('message', message => {
    if (message.type === 'ready') {
      clearTimeout(timeout);
      resolve();
    }
  });
});

try {
  const first = await waitForResult(9, { style: 'balanced' });
  assert.ok(first.selectedMove);
  const legal = new Set(legalMoves(Position.initial()).map(moveToUci));
  assert.ok(legal.has(first.selectedMove), `Expected legal AI move, received ${first.selectedMove}`);
  assert.equal(first.style, 'balanced');
  assert.equal(first.styleLabel, 'Balanced');
  assert.ok(first.depth >= 1);

  const resumed = await waitForResult(10, { style: 'balanced', resumeResult: first });
  assert.ok(resumed.selectedMove);
  assert.ok(legal.has(resumed.selectedMove));
  assert.ok(resumed.depth >= first.depth);
  assert.equal(resumed.style, 'balanced');
  assert.equal(resumed.cached, false, 'Normal analysis results must be freshly searched rather than resumed.');

  const aggressive = await waitForResult(11, {
    style: 'aggressive',
    cacheKey: `${cacheKey}-aggressive`
  });
  assert.ok(legal.has(aggressive.selectedMove));
  assert.equal(aggressive.style, 'aggressive');
  assert.ok(aggressive.lines.length >= 2, 'Style search should retain multiple objective candidates');
  console.log('v19.5 finite play-engine style and fresh-search tests passed.');
} finally {
  await worker.terminate();
}
