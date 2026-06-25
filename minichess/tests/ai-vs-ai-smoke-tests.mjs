import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { Position } from '../js/core/position.js';
import { gameStatus, legalMoves } from '../js/core/rules.js';
import { moveToUci } from '../js/core/notation.js';

const worker = new Worker(new URL('./play-worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
let token = 20;

await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('AI-vs-AI worker ready timeout')), 5_000);
  worker.on('error', reject);
  worker.on('message', message => {
    if (message.type === 'ready') {
      clearTimeout(timeout);
      resolve();
    }
  });
});

function search(position, historyFens) {
  const requestToken = token++;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`AI-vs-AI timeout at token ${requestToken}`)), 8_000);
    const onMessage = message => {
      if (message.type === 'error' && message.token === requestToken) {
        cleanup();
        reject(new Error(message.message));
      }
      if (message.type === 'result' && message.token === requestToken) {
        cleanup();
        resolve(message.result);
      }
    };
    const cleanup = () => {
      clearTimeout(timeout);
      worker.off('message', onMessage);
    };
    worker.on('message', onMessage);
    worker.postMessage({
      type: 'search',
      token: requestToken,
      fen: position.toCompactFEN(),
      historyFens,
      cacheKey: `smoke-${position.canonicalKey()}-${position.halfmove}`,
      level: 1
    });
  });
}

try {
  let position = Position.initial();
  const history = [];
  let played = 0;
  for (; played < 10; played += 1) {
    const status = gameStatus(position);
    if (status.state !== 'playing' && status.state !== 'check') break;
    const result = await search(position, history);
    const legal = legalMoves(position);
    const move = legal.find(candidate => moveToUci(candidate) === result.selectedMove);
    assert.ok(move, `AI returned illegal move ${result.selectedMove} at ply ${played}`);
    history.push(position.toCompactFEN());
    position = position.makeMove(move);
  }
  assert.ok(played >= 4, 'AI-vs-AI smoke game should advance several legal plies');
  console.log(`AI-vs-AI finite-search smoke test passed (${played} plies).`);
} finally {
  await worker.terminate();
}
