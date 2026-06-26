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

function search(position, historyFens, style) {
  const requestToken = token++;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`AI-vs-AI timeout at token ${requestToken}`)), 12_000);
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
      cacheKey: `v9-smoke-${style}-${position.canonicalKey()}-${position.halfmove}`,
      style
    });
  });
}

try {
  let position = Position.initial();
  const history = [];
  const styles = ['balanced', 'pressing', 'conservative', 'aggressive'];
  let played = 0;
  for (; played < styles.length; played += 1) {
    const status = gameStatus(position);
    if (status.state !== 'playing' && status.state !== 'check') break;
    const result = await search(position, history, styles[played]);
    const legal = legalMoves(position);
    const move = legal.find(candidate => moveToUci(candidate) === result.selectedMove);
    assert.ok(move, `AI returned illegal move ${result.selectedMove} at ply ${played}`);
    assert.equal(result.style, styles[played]);
    history.push(position.toCompactFEN());
    position = position.makeMove(move);
  }
  assert.equal(played, styles.length, 'AI-vs-AI smoke game should advance one legal ply per tested style');
  console.log(`v9 AI-vs-AI full-strength style smoke test passed (${played} plies).`);
} finally {
  await worker.terminate();
}
