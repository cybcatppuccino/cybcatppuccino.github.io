import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';
import { Position } from '../js/core/position.js';
import { AI_STYLES, selectLineForStyle } from '../js/engine/difficulty.js';

const BLACK = -1;
function utility(line, side) {
  return side === BLACK ? -Number(line?.score || 0) : Number(line?.score || 0);
}
function assertUtilitySorted(lines, side, label) {
  assert.ok(lines.length >= 2, `${label} should have at least two lines`);
  for (let i = 1; i < lines.length; i += 1) {
    assert.ok(utility(lines[i - 1], side) >= utility(lines[i], side), `${label} order is not side-safe: ${JSON.stringify(lines.map(line => [line.move, line.score, line.scoreText]))}`);
  }
}

const engineSource = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const playClientSource = readFileSync(new URL('../js/engine/play-client.js', import.meta.url), 'utf8');
const playWorkerSource = readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
assert.match(engineSource, /proof insertion root-perspective safe/, 'mate proof insertion should be explicitly root-perspective safe');
assert.match(appSource, /analysisModeAllowed\(\).*?gameMode === 'local'/s, 'manual analysis mode should be local-only');
assert.match(appSource, /playClient\.pause\(\)/, 'analysis panel pause button should pause AI thinking during play');
assert.match(appSource, /handleAiInfoResult/, 'AI internal thinking info should be rendered through the analysis panel');
assert.match(playClientSource, /onInfo/, 'play client should forward streamed AI info');
assert.match(playWorkerSource, /message.type === 'pause'/, 'play worker should accept pause messages');
assert.match(playWorkerSource, /activeElapsed/, 'play worker should track active elapsed time separately from paused time');

const blackMatingFen = 'k4/5/5/4q/4K b - - 0 1';
const searcher = new GardnerSearcher({ hashEntries: 32768 });
const result = searcher.analyze(EnginePosition.fromFEN(blackMatingFen), {
  timeMs: 650,
  maxDepth: 5,
  multipv: 4,
  mateProbeMs: 300,
  mateMaxPlies: 15,
  endgameProbeMs: 0,
  fortressProbeMs: 0
});
assertUtilitySorted(result.lines, BLACK, 'black mating analysis');
assert.ok(result.lines[0].score < 0, 'black winning mate should remain negative from White display perspective');

const best = { move: 'a1a2', score: -280, scoreText: '-2.80', pv: ['a1a2'], styleProfile: { replyGap: 0, goodReplyCount: 4 } };
const trap = { move: 'a1b1', score: -310, scoreText: '-3.10', pv: ['a1b1'], styleProfile: { replyGap: 160, goodReplyCount: 1, bestReplyQuiet: true, opponentLegal: 5, quiet: true, pressureEdge: 2 } };
const blunder = { move: 'a1c1', score: -520, scoreText: '-5.20', pv: ['a1c1'], styleProfile: { replyGap: 240, goodReplyCount: 1, bestReplyQuiet: true, opponentLegal: 6, quiet: true, pressureEdge: 2 } };
assert.equal(selectLineForStyle([best, trap, blunder], 'cunning', 'w'), trap, 'Cunning may choose a near-equivalent trap in bad/equal positions');
const winningBest = { ...best, score: 420, scoreText: '+4.20' };
const flashyInferior = { ...trap, score: 330, scoreText: '+3.30' };
assert.equal(selectLineForStyle([winningBest, flashyInferior], 'cunning', 'w'), winningBest, 'Cunning should preserve a stable advantage before style preferences');

const worker = new Worker(new URL('./play-worker-node-wrapper.mjs', import.meta.url), { type: 'module' });
try {
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('play worker ready timeout')), 5000);
    worker.on('error', reject);
    worker.on('message', message => {
      if (message.type === 'ready') {
        clearTimeout(timeout);
        resolve();
      }
    });
  });
  const seen = [];
  const token = 171;
  const final = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`play worker pause/resume timeout: ${JSON.stringify(seen.slice(-10))}`)), 15000);
    worker.on('message', message => {
      if (message.token !== token) return;
      seen.push(message);
      if (message.type === 'info' && !seen.some(item => item.type === 'state' && item.state === 'paused')) {
        worker.postMessage({ type: 'pause', token });
        setTimeout(() => worker.postMessage({ type: 'resume', token }), 80);
      }
      if (message.type === 'result') {
        clearTimeout(timeout);
        resolve(message.result);
      }
      if (message.type === 'error') {
        clearTimeout(timeout);
        reject(new Error(message.message));
      }
    });
    worker.postMessage({
      type: 'search',
      token,
      fen: Position.initial().toCompactFEN(),
      historyFens: [],
      cacheKey: 'v17.2-play-pause',
      style: 'cunning',
      thinkTimeMs: 5000
    });
  });
  assert.ok(seen.some(message => message.type === 'info'), 'play worker should stream AI internal info');
  assert.ok(seen.some(message => message.type === 'state' && message.state === 'paused'), 'play worker should acknowledge pause');
  assert.ok(final.selectedMove, 'play worker should still return a move after resume');
  assert.equal(final.style, 'cunning');
} finally {
  await worker.terminate();
}

assert.ok(AI_STYLES.every(style => style.maxDepth >= 40 || style.id === 'balanced'), 'v17.2 styles should keep high depth ceilings');
console.log('v17.2 AI pause/style and mate-order tests passed.');
