/*
 * Browser-side UCI adapter for fairy-stockfish-nnue.wasm 1.1.11.
 *
 * v15.1 notes:
 * - This package is a pthread wasm build. Browsers require same-origin HTTP(S)
 *   plus COOP/COEP headers so SharedArrayBuffer is available. If startup fails,
 *   this worker emits an error immediately so Orion JS can fall back instead of
 *   leaving the UI at "Starting the local engine…".
 * - All moves are still validated by Orion before the UI or AI play layer uses
 *   them; this adapter only translates UCI text into the common result shape.
 */

const ENGINE_LABEL = 'Fairy-Stockfish wasm 1.1.11';
const MATE = 30000;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

let stockfishBootable = true;
if (typeof SharedArrayBuffer === 'undefined') {
  // v15.1: do not throw an uncaught worker error here. The provider will
  // fall back to Orion JS while the versioned COI helper prepares a
  // cross-origin isolated reload for future Fairy-Stockfish searches.
  stockfishBootable = false;
  post('error', {
    recoverable: true,
    message: 'Fairy-Stockfish requires SharedArrayBuffer. Run ./serve.sh or serve.bat, open http://127.0.0.1:8000, and allow the COI helper to reload. Orion JS will be used only until SharedArrayBuffer is available.'
  });
} else {
  try {
    importScripts('./stockfish.js');
  } catch (error) {
    stockfishBootable = false;
    post('error', { recoverable: true, message: `Unable to load vendor/fairy-stockfish/stockfish.js: ${error?.message || error}` });
  }
}

let engine = null;
let initialized = false;
let initPromise = null;
let pendingWaits = [];
let activeToken = 0;
let rootTurn = 'w';
let currentLines = new Map();
let currentDepth = 0;
let currentNodes = 0;
let currentNps = 0;
let currentStartedAt = 0;
let currentMultipv = 1;

function send(command) {
  if (!engine) throw new Error(`Fairy-Stockfish is not initialized; cannot send ${command}`);
  engine.postMessage(command);
}

function waitFor(pattern, timeoutMs = 12000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingWaits = pendingWaits.filter(item => item.resolve !== resolve);
      reject(new Error(`Timed out waiting for ${pattern}`));
    }, timeoutMs);
    pendingWaits.push({ pattern, resolve, timer });
  });
}

function resolveWaits(line) {
  for (const item of [...pendingWaits]) {
    const matched = typeof item.pattern === 'string'
      ? line === item.pattern
      : item.pattern.test(line);
    if (matched) {
      clearTimeout(item.timer);
      pendingWaits = pendingWaits.filter(candidate => candidate !== item);
      item.resolve(line);
    }
  }
}

function scoreText(score) {
  if (Math.abs(score) >= 29000) {
    const plies = Math.max(1, MATE - Math.abs(score));
    const moves = Math.ceil(plies / 2);
    return score > 0 ? `#${moves}` : `-#${moves}`;
  }
  const pawns = score / 100;
  return `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}`;
}

function parseInfo(line) {
  if (!line.startsWith('info ')) return;
  const tokens = line.trim().split(/\s+/);
  const depthIndex = tokens.indexOf('depth');
  const multipvIndex = tokens.indexOf('multipv');
  const scoreIndex = tokens.indexOf('score');
  const nodesIndex = tokens.indexOf('nodes');
  const npsIndex = tokens.indexOf('nps');
  const pvIndex = tokens.indexOf('pv');
  const depth = depthIndex >= 0 ? Number(tokens[depthIndex + 1] || 0) : currentDepth;
  const multipv = multipvIndex >= 0 ? Number(tokens[multipvIndex + 1] || 1) : 1;
  if (nodesIndex >= 0) currentNodes = Number(tokens[nodesIndex + 1] || currentNodes);
  if (npsIndex >= 0) currentNps = Number(tokens[npsIndex + 1] || currentNps);
  currentDepth = Math.max(currentDepth, depth || 0);
  if (scoreIndex < 0 || pvIndex < 0) return;

  let score = 0;
  const scoreKind = tokens[scoreIndex + 1];
  const rawScore = Number(tokens[scoreIndex + 2] || 0);
  if (scoreKind === 'mate') {
    const plies = Math.max(1, Math.abs(rawScore) * 2 - 1);
    score = rawScore > 0 ? MATE - plies : -MATE + plies;
  } else {
    score = rawScore;
  }
  // Fairy-Stockfish reports root score from the side-to-move perspective; the
  // Gardner UI uses white-centric scores.
  const whiteScore = rootTurn === 'w' ? score : -score;
  const pv = tokens.slice(pvIndex + 1).filter(Boolean);
  if (!pv.length) return;
  currentLines.set(multipv, {
    move: pv[0],
    score: whiteScore,
    scoreText: scoreText(whiteScore),
    pv,
    depth,
    mateVerified: false,
    source: 'fairy-stockfish'
  });
}

function emitResult(token, bestmove = '') {
  const best = bestmove && bestmove !== '(none)' ? bestmove : '';
  const lines = [...currentLines.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, line]) => line)
    .slice(0, currentMultipv);
  if (best && !lines.find(line => line.move === best)) {
    lines.unshift({
      move: best,
      score: 0,
      scoreText: '0.00',
      pv: [best],
      depth: currentDepth,
      mateVerified: false,
      source: 'fairy-stockfish'
    });
  }
  post('info', {
    token,
    result: {
      engine: ENGINE_LABEL,
      engineLabel: ENGINE_LABEL,
      depth: currentDepth,
      nodes: currentNodes,
      nps: currentNps,
      elapsed: Math.max(1, Math.round(performance.now() - currentStartedAt)),
      lines,
      completed: true,
      external: true,
      source: 'fairy-stockfish'
    }
  });
  post('state', { token, state: 'complete', engine: ENGINE_LABEL, depth: currentDepth });
}

function handleEngineLine(line) {
  const text = String(line || '');
  resolveWaits(text);
  parseInfo(text);
  const match = text.match(/^bestmove\s+(\S+)/);
  if (match && Number(activeToken)) emitResult(activeToken, match[1]);
}

async function init(options = {}) {
  if (!stockfishBootable || typeof Stockfish !== 'function') {
    throw new Error('Fairy-Stockfish wasm is not bootable in this browser context. Orion JS fallback is available.');
  }
  if (initialized) return;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      const base = self.location.href.replace(/[^/]*$/, '');
      engine = await Stockfish({ locateFile: file => base + file });
      engine.addMessageListener(handleEngineLine);

      const uciok = waitFor('uciok');
      send('uci');
      await uciok;

      send('setoption name UCI_Variant value gardner');
      send(`setoption name Hash value ${Math.max(16, Math.min(1024, Number(options.hashMb || 32)))}`);
      send('setoption name Threads value 1');
      send('setoption name Use NNUE value true');
      const readyok = waitFor('readyok');
      send('isready');
      await readyok;

      initialized = true;
      post('ready', { engine: ENGINE_LABEL });
    } catch (error) {
      post('error', { recoverable: true, message: error?.stack || error?.message || String(error) });
      self.close();
      return;
    }
  })();
  return initPromise;
}

async function search(message) {
  if (!initialized) await init({});
  activeToken = Number(message.token || activeToken + 1);
  rootTurn = message.turn === 'b' || /\sb\s/.test(String(message.fen || '')) ? 'b' : 'w';
  currentLines = new Map();
  currentDepth = 0;
  currentNodes = 0;
  currentNps = 0;
  currentStartedAt = performance.now();
  currentMultipv = Math.max(1, Math.min(8, Number(message.multipv || 1)));
  post('state', { token: activeToken, state: 'thinking', engine: ENGINE_LABEL });
  send('stop');
  send(`setoption name MultiPV value ${currentMultipv}`);
  send('ucinewgame');
  send(`position fen ${String(message.fen || '').trim()}`);
  if (Number(message.depth || 0) > 0) send(`go depth ${Math.max(1, Math.min(64, Number(message.depth)))}`);
  else send(`go movetime ${Math.max(50, Math.min(30000, Number(message.timeMs || 1000)))}`);
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'init') {
    init(message).catch(error => post('error', { message: error?.stack || error?.message || String(error) }));
  } else if (message.type === 'search') {
    search(message).catch(error => post('error', { token: message.token, message: error?.stack || error?.message || String(error) }));
  } else if (message.type === 'stop') {
    activeToken = 0;
    if (engine) send('stop');
  }
});

if (stockfishBootable) init({}).catch(() => {});
