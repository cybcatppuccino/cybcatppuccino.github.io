/*
 * Browser-side adapter example for fairy-stockfish-nnue.wasm 1.1.11.
 * Put this file in the same directory as stockfish.js, stockfish.wasm,
 * and stockfish.worker.js, then create it with:
 *   new Worker('/vendor/fairy-stockfish/fairy-uci-worker.example.js')
 *
 * Protocol in:
 *   { type:'init', hashMb?:32, threads?:1 }
 *   { type:'search', token, fen, timeMs?:1000, depth?:0, multipv?:3, turn?:'w'|'b' }
 *   { type:'stop' }
 * Protocol out:
 *   { type:'ready' }
 *   { type:'state', token, state:'thinking'|'complete' }
 *   { type:'info', token, result }
 *   { type:'error', message }
 */

importScripts('./stockfish.js');

const ENGINE_LABEL = 'Fairy-Stockfish wasm 1.1.11';
const MATE = 30000;
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

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function send(command) {
  engine.postMessage(command);
}

function waitFor(pattern, timeoutMs = 8000) {
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
  // Stockfish reports root score from the side-to-move perspective; v12.2 UI
  // uses white-centric scores.
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
  const lines = [...currentLines.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, line]) => line)
    .slice(0, currentMultipv);
  if (bestmove && !lines.find(line => line.move === bestmove)) {
    lines.unshift({
      move: bestmove,
      score: 0,
      scoreText: '0.00',
      pv: [bestmove],
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
  resolveWaits(String(line || ''));
  parseInfo(String(line || ''));
  const match = String(line || '').match(/^bestmove\s+(\S+)/);
  if (match && Number(activeToken)) emitResult(activeToken, match[1]);
}

async function init(options = {}) {
  if (initialized) return;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const base = self.location.href.replace(/[^/]*$/, '');
    engine = await Stockfish({ locateFile: file => base + file });
    engine.addMessageListener(handleEngineLine);
    send('uci');
    await waitFor('uciok');
    send('setoption name UCI_Variant value gardner');
    send(`setoption name Hash value ${Math.max(16, Math.min(1024, Number(options.hashMb || 32)))}`);
    send(`setoption name Threads value ${Math.max(1, Math.min(8, Number(options.threads || 1)))}`);
    send('setoption name Use NNUE value true');
    send('isready');
    await waitFor('readyok');
    initialized = true;
    post('ready', { engine: ENGINE_LABEL });
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

init({}).catch(error => post('error', { message: error?.stack || error?.message || String(error) }));
