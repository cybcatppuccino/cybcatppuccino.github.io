import { EnginePosition, GardnerSearcher, ENGINE_VERSION, validateMateResult } from './engine.js';
import { GardnerTablebase } from './tablebase.js';

const MAX_DEPTH = 48;
const searcher = new GardnerSearcher({ hashEntries: 524288 });
const tablebase = new GardnerTablebase();
// v11: start manifest fetch as soon as the worker is created so the later
// "Checking tablebase" phase usually waits only for the actual block/probe.
tablebase.init().catch(() => {});
const positionCache = new Map();
const CACHE_LIMIT = 72;
let activeToken = 0;
let running = false;
let paused = false;
let current = null;
let nextDepth = 1;
let effortMs = 950;
let currentBudgetMs = 90;
let multipv = 3;
let firstChunk = true;
let totalNodes = 0;
let totalElapsed = 0;

function isSolvedResult(result) {
  return Boolean(result?.tablebase || result?.fortressProof || result?.lines?.[0]?.mateVerified || (result?.solved && result?.lines?.[0]?.mateVerified));
}

async function probeTablebase(token) {
  if (!current || token !== activeToken) return false;
  try {
    post('state', {
      token,
      state: 'probing',
      engine: ENGINE_VERSION,
      depth: Number(current.resumeResult?.depth || 0),
      searchDepth: nextDepth
    });
    const result = await tablebase.analyze(current.position.clone(), { multipv });
    if (!result || token !== activeToken || !current) return false;
    const solved = {
      ...result,
      cacheKey: current.cacheKey,
      cached: false,
      solved: true,
      searchDepth: 0,
      nextDepth: 0
    };
    current.lastResult = solved;
    cacheResult(current.cacheKey, solved);
    running = false;
    paused = false;
    post('info', { token, result: solved });
    post('state', {
      token,
      state: 'complete',
      engine: result.engineLabel || ENGINE_VERSION,
      depth: 0,
      searchDepth: 0,
      tablebase: true
    });
    return true;
  } catch {
    return false;
  }
}

async function startPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  let resumeResult = bestResume(message, cacheKey);
  if (resumeResult?.lines?.length) {
    const validatedLines = resumeResult.lines.filter(line => !line.mateVerified || validateMateResult(position, line));
    if (resumeResult.lines[0]?.mateVerified && validatedLines[0] !== resumeResult.lines[0]) {
      resumeResult = null;
      positionCache.delete(cacheKey);
    } else {
      resumeResult = { ...resumeResult, lines: validatedLines, solved: isSolvedResult({ ...resumeResult, lines: validatedLines }) };
    }
  }
  const solvedResume = isSolvedResult(resumeResult);
  current = {
    position,
    cacheKey,
    resumeResult,
    lastResult: resumeResult,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(fen => {
      const historyPosition = EnginePosition.fromFEN(fen);
      return { a: historyPosition.hashA, b: historyPosition.hashB };
    })
  };
  effortMs = Math.max(200, Math.min(2400, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  nextDepth = Math.max(1, Number(resumeResult?.depth || 0) + 1);
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = Math.max(0, Number(resumeResult?.nodes || 0));
  totalElapsed = Math.max(0, Number(resumeResult?.elapsed || 0));
  paused = Boolean(message.startPaused);
  running = !paused && !solvedResume;

  if (resumeResult?.lines?.length) {
    post('info', {
      token,
      result: {
        ...resumeResult,
        cacheKey,
        cached: true,
        searchDepth: solvedResume ? 0 : nextDepth,
        nextDepth: solvedResume ? 0 : nextDepth
      }
    });
  }
  post('state', {
    token,
    state: solvedResume ? 'complete' : paused ? 'paused' : 'thinking',
    engine: resumeResult?.engineLabel || ENGINE_VERSION,
    depth: Number(resumeResult?.depth || 0),
    searchDepth: solvedResume ? 0 : nextDepth
  });
  if (!running) return;
  if (await probeTablebase(token)) return;
  if (token === activeToken && running && !paused) schedule(token);
}

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function schedule(token, delay = 0) {
  setTimeout(() => runChunk(token), delay);
}

function initialBudget(depth) {
  if (depth <= 1) return 70;
  if (depth === 2) return 100;
  if (depth === 3) return 145;
  return Math.min(effortMs, 190 + (depth - 4) * 82);
}

function cacheResult(key, result) {
  if (!key || !result?.lines?.length) return;
  const previous = positionCache.get(key);
  const previousSolved = isSolvedResult(previous?.result);
  const nextSolved = isSolvedResult(result);
  if (previousSolved && !nextSolved) {
    previous.updatedAt = Date.now();
  } else if (!previous || nextSolved || Number(result.depth || 0) >= Number(previous.result?.depth || 0)) {
    positionCache.set(key, { updatedAt: Date.now(), result });
  } else {
    previous.updatedAt = Date.now();
  }
  if (positionCache.size > CACHE_LIMIT) {
    const oldest = [...positionCache.entries()].sort((a, b) => a[1].updatedAt - b[1].updatedAt)[0];
    if (oldest) positionCache.delete(oldest[0]);
  }
}

function bestResume(message, key) {
  const internalCandidate = positionCache.get(key)?.result || null;
  const externalCandidate = message.resumeResult || null;
  const internal = internalCandidate?.engine === ENGINE_VERSION ? internalCandidate : null;
  const external = externalCandidate?.engine === ENGINE_VERSION ? externalCandidate : null;
  if (!internal) return external;
  if (!external) return internal;
  const internalSolved = isSolvedResult(internal);
  const externalSolved = isSolvedResult(external);
  if (internalSolved !== externalSolved) return internalSolved ? internal : external;
  return Number(internal.depth || 0) >= Number(external.depth || 0) ? internal : external;
}

function runChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    const mateBudget = Math.min(520, Math.max(70, Math.round(currentBudgetMs * 0.28)));
    const mainBudget = Math.max(70, currentBudgetMs - mateBudget);
    const result = searcher.analyze(current.position.clone(), {
      timeMs: mainBudget,
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      bookMoves: current.bookMoves,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
      resumeResult: firstChunk ? current.resumeResult : null,
      endgameProbeMs: 70,
      fortressProbeMs: 150,
      mateProbeMs: mateBudget,
      mateMaxPlies: 81
    });
    firstChunk = false;
    current.resumeResult = null;
    if (!running || paused || token !== activeToken) return;

    totalNodes += result.nodes;
    totalElapsed += result.elapsed;
    if (result.completed) {
      nextDepth = result.nextDepth;
      currentBudgetMs = initialBudget(nextDepth);
    } else {
      currentBudgetMs = Math.min(Math.max(effortMs, 320), Math.round(currentBudgetMs * 1.5));
    }

    const cumulative = {
      ...result,
      nodes: totalNodes,
      elapsed: totalElapsed,
      nps: Math.round(totalNodes * 1000 / Math.max(1, totalElapsed)),
      searchDepth: nextDepth,
      searchBudget: currentBudgetMs,
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(result)
    };
    current.lastResult = cumulative;
    cacheResult(current.cacheKey, cumulative);
    post('info', { token, result: cumulative });

    const mateFound = Boolean(result.lines[0]?.mateVerified);
    if (result.terminal || result.fortressProof || mateFound || nextDepth > MAX_DEPTH) {
      running = false;
      post('state', { token, state: 'complete', engine: ENGINE_VERSION, depth: result.depth, searchDepth: nextDepth });
      return;
    }
    post('state', { token, state: 'thinking', engine: ENGINE_VERSION, depth: result.depth, searchDepth: nextDepth });
    schedule(token, 7);
  } catch (error) {
    running = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'start' || message.type === 'position') {
    void startPosition(message).catch(error => {
      running = false;
      post('error', { token: Number(message.token || activeToken), message: error?.stack || error?.message || String(error) });
    });
    return;
  }
  if (message.type === 'pause') {
    if (Number(message.token) !== activeToken || !current) return;
    paused = true;
    running = false;
    post('state', {
      token: activeToken,
      state: 'paused',
      engine: ENGINE_VERSION,
      depth: Number(current.lastResult?.depth || 0),
      searchDepth: nextDepth
    });
    return;
  }
  if (message.type === 'resume') {
    if (Number(message.token) !== activeToken || !current) return;
    if (isSolvedResult(current.lastResult)) {
      paused = false;
      running = false;
      post('state', {
        token: activeToken,
        state: 'complete',
        engine: ENGINE_VERSION,
        depth: Number(current.lastResult?.depth || 0),
        searchDepth: nextDepth
      });
      return;
    }
    paused = false;
    running = true;
    post('state', {
      token: activeToken,
      state: 'thinking',
      engine: ENGINE_VERSION,
      depth: Number(current.lastResult?.depth || 0),
      searchDepth: nextDepth
    });
    schedule(activeToken);
    return;
  }
  if (message.type === 'stop') {
    activeToken = Number(message.token || activeToken + 1);
    running = false;
    paused = false;
    current = null;
    post('state', { token: activeToken, state: 'idle', engine: ENGINE_VERSION });
    return;
  }
  if (message.type === 'clear') {
    searcher.clear();
    positionCache.clear();
    nextDepth = 1;
    currentBudgetMs = initialBudget(1);
    post('state', { token: activeToken, state: paused ? 'paused' : running ? 'thinking' : 'idle', engine: ENGINE_VERSION });
  }
});

post('ready', { engine: ENGINE_VERSION });
