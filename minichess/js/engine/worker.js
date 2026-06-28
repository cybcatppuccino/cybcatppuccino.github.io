import { EnginePosition, GardnerSearcher, ENGINE_VERSION, validateMateResult } from './engine.js';
import { GardnerTablebase } from './tablebase.js';
import { isSolvedResult, isTrustedExactTablebaseResult, resultPvProfile, withResultQuality } from './result-quality.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';

// v19.5 analysis worker
// Result ownership rule: every published score/PV pair comes from one completed
// iteration (or one exact proof). Incomplete chunks may update progress only.
const MAX_DEPTH = 48;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
searcher.setTablebaseProbe(position => tablebase.probeWdlSync(position));
tablebase.init().catch(() => {});

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
let currentKernel = ENGINE_KERNELS.ORION;

const TABLEBASE_PROMOTION_RETRY_MS = 220;
const historyFenKeyCache = new Map();
const HISTORY_FEN_KEY_CACHE_LIMIT = 256;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function reportFatalWorkerError(error, token = activeToken) {
  const message = error?.stack || error?.message || String(error || 'Unknown worker error.');
  try { post('error', { token: Number(token || activeToken || 0), message }); } catch {}
}

self.addEventListener('error', event => reportFatalWorkerError(event?.error || event?.message || 'Worker script error.'));
self.addEventListener('unhandledrejection', event => reportFatalWorkerError(event?.reason || 'Unhandled worker promise rejection.'));

function historyKeyFromFen(fen) {
  const key = String(fen || '');
  const cached = historyFenKeyCache.get(key);
  if (cached) return cached;
  const position = EnginePosition.fromFEN(key);
  const value = { a: position.hashA, b: position.hashB };
  historyFenKeyCache.set(key, value);
  if (historyFenKeyCache.size > HISTORY_FEN_KEY_CACHE_LIMIT) {
    const oldest = historyFenKeyCache.keys().next().value;
    if (oldest !== undefined) historyFenKeyCache.delete(oldest);
  }
  return value;
}

function lineUtilityForSide(line, sideToMove) {
  const score = Number(line?.score || 0);
  return sideToMove === -1 ? -score : score;
}

function sortResultLinesForSide(result, sideToMove, limit = 3) {
  if (!result || !Array.isArray(result.lines)) return result;
  const maxLines = Math.max(1, Math.min(5, Number(limit || result.lines.length || 3)));
  const lines = result.lines
    .map(line => ({ ...line, pv: Array.isArray(line?.pv) ? line.pv.slice() : [] }))
    .sort((a, b) => lineUtilityForSide(b, sideToMove) - lineUtilityForSide(a, sideToMove))
    .slice(0, maxLines);
  return { ...result, lines };
}

function initialBudget(depth) {
  if (depth <= 1) return 70;
  if (depth === 2) return 100;
  if (depth === 3) return 145;
  return Math.min(effortMs, 190 + (depth - 4) * 82);
}

function schedule(token, delay = 0) {
  setTimeout(() => void runChunk(token), delay);
}

function isTrustedResume(position, candidate) {
  if (!candidate?.lines?.length || candidate.engine !== ENGINE_VERSION || !isSolvedResult(candidate)) return null;
  const normalized = withResultQuality(sortResultLinesForSide(candidate, position.turn, multipv));
  if (isTrustedExactTablebaseResult(normalized)) return normalized;
  const first = normalized.lines[0];
  const verifiedMate = Boolean(first?.mateVerified && (first?.mateProof || first?.endgameProof || normalized?.mateProof || normalized?.endgameProof));
  if (verifiedMate && validateMateResult(position, first)) return normalized;
  if (normalized.fortressProof || normalized.endgameProof || normalized.terminal) return normalized;
  return null;
}

function isStableSearchResult(result) {
  if (!result?.lines?.length) return false;
  if (isSolvedResult(result)) return true;
  return Boolean(
    result.completed !== false &&
    result.pvComplete !== false &&
    !result.pvIncomplete &&
    result.multiPvVerified !== false
  );
}

function progressFromStable(stable, snapshot, requestedDepth) {
  if (!stable?.lines?.length) return null;
  const nodes = totalNodes + Math.max(0, Number(snapshot?.nodes || 0));
  const elapsed = totalElapsed + Math.max(0, Number(snapshot?.elapsed || 0));
  return {
    ...stable,
    // Keep score/PV/proof fields unchanged. Only progress metrics advance.
    nodes,
    elapsed,
    nps: Math.round(nodes * 1000 / Math.max(1, elapsed)),
    selDepth: Math.max(Number(stable.selDepth || 0), Number(snapshot?.selDepth || 0)),
    searchDepth: Math.max(1, Number(requestedDepth || nextDepth)),
    nextDepth: Math.max(1, Number(requestedDepth || nextDepth)),
    nodeTarget: updateDepthNodeEstimate(current, nodes, requestedDepth),
    cacheKey: current.cacheKey,
    cached: false,
    liveProgress: true,
    liveUpdate: true
  };
}

function estimateDepthNodes(state, depth) {
  const requested = Math.max(1, Number(depth || 1));
  const costs = state?.depthNodeCosts || new Map();
  const previous = Math.max(0, Number(costs.get(requested - 1) || 0));
  const beforePrevious = Math.max(0, Number(costs.get(requested - 2) || 0));
  let estimate;
  if (previous > 0 && beforePrevious > 0) {
    const growth = Math.max(1.25, Math.min(3.25, previous / Math.max(1, beforePrevious)));
    estimate = previous * growth;
  } else if (previous > 0) {
    estimate = previous * (requested <= 3 ? 1.55 : 1.85);
  } else if (totalNodes > 0) {
    estimate = Math.max(2_000, totalNodes * 0.45);
  } else {
    estimate = 2_000 * Math.pow(1.62, Math.max(0, requested - 1));
  }
  return Math.max(1_000, Math.min(250_000_000, Math.round(estimate)));
}

function beginDepthNodeEstimate(state, depth) {
  if (!state) return 0;
  const requested = Math.max(1, Number(depth || 1));
  if (state.progressDepth !== requested || state.progressTargetNodes <= totalNodes) {
    state.progressDepth = requested;
    state.progressTargetNodes = totalNodes + estimateDepthNodes(state, requested);
  }
  return state.progressTargetNodes;
}

function updateDepthNodeEstimate(state, visibleNodes, depth) {
  const requested = Math.max(1, Number(depth || state?.progressDepth || 1));
  let target = beginDepthNodeEstimate(state, requested);
  const currentNodes = Math.max(0, Number(visibleNodes || 0));
  if (currentNodes >= target) {
    target = Math.max(currentNodes + 1_000, Math.round(currentNodes * 1.28));
    state.progressTargetNodes = target;
  }
  return target;
}

function postLiveProgress(token, snapshot, requestedDepth) {
  if (!current || token !== activeToken || !running || paused || !snapshot) return;
  // No raw live PV reaches the UI. Until the first completed iteration exists,
  // the panel remains in its existing searching state instead of displaying a
  // one-ply / transient score.
  const live = progressFromStable(current.lastResult, snapshot, requestedDepth);
  if (live) post('info', { token, result: live });
}

function publishExactTablebaseResult(token, result) {
  if (!current || token !== activeToken || !result || !isTrustedExactTablebaseResult(result)) return false;
  const solved = withResultQuality({
    ...result,
    cacheKey: current.cacheKey,
    cached: false,
    solved: true,
    multiPvVerified: true,
    searchDepth: 0,
    nextDepth: 0
  });
  current.lastResult = solved;
  current.tablebasePromotionPending = false;
  current.tablebasePromotionQueued = false;
  current.tablebasePromotionResolved = true;
  post('info', { token, result: solved });
  running = false;
  paused = false;
  post('state', { token, state: 'complete', engine: result.engineLabel || ENGINE_VERSION, depth: 0, searchDepth: 0, tablebase: true });
  return true;
}

async function probeTablebase(token, { announce = true } = {}) {
  if (!current || token !== activeToken) return false;
  try {
    if (announce) post('state', { token, state: 'probing', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    const result = await tablebase.analyze(current.position.clone(), { multipv });
    if (!result || token !== activeToken || !current) return false;
    return publishExactTablebaseResult(token, result);
  } catch {
    return false;
  }
}

function queueExactTablebasePromotion(token) {
  if (!current || token !== activeToken || current.tablebasePromotionQueued || current.tablebasePromotionResolved) return;
  const root = current.position.clone();
  current.tablebasePromotionQueued = true;
  current.tablebasePromotionPending = true;
  void (async () => {
    let promoted = false;
    try {
      const warmed = await tablebase.warmExactWdlNeighborhood(root.clone(), { includeLegalChildren: true });
      if (!current || token !== activeToken) return;
      if (warmed) promoted = await probeTablebase(token, { announce: false });
      if (!promoted && current && token === activeToken) {
        await new Promise(resolve => setTimeout(resolve, TABLEBASE_PROMOTION_RETRY_MS));
        if (current && token === activeToken) promoted = await probeTablebase(token, { announce: false });
      }
    } finally {
      if (!current || token !== activeToken || promoted || isTrustedExactTablebaseResult(current.lastResult)) return;
      current.tablebasePromotionQueued = false;
      current.tablebasePromotionPending = false;
      current.tablebasePromotionResolved = true;
    }
  })();
}

async function startFairyPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.FAIRY;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  current = { position, cacheKey, lastResult: null, bookMoves: [], historyKeys: [] };
  running = true;
  paused = false;
  effortMs = Math.max(200, Math.min(30000, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  post('state', { token, state: 'thinking', engine: FAIRY_STOCKFISH_LABEL, depth: 0, searchDepth: 0 });
  try {
    const raw = await fairyProvider.search({ token, fen: String(message.fen || '').trim(), timeMs: effortMs, multipv });
    if (token !== activeToken || !current) return;
    const result = validateExternalAnalysisResult(position, raw, { maxLines: multipv });
    if (!result) throw new Error('Fairy-Stockfish returned no fully legal Gardner PV.');
    const finalResult = withResultQuality(sortResultLinesForSide({
      ...result,
      cacheKey,
      cached: false,
      rootTurn: position.turn,
      searchDepth: 0,
      nextDepth: 0,
      completed: true,
      multiPvVerified: true,
      solved: false
    }, position.turn, multipv));
    current.lastResult = finalResult;
    running = false;
    paused = false;
    post('info', { token, result: finalResult });
    post('state', { token, state: 'complete', engine: FAIRY_STOCKFISH_LABEL, depth: finalResult.depth, searchDepth: 0 });
  } catch (error) {
    if (token !== activeToken) return;
    await startOrionPosition({ ...message, kernel: ENGINE_KERNELS.ORION });
  }
}

async function startOrionPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.ORION;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  effortMs = Math.max(200, Math.min(2400, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  const resumeResult = isTrustedResume(position, message.resumeResult);
  const tablebaseEligible = Number(position.pieceCount || 0) <= 5;
  current = {
    position,
    cacheKey,
    lastResult: resumeResult,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen),
    depthNodeCosts: new Map(),
    progressDepth: 0,
    progressTargetNodes: 0,
    tablebasePromotionPending: false,
    tablebasePromotionQueued: false,
    tablebasePromotionResolved: false
  };
  nextDepth = 1;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = 0;
  totalElapsed = 0;
  paused = Boolean(message.startPaused);
  const solvedResume = isSolvedResult(resumeResult);
  running = !paused && !solvedResume;

  if (resumeResult?.lines?.length) {
    post('info', { token, result: { ...resumeResult, cacheKey, cached: true, searchDepth: 0, nextDepth: 0 } });
  }
  post('state', {
    token,
    state: paused ? 'paused' : solvedResume ? 'complete' : 'thinking',
    engine: resumeResult?.engineLabel || ENGINE_VERSION,
    depth: Number(resumeResult?.depth || 0),
    searchDepth: solvedResume ? 0 : nextDepth
  });
  if (!running) return;
  if (tablebaseEligible) {
    if (await probeTablebase(token, { announce: true })) return;
    if (token !== activeToken || !current) return;
    queueExactTablebasePromotion(token);
  }
  if (token === activeToken && running && !paused) schedule(token);
}

async function startPosition(message) {
  return selectedKernel(message.kernel) === ENGINE_KERNELS.FAIRY ? startFairyPosition(message) : startOrionPosition(message);
}

async function runChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const mateBudget = Math.min(520, Math.max(70, Math.round(currentBudgetMs * 0.28)));
    const mainBudget = Math.max(70, currentBudgetMs - mateBudget);
    const raw = searcher.analyze(current.position.clone(), {
      timeMs: mainBudget,
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      bookMoves: current.bookMoves,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
      endgameProbeMs: 70,
      fortressProbeMs: 150,
      mateProbeMs: mateBudget,
      mateMaxPlies: 81,
      onProgress: snapshot => postLiveProgress(token, snapshot, requestedDepth)
    });
    firstChunk = false;
    if (!running || paused || token !== activeToken) return;

    const chunkNodes = Math.max(0, Number(raw.nodes || 0));
    totalNodes += chunkNodes;
    totalElapsed += Math.max(0, Number(raw.elapsed || 0));
    if (raw.completed) {
      if (Number(raw.depth || 0) > 0) current.depthNodeCosts.set(Number(raw.depth || 0), chunkNodes);
      nextDepth = Math.max(requestedDepth + 1, Number(raw.nextDepth || requestedDepth + 1));
      currentBudgetMs = initialBudget(nextDepth);
      current.progressDepth = 0;
      beginDepthNodeEstimate(current, nextDepth);
    } else {
      currentBudgetMs = Math.min(Math.max(effortMs, 320), Math.round(currentBudgetMs * 1.5));
      updateDepthNodeEstimate(current, totalNodes, requestedDepth);
    }

    const profile = resultPvProfile(raw);
    const cumulative = withResultQuality({
      ...raw,
      ...profile,
      nodes: totalNodes,
      elapsed: totalElapsed,
      nps: Math.round(totalNodes * 1000 / Math.max(1, totalElapsed)),
      scoreDepth: Number(raw.scoreDepth || raw.depth || 0),
      searchDepth: nextDepth,
      searchBudget: currentBudgetMs,
      nodeTarget: updateDepthNodeEstimate(current, totalNodes, nextDepth),
      rootTurn: Number(raw.rootTurn || current.position.turn),
      lowProgressAudit: Boolean(raw.lowProgressAudit),
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(raw)
    });

    // Atomic publication: never merge PVs, proof labels or scores from prior
    // and current iterations. A partial depth only refreshes stable metrics.
    let visible = null;
    if (isStableSearchResult(cumulative)) {
      current.lastResult = cumulative;
      visible = cumulative;
    } else {
      visible = progressFromStable(current.lastResult, cumulative, requestedDepth);
    }
    if (visible) post('info', { token, result: visible });

    const stable = current.lastResult;
    const mateFound = Boolean(stable?.lines?.[0]?.mateVerified && stable?.solved);
    if (stable?.terminal || stable?.fortressProof || mateFound || isSolvedResult(stable) || nextDepth > MAX_DEPTH) {
      running = false;
      post('state', { token, state: 'complete', engine: ENGINE_VERSION, depth: Number(stable?.depth || raw.depth || 0), searchDepth: nextDepth });
      return;
    }
    post('state', { token, state: 'thinking', engine: ENGINE_VERSION, depth: Number(stable?.depth || 0), searchDepth: nextDepth });
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
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    post('state', { token: activeToken, state: 'paused', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    return;
  }
  if (message.type === 'resume') {
    if (Number(message.token) !== activeToken || !current) return;
    if (isSolvedResult(current.lastResult)) {
      paused = false;
      running = false;
      post('state', { token: activeToken, state: 'complete', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
      return;
    }
    paused = false;
    running = true;
    post('state', { token: activeToken, state: 'thinking', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    schedule(activeToken);
    return;
  }
  if (message.type === 'stop') {
    activeToken = Number(message.token || activeToken + 1);
    running = false;
    paused = false;
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    current = null;
    post('state', { token: activeToken, state: 'idle', engine: ENGINE_VERSION });
    return;
  }
  if (message.type === 'clear') {
    // Clear search accelerators only on an explicit user reset. A normal root
    // transition retains TT/eval ordering but never retains analysis conclusions.
    searcher.clear();
    fairyProvider.stop();
    nextDepth = 1;
    currentBudgetMs = initialBudget(1);
    post('state', { token: activeToken, state: paused ? 'paused' : running ? 'thinking' : 'idle', engine: ENGINE_VERSION });
  }
});

post('ready', { engine: ENGINE_VERSION });
