import { EnginePosition, GardnerSearcher, ENGINE_VERSION, EngineInternals, validateMateResult } from './engine.js';
import { MinifishSearcher, MINIFISH_VERSION } from './minifish.js';
import { GardnerTablebase } from './tablebase.js';
import { compareAnalysisResults, isSolvedResult, isTrustedExactTablebaseResult, resultPvProfile, withResultQuality } from './result-quality.js';
import { isPublishableLine } from './result-contract.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';

// v22.7 analysis worker
// Tablebases are search-tree leaves: they provide exact WDL/DTM cutoffs once a
// <=5-piece position is resident.  There is deliberately no root-to-tablebase
// bridge prover, background AND/OR certificate, or independent mate prover.
const MAX_DEPTH = 48;
const { makeMove, undoMove } = EngineInternals;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const minifish = new MinifishSearcher();
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
function tablebaseProbeAtSearchNode(position, options = {}) {
  const wantsExactDtm = Boolean(options?.forceExactDtm || options?.exactDtm || options?.requireExactDtm);
  const exactOnly = Boolean(options?.skipWdlOnly || options?.requireExactDtm);
  if (wantsExactDtm) {
    const exactHit = tablebase.probeExactSync(position);
    if (exactHit) return exactHit;
    tablebase.requestExactDtmFromSearch(position, options);
    if (exactOnly) return null;
  }
  const hit = tablebase.probeSync(position);
  // A cache miss is queued only for this actual search node. Exact-DTM callers
  // request the DTM block; ordinary callers keep the legacy WDL prefetch path.
  if (!hit) {
    if (wantsExactDtm) tablebase.requestExactDtmFromSearch(position, options);
    else tablebase.requestWdlFromSearch(position);
    return null;
  }
  if (exactOnly && !hit.exactDtm) return null;
  return hit;
}

tablebaseProbeAtSearchNode.ensureExactDtm = (position, options = {}) => tablebase.requestExactDtmFromSearch(position, options);
tablebaseProbeAtSearchNode.loadExactDtm = (position, options = {}) => tablebase.requestExactDtmFromSearch(position, options);
tablebaseProbeAtSearchNode.probeExactDtm = position => tablebase.probeExactSync(position);

searcher.setTablebaseProbe(tablebaseProbeAtSearchNode);
minifish.setTablebaseProbe(tablebaseProbeAtSearchNode);
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

const historyFenKeyCache = new Map();
const HISTORY_FEN_KEY_CACHE_LIMIT = 256;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function activeEngineLabel() {
  if (currentKernel === ENGINE_KERNELS.MINIFISH) return MINIFISH_VERSION;
  if (currentKernel === ENGINE_KERNELS.FAIRY) return FAIRY_STOCKFISH_LABEL;
  return ENGINE_VERSION;
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

function relabelTablebaseResult(result, engineLabel) {
  if (!result) return result;
  return withResultQuality({
    ...result,
    engine: engineLabel,
    engineLabel: `${engineLabel} + GTB`,
    tablebase: true,
    tablebaseRoot: true,
    lines: Array.isArray(result.lines) ? result.lines.map(line => ({
      ...line,
      sourceEngine: result.engine || line.sourceEngine || '',
      scoreKind: line.scoreKind || (Number(line.tablebaseWdl || 0) ? 'mate' : 'evaluation'),
      tablebase: true,
      tablebaseRoot: true,
      tablebaseExactDtm: true,
      mateVerified: Number(line.tablebaseWdl || 0) !== 0
    })) : []
  });
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
  if (first?.mateVerified && validateMateResult(position, first)) return normalized;
  if (normalized.terminal) return normalized;
  return null;
}

function isStableSearchResult(result) {
  if (!result?.lines?.length) return false;
  if (isSolvedResult(result)) return true;
  if (result.completed === false || result.pvComplete === false || result.pvIncomplete || result.multiPvVerified === false) return false;
  const visibleCount = Math.max(1, Math.min(Number(result.multipv || multipv || 1), result.lines.length));
  return result.lines.slice(0, visibleCount).every(line => isPublishableLine(line, result));
}

function stabilizePublishedMetrics(state, result) {
  if (!state || !result) return result;
  const nodes = Math.max(0, Number(result.nodes || 0));
  const elapsed = Math.max(0, Number(result.elapsed || 0));
  const priorNodes = Math.max(0, Number(state.maxPublishedNodes || 0));
  const priorElapsed = Math.max(0, Number(state.maxPublishedElapsed || 0));
  const visibleNodes = Math.max(priorNodes, nodes);
  const visibleElapsed = Math.max(priorElapsed, elapsed);
  const proposedTarget = Math.max(0, Number(result.nodeTarget || 0));
  let target = Math.max(proposedTarget, Number(state.maxPublishedNodeTarget || 0));
  if (target && target <= visibleNodes) target = Math.max(target, visibleNodes + Math.max(1000, Math.round(visibleNodes * 0.18)));
  state.maxPublishedNodes = visibleNodes;
  state.maxPublishedElapsed = visibleElapsed;
  state.maxPublishedNodeTarget = target;
  return {
    ...result,
    nodes: visibleNodes,
    elapsed: visibleElapsed,
    nps: Math.round(visibleNodes * 1000 / Math.max(1, visibleElapsed)),
    nodeTarget: target
  };
}

function progressFromStable(stable, snapshot, requestedDepth) {
  if (!stable?.lines?.length) return null;
  const nodes = totalNodes + Math.max(0, Number(snapshot?.nodes || 0));
  const elapsed = totalElapsed + Math.max(0, Number(snapshot?.elapsed || 0));
  return stabilizePublishedMetrics(current, {
    ...stable,
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
  });
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
    estimate = Math.max(2000, totalNodes * 0.45);
  } else {
    estimate = 2000 * Math.pow(1.62, Math.max(0, requested - 1));
  }
  return Math.max(1000, Math.min(250000000, Math.round(estimate)));
}

function beginDepthNodeEstimate(state, depth) {
  if (!state) return 0;
  const requested = Math.max(1, Number(depth || 1));
  const cumulativeFloor = Math.max(totalNodes, Number(state.maxPublishedNodes || 0));
  if (state.progressDepth !== requested || state.progressTargetNodes <= cumulativeFloor) {
    state.progressDepth = requested;
    state.progressTargetNodes = Math.max(
      Number(state.progressTargetNodes || 0),
      Number(state.maxPublishedNodeTarget || 0),
      cumulativeFloor + estimateDepthNodes(state, requested)
    );
  }
  return state.progressTargetNodes;
}

function updateDepthNodeEstimate(state, visibleNodes, depth) {
  const requested = Math.max(1, Number(depth || state?.progressDepth || 1));
  let target = beginDepthNodeEstimate(state, requested);
  const currentNodes = Math.max(0, Number(visibleNodes || 0), Number(state?.maxPublishedNodes || 0));
  if (currentNodes >= target) {
    target = Math.max(target, currentNodes + 1000, Math.round(currentNodes * 1.28), Number(state?.maxPublishedNodeTarget || 0));
    state.progressTargetNodes = target;
  }
  return target;
}

function postLiveProgress(token, snapshot, requestedDepth) {
  if (!current || token !== activeToken || !running || paused || !snapshot) return;
  const live = progressFromStable(current.lastResult, snapshot, requestedDepth);
  if (live) post('info', { token, result: live });
}

function publishRootTablebaseResult(token, result) {
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
  current.lastResult = stabilizePublishedMetrics(current, solved);
  running = false;
  post('info', { token, result: current.lastResult });
  post('state', { token, state: 'complete', engine: result.engineLabel || ENGINE_VERSION, depth: 0, searchDepth: 0, tablebase: true });
  return true;
}

async function probeRootTablebase(token, { announce = true } = {}) {
  // This is only a direct root answer. There is no six-piece bridge, frontier
  // traversal, preload task, or asynchronous proof path.
  if (!current || token !== activeToken || Number(current.position?.pieceCount || 0) > 5) return false;
  try {
    const result = await tablebase.analyze(current.position.clone(), { multipv });
    if (!current || token !== activeToken || !result) return false;
    const normalized = relabelTablebaseResult(result, activeEngineLabel());
    if (!announce) return Boolean(normalized);
    return publishRootTablebaseResult(token, normalized);
  } catch {
    return false;
  }
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
  } catch {
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
  effortMs = Math.max(200, Math.min(5000, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  const resumeResult = isTrustedResume(position, message.resumeResult);
  current = {
    position,
    cacheKey,
    lastResult: resumeResult,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen),
    depthNodeCosts: new Map(),
    progressDepth: 0,
    progressTargetNodes: 0,
    maxPublishedNodes: 0,
    maxPublishedElapsed: 0,
    maxPublishedNodeTarget: 0
  };
  nextDepth = 1;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = 0;
  totalElapsed = 0;
  paused = Boolean(message.startPaused);
  const solvedResume = isSolvedResult(current.lastResult);
  running = !paused && !solvedResume;
  if (current.lastResult?.lines?.length) {
    post('info', { token, result: { ...current.lastResult, cacheKey, cached: Boolean(resumeResult), searchDepth: 0, nextDepth: 0 } });
  }
  post('state', {
    token,
    state: paused ? 'paused' : solvedResume ? 'complete' : 'thinking',
    engine: current.lastResult?.engineLabel || ENGINE_VERSION,
    depth: Number(current.lastResult?.depth || 0),
    searchDepth: solvedResume ? 0 : nextDepth
  });
  if (!running) return;
  if (await probeRootTablebase(token, { announce: true })) return;
  if (token === activeToken && running && !paused) schedule(token);
}

async function startMinifishPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.MINIFISH;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  effortMs = Math.max(120, Math.min(3000, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  current = {
    position,
    cacheKey,
    lastResult: null,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen),
    depthNodeCosts: new Map(),
    progressDepth: 0,
    progressTargetNodes: 0,
    maxPublishedNodes: 0,
    maxPublishedElapsed: 0,
    maxPublishedNodeTarget: 0,
    minifish: true
  };
  nextDepth = 1;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = 0;
  totalElapsed = 0;
  paused = Boolean(message.startPaused);
  running = !paused;
  post('state', { token, state: paused ? 'paused' : 'thinking', engine: MINIFISH_VERSION, depth: 0, searchDepth: nextDepth });
  if (!running) return;
  if (await probeRootTablebase(token, { announce: true })) return;
  schedule(token, 0);
}

function advanceBudget(raw, requestedDepth) {
  if (raw.completed) {
    if (Number(raw.depth || 0) > 0) current.depthNodeCosts.set(Number(raw.depth || 0), Math.max(0, Number(raw.nodes || 0)));
    nextDepth = Math.max(requestedDepth + 1, Number(raw.nextDepth || requestedDepth + 1));
    currentBudgetMs = initialBudget(nextDepth);
    current.progressDepth = 0;
    beginDepthNodeEstimate(current, nextDepth);
  } else {
    // A depth is retried with a larger slice; it is not capped at 900 ms.
    currentBudgetMs = Math.min(effortMs, Math.max(320, Math.round(currentBudgetMs * 1.35)));
    updateDepthNodeEstimate(current, totalNodes, requestedDepth);
  }
}

function publishChunk(token, raw, requestedDepth, engineLabel, extra = {}) {
  const profile = resultPvProfile(raw);
  const cumulative = withResultQuality(sortResultLinesForSide({
    ...raw,
    ...profile,
    ...extra,
    nodes: totalNodes,
    elapsed: totalElapsed,
    nps: Math.round(totalNodes * 1000 / Math.max(1, totalElapsed)),
    scoreDepth: Number(raw.scoreDepth || raw.depth || 0),
    searchDepth: nextDepth,
    searchBudget: currentBudgetMs,
    nodeTarget: updateDepthNodeEstimate(current, totalNodes, nextDepth),
    rootTurn: Number(raw.rootTurn || current.position.turn),
    cacheKey: current.cacheKey,
    cached: false,
    solved: isSolvedResult(raw)
  }, current.position.turn, multipv));
  const normalized = stabilizePublishedMetrics(current, cumulative);
  let visible = null;
  if (isStableSearchResult(normalized)) {
    const chosen = compareAnalysisResults(current.lastResult, normalized, { preferNextOnTie: true });
    if (chosen === normalized) {
      current.lastResult = normalized;
      visible = normalized;
    } else {
      visible = progressFromStable(current.lastResult, normalized, requestedDepth);
    }
  } else {
    visible = progressFromStable(current.lastResult, normalized, requestedDepth);
  }
  if (visible) post('info', { token, result: visible });
  const stable = current.lastResult;
  if (stable?.terminal || isSolvedResult(stable) || nextDepth > MAX_DEPTH) {
    running = false;
    post('state', { token, state: 'complete', engine: engineLabel, depth: Number(stable?.depth || raw.depth || 0), searchDepth: nextDepth });
    return true;
  }
  post('state', { token, state: 'thinking', engine: engineLabel, depth: Number(stable?.depth || 0), searchDepth: nextDepth });
  return false;
}

async function runMinifishChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const raw = minifish.analyze(current.position.clone(), {
      timeMs: Math.max(45, currentBudgetMs),
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
      onProgress: snapshot => postLiveProgress(token, snapshot, requestedDepth)
    });
    firstChunk = false;
    if (!running || paused || token !== activeToken) return;
    totalNodes += Math.max(0, Number(raw.nodes || 0));
    totalElapsed += Math.max(0, Number(raw.elapsed || 0));
    advanceBudget(raw, requestedDepth);
    if (publishChunk(token, raw, requestedDepth, MINIFISH_VERSION, { minifish: true })) return;
    schedule(token, 0);
  } catch (error) {
    running = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

async function runChunk(token) {
  if (currentKernel === ENGINE_KERNELS.MINIFISH) return runMinifishChunk(token);
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const raw = searcher.analyze(current.position.clone(), {
      timeMs: Math.max(90, currentBudgetMs),
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      bookMoves: current.bookMoves,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
      onProgress: snapshot => postLiveProgress(token, snapshot, requestedDepth)
    });
    firstChunk = false;
    if (!running || paused || token !== activeToken) return;
    totalNodes += Math.max(0, Number(raw.nodes || 0));
    totalElapsed += Math.max(0, Number(raw.elapsed || 0));
    advanceBudget(raw, requestedDepth);
    if (publishChunk(token, raw, requestedDepth, ENGINE_VERSION)) return;
    schedule(token, 7);
  } catch (error) {
    running = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

async function startPosition(message) {
  const kernel = selectedKernel(message.kernel);
  if (kernel === ENGINE_KERNELS.FAIRY) return startFairyPosition(message);
  if (kernel === ENGINE_KERNELS.MINIFISH) return startMinifishPosition(message);
  return startOrionPosition(message);
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
    post('state', { token: activeToken, state: 'paused', engine: activeEngineLabel(), depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    return;
  }
  if (message.type === 'resume') {
    if (Number(message.token) !== activeToken || !current) return;
    if (isSolvedResult(current.lastResult)) {
      paused = false;
      running = false;
      post('state', { token: activeToken, state: 'complete', engine: activeEngineLabel(), depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
      return;
    }
    paused = false;
    running = true;
    post('state', { token: activeToken, state: 'thinking', engine: activeEngineLabel(), depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    schedule(activeToken);
    return;
  }
  if (message.type === 'stop') {
    activeToken = Number(message.token || activeToken + 1);
    running = false;
    paused = false;
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    current = null;
    post('state', { token: activeToken, state: 'idle', engine: activeEngineLabel() });
    return;
  }
  if (message.type === 'clear') {
    searcher.clear();
    fairyProvider.stop();
    nextDepth = 1;
    currentBudgetMs = initialBudget(1);
    post('state', { token: activeToken, state: paused ? 'paused' : running ? 'thinking' : 'idle', engine: activeEngineLabel() });
  }
});

post('ready', { engine: ENGINE_VERSION });
