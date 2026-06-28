import { EnginePosition, GardnerSearcher, ENGINE_VERSION, validateMateResult } from './engine.js';
import { GardnerTablebase } from './tablebase.js';
import {
  compareAnalysisResults,
  isSolvedResult,
  isThinPvResult,
  isTrustedExactTablebaseResult,
  resultPvProfile,
  shouldCacheResult,
  withResultQuality
} from './result-quality.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';

const MAX_DEPTH = 48;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
searcher.setTablebaseProbe(position => tablebase.probeWdlSync(position));
// v12: start manifest and WDL warming as soon as the worker is created.
// Search uses only already-warmed WDL blocks synchronously; misses safely fall
// back to the normal alpha-beta path.
tablebase.init().catch(() => {});
const positionCache = new Map();
const CACHE_LIMIT = 216;
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

// Synchronous WDL probes use +/-220.00 as an internal ordering score. They
// are useful to the search, but should never displace an already-known GTB
// mate/DTM line in the analysis panel.
const TABLEBASE_PSEUDO_SCORE = 22000;
const TABLEBASE_PROMOTION_RETRY_MS = 220;

const historyFenKeyCache = new Map();
const HISTORY_FEN_KEY_CACHE_LIMIT = 256;
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



function stableLineRank(line) {
  if (!line) return 0;
  // A direct GTB line contains the database's optimal DTM continuation and
  // must be stable against later local mate / centipawn snapshots.
  if (line.tablebase && !line.tablebaseBound) return 90;
  if (line.mateVerified) return 80;
  if (line.tablebaseBound || line.dtmUpperBound) return 45;
  if (line.endgameProof || line.fortressProof) return 40;
  return 10;
}

function hasDatabaseMateLine(result) {
  return Boolean(result?.lines?.some(line => Boolean(line?.tablebase)
    && Math.abs(Number(line?.score || 0)) >= 29000
    && Number(line?.dtm || 0) > 0));
}

function hasTablebasePseudoScore(result) {
  return Boolean(result?.lines?.some(line => !line?.tablebase
    && Math.abs(Number(line?.score || 0)) === TABLEBASE_PSEUDO_SCORE));
}

function shouldDeferTablebasePseudoDisplay(result) {
  if (!current?.tablebasePromotionPending || !hasTablebasePseudoScore(result)) return false;
  // A bound annotation already carries concrete database mate information and
  // is preferable to hiding all analysis while the root DTM block is loading.
  return !hasDatabaseMateLine(result) && !hasDatabaseMateLine(current.lastResult);
}

function flushDeferredTablebaseDisplay(token) {
  if (!current || token !== activeToken) return;
  const deferred = current.deferredTablebaseResult;
  current.deferredTablebaseResult = null;
  if (deferred?.lines?.length) post('info', { token, result: deferred });
}

function isPvPrefix(prefix, full) {
  return prefix.length <= full.length && prefix.every((move, index) => move === full[index]);
}

function mergeStableLine(previousLine, nextLine) {
  if (!previousLine) return nextLine;
  if (!nextLine) return previousLine;
  const previousPv = Array.isArray(previousLine?.pv) ? previousLine.pv : [];
  const nextPv = Array.isArray(nextLine?.pv) ? nextLine.pv : [];
  const nextPreservesPreviousPrefix = Boolean(previousPv.length && nextPv.length < previousPv.length && isPvPrefix(nextPv, previousPv));
  const previousRank = stableLineRank(previousLine);
  const nextRank = stableLineRank(nextLine);
  if (previousRank > nextRank) {
    // A verified mate/tablebase bound is stronger than a later live centipawn
    // estimate for the same root move. Keep its score/badge stable, but only
    // retain an older PV tail if the newer live variation is its exact prefix.
    // A different second/third move must never inherit notation from another
    // branch just because both lines share the same root move.
    return {
      ...previousLine,
      pv: nextPv.length > previousPv.length || !nextPreservesPreviousPrefix ? nextPv.slice() : previousPv.slice(),
      pvPreservedFromCache: nextPreservesPreviousPrefix || Boolean(previousLine.pvPreservedFromCache),
      liveScoreSuppressed: true
    };
  }
  if (!previousPv.length || nextRank > previousRank || nextLine?.mateVerified || nextLine?.tablebase || nextLine?.fortressProof) return nextLine;
  if (nextPv.length >= previousPv.length || !nextPreservesPreviousPrefix) return nextLine;
  return { ...nextLine, pv: previousPv.slice(), pvPreservedFromCache: true };
}

function publishExactTablebaseResult(token, result) {
  if (!current || token !== activeToken || !result) return false;
  const tablebaseSolved = isTrustedExactTablebaseResult(result);
  const solved = withResultQuality({
    ...result,
    cacheKey: current.cacheKey,
    cached: false,
    solved: tablebaseSolved,
    searchDepth: tablebaseSolved ? 0 : nextDepth,
    nextDepth: tablebaseSolved ? 0 : nextDepth
  });
  current.lastResult = solved;
  current.resumeResult = solved;
  current.tablebasePromotionPending = false;
  current.tablebasePromotionQueued = false;
  current.tablebasePromotionResolved = true;
  current.deferredTablebaseResult = null;
  cacheResult(current.cacheKey, solved);
  post('info', { token, result: solved });
  if (!tablebaseSolved) return false;
  running = false;
  paused = false;
  post('state', {
    token,
    state: 'complete',
    engine: result.engineLabel || ENGINE_VERSION,
    depth: 0,
    searchDepth: 0,
    tablebase: true
  });
  return true;
}

async function probeTablebase(token, { announce = true } = {}) {
  if (!current || token !== activeToken) return false;
  try {
    if (announce) {
      post('state', {
        token,
        state: 'probing',
        engine: ENGINE_VERSION,
        depth: Number(current.resumeResult?.depth || 0),
        searchDepth: nextDepth
      });
    }
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
      // The initial exact probe can race the first five-piece block download.
      // Warm root/children at background priority, then retry inside the same
      // worker session rather than requiring the user to restart analysis.
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
      // Only after the bounded retry has genuinely failed may a synchronous WDL
      // sentinel be shown as the ordinary engine fallback.
      flushDeferredTablebaseDisplay(token);
    }
  })();
}


async function startFairyPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.FAIRY;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  current = { position, cacheKey, lastResult: null, resumeResult: null, bookMoves: [], historyKeys: [] };
  running = true;
  paused = false;
  effortMs = Math.max(200, Math.min(30000, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  post('state', { token, state: 'thinking', engine: FAIRY_STOCKFISH_LABEL, depth: 0, searchDepth: 0 });
  try {
    const raw = await fairyProvider.search({
      token,
      fen: String(message.fen || '').trim(),
      timeMs: effortMs,
      multipv
    });
    if (token !== activeToken || !current) return;
    const result = validateExternalAnalysisResult(position, raw, { maxLines: multipv });
    if (!result) throw new Error('Fairy-Stockfish returned no fully legal Gardner PV.');
    const finalResult = sortResultLinesForSide({
      ...result,
      cacheKey,
      cached: false,
      rootTurn: position.turn,
      searchDepth: 0,
      nextDepth: 0,
      solved: false
    }, position.turn, multipv);
    current.lastResult = finalResult;
    running = false;
    paused = false;
    post('info', { token, result: finalResult });
    post('state', { token, state: 'complete', engine: FAIRY_STOCKFISH_LABEL, depth: finalResult.depth, searchDepth: 0 });
  } catch (error) {
    if (token !== activeToken) return;
    // External engines are optional providers.  If the wasm worker is blocked
    // by browser policy, fails to initialize, or emits an illegal PV, fall back
    // to the native Orion searcher instead of surfacing a broken move.
    await startOrionPosition({ ...message, kernel: ENGINE_KERNELS.ORION });
  }
}

async function startPosition(message) {
  const kernel = selectedKernel(message.kernel);
  if (kernel === ENGINE_KERNELS.FAIRY) return startFairyPosition(message);
  return startOrionPosition(message);
}
async function startOrionPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.ORION;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  let resumeResult = bestResume(message, cacheKey);
  if (isThinPvResult(resumeResult)) {
    resumeResult = null;
    positionCache.delete(cacheKey);
  }
  if (resumeResult?.lines?.length) {
    const validatedLines = resumeResult.lines.filter(line => !line.mateVerified || validateMateResult(position, line));
    if (resumeResult.lines[0]?.mateVerified && validatedLines[0] !== resumeResult.lines[0]) {
      resumeResult = null;
      positionCache.delete(cacheKey);
    } else {
      resumeResult = { ...resumeResult, lines: validatedLines, solved: isSolvedResult({ ...resumeResult, lines: validatedLines }) };
    }
  }
  effortMs = Math.max(200, Math.min(2400, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  if (resumeResult?.lines?.length) resumeResult = withResultQuality(sortResultLinesForSide(resumeResult, position.turn, multipv));
  const solvedResume = isSolvedResult(resumeResult);
  const tablebaseEligible = Number(position.pieceCount || 0) <= 5;
  const trustedTablebaseResume = tablebaseEligible && isTrustedExactTablebaseResult(resumeResult);
  const resumeSearchDepth = Math.max(
    1,
    Number(resumeResult?.nextDepth || 0),
    Number(resumeResult?.searchDepth || 0),
    Number(resumeResult?.depth || 0) + 1
  );
  current = {
    position,
    cacheKey,
    resumeResult,
    lastResult: resumeResult,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen),
    // v18.4 keeps a small per-depth node-cost history for an honest UI estimate
    // of the total nodes likely needed to finish the currently requested depth.
    depthNodeCosts: new Map(),
    progressDepth: 0,
    progressTargetNodes: 0,
    tablebasePromotionPending: false,
    tablebasePromotionQueued: false,
    tablebasePromotionResolved: false,
    deferredTablebaseResult: null
  };
  nextDepth = resumeSearchDepth;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = Math.max(0, Number(resumeResult?.nodes || 0));
  totalElapsed = Math.max(0, Number(resumeResult?.elapsed || 0));
  paused = Boolean(message.startPaused);
  // v18.3: every direct current-version <=5-piece GTB result is trusted as a
  // solved no-50-move-rule result. Live resumes still continue from their next
  // depth instead of repeating completed work.
  running = !paused && (!solvedResume || (tablebaseEligible && !trustedTablebaseResume));

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
    state: paused ? 'paused' : solvedResume && (!tablebaseEligible || trustedTablebaseResume) ? 'complete' : 'thinking',
    engine: resumeResult?.engineLabel || ENGINE_VERSION,
    depth: Number(resumeResult?.depth || 0),
    searchDepth: solvedResume && (!tablebaseEligible || trustedTablebaseResume) ? 0 : nextDepth
  });
  if (!running) return;
  if (tablebaseEligible && !trustedTablebaseResume) {
    if (await probeTablebase(token, { announce: true })) return;
    if (token !== activeToken || !current) return;
    queueExactTablebasePromotion(token);
  }
  if (token !== activeToken || !running || paused) return;
  if (solvedResume) {
    running = false;
    post('state', { token, state: 'complete', engine: resumeResult?.engineLabel || ENGINE_VERSION, depth: Number(resumeResult?.depth || 0), searchDepth: 0 });
    return;
  }
  // For five-piece roots the queued promotion above owns the background warm
  // and will take over automatically when full DTM data becomes readable.
  if (token === activeToken && running && !paused) schedule(token);
}

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function schedule(token, delay = 0) {
  setTimeout(() => void runChunk(token), delay);
}

function initialBudget(depth) {
  if (depth <= 1) return 70;
  if (depth === 2) return 100;
  if (depth === 3) return 145;
  return Math.min(effortMs, 190 + (depth - 4) * 82);
}


function reportFatalWorkerError(error, token = activeToken) {
  const message = error?.stack || error?.message || String(error || 'Unknown worker error.');
  try { post('error', { token: Number(token || activeToken || 0), message }); } catch {}
}

self.addEventListener('error', event => {
  reportFatalWorkerError(event?.error || event?.message || 'Worker script error.');
});

self.addEventListener('unhandledrejection', event => {
  reportFatalWorkerError(event?.reason || 'Unhandled worker promise rejection.');
});


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

function mergeKnownAnalysisResult(previous, next, limit = 3, sideToMove = 1) {
  if (!next || !Array.isArray(next.lines)) return next;
  const maxLines = Math.max(1, Math.min(3, Number(limit || 3)));
  const merged = new Map();
  for (const line of Array.isArray(previous?.lines) ? previous.lines : []) {
    if (!line?.move) continue;
    merged.set(String(line.move), { ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] });
  }
  for (const line of next.lines) {
    if (!line?.move) continue;
    const key = String(line.move);
    const prior = merged.get(key);
    const candidate = { ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] };
    merged.set(key, mergeStableLine(prior, candidate));
  }
  const lines = [...merged.values()]
    .sort((a, b) => lineUtilityForSide(b, sideToMove) - lineUtilityForSide(a, sideToMove))
    .slice(0, maxLines);
  return { ...next, lines };
}

function cacheResult(key, result) {
  if (!key || !result?.lines?.length || !shouldCacheResult(result)) return;
  const normalized = withResultQuality(result);
  const previous = positionCache.get(key);
  const chosen = compareAnalysisResults(previous?.result || null, normalized);
  if (chosen === previous?.result) {
    if (previous) previous.updatedAt = Date.now();
  } else if (chosen) {
    positionCache.set(key, { updatedAt: Date.now(), result: chosen });
  }
  if (positionCache.size > CACHE_LIMIT) {
    // v14.3: avoid allocating and sorting the whole cache on every streamed
    // result.  A single pass is enough to evict the least recently updated
    // entry and keeps the larger cache cheap.
    let oldestKey = '';
    let oldestTime = Infinity;
    for (const [entryKey, entry] of positionCache) {
      if (entry.updatedAt < oldestTime) {
        oldestKey = entryKey;
        oldestTime = entry.updatedAt;
      }
    }
    if (oldestKey) positionCache.delete(oldestKey);
  }
}

function bestResume(message, key) {
  const internalCandidate = positionCache.get(key)?.result || null;
  const externalCandidate = message.resumeResult || null;
  const internal = internalCandidate?.engine === ENGINE_VERSION ? internalCandidate : null;
  const external = externalCandidate?.engine === ENGINE_VERSION ? externalCandidate : null;
  return compareAnalysisResults(internal, external, { preferNextOnTie: false });
}


function needsDtmAnnotation(result) {
  if (!result?.lines?.length || result.tablebase || result.fortressProof || result.terminal) return false;
  return result.lines.slice(0, Math.max(1, multipv)).some(line => !(line?.mateVerified || line?.fortressProof || line?.tablebaseBound || line?.dtmUpperBound || (line?.tablebase && line?.tablebaseExactDtm)));
}

function requestDtmAnnotation(token, baseResult) {
  if (!current || token !== activeToken || !needsDtmAnnotation(baseResult)) return;
  current.annotationPending = baseResult;
  if (current.annotationRunning) return;
  current.annotationRunning = true;
  void drainDtmAnnotations(token).catch(error => {
    if (error?.message && token === activeToken) {
      // Annotation is an optional enhancement; never fail the main search loop.
    }
  });
}

async function drainDtmAnnotations(token) {
  while (current && token === activeToken && current.annotationPending) {
    const baseResult = current.annotationPending;
    current.annotationPending = null;
    const annotated = await tablebase.annotateResultWithDtmBounds(current.position.clone(), baseResult, {
      maxLines: multipv,
      maxProbePly: 24
    });
    if (!current || token !== activeToken || paused) break;
    if (!annotated || annotated === baseResult || !annotated.tablebaseDtmBound) continue;
    const merged = mergeKnownAnalysisResult(current.lastResult, annotated, multipv, current.position.turn);
    const enriched = withResultQuality({
      ...merged,
      nodes: Math.max(Number(current.lastResult?.nodes || 0), Number(baseResult.nodes || 0)),
      elapsed: Math.max(Number(current.lastResult?.elapsed || 0), Number(baseResult.elapsed || 0)),
      nps: Number(current.lastResult?.nps || baseResult.nps || 0),
      scoreDepth: Number(merged.scoreDepth || merged.depth || baseResult.scoreDepth || 0),
      searchDepth: nextDepth,
      searchBudget: currentBudgetMs,
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(merged)
    });
    const chosen = compareAnalysisResults(current.lastResult, enriched);
    if (chosen !== current.lastResult) {
      current.lastResult = chosen;
      cacheResult(current.cacheKey, chosen);
      post('info', { token, result: chosen });
    }
  }
  if (current) current.annotationRunning = false;
  if (current && token === activeToken && current.annotationPending) requestDtmAnnotation(token, current.annotationPending);
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
    // A resumed result may not have per-depth timing history. Keep the first
    // estimate modest and allow it to adjust upward if the depth retries.
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
  // Estimates are deliberately elastic: a time-slice retry can consume more
  // nodes than a prior depth without pretending the denominator is exact.
  if (currentNodes >= target) {
    target = Math.max(currentNodes + 1_000, Math.round(currentNodes * 1.28));
    state.progressTargetNodes = target;
  }
  return target;
}

function postLiveProgress(token, snapshot, requestedDepth) {
  if (!current || token !== activeToken || !running || paused || !snapshot) return;
  const nodes = totalNodes + Math.max(0, Number(snapshot.nodes || 0));
  const elapsed = totalElapsed + Math.max(0, Number(snapshot.elapsed || 0));
  const nodeTarget = updateDepthNodeEstimate(current, nodes, requestedDepth);
  const live = {
    ...snapshot,
    nodes,
    elapsed,
    nps: Math.round(nodes * 1000 / Math.max(1, elapsed)),
    searchDepth: Math.max(1, Number(snapshot.searchDepth || requestedDepth || nextDepth)),
    nextDepth: Math.max(1, Number(snapshot.nextDepth || requestedDepth || nextDepth)),
    nodeTarget,
    cacheKey: current.cacheKey,
    cached: false,
    liveProgress: true,
    completed: false,
    solved: false
  };
  if (shouldDeferTablebasePseudoDisplay(live)) {
    current.deferredTablebaseResult = live;
    return;
  }
  post('info', { token, result: live });
}

async function runChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const mateBudget = Math.min(520, Math.max(70, Math.round(currentBudgetMs * 0.28)));
    const mainBudget = Math.max(70, currentBudgetMs - mateBudget);
    let result = searcher.analyze(current.position.clone(), {
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
      mateMaxPlies: 81,
      onProgress: snapshot => postLiveProgress(token, snapshot, requestedDepth)
    });
    firstChunk = false;
    current.resumeResult = null;
    if (!running || paused || token !== activeToken) return;
    result = mergeKnownAnalysisResult(current.lastResult, result, multipv, current.position.turn);

    const chunkNodes = Math.max(0, Number(result.nodes || 0));
    totalNodes += chunkNodes;
    totalElapsed += result.elapsed;
    if (result.completed) {
      if (Number(result.depth || 0) > 0) current.depthNodeCosts.set(Number(result.depth || 0), chunkNodes);
      nextDepth = result.nextDepth;
      currentBudgetMs = initialBudget(nextDepth);
      // Once a depth completes, immediately show the estimate for reaching the
      // next one rather than leaving the completed-depth denominator in place.
      current.progressDepth = 0;
      beginDepthNodeEstimate(current, nextDepth);
    } else {
      currentBudgetMs = Math.min(Math.max(effortMs, 320), Math.round(currentBudgetMs * 1.5));
      updateDepthNodeEstimate(current, totalNodes, requestedDepth);
    }

    const profile = resultPvProfile(result);
    const cumulative = withResultQuality({
      ...result,
      ...profile,
      nodes: totalNodes,
      elapsed: totalElapsed,
      nps: Math.round(totalNodes * 1000 / Math.max(1, totalElapsed)),
      scoreDepth: Number(result.scoreDepth || result.depth || 0),
      searchDepth: nextDepth,
      searchBudget: currentBudgetMs,
      nodeTarget: updateDepthNodeEstimate(current, totalNodes, nextDepth),
      rootTurn: Number(result.rootTurn || current.position.turn),
      lowProgressAudit: Boolean(result.lowProgressAudit),
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(result)
    });
    const previousStable = current.lastResult;
    const chosen = compareAnalysisResults(previousStable, cumulative) || cumulative;
    // A partial retry may not outrank a completed shallower result under the
    // durable quality policy. Keep that stable result for caching/resume, but
    // still publish the newer node count and current PVs to the 500 ms UI.
    // Otherwise NODES could visibly jump backwards after an in-depth snapshot.
    const publishLiveProgress = chosen === previousStable
      && Number(cumulative.nodes || 0) > Number(previousStable?.nodes || 0);
    const visibleResult = publishLiveProgress
      ? {
          ...chosen,
          depth: Math.max(Number(chosen?.depth || 0), Number(cumulative.depth || 0)),
          selDepth: Math.max(Number(chosen?.selDepth || 0), Number(cumulative.selDepth || 0)),
          nodes: cumulative.nodes,
          elapsed: cumulative.elapsed,
          nps: cumulative.nps,
          hashfull: cumulative.hashfull,
          searchDepth: cumulative.searchDepth,
          nextDepth: cumulative.nextDepth,
          nodeTarget: cumulative.nodeTarget,
          rootTurn: cumulative.rootTurn,
          lowProgressAudit: cumulative.lowProgressAudit,
          lines: cumulative.lines?.length ? cumulative.lines : chosen.lines,
          completed: false,
          liveUpdate: true,
          liveProgress: true,
          cached: false,
          solved: false
        }
      : chosen;
    current.lastResult = chosen;
    cacheResult(current.cacheKey, chosen);
    if (shouldDeferTablebasePseudoDisplay(visibleResult)) {
      current.deferredTablebaseResult = visibleResult;
    } else {
      post('info', { token, result: visibleResult });
    }
    requestDtmAnnotation(token, chosen);

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
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
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
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    current = null;
    post('state', { token: activeToken, state: 'idle', engine: ENGINE_VERSION });
    return;
  }
  if (message.type === 'clear') {
    searcher.clear();
    fairyProvider.stop();
    positionCache.clear();
    nextDepth = 1;
    currentBudgetMs = initialBudget(1);
    post('state', { token: activeToken, state: paused ? 'paused' : running ? 'thinking' : 'idle', engine: ENGINE_VERSION });
  }
});

post('ready', { engine: ENGINE_VERSION });
