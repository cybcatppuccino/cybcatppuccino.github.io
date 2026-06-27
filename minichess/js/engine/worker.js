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
  if (line.mateVerified) return 80;
  if (line.tablebase && !line.tablebaseBound && !line.dtmUpperBound) return 70;
  if (line.tablebaseBound || line.dtmUpperBound) return 45;
  if (line.endgameProof || line.fortressProof) return 40;
  return 10;
}

function mergeStableLine(previousLine, nextLine) {
  if (!previousLine) return nextLine;
  if (!nextLine) return previousLine;
  const previousPv = Array.isArray(previousLine?.pv) ? previousLine.pv : [];
  const nextPv = Array.isArray(nextLine?.pv) ? nextLine.pv : [];
  const previousRank = stableLineRank(previousLine);
  const nextRank = stableLineRank(nextLine);
  if (previousRank > nextRank) {
    // v18.2: a verified mate/tablebase bound is stronger than a later live
    // centipawn estimate for the same root move.  Keep the proven score and
    // badge stable, but allow a longer legal live PV to fill in notation.
    return {
      ...previousLine,
      pv: nextPv.length > previousPv.length ? nextPv.slice() : previousPv.slice(),
      pvPreservedFromCache: nextPv.length <= previousPv.length || Boolean(previousLine.pvPreservedFromCache),
      liveScoreSuppressed: true
    };
  }
  if (!previousPv.length || nextRank > previousRank || nextLine?.mateVerified || nextLine?.tablebase || nextLine?.fortressProof) return nextLine;
  if (nextPv.length >= previousPv.length) return nextLine;
  return { ...nextLine, pv: previousPv.slice(), pvPreservedFromCache: true };
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
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen)
  };
  nextDepth = resumeSearchDepth;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = Math.max(0, Number(resumeResult?.nodes || 0));
  totalElapsed = Math.max(0, Number(resumeResult?.elapsed || 0));
  paused = Boolean(message.startPaused);
  // v18: exact current-version <=5-piece tablebase resumes are trusted and do
  // not force a duplicate probe. Bound/WDL-only tablebase resumes are still
  // refreshed, while deep 6/7-piece live resumes continue from their next depth
  // instead of restarting lower-depth work.
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
  if (tablebaseEligible && !trustedTablebaseResume && await probeTablebase(token)) return;
  if (token !== activeToken || !running || paused) return;
  if (solvedResume) {
    running = false;
    post('state', { token, state: 'complete', engine: resumeResult?.engineLabel || ENGINE_VERSION, depth: Number(resumeResult?.depth || 0), searchDepth: 0 });
    return;
  }
  // Background warming is useful, but it should not compete with the exact
  // tablebase probe that can immediately solve the current position.
  void tablebase.warmExactWdlNeighborhood(position.clone(), { includeLegalChildren: true }).catch(() => false);
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

async function runChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
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
      mateMaxPlies: 81
    });
    firstChunk = false;
    current.resumeResult = null;
    if (!running || paused || token !== activeToken) return;
    result = mergeKnownAnalysisResult(current.lastResult, result, multipv, current.position.turn);

    totalNodes += result.nodes;
    totalElapsed += result.elapsed;
    if (result.completed) {
      nextDepth = result.nextDepth;
      currentBudgetMs = initialBudget(nextDepth);
    } else {
      currentBudgetMs = Math.min(Math.max(effortMs, 320), Math.round(currentBudgetMs * 1.5));
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
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(result)
    });
    const chosen = compareAnalysisResults(current.lastResult, cumulative);
    current.lastResult = chosen;
    cacheResult(current.cacheKey, chosen);
    post('info', { token, result: chosen });
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
