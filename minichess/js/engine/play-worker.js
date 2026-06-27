import {
  EngineInternals,
  EnginePosition,
  GardnerSearcher,
  ENGINE_VERSION,
  generateLegalMoves,
  moveToUci,
  staticExchangeEval,
  uciToMove,
  validateMateResult
} from './engine.js';
import { GardnerTablebase } from './tablebase.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';
import {
  buildMoveStyleProfile,
  selectLineForStyle,
  styleConfig
} from './difficulty.js';

const { makeMove, undoMove, isCapture, isPromotion, givesCheck } = EngineInternals;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const responseSearcher = new GardnerSearcher({ hashEntries: 262144 });
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
searcher.setTablebaseProbe(position => tablebase.probeWdlSync(position));
responseSearcher.setTablebaseProbe(position => tablebase.probeWdlSync(position));
tablebase.init().catch(() => {});

const resultCache = new Map();
const CACHE_LIMIT = 576;
let activeToken = 0;
let currentPlay = null;
let running = false;
let paused = false;
let scheduled = 0;

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

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
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

function isSolvedResult(result) {
  return Boolean(result?.tablebase || result?.fortressProof || result?.endgameProof || result?.lines?.[0]?.mateVerified || (result?.solved && result?.lines?.[0]?.mateVerified));
}

function pvProfile(result) {
  if (!result || isSolvedResult(result) || result.tablebase || result.fortressProof || result.endgameProof) {
    return { pvDepth: Array.isArray(result?.lines?.[0]?.pv) ? result.lines[0].pv.length : 0, pvTarget: 0, pvComplete: true };
  }
  const depth = Number(result.scoreDepth || result.depth || 0);
  const pvDepth = Array.isArray(result.lines?.[0]?.pv) ? result.lines[0].pv.length : 0;
  const pvTarget = depth >= 8 ? Math.min(12, Math.max(6, depth - 2)) : 0;
  return { pvDepth, pvTarget, pvComplete: !pvTarget || pvDepth >= pvTarget };
}

function isThinPvResume(result) {
  if (!result || isSolvedResult(result)) return false;
  const profile = pvProfile(result);
  return !profile.pvComplete && profile.pvDepth > 0;
}


function shouldCacheWorkerResult(result) {
  if (!result?.lines?.length) return false;
  if (isSolvedResult(result) || result.terminal || result.tablebase || result.fortressProof || result.endgameProof) return true;
  const profile = pvProfile(result);
  return profile.pvComplete && result.completed !== false;
}

function preserveBestPv(previousLine, nextLine) {
  const previousPv = Array.isArray(previousLine?.pv) ? previousLine.pv : [];
  const nextPv = Array.isArray(nextLine?.pv) ? nextLine.pv : [];
  if (!previousPv.length || nextLine?.mateVerified || nextLine?.tablebase || nextLine?.fortressProof) return nextLine;
  if (nextPv.length >= previousPv.length) return nextLine;
  return { ...nextLine, pv: previousPv.slice(), pvPreservedFromCache: true };
}

function lineUtility(line, side) {
  return (side === 1 ? 1 : -1) * Number(line?.score || 0);
}

function sortResultLinesForSide(result, sideToMove, limit = 0) {
  if (!result || !Array.isArray(result.lines)) return result;
  const maxLines = Math.max(0, Number(limit || 0));
  const lines = result.lines
    .map(line => ({ ...line, pv: Array.isArray(line?.pv) ? line.pv.slice() : [] }))
    .sort((a, b) => lineUtility(b, sideToMove) - lineUtility(a, sideToMove));
  if (maxLines > 0 && lines.length > maxLines) lines.length = maxLines;
  return { ...result, lines };
}

function cacheResult(key, result) {
  if (!key || !result?.lines?.length || !shouldCacheWorkerResult(result)) return;
  const previous = resultCache.get(key);
  const previousSolved = isSolvedResult(previous?.result);
  const nextSolved = isSolvedResult(result);
  const previousPv = pvProfile(previous?.result);
  const nextPv = pvProfile(result);
  const previousDepth = Number(previous?.result?.scoreDepth || previous?.result?.depth || 0);
  const nextDepth = Number(result.scoreDepth || result.depth || 0);
  const nextResult = { ...result, ...nextPv };
  if (previousSolved && !nextSolved) {
    if (previous) previous.updatedAt = Date.now();
  } else if (previous?.result && previousPv.pvComplete && !nextPv.pvComplete && !nextSolved) {
    previous.updatedAt = Date.now();
  } else if (!previous || nextSolved || (nextDepth >= previousDepth && (nextPv.pvComplete || !previousPv.pvComplete))) {
    resultCache.set(key, { updatedAt: Date.now(), result: nextResult });
  } else if (previous) {
    previous.updatedAt = Date.now();
  }
  if (resultCache.size > CACHE_LIMIT) {
    let oldestKey = '';
    let oldestTime = Infinity;
    for (const [entryKey, entry] of resultCache) {
      if (entry.updatedAt < oldestTime) {
        oldestKey = entryKey;
        oldestTime = entry.updatedAt;
      }
    }
    if (oldestKey) resultCache.delete(oldestKey);
  }
}

function bestResume(message, key) {
  const internalCandidate = resultCache.get(key)?.result || null;
  const externalCandidate = message.resumeResult || null;
  const internal = internalCandidate?.engine === ENGINE_VERSION ? internalCandidate : null;
  const external = externalCandidate?.engine === ENGINE_VERSION ? externalCandidate : null;
  if (!internal) return external;
  if (!external) return internal;
  const internalSolved = isSolvedResult(internal);
  const externalSolved = isSolvedResult(external);
  if (internalSolved !== externalSolved) return internalSolved ? internal : external;
  const internalPv = pvProfile(internal);
  const externalPv = pvProfile(external);
  if (internalPv.pvComplete !== externalPv.pvComplete) return internalPv.pvComplete ? internal : external;
  return Number(internal.scoreDepth || internal.depth || 0) >= Number(external.scoreDepth || external.depth || 0) ? internal : external;
}

function responseProfile(root, rootLine, timeMs) {
  const move = uciToMove(root, rootLine.move);
  if (!move) return null;
  const state = makeMove(root, move);
  try {
    const childSide = root.turn;
    const response = responseSearcher.analyze(root, {
      timeMs: Math.max(100, Number(timeMs || 120)),
      maxDepth: 8,
      multipv: 5,
      startDepth: 1,
      newPosition: true,
      endgameProbeMs: 0,
      fortressProbeMs: 0,
      mateProbeMs: Math.max(35, Math.floor(Number(timeMs || 120) * 0.22)),
      mateMaxPlies: 21
    });
    const ranked = sortResultLinesForSide(response, childSide, 5).lines
      .map(line => ({ line, utility: lineUtility(line, childSide) }));
    if (!ranked.length) return null;
    const best = ranked[0];
    const second = ranked[1] || best;
    const bestMove = uciToMove(root, best.line.move);
    let bestReplyForcing = false;
    let bestReplyQuiet = false;
    if (bestMove) {
      bestReplyForcing = isCapture(root, bestMove) || isPromotion(bestMove) || givesCheck(root, bestMove);
      bestReplyQuiet = !bestReplyForcing && staticExchangeEval(root, bestMove) >= -20;
    }
    return {
      depth: response.depth || 0,
      gapCp: Math.max(0, best.utility - second.utility),
      goodReplyCount: ranked.filter(item => best.utility - item.utility <= 38).length,
      replyCount: ranked.length,
      bestReply: best.line.move,
      bestReplyForcing,
      bestReplyQuiet
    };
  } finally {
    undoMove(root, move, state);
  }
}

async function decorateStyleProfiles(position, result, config, token, { final = false } = {}) {
  if (!result?.lines?.length) return result;
  const lines = result.lines.map(line => ({ ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] }));
  if (final && config.id === 'cunning' && !result.tablebase) {
    for (const line of lines.slice(0, 5)) {
      if (token !== activeToken) return result;
      line.replyProfile = responseProfile(position, line, config.responseProbeMs);
    }
  }
  for (const line of lines) line.styleProfile = buildMoveStyleProfile(position, line);
  return { ...result, lines };
}

function fallbackResult(position, result = {}) {
  const fallbackMove = generateLegalMoves(position, false)[0] || 0;
  if (!fallbackMove) return result;
  const uci = moveToUci(fallbackMove);
  return {
    ...result,
    depth: Math.max(0, Number(result?.depth || 0)),
    lines: [{
      move: uci,
      score: 0,
      scoreText: '0.00',
      pv: [uci],
      fallback: true,
      mateVerified: false,
      endgameProof: false,
      dtm: 0
    }],
    completed: false,
    fallback: true
  };
}

async function normalizePlayResult(position, result, config, token, { final = false } = {}) {
  let normalized = result;
  if (!normalized?.terminal && !normalized?.lines?.length) normalized = fallbackResult(position, normalized || {});
  if (normalized && !normalized.tablebase && !normalized.fortressProof) {
    normalized = await tablebase.annotateResultWithDtmBounds(position.clone(), normalized, {
      maxLines: config.multipv,
      maxProbePly: 24
    });
  }
  if (token !== activeToken) return normalized;
  normalized = sortResultLinesForSide(normalized, position.turn, config.multipv);
  normalized = await decorateStyleProfiles(position, normalized, config, token, { final });
  if (token !== activeToken) return normalized;
  return sortResultLinesForSide(normalized, position.turn, config.multipv);
}

function playChunkBudget(depth, remainingMs) {
  const base = depth <= 1 ? 90 : depth === 2 ? 130 : depth === 3 ? 180 : 240 + Math.min(360, (depth - 4) * 60);
  return Math.max(60, Math.min(remainingMs, base));
}

function schedulePlay(token, delay = 0) {
  clearTimeout(scheduled);
  scheduled = setTimeout(() => void runPlayChunk(token), delay);
}

async function finishPlay(token) {
  if (!currentPlay || token !== activeToken) return;
  running = false;
  paused = false;
  const state = currentPlay;
  let result = state.lastResult || fallbackResult(state.position, {});
  result = await normalizePlayResult(state.position, result, state.config, token, { final: true });
  if (token !== activeToken || !currentPlay) return;
  result = sortResultLinesForSide(result, state.position.turn, state.config.multipv);
  cacheResult(state.cacheKey, result);
  const selected = selectLineForStyle(result.lines, state.config, state.position.turn === 1 ? 'w' : 'b');
  const selectedMove = selected?.move || result.lines?.[0]?.move || '';
  post('info', {
    token,
    result: {
      ...result,
      cacheKey: state.cacheKey,
      style: state.config.id,
      styleLabel: state.config.label,
      timeLimit: state.config.timeMs,
      maxDepth: state.config.maxDepth,
      selectedMove,
      selectedLine: selected || result.lines?.[0] || null,
      cached: Boolean(state.usedResume || result.cached),
      completed: true
    }
  });
  post('result', {
    token,
    result: {
      ...result,
      cacheKey: state.cacheKey,
      selectedMove,
      selectedLine: selected || result.lines?.[0] || null,
      cached: Boolean(state.usedResume || result.cached),
      style: state.config.id,
      styleLabel: state.config.label,
      timeLimit: state.config.timeMs,
      maxDepth: state.config.maxDepth
    }
  });
  post('state', { token, state: 'complete', engine: ENGINE_VERSION, style: state.config.id, depth: result.depth || 0, searchDepth: 0 });
  currentPlay = null;
}

async function runPlayChunk(token) {
  if (!running || paused || token !== activeToken || !currentPlay) return;
  const state = currentPlay;
  try {
    const remaining = Math.max(0, state.config.timeMs - state.activeElapsed);
    if (remaining <= 0 && state.lastResult?.lines?.length) return finishPlay(token);
    const requestedDepth = state.nextDepth;
    const chunkMs = playChunkBudget(requestedDepth, Math.max(60, remaining || 60));
    const mateBudget = state.rootLegalMoves.length <= 1 ? 0 : Math.min(1200, Math.max(45, Math.round(chunkMs * 0.28)));
    const mainBudget = Math.max(50, chunkMs - mateBudget);
    const started = performance.now();
    let result = searcher.analyze(state.position.clone(), {
      timeMs: mainBudget,
      maxDepth: requestedDepth,
      multipv: state.config.multipv,
      startDepth: requestedDepth,
      historyKeys: state.historyKeys,
      newPosition: state.firstChunk,
      resumeResult: state.firstChunk ? state.resumeResult : null,
      endgameProbeMs: Math.min(90, state.config.endgameProbeMs || 0),
      fortressProbeMs: Math.min(120, state.config.fortressProbeMs || 0),
      mateProbeMs: mateBudget,
      mateMaxPlies: state.config.timeMs >= 10000 ? 81 : state.config.timeMs >= 5000 ? 65 : 45
    });
    state.firstChunk = false;
    state.resumeResult = null;
    if (!running || paused || token !== activeToken) return;
    result = await normalizePlayResult(state.position, result, state.config, token, { final: false });
    if (!running || paused || token !== activeToken) return;
    const elapsed = Math.max(1, Math.round(performance.now() - started));
    state.activeElapsed += elapsed;
    state.totalNodes += Number(result.nodes || 0);
    state.totalElapsed += Number(result.elapsed || elapsed);
    if (result.completed) state.nextDepth = Math.max(1, Number(result.nextDepth || requestedDepth + 1));
    else state.nextDepth = requestedDepth;
    const profile = pvProfile(result);
    const cumulative = {
      ...result,
      ...profile,
      nodes: state.totalNodes,
      elapsed: state.totalElapsed,
      nps: Math.round(state.totalNodes * 1000 / Math.max(1, state.totalElapsed)),
      scoreDepth: Number(result.scoreDepth || result.depth || 0),
      cacheKey: state.cacheKey,
      cached: false,
      searchDepth: state.nextDepth,
      nextDepth: state.nextDepth,
      style: state.config.id,
      styleLabel: state.config.label,
      timeLimit: state.config.timeMs,
      maxDepth: state.config.maxDepth,
      aiInternal: true,
      activeElapsed: state.activeElapsed,
      timeRemaining: Math.max(0, state.config.timeMs - state.activeElapsed)
    };
    state.lastResult = cumulative;
    cacheResult(state.cacheKey, cumulative);
    post('info', { token, result: cumulative });
    post('state', {
      token,
      state: 'thinking',
      engine: ENGINE_VERSION,
      style: state.config.id,
      depth: result.depth || 0,
      searchDepth: state.nextDepth,
      activeElapsed: state.activeElapsed,
      timeRemaining: Math.max(0, state.config.timeMs - state.activeElapsed)
    });
    const solved = isSolvedResult(cumulative) || cumulative.terminal || cumulative.fortressProof || state.nextDepth > state.config.maxDepth;
    if (solved || state.activeElapsed >= state.config.timeMs) return finishPlay(token);
    // Leave a small event-loop window so the UI pause button can interrupt between
    // iterative chunks and pause the AI clock before the next depth starts.
    schedulePlay(token, 24);
  } catch (error) {
    running = false;
    paused = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

async function handleFairySearch(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentPlay = null;
  running = false;
  paused = false;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  const baseConfig = styleConfig(message.style || 'balanced');
  const requestedTime = Number(message.timeMs || message.thinkTimeMs || baseConfig.timeMs);
  const timeMs = [1000, 2000, 3000, 5000, 10000, 20000, 30000].includes(requestedTime)
    ? requestedTime
    : Math.max(500, Math.min(30000, requestedTime || baseConfig.timeMs));
  post('state', { token, state: 'thinking', engine: FAIRY_STOCKFISH_LABEL, style: 'stockfish' });
  try {
    const raw = await fairyProvider.search({
      token,
      fen: String(message.fen || '').trim(),
      timeMs,
      multipv: Math.max(1, Math.min(5, Number(message.multipv || baseConfig.multipv || 3)))
    });
    if (token !== activeToken) return;
    const result = validateExternalAnalysisResult(position, raw, { maxLines: Math.max(1, Math.min(5, Number(message.multipv || baseConfig.multipv || 3))) });
    if (!result) throw new Error('Fairy-Stockfish returned no fully legal Gardner move.');
    const ordered = sortResultLinesForSide(result, position.turn);
    const selected = ordered.lines[0] || null;
    post('info', { token, result: { ...ordered, cacheKey, style: 'stockfish', styleLabel: 'Fairy-Stockfish', timeLimit: timeMs, maxDepth: ordered.depth || 0 } });
    post('result', {
      token,
      result: {
        ...ordered,
        cacheKey,
        selectedMove: selected?.move || '',
        selectedLine: selected,
        style: 'stockfish',
        styleLabel: 'Fairy-Stockfish',
        timeLimit: timeMs,
        maxDepth: ordered.depth || 0
      }
    });
  } catch (error) {
    if (token !== activeToken) return;
    await handleOrionSearch({ ...message, kernel: ENGINE_KERNELS.ORION });
  }
}

async function handleOrionSearch(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  clearTimeout(scheduled);
  scheduled = 0;
  paused = false;
  running = false;
  try {
    const baseConfig = styleConfig(message.style || 'balanced');
    const requestedTime = Number(message.timeMs || message.thinkTimeMs || baseConfig.timeMs);
    const config = {
      ...baseConfig,
      timeMs: [1000, 2000, 3000, 5000, 10000, 20000, 30000].includes(requestedTime)
        ? requestedTime
        : Math.max(500, Math.min(30000, requestedTime || baseConfig.timeMs))
    };
    const position = EnginePosition.fromFEN(message.fen);
    const rootLegalMoves = generateLegalMoves(position, false);
    if (rootLegalMoves.length <= 1) {
      config.timeMs = Math.min(config.timeMs, 180);
      config.maxDepth = Math.min(config.maxDepth, 8);
      config.multipv = 1;
      config.endgameProbeMs = 0;
      config.fortressProbeMs = 0;
    }
    const cacheKey = String(message.cacheKey || position.key());
    let resumeResult = bestResume(message, cacheKey);
    if (resumeResult?.lines?.[0]?.mateVerified && !validateMateResult(position, resumeResult.lines[0])) {
      resumeResult = null;
      resultCache.delete(cacheKey);
    }
    if (isThinPvResume(resumeResult)) {
      resumeResult = null;
      resultCache.delete(cacheKey);
    }
    if (resumeResult?.lines?.length) resumeResult = sortResultLinesForSide(resumeResult, position.turn, config.multipv);
    const historyKeys = (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen);
    post('state', {
      token,
      state: 'thinking',
      engine: ENGINE_VERSION,
      style: config.id,
      resumedDepth: Number(resumeResult?.depth || 0),
      timeRemaining: config.timeMs
    });

    let tbResult = null;
    try {
      tbResult = await tablebase.analyze(position.clone(), { multipv: Math.max(5, config.multipv) });
      if (token !== activeToken) return;
    } catch {
      tbResult = null;
    }
    if (tbResult) {
      let result = await normalizePlayResult(position, tbResult, config, token, { final: true });
      if (token !== activeToken) return;
      result = sortResultLinesForSide(result, position.turn, config.multipv);
      cacheResult(cacheKey, result);
      const selected = selectLineForStyle(result.lines, config, position.turn === 1 ? 'w' : 'b');
      post('info', { token, result: { ...result, cacheKey, style: config.id, styleLabel: config.label, timeLimit: config.timeMs, maxDepth: config.maxDepth } });
      post('result', {
        token,
        result: {
          ...result,
          cacheKey,
          selectedMove: selected?.move || result.lines?.[0]?.move || '',
          selectedLine: selected || result.lines?.[0] || null,
          style: config.id,
          styleLabel: config.label,
          timeLimit: config.timeMs,
          maxDepth: config.maxDepth
        }
      });
      post('state', { token, state: 'complete', engine: result.engineLabel || ENGINE_VERSION, style: config.id, depth: result.depth || 0, searchDepth: 0, tablebase: true });
      return;
    }

    // Do not let broad WDL neighborhood warming compete with a direct exact
    // tablebase solve. Start it only after the current position was not solved
    // by tablebase.
    void tablebase.warmExactWdlNeighborhood(position.clone(), { includeLegalChildren: true }).catch(() => false);
    if (token !== activeToken) return;

    currentPlay = {
      token,
      position,
      cacheKey,
      config,
      historyKeys,
      rootLegalMoves,
      resumeResult,
      firstChunk: true,
      nextDepth: Math.max(1, Math.min(config.maxDepth, Number(resumeResult?.depth || 0) + 1)),
      lastResult: resumeResult || null,
      totalNodes: Math.max(0, Number(resumeResult?.nodes || 0)),
      totalElapsed: Math.max(0, Number(resumeResult?.elapsed || 0)),
      activeElapsed: 0,
      usedResume: Boolean(resumeResult)
    };
    if (resumeResult?.lines?.length) {
      post('info', {
        token,
        result: {
          ...resumeResult,
          cacheKey,
          cached: true,
          searchDepth: currentPlay.nextDepth,
          nextDepth: currentPlay.nextDepth,
          style: config.id,
          styleLabel: config.label,
          timeLimit: config.timeMs,
          maxDepth: config.maxDepth,
          aiInternal: true
        }
      });
    }
    if (isSolvedResult(resumeResult)) return finishPlay(token);
    running = true;
    schedulePlay(token, 0);
  } catch (error) {
    running = false;
    paused = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

async function handleSearch(message) {
  if (selectedKernel(message.kernel) === ENGINE_KERNELS.FAIRY) return handleFairySearch(message);
  return handleOrionSearch(message);
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'cancel') {
    activeToken = Number(message.token || activeToken + 1);
    clearTimeout(scheduled);
    scheduled = 0;
    running = false;
    paused = false;
    currentPlay = null;
    fairyProvider.stop();
    return;
  }
  if (message.type === 'pause') {
    if (Number(message.token) !== activeToken || !currentPlay) return;
    paused = true;
    running = false;
    clearTimeout(scheduled);
    scheduled = 0;
    post('state', {
      token: activeToken,
      state: 'paused',
      engine: ENGINE_VERSION,
      style: currentPlay.config.id,
      depth: Number(currentPlay.lastResult?.depth || 0),
      searchDepth: currentPlay.nextDepth,
      activeElapsed: currentPlay.activeElapsed,
      timeRemaining: Math.max(0, currentPlay.config.timeMs - currentPlay.activeElapsed)
    });
    return;
  }
  if (message.type === 'resume') {
    if (Number(message.token) !== activeToken || !currentPlay) return;
    if (isSolvedResult(currentPlay.lastResult) || currentPlay.activeElapsed >= currentPlay.config.timeMs) {
      void finishPlay(activeToken);
      return;
    }
    paused = false;
    running = true;
    post('state', {
      token: activeToken,
      state: 'thinking',
      engine: ENGINE_VERSION,
      style: currentPlay.config.id,
      depth: Number(currentPlay.lastResult?.depth || 0),
      searchDepth: currentPlay.nextDepth,
      activeElapsed: currentPlay.activeElapsed,
      timeRemaining: Math.max(0, currentPlay.config.timeMs - currentPlay.activeElapsed)
    });
    schedulePlay(activeToken, 0);
    return;
  }
  if (message.type === 'search') void handleSearch(message);
});

post('ready', { engine: ENGINE_VERSION });
