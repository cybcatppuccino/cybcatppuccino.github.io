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
import { MinifishSearcher, MINIFISH_VERSION } from './minifish.js';
import { GardnerTablebase } from './tablebase.js';
import {
  isSolvedResult,
  isTrustedExactTablebaseResult,
  resultPvProfile,
  withResultQuality
} from './result-quality.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';
import {
  buildMoveStyleProfile,
  selectLineForStyle,
  styleConfig
} from './difficulty.js';

const { makeMove, undoMove, isCapture, isPromotion, givesCheck } = EngineInternals;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const minifish = new MinifishSearcher();
const responseSearcher = new GardnerSearcher({ hashEntries: 262144 });
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
searcher.setTablebaseProbe(position => tablebase.probeSync(position));
minifish.setTablebaseProbe(position => tablebase.probeSync(position));
responseSearcher.setTablebaseProbe(position => tablebase.probeSync(position));
// This worker is created only for AI modes by app.js; initialise GTB only then.
tablebase.init().catch(() => {});

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

function isStablePlayResult(result) {
  if (!result?.lines?.length) return false;
  if (isSolvedResult(result)) return true;
  return Boolean(result.completed !== false && result.pvComplete !== false && !result.pvIncomplete && result.multiPvVerified !== false);
}

function trustedResume(position, result, multipv) {
  if (!result?.lines?.length || result.engine !== ENGINE_VERSION || !isSolvedResult(result)) return null;
  const candidate = sortResultLinesForSide(result, position.turn, multipv);
  if (isTrustedExactTablebaseResult(candidate)) return candidate;
  const first = candidate.lines[0] || {};
  const proofBackedMate = Boolean(first.mateVerified && (first.mateProof || first.endgameProof || candidate.mateProof || candidate.endgameProof));
  if (proofBackedMate && validateMateResult(position, first)) return candidate;
  if (candidate.fortressProof || candidate.endgameProof || candidate.terminal) return candidate;
  return null;
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

function relabelTablebaseResult(result, engineLabel) {
  if (!result) return result;
  return withResultQuality({
    ...result,
    engine: engineLabel,
    engineLabel: `${engineLabel} + GTB`,
    lines: Array.isArray(result.lines) ? result.lines.map(line => ({
      ...line,
      sourceEngine: result.engine || line.sourceEngine || '',
      scoreKind: line.scoreKind || 'exact-tablebase',
      tablebase: true,
      tablebaseScope: line.tablebaseScope || 'root-exact'
    })) : []
  });
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
  let result = state.lastResult || state.lastRawResult || fallbackResult(state.position, {});
  result = await normalizePlayResult(state.position, result, state.config, token, { final: true });
  if (token !== activeToken || !currentPlay) return;
  result = sortResultLinesForSide(result, state.position.turn, state.config.multipv);
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
      cached: false,
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
      cached: false,
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
      endgameProbeMs: Math.min(90, state.config.endgameProbeMs || 0),
      fortressProbeMs: Math.min(120, state.config.fortressProbeMs || 0),
      mateProbeMs: mateBudget,
      mateMaxPlies: state.config.timeMs >= 10000 ? 81 : state.config.timeMs >= 5000 ? 65 : 45
    });
    state.firstChunk = false;
    if (!running || paused || token !== activeToken) return;
    result = await normalizePlayResult(state.position, result, state.config, token, { final: false });
    if (!running || paused || token !== activeToken) return;
    const elapsed = Math.max(1, Math.round(performance.now() - started));
    state.activeElapsed += elapsed;
    state.totalNodes += Number(result.nodes || 0);
    state.totalElapsed += Number(result.elapsed || elapsed);
    if (result.completed) state.nextDepth = Math.max(1, Number(result.nextDepth || requestedDepth + 1));
    else state.nextDepth = requestedDepth;
    const profile = resultPvProfile(result);
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
    state.lastRawResult = cumulative;
    let visible = null;
    if (isStablePlayResult(cumulative)) {
      state.lastResult = cumulative;
      visible = cumulative;
    } else if (state.lastResult?.lines?.length) {
      // Mirror the analysis worker policy: unfinished chunks advance timing
      // information only and never replace a completed score/PV pair.
      visible = {
        ...state.lastResult,
        nodes: cumulative.nodes,
        elapsed: cumulative.elapsed,
        nps: cumulative.nps,
        selDepth: Math.max(Number(state.lastResult.selDepth || 0), Number(cumulative.selDepth || 0)),
        searchDepth: state.nextDepth,
        nextDepth: state.nextDepth,
        activeElapsed: state.activeElapsed,
        timeRemaining: Math.max(0, state.config.timeMs - state.activeElapsed),
        liveProgress: true,
        liveUpdate: true,
        cached: false
      };
    }
    if (visible) post('info', { token, result: visible });
    post('state', {
      token,
      state: 'thinking',
      engine: ENGINE_VERSION,
      style: state.config.id,
      depth: Number(state.lastResult?.depth || 0),
      searchDepth: state.nextDepth,
      activeElapsed: state.activeElapsed,
      timeRemaining: Math.max(0, state.config.timeMs - state.activeElapsed)
    });
    const solved = isSolvedResult(state.lastResult) || state.lastResult?.terminal || state.lastResult?.fortressProof || state.nextDepth > state.config.maxDepth;
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

async function handleMinifishSearch(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  clearTimeout(scheduled);
  scheduled = 0;
  currentPlay = null;
  running = false;
  paused = false;
  try {
    const baseConfig = styleConfig(message.style || 'balanced');
    const requestedTime = Number(message.timeMs || message.thinkTimeMs || baseConfig.timeMs);
    const config = {
      ...baseConfig,
      id: baseConfig.id,
      label: `Minifish · ${baseConfig.label}`,
      shortLabel: 'Minifish',
      timeMs: [1000, 2000, 3000, 5000, 10000, 20000, 30000].includes(requestedTime)
        ? requestedTime
        : Math.max(500, Math.min(30000, requestedTime || baseConfig.timeMs))
    };
    const position = EnginePosition.fromFEN(message.fen);
    const cacheKey = String(message.cacheKey || position.key());
    const rootLegalMoves = generateLegalMoves(position, false);
    if (rootLegalMoves.length <= 1) {
      config.timeMs = Math.min(config.timeMs, 180);
      config.maxDepth = Math.min(config.maxDepth, 8);
      config.multipv = 1;
    }
    const historyKeys = (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen);
    post('state', { token, state: 'thinking', engine: MINIFISH_VERSION, style: config.id, timeRemaining: config.timeMs });

    if (Number(position.pieceCount || 0) <= 5) {
      try {
        const tb = await tablebase.analyze(position.clone(), { multipv: Math.max(5, config.multipv) });
        if (token !== activeToken) return;
        if (tb) {
          let result = await normalizePlayResult(position, relabelTablebaseResult(tb, MINIFISH_VERSION), config, token, { final: true });
          if (token !== activeToken) return;
          result = sortResultLinesForSide(result, position.turn, config.multipv);
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
          post('state', { token, state: 'complete', engine: result.engineLabel || MINIFISH_VERSION, style: config.id, depth: result.depth || 0, searchDepth: 0, tablebase: true });
          return;
        }
      } catch {}
    }

    if (Number(position.pieceCount || 0) <= 6) {
      void tablebase.warmExactWdlNeighborhood(position.clone(), { includeLegalChildren: true }).catch(() => false);
    }

    const started = performance.now();
    const raw = minifish.analyze(position.clone(), {
      timeMs: config.timeMs,
      maxDepth: config.maxDepth,
      multipv: config.multipv,
      startDepth: 1,
      historyKeys,
      newPosition: true
    });
    if (token !== activeToken) return;
    const elapsed = Math.max(1, Math.round(performance.now() - started));
    let result = await normalizePlayResult(position, {
      ...raw,
      elapsed: Math.max(Number(raw.elapsed || 0), elapsed),
      cacheKey,
      cached: false,
      searchDepth: Number(raw.nextDepth || raw.searchDepth || 0),
      nextDepth: Number(raw.nextDepth || raw.searchDepth || 0),
      style: config.id,
      styleLabel: config.label,
      timeLimit: config.timeMs,
      maxDepth: config.maxDepth,
      aiInternal: true,
      minifish: true
    }, config, token, { final: true });
    if (token !== activeToken) return;
    result = sortResultLinesForSide(result, position.turn, config.multipv);
    const selected = selectLineForStyle(result.lines, config, position.turn === 1 ? 'w' : 'b');
    const selectedMove = selected?.move || result.lines?.[0]?.move || '';
    post('info', { token, result: { ...result, selectedMove, selectedLine: selected || result.lines?.[0] || null, completed: true } });
    post('result', {
      token,
      result: {
        ...result,
        selectedMove,
        selectedLine: selected || result.lines?.[0] || null,
        completed: true
      }
    });
    post('state', { token, state: 'complete', engine: MINIFISH_VERSION, style: config.id, depth: result.depth || 0, searchDepth: 0 });
  } catch (error) {
    if (token !== activeToken) return;
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
    const resumeResult = trustedResume(position, message.resumeResult, config.multipv);
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

    // Do not let broad tablebase warming compete with a direct exact root
    // solve. For a 6-piece root, also preload first-hit <=5-piece leaves so the
    // playing search can cut them off directly instead of re-searching them.
    void (async () => {
      await tablebase.warmExactWdlNeighborhood(position.clone(), { includeLegalChildren: true });
      await tablebase.warmExactFrontier(position.clone(), { maxPly: 4, maxStates: 240, priority: 1 });
    })().catch(() => false);
    if (token !== activeToken) return;

    currentPlay = {
      token,
      position,
      cacheKey,
      config,
      historyKeys,
      rootLegalMoves,
      firstChunk: true,
      nextDepth: 1,
      lastResult: resumeResult || null,
      lastRawResult: resumeResult || null,
      totalNodes: 0,
      totalElapsed: 0,
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
  const kernel = selectedKernel(message.kernel);
  if (kernel === ENGINE_KERNELS.FAIRY) return handleFairySearch(message);
  if (kernel === ENGINE_KERNELS.MINIFISH) return handleMinifishSearch(message);
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
