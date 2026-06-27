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
// v12: start manifest and WDL warming as soon as the worker is created.
// Search uses only already-warmed WDL blocks synchronously; misses safely fall
// back to the normal alpha-beta path.
tablebase.init().catch(() => {});
const resultCache = new Map();
const CACHE_LIMIT = 576;
let activeToken = 0;


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

function cacheResult(key, result) {
  if (!key || !result?.lines?.length) return;
  const previous = resultCache.get(key);
  if (!previous || Number(result.depth || 0) >= Number(previous.result?.depth || 0) || result.tablebase || result.lines[0]?.mateVerified) {
    resultCache.set(key, { updatedAt: Date.now(), result });
  } else {
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

function isSolvedResult(result) {
  return Boolean(result?.tablebase || result?.fortressProof || result?.endgameProof || result?.lines?.[0]?.mateVerified || (result?.solved && result?.lines?.[0]?.mateVerified));
}

function isThinPvResume(result) {
  if (!result || isSolvedResult(result)) return false;
  const depth = Number(result.depth || 0);
  const pvLength = Array.isArray(result.lines?.[0]?.pv) ? result.lines[0].pv.length : 0;
  return depth >= 10 && pvLength > 0 && pvLength < Math.min(10, Math.max(6, depth - 2));
}

function bestResume(message, key) {
  const internalCandidate = resultCache.get(key)?.result || null;
  const externalCandidate = message.resumeResult || null;
  const internal = internalCandidate?.engine === ENGINE_VERSION ? internalCandidate : null;
  const external = externalCandidate?.engine === ENGINE_VERSION ? externalCandidate : null;
  if (!internal) return external;
  if (!external) return internal;
  if (internal.lines?.[0]?.mateVerified || internal.tablebase) return internal;
  if (external.lines?.[0]?.mateVerified || external.tablebase) return external;
  return Number(internal.depth || 0) >= Number(external.depth || 0) ? internal : external;
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

function responseProfile(root, rootLine, timeMs) {
  const move = uciToMove(root, rootLine.move);
  if (!move) return null;
  const state = makeMove(root, move);
  try {
    const childSide = root.turn;
    const response = responseSearcher.analyze(root, {
      timeMs: Math.max(80, Number(timeMs || 100)),
      maxDepth: 7,
      multipv: 4,
      startDepth: 1,
      newPosition: true,
      endgameProbeMs: 0,
      fortressProbeMs: 0
    });
    const ranked = (response.lines || [])
      .map(line => ({ line, utility: lineUtility(line, childSide) }))
      .sort((a, b) => b.utility - a.utility);
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

async function decorateStyleProfiles(position, result, config, token) {
  if (!result?.lines?.length) return result;
  const lines = result.lines.map(line => ({ ...line }));
  if (config.id === 'cunning' && !result.tablebase) {
    for (const line of lines.slice(0, 5)) {
      if (token !== activeToken) return result;
      line.replyProfile = responseProfile(position, line, config.responseProbeMs);
    }
  }
  for (const line of lines) line.styleProfile = buildMoveStyleProfile(position, line);
  return { ...result, lines };
}


async function handleFairySearch(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
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
    // If the optional wasm provider cannot be used or emits an illegal PV,
    // fall back to Orion JS so AI play never receives an unchecked move.
    await handleOrionSearch({ ...message, kernel: ENGINE_KERNELS.ORION });
  }
}

async function handleSearch(message) {
  if (selectedKernel(message.kernel) === ENGINE_KERNELS.FAIRY) return handleFairySearch(message);
  return handleOrionSearch(message);
}
async function handleOrionSearch(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
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
      config.timeMs = Math.min(config.timeMs, 160);
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
      resumedDepth: Number(resumeResult?.depth || 0)
    });

    await tablebase.warmExactWdlNeighborhood(position.clone(), { includeLegalChildren: true }).catch(() => false);
    if (token !== activeToken) return;

    let result = null;
    try {
      result = await tablebase.analyze(position.clone(), { multipv: Math.max(5, config.multipv) });
      if (token !== activeToken) return;
    } catch {
      result = null;
    }

    if (!result) {
      const resumeDepth = Number(resumeResult?.depth || 0);
      const hasRequiredBreadth = Number(resumeResult?.lines?.length || 0) >= config.multipv;
      if (resumeResult?.lines?.length && (resumeResult.terminal || resumeResult.tablebase || resumeResult.fortressProof || resumeResult.lines[0]?.mateVerified || (resumeDepth >= config.maxDepth && hasRequiredBreadth))) {
        result = {
          ...resumeResult,
          engine: ENGINE_VERSION,
          cached: true,
          completed: true,
          searchDepth: resumeDepth + 1
        };
      } else {
        const startDepth = Math.max(1, resumeDepth >= config.maxDepth ? config.maxDepth : resumeDepth + 1);
        const mateBudget = rootLegalMoves.length <= 1
          ? 0
          : Math.min(9000, Math.max(120, Math.round(config.timeMs * 0.30)));
        const mainBudget = Math.max(80, config.timeMs - mateBudget);
        result = searcher.analyze(position, {
          timeMs: mainBudget,
          maxDepth: config.maxDepth,
          multipv: config.multipv,
          startDepth,
          historyKeys,
          newPosition: true,
          resumeResult,
          endgameProbeMs: config.endgameProbeMs,
          fortressProbeMs: config.fortressProbeMs,
          mateProbeMs: mateBudget,
          mateMaxPlies: config.timeMs >= 10000 ? 81 : config.timeMs >= 5000 ? 65 : 45
        });
        result.cached = Boolean(resumeResult);
      }
    }

    if (result && !result.tablebase && !result.fortressProof) {
      result = await tablebase.annotateResultWithDtmBounds(position.clone(), result, {
        maxLines: config.multipv,
        maxProbePly: 24
      });
      if (token !== activeToken) return;
    }

    if (!result?.terminal && !result?.lines?.length) {
      const fallbackMove = generateLegalMoves(position, false)[0] || 0;
      if (fallbackMove) {
        const uci = moveToUci(fallbackMove);
        result = {
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
    }

    result = await decorateStyleProfiles(position, result, config, token);
    if (token !== activeToken) return;
    result = sortResultLinesForSide(result, position.turn);
    cacheResult(cacheKey, result);
    const selected = selectLineForStyle(result.lines, config, position.turn === 1 ? 'w' : 'b');
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
  } catch (error) {
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'cancel') {
    activeToken = Number(message.token || activeToken + 1);
    fairyProvider.stop();
    return;
  }
  if (message.type === 'search') void handleSearch(message);
});

post('ready', { engine: ENGINE_VERSION });
