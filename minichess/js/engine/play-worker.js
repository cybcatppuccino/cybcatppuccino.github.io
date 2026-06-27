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
import {
  buildMoveStyleProfile,
  selectLineForStyle,
  styleConfig
} from './difficulty.js';

const { makeMove, undoMove, isCapture, isPromotion, givesCheck } = EngineInternals;
const searcher = new GardnerSearcher({ hashEntries: 524288 });
const responseSearcher = new GardnerSearcher({ hashEntries: 131072 });
const tablebase = new GardnerTablebase();
searcher.setTablebaseProbe(position => tablebase.probeWdlSync(position));
responseSearcher.setTablebaseProbe(position => tablebase.probeWdlSync(position));
// v12: start manifest and WDL warming as soon as the worker is created.
// Search uses only already-warmed WDL blocks synchronously; misses safely fall
// back to the normal alpha-beta path.
tablebase.init()
  .then(() => tablebase.warmExactWdl({ pieceLimit: 4 }))
  .catch(() => {});
const resultCache = new Map();
const CACHE_LIMIT = 192;
let activeToken = 0;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

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

async function handleSearch(message) {
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
    const historyKeys = (Array.isArray(message.historyFens) ? message.historyFens : []).map(fen => {
      const historyPosition = EnginePosition.fromFEN(fen);
      return { a: historyPosition.hashA, b: historyPosition.hashB };
    });
    post('state', {
      token,
      state: 'thinking',
      engine: ENGINE_VERSION,
      style: config.id,
      resumedDepth: Number(resumeResult?.depth || 0)
    });

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
    return;
  }
  if (message.type === 'search') void handleSearch(message);
});

post('ready', { engine: ENGINE_VERSION });
