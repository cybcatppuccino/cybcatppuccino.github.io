import { EnginePosition, GardnerSearcher, ENGINE_VERSION, generateLegalMoves, moveToUci, validateMateResult } from './engine.js';
import { GardnerTablebase } from './tablebase.js';
import { levelConfig, selectLineForLevel } from './difficulty.js';

const searcher = new GardnerSearcher({ hashEntries: 524288 });
const tablebase = new GardnerTablebase();
const resultCache = new Map();
const CACHE_LIMIT = 96;
let activeToken = 0;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function cacheResult(key, result) {
  if (!key || !result?.lines?.length) return;
  const previous = resultCache.get(key);
  if (!previous || Number(result.depth || 0) >= Number(previous.result?.depth || 0)) {
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
  return Number(internal.depth || 0) >= Number(external.depth || 0) ? internal : external;
}

function chooseLine(lines, config, sideToMove) {
  return selectLineForLevel(lines, config, sideToMove);
}


async function handleSearch(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  try {
    const config = levelConfig(message.level);
    const position = EnginePosition.fromFEN(message.fen);
    const cacheKey = String(message.cacheKey || position.key());
    let resumeResult = bestResume(message, cacheKey);
    if (config.level < 10) resumeResult = null;
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
      level: config.level,
      resumedDepth: Number(resumeResult?.depth || 0)
    });

    let result = null;
    if (config.level >= 8) {
      try {
        result = await tablebase.analyze(position.clone(), { multipv: Math.max(3, config.multipv) });
        if (token !== activeToken) return;
      } catch {
        result = null;
      }
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
        result = searcher.analyze(position, {
          timeMs: config.timeMs,
          maxDepth: config.maxDepth,
          multipv: config.multipv,
          startDepth,
          historyKeys,
          newPosition: true,
          resumeResult,
          endgameProbeMs: config.endgameProbeMs,
          fortressProbeMs: config.level >= 7 ? 100 : 35
        });
        result.cached = Boolean(resumeResult);
      }
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
    cacheResult(cacheKey, result);
    const selected = chooseLine(result.lines, config, position.turn === 1 ? 'w' : 'b');
    post('result', {
      token,
      result: {
        ...result,
        cacheKey,
        selectedMove: selected?.move || result.lines?.[0]?.move || '',
        selectedLine: selected || result.lines?.[0] || null,
        level: config.level,
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
