import { EnginePosition, GardnerSearcher, ENGINE_VERSION } from './engine.js';
import { levelConfig } from './difficulty.js';

const searcher = new GardnerSearcher({ hashEntries: 524288 });
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
  const internal = resultCache.get(key)?.result || null;
  const external = message.resumeResult || null;
  if (!internal) return external;
  if (!external) return internal;
  return Number(internal.depth || 0) >= Number(external.depth || 0) ? internal : external;
}

function chooseLine(lines, config) {
  if (!lines?.length) return null;
  if (config.level >= 10 || lines.length === 1) return lines[0];
  const best = Number(lines[0].score || 0);
  const eligible = lines.filter((line, index) => {
    if (index === 0) return true;
    return Math.abs(Number(line.score || 0) - best) <= config.margin;
  });
  if (eligible.length === 1 || config.temperature <= 0) return eligible[0];
  const weights = eligible.map((_, index) => Math.exp(-index / Math.max(0.08, config.temperature)));
  const total = weights.reduce((sum, value) => sum + value, 0);
  let pick = Math.random() * total;
  for (let i = 0; i < eligible.length; i += 1) {
    pick -= weights[i];
    if (pick <= 0) return eligible[i];
  }
  return eligible[0];
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'cancel') {
    activeToken = Number(message.token || activeToken + 1);
    return;
  }
  if (message.type !== 'search') return;
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  try {
    const config = levelConfig(message.level);
    const position = EnginePosition.fromFEN(message.fen);
    const cacheKey = String(message.cacheKey || position.key());
    const resumeResult = bestResume(message, cacheKey);
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

    let result;
    const resumeDepth = Number(resumeResult?.depth || 0);
    const hasRequiredBreadth = Number(resumeResult?.lines?.length || 0) >= config.multipv;
    if (resumeResult?.lines?.length && (resumeResult.terminal || (resumeDepth >= config.maxDepth && hasRequiredBreadth))) {
      result = {
        ...resumeResult,
        engine: ENGINE_VERSION,
        cached: true,
        completed: true,
        searchDepth: resumeDepth + 1
      };
    } else {
      // A one-line descendant PV is excellent ordering information, but lower
      // levels still need their configured MultiPV breadth to preserve genuine
      // strength separation and controlled near-best exploration.
      const startDepth = Math.max(1, resumeDepth >= config.maxDepth ? config.maxDepth : resumeDepth + 1);
      result = searcher.analyze(position, {
        timeMs: config.timeMs,
        maxDepth: config.maxDepth,
        multipv: config.multipv,
        startDepth,
        historyKeys,
        newPosition: true,
        resumeResult
      });
      result.cached = Boolean(resumeResult);
    }
    cacheResult(cacheKey, result);
    const selected = chooseLine(result.lines, config);
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
});

post('ready', { engine: ENGINE_VERSION });
