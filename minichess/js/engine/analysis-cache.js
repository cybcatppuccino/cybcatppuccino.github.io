import { ENGINE_VERSION, scoreToDisplay } from './engine.js';
import {
  compareAnalysisResults,
  lineHasCompletePv,
  resultPvProfile,
  shouldCacheResult,
  withResultQuality
} from './result-quality.js';

// v22.2 stores only direct exact-root tablebase answers and verified mates.
// Ordinary alpha-beta evaluations remain session-local and never resume a root search.
const STORAGE_KEY = 'gardner-analysis-cache-v22.2';
const MIGRATE_STORAGE_KEYS = Object.freeze([]);
const OLD_STORAGE_KEYS = Object.freeze([
  'gardner-analysis-cache-v21.2',
  'gardner-analysis-cache-v20.5',
  'gardner-analysis-cache-v20.3',
  'gardner-analysis-cache-v19.8',
  'gardner-analysis-cache-v19.4',
  'gardner-analysis-cache-v19.3',
  'gardner-analysis-cache-v19.2',
  'gardner-analysis-cache-v19.1',
  'gardner-analysis-cache-v19',
  'gardner-analysis-cache-v18.4',
  'gardner-analysis-cache-v18.3',
  'gardner-analysis-cache-v18.2',
  'gardner-analysis-cache-v18.1',
  'gardner-analysis-cache-v18',
  'gardner-analysis-cache-v17.4',
  'gardner-analysis-cache-v17.3',
  'gardner-analysis-cache-v17.2',
  'gardner-analysis-cache-v15_1',
  'gardner-analysis-cache-v15',
  'gardner-analysis-cache-v14_3',
  'gardner-analysis-cache-v14_2',
  'gardner-analysis-cache-v14_1',
  'gardner-analysis-cache-v14',
  'gardner-analysis-cache-v13',
  'gardner-analysis-cache-v12_2',
  'gardner-analysis-cache-v12_1'
]);
const CACHE_SCHEMA = 35;
const MAX_ENTRIES = 256;
const MAX_PV_PLIES = 28;
const COMPATIBLE_ORION_ENGINES = Object.freeze([ENGINE_VERSION]);

function isCompatibleOrionEngine(engine) {
  return COMPATIBLE_ORION_ENGINES.includes(String(engine || ''));
}

function cloneLine(line) {
  const score = Number(line?.score || 0);
  if (!Number.isFinite(score)) return null;
  return {
    move: String(line?.move || ''),
    score,
    scoreText: String(line?.scoreText || scoreToDisplay(score)),
    scoreKind: String(line?.scoreKind || ''),
    scoreNumeric: line?.scoreNumeric === false ? false : true,
    pv: Array.isArray(line?.pv) ? line.pv.slice(0, MAX_PV_PLIES).map(String) : [],
    mateVerified: Boolean(line?.mateVerified),
    tablebase: Boolean(line?.tablebase),
    tablebaseRoot: Boolean(line?.tablebaseRoot),
    tablebaseWdl: Number(line?.tablebaseWdl || 0),
    tablebaseExactDtm: Boolean(line?.tablebaseExactDtm),
    source: String(line?.source || ''),
    dtm: Math.max(0, Number(line?.dtm || 0)),
    pvComplete: Boolean(line?.pvComplete),
    rootScoreExact: line?.rootScoreExact !== false,
    resultContract: String(line?.resultContract || ''),
    resultKindV2: String(line?.resultKindV2 || '')
  };
}

function sanitizeResult(result) {
  if (!result || !isCompatibleOrionEngine(result.engine) || !Array.isArray(result.lines)) return null;
  const lines = result.lines.slice(0, 5).map(cloneLine).filter(Boolean);
  if (!lines.length) return null;
  const candidate = {
    ...result,
    lines,
    terminal: Boolean(result.terminal),
    tablebase: Boolean(result.tablebase),
    tablebaseRoot: Boolean(result.tablebaseRoot),
    tablebaseWdl: Number(result.tablebaseWdl || 0)
  };
  // This is the important gate: no ordinary analysis result survives it.
  if (!shouldCacheResult(candidate)) return null;
  const completeness = resultPvProfile(candidate, lines);
  for (const line of lines) line.pvComplete = lineHasCompletePv(line, candidate);
  return withResultQuality({
    schema: CACHE_SCHEMA,
    engine: ENGINE_VERSION,
    engineLabel: String(result.engineLabel || ENGINE_VERSION),
    depth: Math.max(0, Number(result.depth || 0)),
    scoreDepth: completeness.scoreDepth,
    pvDepth: completeness.pvDepth,
    pvTarget: completeness.pvTarget,
    pvComplete: true,
    selDepth: Math.max(0, Number(result.selDepth || 0)),
    nodes: Math.max(0, Number(result.nodes || 0)),
    nps: Math.max(0, Number(result.nps || 0)),
    elapsed: Math.max(0, Number(result.elapsed || 0)),
    hashfull: Math.max(0, Number(result.hashfull || 0)),
    rootTurn: Number(result.rootTurn) === -1 ? -1 : 1,
    searchDepth: Math.max(0, Number(result.searchDepth || result.depth || 0)),
    nextDepth: Math.max(0, Number(result.nextDepth || result.depth || 0)),
    attemptedDepth: Math.max(0, Number(result.attemptedDepth || result.depth || 0)),
    completed: true,
    terminal: Boolean(result.terminal),
    tablebase: Boolean(result.tablebase),
    tablebaseRoot: Boolean(result.tablebaseRoot),
    tablebaseSource: String(result.tablebaseSource || ''),
    tablebaseSignature: String(result.tablebaseSignature || ''),
    tablebaseWdl: Number(result.tablebaseWdl || 0),
    rejectedMateClaims: Math.max(0, Number(result.rejectedMateClaims || 0)),
    cached: true,
    solved: true,
    multiPvVerified: true,
    lines
  });
}

// Retained for callers that need to rebase a verified mate after consuming plies.
export function rebaseVerifiedMateLine(line, consumedPlies = 1) {
  const consumed = Math.max(0, Math.floor(Number(consumedPlies || 0)));
  const pv = Array.isArray(line?.pv) ? line.pv.slice(consumed) : [];
  if (!line?.mateVerified || !pv.length) return null;
  const original = Number(line.score || 0);
  if (!Number.isFinite(original) || Math.abs(original) < 29_000) return null;
  const score = original > 0 ? original + consumed : original - consumed;
  return {
    ...line,
    move: pv[0],
    score,
    scoreText: scoreToDisplay(score),
    pv,
    dtm: Math.max(1, Number(line.dtm || (pv.length + consumed)) - consumed),
    mateVerified: true,
  };
}

function compareCacheResults(previous, next, updatedAt = Date.now()) {
  return compareAnalysisResults(previous, next, { preferNextOnTie: updatedAt >= 0 });
}

function repetitionContextSignature(position, historyFens = []) {
  const reversiblePlies = Math.max(0, Number(position?.halfmove || 0));
  const relevant = reversiblePlies ? historyFens.slice(-reversiblePlies) : [];
  const counts = new Map();
  for (const fen of relevant) {
    const identity = String(fen).split(/\s+/).slice(0, 2).join(' ');
    if (!identity) continue;
    counts.set(identity, (counts.get(identity) || 0) + 1);
  }
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([identity, count]) => `${identity}x${count}`)
    .join('>');
}

export function buildAnalysisKey(position, historyFens = []) {
  const fenParts = position.toCompactFEN().split(/\s+/);
  const base = `${fenParts[0]} ${fenParts[1]} hm${fenParts[4] || 0}`;
  return `${base}|rep:${repetitionContextSignature(position, historyFens)}`;
}

export class AnalysisCache {
  constructor(storage = globalThis.localStorage) {
    this.storage = storage;
    this.entries = new Map();
    this.persistTimer = 0;
    this.persistIdleHandle = 0;
    this.dirty = false;
    this.deferPersistence = typeof window !== 'undefined' && storage === globalThis.localStorage;
    this.load();
  }

  readStorageKey(key) {
    if (!this.storage) return [];
    try {
      const payload = JSON.parse(this.storage.getItem(key) || '[]');
      return Array.isArray(payload) ? payload : [];
    } catch {
      return [];
    }
  }

  ingestPayload(payload) {
    for (const item of payload) {
      if (!item?.key || !item?.result) continue;
      const result = sanitizeResult(item.result);
      if (!result) continue;
      const key = String(item.key).replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
      const updatedAt = Number(item.updatedAt || 0);
      const previous = this.entries.get(key);
      const chosen = compareCacheResults(previous?.result || null, result, updatedAt);
      if (chosen === previous?.result) {
        if (previous) previous.updatedAt = Math.max(previous.updatedAt || 0, updatedAt || 0);
      } else if (chosen) {
        this.entries.set(key, { key, updatedAt, result: chosen });
      }
    }
  }

  load() {
    if (!this.storage) return;
    for (const key of OLD_STORAGE_KEYS) {
      try { this.storage.removeItem(key); } catch {}
    }
    this.ingestPayload(this.readStorageKey(STORAGE_KEY));
    for (const key of MIGRATE_STORAGE_KEYS) this.ingestPayload(this.readStorageKey(key));
    this.trim(false);
    this.dirty = true;
    this.persist();
  }

  cancelScheduledPersist() {
    if (this.persistTimer) {
      clearTimeout(this.persistTimer);
      this.persistTimer = 0;
    }
    if (this.persistIdleHandle && typeof globalThis.cancelIdleCallback === 'function') {
      globalThis.cancelIdleCallback(this.persistIdleHandle);
      this.persistIdleHandle = 0;
    }
  }

  persist({ force = false } = {}) {
    if (!this.storage || (!force && !this.dirty)) return;
    this.cancelScheduledPersist();
    try {
      const payload = [...this.entries.values()].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_ENTRIES);
      this.storage.setItem(STORAGE_KEY, JSON.stringify(payload));
      this.dirty = false;
    } catch {
      // In-memory exact-result caching remains available in restrictive browser contexts.
    }
  }

  schedulePersist() {
    this.dirty = true;
    if (!this.deferPersistence) {
      this.persist();
      return;
    }
    if (this.persistTimer || this.persistIdleHandle) return;
    const flush = () => {
      this.persistTimer = 0;
      this.persistIdleHandle = 0;
      this.persist();
    };
    if (typeof globalThis.requestIdleCallback === 'function') {
      this.persistIdleHandle = globalThis.requestIdleCallback(flush, { timeout: 1200 });
    } else {
      this.persistTimer = setTimeout(flush, 850);
    }
  }

  flush() { this.persist({ force: true }); }

  trim(persist = true) {
    if (this.entries.size > MAX_ENTRIES) {
      const ordered = [...this.entries.values()].sort((a, b) => b.updatedAt - a.updatedAt);
      this.entries.clear();
      ordered.slice(0, MAX_ENTRIES).forEach(entry => this.entries.set(entry.key, entry));
    }
    if (persist) this.schedulePersist();
  }

  get(key) {
    const normalizedKey = String(key).replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
    const entry = this.entries.get(normalizedKey);
    if (!entry) return null;
    const safe = sanitizeResult(entry.result);
    if (!safe) {
      this.entries.delete(normalizedKey);
      this.schedulePersist();
      return null;
    }
    entry.updatedAt = Date.now();
    return safe;
  }

  set(key, result) {
    const normalizedKey = String(key || '').replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
    const previous = normalizedKey ? this.entries.get(normalizedKey) : null;
    const clean = sanitizeResult(result);
    // Ordinary fresh results never enter or influence persistent cache state.
    // Only direct tablebase roots and verified mates are returned from set().
    if (!clean || !normalizedKey) return null;
    const chosen = compareCacheResults(previous?.result || null, clean, Date.now()) || clean;
    this.entries.set(normalizedKey, { key: normalizedKey, updatedAt: Date.now(), result: chosen });
    this.trim();
    return chosen;
  }

  delete(key) {
    const normalizedKey = String(key).replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
    this.entries.delete(normalizedKey);
    this.schedulePersist();
  }

  clear() {
    this.entries.clear();
    this.cancelScheduledPersist();
    this.dirty = false;
    try { this.storage?.removeItem(STORAGE_KEY); } catch {}
  }
}
