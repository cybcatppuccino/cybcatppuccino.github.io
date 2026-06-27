import { ENGINE_VERSION, scoreToDisplay } from './engine.js';
import {
  compareAnalysisResults,
  isThinPvResult,
  lineHasCompletePv,
  resultPvProfile,
  shouldCacheResult,
  withResultQuality
} from './result-quality.js';

const STORAGE_KEY = 'gardner-analysis-cache-v18.1';
// v18.1 intentionally starts a fresh persistent analysis cache. Older v17.x
// entries could contain incomplete live PVs or stale tablebase-bound mate
// distances that are now classified by the stricter shared result-quality model.
const MIGRATE_STORAGE_KEYS = Object.freeze([]);
const OLD_STORAGE_KEYS = Object.freeze([
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
const CACHE_SCHEMA = 25;
// v18.1 keeps the shared Orion persistent cache budget unchanged.
// Persistence remains debounced in browsers so the larger cache does not stall
// the UI on streamed analysis updates.
const MAX_ENTRIES = 576;
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
    pv: Array.isArray(line?.pv) ? line.pv.slice(0, MAX_PV_PLIES).map(String) : [],
    mateVerified: Boolean(line?.mateVerified),
    mateRejected: Boolean(line?.mateRejected),
    endgameProof: Boolean(line?.endgameProof),
    mateProof: Boolean(line?.mateProof),
    fortressProof: Boolean(line?.fortressProof),
    matePending: Boolean(line?.matePending),
    tablebase: Boolean(line?.tablebase),
    tablebaseWdl: Number(line?.tablebaseWdl || 0),
    tablebaseBound: Boolean(line?.tablebaseBound),
    tablebaseExactDtm: Boolean(line?.tablebaseExactDtm),
    dtmUpperBound: Boolean(line?.dtmUpperBound),
    source: String(line?.source || ''),
    dtm: Math.max(0, Number(line?.dtm || 0)),
    pvComplete: Boolean(line?.pvComplete)
  };
}

function sanitizeResult(result) {
  // v14.2 restores the Orion cache as a shared analysis/play artifact.  Only
  // trusted Orion results are persisted; optional external kernels may consume
  // these as fallback/resume hints but never overwrite them.
  if (!result || !isCompatibleOrionEngine(result.engine) || !Array.isArray(result.lines)) return null;
  const lines = result.lines.slice(0, 5).map(cloneLine).filter(Boolean);
  if (!lines.length) return null;
  const resultForCompleteness = { ...result, lines };
  if (isThinPvResult(resultForCompleteness)) return null;
  const completeness = resultPvProfile(resultForCompleteness, lines);
  for (const line of lines) line.pvComplete = lineHasCompletePv(line, resultForCompleteness);
  return withResultQuality({
    schema: CACHE_SCHEMA,
    engine: ENGINE_VERSION,
    engineLabel: String(result.engineLabel || ENGINE_VERSION),
    depth: Math.max(0, Number(result.depth || 0)),
    scoreDepth: completeness.scoreDepth,
    pvDepth: completeness.pvDepth,
    pvTarget: completeness.pvTarget,
    pvComplete: completeness.pvComplete,
    selDepth: Math.max(0, Number(result.selDepth || 0)),
    nodes: Math.max(0, Number(result.nodes || 0)),
    nps: Math.max(0, Number(result.nps || 0)),
    elapsed: Math.max(0, Number(result.elapsed || 0)),
    hashfull: Math.max(0, Number(result.hashfull || 0)),
    searchDepth: Math.max(1, Number(result.searchDepth || result.nextDepth || (Number(result.depth || 0) + 1))),
    nextDepth: Math.max(1, Number(result.nextDepth || (Number(result.depth || 0) + 1))),
    attemptedDepth: Math.max(1, Number(result.attemptedDepth || result.searchDepth || 1)),
    completed: result.completed !== false && completeness.pvComplete,
    terminal: Boolean(result.terminal),
    tablebase: Boolean(result.tablebase),
    tablebaseSource: String(result.tablebaseSource || ''),
    tablebaseSignature: String(result.tablebaseSignature || ''),
    tablebaseWdl: Number(result.tablebaseWdl || 0),
    tablebaseDtmBound: Boolean(result.tablebaseDtmBound || lines[0]?.tablebaseBound),
    fortressProof: Boolean(result.fortressProof || lines[0]?.fortressProof),
    fortressNodes: Math.max(0, Number(result.fortressNodes || 0)),
    endgameProof: Boolean(result.endgameProof || lines[0]?.endgameProof),
    mateProof: Boolean(result.mateProof || lines[0]?.mateProof),
    rejectedMateClaims: Math.max(0, Number(result.rejectedMateClaims || 0)),
    cached: true,
    solved: Boolean(result.tablebase || result.fortressProof || result.endgameProof || lines[0]?.mateVerified),
    lines
  });
}

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
    mateVerified: true
  };
}


function compareCacheResults(previous, next, updatedAt = Date.now()) {
  return compareAnalysisResults(previous, next, { preferNextOnTie: updatedAt >= 0 });
}

function shouldPersistResult(result) {
  return shouldCacheResult(result);
}

export function buildAnalysisKey(position, historyFens = []) {
  const fenParts = position.toCompactFEN().split(/\s+/);
  const base = `${fenParts[0]} ${fenParts[1]} hm${fenParts[4] || 0}`;
  const history = historyFens
    .slice(-10)
    .map(fen => String(fen).split(/\s+/).slice(0, 2).join(' '))
    .join('>');
  return `${base}|${history}`;
}

export class AnalysisCache {
  constructor(storage = globalThis.localStorage) {
    this.storage = storage;
    this.entries = new Map();
    this.persistTimer = 0;
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
      const key = String(item.key)
        // v14/v14.1 accidentally made the engine kernel part of the persistent
        // key.  Strip it during migration so analysis, human-vs-AI and AI-vs-AI
        // can all reuse the same Orion search/mate artifacts again.
        .replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
      const updatedAt = Number(item.updatedAt || 0);
      if (!shouldPersistResult(result)) continue;
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
    this.persist();
  }

  persist() {
    if (!this.storage) return;
    if (this.persistTimer) {
      clearTimeout(this.persistTimer);
      this.persistTimer = 0;
    }
    try {
      const payload = [...this.entries.values()]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_ENTRIES);
      this.storage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Browsers may reject storage in private contexts. In-memory caching remains active.
    }
  }

  schedulePersist() {
    if (!this.deferPersistence) {
      this.persist();
      return;
    }
    if (this.persistTimer) return;
    this.persistTimer = setTimeout(() => {
      this.persistTimer = 0;
      this.persist();
    }, 300);
  }

  flush() {
    this.persist();
  }

  trim(persist = true) {
    if (this.entries.size > MAX_ENTRIES) {
      const ordered = [...this.entries.values()].sort((a, b) => b.updatedAt - a.updatedAt);
      this.entries.clear();
      ordered.slice(0, MAX_ENTRIES).forEach(entry => this.entries.set(entry.key, entry));
    }
    if (persist) this.schedulePersist();
  }

  get(key) {
    const entry = this.entries.get(String(key).replace(/\|K(?:orion-js|fairy-stockfish)$/g, ''));
    if (!entry) return null;
    entry.updatedAt = Date.now();
    return sanitizeResult(entry.result);
  }

  set(key, result) {
    const normalizedKey = String(key || '').replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
    const previous = normalizedKey ? this.entries.get(normalizedKey) : null;
    const clean = sanitizeResult(result);
    if (!clean || !normalizedKey) return previous?.result || null;
    if (!shouldPersistResult(clean)) return previous?.result || clean;
    const chosen = compareCacheResults(previous?.result || null, clean, Date.now());
    if (chosen === previous?.result) {
      if (previous) {
        previous.updatedAt = Date.now();
        this.schedulePersist();
        return previous.result;
      }
      // Defensive fallback for malformed/no-entry edge cases. This branch should
      // not be reachable with a valid clean result, but it prevents a streamed
      // first result from ever throwing because a previous cache entry is absent.
      return clean;
    }
    const safeChosen = chosen || clean;
    this.entries.set(normalizedKey, { key: normalizedKey, updatedAt: Date.now(), result: safeChosen });
    this.trim();
    return safeChosen;
  }

  delete(key) {
    const normalizedKey = String(key).replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
    this.entries.delete(normalizedKey);
    this.schedulePersist();
  }

  clear() {
    this.entries.clear();
    if (this.persistTimer) {
      clearTimeout(this.persistTimer);
      this.persistTimer = 0;
    }
    try { this.storage?.removeItem(STORAGE_KEY); } catch {}
  }
}
