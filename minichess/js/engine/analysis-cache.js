import { ENGINE_VERSION, scoreToDisplay } from './engine.js';

const STORAGE_KEY = 'gardner-analysis-cache-v15_1';
const MIGRATE_STORAGE_KEYS = Object.freeze([
  'gardner-analysis-cache-v15',
  'gardner-analysis-cache-v14_3',
  'gardner-analysis-cache-v14_2',
  'gardner-analysis-cache-v14_1',
  'gardner-analysis-cache-v14'
]);
const OLD_STORAGE_KEYS = Object.freeze(['gardner-analysis-cache-v12_1', 'gardner-analysis-cache-v12_2', 'gardner-analysis-cache-v13']);
const CACHE_SCHEMA = 19;
// v15.1 keeps the v14.3/v15 shared Orion persistent cache budget unchanged.
// Persistence remains debounced in browsers so the larger cache does not stall
// the UI on streamed analysis updates.
const MAX_ENTRIES = 576;
const MAX_PV_PLIES = 28;
const COMPATIBLE_ORION_ENGINES = Object.freeze(['Orion JS 14', 'Orion JS 14.1', 'Orion JS 14.2', 'Orion JS 14.3', 'Orion JS 15', ENGINE_VERSION]);

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
    dtm: Math.max(0, Number(line?.dtm || 0))
  };
}

function sanitizeResult(result) {
  // v14.2 restores the Orion cache as a shared analysis/play artifact.  Only
  // trusted Orion results are persisted; optional external kernels may consume
  // these as fallback/resume hints but never overwrite them.
  if (!result || !isCompatibleOrionEngine(result.engine) || !Array.isArray(result.lines)) return null;
  const lines = result.lines.slice(0, 5).map(cloneLine).filter(Boolean);
  if (!lines.length) return null;
  return {
    schema: CACHE_SCHEMA,
    engine: ENGINE_VERSION,
    engineLabel: String(result.engineLabel || ENGINE_VERSION),
    depth: Math.max(0, Number(result.depth || 0)),
    selDepth: Math.max(0, Number(result.selDepth || 0)),
    nodes: Math.max(0, Number(result.nodes || 0)),
    nps: Math.max(0, Number(result.nps || 0)),
    elapsed: Math.max(0, Number(result.elapsed || 0)),
    hashfull: Math.max(0, Number(result.hashfull || 0)),
    searchDepth: Math.max(1, Number(result.searchDepth || result.nextDepth || (Number(result.depth || 0) + 1))),
    nextDepth: Math.max(1, Number(result.nextDepth || (Number(result.depth || 0) + 1))),
    attemptedDepth: Math.max(1, Number(result.attemptedDepth || result.searchDepth || 1)),
    completed: result.completed !== false,
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
  };
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
      const previous = this.entries.get(key);
      const previousDepth = Number(previous?.result?.depth || 0);
      const nextDepth = Number(result.depth || 0);
      if (!previous || result.solved || nextDepth >= previousDepth || updatedAt > previous.updatedAt) {
        this.entries.set(key, { key, updatedAt, result });
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
    const clean = sanitizeResult(result);
    if (!clean || !key) return null;
    const normalizedKey = String(key).replace(/\|K(?:orion-js|fairy-stockfish)$/g, '');
    const previous = this.entries.get(normalizedKey);
    // A replay-verified mate is a solved artifact, not a transient depth result.
    // Never overwrite it with a later shallow/non-mate update.
    if (previous?.result?.solved && !clean.solved) return previous.result;
    // Never replace a deeper completed result with a shallower transient update.
    if (previous && previous.result.depth > clean.depth && !clean.terminal && !clean.endgameProof && !clean.solved) return previous.result;
    this.entries.set(normalizedKey, { key: normalizedKey, updatedAt: Date.now(), result: clean });
    this.trim();
    return clean;
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
