import { ENGINE_VERSION, scoreToDisplay } from './engine.js';

const STORAGE_KEY = 'gardner-analysis-cache-v10';
const CACHE_SCHEMA = 9;
const MAX_ENTRIES = 96;
const MAX_PV_PLIES = 24;

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
    fortressProof: Boolean(line?.fortressProof),
    tablebase: Boolean(line?.tablebase),
    tablebaseWdl: Number(line?.tablebaseWdl || 0),
    dtmUpperBound: Boolean(line?.dtmUpperBound),
    source: String(line?.source || ''),
    dtm: Math.max(0, Number(line?.dtm || 0))
  };
}

function sanitizeResult(result) {
  if (!result || result.engine !== ENGINE_VERSION || !Array.isArray(result.lines)) return null;
  const lines = result.lines.slice(0, 5).map(cloneLine).filter(Boolean);
  return {
    schema: CACHE_SCHEMA,
    engine: ENGINE_VERSION,
    engineLabel: String(result.engineLabel || ''),
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
    fortressProof: Boolean(result.fortressProof || lines[0]?.fortressProof),
    fortressNodes: Math.max(0, Number(result.fortressNodes || 0)),
    endgameProof: Boolean(lines[0]?.endgameProof),
    rejectedMateClaims: Math.max(0, Number(result.rejectedMateClaims || 0)),
    cached: true,
    solved: Boolean(result.tablebase || result.fortressProof || lines[0]?.mateVerified),
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
    this.load();
  }

  load() {
    if (!this.storage) return;
    try {
      const payload = JSON.parse(this.storage.getItem(STORAGE_KEY) || '[]');
      if (!Array.isArray(payload)) return;
      for (const item of payload) {
        if (!item?.key || !item?.result) continue;
        const result = sanitizeResult(item.result);
        if (!result) continue;
        this.entries.set(String(item.key), {
          key: String(item.key),
          updatedAt: Number(item.updatedAt || 0),
          result
        });
      }
      this.trim(false);
    } catch {
      this.entries.clear();
    }
  }

  persist() {
    if (!this.storage) return;
    try {
      const payload = [...this.entries.values()]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_ENTRIES);
      this.storage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Browsers may reject storage in private contexts. In-memory caching remains active.
    }
  }

  trim(persist = true) {
    if (this.entries.size > MAX_ENTRIES) {
      const ordered = [...this.entries.values()].sort((a, b) => b.updatedAt - a.updatedAt);
      this.entries.clear();
      ordered.slice(0, MAX_ENTRIES).forEach(entry => this.entries.set(entry.key, entry));
    }
    if (persist) this.persist();
  }

  get(key) {
    const entry = this.entries.get(String(key));
    if (!entry) return null;
    entry.updatedAt = Date.now();
    return sanitizeResult(entry.result);
  }

  set(key, result) {
    const clean = sanitizeResult(result);
    if (!clean || !key) return null;
    const previous = this.entries.get(String(key));
    // A replay-verified mate is a solved artifact, not a transient depth result.
    // Never overwrite it with a later shallow/non-mate update.
    if (previous?.result?.solved && !clean.solved) return previous.result;
    // Never replace a deeper completed result with a shallower transient update.
    if (previous && previous.result.depth > clean.depth && !clean.terminal && !clean.endgameProof && !clean.solved) return previous.result;
    this.entries.set(String(key), { key: String(key), updatedAt: Date.now(), result: clean });
    this.trim();
    return clean;
  }

  delete(key) {
    this.entries.delete(String(key));
    this.persist();
  }

  clear() {
    this.entries.clear();
    try { this.storage?.removeItem(STORAGE_KEY); } catch {}
  }
}
