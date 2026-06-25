const STORAGE_KEY = 'gardner-analysis-cache-v4';
const MAX_ENTRIES = 96;
const MAX_PV_PLIES = 24;

function cloneLine(line) {
  return {
    move: String(line?.move || ''),
    score: Number(line?.score || 0),
    scoreText: String(line?.scoreText || '0.00'),
    wdl: {
      win: Number(line?.wdl?.win || 0),
      draw: Number(line?.wdl?.draw || 0),
      loss: Number(line?.wdl?.loss || 0)
    },
    pv: Array.isArray(line?.pv) ? line.pv.slice(0, MAX_PV_PLIES).map(String) : []
  };
}

function sanitizeResult(result) {
  if (!result || !Array.isArray(result.lines)) return null;
  return {
    engine: String(result.engine || ''),
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
    cached: true,
    lines: result.lines.slice(0, 5).map(cloneLine)
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
    // Never replace a deeper completed result with a shallower transient update.
    if (previous && previous.result.depth > clean.depth && !clean.terminal) return previous.result;
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
