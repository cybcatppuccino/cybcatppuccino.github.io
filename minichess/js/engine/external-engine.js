import { EngineInternals, uciToMove, scoreToDisplay } from './engine.js';

const { makeMove } = EngineInternals;

export const ENGINE_KERNELS = Object.freeze({
  ORION: 'orion-js',
  FAIRY: 'fairy-stockfish'
});

export const FAIRY_STOCKFISH_LABEL = 'Fairy-Stockfish wasm 1.1.11';

function normalizeKernel(value) {
  return value === ENGINE_KERNELS.FAIRY ? ENGINE_KERNELS.FAIRY : ENGINE_KERNELS.ORION;
}

export function selectedKernel(value) {
  return normalizeKernel(value);
}

function validatePv(root, pv) {
  const cursor = root.clone();
  const clean = [];
  for (const uci of pv) {
    const text = String(uci || '').trim().toLowerCase();
    const move = uciToMove(cursor, text);
    if (!move) return null;
    clean.push(text);
    makeMove(cursor, move);
  }
  return clean;
}

/**
 * Validates every external UCI move against Orion's legal move generator before
 * the UI or AI play code can consume it.  If any PV is illegal, that line is
 * dropped; callers should fall back to Orion JS when no legal line remains.
 */
export function validateExternalAnalysisResult(root, result, { maxLines = 3 } = {}) {
  if (!result || !Array.isArray(result.lines)) return null;
  const lines = [];
  for (const sourceLine of result.lines) {
    const rawPv = Array.isArray(sourceLine?.pv) && sourceLine.pv.length
      ? sourceLine.pv
      : sourceLine?.move
        ? [sourceLine.move]
        : [];
    const pv = validatePv(root, rawPv);
    if (!pv?.length) continue;
    const score = Number(sourceLine.score || 0);
    lines.push({
      ...sourceLine,
      move: pv[0],
      score: Number.isFinite(score) ? score : 0,
      scoreText: String(sourceLine.scoreText || scoreToDisplay(Number.isFinite(score) ? score : 0)),
      pv,
      depth: Math.max(0, Number(sourceLine.depth || result.depth || 0)),
      mateVerified: false,
      external: true,
      source: 'fairy-stockfish'
    });
    if (lines.length >= maxLines) break;
  }
  if (!lines.length) return null;
  return {
    ...result,
    engine: FAIRY_STOCKFISH_LABEL,
    engineLabel: FAIRY_STOCKFISH_LABEL,
    depth: Math.max(0, Number(result.depth || Math.max(...lines.map(line => line.depth || 0), 0))),
    nodes: Math.max(0, Number(result.nodes || 0)),
    nps: Math.max(0, Number(result.nps || 0)),
    elapsed: Math.max(1, Number(result.elapsed || 1)),
    lines,
    completed: true,
    external: true,
    source: 'fairy-stockfish',
    solved: false,
    terminal: false,
    fortressProof: false,
    tablebase: false,
    mateProof: false,
    endgameProof: false
  };
}

export class FairyStockfishProvider {
  constructor({ onState = null, onInfo = null } = {}) {
    this.onState = onState;
    this.onInfo = onInfo;
    this.worker = null;
    this.ready = false;
    this.readyPromise = null;
    this.pending = new Map();
    this.lastInfo = new Map();
    this.tokenSeed = 0;
  }

  ensureWorker() {
    if (this.worker) return;
    this.readyPromise = new Promise((resolve, reject) => {
      try {
        this.worker = new Worker(new URL('../../vendor/fairy-stockfish/fairy-uci-worker.js', import.meta.url));
        this.worker.addEventListener('message', event => this.handleMessage(event.data || {}, resolve));
        this.worker.addEventListener('error', event => {
          const error = new Error(event.message || 'Fairy-Stockfish worker failed to start.');
          for (const item of this.pending.values()) item.reject(error);
          this.pending.clear();
          reject(error);
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  handleMessage(message, resolveReady) {
    if (message.type === 'ready') {
      this.ready = true;
      resolveReady?.(message);
      return;
    }
    if (message.type === 'state') {
      this.onState?.(message);
      const token = Number(message.token || 0);
      if (message.state === 'complete' && this.pending.has(token)) {
        const item = this.pending.get(token);
        this.pending.delete(token);
        item.resolve(this.lastInfo.get(token) || {
          engine: FAIRY_STOCKFISH_LABEL,
          engineLabel: FAIRY_STOCKFISH_LABEL,
          depth: Number(message.depth || 0),
          nodes: 0,
          nps: 0,
          elapsed: 1,
          lines: [],
          completed: true,
          external: true
        });
      }
      return;
    }
    if (message.type === 'info') {
      const token = Number(message.token || 0);
      if (message.result) {
        this.lastInfo.set(token, message.result);
        this.onInfo?.(message.result, token);
      }
      return;
    }
    if (message.type === 'error') {
      const token = Number(message.token || 0);
      const error = new Error(message.message || 'Fairy-Stockfish search failed.');
      const item = this.pending.get(token);
      if (item) {
        this.pending.delete(token);
        item.reject(error);
      }
    }
  }

  async search({ token = 0, fen, timeMs = 1000, depth = 0, multipv = 1 } = {}) {
    this.ensureWorker();
    await this.readyPromise;
    const searchToken = Number(token || ++this.tokenSeed);
    this.lastInfo.delete(searchToken);
    const promise = new Promise((resolve, reject) => {
      this.pending.set(searchToken, { resolve, reject });
      this.worker.postMessage({
        type: 'search',
        token: searchToken,
        fen: String(fen || '').trim(),
        timeMs: Math.max(50, Math.min(30000, Number(timeMs || 1000))),
        depth: Math.max(0, Math.min(64, Number(depth || 0))),
        multipv: Math.max(1, Math.min(8, Number(multipv || 1)))
      });
    });
    return promise;
  }

  stop() {
    try { this.worker?.postMessage({ type: 'stop' }); } catch {}
    for (const item of this.pending.values()) item.reject(new Error('Fairy-Stockfish search was stopped.'));
    this.pending.clear();
    this.lastInfo.clear();
  }
}
