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
  constructor({ onState = null, onInfo = null, startupTimeoutMs = 15000 } = {}) {
    this.onState = onState;
    this.onInfo = onInfo;
    this.startupTimeoutMs = Math.max(2000, Number(startupTimeoutMs || 15000));
    this.worker = null;
    this.ready = false;
    this.readyPromise = null;
    this.rejectReady = null;
    this.pending = new Map();
    this.lastInfo = new Map();
    this.tokenSeed = 0;
  }

  failAll(error) {
    for (const item of this.pending.values()) item.reject(error);
    this.pending.clear();
    this.lastInfo.clear();
  }

  teardown() {
    try { this.worker?.terminate?.(); } catch {}
    this.worker = null;
    this.ready = false;
    this.readyPromise = null;
    this.rejectReady = null;
  }

  ensureWorker() {
    if (this.worker && this.readyPromise) return;
    this.ready = false;
    this.readyPromise = new Promise((resolve, reject) => {
      this.rejectReady = reject;
      let settled = false;
      const settleReady = message => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.ready = true;
        resolve(message);
      };
      const failReady = error => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.failAll(error);
        this.teardown();
        reject(error);
      };
      const timer = setTimeout(() => {
        failReady(new Error('Fairy-Stockfish did not finish UCI initialization. Serve the app over HTTP with COOP/COEP headers, not file://.'));
      }, this.startupTimeoutMs);
      try {
        this.worker = new Worker(new URL('../../vendor/fairy-stockfish/fairy-uci-worker.js', import.meta.url));
        this.worker.addEventListener('message', event => this.handleMessage(event.data || {}, settleReady, failReady));
        this.worker.addEventListener('error', event => {
          failReady(new Error(event.message || 'Fairy-Stockfish worker failed to start.'));
        });
        this.worker.addEventListener('messageerror', () => {
          failReady(new Error('Fairy-Stockfish worker emitted an unreadable message.'));
        });
      } catch (error) {
        failReady(error);
      }
    });
  }

  handleMessage(message, resolveReady, rejectReady) {
    if (message.type === 'ready') {
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
        return;
      }
      // Initialization failures are often emitted before a search token exists
      // (for example, missing COOP/COEP headers for the pthread wasm build).
      // Reject the ready promise so callers can fall back instead of leaving the
      // UI stuck at "Starting the local engine…" forever.
      if (!this.ready) rejectReady?.(error);
    }
  }

  async search({ token = 0, fen, timeMs = 1000, depth = 0, multipv = 1 } = {}) {
    this.ensureWorker();
    await this.readyPromise;
    const searchToken = Number(token || ++this.tokenSeed);
    this.lastInfo.delete(searchToken);
    const timeoutMs = Math.max(5000, Math.min(45000, Number(timeMs || 1000) + 10000));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.has(searchToken)) return;
        this.pending.delete(searchToken);
        reject(new Error('Fairy-Stockfish search timed out before bestmove.'));
      }, timeoutMs);
      this.pending.set(searchToken, {
        resolve: value => { clearTimeout(timer); resolve(value); },
        reject: error => { clearTimeout(timer); reject(error); }
      });
      this.worker.postMessage({
        type: 'search',
        token: searchToken,
        fen: String(fen || '').trim(),
        timeMs: Math.max(50, Math.min(30000, Number(timeMs || 1000))),
        depth: Math.max(0, Math.min(64, Number(depth || 0))),
        multipv: Math.max(1, Math.min(8, Number(multipv || 1)))
      });
    });
  }

  stop() {
    try { this.worker?.postMessage({ type: 'stop' }); } catch {}
    this.failAll(new Error('Fairy-Stockfish search was stopped.'));
  }
}
