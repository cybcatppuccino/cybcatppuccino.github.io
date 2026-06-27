export class AnalysisClient {
  constructor({ onReady, onState, onInfo, onError } = {}) {
    this.onReady = onReady;
    this.onState = onState;
    this.onInfo = onInfo;
    this.onError = onError;
    this.worker = null;
    this.token = 0;
    this.active = false;
    this.paused = false;
    this.ready = false;
    this.pending = null;
    this.lastRequest = null;
    this.createWorker();
  }

  createWorker() {
    try {
      this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      this.worker.addEventListener('message', event => this.handleMessage(event.data || {}));
      this.worker.addEventListener('error', event => {
        this.onError?.(event.message || 'The analysis worker failed to start.');
      });
    } catch (error) {
      this.onError?.(error?.message || String(error));
    }
  }

  handleMessage(message) {
    if (message.type === 'ready') {
      this.ready = true;
      this.onReady?.(message);
      if (this.pending && this.active) {
        const pending = this.pending;
        this.pending = null;
        this.start(pending);
      }
      return;
    }
    if (message.token !== undefined && message.token !== this.token) return;
    if (message.type === 'state') {
      if (message.state === 'paused') this.paused = true;
      if (message.state === 'thinking') this.paused = false;
      this.onState?.(message);
    } else if (message.type === 'info') this.onInfo?.(message.result);
    else if (message.type === 'error') this.onError?.(message.message || 'Unknown engine error.');
  }

  start({
    fen,
    bookMoves = [],
    historyFens = [],
    effortMs = 950,
    multipv = 3,
    cacheKey = '',
    resumeResult = null,
    kernel = 'orion-js'
  }) {
    this.active = true;
    const request = { fen, bookMoves, historyFens, effortMs, multipv, cacheKey, resumeResult, kernel };
    this.lastRequest = request;
    if (!this.ready || !this.worker) {
      this.pending = request;
      return;
    }
    this.token += 1;
    this.worker.postMessage({ type: 'start', token: this.token, ...request, startPaused: this.paused });
  }

  update(request) {
    this.start(request);
  }

  pause() {
    if (!this.active || this.paused) return;
    this.paused = true;
    this.worker?.postMessage({ type: 'pause', token: this.token });
  }

  resume() {
    if (!this.active) {
      if (this.lastRequest) this.start(this.lastRequest);
      return;
    }
    if (!this.paused) return;
    this.paused = false;
    this.worker?.postMessage({ type: 'resume', token: this.token });
  }

  stop() {
    this.active = false;
    this.paused = false;
    this.pending = null;
    this.token += 1;
    this.worker?.postMessage({ type: 'stop', token: this.token });
  }

  clearHash() {
    this.worker?.postMessage({ type: 'clear', token: this.token });
  }
}
