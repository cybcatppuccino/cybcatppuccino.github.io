export class PlayEngineClient {
  constructor({ onReady, onState, onInfo, onResult, onError } = {}) {
    this.onReady = onReady;
    this.onState = onState;
    this.onInfo = onInfo;
    this.onResult = onResult;
    this.onError = onError;
    this.worker = null;
    this.ready = false;
    this.token = 0;
    this.pending = null;
    this.restartAttempts = 0;
    this.createWorker();
  }

  createWorker() {
    try {
      this.worker = new Worker(new URL('./play-worker.js', import.meta.url), { type: 'module' });
      this.worker.addEventListener('message', event => this.handleMessage(event.data || {}));
      this.worker.addEventListener('error', event => this.handleWorkerFailure(event.message || 'The play engine failed to start.'));
      try {
        this.worker.addEventListener('messageerror', () => this.handleWorkerFailure('The play engine sent an unreadable message.'));
      } catch {}
    } catch (error) {
      this.handleWorkerFailure(error?.message || String(error));
    }
  }

  handleWorkerFailure(message) {
    const text = String(message || 'The play engine crashed.');
    this.ready = false;
    this.pending = null;
    try { this.worker?.terminate?.(); } catch {}
    this.worker = null;
    this.onError?.(`${text} Turn AI off and on again to restart the engine safely.`);
    if (this.restartAttempts < 1) {
      this.restartAttempts += 1;
      setTimeout(() => this.createWorker(), 80);
    }
  }

  handleMessage(message) {
    if (message.type === 'ready') {
      this.ready = true;
      this.restartAttempts = 0;
      this.onReady?.(message);
      if (this.pending) {
        const request = this.pending;
        this.pending = null;
        this.search(request);
      }
      return;
    }
    if (message.token !== undefined && message.token !== this.token) return;
    if (message.type === 'state') this.onState?.(message);
    else if (message.type === 'info') this.onInfo?.(message.result);
    else if (message.type === 'result') this.onResult?.(message.result);
    else if (message.type === 'error') this.onError?.(message.message || 'Unknown play-engine error.');
  }

  search(request) {
    if (!this.ready || !this.worker) {
      this.pending = request;
      return;
    }
    this.token += 1;
    this.worker.postMessage({ type: 'search', token: this.token, ...request });
  }

  pause() {
    if (!this.worker) return;
    this.worker.postMessage({ type: 'pause', token: this.token });
  }

  resume() {
    if (!this.worker) return;
    this.worker.postMessage({ type: 'resume', token: this.token });
  }

  dispose() {
    this.pending = null;
    this.token += 1;
    try { this.worker?.terminate?.(); } catch {}
    this.worker = null;
    this.ready = false;
  }

  cancel() {
    this.pending = null;
    this.token += 1;
    this.worker?.postMessage({ type: 'cancel', token: this.token });
  }
}
