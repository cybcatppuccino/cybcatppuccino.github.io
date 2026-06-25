export class PlayEngineClient {
  constructor({ onReady, onState, onResult, onError } = {}) {
    this.onReady = onReady;
    this.onState = onState;
    this.onResult = onResult;
    this.onError = onError;
    this.worker = null;
    this.ready = false;
    this.token = 0;
    this.pending = null;
    this.createWorker();
  }

  createWorker() {
    try {
      this.worker = new Worker(new URL('./play-worker.js', import.meta.url), { type: 'module' });
      this.worker.addEventListener('message', event => this.handleMessage(event.data || {}));
      this.worker.addEventListener('error', event => this.onError?.(event.message || 'The play engine failed to start.'));
    } catch (error) {
      this.onError?.(error?.message || String(error));
    }
  }

  handleMessage(message) {
    if (message.type === 'ready') {
      this.ready = true;
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

  cancel() {
    this.pending = null;
    this.token += 1;
    this.worker?.postMessage({ type: 'cancel', token: this.token });
  }
}
