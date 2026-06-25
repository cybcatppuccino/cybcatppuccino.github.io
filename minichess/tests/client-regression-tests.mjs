import assert from 'node:assert/strict';

class FakeWorker {
  constructor() {
    this.listeners = { message: [], error: [] };
    this.sent = [];
    queueMicrotask(() => this.emit('message', { data: { type: 'ready', engine: 'test-engine' } }));
  }
  addEventListener(type, listener) { this.listeners[type].push(listener); }
  postMessage(message) { this.sent.push(message); }
  emit(type, event) { for (const listener of this.listeners[type] || []) listener(event); }
}

globalThis.Worker = FakeWorker;
const { AnalysisClient } = await import('../js/engine/client.js');
const client = new AnalysisClient();
client.update({ fen: 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1', effortMs: 550, multipv: 3 });
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(client.active, true, 'update() must activate analysis on the first click');
assert.equal(client.worker.sent.length, 1, 'The pending first request must be sent after worker readiness');
assert.equal(client.worker.sent[0].type, 'start');
assert.equal(client.worker.sent[0].fen, 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
console.log('AnalysisClient first-click regression test passed.');
