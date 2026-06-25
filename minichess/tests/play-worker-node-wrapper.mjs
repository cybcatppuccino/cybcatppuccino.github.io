import { parentPort } from 'node:worker_threads';
const listeners = new Set();
globalThis.self = {
  postMessage(message) { parentPort.postMessage(message); },
  addEventListener(type, handler) { if (type === 'message') listeners.add(handler); }
};
parentPort.on('message', data => {
  for (const handler of listeners) handler({ data });
});
await import('../js/engine/play-worker.js');
