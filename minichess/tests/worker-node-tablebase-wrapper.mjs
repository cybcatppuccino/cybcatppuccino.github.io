import { parentPort, workerData } from 'node:worker_threads';
globalThis.location = { href: workerData.baseUrl };
const listeners = new Set();
globalThis.self = {
  postMessage(message) { parentPort.postMessage(message); },
  addEventListener(type, handler) { if (type === 'message') listeners.add(handler); }
};
parentPort.on('message', data => { for (const handler of listeners) handler({ data }); });
await import('../js/engine/worker.js');
