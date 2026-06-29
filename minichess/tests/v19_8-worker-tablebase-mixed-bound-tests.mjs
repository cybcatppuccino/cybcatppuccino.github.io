import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { readFile, stat } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
const root=path.resolve(fileURLToPath(new URL('../tools/gardner_tablebase/tables/', import.meta.url)));
const server=http.createServer(async(req,res)=>{try{const u=new URL(req.url,'http://x');let p=decodeURIComponent(u.pathname); if(!p.startsWith('/tools/gardner_tablebase/tables/')) throw 0; const f=path.resolve(root,p.replace('/tools/gardner_tablebase/tables/','')); if(!f.startsWith(root+path.sep))throw 0; const st=await stat(f);if(!st.isFile())throw 0; res.writeHead(200,{'Content-Type':f.endsWith('.json')?'application/json':'application/octet-stream'});res.end(await readFile(f));}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r)); const base=`http://127.0.0.1:${server.address().port}/`;
const w=new Worker(new URL('./worker-node-tablebase-wrapper.mjs',import.meta.url),{type:'module',workerData:{baseUrl:base}});
const got = [];
await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error(`Timed out waiting for finite WDL audit: ${JSON.stringify(got.slice(-12))}`)), 20_000);
  w.on('error', reject);
  w.on('message', message => {
    got.push(message);
    if (message.type === 'ready') {
      w.postMessage({ type: 'start', token: 98, fen: '5/k4/p1p2/2P1P/2K2 w - - 0 3', cacheKey: 'tb-test', bookMoves: [], historyFens: [], effortMs: 2400, multipv: 1 });
    }
    if (message.type === 'info' && message.token === 98 && message.result?.tablebaseMixedAudit) {
      clearTimeout(timer);
      resolve();
    }
  });
});
const audit = got.filter(message => message.type === 'info' && message.result?.tablebaseMixedAudit).at(-1)?.result;
assert.ok(audit, 'Expected an actual Worker mixed-WDL audit result.');
assert.equal(audit.lines[0]?.scoreText, '+10.00');
assert.notEqual(audit.lines[0]?.scoreText, 'TB bridge · verifying');
assert.ok(Math.abs(Number(audit.lines[0]?.score || 0)) < 22_000);
console.log('v19.8 Worker mixed-WDL audit integration test passed.', {
  depth: audit.tablebaseMixedAuditDepth,
  score: audit.lines[0]?.scoreText,
  boundedHits: audit.tablebaseMixedAuditProbeHits
});
w.postMessage({type:'stop',token:99}); await w.terminate(); await new Promise(r=>server.close(r));
