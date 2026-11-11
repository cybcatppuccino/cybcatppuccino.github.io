'use strict';
// Actual C/WASM worker, <18 discs only. The 18+ route is tested in app integration.
const assert=require('node:assert/strict');
const fs=require('fs'),path=require('path'),{Worker}=require('worker_threads');
const assets=path.resolve(__dirname,'../assets');
const src=`const {parentPort,workerData}=require('worker_threads');
const fs=require('fs'),path=require('path');
global.self={location:{href:'http://localhost/assets/egaroucid-worker.js'},postMessage:m=>parentPort.postMessage(m)};
global.fetch=async u=>{const file=path.join(workerData.assets,path.basename(String(u)));const b=fs.readFileSync(file);return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};};
require(path.join(workerData.assets,'egaroucid-worker.js'));
parentPort.on('message',message=>self.onmessage({data:message}));`;
const D=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
function flips(b,s,p){if(b[p])return [];let out=[],x=p%6,y=p/6|0;
 for(const [dx,dy] of D){let ray=[],xx=x+dx,yy=y+dy;while(xx>=0&&xx<6&&yy>=0&&yy<6){const n=yy*6+xx;if(b[n]===-s)ray.push(n);else {if(b[n]===s&&ray.length)out.push(...ray);break;}xx+=dx;yy+=dy;}}
 return out;}
function legal(b,s){const a=[];for(let i=0;i<36;i++)if(flips(b,s,i).length)a.push(i);return a;}
function move(b,s,m){if(m===-1)return b.slice();let c=b.slice();c[m]=s;for(const x of flips(b,s,m))c[x]=s;return c;}
function randomState(seed,pieces){const b=Array(36).fill(0);b[14]=-1;b[15]=1;b[20]=1;b[21]=-1;let side=1;
 for(let i=4;i<pieces;i++){let ms=legal(b,side);if(!ms.length){if(!legal(b,-side).length)return null;side=-side;i--;continue;}
  seed=(Math.imul(seed,1664525)+1013904223)|0;const pick=ms[(seed>>>0)%ms.length];b.splice(0,36,...move(b,side,pick));side=-side;}
 return {b,side};}
function spawn(){return new Worker(src,{eval:true,workerData:{assets}})}
function wasmMemory(){const bytes=fs.readFileSync(path.join(assets,'egaroucid-6x6.wasm'));
 const mod=new WebAssembly.Module(bytes);
 const inst=new WebAssembly.Instance(mod,{env:{notify(){}}});
 assert.equal(inst.exports.memory.buffer.byteLength,48*1024*1024);
 const input=inst.exports.board_ptr();assert(input>=0&&input<48*1024*1024, 'Input pointer must be within WASM memory');
 return {bytes:bytes.length,memory:inst.exports.memory.buffer.byteLength,input};}
(async()=>{
 const mem=wasmMemory();
 const st=randomState(91123*2+13,17); assert(st);
 const w=spawn(); const messages=[],started=Date.now();
 const result=await new Promise((resolve,reject)=>{
  const deadline=setTimeout(()=>resolve('enough-iterations'),6500);
  w.on('error',reject);
  w.on('message',msg=>{
   if(msg.type==='error')reject(new Error(msg.message));
   if(msg.result){messages.push({time:Date.now()-started,...msg});
     if(msg.result.depth>=8 && messages.filter(x=>x.type==='update').length>=2) {clearTimeout(deadline);resolve('depth8');}}
  });
  w.postMessage({type:'start',key:'s',token:'1',board:st.b.map(x=>x===1?'black':x===-1?'white':null),side:st.side===1?'black':'white'});
 });
 await w.terminate();
 assert(messages.length>=2,'Expected repeated WASM analysis updates');
 assert(messages.some(x=>x.result.depth>=1),'WASM did not finish one full iteration');
 for(const m of messages){
  assert.equal(m.result.mode,'6x6 Egaroucid search');
  const d=m.result.depth;
  assert(m.result.lines.every(x=>d===0?x.pending===true:(!x.pending&&Number.isFinite(x.evalWhite))));
  if(!m.result.exact)assert(m.result.lines.every(x=>!x.exact));
 }
 const updates=messages.filter(x=>x.type==='update');
 for(let i=1;i<updates.length;i++)assert(updates[i].time-updates[i-1].time>=400,'WASM worker updates must be throttled to ~500 ms');
 // Direct routing guard: the separate exact endgame worker owns >=18 discs.
 const late=randomState(1733,18);assert(late);
 const other=spawn();
 const msg=await new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>reject(Error('Expected a routing error for >=18 discs')),5000);
  other.on('error',reject);
  other.on('message',m=>{clearTimeout(timeout);resolve(m)});
  other.postMessage({type:'start',key:'x',token:'2',board:late.b.map(x=>x===1?'black':x===-1?'white':null),side:late.side===1?'black':'white'});
 });
 await other.terminate();
 assert.equal(msg.type,'error');assert.match(msg.message,/18\+ discs/);
 console.log('EGAROUCID WORKER PASS',JSON.stringify({reason:result,initialMiB:mem.memory/1048576,wasmBytes:mem.bytes,frames:messages.length,depths:messages.map(x=>x.result.depth),lastMs:messages.at(-1).time,lateRejected:true}));
})().catch(err=>{console.error(err);process.exitCode=1});
