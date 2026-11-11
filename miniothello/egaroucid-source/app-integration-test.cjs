'use strict';
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict'),{Worker:NodeWorker}=require('node:worker_threads');
const root=path.resolve(__dirname,'..'),assets=path.join(root,'assets');
const mk=()=>({innerHTML:'',textContent:'',value:'',dataset:{},style:{},hidden:false,disabled:false,classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(){},addEventListener(){},appendChild(){},remove(){},querySelector(){return mk()},children:[],focus(){}});
const elements=new Map();const doc={querySelector(q){if(!elements.has(q))elements.set(q,mk());return elements.get(q)},createElement(){return mk()}};
const runner=`const {parentPort,workerData}=require('node:worker_threads');
const fs=require('fs'),path=require('path');
global.self={location:{href:'http://localhost/assets/'+workerData.name},handlers:{},addEventListener(type,fn){this.handlers[type]=fn},postMessage(message){parentPort.postMessage(message)}};
global.fetch=async url=>{const b=fs.readFileSync(path.join(workerData.assets,path.basename(String(url))));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
require(path.join(workerData.assets,workerData.name));
parentPort.on('message',data=>{
 const handler=self.onmessage||self.handlers.message;
 if(handler)Promise.resolve(handler({data})).catch(error=>parentPort.postMessage({type:'error',key:data.key,token:data.token,message:String(error)}));
});`;
class ProxyWorker {
 constructor(url){this.onmessage=null;this.onerror=null;this.terminated=false;this.listeners={};
  const name=path.basename(url);
  if(name==='orion-worker.js'){this.real=null;return;}
  if(!['egaroucid-worker.js','endgame-worker.js','black-oracle-worker.js'].includes(name))throw Error('Unexpected worker '+name);
  if(name==='egaroucid-worker.js')ProxyWorker.egaroucidCount=(ProxyWorker.egaroucidCount||0)+1;
  if(name==='endgame-worker.js')ProxyWorker.endgameCount=(ProxyWorker.endgameCount||0)+1;
  this.real=new NodeWorker(runner,{eval:true,workerData:{assets,name}});
  this.real.on('message',data=>{if(!this.terminated){this.onmessage?.({data});for(const f of this.listeners.message||[])f({data})}});
  this.real.on('error',err=>{if(!this.terminated){this.onerror?.(err);for(const f of this.listeners.error||[])f(err)}});
 }
 addEventListener(t,fn){(this.listeners[t]??=[]).push(fn)}
 postMessage(data){if(!this.terminated)this.real?.postMessage(data)}
 terminate(){this.terminated=true;this.real?.terminate()}
}

const ls={getItem(){return null},setItem(){}};
const ctx={document:doc,window:{addEventListener(){}},localStorage:ls,Worker:ProxyWorker,console,fetch:()=>{},WebAssembly,URL,Int8Array,Uint8Array,Uint32Array,Int32Array,Date,Math,Number,Array,Promise,BigInt,Map,Set,JSON,Intl,Boolean,Error,setTimeout,clearTimeout,queueMicrotask};
vm.createContext(ctx);
let source=fs.readFileSync(path.join(root,'app.js'),'utf8');source=source.slice(0,source.lastIndexOf('\nloadState();'));
vm.runInContext(source+'\n;globalThis.T={createInitialBoard,applyMoveToBoard,legalMovesFor,sideToMove,analyzePosition,requestAnalysis,chooseAiMove,formatScore,countsOf,lookupOracleInOrder};',ctx);
const T=ctx.T;
function setPosition(moves){ctx.givenMoves=moves;vm.runInContext(`(() => {
 root=new TreeNode({id:'root'});let n=root;
 for(const move of givenMoves){const child=new TreeNode({id:'p'+n.id+move,parent:n,move});n.children.push(child);n=child;}
 current=n;board=rebuildBoard(getLineMoves());latestAnalysis=null;latestAnalysisKey='';
})()`,ctx)}
async function until(expr,ms=30000){const start=Date.now();while(Date.now()-start<ms){if(vm.runInContext(expr,ctx))return;await new Promise(resolve=>setTimeout(resolve,40))}throw Error('Timed out '+expr)}
(async()=>{
 assert.equal(vm.runInContext('aiKernel',ctx),'egaroucid');
 assert.equal(T.formatScore(16,false),'+16.00');
 assert.equal(T.formatScore(16,true),'+16');
 const opening=await T.analyzePosition([],T.createInitialBoard());
 assert.equal(opening.source,'Black Oracle DB');
 assert.equal(opening.lines.length,4);
 assert(opening.lines.every(l=>l.exact&&l.evalWhite===4));
 const moves=[13,7,1];setPosition(moves);
 const off=await T.lookupOracleInOrder(moves);assert.equal(off,null);
 vm.runInContext('analysisEnabled=true;requestAnalysis()',ctx);
 await until('latestAnalysis?.wasmEgaroucid && latestAnalysis?.depth>=1',18000);
 const state=vm.runInContext('latestAnalysis',ctx);
 assert(state.lines.every(l=>!l.pending && !l.exact && Number.isFinite(l.evalWhite)));
 console.log('STANDARD',JSON.stringify({depth:state.depth,lines:state.lines.map(x=>[x.move,x.evalWhite])}));
 const beforeReuse=ProxyWorker.egaroucidCount;
 const earlyAiMove=await T.chooseAiMove();
 assert(state.lines.some(line=>line.move===earlyAiMove));
 assert.equal(ProxyWorker.egaroucidCount,beforeReuse,'AI should reuse live UI analysis WASM worker');
 console.log('AI REUSE',earlyAiMove,'WASM worker count',beforeReuse);
 // Establish an off-book state with exactly 18 discs: no Egaroucid
 // iterative worker is allowed. Verify router directly without waiting for
 // the potentially very expensive first exact 18-disc score.
 ctx.path18=[];ctx.path22=[];
 vm.runInContext(`(() => {
  let path=[13,7,1],b=rebuildBoard(path),side=sideToMove(path);
  for(let k=0;k<17;k++){
    const a=legalMovesFor(b,side),m=a.length?a[(k*17+3)%a.length]:-1;
    b=applyMoveToBoard(b,side,m);path.push(m);
    side=side==='black'?'white':'black';
    const discs=countsOf(b).black+countsOf(b).white;
    if(discs===18)path18=path.slice();
    if(discs===22)path22=path.slice();
  }
 })()`,ctx);
 assert(ctx.path18.length>0 && ctx.path22.length>0);
 setPosition(ctx.path18);
 assert.equal(vm.runInContext('countsOf(board).black+countsOf(board).white',ctx),18);
 assert.equal((await T.lookupOracleInOrder(ctx.path18)),null,'18 discs example must be outside both books');
 const at18=await T.analyzePosition(ctx.path18,vm.runInContext('board',ctx));
 assert(at18.endgameWasm && at18.backgroundFallback?.type==='endgame','18 discs must enter exact WASM route');
 assert.equal(at18.lines.some(x=>x.exact),false);
 console.log('EXACT-THRESHOLD',JSON.stringify({discs:18,route:at18.backgroundFallback.type,lines:at18.lines.length}));
 vm.runInContext('stopFallbackSearch();stopEndgameSearch()',ctx);
 // A 22-disc off-book state is fast enough for real independent endgame WASM.
 setPosition(ctx.path22);
 assert.equal(vm.runInContext('countsOf(board).black+countsOf(board).white',ctx),22);
 const beforeExact=ProxyWorker.endgameCount||0;
 const beforeEgar=ProxyWorker.egaroucidCount||0;
 vm.runInContext('requestAnalysis()',ctx);
 await until('endgameWorker && activeEndgameKey',10000);
 // Two concurrent callers must reuse the same real exact WASM work.
 const liveAiMove=T.chooseAiMove();
 await until('latestAnalysis?.endgameWasm && latestAnalysis?.exact',30000);
 const late=vm.runInContext('latestAnalysis',ctx);
 assert(late.exact && late.lines.every(x=>x.exact&&x.evalWhite%2===0));
 assert((ProxyWorker.endgameCount||0)>beforeExact,'18+ must launch exact endgame worker');
 assert.equal(ProxyWorker.egaroucidCount,beforeEgar,'18+ must not launch Egaroucid worker');
 console.log('ENDGAME',JSON.stringify({discs:22,exact:late.exact,candidates:late.lines.map(l=>[l.move,l.evalWhite]),workers:ProxyWorker.endgameCount}));
 const best=await liveAiMove;assert(late.lines.some(l=>l.move===best));
 assert.equal(ProxyWorker.endgameCount,beforeExact+1,'UI and AI must not duplicate active endgame WASM');
 console.log('AI SHARED EXACT',best);
 vm.runInContext('aiKernel="poseidon"',ctx);
 assert.equal(vm.runInContext('selectedSearchWorkerPath()',ctx),'assets/poseidon-worker.js');
 vm.runInContext('aiKernel="orion"',ctx);
 assert.equal(vm.runInContext('selectedSearchWorkerPath()',ctx),'assets/fallback-worker.js');
 console.log('APP INTEGRATION PASS');
 vm.runInContext('stopFallbackSearch();stopEndgameSearch();blackOracleKernel.worker?.terminate();oracleKernel.worker?.terminate()',ctx);
})().catch(err=>{console.error(err);process.exitCode=1;vm.runInContext('stopFallbackSearch();stopEndgameSearch();blackOracleKernel.worker?.terminate();oracleKernel.worker?.terminate()',ctx)});
