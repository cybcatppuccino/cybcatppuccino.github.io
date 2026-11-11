'use strict';
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),assets=path.join(root,'assets');
function fakeDom(){return {innerHTML:'',textContent:'',value:'',dataset:{},style:{},hidden:false,disabled:false,classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(){},addEventListener(){},appendChild(){},remove(){},querySelector(){return fakeDom()},children:[],focus(){}}}
const elements=new Map(),doc={querySelector(s){if(!elements.has(s))elements.set(s,fakeDom());return elements.get(s)},createElement(){return fakeDom()}};
class VMWorker {
 constructor(url){this.url=url;this.listeners={};this.onmessage=null;this.onerror=null;this.terminated=false;
  const src=path.join(assets,path.basename(url));if(!url.endsWith('black-oracle-worker.js') && !url.endsWith('endgame-worker.js'))return;
  const workerSelf={location:{href:'http://localhost/assets/'+path.basename(url)},addEventListener:(t,h)=>{if(t==='message')this.onWorkerMessage=h},postMessage:(m)=>queueMicrotask(()=>{if(this.terminated)return;for(const f of this.listeners.message||[])f({data:m});this.onmessage?.({data:m})})};
  const fetch=async url=>{const b=fs.readFileSync(path.join(assets,path.basename(url.toString())));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
  vm.runInNewContext(fs.readFileSync(src,'utf8'),{self:workerSelf,fetch,WebAssembly,URL,Int8Array,Int16Array,Uint8Array,Uint32Array,DataView,Promise,Array,Boolean,Number,String,Math,Error,console});
 }
 addEventListener(type,fn){(this.listeners[type]??=[]).push(fn)}
 postMessage(msg){if(this.onWorkerMessage)queueMicrotask(()=>{if(!this.terminated)this.onWorkerMessage({data:msg}).catch(e=>console.error('Worker error',e))})}
 terminate(){this.terminated=true}
}
const ctx={document:doc,window:{addEventListener(){}},localStorage:{getItem(){return null},setItem(){}},Worker:VMWorker,console,fetch:()=>{},WebAssembly,URL,Uint8Array,Int8Array,Date,Math,Number,Array,Promise,BigInt,Map,Set,JSON,Intl,Boolean,Error,setTimeout,clearTimeout,queueMicrotask};
vm.createContext(ctx);
let app=fs.readFileSync(path.join(root,'app.js'),'utf8');app=app.slice(0,app.lastIndexOf('\nloadState();'));
vm.runInContext(app+'\n;globalThis.t={createInitialBoard,applyMoveToBoard,legalMovesFor,sideToMove,lookupOracleInOrder,analyzePosition,normalizeEndgameAnalysis,requestEndgameForAi,countsOf,hasEndgameThreshold,blackOracleBlackAnalysis,completeOracleAnalysis,requestAnalysis,chooseAiMove};',ctx);
const t=ctx.t;
function boardFor(moves){let b=t.createInitialBoard();moves.forEach((m,i)=>b=t.applyMoveToBoard(b,i%2?'white':'black',m));return b;}
function buildRandom(seed){let moves=[],b=t.createInitialBoard();for(let i=0;i<14;i++){const side=t.sideToMove(moves),legal=t.legalMovesFor(b,side);seed=(Math.imul(seed,1664525)+1013904223)|0;const move=legal.length?legal[(seed>>>0)%legal.length]:-1;moves.push(move);b=t.applyMoveToBoard(b,side,move);}return {moves,b};}
(async()=>{
 let off=null;
 for(let i=1;i<40;i++){let trial=buildRandom(i*73);if(!(await t.lookupOracleInOrder(trial.moves))){off=trial;break;}}
 assert(off,'Could not find 18-disc position outside both books');
 assert.equal(t.countsOf(off.b).empty,18);
 const pending=await t.analyzePosition(off.moves,off.b);
 assert.equal(pending.backgroundFallback.type,'endgame');
 assert(pending.lines.every(l=>l.pending));
 const raw=await t.requestEndgameForAi(off.b,t.sideToMove(off.moves));
 const ordinary=t.normalizeEndgameAnalysis(raw);
 const legal=t.legalMovesFor(off.b,t.sideToMove(off.moves));
 assert.deepEqual(ordinary.lines.map(l=>l.move).sort((a,b)=>a-b),legal.slice().sort((a,b)=>a-b));
 assert(ordinary.lines.every(l=>!l.pending && Number.isFinite(l.evalWhite)));
 assert(ordinary.lines.every(l=>l.source==='Poseidon endgame WASM'));
 assert.equal(ordinary.endgameWasm,true);
 assert.equal(ordinary.exact,true);
 assert(raw.exact && raw.lines.every(l=>l.exact && Number.isInteger(l.evalWhite)));
 assert(raw.moves && raw.moves.length===legal.length);
 // A finished candidate is shown exactly; unsearched squares MUST stay pending.
 const partial=t.normalizeEndgameAnalysis({...raw,lines:raw.lines.slice(0,1),
   exact:false,completed:1});
 assert.equal(partial.exact,false);
 assert.equal(partial.bestMove,null);
 assert.equal(partial.lines.filter(l=>l.pending).length,legal.length-1);
 assert.equal(partial.lines.filter(l=>l.exact).length,1);
 assert(partial.lines.filter(l=>l.exact).every(l=>Number.isInteger(l.evalWhite)));
 console.log('OFFBOOK:',{moves:off.moves,source:ordinary.source,candidates:ordinary.lines.length,nodes:ordinary.nodes,depth:ordinary.depth,exact:ordinary.exact});
 // Overlaid oracle lines must win even if the WASM attempts to replace them.
 const m=ordinary.lines[0].move;
 const overlay={side:ordinary.side,legalMoves:legal,source:'Black Oracle DB',bestMove:m,
  oracleLines:[{move:m,evalWhite:27,source:'Black Oracle DB',exact:true,pv:[m]}]};
 const mixed=t.normalizeEndgameAnalysis(raw,overlay);
 assert.equal(mixed.lines.find(l=>l.move===m).evalWhite,27);
 assert.equal(mixed.lines.find(l=>l.move===m).source,'Black Oracle DB');
 assert(mixed.lines.every(l=>!l.pending));
 assert.equal(mixed.bestMove,m);
 console.log('OVERLAY:',{lines:mixed.lines.length,oracleValue:mixed.lines.find(l=>l.move===m).evalWhite,source:mixed.source});
 // Drive the real UI analysis dispatcher and dedicated WASM worker through VM.
 ctx.historyForTest=off.moves;
 vm.runInContext(`(() => {analysisEnabled=true;let n=root;for(const m of historyForTest){const child=new TreeNode({id:'test'+m+Math.random(),parent:n,move:m});n.children.push(child);n=child;}current=n;board=rebuildBoard(getLineMoves());requestAnalysis()})()`,ctx);
 await new Promise((resolve,reject)=>{const start=Date.now();const timer=setInterval(()=>{
  const done=vm.runInContext('latestAnalysis?.endgameWasm && latestAnalysis?.lines?.every(l=>!l.pending)',ctx);
  if(done){clearInterval(timer);resolve()}
  else if(Date.now()-start>12000){clearInterval(timer);reject(Error('UI worker did not complete'))}
 },15)});
 const source=vm.runInContext('latestAnalysis.source',ctx);
 const count=vm.runInContext('latestAnalysis.lines.length',ctx);
 assert.equal(count,legal.length);assert.match(source,/Poseidon endgame WASM/);
 // Orion mode uses same WASM but reports the selected engine's label.
 vm.runInContext('aiKernel="orion";requestAnalysis()',ctx);
 await new Promise((resolve,reject)=>{const start=Date.now();const timer=setInterval(()=>{
  const done=vm.runInContext('latestAnalysis?.endgameWasm && latestAnalysis?.source?.includes("Orion")',ctx);
  if(done){clearInterval(timer);resolve()}
  else if(Date.now()-start>12000){clearInterval(timer);reject(Error('Orion UI worker did not complete'))}
 },15)});
 console.log('DISPATCH:',{poseidon:source,orion:vm.runInContext('latestAnalysis.source',ctx),uiLines:count});
 // With no parent book hit, AI should select the best WASM candidate directly.
 const aiMove=await t.chooseAiMove();
 assert(legal.includes(aiMove));
 console.log('AI:',{move:aiMove,source:'Orion endgame WASM'});
 // Walk a REAL opening-book path to 18+ discs and preserve every exact line.
 let actual=null;
 for(let seed=0;seed<18 && !actual;seed++) {
   let moves=[],b=t.createInitialBoard();
   for(let ply=0;ply<16;ply++){
     const side=t.sideToMove(moves),hit=await t.lookupOracleInOrder(moves);
     if(!hit)break;
     let move;
     if(side==='black')move=hit.source==='black'?hit.response.bestMove:hit.response.options[0].move;
     else move=hit.source==='black'?hit.response.options[(seed+ply)%hit.response.options.length].move:hit.response.bestMove;
     if(move===undefined)break;
     moves.push(move);b=t.applyMoveToBoard(b,side,move);
     if(moves.length>=14 && !(moves.length%2)){
       const books=await t.lookupOracleInOrder(moves);
       if(books){const analysis=await t.analyzePosition(moves,b);
         if(analysis?.backgroundFallback?.type==='oracle-mixed') {actual={moves,b,analysis};break;}}
     }
   }
 }
 assert(actual,'Could not find book path with off-book branches at 18+ discs');
 const overlayRaw=await t.requestEndgameForAi(actual.b,t.sideToMove(actual.moves));
 const afterOverlay=t.normalizeEndgameAnalysis(overlayRaw,actual.analysis.backgroundFallback);
 for(const line of actual.analysis.lines.filter(l=>l.exact)) {
   const merged=afterOverlay.lines.find(x=>x.move===line.move);
   assert(merged,`Missing book move ${line.move}`);
   assert.equal(merged.source,line.source);
   assert.equal(merged.evalWhite,line.evalWhite);
   assert.equal(merged.exact,true);
 }
 assert.equal(afterOverlay.lines.length,t.legalMovesFor(actual.b,t.sideToMove(actual.moves)).length);
 // The full-depth terminal WASM agrees with the independently generated oracle values.
 for(const bookLine of actual.analysis.lines.filter(l=>l.exact)) {
   const candidate=overlayRaw.lines.find(l=>l.move===bookLine.move);
   assert(candidate?.exact,`Missing exact WASM result for book move ${bookLine.move}`);
   assert.equal(candidate.evalWhite,bookLine.evalWhite,`WASM/Oracle exact score mismatch at ${bookLine.move}`);
 }
 console.log('REAL BOOK:',{discs:36-t.countsOf(actual.b).empty,exactBookMoves:actual.analysis.lines.filter(l=>l.exact).length,source:afterOverlay.source,candidates:afterOverlay.lines.length});
 console.log('PASS: 18-disc off-book WASM, all squares, oracle overlay, both engine labels, AI priority');
})().catch(err=>{console.error(err);process.exitCode=1});
