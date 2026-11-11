'use strict';
const fs=require('fs'), vm=require('vm'), path=require('path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'), assets=path.join(root,'assets');
const fakeDom=()=>({innerHTML:'',textContent:'',value:'',dataset:{},style:{},hidden:false,classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(){},addEventListener(){},appendChild(){},remove(){},querySelector(){return fakeDom()},children:[],focus(){}});
const elements=new Map();const doc={querySelector(s){if(!elements.has(s))elements.set(s,fakeDom());return elements.get(s)},createElement(){return fakeDom()}};
class VMWorker {
  constructor(url){this.url=url;this.listeners={}; this.onmessage=null;this.onerror=null;
    if(url.endsWith('black-oracle-worker.js')){
      const workerSelf={location:{href:'http://localhost/assets/black-oracle-worker.js'},addEventListener:(t,h)=>{if(t==='message') this.onWorkerMessage=h},postMessage:(m)=>{queueMicrotask(()=>{for(const f of this.listeners.message||[])f({data:m});this.onmessage?.({data:m})})}};
      const fetch=async (url)=>{const b=fs.readFileSync(path.join(assets,path.basename(url.toString())));return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
      vm.runInNewContext(fs.readFileSync(path.join(assets,'black-oracle-worker.js'),'utf8'),{self:workerSelf,fetch,WebAssembly,URL,Int8Array,Uint8Array,Uint32Array,DataView,Promise,Array,Boolean,Number,String,Math,Error,console});
    }
  }
  addEventListener(type,fn){(this.listeners[type]??=[]).push(fn)}
  postMessage(msg){if(this.onWorkerMessage){this.onWorkerMessage({data:msg}).catch(e=>{console.error('worker err',e)})};}
  terminate(){}
}
const ctx={document:doc,window:{addEventListener(){}},localStorage:{getItem(){return null},setItem(){}},Worker:VMWorker,console,fetch:()=>{},WebAssembly,URL,Uint8Array,Int8Array,Date,Math,Number,Array,Promise,BigInt,Map,Set,JSON,Intl,Math,Boolean,Error,setTimeout,clearTimeout,queueMicrotask};
vm.createContext(ctx);
let js=fs.readFileSync(path.join(root,'app.js'),'utf8');
js=js.slice(0,js.lastIndexOf('\nloadState();'));
vm.runInContext(js+'\n;globalThis.t={analyzePosition,createInitialBoard,applyMoveToBoard,legalMovesFor,sideToMove,lookupOracleInOrder,requestBlackOraclePosition,requestWhiteOracleBookPosition,mergeOracleWithFallback};',ctx);
const t=ctx.t;
function b(moves){let board=t.createInitialBoard();moves.forEach((m,i)=>board=t.applyMoveToBoard(board,i%2?'white':'black',m));return board}
function view(r){return {source:r.source,side:r.side,best:r.bestMove,exact:r.exact,cache:r.cache,lines:r.lines.map(l=>[l.move,l.evalWhite,l.source,l.exact,l.pending||false]),background:r.backgroundFallback?.type}}
async function assertPosition(moves,opts={}){
 const board=b(moves),legal=t.legalMovesFor(board,t.sideToMove(moves));
 const result=await t.analyzePosition(moves,board);
 assert.equal(result.lines.length,legal.length,'every legal move must be represented, even off-book');
 assert.deepEqual(result.lines.map(l=>l.move).sort((a,b)=>a-b),legal.slice().sort((a,b)=>a-b));
 assert(result.lines.find(l=>l.move===result.bestMove)?.exact);
 console.log('CASE',moves.join(','),JSON.stringify(view(result)));
 if(opts.source)assert.equal(result.source,opts.source);
 if(opts.complete!==undefined)assert.equal(Boolean(result.backgroundFallback),!opts.complete);
 return result;
}
(async()=>{
 let r=await assertPosition([],{source:'Black Oracle DB',complete:true});
 assert.equal(r.lines.length,4);assert(r.lines.every(l=>l.evalWhite===4 && l.exact));
 r=await assertPosition([13],{source:'Black Oracle DB',complete:true});assert.equal(r.lines.length,3);
 r=await assertPosition([13,7],{source:'Black Oracle DB',complete:false});
 assert(r.lines.some(l=>l.pending));
 r=await assertPosition([13,9],{source:'Black Oracle DB',complete:true});
 assert(r.lines.some(l=>l.source==='White Oracle DB'));
 r=await assertPosition([13,9,22],{source:'White Oracle DB',complete:false});
 let notHit=await t.lookupOracleInOrder([13,7,1]);assert.equal(notHit,null);
 try{await t.analyzePosition([13,7,1],b([13,7,1]));assert.fail('off both must switch to fallback')}catch(e){assert.match(e.message,/outside both Oracle/)}
 // White book: all Black variations, single White optimal response, recursively.
 let whiteChecked=0, whiteChildren=0, blackChecked=0;
 for(let seed=0;seed<45;seed++) {
   let moves=[[8,13,22,27][seed%4]];
   for(let depth=1;depth<16;depth++) {
     const side=t.sideToMove(moves),actualLegal=t.legalMovesFor(b(moves),side),legal=actualLegal.length?actualLegal:[-1];
     const w=await t.requestWhiteOracleBookPosition(moves);
     if(!w) break;
     if(side==='white') {
       assert(legal.includes(w.bestMove));
       const after=await t.requestWhiteOracleBookPosition(moves.concat(w.bestMove));
       if(after?.options?.length) {
         assert.equal(w.scoreWhite, -Math.max(...after.options.map(v=>v.blackValue)));
       }
       whiteChecked++;
       moves.push(w.bestMove);
     } else {
       const wMoves=w.options.map(v=>v.move).sort((a,b)=>a-b);
       assert.equal(JSON.stringify(wMoves),JSON.stringify(legal.slice().sort((a,b)=>a-b)));
       for(const candidate of w.options){
         const next=await t.requestWhiteOracleBookPosition(moves.concat(candidate.move));
         assert(next && next.side==='white');
         assert.equal(next.scoreWhite,-candidate.blackValue);
         whiteChildren++;
       }
       moves.push(w.options[(seed*7+depth*3)%w.options.length].move);
     }
   }
 }
 for(let seed=0;seed<35;seed++) {
   let moves=[];
   for(let depth=0;depth<19;depth++) {
     const side=t.sideToMove(moves),actualLegal=t.legalMovesFor(b(moves),side),legal=actualLegal.length?actualLegal:[-1];
     const black=await t.requestBlackOraclePosition(moves);
     if(!black)break;
     if(side==='black') {
       assert(legal.includes(black.bestMove));
       moves.push(black.bestMove);
     } else {
       const actual=black.options.map(v=>v.move).sort((a,b)=>a-b);
       assert.equal(JSON.stringify(actual),JSON.stringify(legal.slice().sort((a,b)=>a-b)));
       for(const option of black.options){
         const child=await t.requestBlackOraclePosition(moves.concat(option.move));
         assert(child && child.side==='black');
         assert.equal(child.scoreBlack,option.blackValue);
         blackChecked++;
       }
       moves.push(black.options[(seed*11+depth)%black.options.length].move);
     }
   }
 }
 console.log('Deep tree checked:',{whiteChecked,whiteChildren,blackChecked});
 assert(whiteChecked>150 && whiteChildren>250 && blackChecked>150);
 const mixed=await assertPosition([13,7],{complete:false});
 const ordinary={side:'black',bestMove:10,source:'JS fallback iterative',evalWhite:0,lines:mixed.lines.map(l=>({move:l.move,evalWhite:11,source:'ordinary',exact:false}))};
 const merged=t.mergeOracleWithFallback(ordinary,mixed.backgroundFallback);
 assert(merged.lines.find(l=>l.move===mixed.bestMove).exact);
 assert(merged.lines.filter(l=>l.move!==mixed.bestMove).every(l=>l.source==='ordinary' || l.exact));
 console.log('PASS: real WASM Black book + real LOUDS White book, four root squares, both sides, overlap, mixed fallback and book miss.');
})().catch(e=>{console.error(e);process.exit(1)})
