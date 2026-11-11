'use strict';
const fs=require('fs'),path=require('path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),assets=path.join(root,'assets');
const newW=new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(path.join(assets,'egaroucid-6x6.wasm'))),{env:{notify(){}}}).exports;
const exactW=new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(path.join(assets,'endgame-evaluator.wasm'))),{}).exports;
const D=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
function flips(b,s,p){if(b[p])return [];let out=[],x=p%6,y=p/6|0;
 for(const [dx,dy] of D){let ray=[],xx=x+dx,yy=y+dy;while(xx>=0&&xx<6&&yy>=0&&yy<6){const n=yy*6+xx;if(b[n]===-s)ray.push(n);else {if(b[n]===s&&ray.length)out.push(...ray);break;}xx+=dx;yy+=dy;}}
 return out;}
function legal(b,s){const a=[];for(let i=0;i<36;i++)if(flips(b,s,i).length)a.push(i);return a;}
function move(b,s,m){if(m===-1)return b.slice();let c=b.slice();c[m]=s;for(const x of flips(b,s,m))c[x]=s;return c;}
function randomState(seed,pieces){const b=Array(36).fill(0);b[14]=-1;b[15]=1;b[20]=1;b[21]=-1;let side=1;
 for(let i=4;i<pieces;i++){let ms=legal(b,side);if(!ms.length){if(!legal(b,-side).length)return null;side=-side;i--;continue;}
  seed=(Math.imul(seed,1664525)+1013904223)|0;const pick=ms[(seed>>>0)%ms.length];b.splice(0,36,...move(b,side,pick));side=-side;
 }
 return {b,side};}
function compare(b,side,withExact=true){const ni=new Int8Array(newW.memory.buffer,newW.board_ptr(),36);ni.set(b);
 const ei=new Int8Array(exactW.memory.buffer,exactW.board_ptr(),36);ei.set(b);
 const count=newW.begin(side,1);
 const cnt2=exactW.begin(side);
 const nm=Array.from(new Int8Array(newW.memory.buffer,newW.moves_ptr(),count));
 const em=Array.from(new Int8Array(exactW.memory.buffer,exactW.moves_ptr(),cnt2));
 assert.deepEqual(nm,em,'WASM legal candidates');
 const empties=b.filter(x=>!x).length;
 if(empties>0){newW.search_depth(1);assert.equal(newW.result_depth(),1);assert.equal(newW.result_exact(),0);}
 const first=Array.from(new Int32Array(newW.memory.buffer,newW.scores_ptr(),count));
 for(let d=2;d<=empties;d++){
   newW.search_depth(d);
   assert.equal(newW.result_depth(),d,'iteration depth must really grow');
 }
 assert.equal(newW.result_exact(),1,'terminal exact flag');
 const ns=Array.from(new Int32Array(newW.memory.buffer,newW.scores_ptr(),count));
 const ex=Array.from(new Uint8Array(newW.memory.buffer,newW.exact_ptr(),count));
 assert(ex.every(x=>x===1));
 if(withExact)for(let i=0;i<cnt2;i++){
   assert.equal(exactW.solve_next(),i);
   const s=new Int16Array(exactW.memory.buffer,exactW.scores_ptr(),cnt2)[i];
   assert.equal(ns[i],s*100,`score mismatch move=${nm[i]} side=${side} pieces=${36-empties}`);
   assert((ns[i]/100)%2===0,'even exact final score');
 }
 return {count,empties,nodes:newW.result_nodes(),sampleFirst:first.slice(0,4),final:ns};}
const start=Date.now();let checked=0,candidates=0;
for(let seed=1;seed<=22;seed++){
 const pieces=seed<=2?20:seed<=5?24:seed<=11?27:seed<=17?30:33;
 const st=randomState(seed*91123+16,pieces);
 if(!st)continue;
 const result=compare(st.b,st.side);
 checked++;candidates+=result.count;
 if(seed<7)console.log('POSITION',seed,'discs',pieces,'moves',result.count,'nodes',Math.round(result.nodes),'timeMS',Date.now()-start);
}
// A larger randomized late-game corpus catches subtle wrong alpha-beta/TT
// cutoffs, pass handling and stable-region bounds.
for(let seed=35;seed<175;seed++) {
 const pieces=seed%3===0?28:seed%3===1?31:34;
 const st=randomState(seed*73319+300,pieces);
 if(!st)continue;
 const result=compare(st.b,st.side);
 checked++;candidates+=result.count;
}
// Forced pass: same discs, search with both turn orders. The next side can
// have no legal move even when its opponent does have legal moves.
let forcedPass=0;
for(let seed=1;seed<350 && forcedPass<4;seed++) {
 const st=randomState(seed*9949+42,31);
 if(!st)continue;
 for(const side of [-1,1]) {
   if(legal(st.b,side).length===0 && legal(st.b,-side).length>0) {
     const result=compare(st.b,side);
     assert.equal(result.count,1);
     forcedPass++;
   }
 }
}
assert(forcedPass>0,'No forced-pass test position found');
// Before 18 occupied discs, run Egaroucid WASM through a sequence of deepening
// iterations, without attempting a potentially astronomical opening solve.
{
 const st=randomState(982934,17);assert(st);
 new Int8Array(newW.memory.buffer,newW.board_ptr(),36).set(st.b);
 const count=newW.begin(st.side,1);
 const moves=Array.from(new Int8Array(newW.memory.buffer,newW.moves_ptr(),count));
 assert.deepEqual(moves,legal(st.b,st.side));
 for(let d=1;d<=7;d++) {
   assert.equal(newW.search_depth(d),d);
   assert.equal(newW.result_exact(),0);
 }
}
// Drawn / terminal positions (no legal move) and forced pass.
const full=Array(36).fill(1);full[7]=-1;
for(const side of [1,-1]){
 new Int8Array(newW.memory.buffer,newW.board_ptr(),36).set(full);
 assert.equal(newW.begin(side,1),0);
}
console.log('VERIFIED',JSON.stringify({positions:checked,exactCandidateScores:candidates,forcedPassPositions:forcedPass,elapsedMs:Date.now()-start}));
