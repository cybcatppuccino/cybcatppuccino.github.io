'use strict';
/* Independent grid-rule verifier for exact WASM (no estimates, no cutoffs). */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const wasm=fs.readFileSync(path.join(__dirname,'../assets/endgame-evaluator.wasm'));
const e=new WebAssembly.Instance(new WebAssembly.Module(wasm),{}).exports;
const input=new Int8Array(e.memory.buffer,e.board_ptr(),36);
const DIRS=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
function flips(b,s,p){if(b[p])return [];let out=[],x=p%6,y=p/6|0;
 for(const [dx,dy] of DIRS){let found=[],xx=x+dx,yy=y+dy;
  while(xx>=0&&xx<6&&yy>=0&&yy<6&&b[yy*6+xx]===-s){found.push(yy*6+xx);xx+=dx;yy+=dy;}
  if(found.length&&xx>=0&&xx<6&&yy>=0&&yy<6&&b[yy*6+xx]===s)out.push(...found);
 }return out;
}
function legal(b,s){const a=[];for(let i=0;i<36;i++)if(flips(b,s,i).length)a.push(i);return a;}
function move(b,s,m){if(m===-1)return b.slice();let c=b.slice();c[m]=s;for(const x of flips(b,s,m))c[x]=s;return c;}
function terminal(b,s){let diff=b.reduce((v,x)=>v+(x===s?1:x===-s?-1:0),0),empty=b.filter(x=>!x).length;return diff>0?diff+empty:diff<0?diff-empty:0;}
const referenceCache=new Map();
function brute(b,s){const key=`${s}:${b.join('')}`;if(referenceCache.has(key))return referenceCache.get(key);
 let m=legal(b,s);if(!m.length){const v=legal(b,-s).length?-brute(b,-s):terminal(b,s);referenceCache.set(key,v);return v;}
 let best=-100;for(const p of m)best=Math.max(best,-brute(move(b,s,p),-s));referenceCache.set(key,best);return best;
}
function randomState(seed,pieces){const b=Array(36).fill(0);b[14]=-1;b[15]=1;b[20]=1;b[21]=-1;let side=1;
 for(let i=4;i<pieces;i++){let m=legal(b,side);if(!m.length){if(!legal(b,-side).length)return null;side=-side;i--;continue;}
  seed=(Math.imul(seed,1664525)+1013904223)|0;const pick=m[(seed>>>0)%m.length];b.splice(0,36,...move(b,side,pick));side=-side;
 }return {b,side};
}
function query(b,side){input.set(b);const cnt=e.begin(side);const arr=Array.from({length:cnt},()=>0);
 const moves=[...new Int8Array(e.memory.buffer,e.moves_ptr(),cnt)];
 assert.deepEqual(moves,legal(b,side).length?legal(b,side):legal(b,-side).length?[-1]:[],'6x6 legal moves mismatch');
 for(let i=0;i<cnt;i++){
  assert.equal(e.solve_next(),i);
  assert.equal(e.result_done(),i+1);
  assert.deepEqual([...new Uint8Array(e.memory.buffer,e.exact_ptr(),cnt)],Array.from({length:cnt},(_,j)=>j<=i?1:0));
  arr[i]=new Int16Array(e.memory.buffer,e.scores_ptr(),cnt)[i];
  assert(Number.isInteger(arr[i]) && arr[i]>=-36 && arr[i]<=36);
 }
 assert.equal(e.solve_next(),-1);assert.equal(e.result_exact(),1);
 return {moves,scores:arr,nodes:e.result_nodes()};
}
(async()=>{
 const started=Date.now();let reference=0,checked=0,passCases=0,disc18=0;
 for(let seed=1;seed<=36;seed++){
  let pieces=seed===1?18:(seed<=4?20+seed%3:(seed<=12?23+seed%3:28+seed%8));
  const state=randomState(seed*9923+50,pieces);if(!state)continue;
  const {b,side}=state,wasm=query(b,side);
  for(let i=0;i<wasm.moves.length;i++){
   if(pieces>=28){const expected=-brute(move(b,side,wasm.moves[i]),-side);
     assert.equal(wasm.scores[i],expected || 0,`Independent final score at seed ${seed} move ${wasm.moves[i]}`);reference++;
   }
  }
  checked++;disc18+=(pieces===18);
  if(seed%6===0) console.log(`verified ${seed} positions, latest ${pieces} discs, ${wasm.nodes} WASM nodes`);
  if(wasm.moves.includes(-1))passCases++;
 }
 // Explicit final pass when neither player has a move: zero candidate lines.
 const full=Array(36).fill(1);full[7]=-1;
 assert.deepEqual(query(full,1).moves,[]);
 // Empty-square allocation is verified by independent reference tests above.
 console.log(`PASS exact endgame WASM: ${checked} random positions; ${disc18} at 18 discs; ${reference} independently brute-forced candidate scores; ${passCases} forced-pass roots; ${Date.now()-started}ms`);
})().catch(error=>{console.error(error);process.exitCode=1});
