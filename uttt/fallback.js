import {Game,LINES,POW} from './game.js';
// Clear, verifiable JS search fallback. WASM is the optimized native engine.
const WIN=30000, INF=32000,weights=[4,2,4,2,6,2,4,2,4];
let table=null;
function createTable(){
 if(table)return table;
 const t=new Array(19683);
 for(let code=0;code<19683;code++){
  let x=0,o=0,empty=0,val=0;
  for(let p=0;p<9;p++){
   const d=Math.floor(code/POW[p])%3;
   if(d===1){x|=1<<p;val+=weights[p];}else if(d===2){o|=1<<p;val-=weights[p];}else empty|=1<<p;
  }
  let status=0,tx=0,to=0;
  for(const m of LINES){if((x&m)===m)status=1;if((o&m)===m&&!status)status=2;}
  if(!status&&!empty)status=3;
  if(!status)for(const m of LINES){
    const xc=bitcount(x&m),oc=bitcount(o&m);
    if(!oc){val+=xc===2?30:xc===1?7:0;if(xc===2)tx|=m&empty;}
    if(!xc){val-=oc===2?30:oc===1?7:0;if(oc===2)to|=m&empty;}
  }
  t[code]={status,eval:val,tx,to,empty};
 }
 table=t;return t;
}
function bitcount(n){let i=0;while(n){n&=n-1;i++;}return i;}
export class JSEngine{
 constructor(){this.table=createTable();this.tt=new Map();this.n=0;this.budget=0;this.abort=false;this.candidates=[];this.pv=[];this.killers=Array.from({length:84},()=>[-1,-1]);this.history=new Int32Array(162);}
 from(game){this.game=game.clone();this.codes=new Int32Array(9);for(let m=0;m<9;m++)for(let p=0;p<9;p++)this.codes[m]+=game.cells[m*9+p]*POW[p];}
 make(m){const old=this.game.clone();const b=m/9|0;const code=this.codes[b];this.game.play(m);this.codes[b]+=old.turn*POW[m%9];return [old,b,code];}
 undo(u){this.game=u[0];this.codes[u[1]]=u[2];}
 eval(){const g=this.game;let v=0,x=0,o=0;const weights=[13,9,13,9,17,9,13,9,13];
  for(let b=0;b<9;b++){if(g.boards[b]===1){v+=weights[b]*14;x|=1<<b;}
   else if(g.boards[b]===2){v-=weights[b]*14;o|=1<<b;}
   else if(!g.boards[b])v+=this.table[this.codes[b]].eval*weights[b]/13;
  }
  for(const line of LINES){const xx=bitcount(x&line),oo=bitcount(o&line);if(!oo)v+=xx===2?280:xx===1?33:0;if(!xx)v-=oo===2?280:oo===1?33:0;}
  return (g.turn===1?1:-1)*v|0;
 }
 direct(){const g=this.game,side=g.turn;let claimed=0;for(let b=0;b<9;b++)if(g.boards[b]===side)claimed|=1<<b;
  for(const m of g.legal()){const b=m/9|0,p=m%9;if((side===1?this.table[this.codes[b]].tx:this.table[this.codes[b]].to)&(1<<p)){
   const bits=claimed|1<<b;if(LINES.some(l=>(bits&l)===l))return m;
  }}return -1;
 }
 key(){return `${this.codes.join(',')}.${this.game.turn}.${this.game.target}`;}
 order(moves,ttmove,ply){const side=this.game.turn;
  return moves.sort((a,b)=>this.ord(b,side,ttmove,ply)-this.ord(a,side,ttmove,ply));}
 ord(m,side,ttmove,ply){if(m===ttmove)return 1e9;const b=m/9|0,p=m%9;const t=this.table[this.codes[b]];
  let v=weights[p]*12+(b===4?140:0)+this.history[side*81+m];
  if((side===1?t.tx:t.to)&(1<<p))v+=50000;
  if(this.killers[ply]?.includes(m))v+=1500;
  return v;
 }
 searchNode(depth,alpha,beta,ply){
  if(++this.n>this.budget){this.abort=true;return 0;}
  const g=this.game;
  if(g.winner)return g.winner===3?0:g.winner===g.turn?WIN-ply:-WIN+ply;
  const immediate=this.direct();if(immediate>=0)return WIN-ply-1;
  if(depth<=0||ply>=81)return this.eval();
  const k=this.key(),entry=this.tt.get(k),oldAlpha=alpha;
  if(entry&&entry.d>=depth){if(entry.flag==='exact')return entry.v;if(entry.flag==='lower'&&entry.v>=beta)return entry.v;if(entry.flag==='upper'&&entry.v<=alpha)return entry.v;}
  const moves=this.order(g.legal(),entry?.move??-1,ply);
  let best=-INF,bestMove=-1;
  for(let i=0;i<moves.length;i++){
   const m=moves[i],u=this.make(m);
   let v=-this.searchNode(depth-1,-beta,-alpha,ply+1);
   this.undo(u);if(this.abort)return 0;
   if(v>best){best=v;bestMove=m;}if(v>alpha)alpha=v;
   if(alpha>=beta){const idx=this.game.turn*81+m;this.history[idx]+=depth*depth;
    this.killers[ply]=[m,this.killers[ply][0]];break;
   }
  }
  if(this.tt.size>70000)this.tt.clear();
  this.tt.set(k,{d:depth,v:best,move:bestMove,flag:best<=oldAlpha?'upper':best>=beta?'lower':'exact'});
  return best;
 }
 search(game,depth,budget){depth=Math.min(depth,81-game.moves.length);this.from(game);this.n=0;this.budget=budget;this.abort=false;
  const k=this.key(),ttmove=this.tt.get(k)?.move??-1;
  const moves=this.order(this.game.legal(),ttmove,0);let bestMove=-1,bestVal=-INF,alpha=-INF;this.candidates=[];
  for(const m of moves){const oldAlpha=alpha,u=this.make(m);const val=-this.searchNode(depth-1,-INF,INF,1);
   this.undo(u);if(this.abort)return {complete:false,nodes:this.n};
   this.candidates.push({move:m,score:val,bound:false});
   if(val>bestVal){bestVal=val;bestMove=m;}alpha=Math.max(alpha,val);
  }
  this.tt.set(k,{d:depth,v:bestVal,move:bestMove,flag:'exact'});
  this.pv=[bestMove];const undo=[];
  if(bestMove>=0){let m=bestMove;for(let i=0;i<depth&&m>=0&&this.game.isLegal(m);i++){
   undo.push(this.make(m));if(this.game.winner)break;m=this.tt.get(this.key())?.move??-1;this.pv.push(m);
  }}while(undo.length)this.undo(undo.pop());this.pv=this.pv.filter(m=>m>=0);
  return {complete:true,nodes:this.n,best:bestMove,score:bestVal,pv:this.pv,candidates:this.candidates};
 }
 // Tri-state exact proof: 1 yes, 0 no within horizon, -1 budget exhausted.
 prove(game,plies,budget){this.from(game);this.pn=0;this.pb=budget;this.proofTT=new Map();let best=-1;const attacker=this.game.turn;
  const dfs=(d)=>{if(++this.pn>this.pb)return -1;
   if(this.game.winner)return this.game.winner===attacker?1:0;
   if(!d)return 0;
   const key=this.key()+':'+d,hit=this.proofTT.get(key);if(hit!==undefined)return hit;
   const isAttacker=this.game.turn===attacker;
   const imm=this.direct();if(imm>=0){if(d===plies&&isAttacker)best=imm;return isAttacker?1:0;}
   let unknown=false;
   const moves=this.order(this.game.legal(),-1,0);
   for(const m of moves){const u=this.make(m),r=dfs(d-1);this.undo(u);
    if(r===-1)unknown=true;
    else if((isAttacker&&r===1)||(!isAttacker&&r===0)){
      this.proofTT.set(key,r);if(d===plies)best=m;return r;
    }
   }
   if(unknown)return -1;const r=isAttacker?0:1;this.proofTT.set(key,r);return r;
  };
  const r=dfs(plies);return {result:r,nodes:this.pn,best};
 }
}
