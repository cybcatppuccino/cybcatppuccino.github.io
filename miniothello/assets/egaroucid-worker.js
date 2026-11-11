/* Pure WebAssembly 6x6 analysis worker, Egaroucid-inspired algorithms.
 * No JS search, no search deadline, no partial-depth score publication.
 * A dedicated Web Worker is terminated by the host on position change.
 */
let kernel=null, context=null;
const TICK_MS=500;
let lastPublish=0;

function scored(w, boardSide, mode) {
  const count=w.result_count();
  if(count<0 || count>36)throw new Error('Invalid Egaroucid WASM candidate count');
  const depth=w.result_depth();
  const complete=Boolean(w.result_exact());
  const moves=Array.from(new Int8Array(w.memory.buffer,w.moves_ptr(),count));
  const values=Array.from(new Int32Array(w.memory.buffer,w.scores_ptr(),count));
  const exacts=Array.from(new Uint8Array(w.memory.buffer,w.exact_ptr(),count));
  const sideSign=boardSide==='white'?1:-1;
  const lines=moves.map((move,i)=>({
    move,evalWhite:depth?sideSign*values[i]/100:null,
    source:complete?'Egaroucid WASM exact':'Egaroucid WASM eval',
    pv:[move],exact:depth>0&&Boolean(exacts[i]),pending:depth===0
  }));
  lines.sort((a,b)=>a.pending!==b.pending?(a.pending?1:-1):
    (boardSide==='white'?b.evalWhite-a.evalWhite:a.evalWhite-b.evalWhite));
  const best=lines.find(line=>!line.pending) || null;
  return {
    side:boardSide,lines,bestMove:best?.move??null,
    evalWhite:best?.evalWhite??null,evalExact:Boolean(best?.exact),
    exact:complete,source:complete?'Egaroucid 6x6 exact WASM':'Egaroucid 6x6 WASM',
    cache:'WASM',wasmEgaroucid:true,
    depth,searchingDepth:w.searching_depth()||null,
    nodes:w.result_nodes(),terminal:false,
    mode:'6x6 Egaroucid search'
  };
}

function publish(type='update',force=false) {
  if(!context||!kernel)return;
  const now=Date.now();
  if(!force && now-lastPublish<TICK_MS)return;
  lastPublish=now;
  const result=scored(kernel,context.side,context.mode);
  self.postMessage({type,key:context.key,token:context.token,result});
}

function notify() { publish('update'); }
async function loadKernel(){
  if(kernel)return kernel;
  const response=await fetch(new URL('egaroucid-6x6.wasm',self.location.href));
  if(!response.ok)throw Error(`Egaroucid WASM HTTP ${response.status}`);
  const {instance}=await WebAssembly.instantiate(await response.arrayBuffer(),{env:{notify}});
  kernel=instance.exports;
  return kernel;
}
self.onmessage=async({data})=>{
  if(data?.type!=='start')return;
  const {key,token,board,side}=data;
  try {
    const w=await loadKernel();
    if(!Array.isArray(board)||board.length!==36||!['black','white'].includes(side))throw Error('Invalid 6x6 input');
    const discs=board.filter(Boolean).length;
    if (discs >= 18) throw Error('18+ discs must use the exact endgame WASM worker');
    const mode='advanced';
    context={key,token,side,mode};
    const dst=new Int8Array(w.memory.buffer,w.board_ptr(),36);
    for(let i=0;i<36;i++)dst[i]=board[i]==='black'?1:board[i]==='white'?-1:0;
    const candidates=w.begin(side==='black'?1:-1,1);
    if(!candidates){
      self.postMessage({type:'done',key,token,result:{side,lines:[],bestMove:null,exact:true,terminal:true,depth:0,source:'Egaroucid 6x6 WASM'}});
      return;
    }
    lastPublish=Date.now();
    const maxDepth=w.result_empties();
    for(let depth=1;depth<=maxDepth;depth++) {
      // WASM runs synchronously, but imported notify() can post 500ms
      // heartbeats during each iteration. They expose the previous fully
      // completed depth, never a mixture of partial new depth scores.
      w.search_depth(depth);
      publish('update',false);
    }
    publish('done',true);
  }catch(error){
    self.postMessage({type:'error',key,token,message:String(error?.stack||error)});
  }
};
