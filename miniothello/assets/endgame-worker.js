/* Exact 6x6 endgame WASM worker. No deadlines, budgets or estimates.
 * Every posted line is solved to a terminal score including forced passes.
 * A live board can terminate this isolated worker when the position changes.
 */
let wasmPromise;
async function loadEvaluator() {
  if (!wasmPromise) wasmPromise = (async () => {
    const response = await fetch(new URL('endgame-evaluator.wasm', self.location.href));
    if (!response.ok) throw new Error(`Endgame WASM HTTP ${response.status}`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
    return instance.exports;
  })();
  return wasmPromise;
}
function readSolved(wasm,side) {
  const count=wasm.result_count(),done=wasm.result_done();
  if (count<0 || count>36 || done<0 || done>count) throw Error('Invalid endgame result count');
  const moves=new Int8Array(wasm.memory.buffer,wasm.moves_ptr(),count);
  const scores=new Int16Array(wasm.memory.buffer,wasm.scores_ptr(),count);
  const exact=new Uint8Array(wasm.memory.buffer,wasm.exact_ptr(),count);
  const lines=[];
  for(let i=0;i<done;i++) {
    if(exact[i]!==1 || scores[i]<-36 || scores[i]>36) throw Error('Unsolved or invalid terminal score');
    lines.push({move:moves[i],evalWhite:side==='white'?scores[i]:-scores[i],
      exact:true,pv:[moves[i]]});
  }
  return {side,moves:Array.from(moves),lines,depth:wasm.result_depth(),nodes:wasm.result_nodes(),
    exact:done===count&&Boolean(wasm.result_exact()),completed:done,total:count};
}
self.addEventListener('message', async (event) => {
  const {type,token,key,board,side}=event.data||{};
  if(type!=='analyze')return;
  try {
    if(!Array.isArray(board)||board.length!==36||!['black','white'].includes(side))
      throw Error('Invalid exact endgame input');
    const wasm=await loadEvaluator();
    const input=new Int8Array(wasm.memory.buffer,wasm.board_ptr(),36);
    for(let i=0;i<36;i++)input[i]=board[i]==='black'||board[i]===1?1:
      board[i]==='white'||board[i]===-1?-1:0;
    const count=wasm.begin(side==='black'?1:-1);
    if(count<0||count>36)throw Error('Invalid exact endgame move count');
    // Every solve_next() is a full-depth, unbounded proof for ONE move.
    // No partial depth evaluations are produced, shown or cached.
    for(let i=0;i<count;i++){
      if(wasm.solve_next()!==i)throw Error('WASM solver did not finish candidate');
      self.postMessage({type:'update',token,key,result:readSolved(wasm,side)});
      // Let messages be painted between fully solved root candidates.
      await Promise.resolve();
    }
    const result=readSolved(wasm,side);
    if(!result.exact)throw Error('Endgame WASM failed to finish all candidates');
    self.postMessage({type:'done',token,key,result});
  } catch(error) {
    self.postMessage({type:'error',token,key,message:error?.message||String(error)});
  }
});
