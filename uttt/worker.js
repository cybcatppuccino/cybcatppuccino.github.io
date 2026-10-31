import {Game} from './game.js';
import {JSEngine} from './fallback.js';
import {Stage2Net} from './stage2_net.js';
import {Stage2MCTS} from './stage2_mcts.js';
import {endgameProfile} from './endgame.js';
import {RollingAllocator, adaptiveShares, PVScout} from './proof_scout.js';

let wasm=null,loading=null,js=null,activeJob=null,cachedStage2=null;
let sliceEnd=Infinity,hardEnd=Infinity,nextProgress=0,currentId=0,overallNodes=0,workingDepth=0;
let previousKey='',previousDepth=0,previousReport=null,proofTreeDirty=false;
let mlSessionActive=false;
// Yield to the Worker event loop without repeatedly paying the 4ms timer clamp
// in browsers. This still delivers cancel/UI messages between search batches.
const sleep=(()=>{
 if(typeof MessageChannel!=='function')return ()=>new Promise(r=>setTimeout(r,0));
 const channel=new MessageChannel(),waiting=[];
 channel.port1.onmessage=()=>{const done=waiting.shift();if(done)done();};
 return ()=>new Promise(done=>{waiting.push(done);channel.port2.postMessage(0);});
})();
function countOpenSquares(game){let n=0;for(let b=0;b<9;b++)if(!game.boards[b])for(let i=b*9;i<b*9+9;i++)n+=game.cells[i]===0;return n;}
async function load(){
 if(!loading)loading=(async()=>{
  const r=await fetch(new URL('./apex.wasm',import.meta.url));if(!r.ok)throw Error('WASM '+r.status);
  const env={apex_poll(){const now=performance.now();
   if(now>=nextProgress){nextProgress=now+175;self.postMessage({type:'progress',id:currentId,elapsed:now-startClock,nodes:overallNodes+(wasm?wasm.apex_nodes():0)});}
   return now>=sliceEnd||now>=hardEnd?1:0;
  }};
  const {instance}=await WebAssembly.instantiate(await r.arrayBuffer(),{env});wasm=instance.exports;wasm.apex_init();return wasm;
 })();return loading;
}
let startClock=0;
function newGame(m){const g=new Game();g.cells.set(m.cells);g.turn=m.turn;g.target=m.target;for(let i=0;i<9;i++)g.boards[i]=g.localStatus(i);g.winner=g.globalStatus();g.moves=Array(g.cells.reduce((s,v)=>s+!!v,0)).fill(-1);return g;}
function keyOf(m){return `${m.turn}/${m.target}/${m.cells.join('')}`;}
function setWasm(g){new Uint8Array(wasm.memory.buffer,wasm.apex_input_ptr(),81).set(g.cells);if(!wasm.apex_set_position(g.turn,g.target))throw Error('Invalid game state');}
function report(depth,elapsed){
 const solved=!!wasm.apex_root_solved(),candidates=[];
 for(let i=0;i<wasm.apex_root_count();i++){
  const bound=!!wasm.apex_root_bound(i);
  candidates.push({move:wasm.apex_root_move(i),score:wasm.apex_root_score(i),bound,solved:solved&&!bound});
 }
 candidates.sort((a,b)=>b.score-a.score);
 return {depth,best:wasm.apex_best(),score:wasm.apex_score(),nodes:wasm.apex_nodes(),elapsed,candidates,solved,
  v12:{dominatedPruned:wasm.apex_dominated_pruned?.()||0,rootResumeHits:wasm.apex_root_resume_hits?.()||0,
   internalProofAttempts:wasm.apex_oracle_attempts?.()||0,internalProofSuccesses:wasm.apex_oracle_success?.()||0,
   internalProofNodes:wasm.apex_oracle_nodes?.()||0}};
}
// A root certificate is NOT automatically a certificate for every move.
// Only full-horizon exact child scores, individually proved move WDL, and
// mathematically entailed consequences (root strictly lost -> every move loses)
// can produce 100% bars. Other legal cells remain distinctly unproved.
function certifiedCandidates(game,r){
 const legal=game.legal(),legalSet=new Set(legal);
 const source=(r?.candidates||[]).filter(c=>legalSet.has(c.move));
 if(!r?.solved)return source.map(c=>({...c}));
 const map=new Map(source.map(c=>[c.move,{...c}]));
 const ordered=source.map(c=>c.move).concat(legal.filter(m=>!map.has(m)));
 const outcome=r?.solved?(r.score>29872?1:r.score< -29872?-1:r.score===0?0:null):null;
 const proofKind='WDL (distance unproven)';
 return ordered.map(move=>{
  const c=map.get(move)||{move,score:null,bound:false,solved:false};
  if(c.bound&&!c.solved){c.score=null;c.bound=false;}
  if(outcome===-1&&!c.solved){c.score=-29999;c.solved=true;c.proofKind=proofKind;}
  else if(outcome!==null&&move===r.best&&!c.solved){
   c.score=outcome===0?0:outcome*29999;c.solved=true;c.proofKind=proofKind;
  }
  return c;
 });
}
function attachCertifiedCandidates(game,r){return {...r,candidates:certifiedCandidates(game,r)};}
function mergeMoveCertificates(exactCandidates=[],provenCandidates=[]){
 const map=new Map(provenCandidates.filter(c=>c.solved).map(c=>[c.move,c]));
 for(const c of exactCandidates)if(c.solved)map.set(c.move,c);
 return exactCandidates.map(c=>c.solved?c:(map.get(c.move)||c)).concat(
  [...map.values()].filter(c=>!exactCandidates.some(x=>x.move===c.move)));
}

// Use proof-TT reuse to certify actual child positions without guessing their
// mate distance. The bounded pass never holds back the primary search when
// no meaningful wall time remains.
function certifyRootChildren(game,r,allocationMs,id){
 let nodes=0,checked=0,proved=0;
 if(!r.solved||!wasm?.apex_proof_move_wdl||allocationMs<4)return {result:attachCertifiedCandidates(game,r),nodes,checked,proved};
 const start=performance.now(),deadline=Math.min(hardEnd-3,start+allocationMs);
 const candidates=certifiedCandidates(game,r),byMove=new Map(candidates.map(c=>[c.move,c]));
 const pending=candidates.filter(c=>!c.solved),order=pending.sort((a,b)=>(b.move===r.best)-(a.move===r.best)||
  (Number.isFinite(b.score)?b.score:-32001)-(Number.isFinite(a.score)?a.score:-32001));
 for(const c of order){
  if(activeJob!==id||performance.now()>deadline-3)break;
  const now=performance.now(),available=deadline-now;
  sliceEnd=deadline;nextProgress=now+130;
  const limit=Math.max(1500,Math.min(1800000,Math.floor(4100*Math.max(2,available)*.82)));
  const result=wasm.apex_proof_move_wdl(c.move,limit);
  nodes+=wasm.apex_proof_nodes();checked++;
  if(result!==2){
   const target=byMove.get(c.move);target.score=result===0?0:result*29999;
   target.solved=true;target.proofKind='WDL (distance unproven)';target.bound=false;proved++;
  }
 }
 return {result:{...r,candidates},nodes,checked,proved};
}

// Model-led hybrid search: Stage2 PUCT is the primary decision/evaluation engine.
// PVS is a separately-scaled verifier for tactics, policy disagreements and exact endgames.
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function stage2PolicyStats(snapshot) {
 const ranked=(snapshot?.candidates||[]).filter(c=>Number.isFinite(c.prior)&&c.prior>0).slice()
  .sort((a,b)=>b.prior-a.prior);
 const n=ranked.length||1,sum=ranked.reduce((v,c)=>v+c.prior,0)||1;
 let entropy=0;for(const c of ranked){const p=c.prior/sum;if(p>0)entropy-=p*Math.log(p);}
 return {ranked,entropy:n>1?entropy/Math.log(n):0,
  topGap:n>1?ranked[0].prior/sum-ranked[1].prior/sum:1,
  topRatio:n>1?ranked[0].prior/Math.max(1e-12,ranked[1].prior):Infinity};
}
function writeStage2Policy(snapshot) {
 const ptr=wasm.apex_root_policy_ptr?wasm.apex_root_policy_ptr():wasm.apex_proof_prior_ptr();
 const out=new Int32Array(wasm.memory.buffer,ptr,81);out.fill(0);
 const stats=stage2PolicyStats(snapshot),top=stats.ranked[0]?.prior||1;
 for(const c of stats.ranked){
  const rel=Math.log(Math.max(1e-12,c.prior/top));
  // Bounded ordering hint only. Native tactical bonuses and TT moves remain ahead.
  out[c.move]=Math.max(0,Math.min(30000,Math.round(30000+5000*rel)));
 }
 return stats;
}
function writeVerifyMoves(moves) {
 if(!wasm.apex_root_verify_ptr)return;
 const out=new Uint8Array(wasm.memory.buffer,wasm.apex_root_verify_ptr(),81);out.fill(0);
 for(const m of moves)if(Number.isInteger(m)&&m>=0&&m<81)out[m]=1;
}
function pvsUncertainty(pvs,snapshot,bestChanges,puctBestChanges,policyStats,rootLegal) {
 const exact=(pvs?.candidates||[]).filter(c=>!c.bound).sort((a,b)=>b.score-a.score);
 const puctRank=(snapshot?.candidates||[]).slice().sort((a,b)=>b.visits-a.visits||b.prior-a.prior);
 const pvsTop=new Set(exact.slice(0,3).map(c=>c.move));
 const puctTop=new Set(puctRank.slice(0,5).map(c=>c.move));
 const policyTop=policyStats.ranked.slice(0,5).map(c=>c.move);
 const pvsBest=exact[0]?.move??-1,puctBest=puctRank[0]?.move??snapshot?.best??-1;
 const policyBest=policyTop[0]??-1;
 const pvsGap=exact.length>1?exact[0].score-exact[1].score:Infinity;
 const pvsClose=pvsGap<Math.max(28,Math.abs(exact[0]?.score||0)*.075);
 const puctVisits=puctRank.reduce((n,c)=>n+(c.visits||0),0)||1;
 const puctP=puctRank.map(c=>(c.visits||0)/puctVisits).filter(p=>p>0);
 let puctEntropy=0;for(const p of puctP)puctEntropy-=p*Math.log(p);
 puctEntropy=puctP.length>1?puctEntropy/Math.log(puctP.length):0;
 const diffuse=policyStats.entropy>=.76||policyStats.topGap<.035||puctEntropy>=.78;
 const conflict=exact.length>0&&puctRank.length>0&&
  (!pvsTop.has(puctBest)||!puctTop.has(pvsBest)||!pvsTop.has(policyBest));
 const consensus=exact.length>0&&puctRank.length>0&&pvsTop.has(puctBest)&&puctTop.has(pvsBest)&&pvsTop.has(policyBest);
 const pvsMate=exact.length>0&&Math.abs(exact[0].score)>=29872;
 const forced=snapshot?.solved||rootLegal<=2||
  (pvsMate&&!conflict)||
  (policyStats.topRatio>=16&&!diffuse&&!conflict&&exact.length>0);
 const uncertain=conflict||diffuse||pvsClose||bestChanges>=2||puctBestChanges>=2;
 return {exact,puctRank,pvsTop,puctTop,policyTop,pvsBest,puctBest,policyBest,pvsGap,
  puctEntropy,pvsClose,diffuse,conflict,consensus,forced,pvsMate,uncertain};
}
function chooseVerifyMoves(uncertainty,snapshot) {
 const selected=[],seen=new Set();
 const add=m=>{if(Number.isInteger(m)&&m>=0&&m<81&&!seen.has(m)){seen.add(m);selected.push(m);}};
 const puctRank=uncertainty.puctRank||(snapshot?.candidates||[]).slice().sort((a,b)=>b.visits-a.visits||b.prior-a.prior);
 const policyTop=uncertainty.policyTop||[];
 const pvsTop=uncertainty.pvsTop||new Set();
 if(uncertainty.conflict){
  for(const c of puctRank.slice(0,4))add(c.move);
  for(const m of policyTop.slice(0,3))add(m);
 }else if(uncertainty.diffuse||uncertainty.pvsClose||uncertainty.uncertain){
  for(const c of puctRank.slice(0,3))add(c.move);
  for(const m of policyTop.slice(0,2))add(m);
 }else if(uncertainty.puctBest>=0&&!pvsTop.has(uncertainty.puctBest))add(uncertainty.puctBest);
 for(const c of snapshot?.candidates||[])if(c.solved&&c.score>0)add(c.move);
 return selected.slice(0,7);
}
function desiredModelShare(mcts,pvsRate,uncertainty,jobEvaluations=0,pvsNodes=0) {
 const inferMs=mcts.evaluations?mcts.evaluationTimeMs/mcts.evaluations:5.5;
 if(!(pvsRate>0)||!(inferMs>0))return {share:.90,targetEvalPerMillion:300,inferMs};
 // Calibrate the desired neural/PVS time ratio from observed device speeds.
 // Depth and disagreement gradually raise the neural share; the resulting
 // call density is clamped to the requested 200–500 / million-node region.
 const depthBoost=.65*(1-Math.exp(-Math.max(0,mcts.maxDepth)/10));
 const conflictBoost=uncertainty?.conflict ? .45 : uncertainty?.uncertain ? .22 : 0;
 const targetRatio=3.0+depthBoost+conflictBoost;
 const targetEvalPerMillion=clamp(targetRatio*1e6/(pvsRate*inferMs),200,500);
 const realizedRatio=targetEvalPerMillion*pvsRate*inferMs/1e6;
 let share=clamp(realizedRatio/(1+realizedRatio),.72,.90);
 // Close the loop using actual new model evaluations per completed PVS node.
 // This corrects for device-specific differences between the instantaneous
 // node-rate estimate and the full iterative-deepening workload.
 if(pvsNodes>50000&&jobEvaluations>0){
  const actualDensity=jobEvaluations/pvsNodes*1e6;
  if(actualDensity<targetEvalPerMillion*.90)
   share+=clamp((targetEvalPerMillion-actualDensity)/targetEvalPerMillion*.20,.03,.12);
  else if(actualDensity>targetEvalPerMillion*1.18)
   share-=clamp((actualDensity-targetEvalPerMillion)/targetEvalPerMillion*.10,.02,.06);
  share=clamp(share,.72,.92);
 }
 return {share,targetEvalPerMillion,inferMs};
}
function assessExactComplexity(r,depth,remaining,rootLegal) {
 const branchNodes=Math.max(1,wasm.apex_branch_nodes?.()||0);
 const avgBranch=(wasm.apex_branch_sum?.()||0)/branchNodes;
 const forced=((wasm.apex_single_nodes?.()||0)+(wasm.apex_two_nodes?.()||0))/branchNodes;
 const ttProbes=wasm.apex_tt_probes?.()||0,ttHits=wasm.apex_tt_hits?.()||0;
 const hitRate=ttProbes?ttHits/ttProbes:0;
 const ebf=Math.pow(Math.max(1,r.nodes||1),1/Math.max(1,depth));
 const blended=.58*Math.max(1,avgBranch)+.42*Math.min(8,ebf);
 const effectiveBranch=clamp(blended*(1-.46*Math.min(.96,forced))*(1-.30*hitRate),1.04,8);
 const extra=Math.max(0,remaining-depth);
 const logProjected=Math.log(Math.max(1,r.nodes||1))+extra*Math.log(effectiveBranch);
 const projectedNodes=logProjected<29?Math.exp(logProjected):Infinity;
 const structure=(rootLegal<=4?.85:0)+(avgBranch<=2.5?.70:0)+Math.min(1,forced*1.35)+
  (hitRate>=.12?.50:0)+(effectiveBranch<=1.65?.65:0);
 return {avgBranch,forced,hitRate,effectiveBranch,projectedNodes,structure,branchNodes,ttProbes,ttHits};
}
// Same strict AND/OR solver, TT and WDL interpretation for both modes.
// A proof of win is *not* a proof of its exact mate distance.
function unifiedProof(game,remaining,pvsRate,maxMs,id){
 const t0=performance.now(),deadline=Math.min(hardEnd,t0+Math.max(1,maxMs));
 const side=game.turn,results=[],stats=[];
 let best=-1,nodes=0;
 for(let pass=0;pass<2&&activeJob===id;pass++){
  if(performance.now()>=deadline-2)break;
  const who=pass?3-side:side,now=performance.now();
  const end=pass?deadline:now+Math.max(5,(deadline-now)*.53);
  sliceEnd=Math.min(deadline,end);nextProgress=now+125;
  const msAvailable=Math.max(1,sliceEnd-now);
  const limit=Math.min(1800000000,Math.max(6000,Math.floor((pvsRate||1600)*msAvailable*.90)));
  const r=wasm.apex_prove_for(who,remaining,limit),count=wasm.apex_proof_nodes();
  nodes+=count;results[pass]=r;stats.push({who,result:r,nodes:count});
  if(r>=0&&wasm.apex_proof_best()>=0)best=wasm.apex_proof_best();
  if(pass===0&&r===1)break;
 }
 const a=results[0],b=results[1];
 const outcome=a===1?1:b===1?-1:a===0&&b===0?0:null;
 if(!game.isLegal(best))best=-1;
 return {solved:outcome!==null,score:outcome===null?null:outcome===0?0:outcome>0?29999:-29999,
  best,nodes,elapsed:performance.now()-t0,stats,proofKind:'WDL (distance unproven)'};
}
// Once WDL is established, spend otherwise unused computation on an exact
// mate-distance certificate. Keep the WDL result even if full-horizon PVS
// times out. In continuous mode, refine repeatedly until solved/cancelled.
async function refineWdlToExact(remaining,rate,id,start,endless){
 const refinementStart=performance.now();
 let spentNodes=0,attempts=0,reportExact=null;
 while(activeJob===id){
  const now=performance.now();
  const available=endless?440:Math.max(0,hardEnd-now-12);
  if(available<25)break;
  const ms=endless?Math.min(440,240+Math.min(200,attempts*60)):available;
  sliceEnd=now+ms;nextProgress=now+130;
  // Timed exact search, not a guessed node-count ceiling: difficult positions
  // may require far more nodes than the PVS sample rate predicts.
  const complete=!!wasm.apex_search(remaining,1800000000);
  const r=report(remaining,performance.now()-start);
  spentNodes+=r.nodes||0;
  overallNodes+=r.nodes||0;
  if(complete&&r.solved){reportExact=r;break;}
  if(!endless)break;
  attempts++;
  await sleep();
 }
 return {reportExact,spentNodes,attempts,elapsedMs:performance.now()-refinementStart};
}
function hybridReport(snapshot,pvs,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,totalPuctMs,totalPvsMs,totalProofMs,policyStats,targetEvalPerMillion) {
 const pvsMap=new Map((pvs?.candidates||[]).map((c,i)=>[c.move,{...c,rank:i+1}]));
 const candidates=(snapshot?.candidates||[]).map(c=>{
  const p=pvsMap.get(c.move);
  return {...c,score:p?.solved?p.score:(c.solved?c.score:null),solved:!!(p?.solved||c.solved),
   proofKind:c.solved?'WDL (distance unproven)':undefined,bound:!!p?.bound,
   pvsScore:p?.score,pvsBound:p?.bound??true,pvsRank:p?.rank??null,pvsDepth:pvs?.depth??0};
 });
 const immediateWin=candidates.find(c=>c.move===snapshot?.best&&c.solved&&c.score>0);
 const pvsNodes=Math.max(0,totalNodes-(totalProofNodes||0));
 return {depth:snapshot?.depth||0,puctDepth:snapshot?.depth||0,pvsDepth:pvs?.depth||pvsDepth-1,
  best:snapshot?.best??-1,score:immediateWin?29999:null,solved:!!immediateWin,
  authority:immediateWin?'terminal-child':undefined,
  nodes:pvs?.nodes||0,totalNodes,pvsNodes,candidates,valueX:snapshot?.valueX??0,
  visits:snapshot?.visits||0,model:{format:'Stage2 FP16',evaluations:snapshot?.model?.evaluations||0,
   newEvaluations:jobEvaluations,inferenceMs:snapshot?.model?.inferenceMs||0,
   newInferenceMs:jobInferenceMs,policyEntropy:policyStats.entropy,
   evaluationsPerMillionPvsNodes:pvsNodes?jobEvaluations/pvsNodes*1e6:0,
   targetEvaluationsPerMillionPvsNodes:targetEvalPerMillion,
   modelTimeShare:(jobInferenceMs+totalPvsMs+totalProofMs)>0?jobInferenceMs/(jobInferenceMs+totalPvsMs+totalProofMs):1,
   puctWallShare:(totalPuctMs+totalPvsMs+totalProofMs)>0?totalPuctMs/(totalPuctMs+totalPvsMs+totalProofMs):1},
  pvsBest:pvs?.best??-1,pvsScore:pvs?.score??null,backend:'stage2-puct+pvs'};
}
async function runPuctSlice(mcts,deadline,id,onBatch) {
 let simulations=0;
 while(activeJob===id&&performance.now()<deadline){
  mcts.simulation();simulations++;
  if((simulations&3)===0){onBatch?.();await sleep();}
 }
 return simulations;
}
async function runStage2(game,m,id,start,ms,endless) {
 self.postMessage({type:'ready',id,backend:'STAGE2-PUCT-V14'});
 let net;
 try{net=await Stage2Net.load();}
 catch(e){self.postMessage({type:'notice',id,text:'Stage2 原模型加载失败，改用无模型搜索：'+e.message});return false;}
 if(activeJob!==id)return true;
 try{await load();}
 catch(e){self.postMessage({type:'notice',id,text:'WASM 加载失败，改用无模型搜索：'+e.message});return false;}
 if(activeJob!==id)return true;
 const expire=endless?Infinity:start+ms;
 hardEnd=expire;sliceEnd=Infinity;
 // Reuse an exact-position subtree only; completed proofs remain valid across turns.
 let reused=null;
 const oldMcts=cachedStage2;
 if(oldMcts?.net===net){
  const matches=(a,b)=>a.turn===b.turn&&a.target===b.target&&a.cells.every((v,i)=>v===b.cells[i]);
  if(matches(oldMcts.root.game,game))reused=oldMcts.root;
  else for(const edge of oldMcts.root.edges||[])if(edge.child&&matches(edge.child.game,game)){reused=edge.child;break;}
 }
 const mcts=new Stage2MCTS(game,net);
 if(reused)mcts.adoptSubtree(reused);
 cachedStage2=mcts;
 const evalBefore=mcts.evaluations,inferBefore=mcts.evaluationTimeMs;
 const tModel=performance.now();
 try{mcts.start();}
 catch(e){self.postMessage({type:'notice',id,text:'网络推理失败，切换无模型搜索：'+e.message});return false;}
 const initialModelMs=performance.now()-tModel;
 self.postMessage({type:'model',id,result:{format:'Stage2 FP16 WASM',elapsed:initialModelMs}});
 if(activeJob!==id)return true;
 wasm.apex_begin_search();setWasm(game);
 if(wasm.apex_clear_root_guidance)wasm.apex_clear_root_guidance();
 const policyStats=writeStage2Policy(mcts.snapshot());
 mlSessionActive=true;
 const rootLegal=game.legal().length, remaining=wasm.apex_possible_remaining();
 const allocator=new RollingAllocator(initialModelMs), scout=new PVScout(mcts);
 let jobEvaluations=mcts.evaluations-evalBefore,jobInferenceMs=mcts.evaluationTimeMs-inferBefore;
 let totalNodes=0,totalProofNodes=0,pvsRate=0,proofRate=0,pvsDepth=1,pvsRetry=0;
 let oracleNodes=0,oracleAttempts=0,oracleSuccess=0;
 let proofAttempts=0,proofCertificates=0,proofUsefulCertificates=0,proofProbeFailures=0;
 const proofDepthCounts={},proofCompletedDepthCounts={},proofEvents=[];
 let lastPvs=null,lastComplexity=null,lastUncertainty=null,last=null,lastProgress=-Infinity;
 let lastPvsBest=-1,lastPuctBest=-1,pvsBestChanges=0,puctBestChanges=0;
 const proofStats={get attempts(){return proofAttempts;},get certificates(){return proofCertificates;},get usefulCertificates(){return proofUsefulCertificates;}};
 const makeReport=()=>{
  const snap=mcts.snapshot();
  const modelBudget=desiredModelShare(mcts,pvsRate,lastUncertainty,jobEvaluations,
   Math.max(0,totalNodes-totalProofNodes));
  const phase=endgameProfile(wasm,lastComplexity);
  const shares=adaptiveShares(phase,modelBudget.share,scout.lastFrontier,mcts.root,proofStats);
  const times=allocator.used;
  const result=hybridReport(snap,lastPvs,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,
   times.model,times.pvs,times.proof,policyStats,modelBudget.targetEvalPerMillion);
  result.best=snap.best; result.solved=snap.solved;
  result.score=snap.solved?snap.score:null;
  result.lowerWdl=snap.lowerWdl;result.upperWdl=snap.upperWdl;
  result.valueX=snap.valueX;
  result.authority=snap.solved?'strict-wdl-bounds':undefined;
  result.proofKind=snap.solved?'WDL (distance unproven)':undefined;
  result.depth=Math.max(snap.depth,lastPvs?.depth||0);
  result.candidates=snap.candidates.map(c=>{
   const pv=(lastPvs?.candidates||[]).find(p=>p.move===c.move);
   return {...c,pvsScore:pv?.score??null,pvsBound:pv?.bound??true,pvsDepth:lastPvs?.depth||0};
  });
  result.model.allocatedModelShare=shares.model;
  result.model.allocatedPvsShare=shares.pvs;
  result.model.allocatedProofShare=shares.proof;
  result.model.puctWallMs=times.model;
  result.model.pvsWallMs=times.pvs; // Contains internal PVS Oracle time.
  result.model.proofWallMs=times.proof; // Independent proof, including state switching; excludes scouting.
  result.model.scoutWallMs=scoutWallMs;
  result.model.pvsInternalOracleNodes=oracleNodes;
  result.model.pvsInternalOracleAttempts=oracleAttempts;
  result.model.pvsInternalOracleSuccesses=oracleSuccess;
  result.model.wallMsAccounted=allocator.total();
  result.model.startupAndUnaccountedMs=Math.max(0,performance.now()-start-allocator.total());
  result.searchDiagnostics={...lastComplexity,phase:phase.urgency,
   proofAttempts,proofCertificates,proofUsefulCertificates,proofProbeFailures,
   proofFrontierDepth:scout.lastFrontier?.depth||0,proofDepthCounts,proofCompletedDepthCounts,proofEvents:proofEvents.slice(-8),
   proofBoundUpdates:mcts.boundsUpdated,
   resourceShares:shares,remaining,oracleNodes};
  return result;
 };
 let scoutWallMs=0;
 const publish=(force=false)=>{
  const now=performance.now();if(!force&&now-lastProgress<140)return;
  lastProgress=now;
  last=makeReport();
  self.postMessage({type:'info',id,...last,elapsed:now-start,proofNodes:totalProofNodes});
 };
 publish(true);

 // One bounded task each iteration; weighted wall-clock debt, rather than
 // fixed cycles, keeps budget shares meaningful at ANY interrupt point.
 while(activeJob===id){
  const now=performance.now();
  if(!endless&&now>=expire-7)break;
  if(mcts.root.solved)break;
  const snap=mcts.snapshot();
  const phase=endgameProfile(wasm,lastComplexity);
  const budget=desiredModelShare(mcts,pvsRate,lastUncertainty,jobEvaluations,
   Math.max(0,totalNodes-totalProofNodes));
  const shares=adaptiveShares(phase,budget.share,scout.lastFrontier,mcts.root,proofStats);
  let activity=allocator.choose(shares);
  // Balanced quantum lengths are bounded only to keep the Worker responsive;
  // an activity's cumulative budget is NOT hard-capped per game stage.
  const quantum=Math.min(140,Math.max(25,28+9*Math.log1p(allocator.total()/80)));
  const timeLeft=endless?quantum:Math.max(0,expire-now-7);
  const grant=Math.min(quantum,timeLeft);
  if(grant<5)break;
  currentId=id;overallNodes=totalNodes;
  if(activity==='model'){
   const t=performance.now(),preE=mcts.evaluations,preI=mcts.evaluationTimeMs;
   await runPuctSlice(mcts,Math.min(expire-7,t+grant),id,()=>publish());
   if(activeJob!==id)return true;
   // Cost of creating policy-led scout continuations is real neural time.
   const scoutStart=performance.now();
   if(!mcts.root.solved)scout.advance();
   scoutWallMs+=performance.now()-scoutStart;
   allocator.add('model',performance.now()-t);
   jobEvaluations+=mcts.evaluations-preE;
   jobInferenceMs+=mcts.evaluationTimeMs-preI;
   if(!mcts.root.solved){
    const nowBest=mcts.snapshot().best;
    if(nowBest!==lastPuctBest){if(lastPuctBest>=0)puctBestChanges++;lastPuctBest=nowBest;}
   }
  }else if(activity==='pvs'){
   const t=performance.now(),deadline=Math.min(expire-7,t+grant);
   // Restore the true root after any deep frontier proof. PVS policy hints
   // belong to the root only; proof TT remains globally reusable.
   setWasm(game);
   writeStage2Policy(mcts.snapshot());
   while(activeJob===id&&performance.now()<deadline-3&&!mcts.root.solved){
    const started=performance.now(),available=Math.max(1,deadline-started);
    sliceEnd=deadline;hardEnd=expire;nextProgress=started+130;
    const searchDepth=pvsDepth;
    const nodeBudget=Math.min(1800000000,Math.max(2500,Math.floor((pvsRate||1750)*available*.92)));
    workingDepth=searchDepth;
    const complete=!!wasm.apex_search(searchDepth,nodeBudget);
    const r=report(searchDepth,performance.now()-start);
    const dt=Math.max(.5,performance.now()-started);
    const inside=r.v12?.internalProofNodes||0;
    totalNodes+=(r.nodes||0)+inside;
    totalProofNodes+=inside;
    oracleNodes+=inside;
    oracleAttempts+=r.v12?.internalProofAttempts||0;
    oracleSuccess+=r.v12?.internalProofSuccesses||0;
    pvsRate=pvsRate?.50*pvsRate+.50*((r.nodes||0)/dt):(r.nodes||0)/dt;
    if(!complete){pvsRetry++;break;}
    lastPvs=r;pvsDepth=searchDepth+1;pvsRetry=0;
    const latest=mcts.snapshot();
    if(r.best!==lastPvsBest){if(lastPvsBest>=0)pvsBestChanges++;lastPvsBest=r.best;}
    lastUncertainty=pvsUncertainty(r,latest,pvsBestChanges,puctBestChanges,policyStats,rootLegal);
    writeVerifyMoves(chooseVerifyMoves(lastUncertainty,latest));
    lastComplexity=assessExactComplexity(r,searchDepth,remaining,rootLegal);
    if(r.solved){
     const wdl=r.score>29872?1:r.score< -29872?-1:0;
     mcts.certifyWdl(mcts.root,wdl,r.best);
     for(const cand of r.candidates)if(cand.solved&&!cand.bound&&mcts.root.edges){
      const edge=mcts.root.edges.find(e=>e.move===cand.move);
      if(edge){const child=mcts.child(mcts.root,edge);
       mcts.certifyWdl(child,-(cand.score>29872?1:cand.score< -29872?-1:0));}
     }
     break;
    }
    if(pvsDepth>remaining)break;
   }
   allocator.add('pvs',performance.now()-t);
  }else{
   const t=performance.now();
   const picked=scout.pick();
   if(picked){
    const node=picked.node,side=node.game.turn;
    // Prefer refuting the optimistic side first. If one side is already ruled
    // out, work on the other; completed attack decisions are never repeated.
    let attacker=side;
    if(node.attackResults[side] >= 0)attacker=3-side;
    else if(node.attackResults[3-side] < 0 && node.mean< -.12)attacker=3-side;
    else if(node.proofAttempts%4===3&&node.attackResults[3-side]<0)attacker=3-side;
    if(node.attackResults[attacker] < 0){
     setWasm(node.game);
     const stamp=performance.now(),deadline=Math.min(expire-7,t+grant);
     sliceEnd=deadline;hardEnd=expire;nextProgress=stamp+130;
     const msAvailable=Math.max(1,deadline-stamp);
     const limit=Math.min(1800000000,Math.max(4096,Math.floor((proofRate||2000)*msAvailable*1.18)));
     const result=wasm.apex_prove_for(attacker,wasm.apex_possible_remaining(),limit);
     const spent=wasm.apex_proof_nodes(),witness=wasm.apex_proof_best();
     const elapsed=Math.max(.1,performance.now()-stamp);
     proofRate=proofRate?.55*proofRate+.45*spent/elapsed:spent/elapsed;
     node.proofAttempts++;node.proofSpentMs+=elapsed;
     proofAttempts++;totalProofNodes+=spent;totalNodes+=spent;
     proofDepthCounts[picked.depth]=(proofDepthCounts[picked.depth]||0)+1;
     if(result>=0){
      mcts.certifyAttack(node,attacker,result,witness);
      proofCertificates++;proofUsefulCertificates+=1/(1+.35*picked.depth);
      if(node.solved){proofCompletedDepthCounts[picked.depth]=(proofCompletedDepthCounts[picked.depth]||0)+1;}
      proofEvents.push({depth:picked.depth,attacker,result,solved:node.solved,elapsed:+elapsed.toFixed(1),
       path:(()=>{const line=[];let n=node;while(n.parent){line.push(n.move);n=n.parent;}return line.reverse();})()});
      if(proofEvents.length>24)proofEvents.shift();
     }else proofProbeFailures++;
     setWasm(game);
    }
   }
   allocator.add('proof',performance.now()-t);
  }
  if(activeJob!==id)return true;
  publish();
  await sleep();
 }
 if(activeJob!==id)return true;
 publish(true);
 if(!endless)self.postMessage({type:'done',id,...attachCertifiedCandidates(game,last),
  elapsed:performance.now()-start,proofNodes:totalProofNodes,backend:'stage2-puct+pvs+pv-proof-v14'});
 else self.postMessage({type:'notice',id,text:last?.solved?'WDL 已严格证明':'当前分析已停止'});
 return true;
}
self.onmessage=async({data:m})=>{
 if(m.type==='cancel'){activeJob=null;return;}
 if(m.type!=='search')return;
 const id=m.id;activeJob=id;currentId=id;const start=performance.now();startClock=start;
 const game=newGame(m),key=keyOf(m),endless=!!m.endless,ms=Math.max(100,Math.min(20000,Number(m.time)||1500));
 if(m.student){
  // For topology-confirmed endgames, bypass neural initialization entirely.
  // Both UI modes then execute the IDENTICAL PVS + AND/OR solver with the
  // IDENTICAL wall-time budget. No neural startup or MCTS can steal time from
  // strict endgame proof. In earlier game stages Stage2 still runs normally.
  let sharedEndgame=false;
  try{
   await load();
   if(activeJob!==id)return;
   setWasm(game);
   const profile=endgameProfile(wasm);
   sharedEndgame=profile.definite;
  }catch(e){self.postMessage({type:'notice',id,text:'残局分析预检不可用：'+e.message});}
  previousKey='';previousDepth=0;previousReport=null;proofTreeDirty=true;
  if(sharedEndgame){
   self.postMessage({type:'notice',id,text:'残局：增强模式和普通模式使用同一严格搜索器'});
  }else if(await runStage2(game,m,id,start,ms,endless))return;
 }

 hardEnd=endless?Infinity:start+ms;nextProgress=start+170;
 mlSessionActive=false;
 let backend=m.backend==='js'?'js':'wasm';
 if(backend==='wasm'){try{await load();}catch(e){backend='js';self.postMessage({type:'notice',id,text:'JS 后备: '+e.message});}}
 if(activeJob!==id)return;
 if(backend==='wasm'){
  try{proofTreeDirty=false;wasm.apex_clear_root_guidance?wasm.apex_clear_root_guidance():wasm.apex_clear_proof_priors();setWasm(game);}catch(e){self.postMessage({type:'notice',id,text:e.message});return;}
 }else if(!js)js=new JSEngine();
 if(previousKey!==key){previousKey=key;previousDepth=0;previousReport=null;if(js)js.tt.clear();}
 self.postMessage({type:'ready',id,backend});
 if(activeJob!==id)return;
 const remain=backend==='wasm'?wasm.apex_possible_remaining():81-game.moves.length;
 let depth=Math.max(1,previousDepth+1),last=previousReport,total=0,iterBudget=100000,rate=0,retry=0;
 let proofAttempts=0,lastProofDepthPlain=0,lastProofMsPlain=-Infinity;
 if(last){self.postMessage({type:'info',id,...last,resumed:true});}
 // The strict proof is cheap on many topology-constrained 20-30-square
 // endgames. Probe BEFORE the first high-depth PVS search: otherwise a single
 // incomplete PVS iteration may consume the time budget without producing a
 // certificate. Both ML and ordinary endgames take this same execution path.
 if(backend==='wasm'&&activeJob===id){
  const phase=endgameProfile(wasm);
  if(phase.definite){
   const earlyAvailable=endless?350:Math.max(0,hardEnd-performance.now()-15);
   const earlyMs=endless?300:Math.min(420,earlyAvailable*.66);
   if(earlyMs>=12){
    proofAttempts++;lastProofMsPlain=performance.now();
    const certified=unifiedProof(game,remain,1800,earlyMs,id);
    total+=certified.nodes;overallNodes=total;
    if(certified.solved&&activeJob===id){
     const move=certified.best>=0?certified.best:game.legal()[0];
     last={depth:0,best:move,score:certified.score,solved:true,
      authority:'and-or-proof',proofKind:certified.proofKind,backend:'wasm',
      proofNodes:certified.nodes,totalNodes:total,pv:[move],candidates:[],
      searchDiagnostics:{...phase,remaining:remain,proofAttempts,shouldProbe:true}};
     const cBudget=endless?70:Math.min(60,Math.max(0,hardEnd-performance.now()-25)*.20);
     const childCertificates=certifyRootChildren(game,last,cBudget,id);
     last=childCertificates.result;total+=childCertificates.nodes;
     last.totalNodes=total;last.searchDiagnostics={...last.searchDiagnostics,certifiedMoves:childCertificates.proved};
     previousReport=last;
     self.postMessage({type:'info',id,...last});
     const refined=await refineWdlToExact(remain,1800,id,start,endless);
     if(activeJob!==id)return;
     total+=refined.spentNodes;overallNodes=total;
     if(refined.reportExact){
      last=attachCertifiedCandidates(game,{...refined.reportExact,authority:'exact-pvs',backend:'wasm',
       candidates:mergeMoveCertificates(refined.reportExact.candidates,last.candidates),
       totalNodes:total,searchDiagnostics:{...last.searchDiagnostics,exactRefined:true}});
      const more=certifyRootChildren(game,last,endless?180:Math.min(540,Math.max(0,hardEnd-performance.now()-12)*.85),id);
      last=more.result;total+=more.nodes;last.totalNodes=total;
     }else {last.totalNodes=total;last=attachCertifiedCandidates(game,last);}
     previousReport=last;
     self.postMessage({type:'info',id,...last});
     if(!endless)self.postMessage({type:'done',id,...last,elapsed:performance.now()-start});
     return;
    }
   }
  }
 }
 while(activeJob===id){
  if(!endless&&performance.now()>=hardEnd-12)break;
  if(depth>remain){self.postMessage({type:'notice',id,text:'搜索完成'});break;}
  const iterStart=performance.now(),remaining=hardEnd-iterStart;
  const chunk=endless?Math.min(650,Math.max(100,220+retry*90)):
   Math.max(5,(remaining-15)*(last?.searchDiagnostics?.definite?.60:1));
  sliceEnd=iterStart+chunk;
  let budget=endless?Math.min(1800000000,Math.max(100000,iterBudget)):
    Math.min(1800000000,Math.max(1200,Math.floor(rate?rate*Math.max(20,remaining-10)*.85:Math.min(900000,ms*280))));
  currentId=id;workingDepth=depth;overallNodes=total;self.postMessage({type:'thinking',id,depth,budget,elapsed:iterStart-start,totalNodes:total});
  let complete,r;
  if(backend==='wasm'){
   complete=wasm.apex_search(depth,budget)!==0;r=report(depth,performance.now()-start);
  }else{
   const j=js.search(game,depth,Math.min(budget,1200000));complete=j.complete;r={...j,depth,elapsed:performance.now()-start,solved:j.complete&&depth>=countOpenSquares(game)};
  }
  total+=(r.nodes||0);
  const dt=Math.max(0.5,performance.now()-iterStart),nr=(r.nodes||0)/dt;rate=rate?rate*.4+nr*.6:nr;
  if(complete){
   last={...r,totalNodes:total,backend};
   if(r.solved){const cc=certifyRootChildren(game,last,endless?180:Math.min(540,Math.max(0,hardEnd-performance.now()-12)*.85),id);last=cc.result;total+=cc.nodes;last.totalNodes=total;}
   previousDepth=depth;previousReport=last;
   self.postMessage({type:'info',id,...last});depth++;retry=0;
   iterBudget=Math.max(iterBudget,Math.ceil((r.nodes||1000)*2.2));
   if(backend==='wasm'&&!r.solved){
    const raw=assessExactComplexity(r,depth,wasm.apex_possible_remaining(),wasm.apex_root_legal_count());
    const phase=endgameProfile(wasm,raw);
    last.searchDiagnostics={...raw,...phase,depth,remaining:wasm.apex_possible_remaining(),proofAttempts};
    if(phase.proofPriority&&(!Number.isFinite(lastProofDepthPlain)||depth>lastProofDepthPlain||
       phase.definite&&performance.now()-lastProofMsPlain>240)){
     lastProofDepthPlain=depth;lastProofMsPlain=performance.now();proofAttempts++;
     const now=performance.now();
     const available=endless?240:Math.max(0,hardEnd-now-10);
     const probeTime=Math.min(phase.definite?240:70,available*(phase.definite?.65:.14));
     if(probeTime>=8){
      const proven=unifiedProof(game,wasm.apex_possible_remaining(),rate,probeTime,id);
      total+=proven.nodes;overallNodes=total;
      if(proven.solved){
       const move=proven.best>=0?proven.best:r.best;
       last={...r,best:move,score:proven.score,solved:true,proofKind:proven.proofKind,
        authority:'and-or-proof',proofNodes:proven.nodes,totalNodes:total,backend,
        searchDiagnostics:{...last.searchDiagnostics,proofAttempts}};
       const cc=certifyRootChildren(game,last,endless?70:Math.min(55,Math.max(0,hardEnd-performance.now()-25)*.20),id);
       last=cc.result;total+=cc.nodes;last.totalNodes=total;
       previousReport=last;
       self.postMessage({type:'info',id,...last});
       const refined=await refineWdlToExact(wasm.apex_possible_remaining(),rate,id,start,endless);
       if(activeJob!==id)return;
       total+=refined.spentNodes;overallNodes=total;
       if(refined.reportExact){
        last=attachCertifiedCandidates(game,{...refined.reportExact,solved:true,authority:'exact-pvs',backend,totalNodes:total,
         candidates:mergeMoveCertificates(refined.reportExact.candidates,last.candidates),
         searchDiagnostics:{...last.searchDiagnostics,exactRefined:true}});
        const more=certifyRootChildren(game,last,endless?180:Math.min(500,Math.max(0,hardEnd-performance.now()-12)*.85),id);
        last=more.result;total+=more.nodes;last.totalNodes=total;
       }else {last.totalNodes=total;last=attachCertifiedCandidates(game,last);}
       previousReport=last;self.postMessage({type:'info',id,...last});
       break;
      }
     }
    }
   }
   if(r.solved){self.postMessage({type:'notice',id,text:'已严格证明终局'});break;}
  }else{
   retry++;iterBudget=Math.min(1800000000,Math.max(iterBudget*2,Math.ceil((r.nodes||1000)*2.5)));
   // A stalled high-depth PVS must not suppress the independent proof engine.
   if(backend==='wasm'&&last?.searchDiagnostics?.proofPriority&&
      performance.now()-lastProofMsPlain>=180){
    const t=performance.now();lastProofMsPlain=t;proofAttempts++;
    const available=endless?210:Math.max(0,hardEnd-t-12);
    const probeMs=Math.min(210,available*(last.searchDiagnostics.definite?.70:.14));
    if(probeMs>=8){
     const exact=unifiedProof(game,wasm.apex_possible_remaining(),rate,probeMs,id);
     total+=exact.nodes;overallNodes=total;
     if(exact.solved){
      last={...last,best:exact.best>=0?exact.best:last.best,score:exact.score,solved:true,
       proofKind:exact.proofKind,authority:'and-or-proof',proofNodes:(last.proofNodes||0)+exact.nodes,
       totalNodes:total,searchDiagnostics:{...last.searchDiagnostics,proofAttempts}};
      const cc=certifyRootChildren(game,last,endless?70:Math.min(55,Math.max(0,hardEnd-performance.now()-25)*.20),id);
      last=cc.result;total+=cc.nodes;last.totalNodes=total;
      previousReport=last;self.postMessage({type:'info',id,...last});
      const refined=await refineWdlToExact(wasm.apex_possible_remaining(),rate,id,start,endless);
      if(activeJob!==id)return;
      total+=refined.spentNodes;overallNodes=total;
      if(refined.reportExact){
       last={...refined.reportExact,solved:true,authority:'exact-pvs',backend,totalNodes:total,
        searchDiagnostics:{...last.searchDiagnostics,exactRefined:true}};
      }else {last.totalNodes=total;last=attachCertifiedCandidates(game,last);}
      previousReport=last;self.postMessage({type:'info',id,...last});break;
     }
    }
   }
   if(!endless&&performance.now()>=hardEnd-15)break;
  }
  if(activeJob!==id)return;
  await sleep();
 }
 if(activeJob!==id)return;
 if(!last){const best=game.legal()[0]??-1;last={depth:0,best,score:null,pv:best>=0?[best]:[],candidates:[],totalNodes:total,backend,solved:false};}
 if(!endless)self.postMessage({type:'done',id,...attachCertifiedCandidates(game,last),elapsed:performance.now()-start});
 else self.postMessage({type:'notice',id,text:'全部剩余深度已检索'});
};
