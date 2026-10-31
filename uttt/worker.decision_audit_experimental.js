import {Game} from './game.js';
import {JSEngine} from './fallback.js';
import {Stage2Net} from './stage2_net.js';
import {Stage2MCTS} from './stage2_mcts.js';
import {endgameProfile} from './endgame.js';

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
// V12 decision-aware strict root certificates. This is NOT a heuristic
// evaluation and never interprets UNKNOWN as draw. We target disputed root
// candidates, not every leaf of the middle-game tree. W > unproven is safe
// for WDL; a proven D only displaces a proven L (not an unknown W).
function applyDecisionCertificates(r,certificates,snapshot) {
 if(!r||!certificates.size||r.solved)return r;
 const edges=(snapshot?.candidates||[]);
 const byVisits=new Map(edges.map(c=>[c.move,c.visits||0]));
 const candidates=(r.candidates||[]).map(c=>{
  const wdl=certificates.get(c.move);
  return wdl===undefined?c:{...c,score:wdl*29999,solved:true,bound:false,
   proofKind:'WDL (distance unproven)',authority:'root-child-exact-proof',wdl};
 });
 let best=r.best,winning=[];
 for(const [move,value] of certificates)if(value===1)winning.push(move);
 if(winning.length){
  winning.sort((a,b)=>(byVisits.get(b)||0)-(byVisits.get(a)||0));
  best=winning[0];
  return {...r,best,score:29999,solved:true,proofKind:'WDL (distance unproven)',
   authority:'decision-aware-root-proof',backend:'stage2+proof',candidates};
 }
 // We can replace a certified loss with a certified draw, but cannot rank a
 // proved draw above a potentially winning unknown alternative.
 if(certificates.get(best)===-1){
  const draws=[...certificates].filter(([,v])=>v===0).map(([mv])=>mv);
  if(draws.length){draws.sort((a,b)=>(byVisits.get(b)||0)-(byVisits.get(a)||0));best=draws[0];}
 }
 return {...r,best,candidates,decisionProof:{certified:certificates.size}};
}
function runDecisionAudit(game,wasm,phase,uncertainty,snapshot,pvs,certificates,deadline,id){
 if(!wasm?.apex_proof_move_wdl||!pvs||certificates.size||activeJob!==id)return {checked:0,proved:0,nodes:0,elapsed:0};
 const conflict=!!uncertainty?.conflict||!!uncertainty?.pvsClose;
 const pressure=phase.urgency + (conflict?.15:0) + (phase.effectivelyNarrow?.13:0);
 // Calibration from independent actual-game positions: before macro pressure
 // reaches its proofable region, thousands of proof nodes buy no certificates.
 // Use a smooth yield estimate; no fixed ply / empty-cell cutoff is involved.
 const signal=wasm.apex_endgame_signal?.()??0;
 const predictedYield=1/(1+Math.exp(-(signal-655)/87));
 const decisionGain=(conflict?1.3:1)*(phase.effectivelyNarrow?1.15:1);
 if(!phase.borderline||predictedYield*decisionGain<.22)
  return {checked:0,proved:0,nodes:0,elapsed:0};
 const t0=performance.now(),moves=[];
 const add=m=>{if(Number.isInteger(m)&&game.isLegal(m)&&!moves.includes(m))moves.push(m)};
 add(snapshot?.best);add(pvs.best);
 const visitRank=(snapshot?.candidates||[]).slice().sort((a,b)=>b.visits-a.visits);
 const pvsRank=(pvs.candidates||[]).filter(c=>!c.bound).slice().sort((a,b)=>b.score-a.score);
 for(const c of visitRank.slice(0,3))add(c.move);
 for(const c of pvsRank.slice(0,2))add(c.move);
 let checked=0,proved=0,nodes=0;
 const maxTime=Math.min(48,Math.max(6,(deadline-t0)*.38* Math.min(1,predictedYield*2.2)));
 const end=Math.min(deadline-3,t0+maxTime);
 for(const move of moves){
  const msLeft=end-performance.now();if(msLeft<3||activeJob!==id)break;
  const grant=Math.max(1300,Math.min(32000,Math.floor(msLeft*2300/Math.max(1,moves.length-checked))));
  sliceEnd=end;
  const wdl=wasm.apex_proof_move_wdl(move,grant),used=wasm.apex_proof_nodes();
  checked++;nodes+=used;
  if(wdl!==2){certificates.set(move,wdl);proved++;if(wdl===1)break;}
 }
 return {checked,proved,nodes,elapsed:performance.now()-t0,signal,predictedYield,pressure,conflict};
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
 self.postMessage({type:'ready',id,backend:'STAGE2-PUCT'});
 let net;
 try{net=await Stage2Net.load();}
 catch(e){self.postMessage({type:'notice',id,text:'Stage2 原模型加载失败，改用无模型搜索：'+e.message});return false;}
 if(activeJob!==id)return true;
 try{await load();}
 catch(e){self.postMessage({type:'notice',id,text:'Stage2-PVS WASM 加载失败，改用无模型搜索：'+e.message});return false;}
 if(activeJob!==id)return true;
 const expire=endless?Infinity:start+ms;
 hardEnd=expire;sliceEnd=Infinity;
 // Reuse only a subtree whose complete position (cells, side and target) matches.
 let reused=null;const oldMcts=cachedStage2;
 if(oldMcts?.net===net){
  const matches=(a,b)=>a.turn===b.turn&&a.target===b.target&&a.cells.every((v,i)=>v===b.cells[i]);
  if(matches(oldMcts.root.game,game))reused=oldMcts.root;
  else for(const edge of oldMcts.root.edges||[])if(edge.child&&matches(edge.child.game,game)){reused=edge.child;break;}
 }
 const mcts=new Stage2MCTS(game,net);
 if(reused)mcts.adoptSubtree(reused);
 cachedStage2=mcts;
 const evalsBeforeStart=mcts.evaluations,inferenceBeforeStart=mcts.evaluationTimeMs;
 const tModel=performance.now();
 let neuralSnapshot;
 try{neuralSnapshot=mcts.start();}
 catch(e){self.postMessage({type:'notice',id,text:'Stage2 推理失败，改用无模型搜索：'+e.message});return false;}
 const rootInferenceWallMs=Math.max(0,performance.now()-tModel);
 const rootInferenceCount=mcts.evaluations-evalsBeforeStart;
 const rootInferenceMs=mcts.evaluationTimeMs-inferenceBeforeStart;
 self.postMessage({type:'model',id,result:{format:'Stage2 FP16 WASM',elapsed:rootInferenceWallMs}});
 if(activeJob!==id)return true;
 // The ML TT persists across ML roots. Pure-PVS state is cleared on the first
 // transition only; the non-ML Worker path below remains untouched.
 // Shared canonical TT persists across mode switches and successive root positions.
 wasm.apex_begin_search();setWasm(game);
 if(wasm.apex_clear_root_guidance)wasm.apex_clear_root_guidance();else wasm.apex_clear_proof_priors();
 const policyStats=writeStage2Policy(neuralSnapshot);
 mlSessionActive=true;
 const rootLegal=wasm.apex_root_legal_count?wasm.apex_root_legal_count():game.legal().length;
 const remaining=wasm.apex_possible_remaining();
 let totalNodes=0,totalProofNodes=0,totalPuctMs=rootInferenceWallMs,totalPvsMs=0,totalProofMs=0;
 let jobEvaluations=rootInferenceCount,jobInferenceMs=rootInferenceMs,pvsRate=0,pvsDepth=1,pvsRetry=0;
 let lastPvs=null,lastComplexity=null,lastUncertainty=null,last=null;
 let lastPvsBest=-1,lastPuctBest=-1,pvsBestChanges=0,puctBestChanges=0;
 let proofAttempts=0,lastProofDepth=0,lastProofTime=-Infinity,cycle=0,lastProgress=0;
 const decisionCertificates=new Map();let decisionAuditAttempts=0,decisionAuditStats={checked:0,proved:0,nodes:0,elapsed:0};
 const publish=(force=false)=>{
  const now=performance.now();if(!force&&now-lastProgress<140)return;lastProgress=now;
  const snap=mcts.snapshot();
  const budget=desiredModelShare(mcts,pvsRate,lastUncertainty,jobEvaluations,Math.max(0,totalNodes-totalProofNodes));
  last=applyDecisionCertificates(hybridReport(snap,lastPvs,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,totalPuctMs,totalPvsMs,totalProofMs,policyStats,budget.targetEvalPerMillion),decisionCertificates,snap);
  last.model.allocatedModelShare=budget.share;
  last.model.pvsNodesPerMs=totalPvsMs>0?last.pvsNodes/totalPvsMs:0;
  last.model.puctWallMs=totalPuctMs;last.model.pvsWallMs=totalPvsMs;last.model.proofWallMs=totalProofMs;
  if(lastComplexity)last.searchDiagnostics={...lastComplexity,proofAttempts,
   modelShare:budget.share,targetEvalPerMillion:budget.targetEvalPerMillion};
  self.postMessage({type:'info',id,...last,elapsed:now-start,proofNodes:totalProofNodes});
 };
 publish(true);
 while(activeJob===id){
  const now=performance.now();if(!endless&&now>=expire-8)break;
  const remainingMs=endless?Infinity:Math.max(0,expire-now-8);
  const cycleMs=endless?Math.min(390,Math.max(180,300+pvsRetry*35)):
   Math.min(310,Math.max(0,remainingMs));
  if(cycleMs<12)break;
  const beforeSnap=mcts.snapshot();
  const priorUncertainty=lastPvs?pvsUncertainty(lastPvs,beforeSnap,pvsBestChanges,puctBestChanges,policyStats,rootLegal):
   {conflict:false,uncertain:true,diffuse:policyStats.entropy>=.76,pvsClose:false,forced:rootLegal<=2};
  const budget=desiredModelShare(mcts,pvsRate,priorUncertainty,jobEvaluations,Math.max(0,totalNodes-totalProofNodes));
  let modelShare=budget.share;
  const phase=endgameProfile(wasm,lastComplexity);
  if(phase.definite||phase.borderline)modelShare=Math.min(modelShare,1-phase.pvsShare);
  // Clear forced lines should be verified/proved rather than spending more
  // teacher calls on an already decisive branch.
  if(priorUncertainty.forced)modelShare=Math.min(modelShare,phase.definite?.10:.62);
  const puctBudgetMs=Math.max(0,cycleMs*modelShare);
  const puctStart=performance.now(),evalBefore=mcts.evaluations,evalMsBefore=mcts.evaluationTimeMs;
  currentId=id;workingDepth=mcts.maxDepth;overallNodes=totalNodes;nextProgress=puctStart+140;
  self.postMessage({type:'thinking',id,depth:mcts.maxDepth,pvsDepth,budget:Math.round(puctBudgetMs),
   elapsed:puctStart-start,totalNodes,visits:mcts.simulations,evaluations:mcts.evaluations});
  await runPuctSlice(mcts,puctStart+puctBudgetMs,id,()=>publish());
  const puctElapsed=Math.max(0,performance.now()-puctStart);
  totalPuctMs+=puctElapsed;
  jobEvaluations+=mcts.evaluations-evalBefore;
  jobInferenceMs+=mcts.evaluationTimeMs-evalMsBefore;
  if(activeJob!==id)return true;

  const snapshot=mcts.snapshot();
  if(snapshot.best!==lastPuctBest){if(lastPuctBest>=0)puctBestChanges++;lastPuctBest=snapshot.best;}
  if(snapshot.solved){
   last=hybridReport(snapshot,lastPvs,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,totalPuctMs,totalPvsMs,totalProofMs,policyStats,budget.targetEvalPerMillion);
   last=attachCertifiedCandidates(game,{...last,best:snapshot.best,score:29999,solved:true,
    proofKind:'WDL (distance unproven)',authority:'terminal-child',backend:'stage2+proof'});
   self.postMessage({type:'info',id,...last,elapsed:performance.now()-start,proofNodes:totalProofNodes});break;
  }
  const uncertainty=lastPvs?pvsUncertainty(lastPvs,snapshot,pvsBestChanges,puctBestChanges,policyStats,rootLegal):priorUncertainty;
  const verifyMoves=chooseVerifyMoves(uncertainty,snapshot);
  writeVerifyMoves(verifyMoves);

  // PVS is a bounded tactical verifier. Iterate deeper until its allocated
  // slice is consumed; an early completed iteration must not waste the slice.
  const pvsBudgetMs=Math.min(Math.max(0,cycleMs-puctElapsed),endless?Infinity:Math.max(0,expire-performance.now()-8));
  if(pvsBudgetMs>=8){
   const pvsStart=performance.now();
   const proofDeadline=pvsStart+pvsBudgetMs;
   // Guarantee the exact solver a slice even when the next PVS iteration
   // cannot finish; otherwise difficult endgames would starve proof forever.
   const reserveProof=phase.proofPriority?pvsBudgetMs*(phase.definite?.42:.12):0;
   const pvsDeadline=proofDeadline-reserveProof;
   let proofCandidate=null,exactPvsResult=null;
   while(activeJob===id&&performance.now()<pvsDeadline-2){
    const searchDepth=pvsDepth,remainingSlice=Math.max(0,pvsDeadline-performance.now());
    sliceEnd=pvsDeadline;hardEnd=expire;nextProgress=performance.now()+130;
    const nodeBudget=Math.min(1800000000,Math.max(2500,Math.floor((pvsRate||1750)*remainingSlice*.88)));
    currentId=id;workingDepth=searchDepth;overallNodes=totalNodes;
    const pvsIterStart=performance.now();
    const complete=!!wasm.apex_search(searchDepth,nodeBudget);
    const r=report(searchDepth,performance.now()-start);
    const dt=Math.max(.5,performance.now()-pvsIterStart),measured=(r.nodes||0)/dt;
    pvsRate=pvsRate?pvsRate*.42+measured*.58:measured;
    totalNodes+=r.nodes||0;overallNodes=totalNodes;
    if(!complete){pvsRetry++;break;}

    if(r.best!==lastPvsBest){if(lastPvsBest>=0)pvsBestChanges++;lastPvsBest=r.best;}
    lastPvs=r;pvsDepth=searchDepth+1;pvsRetry=0;
    const fresh=mcts.snapshot();
    lastUncertainty=pvsUncertainty(r,fresh,pvsBestChanges,puctBestChanges,policyStats,rootLegal);
    writeVerifyMoves(chooseVerifyMoves(lastUncertainty,fresh));

    // A completed full-horizon PVS result is exact. A shallow M± is only a
    // tactical signal; it is never promoted to a proof by score threshold.
    if(r.solved){exactPvsResult=r;break;}

    const complexity=assessExactComplexity(r,r.depth,remaining,rootLegal);
    const available=endless?Infinity:Math.max(0,expire-performance.now()-10);
    const affordableNodes=endless?900000:Math.max(12000,pvsRate*Math.max(20,available*.24));
    const projectedAffordable=complexity.projectedNodes<=Math.max(24000,affordableNodes*7);
    const structurallySimple=complexity.effectiveBranch<=1.58||complexity.structure>=1.85||
      (rootLegal<=3&&complexity.avgBranch<=3.1&&complexity.forced>=.08)||
      (rootLegal<=2&&complexity.avgBranch<=4.2);
    const phaseNow=endgameProfile(wasm,complexity);
    const shouldProbe=searchDepth>=1&&(phaseNow.proofPriority||projectedAffordable||structurallySimple);
    lastComplexity={...complexity,...phaseNow,depth:searchDepth,remaining,rootLegal,projectedAffordable,
      shouldProbe,proofAttempts,pvsRate};
    if(shouldProbe&&(searchDepth>lastProofDepth||phaseNow.definite&&
      performance.now()-lastProofTime>240)&&(endless||available>28)){
     proofCandidate={r,complexity,lastComplexity,structurallySimple,projectedAffordable};
     break;
    }
   }
   if(!proofCandidate&&!exactPvsResult&&phase.proofPriority&&lastPvs&&
      performance.now()-lastProofTime>= (phase.definite?180:420)){
    const complexity=lastComplexity||assessExactComplexity(lastPvs,lastPvs.depth,remaining,rootLegal);
    proofCandidate={r:lastPvs,complexity,structurallySimple:phase.effectivelyNarrow,
      projectedAffordable:phase.projectedCheap};
   }
   totalPvsMs+=Math.max(0,performance.now()-pvsStart);

   if(exactPvsResult){
    const snap=mcts.snapshot();
    last=hybridReport(snap,exactPvsResult,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,
     totalPuctMs,totalPvsMs,totalProofMs,policyStats,budget.targetEvalPerMillion);
    last=attachCertifiedCandidates(game,{...last,best:exactPvsResult.best,score:exactPvsResult.score,solved:true,authority:'exact-pvs',
     backend:'stage2+proof',pvsDepth:exactPvsResult.depth});
    self.postMessage({type:'info',id,...last,elapsed:performance.now()-start,proofNodes:totalProofNodes});break;
   }

   if(proofCandidate&&activeJob===id){
    const {r,complexity,structurallySimple,projectedAffordable}=proofCandidate;
    lastProofDepth=r.depth;lastProofTime=performance.now();proofAttempts++;
    const probeStart=performance.now();
    const phaseNow=endgameProfile(wasm,complexity);
    const sliceAvailable=Math.max(0,proofDeadline-probeStart-3);
    const probeMs=Math.min(sliceAvailable,phaseNow.definite?
      (endless?220:Math.max(18,(expire-probeStart-8)*.68)):
      (endless?60:Math.max(8,(expire-probeStart-8)*.14)));
    if(probeMs>=8){
     const result=unifiedProof(game,remaining,pvsRate,probeMs,id);
     totalProofNodes+=result.nodes;totalNodes+=result.nodes;overallNodes=totalNodes;
     totalProofMs+=result.elapsed;
     if(result.solved){
      const snap=mcts.snapshot();
      const move=result.best>=0?result.best:(lastPvs?.best>=0?lastPvs.best:(snap.best>=0?snap.best:(game.legal()[0]??-1)));
      last=hybridReport(snap,lastPvs,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,
       totalPuctMs,totalPvsMs,totalProofMs,policyStats,budget.targetEvalPerMillion);
      last=attachCertifiedCandidates(game,{...last,depth:remaining,puctDepth:snap.depth,best:move,score:result.score,solved:true,
       authority:'and-or-proof',backend:'stage2+proof',proofNodes:totalProofNodes,
       proofKind:result.proofKind,searchDiagnostics:{...lastComplexity,proofAttempts}});
      self.postMessage({type:'info',id,...last,elapsed:performance.now()-start,proofNodes:totalProofNodes});
      const cc=certifyRootChildren(game,last,endless?65:Math.min(60,Math.max(0,hardEnd-performance.now()-25)*.20),id);
     last=cc.result;totalProofNodes+=cc.nodes;totalNodes+=cc.nodes;
     self.postMessage({type:'info',id,...last,elapsed:performance.now()-start,proofNodes:totalProofNodes});
     const refined=await refineWdlToExact(remaining,pvsRate,id,start,endless);
      if(activeJob!==id)return true;
      totalNodes+=refined.spentNodes;overallNodes=totalNodes;
      totalPvsMs+=refined.elapsedMs;
      if(refined.reportExact){
       const ex=refined.reportExact,byMove=new Map(ex.candidates.map(c=>[c.move,c]));
       const candidates=last.candidates.map(c=>{
        const p=byMove.get(c.move);
        return {...c,pvsScore:p?.score,pvsBound:p?.bound??true,pvsDepth:ex.depth,
         score:p?.solved?p.score:c.score,solved:!!(p?.solved||c.solved),
         proofKind:p?.solved?undefined:c.proofKind};
       });
       last={...last,best:ex.best,score:ex.score,solved:true,depth:ex.depth,
        authority:'exact-pvs',proofKind:undefined,pvsDepth:ex.depth,totalNodes,
        pvsNodes:totalNodes-totalProofNodes,candidates,
        searchDiagnostics:{...lastComplexity,proofAttempts,exactRefined:true}};
       const more=certifyRootChildren(game,last,endless?180:Math.min(500,Math.max(0,hardEnd-performance.now()-12)*.85),id);
       last=more.result;totalProofNodes+=more.nodes;totalNodes+=more.nodes;last.totalNodes=totalNodes;
       last.model.modelTimeShare=jobInferenceMs/(jobInferenceMs+totalPvsMs+totalProofMs);
       last.model.pvsWallMs=totalPvsMs;
       last.model.evaluationsPerMillionPvsNodes=jobEvaluations/Math.max(1,last.pvsNodes)*1e6;
       self.postMessage({type:'info',id,...last,elapsed:performance.now()-start});
      }else last.totalNodes=totalNodes;
      break;
     }

     // If proof-number search is inconclusive, try exact PVS only when the
     // measured effective tree cost justifies it and slice time remains.
     const exactPlausible=projectedAffordable||complexity.effectiveBranch<=1.30||rootLegal<=2;
     const exactAvailable=Math.max(0,proofDeadline-performance.now()-3);
     const exactMs=Math.min(75,exactAvailable);
     if(exactPlausible&&exactMs>=16){
      const exactStart=performance.now();sliceEnd=exactStart+exactMs;hardEnd=expire;
      const exactBudget=Math.min(1400000,Math.max(4500,Math.floor((pvsRate||1800)*exactMs*.78)));
      const exactComplete=!!wasm.apex_search(remaining,exactBudget);
      const exactReport=report(remaining,performance.now()-start);
      totalNodes+=exactReport.nodes||0;overallNodes=totalNodes;
      totalPvsMs+=Math.max(0,performance.now()-exactStart);
      if(exactComplete&&exactReport.solved){
       const snap=mcts.snapshot();
       last=hybridReport(snap,exactReport,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,
        totalPuctMs,totalPvsMs,totalProofMs,policyStats,budget.targetEvalPerMillion);
       last=attachCertifiedCandidates(game,{...last,best:exactReport.best,score:exactReport.score,solved:true,authority:'exact-pvs',
        backend:'stage2+proof',proofNodes:totalProofNodes,searchDiagnostics:{...lastComplexity,proofAttempts}});
       self.postMessage({type:'info',id,...last,elapsed:performance.now()-start,proofNodes:totalProofNodes});break;
      }
     }
    }
   }
  }
  if(activeJob!==id)return true;
  // Once a completed PVS iteration and an ML policy distribution exist, spend
  // a bounded slice on the actual conflicting root candidates. Never retry a
  // failed audit in the same move: repeated UNKNOWNs would waste budget.
  if(!decisionAuditAttempts&&lastPvs&&cycle>=0){
   const phaseAudit=endgameProfile(wasm,lastComplexity);
   const auditLeft=endless?80:Math.max(0,expire-performance.now()-10);
   if(auditLeft>34){
    const measured=runDecisionAudit(game,wasm,phaseAudit,lastUncertainty,snapshot,lastPvs,
      decisionCertificates,performance.now()+Math.min(48,auditLeft*.48),id);
    if(measured.checked){decisionAuditAttempts++;decisionAuditStats=measured;
     totalProofNodes+=measured.nodes;totalNodes+=measured.nodes;totalProofMs+=measured.elapsed;
     if(decisionCertificates.has(snapshot.best)&&decisionCertificates.get(snapshot.best)===1)break;
     if([...decisionCertificates.values()].includes(1))break;
    }
   }
  }
  const measuredPvsNodes=Math.max(0,totalNodes-totalProofNodes);
  if(totalPvsMs>0){
   const cumulativeRate=measuredPvsNodes/totalPvsMs;
   if(Number.isFinite(cumulativeRate)&&cumulativeRate>0)pvsRate=pvsRate?pvsRate*.20+cumulativeRate*.80:cumulativeRate;
  }
  publish(true);
  cycle++;
  if(!endless&&performance.now()>=expire-8)break;
  await sleep();
 }
 if(activeJob!==id)return true;
 const snap=mcts.snapshot();
 if(!last||!last.solved){
  const budget=desiredModelShare(mcts,pvsRate,lastUncertainty,jobEvaluations,Math.max(0,totalNodes-totalProofNodes));
  last=applyDecisionCertificates(hybridReport(snap,lastPvs,pvsDepth,totalNodes,totalProofNodes,jobEvaluations,jobInferenceMs,totalPuctMs,totalPvsMs,totalProofMs,policyStats,budget.targetEvalPerMillion),decisionCertificates,snap);
  last.model.allocatedModelShare=budget.share;
  last.model.pvsNodesPerMs=totalPvsMs>0?last.pvsNodes/totalPvsMs:0;
  last.model.puctWallMs=totalPuctMs;last.model.pvsWallMs=totalPvsMs;last.model.proofWallMs=totalProofMs;
  if(lastComplexity)last.searchDiagnostics={...lastComplexity,proofAttempts,decisionAuditAttempts,decisionAuditStats,modelShare:budget.share,
   targetEvalPerMillion:budget.targetEvalPerMillion};
 }
 if(!endless){
  last=applyDecisionCertificates(last,decisionCertificates,snap);
  self.postMessage({type:'done',id,...attachCertifiedCandidates(game,last),visits:snap.visits,elapsed:performance.now()-start,
   proofNodes:totalProofNodes,backend:last.backend||'stage2-puct+pvs'});
 }else self.postMessage({type:'notice',id,text:last.solved?'终局已严格证明':'持续分析已完成当前轮'});
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
