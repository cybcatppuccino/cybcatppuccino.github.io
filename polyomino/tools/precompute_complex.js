// Offline complex-periodic showcase generator. Uses cached exact seeds as reliable starts,
// then spends a short deterministic LNS budget per candidate and stores improved exact tori.
const fs=require('fs'),vm=require('vm'),{performance}=require('perf_hooks');
global.performance=performance;global.self={};global.postMessage=()=>{};
vm.runInThisContext(fs.readFileSync('/mnt/data/polyomino-lab/solver.worker.js','utf8'));
const parse=s=>s.split(';').map(z=>z.split(',').map(Number));
const leaderboard=JSON.parse(fs.readFileSync('/mnt/data/polyomino-lab/data/known/complex-periodic-leaderboard.json')).top;
const limit=+(process.argv[2]||24), budget=+(process.argv[3]||240), seen=new Set(), targets=[];
for(const r of leaderboard){const k=r.id+'|'+r.group;if(seen.has(k))continue;seen.add(k);targets.push(r);if(targets.length>=limit)break;}
const cats=new Map(),seeds=new Map();
function cat(n){if(!cats.has(n)){const d=JSON.parse(fs.readFileSync(`/mnt/data/polyomino-lab/data/catalog/free-holeless-n${n}.json`));cats.set(n,new Map(d.shapes.map(s=>[s.id,s])));}return cats.get(n);}
function seeddb(n){if(!seeds.has(n))seeds.set(n,JSON.parse(fs.readFileSync(`/mnt/data/polyomino-lab/data/known/periodic-seeds-n${n}.json`)));return seeds.get(n);}
const reporter={progress(){},preview(){},partial(){}};const records={};let improved=0;
for(let i=0;i<targets.length;i++){
 const t=targets[i],sh=cat(t.n).get(t.id),shape=parse(sh.c),oris=orientations(shape,t.group),sd=seeddb(t.n).seeds[t.id]?.[t.group];if(!sd)continue;
 const ps=seedToPlacements(sd,oris),base=makePeriodicResult(sd.w,sd.h,ps,'cache','cached exact'),start=performance.now();
 const ctx={shape,shapeSig:encode(shape)+'|'+t.group,oris,n:t.n,group:t.group,mode:'complex',algorithm:'hybrid',opt:{maxArea:288,maxSide:26,maxResults:5},start,deadline:start+budget,reporter,catalogId:t.id,seedResult:base,knownNonTiler:false};
 const rs=solveComplex(ctx),best=rs.sort((a,b)=>b.metrics.score-a.metrics.score)[0]||base;
 if(best.metrics.fundArea>base.metrics.fundArea+1e-9){improved++;records[t.id]=records[t.id]||{};records[t.id][t.group]={w:best.w,h:best.h,p:best.placements.map(p=>[p.ori,p.anchor[0],p.anchor[1]]),metrics:{mu:best.metrics.fundArea,S:best.metrics.score,Ha:best.metrics.adjEntropy,Ho:best.metrics.oriEntropy,tau:best.metrics.shortestPeriod}};}
 console.error(`${i+1}/${targets.length} ${t.id}/${t.group} μ ${base.metrics.fundArea} -> ${best.metrics.fundArea}`);
}
const out={version:1,generated:'2026-10-02',scope:`deterministic LNS showcase seeds from the top cached periodic candidates; ${budget}ms nominal budget per candidate`,targets:targets.length,improved,records};
fs.writeFileSync('/mnt/data/polyomino-lab/data/known/complex-showcase-seeds.json',JSON.stringify(out));console.error(`DONE improved=${improved}/${targets.length}`);
