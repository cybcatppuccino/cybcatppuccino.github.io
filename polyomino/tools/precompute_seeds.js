// Offline exact-periodic seed generator for tiling.lab.
// Usage: node tools/precompute_seeds.js [startN=1] [endN=8] [budgetMs=18] [maxSide=18] [maxArea=112]
const fs=require('fs'),vm=require('vm'),{performance}=require('perf_hooks');
global.performance=performance;global.self={};global.postMessage=()=>{};
vm.runInThisContext(fs.readFileSync('/mnt/data/polyomino-lab/solver.worker.js','utf8'));
function parseCells(s){return s.split(';').map(p=>p.split(',').map(Number));}
const groups=['C1','K4','C4','D4'];
const startN=+(process.argv[2]||1),endN=+(process.argv[3]||8),budget=+(process.argv[4]||18),maxSide=+(process.argv[5]||18),maxArea=+(process.argv[6]||112);
for(let n=startN;n<=endN;n++){
  const data=JSON.parse(fs.readFileSync(`/mnt/data/polyomino-lab/data/catalog/free-holeless-n${n}.json`,'utf8'));
  const out={n,range:{maxArea,maxSide,budgetMsPerGroup:budget,engine:'DLX-v2'},seeds:{}};
  let found=0,total=0;const t0=performance.now();
  for(let si=0;si<data.shapes.length;si++){
    const sh=data.shapes[si],shape=parseCells(sh.c),rec={};
    for(const g of groups){
      total++;const oris=orientations(shape,g),dims=dimensionCandidates(n,'periodic',maxSide,maxArea,g),groupDeadline=performance.now()+budget;let sol=null,dim=null;
      for(const d of dims){
        if(performance.now()>groupDeadline)break;
        const dat=makeTorusData(oris,d.w,d.h),localDeadline=Math.min(groupDeadline,performance.now()+Math.max(2,budget/2));
        const r=exactCoverDLX(dat,localDeadline,hashSeed(n,si,d.w,d.h,groups.indexOf(g)),90000,null,false);
        if(r.solution){sol=r.solution;dim=d;break;}
      }
      if(sol){found++;rec[g]={w:dim.w,h:dim.h,p:sol.map(p=>[p.ori,p.anchor[0],p.anchor[1]])};}
    }
    if(Object.keys(rec).length)out.seeds[sh.id]=rec;
    if((si+1)%250===0||si+1===data.shapes.length)console.error(`n=${n} ${si+1}/${data.shapes.length} found=${found}/${total}`);
  }
  // Regression seed: public data reports a 7×14 torus for the U-heptomino.
  if(n===7&&data.shapes.some(s=>s.id==='P7-0098')&&!out.seeds['P7-0098']?.D4){
    const sh=data.shapes.find(s=>s.id==='P7-0098'),oris=orientations(parseCells(sh.c),'D4'),dat=makeTorusData(oris,7,14);
    const r=exactCoverDLX(dat,performance.now()+1200,12345,1000000,null,false);
    if(r.solution){out.seeds['P7-0098']=out.seeds['P7-0098']||{};out.seeds['P7-0098'].D4={w:7,h:14,p:r.solution.map(p=>[p.ori,p.anchor[0],p.anchor[1]])};}
  }
  fs.writeFileSync(`/mnt/data/polyomino-lab/data/known/periodic-seeds-n${n}.json`,JSON.stringify(out));
  console.error(`DONE n=${n} shapes=${Object.keys(out.seeds).length}/${data.count} entries=${found}/${total} time=${((performance.now()-t0)/1000).toFixed(1)}s`);
}
