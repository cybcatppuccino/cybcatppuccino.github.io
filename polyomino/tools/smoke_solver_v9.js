const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'..');
global.self={};let resolveRun=null,lastPreview=null;
global.postMessage=m=>{if(m.type==='preview')lastPreview=m.result||lastPreview;if(m.type==='result')resolveRun?.(m);if(m.type==='error'){console.error('worker error',m.message);resolveRun?.(m);}};
vm.runInThisContext(fs.readFileSync(path.join(root,'solver.worker.js'),'utf8'));
function run(name,msg){return new Promise(async(resolve)=>{lastPreview=null;const t0=performance.now();resolveRun=m=>{const rs=m.results||[],r=rs[0]||lastPreview,ms=performance.now()-t0;console.log(`${name.padEnd(25)} ${(r?.metrics?.coverage??0).toFixed(4)} +${(r?.metrics?.combinedCoverage??r?.metrics?.coverage??0).toFixed(4)} m=${r?.metrics?.primitiveTileCount??'-'} ${ms.toFixed(0)}ms ${r?.kind||'-'} ${r?.engine||'-'}`);resolve(rs.length?rs:[r].filter(Boolean));};await self.onmessage({data:msg});});}
const options=(timeMs,cellPieces=10)=>({timeMs,cellPieces,aggression:.60,complexity:.56,regularity:.46,coverageBand:.03,targetCoverage:.9995,previewMs:250});
function assert(cond,msg){if(!cond)throw new Error(msg);}
(async()=>{
  const I5=[[0,0],[1,0],[2,0],[3,0],[4,0]],L5=[[0,0],[0,1],[0,2],[0,3],[1,3]];
  let rs=await run('periodic m=1 strip',{type:'solve',shape:I5,group:'D4',mode:'cell',algorithm:'auto',options:options(800,1)}),one=rs.find(r=>r.metrics?.primitiveTileCount===1);
  assert(one&&one.metrics.coverage>.999999&&one.placements.length===1,'m=1 periodic regression');
  rs=await run('periodic m<=2',{type:'solve',shape:L5,group:'D4',mode:'cell',algorithm:'auto',options:options(1000,2)});
  assert(rs.some(r=>r.metrics?.coverage>.999999&&r.metrics?.primitiveTileCount<=2),'small spectrum regression');

  const H7='0,0;0,1;0,2;1,2;2,2;3,2;1,3'.split(';').map(z=>z.split(',').map(Number));
  rs=await run('companion',{type:'solve',shape:H7,group:'D4',mode:'companion',algorithm:'auto',runNonce:3,options:options(1300,5)});
  const c=rs.slice().sort((a,b)=>(b.metrics?.combinedCoverage||0)-(a.metrics?.combinedCoverage||0)||(a.metrics?.companionArea??1e9)-(b.metrics?.companionArea??1e9))[0];
  assert(c&&c.companion,'companion result missing');
  assert((c.metrics.combinedCoverage||0)+1e-12>=c.metrics.coverage,'combined coverage below primary');
  assert((c.metrics.companionArea||0)>=1,'companion area regression');
  assert((c.companion.totalCells||0)===(c.metrics.companionArea||0)*(c.companion.copies||1),'whole-residual companion accounting regression');

  const R20=[];for(let y=0;y<4;y++)for(let x=0;x<5;x++)R20.push([x,y]);
  for(const alg of ['auto','dense','constructive']){rs=await run(`n20 ${alg}`,{type:'solve',shape:R20,group:'D4',mode:'cell',algorithm:alg,options:options(900,4)});assert(rs.some(r=>r.metrics?.coverage>.999999),`${alg} n20 regression`);}
  console.log('OK: periodic spectrum + companion smoke tests');
})();
