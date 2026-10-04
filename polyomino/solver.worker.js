'use strict';

// tiling.lab solver worker v9.0
// Philosophy: visible approximate answer first, proof/search second.
// No network/fetch/importScripts dependencies: safe to run as a Blob worker.

let STOP=false, RESULT_SEQ=0, ACTIVE_SHAPE_N=0, ACTIVE_ORIS=null, ACTIVE_GROUP='D4';
const DIR4=[[1,0],[-1,0],[0,1],[0,-1]];
const TORUS_CACHE=new Map(), FINITE_CACHE=new Map(), LATTICE_CACHE=new Map(), HNF_DATA_CACHE=new Map(), COMPANION_CACHE=new Map();
const TORUS_CACHE_LIMIT=6, FINITE_CACHE_LIMIT=2, LATTICE_CACHE_LIMIT=48, HNF_DATA_CACHE_LIMIT=28, COMPANION_CACHE_LIMIT=72;
const now=()=>performance.now();

self.onmessage=async e=>{
  const m=e.data||{};
  if(m.type==='ping'){postMessage({type:'ready',version:'9.0'});return;}
  if(m.type==='stop'){STOP=true;return;}
  if(m.type!=='solve')return;
  STOP=false;
  try{await solveRequest(m);}catch(err){postMessage({type:'error',message:err?.message||String(err),stack:err?.stack||''});}
};
postMessage({type:'ready',version:'9.0'});

function key(x,y){return `${x},${y}`;}
function mod(a,n){return ((a%n)+n)%n;}
function normalize(cells){
  if(!cells.length)return[];let minX=Infinity,minY=Infinity;
  for(const [x,y] of cells){if(x<minX)minX=x;if(y<minY)minY=y;}
  return cells.map(([x,y])=>[x-minX,y-minY]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]);
}
function encode(cells){return normalize(cells).map(([x,y])=>`${x},${y}`).join(';');}
function bounds(cells){let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const[x,y]of cells){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);}return{minX,minY,maxX,maxY,w:maxX-minX+1,h:maxY-minY+1};}
function groupTransforms(group){
  const I=(x,y)=>[x,y],R90=(x,y)=>[-y,x],R180=(x,y)=>[-x,-y],R270=(x,y)=>[y,-x],MX=(x,y)=>[-x,y],MY=(x,y)=>[x,-y],D=(x,y)=>[y,x],AD=(x,y)=>[-y,-x];
  if(group==='C1')return[I];if(group==='K4')return[I,R180,MX,MY];if(group==='C4')return[I,R90,R180,R270];return[I,R90,R180,R270,MX,MY,D,AD];
}
function orientations(shape,group){const out=[],seen=new Set();for(const f of groupTransforms(group)){const c=normalize(shape.map(([x,y])=>f(x,y))),s=encode(c);if(!seen.has(s)){seen.add(s);out.push(c);}}return out;}
function hashSeed(...v){let h=2166136261>>>0;for(const x of v){if(typeof x==='string'){for(let i=0;i<x.length;i++){h^=x.charCodeAt(i);h=Math.imul(h,16777619)>>>0;}}else{h^=(Number(x)||0)>>>0;h=Math.imul(h,16777619)>>>0;}}return h||1;}
function rngStep(s){return(Math.imul(1664525,s>>>0)+1013904223)>>>0;}
function shuffled(arr,seed){const out=arr.slice();let s=seed>>>0||1;for(let i=out.length-1;i>0;i--){s=rngStep(s);const j=s%(i+1);[out[i],out[j]]=[out[j],out[i]];}return out;}
function entropy(counts){let total=0;for(const v of counts.values())total+=v;if(!total||counts.size<=1)return 0;let H=0;for(const v of counts.values()){const p=v/total;H-=p*Math.log2(p);}return H/Math.log2(counts.size);}
function lruSet(map,k,v,limit){if(map.has(k))map.delete(k);map.set(k,v);while(map.size>limit)map.delete(map.keys().next().value);return v;}

class Reporter{
  constructor(start,deadline,cadence=500){this.start=start;this.deadline=deadline;this.cadence=Math.max(180,Math.min(1200,cadence||500));this.last=-1e9;this.lastPreview=-1e9;this.lastPartial=-1e9;}
  elapsed(){return now()-this.start;}
  progress(data={},force=false){const t=now();if(!force&&t-this.last<this.cadence)return false;this.last=t;postMessage({type:'progress',elapsed:this.elapsed(),remaining:Math.max(0,this.deadline-t),...data});return true;}
  preview(result,data={},force=false){result=validateResultForOutput(result);if(!result)return false;const t=now();if(!force&&t-this.lastPreview<this.cadence)return false;this.lastPreview=t;postMessage({type:'preview',result,...data,elapsed:this.elapsed()});return true;}
  partial(results,data={},force=false){results=(results||[]).map(validateResultForOutput).filter(Boolean);if(!results.length)return false;const t=now();if(!force&&t-this.lastPartial<Math.max(220,this.cadence*.72))return false;this.lastPartial=t;postMessage({type:'partial',results,...data,elapsed:this.elapsed()});return true;}
}

function placementFromAnchor(oris,oi,ax,ay,w,h){
  const cells=[],rawCells=[],idx=[],used=new Set();for(const[dx,dy]of oris[oi]){const rx=ax+dx,ry=ay+dy,x=mod(rx,w),y=mod(ry,h),id=y*w+x;if(used.has(id))return null;used.add(id);cells.push([x,y]);rawCells.push([rx,ry]);idx.push(id);}idx.sort((a,b)=>a-b);return{cells,rawCells,idx,ori:oi,anchor:[ax,ay]};
}
function seedToPlacements(seed,oris){if(!seed)return null;const out=[];for(const[oi,ax,ay]of seed.p||[]){if(!oris[oi])return null;const p=placementFromAnchor(oris,oi,ax,ay,seed.w,seed.h);if(!p)return null;out.push(p);}return out;}

function countBitsOcc(occ){let n=0;for(const v of occ)if(v)n++;return n;}
function placementCellSet(cells,w,h,periodic){const out=new Set();for(const c of cells||[]){if(!Array.isArray(c)||c.length<2)return null;let x=Number(c[0]),y=Number(c[1]);if(!Number.isInteger(x)||!Number.isInteger(y))return null;if(periodic){x=mod(x,w);y=mod(y,h);}else if(x<0||x>=w||y<0||y>=h)return null;out.add(y*w+x);}return out;}
function placementMatchesActiveShape(p,w,h,periodic){
  if(!ACTIVE_SHAPE_N||!ACTIVE_ORIS)return true;const oi=Number(p?.ori);if(!Number.isInteger(oi)||!ACTIVE_ORIS[oi]||!Array.isArray(p.anchor)||p.anchor.length<2)return false;
  const ax=Number(p.anchor[0]),ay=Number(p.anchor[1]);if(!Number.isInteger(ax)||!Number.isInteger(ay))return false;const want=ACTIVE_ORIS[oi].map(([dx,dy])=>[ax+dx,ay+dy]),A=placementCellSet(want,w,h,periodic),B=placementCellSet(p.cells,w,h,periodic);if(!A||!B||A.size!==ACTIVE_SHAPE_N||B.size!==ACTIVE_SHAPE_N)return false;for(const id of A)if(!B.has(id))return false;return true;
}
function validatePlacementSet(w,h,placements,periodic=false){
  if(!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||!Array.isArray(placements))return null;
  const area=w*h,owner=new Int32Array(area);owner.fill(-1);let covered=0;
  for(let i=0;i<placements.length;i++){
    const p=placements[i],cells=p?.cells;if(!Array.isArray(cells)||!cells.length||(ACTIVE_SHAPE_N&&cells.length!==ACTIVE_SHAPE_N)||!placementMatchesActiveShape(p,w,h,periodic))return null;
    const seen=new Set();
    for(const c of cells){if(!Array.isArray(c)||c.length<2)return null;let x=Number(c[0]),y=Number(c[1]);if(!Number.isInteger(x)||!Number.isInteger(y))return null;if(periodic){x=mod(x,w);y=mod(y,h);}else if(x<0||x>=w||y<0||y>=h)return null;const id=y*w+x;if(seen.has(id)||owner[id]>=0)return null;seen.add(id);owner[id]=i;covered++;}
    if(Array.isArray(p.rawCells)&&periodic){if(ACTIVE_SHAPE_N&&p.rawCells.length!==ACTIVE_SHAPE_N)return null;const rawSet=new Set();for(const [rx,ry] of p.rawCells){if(!Number.isInteger(rx)||!Number.isInteger(ry))return null;const id=mod(ry,h)*w+mod(rx,w);rawSet.add(id);}if(rawSet.size!==seen.size)return null;for(const id of seen)if(!rawSet.has(id))return null;}
  }
  return{owner,covered,coverage:covered/area};
}
function validatePeriodicNeighborhood(w,h,placements){
  // A torus solution should remain disjoint when lifted to neighbouring copies. This is
  // deliberately redundant with the quotient check: it protects renderer/history paths.
  const world=new Set();for(let sy=-1;sy<=1;sy++)for(let sx=-1;sx<=1;sx++)for(const p of placements){const raw=Array.isArray(p.rawCells)?p.rawCells:p.cells;if(ACTIVE_SHAPE_N&&raw.length!==ACTIVE_SHAPE_N)return false;for(const [x,y] of raw){const k=`${x+sx*w},${y+sy*h}`;if(world.has(k))return false;world.add(k);}}return true;
}
function rebuildOccFromChosen(chosen,area){const occ=new Uint8Array(area);for(const p of chosen||[])for(const id of p.idx||[]){if(id<0||id>=area||occ[id])return null;occ[id]=1;}return occ;}
function validateState(w,h,state){if(!state?.chosen)return null;const occ=rebuildOccFromChosen(state.chosen,w*h);if(!occ)return null;return{occ,chosen:state.chosen.slice()};}
function voidConnectivity(occ,w,h,wrap=false){
  const N=w*h,seen=new Uint8Array(N),holes=[];for(let i=0;i<N;i++)if(!occ[i])holes.push([i%w,(i/w)|0]);
  let components=0,largest=0,enclosed=0,perimeter=0,boundaryConnected=0;
  for(let start=0;start<N;start++){if(occ[start]||seen[start])continue;components++;let size=0,touches=false;const q=[start];seen[start]=1;for(let qi=0;qi<q.length;qi++){const id=q[qi],x=id%w,y=(id/w)|0;size++;if(!wrap&&(x===0||x===w-1||y===0||y===h-1))touches=true;for(const [dx,dy] of DIR4){let nx=x+dx,ny=y+dy;if(wrap){nx=mod(nx,w);ny=mod(ny,h);}else if(nx<0||nx>=w||ny<0||ny>=h){perimeter++;continue;}const nb=ny*w+nx;if(occ[nb])perimeter++;else if(!seen[nb]){seen[nb]=1;q.push(nb);}}}largest=Math.max(largest,size);if(!wrap&&!touches)enclosed++;if(!wrap&&touches)boundaryConnected++;}
  const total=holes.length,largestShare=total?largest/total:1,cohesion=total?(1/Math.max(1,components))*.45+largestShare*.55:1;
  return{holes,components,largest,largestShare,enclosed,boundaryConnected,perimeter,cohesion};
}
function voidComponentCells(occ,w,h,wrap=false){
  const N=w*h,seen=new Uint8Array(N),out=[];for(let start=0;start<N;start++){if(occ[start]||seen[start])continue;const cells=[],q=[start];seen[start]=1;for(let qi=0;qi<q.length;qi++){const id=q[qi],x=id%w,y=(id/w)|0;cells.push(id);for(const[dx,dy]of DIR4){let nx=x+dx,ny=y+dy;if(wrap){nx=mod(nx,w);ny=mod(ny,h);}else if(nx<0||nx>=w||ny<0||ny>=h)continue;const nb=ny*w+nx;if(!occ[nb]&&!seen[nb]){seen[nb]=1;q.push(nb);}}}out.push(cells);}out.sort((a,b)=>a.length-b.length);return out;
}
function chooseVoidFocus(occ,w,h,wrap,cellTo,seed){
  const comps=voidComponentCells(occ,w,h,wrap);if(!comps.length)return-1;const pool=comps.length>1?comps[0]:comps[0];let s=seed>>>0||1,best=pool[0],bestDeg=1e9;for(let z=0;z<Math.min(48,pool.length);z++){s=rngStep(s);const c=pool[s%pool.length],deg=cellTo?.[c]?.length??999999;if(deg<bestDeg){bestDeg=deg;best=c;}}return best;
}

function periodicPatternInfo(w,h,placements){
  const valid=validatePlacementSet(w,h,placements,true);if(!valid)return null;const area=w*h,n=placements.length,sigToIndex=new Map();
  // Translation symmetry is a symmetry of the tile partition, not merely of the set of
  // occupied quotient cells.  In very small quotients (e.g. one I5 tile on a 1x5 torus)
  // cell-set signatures incorrectly claim unit translations are symmetries because the
  // one tile occupies every quotient cell.  Anchors + orientation preserve tile boundaries.
  const tileSig=(p,dx=0,dy=0)=>`${Number(p.ori)||0}@${mod((p.anchor?.[0]||0)+dx,w)},${mod((p.anchor?.[1]||0)+dy,h)}`;
  placements.forEach((p,i)=>sigToIndex.set(tileSig(p),i));const shifts=[];let shortest=Infinity;
  for(let dy=0;dy<h;dy++)for(let dx=0;dx<w;dx++){let ok=true;for(const p of placements){if(!sigToIndex.has(tileSig(p,dx,dy))){ok=false;break;}}if(ok){shifts.push([dx,dy]);if(dx||dy){const ddx=Math.min(dx,w-dx),ddy=Math.min(dy,h-dy);shortest=Math.min(shortest,Math.hypot(ddx,ddy));}}}
  const parent=Int32Array.from({length:n},(_,i)=>i);const find=x=>{while(parent[x]!==x){parent[x]=parent[parent[x]];x=parent[x];}return x;};const unite=(a,b)=>{a=find(a);b=find(b);if(a!==b)parent[b]=a;};
  for(const [dx,dy] of shifts)for(let i=0;i<n;i++){const j=sigToIndex.get(tileSig(placements[i],dx,dy));if(j!==undefined)unite(i,j);}
  const roots=new Map(),classes=new Int16Array(n);let k=0;for(let i=0;i<n;i++){const r=find(i);if(!roots.has(r))roots.set(r,k++);classes[i]=roots.get(r);}
  if(!Number.isFinite(shortest))shortest=Math.min(w,h);return{...valid,translationSymmetries:Math.max(1,shifts.length),shortestPeriod:shortest,fundArea:area/Math.max(1,shifts.length),colorClasses:Array.from(classes),primitiveTileCount:k};
}
function validateResultForOutput(r){
  if(!r||!Number.isInteger(r.w)||!Number.isInteger(r.h)||r.w<1||r.h<1||!Array.isArray(r.placements))return null;
  const periodic=r.kind==='periodic'||r.kind==='cell'||r.kind==='field',lat=r.lattice&&Number.isInteger(r.lattice.a)?r.lattice:null;
  const v=periodic&&lat?validateLatticePlacementSet(lat.a,lat.b||0,lat.c,r.placements):validatePlacementSet(r.w,r.h,r.placements,periodic);if(!v)return null;
  if(periodic){if(lat){if(!validateLatticeNeighborhood(lat.a,lat.b||0,lat.c,r.placements))return null;}else if(!validatePeriodicNeighborhood(r.w,r.h,r.placements))return null;}
  if(!r.metrics)r.metrics={};
  if(periodic)r.metrics.coverage=Math.min(1,v.coverage);else r.metrics.coverage=Math.min(1,v.coverage);
  if(!Number.isFinite(r.metrics.coverage)||r.metrics.coverage<0||r.metrics.coverage>1+1e-9)return null;
  r.metrics.coverage=Math.min(1,r.metrics.coverage);r.metrics.coveredCells=v.covered;r.metrics.overlapFree=true;
  if(periodic&&r.companion){const seen=new Set();for(let i=0;i<v.owner.length;i++)if(v.owner[i]>=0)seen.add(i);let cc=0,shapeSig=null;const group=r.companion.group||'D4';for(const comp of r.companion.components||[]){const qcells=comp.quotientCells||[],raw=comp.cells||comp.rawCells||qcells;if(r.companion.area>0&&qcells.length!==r.companion.area)return null;const sig=canonicalCompanionSig(raw,group);if(shapeSig===null)shapeSig=sig;else if(sig!==shapeSig)return null;for(const q of qcells){const x=Number(q[0]),y=Number(q[1]),id=y*r.w+x;if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||x>=r.w||y<0||y>=r.h||seen.has(id))return null;seen.add(id);cc++;}}if(shapeSig!==null&&r.companion.shapeSig&&shapeSig!==r.companion.shapeSig)return null;r.companion.totalCells=cc;r.metrics.combinedCoverage=Math.min(1,(v.covered+cc)/(r.w*r.h));}
  return r;
}


async function solveRequest(m){
  const shape=normalize(m.shape||[]);if(!shape.length)throw new Error('empty polyomino');
  const n=shape.length,group=m.group||'C1',mode=m.mode||'periodic',algorithm=m.algorithm||'auto',opt=m.options||{};
  const oris=orientations(shape,group);ACTIVE_SHAPE_N=n;ACTIVE_ORIS=oris;ACTIVE_GROUP=group;const start=now(),budget=Math.max(350,Math.min(60000,opt.timeMs||60000)),deadline=start+budget,reporter=new Reporter(start,deadline,opt.previewMs||500),shapeSig=`${encode(shape)}|${group}`,runNonce=Number(m.runNonce)||0;
  const ctx={shape,shapeSig,oris,n,group,mode,algorithm,opt,start,deadline,reporter,runNonce,resumeResults:Array.isArray(m.resumeResults)?m.resumeResults:[],catalogId:m.catalogId||null,knownNonTiler:!!m.knownNonTiler,heesch:m.heesch||null,cachedSeed:m.cachedSeed||null,complexSeed:m.complexSeed||null,knownComplexity:m.knownComplexity||null};
  reporter.progress({phase:'prepare',engine:'A',group,orientations:oris.length,percent:0},true);

  let seedResult=null;
  const resumedPeriodic=ctx.resumeResults.find(r=>(r?.kind==='periodic'||r?.kind==='cell')&&r.w&&r.h&&Array.isArray(r.placements));
  if(resumedPeriodic){
    // Preserve HNF/sheared resume results instead of reinterpreting them as rectangular
    // tori. This matters after the first Continue press, where b!=0 lattices are common.
    if(resumedPeriodic.lattice?.a&&resumedPeriodic.lattice?.c){const q=validateResultForOutput({...resumedPeriodic,engine:'resume',metrics:{...(resumedPeriodic.metrics||{})}});if(q)seedResult=q;}
    else{const ps=[],occ=new Uint8Array(resumedPeriodic.w*resumedPeriodic.h);let ok=true;for(const rp of resumedPeriodic.placements){const p=placementFromAnchor(oris,Number(rp.ori)||0,Number(rp.anchor?.[0])||0,Number(rp.anchor?.[1])||0,resumedPeriodic.w,resumedPeriodic.h);if(!p||!canPlace(p,occ)){ok=false;break;}place(p,occ,ps);}if(ok&&ps.length)seedResult=makePeriodicResult(resumedPeriodic.w,resumedPeriodic.h,ps,'resume','periodic cell',opt);}
  }
  const limitSeed=Math.max(1,opt.cellPieces||10),complexPreferred=(opt.complexity??.5)>(opt.regularity??.5)&&m.complexSeed?.p?.length<=limitSeed,preferredSeed=complexPreferred?m.complexSeed:(m.cachedSeed||m.complexSeed||null);ctx.cachedSeed=complexPreferred?(m.complexSeed||m.cachedSeed):(m.cachedSeed||m.complexSeed);
  if(!seedResult&&(mode==='cell'||mode==='companion')&&preferredSeed){const p=seedToPlacements(preferredSeed,oris),limit=Math.max(1,opt.cellPieces||10),occ=new Uint8Array((preferredSeed.w||0)*(preferredSeed.h||0));let ok=!!p?.length&&p.length<=limit;if(ok)for(const q of p){if(!canPlace(q,occ)){ok=false;break;}place(q,occ,[]);}if(ok)seedResult=makePeriodicResult(preferredSeed.w,preferredSeed.h,p,'cache','periodic cell',opt);}
  if(seedResult){ctx.seedResult=seedResult;reporter.preview(seedResult,{phase:seedResult.engine==='resume'?'prepare':'cache',engine:seedResult.engine},true);reporter.partial([seedResult],{phase:'cache',engine:seedResult.engine},true);}ctx.seedResult=seedResult;

  let results=[],reason='';
  if(mode==='companion'){
    results=solvePeriodicSpectrum(ctx,'companion',seedResult);reason='companion-spectrum';
  }else{
    results=solvePeriodicSpectrum(ctx,'cell',seedResult);reason=results.some(r=>r?.kind==='periodic')?'periodic-spectrum':'periodic-spectrum-near';
  }
  if(STOP){postMessage({type:'stopped'});return;}
  results=(results||[]).map(validateResultForOutput).filter(Boolean);
  postMessage({type:'result',results,elapsed:now()-start,reason});
}

function dimensionCandidates(n,mode,maxSide=24,maxArea=240,group='D4',opt={}){
  const a=[];for(let w=1;w<=maxSide;w++)for(let h=1;h<=maxSide;h++){const area=w*h;if(area<n||area>maxArea||area%n)continue;const pieces=area/n;if(pieces>80)continue;if((group==='C4'||group==='D4')&&w>h)continue;a.push({w,h,area,pieces,aspect:Math.abs(Math.log(w/h))});}
  const cx=opt.complexity??.5,reg=opt.regularity??.5;if(mode==='periodic')a.sort((A,B)=>(A.area-B.area)*(1.15-cx*.6)+(A.aspect-B.aspect)*(1+reg*3)||A.w-B.w);else a.sort((A,B)=>(B.area-A.area)*(0.55+cx)+(A.aspect-B.aspect)*(1+reg*2)||A.w-B.w);return a;
}
function makeTorusData(oris,w,h){const area=w*h,placements=[],cellTo=Array.from({length:area},()=>[]),seen=new Set();for(let oi=0;oi<oris.length;oi++)for(let ay=0;ay<h;ay++)for(let ax=0;ax<w;ax++){const p=placementFromAnchor(oris,oi,ax,ay,w,h);if(!p)continue;const sig=p.idx.join(',');if(seen.has(sig))continue;seen.add(sig);const pi=placements.length;p.pi=pi;placements.push(p);for(const id of p.idx)cellTo[id].push(pi);}return{area,placements,cellTo};}
function torusData(ctx,w,h){const ck=`${ctx.shapeSig}|${w}x${h}`;let d=TORUS_CACHE.get(ck);if(d){TORUS_CACHE.delete(ck);TORUS_CACHE.set(ck,d);return d;}return lruSet(TORUS_CACHE,ck,makeTorusData(ctx.oris,w,h),TORUS_CACHE_LIMIT);}

function reduceHNF(x,y,a,b,c){const q=Math.floor(y/c),yy=y-q*c,xx=mod(x-q*b,a);return[xx,yy];}
function placementFromAnchorHNF(oris,oi,ax,ay,a,b,c){const cells=[],rawCells=[],idx=[],used=new Set();for(const[dx,dy]of oris[oi]){const rx=ax+dx,ry=ay+dy,[x,y]=reduceHNF(rx,ry,a,b,c),id=y*a+x;if(used.has(id))return null;used.add(id);cells.push([x,y]);rawCells.push([rx,ry]);idx.push(id);}idx.sort((x,y)=>x-y);return{cells,rawCells,idx,ori:oi,anchor:[ax,ay]};}
function makeHNFData(oris,a,b,c){const area=a*c,placements=[],cellTo=Array.from({length:area},()=>[]),seen=new Set();for(let oi=0;oi<oris.length;oi++)for(let ay=0;ay<c;ay++)for(let ax=0;ax<a;ax++){const p=placementFromAnchorHNF(oris,oi,ax,ay,a,b,c);if(!p)continue;const sig=p.idx.join(',');if(seen.has(sig))continue;seen.add(sig);const pi=placements.length;p.pi=pi;placements.push(p);for(const id of p.idx)cellTo[id].push(pi);}return{area,placements,cellTo,a,b,c};}

function reduceHNFCarry(x,y,a,b,c){const q=Math.floor(y/c),yy=y-q*c,x1=x-q*b,p=Math.floor(x1/a),xx=x1-p*a;return{x:xx,y:yy,p,q};}
function hnfNeighborId(id,dx,dy,a,b,c){const x=id%a,y=(id/a)|0,z=reduceHNFCarry(x+dx,y+dy,a,b,c);return z.y*a+z.x;}
function hnfNeighborWithCarry(id,dx,dy,a,b,c){const x=id%a,y=(id/a)|0,z=reduceHNFCarry(x+dx,y+dy,a,b,c);return{id:z.y*a+z.x,p:z.p,q:z.q};}
function dataNeighbor(data,id,dx,dy,w,h){return data?.a?hnfNeighborId(id,dx,dy,data.a,data.b||0,data.c):torusNeighbor(id,dx,dy,w,h);}
function hnfData(ctx,a,b,c){const ck=`${ctx.shapeSig}|H${a},${b},${c}`;let d=HNF_DATA_CACHE.get(ck);if(d){HNF_DATA_CACHE.delete(ck);HNF_DATA_CACHE.set(ck,d);return d;}d=makeHNFData(ctx.oris,a,b,c);return lruSet(HNF_DATA_CACHE,ck,d,HNF_DATA_CACHE_LIMIT);}
function validateLatticePlacementSet(a,b,c,placements){
  if(!Number.isInteger(a)||!Number.isInteger(b)||!Number.isInteger(c)||a<1||c<1||b<0||b>=a||!Array.isArray(placements))return null;const area=a*c,owner=new Int32Array(area);owner.fill(-1);let covered=0;
  for(let i=0;i<placements.length;i++){const p=placements[i],raw=Array.isArray(p?.rawCells)?p.rawCells:p?.cells;if(!Array.isArray(raw)||!raw.length||(ACTIVE_SHAPE_N&&raw.length!==ACTIVE_SHAPE_N))return null;if(ACTIVE_ORIS){const oi=Number(p.ori);if(!Number.isInteger(oi)||!ACTIVE_ORIS[oi]||encode(raw)!==encode(ACTIVE_ORIS[oi]))return null;}const seen=new Set();for(const q of raw){if(!Array.isArray(q)||q.length<2)return null;const rx=Number(q[0]),ry=Number(q[1]);if(!Number.isInteger(rx)||!Number.isInteger(ry))return null;const[x,y]=reduceHNF(rx,ry,a,b,c),id=y*a+x;if(seen.has(id)||owner[id]>=0)return null;seen.add(id);owner[id]=i;covered++;}if(Array.isArray(p.cells)){const qset=new Set();for(const q of p.cells){if(!Array.isArray(q)||q.length<2)return null;const x=Number(q[0]),y=Number(q[1]);if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||x>=a||y<0||y>=c)return null;qset.add(y*a+x);}if(qset.size!==seen.size)return null;for(const id of seen)if(!qset.has(id))return null;}}
  return{owner,covered,coverage:covered/area};
}
function validateLatticeNeighborhood(a,b,c,placements){const world=new Set();for(let sv=-2;sv<=2;sv++)for(let su=-2;su<=2;su++)for(const p of placements){const raw=p.rawCells||p.cells||[];for(const[x,y]of raw){const k=`${x+su*a+sv*b},${y+sv*c}`;if(world.has(k))return false;world.add(k);}}return true;}
function minimalHNFfromShifts(a,b,c,shifts){const D=a*c,S=Math.max(1,shifts?.length||1),Dp=D/S;if(!Number.isInteger(Dp)||Dp<1)return{a,b,c,det:D};const gens=[[a,0],[b,c],...(shifts||[])];for(let A=1;A<=Dp;A++){if(Dp%A)continue;const C=Dp/A;for(let B=0;B<A;B++){let ok=true;for(const[x,y]of gens){const q=reduceHNF(x,y,A,B,C);if(q[0]||q[1]){ok=false;break;}}if(ok)return{a:A,b:B,c:C,det:Dp};}}return{a,b,c,det:D};}
function periodicPatternInfoHNF(a,b,c,placements){
  const valid=validateLatticePlacementSet(a,b,c,placements);if(!valid)return null;const area=a*c,n=placements.length,sigToIndex=new Map();
  const tileSig=(p,dx=0,dy=0)=>{const q=reduceHNF((p.anchor?.[0]||0)+dx,(p.anchor?.[1]||0)+dy,a,b,c);return `${Number(p.ori)||0}@${q[0]},${q[1]}`;};
  placements.forEach((p,i)=>sigToIndex.set(tileSig(p),i));const shifts=[];let shortest=Infinity;
  const cosetNorm=(dx,dy)=>{let best=Infinity;for(let sv=-2;sv<=2;sv++)for(let su=-2;su<=2;su++)best=Math.min(best,Math.hypot(dx+su*a+sv*b,dy+sv*c));return best;};
  for(let dy=0;dy<c;dy++)for(let dx=0;dx<a;dx++){let ok=true;for(const p of placements){if(!sigToIndex.has(tileSig(p,dx,dy))){ok=false;break;}}if(ok){shifts.push([dx,dy]);if(dx||dy)shortest=Math.min(shortest,cosetNorm(dx,dy));}}
  const parent=Int32Array.from({length:n},(_,i)=>i),find=x=>{while(parent[x]!==x){parent[x]=parent[parent[x]];x=parent[x];}return x;},unite=(x,y)=>{x=find(x);y=find(y);if(x!==y)parent[y]=x;};
  for(const[dx,dy]of shifts)for(let i=0;i<n;i++){const j=sigToIndex.get(tileSig(placements[i],dx,dy));if(j!==undefined)unite(i,j);}const roots=new Map(),classes=new Int16Array(n);let k=0;for(let i=0;i<n;i++){const r=find(i);if(!roots.has(r))roots.set(r,k++);classes[i]=roots.get(r);}if(!Number.isFinite(shortest))shortest=Math.min(a,Math.hypot(b,c));const minimalLattice=minimalHNFfromShifts(a,b,c,shifts);return{...valid,translationSymmetries:Math.max(1,shifts.length),shortestPeriod:shortest,fundArea:area/Math.max(1,shifts.length),colorClasses:Array.from(classes),primitiveTileCount:k,minimalLattice,translationShifts:shifts};
}
function periodicVoidComponentsHNF(occ,a,b,c){
  const N=a*c,seen=new Uint8Array(N),components=[];
  for(let start=0;start<N;start++){
    if(occ[start]||seen[start])continue;
    const ids=[],quotientCells=[],cells=[],q=[start],lift=new Map([[start,[start%a,(start/a)|0]]]);let wraps=false;seen[start]=1;
    for(let qi=0;qi<q.length;qi++){
      const id=q[qi],rep=[id%a,(id/a)|0],xy=lift.get(id)||rep;ids.push(id);quotientCells.push(rep);cells.push(xy);
      for(const[dx,dy]of DIR4){
        const nb=hnfNeighborId(id,dx,dy,a,b,c);if(occ[nb])continue;const want=[xy[0]+dx,xy[1]+dy],old=lift.get(nb);
        // A finite residual component has a consistent lift to Z². If the same quotient
        // cell is reached at two coordinates differing by a lattice vector, this component
        // winds around the torus and is an infinite network in the periodic lift, not one
        // finite companion polyomino.
        if(old){if(old[0]!==want[0]||old[1]!==want[1])wraps=true;continue;}
        lift.set(nb,want);if(!seen[nb]){seen[nb]=1;q.push(nb);}
      }
    }
    components.push({ids,quotientCells,cells,area:ids.length,wraps,finiteLift:!wraps});
  }
  components.sort((x,y)=>y.area-x.area);return components;
}
function periodicVoidSummaryHNF(occ,a,b,c){const comps=periodicVoidComponentsHNF(occ,a,b,c),total=comps.reduce((z,q)=>z+q.area,0),largest=comps[0]?.area||0;let perimeter=0;for(let id=0;id<occ.length;id++)if(!occ[id])for(const[dx,dy]of DIR4)if(occ[hnfNeighborId(id,dx,dy,a,b,c)])perimeter++;const components=comps.length,largestShare=total?largest/total:1,cohesion=total?(1/Math.max(1,components))*.45+largestShare*.55:1;return{componentsData:comps,components,largest,largestShare,perimeter,cohesion,holes:comps.flatMap(q=>q.quotientCells)};}
function canonicalCompanionSig(cells,group){let best=null;for(const f of groupTransforms(group)){const q=encode(cells.map(([x,y])=>f(x,y)));if(best===null||q<best)best=q;}return best||'';}
function companionResidualPattern(occ,a,b,c,group){
  const v=periodicVoidSummaryHNF(occ,a,b,c),H=v.holes.length;
  if(!H)return{valid:true,exact:true,patternKind:'empty',area:0,copies:0,totalCells:0,coverage:1,components:[],placements:[],remaining:[],voidSummary:v,shapeSig:'',componentClasses:0};
  if(v.componentsData.some(comp=>comp.wraps))return{valid:true,exact:false,patternKind:'winding-residual',area:NaN,copies:0,totalCells:0,coverage:0,components:[],placements:[],remaining:v.holes.slice(),voidSummary:v,shapeSig:'',componentClasses:v.components};
  const classes=new Map();for(const comp of v.componentsData){const sig=canonicalCompanionSig(comp.cells,group),q=classes.get(sig)||{sig,count:0,area:comp.area};q.count++;classes.set(sig,q);}
  const uniform=classes.size===1,eligible=v.components===1||uniform;
  if(!eligible)return{valid:true,exact:false,patternKind:'mixed-components',area:NaN,copies:0,totalCells:0,coverage:0,components:[],placements:[],remaining:v.holes.slice(),voidSummary:v,shapeSig:'',componentClasses:classes.size};
  const first=v.componentsData[0],shapeSig=canonicalCompanionSig(first.cells,group),components=v.componentsData.map((comp,i)=>({cells:comp.cells.map(q=>q.slice()),rawCells:comp.cells.map(q=>q.slice()),quotientCells:comp.quotientCells.map(q=>q.slice()),ori:0,anchor:(comp.cells[0]||[0,0]).slice(),componentIndex:i}));
  return{valid:true,exact:true,patternKind:v.components===1?'single-component':'congruent-components',area:first.area,copies:components.length,totalCells:H,coverage:1,components,placements:components,remaining:[],voidSummary:v,shapeSig,componentClasses:1};
}
function companionTopologyScore(occ,a,b,c,group){
  const q=companionResidualPattern(occ,a,b,c,group),v=q.voidSummary,H=v.holes.length;if(!H)return 3.2;
  if(q.patternKind==='winding-residual')return-.45;
  if(q.exact){const compact=1-Math.min(1,(v.perimeter||0)/Math.max(1,H*4));return 2.15+(q.patternKind==='congruent-components'?.18:.10)+compact*.22+1/Math.max(8,H);}
  return Math.max(-.2,(v.cohesion||0)*.45-Math.max(0,(q.componentClasses||v.components)-1)*.10);
}
function companionFromVoid(occ,a,b,c,group,deadline=Infinity,seed=1){
  const holeIds=[];for(let id=0;id<occ.length;id++)if(!occ[id])holeIds.push(id);const ck=`${a},${b},${c}|${group}|whole|${holeIds.join('.')}`;
  if(COMPANION_CACHE.has(ck)){const cached=COMPANION_CACHE.get(ck);COMPANION_CACHE.delete(ck);COMPANION_CACHE.set(ck,cached);return cached;}
  const out=companionResidualPattern(occ,a,b,c,group);lruSet(COMPANION_CACHE,ck,out,COMPANION_CACHE_LIMIT);return out;
}
function expandHNFSolution(solution,oris,a,b,c){const W=a,H=a*c,out=[],occ=new Uint8Array(W*H);for(let k=0;k<a;k++)for(const p of solution){const q=placementFromAnchor(oris,p.ori,p.anchor[0]+k*b,p.anchor[1]+k*c,W,H);if(!q||!canPlace(q,occ))return null;place(q,occ,out);}return{w:W,h:H,placements:out,lattice:{a,b,c,area:a*c}};}
function hnfExactProbe(ctx,deadline){
  if(ctx.knownNonTiler||now()>=deadline)return null;const limit=Math.max(1,Math.min(20,ctx.opt.cellPieces||10)),cands=[];
  for(let m=1;m<=limit;m++){const D=ctx.n*m;for(let a=1;a<=Math.min(D,24);a++)if(D%a===0){const c=D/a;if(a*D>1300)continue;for(let b=1;b<a;b++){const shear=Math.min(b,a-b)/a;cands.push({m,D,a,b,c,rank:m*3+Math.abs(Math.log(a/Math.max(1,c)))+shear*.35});}}}
  cands.sort((x,y)=>x.rank-y.rank);let seed=hashSeed(ctx.shapeSig,0x484e46,ctx.runNonce),attempt=0;
  for(const d of cands.slice(0,90)){if(STOP||now()>=deadline)break;attempt++;const data=makeHNFData(ctx.oris,d.a,d.b,d.c);if(!data.placements.length||data.cellTo.some(x=>!x.length))continue;seed=rngStep(seed);const left=deadline-now(),slice=Math.min(105,Math.max(22,left/Math.max(1,Math.min(10,cands.length-attempt+1)))),r=exactCoverDLX(data,now()+slice,seed,180000,null,false);if(r.solution){const ex=expandHNFSolution(r.solution,ctx.oris,d.a,d.b,d.c);if(!ex)continue;const out=makePeriodicResult(ex.w,ex.h,ex.placements,'HNF','periodic lattice',ctx.opt);if(out){out.lattice=ex.lattice;out.metrics.latticeArea=d.D;out.metrics.tileCount=d.m;return out;}}}
  return null;
}

function refineExactPeriodic(base,ctx,deadline){
  if(!base||base.kind!=='periodic'||now()>=deadline)return base;const complexity=Math.max(0,Math.min(1,ctx.opt.complexity??.56)),regularity=Math.max(0,Math.min(1,ctx.opt.regularity??.48)),limit=Math.max(1,Math.min(20,ctx.opt.cellPieces||10));if(complexity<=regularity*.72)return base;let best=base;
  const mults=[[2,1],[1,2],[2,2],[3,1],[1,3]];for(const[kx,ky]of mults){if(STOP||now()>=deadline)break;if((base.placements?.length||0)*kx*ky>limit)continue;const sub={...ctx,deadline:Math.min(deadline,now()+160),opt:{...ctx.opt,maxArea:Math.max(ctx.n*limit,base.w*base.h*kx*ky)}};const r=mutateLifted(base,sub,kx,ky);if(r&&r.metrics.coverage>=.999999&&r.metrics.fundArea>(best.metrics.fundArea||0)+1e-9)best=r;}
  return best;
}

// Sparse Algorithm X / Dancing Links.
function exactCoverDLX(data,deadline,seed,nodeLimit,pulse,randomize=true){
  const cols=data.area,rows=data.placements.length,nPerRow=rows?data.placements[0].idx.length:0;if(!rows||!nPerRow)return{solution:null,nodes:0,bestDepth:0,best:[],timedOut:false};for(const list of data.cellTo)if(!list.length)return{solution:null,nodes:0,bestDepth:0,best:[],timedOut:false};
  const total=1+cols+rows*nPerRow,L=new Int32Array(total),R=new Int32Array(total),U=new Int32Array(total),D=new Int32Array(total),C=new Int32Array(total),ROW=new Int32Array(total),S=new Int32Array(cols+1);L[0]=cols;R[0]=cols?1:0;for(let c=1;c<=cols;c++){L[c]=c-1;R[c]=c===cols?0:c+1;U[c]=D[c]=c;C[c]=c;}
  let ptr=cols+1;for(let ri=0;ri<rows;ri++){let first=0,prev=0;for(const id of data.placements[ri].idx){const node=ptr++,c=id+1;C[node]=c;ROW[node]=ri;U[node]=U[c];D[node]=c;D[U[c]]=node;U[c]=node;S[c]++;if(!first){first=node;L[node]=R[node]=node;}else{L[node]=prev;R[node]=first;R[prev]=node;L[first]=node;}prev=node;}}
  let nodes=0,bestDepth=0,bestRows=[],solutionRows=[],timedOut=false,lastPulse=now();const stack=[];
  function cover(c){L[R[c]]=L[c];R[L[c]]=R[c];for(let i=D[c];i!==c;i=D[i])for(let j=R[i];j!==i;j=R[j]){U[D[j]]=U[j];D[U[j]]=D[j];S[C[j]]--;}}
  function uncover(c){for(let i=U[c];i!==c;i=U[i])for(let j=L[i];j!==i;j=L[j]){S[C[j]]++;U[D[j]]=j;D[U[j]]=j;}L[R[c]]=c;R[L[c]]=c;}
  function chooseColumn(){let c=R[0],best=c,min=1e9;for(;c!==0;c=R[c]){const s=S[c];if(s<min){min=s;best=c;if(s<=1)break;}}return best;}
  function pulseMaybe(force=false){const t=now();if(!force&&t-lastPulse<470)return;lastPulse=t;pulse?.({nodes,bestDepth,best:bestRows.map(i=>data.placements[i])});}
  function search(depth,localSeed){nodes++;if(depth>bestDepth){bestDepth=depth;bestRows=stack.slice();}if((nodes&127)===0){pulseMaybe();if(STOP||now()>deadline||nodes>nodeLimit){timedOut=true;return false;}}if(R[0]===0){solutionRows=stack.slice();return true;}const c=chooseColumn();if(!c||S[c]===0)return false;cover(c);let choices=[];for(let r=D[c];r!==c;r=D[r])choices.push(r);if(randomize&&choices.length>1)choices=shuffled(choices,localSeed^depth);for(const r of choices){stack.push(ROW[r]);for(let j=R[r];j!==r;j=R[j])cover(C[j]);if(search(depth+1,rngStep(localSeed^ROW[r]^depth))){for(let j=L[r];j!==r;j=L[j])uncover(C[j]);stack.pop();uncover(c);return true;}for(let j=L[r];j!==r;j=L[j])uncover(C[j]);stack.pop();if(timedOut){uncover(c);return false;}}uncover(c);return false;}
  const ok=search(0,seed>>>0||1);pulseMaybe(true);return{solution:ok?solutionRows.map(i=>data.placements[i]):null,nodes,bestDepth,best:bestRows.map(i=>data.placements[i]),timedOut};
}
function maxCoverDLX(data,deadline,seed,nodeLimit,initialState=null,pulse=null){
  const cols=data.area,tileRows=data.placements.length,nPerRow=tileRows?data.placements[0].idx.length:0;if(!tileRows||!nPerRow)return initialState;
  const total=1+cols+tileRows*nPerRow+cols,L=new Int32Array(total),R=new Int32Array(total),U=new Int32Array(total),D=new Int32Array(total),C=new Int32Array(total),ROW=new Int32Array(total),S=new Int32Array(cols+1);L[0]=cols;R[0]=cols?1:0;for(let c=1;c<=cols;c++){L[c]=c-1;R[c]=c===cols?0:c+1;U[c]=D[c]=c;C[c]=c;}
  let ptr=cols+1;function addRow(ri,ids){let first=0,prev=0;for(const id of ids){const node=ptr++,c=id+1;C[node]=c;ROW[node]=ri;U[node]=U[c];D[node]=c;D[U[c]]=node;U[c]=node;S[c]++;if(!first){first=node;L[node]=R[node]=node;}else{L[node]=prev;R[node]=first;R[prev]=node;L[first]=node;}prev=node;}}
  for(let ri=0;ri<tileRows;ri++)addRow(ri,data.placements[ri].idx);for(let c=0;c<cols;c++)addRow(tileRows+c,[c]);
  let bestCount=initialState?.chosen?.length||0,best=initialState?.chosen?.slice()||[],nodes=0,lastPulse=now(),stop=false;const stack=[];
  function cover(c){L[R[c]]=L[c];R[L[c]]=R[c];for(let i=D[c];i!==c;i=D[i])for(let j=R[i];j!==i;j=R[j]){U[D[j]]=U[j];D[U[j]]=D[j];S[C[j]]--;}}
  function uncover(c){for(let i=U[c];i!==c;i=U[i])for(let j=L[i];j!==i;j=L[j]){S[C[j]]++;U[D[j]]=j;D[U[j]]=j;}L[R[c]]=c;R[L[c]]=c;}
  function chooseColumn(){let c=R[0],best=c,min=1e9;for(;c!==0;c=R[c]){const z=S[c];if(z<min){min=z;best=c;if(z<=2)break;}}return best;}
  function emit(){const t=now();if(t-lastPulse<480)return;lastPulse=t;pulse?.({nodes,bestCount,best});}
  function search(tileCount,remainingCells,s){nodes++;if((nodes&127)===0){emit();if(STOP||now()>=deadline||nodes>=nodeLimit){stop=true;return;}}if(tileCount+Math.floor(remainingCells/nPerRow)<=bestCount)return;if(R[0]===0){if(tileCount>bestCount){bestCount=tileCount;best=stack.filter(ri=>ri<tileRows).map(ri=>data.placements[ri]);}return;}
    const c=chooseColumn();if(!c)return;cover(c);let real=[],hole=[];for(let r=D[c];r!==c;r=D[r]){const ri=ROW[r];(ri<tileRows?real:hole).push(r);}if(real.length>1)real=shuffled(real,s^tileCount);const choices=real.concat(hole);
    for(const r of choices){const ri=ROW[r],isTile=ri<tileRows,nextRemain=remainingCells-(isTile?nPerRow:1);stack.push(ri);for(let j=R[r];j!==r;j=R[j])cover(C[j]);search(tileCount+(isTile?1:0),nextRemain,rngStep(s^ri^nodes));for(let j=L[r];j!==r;j=L[j])uncover(C[j]);stack.pop();if(stop||bestCount*nPerRow===cols)break;}uncover(c);
  }
  search(0,cols,seed>>>0||1);emit();const occ=new Uint8Array(cols);for(const p of best)for(const id of p.idx)occ[id]=1;return{occ,chosen:best,nodes};
}

function exactCoverBitset(data,deadline,seed,nodeLimit,pulse){
  const area=data.area;if(area>180)return{solution:null,nodes:0,bestDepth:0,best:[],timedOut:true,skipped:true};const masks=data.placements.map(p=>{let m=0n;for(const id of p.idx)m|=1n<<BigInt(id);return m;}),full=(1n<<BigInt(area))-1n;let nodes=0,sol=[],best=[],bestDepth=0,timedOut=false,lastPulse=now();
  function chooseCell(cov){let bestList=null,min=1e9;for(let c=0;c<area;c++){if(cov&(1n<<BigInt(c)))continue;const ls=[];for(const pi of data.cellTo[c])if((masks[pi]&cov)===0n)ls.push(pi);if(!ls.length)return[];if(ls.length<min){min=ls.length;bestList=ls;if(min===1)break;}}return bestList||[];}
  function pulseMaybe(force=false){const t=now();if(!force&&t-lastPulse<470)return;lastPulse=t;pulse?.({nodes,bestDepth,best:best.map(i=>data.placements[i])});}
  function dfs(cov,depth,s){nodes++;if(depth>bestDepth){bestDepth=depth;best=sol.slice();}if((nodes&255)===0){pulseMaybe();if(STOP||now()>deadline||nodes>nodeLimit){timedOut=true;return false;}}if(cov===full)return true;const ls=chooseCell(cov);if(!ls.length)return false;for(const pi of shuffled(ls,s^depth)){const m=masks[pi];if(m&cov)continue;sol.push(pi);if(dfs(cov|m,depth+1,rngStep(s^pi)))return true;sol.pop();if(timedOut)return false;}return false;}
  const ok=dfs(0n,0,seed);pulseMaybe(true);return{solution:ok?sol.map(i=>data.placements[i]):null,nodes,bestDepth,best:best.map(i=>data.placements[i]),timedOut};
}

function solutionMetricsPeriodic(w,h,placements){
  const info=periodicPatternInfo(w,h,placements);if(!info)return null;const owner=info.owner,oriCounts=new Map(),adjCounts=new Map();
  for(const p of placements)oriCounts.set(p.ori,(oriCounts.get(p.ori)||0)+1);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const a=owner[y*w+x];if(a<0)continue;for(const[dx,dy]of[[1,0],[0,1]]){const b=owner[mod(y+dy,h)*w+mod(x+dx,w)];if(b<0||a===b)continue;const oa=placements[a].ori,ob=placements[b].ori,k=oa<=ob?`${oa}:${ob}`:`${ob}:${oa}`;adjCounts.set(k,(adjCounts.get(k)||0)+1);}}
  const adjEntropy=entropy(adjCounts),oriEntropy=entropy(oriCounts),voids=voidConnectivity(Uint8Array.from(owner,x=>x>=0?1:0),w,h,true),score=6.2*Math.log2(info.fundArea+1)+2.2*adjEntropy+1.7*oriEntropy+.08*Math.sqrt(w*h);
  return{coverage:info.coverage,fundArea:info.fundArea,adjEntropy,oriEntropy,shortestPeriod:info.shortestPeriod,translationSymmetries:info.translationSymmetries,primitiveTileCount:info.primitiveTileCount,colorClasses:info.colorClasses,voidComponents:voids.components,voidLargestShare:voids.largestShare,voidPerimeter:voids.perimeter,score};
}
function makePeriodicResult(w,h,placements,engine,label='periodic',opt={}){const metrics=solutionMetricsPeriodic(w,h,placements);if(!metrics||metrics.coverage<.999999||metrics.coverage>1+1e-9)return null;const cx=opt.complexity??.5,reg=opt.regularity??.5;metrics.score=metrics.coverage*120+Math.log2(metrics.fundArea+1)*(3+cx*8)+metrics.adjEntropy*(1+cx*4)+metrics.oriEntropy*(.6+cx*3)+reg*(8/(1+metrics.shortestPeriod));return{id:`T${++RESULT_SEQ}`,kind:'periodic',label,w,h,engine,placements:placements.map(p=>({cells:p.cells,rawCells:p.rawCells||p.cells,ori:p.ori,anchor:p.anchor})),colorClasses:metrics.colorClasses,metrics};}
function makePeriodicPreview(w,h,placements,area,n,engine){const valid=validatePlacementSet(w,h,placements,true);if(!valid)return null;const coverage=valid.coverage;return{id:'preview',kind:'cell',label:'search preview',w,h,engine,placements:placements.map(p=>({cells:p.cells,rawCells:p.rawCells||p.cells,ori:p.ori,anchor:p.anchor})),metrics:{coverage,fundArea:NaN,adjEntropy:NaN,oriEntropy:NaN,shortestPeriod:NaN,voidComponents:NaN,score:coverage*100}};}
function addResult(results,r,mode,maxResults=8){if(!r)return false;const sig=`${r.kind}|${r.w}x${r.h}|`+r.placements.map(p=>p.cells.map(([x,y])=>y*r.w+x).sort((a,b)=>a-b).join('.')).sort().join('|');if(results.some(x=>x._sig===sig))return false;r._sig=sig;results.push(r);results.sort((a,b)=>mode==='complex'?b.metrics.score-a.metrics.score:(a.kind!==b.kind?(a.kind==='periodic'?-1:1):(a.w*a.h-b.w*b.h)||b.metrics.score-a.metrics.score));if(results.length>maxResults)results.length=maxResults;return true;}
function chooseExactEngine(algorithm,area){if(algorithm==='bitset'&&area<=180)return'bitset';return'dlx';}

function exactSearchBudget(ctx,complexFlag){const total=ctx.deadline-now(),e=ctx.opt.exactness??.5,a=ctx.opt.aggression??.5;if(ctx.algorithm==='greedy'||ctx.algorithm==='frontier')return 0;if(ctx.algorithm==='exact'||ctx.algorithm==='bitset')return Math.max(200,total*.94);const base=ctx.n<=6?.62:ctx.n<=10?.42:ctx.n<=14?.27:.12;return total*Math.min(.82,base*(.55+e*.9+a*.35)*(complexFlag?.72:1));}
function solvePeriodic(ctx,complexFlag=false){
  const results=[],maxResults=ctx.opt.maxResults||8;if(ctx.seedResult){addResult(results,ctx.seedResult,complexFlag?'complex':'periodic',maxResults);if(!complexFlag&&(ctx.algorithm==='auto'||ctx.algorithm==='hybrid')&&((ctx.opt.complexity??.5)<.62||ctx.seedResult.metrics.fundArea>=(ctx.opt.targetMu||64)))return results;}
  const budget=exactSearchBudget(ctx,complexFlag);if(budget<80)return results;const end=Math.min(ctx.deadline,now()+budget),maxArea=Math.min(ctx.opt.maxArea||240,ctx.n>=15?Math.max(ctx.n*8,160):240),dims=dimensionCandidates(ctx.n,complexFlag?'complex':'periodic',ctx.opt.maxSide||24,maxArea,ctx.group,ctx.opt),cap=Math.max(6,Math.round((ctx.n<=8?18:ctx.n<=12?12:7)*(0.65+(ctx.opt.aggression??.5)*1.2)));let attempts=0,totalNodes=0,bestArea=results[0]?.kind==='periodic'?results[0].w*results[0].h:Infinity;
  for(const d of dims.slice(0,cap)){if(STOP||now()>end)break;if(!complexFlag&&d.area>=bestArea)break;attempts++;const remain=end-now(),slice=Math.min(complexFlag?260:180,remain),localEnd=now()+Math.max(18,slice),data=torusData(ctx,d.w,d.h),eng=chooseExactEngine(ctx.algorithm,d.area),pieces=d.pieces;
    const pulse=info=>{ctx.reporter.progress({phase:'exact',engine:eng==='dlx'?'DLX':'bit-X',current:attempts,total:Math.min(cap,dims.length),percent:attempts/Math.min(cap,dims.length),w:d.w,h:d.h,nodes:totalNodes+info.nodes,bestCoverage:pieces?info.bestDepth/pieces:0});if(info.best?.length)ctx.reporter.preview(makePeriodicPreview(d.w,d.h,info.best,d.area,ctx.n,eng==='dlx'?'DLX':'bit-X'),{phase:'exact',engine:eng==='dlx'?'DLX':'bit-X',nodes:totalNodes+info.nodes});};
    const nodeScale=1,nodeLimit=Math.round((complexFlag?220000:150000)*nodeScale),rs=hashSeed(ctx.shapeSig,d.w,d.h,attempts,ctx.runNonce),r=eng==='bitset'?exactCoverBitset(data,localEnd,rs,nodeLimit,pulse):exactCoverDLX(data,localEnd,rs,nodeLimit,pulse,complexFlag||ctx.algorithm==='hybrid'||(ctx.opt.temperature??.5)>.35);totalNodes+=r.nodes||0;
    if(r.solution){const sol=makePeriodicResult(d.w,d.h,r.solution,eng==='dlx'?'DLX':'bit-X',complexFlag?'complex periodic':'periodic',ctx.opt);if(sol&&addResult(results,sol,complexFlag?'complex':'periodic',maxResults)){ctx.reporter.preview(sol,{phase:'exact',engine:sol.engine},true);ctx.reporter.partial(results,{phase:'exact',engine:sol.engine},true);bestArea=Math.min(bestArea,d.area);}if(sol&&!complexFlag&&((ctx.opt.complexity??.5)<.65||sol.metrics.fundArea>=(ctx.opt.targetMu||64)))break;}
  }
  return results;
}

function liftSolution(base,oris,kx,ky){const W=base.w*kx,H=base.h*ky,out=[];for(let sy=0;sy<ky;sy++)for(let sx=0;sx<kx;sx++)for(const p of base.placements){const q=placementFromAnchor(oris,p.ori,p.anchor[0]+sx*base.w,p.anchor[1]+sy*base.h,W,H);if(q)out.push(q);}return{w:W,h:H,placements:out};}
function buildOwner(w,h,sol){const owner=new Int32Array(w*h);owner.fill(-1);for(let i=0;i<sol.length;i++)for(const id of sol[i].idx)owner[id]=i;return owner;}
function localAlternative(w,h,solution,selected,allData,seed,deadline){
  const regionIds=[],regionSet=new Set();for(const ti of selected)for(const id of solution[ti].idx)if(!regionSet.has(id)){regionSet.add(id);regionIds.push(id);}const m=regionIds.length;if(m===0||m>120)return null;const localIndex=new Map(regionIds.map((id,i)=>[id,i])),rows=[],cellTo=Array.from({length:m},()=>[]),candidateIds=new Set();for(const id of regionIds)for(const pi of allData.cellTo[id])candidateIds.add(pi);
  for(const pi of candidateIds){const p=allData.placements[pi],idx=[];let ok=true;for(const id of p.idx){const li=localIndex.get(id);if(li===undefined){ok=false;break;}idx.push(li);}if(!ok)continue;const ci=rows.length;rows.push({idx,ref:p});for(const li of idx)cellTo[li].push(ci);}if(!rows.length||cellTo.some(x=>!x.length))return null;const data={area:m,placements:rows,cellTo},oldSig=selected.map(i=>solution[i].idx.join(',')).sort().join('|'),r=exactCoverDLX(data,deadline,seed,16000,null,true);if(!r.solution)return null;const out=r.solution.map(x=>x.ref),sig=out.map(p=>p.idx.join(',')).sort().join('|');return sig===oldSig?null:out;
}
function mutateLifted(base,ctx,kx,ky){
  const lift=liftSolution(base,ctx.oris,kx,ky),w=lift.w,h=lift.h;if(w*h>(ctx.opt.maxArea||240))return null;let current=lift.placements,best=makePeriodicResult(w,h,current,'lift','lifted periodic',ctx.opt),seed=hashSeed(w,h,ctx.n,73,ctx.runNonce),moves=0,allData=torusData(ctx,w,h),maxMoves=Math.min(36,Math.max(8,Math.floor((ctx.deadline-now())/55)));
  for(let it=0;it<maxMoves&&now()<ctx.deadline;it++){const owner=buildOwner(w,h,current);seed=rngStep(seed);const start=seed%current.length,q=Math.min(current.length,3+(seed%7)),selected=[start],sel=new Set(selected),front=[start];while(selected.length<q&&front.length){const ti=front.shift();for(const[x,y]of current[ti].cells)for(const[dx,dy]of DIR4){const nb=owner[mod(y+dy,h)*w+mod(x+dx,w)];if(nb>=0&&!sel.has(nb)){sel.add(nb);selected.push(nb);front.push(nb);if(selected.length>=q)break;}}}const alt=localAlternative(w,h,current,selected,allData,seed,Math.min(ctx.deadline,now()+48));if(alt){current=current.filter((_,i)=>!sel.has(i)).concat(alt);moves++;const c=makePeriodicResult(w,h,current,'LNS','complex periodic',ctx.opt);if(c&&best&&c.metrics.score>best.metrics.score){best=c;ctx.reporter.preview(best,{phase:'mutate',engine:'LNS',current:it+1,total:maxMoves,bestScore:best.metrics.score});}}ctx.reporter.progress({phase:'mutate',engine:'LNS',current:it+1,total:maxMoves,percent:(it+1)/maxMoves,w,h,bestScore:best.metrics.score});}
  return best;
}
function solveComplex(ctx){
  const results=[],maxResults=ctx.opt.maxResults||8;let seed=ctx.seedResult;if(seed)addResult(results,seed,'complex',maxResults);if(!seed){const rs=solvePeriodic(ctx,true);for(const r of rs)addResult(results,r,'complex',maxResults);seed=rs.find(r=>r.kind==='periodic')||null;}
  if(!seed)return results;const target=ctx.opt.targetMu||64,baseArea=seed.w*seed.h,allMults=[[2,1],[3,1],[2,2],[3,2],[4,1],[3,3],[4,2],[5,1],[5,2]].filter(([a,b])=>baseArea*a*b<=(ctx.opt.maxArea||240));if(!allMults.length)return results;
  let round=0,stale=0,bestMu=Math.max(...results.map(r=>r.metrics?.fundArea||0));while(!STOP&&now()<ctx.deadline-35&&round<20000){const [kx,ky]=allMults[(round+ctx.runNonce)%allMults.length],sub={...ctx,runNonce:ctx.runNonce+round*104729};const r=mutateLifted(seed,sub,kx,ky);round++;if(r&&addResult(results,r,'complex',maxResults)){const mu=r.metrics?.fundArea||0;if(mu>bestMu+1e-9){bestMu=mu;stale=0;}else stale++;ctx.reporter.partial(results,{phase:'mutate',engine:r.engine});ctx.reporter.preview(results[0],{phase:'mutate',engine:r.engine,current:round,bestScore:results[0]?.metrics?.score,bestCoverage:1});}else stale++;ctx.reporter.progress({phase:'mutate',engine:'LNS',current:round,total:20000,percent:Math.min(.999,(now()-ctx.start)/(ctx.deadline-ctx.start)),w:r?.w||seed.w,h:r?.h||seed.h,bestScore:results[0]?.metrics?.score,bestCoverage:1});if(bestMu>=target&&now()-ctx.start>650)break;if(stale>24&&now()-ctx.start>1500&&(ctx.opt.aggression??.5)<.55)break;}
  return results;
}

function makeFiniteData(oris,w,h){const placements=[],cellTo=Array.from({length:w*h},()=>[]);for(let oi=0;oi<oris.length;oi++){const b=bounds(oris[oi]);for(let y=0;y<=h-b.h;y++)for(let x=0;x<=w-b.w;x++){const cells=oris[oi].map(([dx,dy])=>[x+dx,y+dy]),idx=cells.map(([xx,yy])=>yy*w+xx),p={cells,idx,ori:oi,anchor:[x,y]},pi=placements.length;placements.push(p);for(const id of idx)cellTo[id].push(pi);}}return{area:w*h,placements,cellTo};}
function finiteData(ctx,w,h){if(w*h>1500)return makeFiniteData(ctx.oris,w,h);const ck=`${ctx.shapeSig}|${w}x${h}`;let d=FINITE_CACHE.get(ck);if(d){FINITE_CACHE.delete(ck);FINITE_CACHE.set(ck,d);return d;}return lruSet(FINITE_CACHE,ck,makeFiniteData(ctx.oris,w,h),FINITE_CACHE_LIMIT);}
function canPlace(p,occ){for(const id of p.idx)if(occ[id])return false;return true;}
function place(p,occ,chosen){for(const id of p.idx)occ[id]=1;chosen.push(p);}

// --- v9 adaptive infinite field ----------------------------------------------------
// Field dimensions are not a UI parameter. We generate a mixed portfolio of compact,
// elongated and near-square tori, then let observed packing quality allocate budget.
function densityDimensionPortfolio(ctx,fast=false){
  const n=ctx.n,opt=ctx.opt||{},motif=Math.max(0,Math.min(1,opt.complexity??.56)),b=bounds(ctx.shape),resume=[],cellLimit=Math.max(1,Math.min(20,opt.cellPieces||10));
  for(const r of ctx.resumeResults||[])if(r?.w&&r?.h)resume.push([r.w,r.h]);
  const wantedPieces=ctx.mode==='cell'?Math.max(1,cellLimit*(.35+.65*motif)):6+motif*92,pieceSeeds=ctx.mode==='cell'?Array.from({length:cellLimit},(_,i)=>i+1):[4,6,8,10,12,16,20,24,32,40,52,68,88,112];
  pieceSeeds.push(Math.max(1,Math.round(wantedPieces*.6)),Math.max(1,Math.round(wantedPieces)),Math.max(1,Math.round(wantedPieces*1.35)));if(ctx.knownComplexity?.mu)pieceSeeds.push(Math.max(1,Math.min(cellLimit,Math.round(ctx.knownComplexity.mu/n))));
  const shapeRatio=Math.max(.16,Math.min(6,b.w/Math.max(1,b.h))),ratios=[.25,.34,.45,.58,.72,1,1.38,1.72,2.2,3,4,shapeRatio,1/shapeRatio];
  const map=new Map();
  function add(w,h,source=0){w=Math.max(1,Math.min(96,Math.round(w)));h=Math.max(1,Math.min(96,Math.round(h)));const area=w*h;if(area<n||area>1100||(ctx.mode==='cell'&&Math.floor(area/n)>cellLimit))return;const k=`${w}x${h}`;if(map.has(k))return;const pieces=area/n,scale=Math.max(0,Math.min(1,Math.log2(Math.max(2,pieces))/Math.log2(112))),scaleFit=1-Math.abs(scale-motif),aspect=Math.abs(Math.log(w/h)),exact=(area%n===0)?1:0,shapeFit=Math.min(Math.abs(Math.log((w/h)/shapeRatio)),Math.abs(Math.log((h/w)/shapeRatio)));map.set(k,{w,h,area,pieces,exact,source,rank:source*4+scaleFit*2.8+exact*.38-aspect*.045-shapeFit*.025});}
  for(const [rw,rh] of resume){add(rw,rh,4);for(const d of [-3,-2,-1,1,2,3]){add(rw+d,rh,3);add(rw,rh+d,3);}}
  for(const pieces0 of pieceSeeds){const area=ctx.mode==='cell'?Math.max(n,Math.round(n*pieces0)):Math.max(n*3,Math.round(n*pieces0));for(const ratio of ratios){const w0=Math.max(2,Math.round(Math.sqrt(area*ratio))),h0=Math.max(2,Math.round(area/w0));for(const dw of [-2,-1,0,1,2])for(const dh of [-2,-1,0,1,2])if(Math.abs(dw)+Math.abs(dh)<=2)add(w0+dw,h0+dh,0);}}
  // Explicitly include narrow motifs based on the tile bounding box; these often expose
  // simple strip constructions that a square-first search misses.
  for(let k=2;k<=10;k++){add(Math.max(2,b.w+k),Math.max(2,b.h+Math.round(n*k/Math.max(2,b.w+k))),1);add(Math.max(2,b.w+Math.round(n*k/Math.max(2,b.h+k))),Math.max(2,b.h+k),1);}
  let arr=[...map.values()];arr.sort((a,b)=>b.rank-a.rank||a.area-b.area);
  // Preserve geometry diversity: successive candidates should not all be the same aspect.
  const out=[],bins=new Set(),limit=fast?10:(ctx.mode==='cell'?Math.min(42,12+cellLimit*2):30);for(const d of arr){const bin=Math.round(Math.log(d.w/d.h)*3);if(bins.has(bin)&&out.length<Math.floor(limit*.45)&&d.source<3)continue;out.push(d);bins.add(bin);if(out.length>=limit)break;}
  return out.length?out:[{w:Math.max(4,b.w+2),h:Math.max(4,b.h+2),area:Math.max(4,b.w+2)*Math.max(4,b.h+2),rank:0}];
}
function densityDimensions(ctx,fast=false){return densityDimensionPortfolio(ctx,fast)[0];}
function torusNeighbor(id,dx,dy,w,h){const x=id%w,y=(id/w)|0;return mod(y+dy,h)*w+mod(x+dx,w);}
function packFromPlacements(data,w,h,seed,style=0){
  const occ=new Uint8Array(w*h),chosen=[],dead=new Uint8Array(w*h);let s=seed>>>0||1,remaining=w*h;
  const order=new Int32Array(w*h);for(let i=0;i<order.length;i++)order[i]=i;
  const ox=s%w,oy=(s>>>8)%h,axis=(s>>>16)&3;
  function orderedCell(k){let a=k%w,b=(k/w)|0;if(axis===1)a=(b&1)?w-1-a:a;else if(axis===2){const t=a;a=b%w;b=t%h;}else if(axis===3){a=w-1-a;b=h-1-b;}return mod(b+oy,h)*w+mod(a+ox,w);}
  for(let kk=0;kk<order.length&&remaining>=data.placements[0]?.idx.length;kk++){
    const c=orderedCell(kk);if(occ[c]||dead[c])continue;let bestPi=-1,best=-1e9;
    const list=data.cellTo[c];for(let z=0;z<list.length;z++){
      const pi=list[z],p=data.placements[pi];if(!canPlace(p,occ))continue;let contact=0;
      for(const id of p.idx)for(const[dx,dy]of DIR4){const nb=dataNeighbor(data,id,dx,dy,w,h);if(occ[nb])contact++;}
      // Prefer compact fronts; random noise creates distinct restarts without overwhelming density.
      s=rngStep(s^pi);const noise=((s&4095)/4095-.5)*(1.2+style*.5),score=contact*(5.2+style*.7)+noise;
      if(score>best){best=score;bestPi=pi;}
    }
    if(bestPi>=0){place(data.placements[bestPi],occ,chosen);remaining-=data.placements[bestPi].idx.length;}else dead[c]=1;
  }
  // A second pass catches cells made feasible by a different scan ordering.
  for(let pass=0;pass<2;pass++)for(let c=0;c<occ.length;c++)if(!occ[c]){const list=data.cellTo[c];for(let z=0;z<list.length;z++){const p=data.placements[list[(z+(s%Math.max(1,list.length)))%list.length]];if(canPlace(p,occ)){place(p,occ,chosen);break;}}}
  return{occ,chosen};
}
function mrvPack(data,w,h,seed,samples=24){
  const occ=new Uint8Array(w*h),chosen=[],dead=new Uint8Array(w*h);let s=seed>>>0||1,piecesMax=Math.floor(w*h/(data.placements[0]?.idx.length||1)),guard=0;
  while(chosen.length<piecesMax&&guard++<w*h*2){let bestCell=-1,bestFeasible=null,bestCount=1e9;
    for(let q=0;q<samples;q++){s=rngStep(s);let c=s%(w*h),tries=0;while((occ[c]||dead[c])&&tries++<10){s=rngStep(s);c=s%(w*h);}if(occ[c]||dead[c])continue;const feasible=[];for(const pi of data.cellTo[c]){if(canPlace(data.placements[pi],occ)){feasible.push(pi);if(feasible.length>=bestCount)break;}}if(!feasible.length){dead[c]=1;continue;}if(feasible.length<bestCount){bestCount=feasible.length;bestCell=c;bestFeasible=feasible;if(bestCount===1)break;}}
    if(bestCell<0){for(let c=0;c<occ.length;c++)if(!occ[c]&&!dead[c]){const feasible=[];for(const pi of data.cellTo[c])if(canPlace(data.placements[pi],occ))feasible.push(pi);if(feasible.length){bestCell=c;bestFeasible=feasible;break;}dead[c]=1;}if(bestCell<0)break;}
    let bp=-1,bs=-1e9;for(const pi of bestFeasible){const p=data.placements[pi];let contact=0;for(const id of p.idx)for(const[dx,dy]of DIR4)if(occ[dataNeighbor(data,id,dx,dy,w,h)])contact++;s=rngStep(s^pi);const sc=contact*4.6+(s&2047)/4096;if(sc>bs){bs=sc;bp=pi;}}if(bp>=0)place(data.placements[bp],occ,chosen);else dead[bestCell]=1;
  }
  for(let c=0;c<occ.length;c++)if(!occ[c])for(const pi of data.cellTo[c]){const p=data.placements[pi];if(canPlace(p,occ)){place(p,occ,chosen);break;}}
  return{occ,chosen};
}
function constructivePack(data,w,h,seed,opt={}){
  const occ=new Uint8Array(w*h),chosen=[],dead=new Uint8Array(w*h),oriUse=new Int32Array(Math.max(1,...data.placements.map(p=>p.ori+1)));let s=seed>>>0||1,guard=0;
  const variety=Math.max(0,Math.min(1,opt.complexity??.56)),regularity=Math.max(0,Math.min(1,opt.regularity??.48)),samples=Math.max(14,Math.min(42,Math.round(Math.sqrt(w*h)*1.15)));
  while(guard++<w*h*3&&!STOP){let cell=-1,feasible=null,deg=1e9;
    for(let q=0;q<samples;q++){s=rngStep(s);let c=s%(w*h),tries=0;while((occ[c]||dead[c])&&tries++<8){s=rngStep(s);c=s%(w*h);}if(occ[c]||dead[c])continue;const f=[];for(const pi of data.cellTo[c]){if(canPlace(data.placements[pi],occ)){f.push(pi);if(f.length>=deg)break;}}if(!f.length){dead[c]=1;continue;}if(f.length<deg){deg=f.length;cell=c;feasible=f;if(deg===1)break;}}
    if(cell<0){for(let c=0;c<occ.length;c++)if(!occ[c]&&!dead[c]){const f=[];for(const pi of data.cellTo[c])if(canPlace(data.placements[pi],occ))f.push(pi);if(f.length){cell=c;feasible=f;break;}dead[c]=1;}if(cell<0)break;}
    if(feasible.length>28)feasible=shuffled(feasible,s).slice(0,28);let bp=-1,bs=-1e12;
    for(const pi of feasible){const p=data.placements[pi];let contact=0,frontier=0;for(const id of p.idx){for(const[dx,dy]of DIR4){const nb=dataNeighbor(data,id,dx,dy,w,h);if(occ[nb])contact++;else frontier++;}}for(const id of p.idx)occ[id]=1;
      let traps=0,forced=0,checked=0;const around=new Set();for(const id of p.idx)for(const[dx,dy]of DIR4){const nb=dataNeighbor(data,id,dx,dy,w,h);if(!occ[nb])around.add(nb);}for(const c of around){if(checked++>18)break;let f=0;for(const qi of data.cellTo[c]){if(canPlace(data.placements[qi],occ)&&++f>=2)break;}if(f===0)traps++;else if(f===1)forced++;}
      for(const id of p.idx)occ[id]=0;s=rngStep(s^pi);let maxUse=0;for(const u of oriUse)if(u>maxUse)maxUse=u;const oriTerm=variety*((maxUse+1)/(oriUse[p.ori]+1))+regularity*(oriUse[p.ori]+1)/(maxUse+1),noise=((s&2047)/2047-.5)*.7,score=contact*5.4-frontier*.08-traps*45-forced*.7+oriTerm*2.8+noise;if(score>bs){bs=score;bp=pi;}}
    if(bp>=0){const p=data.placements[bp];place(p,occ,chosen);oriUse[p.ori]++;}else dead[cell]=1;
  }
  // Human-like finishing pass: fill obvious cavities before handing the state to LNS.
  for(let pass=0;pass<3;pass++){let changed=false;for(let c=0;c<occ.length;c++)if(!occ[c]){let only=-1,count=0;for(const pi of data.cellTo[c])if(canPlace(data.placements[pi],occ)){only=pi;if(++count>1)break;}if(count===1){place(data.placements[only],occ,chosen);changed=true;}}if(!changed)break;}
  return{occ,chosen};
}

function ownerForState(state,area){const owner=new Int32Array(area);owner.fill(-1);for(let i=0;i<state.chosen.length;i++)for(const id of state.chosen[i].idx)owner[id]=i;return owner;}
function refillLocal(data,w,h,base,seed,deadline){
  let occ=base.occ,chosen=base.chosen.slice(),s=seed>>>0||1,bestCount=chosen.length,neutral=0;
  while(now()<deadline&&!STOP){
    const holes=[];for(let i=0;i<occ.length;i++)if(!occ[i])holes.push(i);if(!holes.length)break;
    s=rngStep(s);const hc=holes[s%holes.length],owner=ownerForState({chosen},w*h),list=data.cellTo[hc];if(!list.length)break;
    let accepted=false;
    const testCount=Math.min(list.length,10);for(let zz=0;zz<testCount&&now()<deadline;zz++){
      s=rngStep(s);const p=data.placements[list[s%list.length]],conf=new Set();for(const id of p.idx){const oi=owner[id];if(oi>=0)conf.add(oi);}if(conf.size>3)continue;
      const occ2=occ.slice(),keep=[];for(let i=0;i<chosen.length;i++){if(conf.has(i)){for(const id of chosen[i].idx)occ2[id]=0;}else keep.push(chosen[i]);}
      if(!canPlace(p,occ2))continue;place(p,occ2,keep);
      const local=new Set();for(const id of p.idx){local.add(id);for(let r=1;r<=3;r++)for(const[dx,dy]of DIR4)local.add(dataNeighbor(data,id,dx*r,dy*r,w,h));}
      for(const id of local)if(!occ2[id]){
        let bp=-1,bs=-1e9;for(const pi of data.cellTo[id]){const q=data.placements[pi];if(!canPlace(q,occ2))continue;let contact=0;for(const qid of q.idx)for(const[dx,dy]of DIR4)if(occ2[dataNeighbor(data,qid,dx,dy,w,h)])contact++;const sc=contact*5+((rngStep(s^pi)&1023)/8192);if(sc>bs){bs=sc;bp=pi;}}
        if(bp>=0)place(data.placements[bp],occ2,keep);
      }
      if(keep.length>bestCount||(keep.length===bestCount&&neutral<18&&((s>>>3)&7)===0)){
        occ=occ2;chosen=keep;if(keep.length>bestCount){bestCount=keep.length;neutral=0;}else neutral++;accepted=true;break;
      }
    }
    if(!accepted&&++neutral>42)break;
  }
  return{occ,chosen};
}

function improveOneForTwo(data,w,h,state,seed,deadline){
  let s=seed>>>0||1,chosen=state.chosen.slice(),occ=state.occ.slice(),passes=0;
  while(now()<deadline&&!STOP&&passes<12){passes++;const owner=new Int32Array(w*h);owner.fill(-1);for(let i=0;i<chosen.length;i++)for(const id of chosen[i].idx)owner[id]=i;const holes=[];for(let id=0;id<owner.length;id++)if(owner[id]<0)holes.push(id);if(holes.length<data.placements[0].idx.length)break;
    let improved=false;const hs=shuffled(holes,s).slice(0,Math.min(holes.length,180));
    outer:for(const hc of hs){s=rngStep(s^hc);const plist=shuffled(data.cellTo[hc],s).slice(0,Math.min(18,data.cellTo[hc].length));for(const pi of plist){const p=data.placements[pi],conf=new Set();for(const id of p.idx){const o=owner[id];if(o>=0)conf.add(o);if(conf.size>1)break;}if(conf.size!==1)continue;const ti=conf.values().next().value,tile=chosen[ti],pSet=new Set(p.idx),qCand=new Set();for(const id of tile.idx)for(const qi of data.cellTo[id])qCand.add(qi);for(const qi of qCand){if(qi===pi)continue;const q=data.placements[qi];let ok=true;for(const id of q.idx){if(pSet.has(id)){ok=false;break;}const o=owner[id];if(o>=0&&o!==ti){ok=false;break;}}if(!ok)continue;
          // Replace one old tile with two mutually disjoint placements: strict +1 improvement.
          const next=[];for(let i=0;i<chosen.length;i++)if(i!==ti)next.push(chosen[i]);next.push(p,q);chosen=next;occ=new Uint8Array(w*h);for(const z of chosen)for(const id of z.idx)occ[id]=1;improved=true;break outer;
        }}
    }
    if(!improved)break;
  }
  return{occ,chosen};
}
function annealPacking(data,w,h,state,seed,deadline,temperature=.45){
  let s=seed>>>0||1,chosen=state.chosen.slice(),best=chosen.slice(),owner=new Int32Array(w*h);owner.fill(-1);const rebuild=()=>{owner.fill(-1);for(let i=0;i<chosen.length;i++)for(const id of chosen[i].idx)owner[id]=i;};rebuild();let steps=0,noGain=0;
  while(now()<deadline&&!STOP){steps++;let hc=-1;for(let z=0;z<12;z++){s=rngStep(s);const c=s%(w*h);if(owner[c]<0){hc=c;break;}}if(hc<0){for(let c=0;c<owner.length;c++)if(owner[c]<0){hc=c;break;}}if(hc<0)break;const list=data.cellTo[hc];if(!list.length)break;s=rngStep(s);const p=data.placements[list[s%list.length]],conf=new Set();for(const id of p.idx){const o=owner[id];if(o>=0)conf.add(o);}const delta=1-conf.size,accept=delta>0||delta===0||(((s>>>8)&65535)/65535)<Math.max(.002,temperature*.012/(1+conf.size));if(!accept||conf.size>3)continue;
    const next=[];for(let i=0;i<chosen.length;i++)if(!conf.has(i))next.push(chosen[i]);next.push(p);chosen=next;rebuild();
    // Fill any newly exposed cell immediately when possible.
    for(let sweep=0;sweep<2;sweep++){let added=false;for(let c=0;c<owner.length;c++)if(owner[c]<0){const ls=data.cellTo[c];let bp=-1,bs=-1e9;for(let z=0;z<ls.length;z++){const q=data.placements[ls[z]];let ok=true,contact=0;for(const id of q.idx){if(owner[id]>=0){ok=false;break;}for(const[dx,dy]of DIR4)if(owner[dataNeighbor(data,id,dx,dy,w,h)]>=0)contact++;}if(ok&&contact>bs){bs=contact;bp=ls[z];}}if(bp>=0){const q=data.placements[bp];next.push(q);chosen=next;rebuild();added=true;break;}}if(!added)break;}
    if(chosen.length>best.length){best=chosen.slice();noGain=0;}else noGain++;if(noGain>1200)break;
  }
  const occ=new Uint8Array(w*h);for(const p of best)for(const id of p.idx)occ[id]=1;return{occ,chosen:best};
}

function ruinRecreate(data,w,h,state,seed,deadline,aggression=.6){
  let s=seed>>>0||1,chosen=state.chosen.slice(),occ=state.occ.slice(),round=0;
  while(now()<deadline&&!STOP){round++;const owner=ownerForState({chosen},w*h),holes=[];for(let i=0;i<owner.length;i++)if(owner[i]<0)holes.push(i);if(!holes.length)break;s=rngStep(s);const hc=chooseVoidFocus(occ,w,h,true,data.cellTo,s);if(hc<0)break;const hx=hc%w,hy=(hc/w)|0,qTarget=Math.min(chosen.length,3+(s%Math.max(2,Math.round(3+aggression*5)))),near=[];
    // Select nearby occupied tiles on the torus.
    for(let rad=1;rad<=7&&near.length<qTarget;rad++)for(let dy=-rad;dy<=rad&&near.length<qTarget;dy++)for(let dx=-rad;dx<=rad&&near.length<qTarget;dx++){if(Math.abs(dx)+Math.abs(dy)!==rad)continue;const id=mod(hy+dy,h)*w+mod(hx+dx,w),ti=owner[id];if(ti>=0&&!near.includes(ti))near.push(ti);}
    if(!near.length)continue;const selected=new Set(near.slice(0,qTarget)),region=new Set([hc]);for(const ti of selected)for(const id of chosen[ti].idx)region.add(id);for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++)if(Math.abs(dx)+Math.abs(dy)<=5){const id=mod(hy+dy,h)*w+mod(hx+dx,w);if(owner[id]<0)region.add(id);}
    const candSet=new Set();for(const id of region)for(const pi of data.cellTo[id])candSet.add(pi);const cand=[];for(const pi of candSet){const p=data.placements[pi];let ok=true;for(const id of p.idx){const o=owner[id];if(o>=0&&!selected.has(o)){ok=false;break;}}if(ok)cand.push(p);}if(!cand.length)continue;
    let bestLocal=[],tries=Math.min(36,8+Math.round(aggression*28));for(let tr=0;tr<tries&&now()<deadline;tr++){s=rngStep(s);const order=shuffled(cand,s),used=new Set(),pick=[];for(const p of order){let ok=true;for(const id of p.idx)if(used.has(id)){ok=false;break;}if(!ok)continue;for(const id of p.idx)used.add(id);pick.push(p);}if(pick.length>bestLocal.length)bestLocal=pick;if(bestLocal.length>selected.size)break;}
    if(bestLocal.length>=selected.size){const next=[];for(let i=0;i<chosen.length;i++)if(!selected.has(i))next.push(chosen[i]);next.push(...bestLocal);if(bestLocal.length>selected.size||((s>>>10)&15)===0){chosen=next;occ=new Uint8Array(w*h);for(const p of chosen)for(const id of p.idx)occ[id]=1;}}
    if(round>80&&now()+4>=deadline)break;
  }
  return{occ,chosen};
}
function localExactRepack(data,w,h,state,seed,deadline,strength=.75){
  let s=seed>>>0||1,chosen=state.chosen.slice(),occ=state.occ.slice(),attempts=0;
  while(now()<deadline&&!STOP&&attempts++<18){
    const owner=ownerForState({chosen},w*h),holes=[];for(let i=0;i<owner.length;i++)if(owner[i]<0)holes.push(i);if(!holes.length)break;
    // Eliminate small disconnected void components first; within that component choose a scarce cell.
    const hc=chooseVoidFocus(occ,w,h,true,data.cellTo,s);if(hc<0)break;
    let starters=shuffled(data.cellTo[hc],s).slice(0,Math.min(18,data.cellTo[hc].length)).map(pi=>data.placements[pi]),improved=false;
    for(const p0 of starters){if(now()>=deadline)break;const conf=new Set();for(const id of p0.idx){const o=owner[id];if(o>=0)conf.add(o);}const kicked=conf.size,maxKick=2+Math.round(strength*2);if(kicked===0){if(canPlace(p0,occ)){place(p0,occ,chosen);improved=true;break;}continue;}if(kicked>maxKick)continue;
      const selected=new Set(conf),region=new Set();for(const ti of selected)for(const id of chosen[ti].idx)region.add(id);const hx=hc%w,hy=(hc/w)|0,rad=2+Math.round(strength*2);for(let dy=-rad;dy<=rad;dy++)for(let dx=-rad;dx<=rad;dx++)if(Math.abs(dx)+Math.abs(dy)<=rad){const id=mod(hy+dy,h)*w+mod(hx+dx,w);if(owner[id]<0)region.add(id);}
      const candSet=new Set();for(const id of region)for(const pi of data.cellTo[id])candSet.add(pi);let cand=[];for(const pi of candSet){const p=data.placements[pi];let ok=true,hits=0;for(const id of p.idx){const o=owner[id];if(o>=0&&!selected.has(o)){ok=false;break;}if(owner[id]<0)hits++;}if(ok)cand.push({p,hits,noise:((rngStep(s^pi)&1023)/1024)});}cand.sort((a,b)=>b.hits-a.hits||b.noise-a.noise);cand=cand.slice(0,Math.min(110,cand.length));
      const target=kicked+1,used=new Set(p0.idx),pick=[p0];let nodes=0,found=null;
      function dfs(at){if(pick.length>=target){found=pick.slice();return true;}if(now()>=deadline||++nodes>5200)return false;for(let i=at;i<cand.length;i++){const q=cand[i].p;if(q===p0)continue;let ok=true;for(const id of q.idx)if(used.has(id)){ok=false;break;}if(!ok)continue;for(const id of q.idx)used.add(id);pick.push(q);if(dfs(i+1))return true;pick.pop();for(const id of q.idx)used.delete(id);}return false;}
      dfs(0);if(found){const next=[];for(let i=0;i<chosen.length;i++)if(!selected.has(i))next.push(chosen[i]);next.push(...found);chosen=next;occ=new Uint8Array(w*h);for(const q of chosen)for(const id of q.idx)occ[id]=1;improved=true;break;}
    }
    if(!improved)break;
  }
  return{occ,chosen};
}
function densityMetrics(w,h,chosen,occ,mode,opt={}){
  const valid=validateState(w,h,{chosen});if(!valid)return{invalid:true,coverage:0,holes:[],score:-1e12};occ=valid.occ;const covered=countBitsOcc(occ),coverage=covered/(w*h),oriCounts=new Map(),owner=new Int32Array(w*h);owner.fill(-1);
  for(let i=0;i<chosen.length;i++){const p=chosen[i];oriCounts.set(p.ori,(oriCounts.get(p.ori)||0)+1);for(const id of p.idx)owner[id]=i;}
  const adjCounts=new Map();for(let id=0;id<owner.length;id++){const a=owner[id];if(a<0)continue;for(const[dx,dy]of[[1,0],[0,1]]){const b=owner[torusNeighbor(id,dx,dy,w,h)];if(b<0||b===a)continue;const oa=chosen[a].ori,ob=chosen[b].ori,k=oa<=ob?`${oa}:${ob}`:`${ob}:${oa}`;adjCounts.set(k,(adjCounts.get(k)||0)+1);}}
  const adjEntropy=entropy(adjCounts),oriEntropy=entropy(oriCounts),voids=voidConnectivity(occ,w,h,true),entropyMix=(adjEntropy+oriEntropy)*.5,complexity=Math.max(0,Math.min(1,opt.complexity??.56)),regularity=Math.max(0,Math.min(1,opt.regularity??.48)),pieces=Math.max(1,chosen.length),scaleNorm=Math.max(0,Math.min(1,Math.log2(pieces+1)/Math.log2(41))),structure=complexity*entropyMix+regularity*(1-entropyMix),scaleFit=complexity*scaleNorm+regularity*(1-scaleNorm),voidQuality=(voids.components===0?1.25:voids.cohesion)-Math.min(1,voids.perimeter/Math.max(1,w*h))*.28-Math.max(0,voids.components-1)*.045;
  const score=coverage*64+structure*18+scaleFit*9+voidQuality*13+(coverage>=.999999?12:0);
  return{coverage,fundArea:NaN,adjEntropy,oriEntropy,shortestPeriod:NaN,holeSpread:NaN,voidComponents:voids.components,voidLargestShare:voids.largestShare,voidPerimeter:voids.perimeter,score,aesthetics:score-coverage*58,holes:voids.holes};
}
function makeDensityResult(w,h,state,mode,engine,labelOverride=null,opt={}){const safe=validateState(w,h,state);if(!safe)return null;const m=densityMetrics(w,h,safe.chosen,safe.occ,mode,opt);if(m.invalid)return null;if(mode==='cell'&&m.coverage>=.999999){const r=makePeriodicResult(w,h,safe.chosen,engine,labelOverride||'periodic cell',opt);if(!r)return null;r.metrics.coverageBand=opt.coverageBand??.03;r.metrics.tileCount=r.metrics.primitiveTileCount||safe.chosen.length;return r;}const info=periodicPatternInfo(w,h,safe.chosen),kind=mode==='cell'?'cell':'field';return{id:`D${++RESULT_SEQ}`,kind,label:labelOverride||(mode==='cell'?'periodic cell':mode==='drift'?'density drift':'density field'),w,h,engine,placements:safe.chosen.map(p=>({cells:p.cells,rawCells:p.rawCells||p.cells,ori:p.ori,anchor:p.anchor})),colorClasses:info?.colorClasses||null,holes:m.holes,metrics:{coverage:m.coverage,fundArea:info?.fundArea??m.fundArea,adjEntropy:m.adjEntropy,oriEntropy:m.oriEntropy,shortestPeriod:info?.shortestPeriod??m.shortestPeriod,voidComponents:m.voidComponents,voidLargestShare:m.voidLargestShare,voidPerimeter:m.voidPerimeter,score:m.score,coverageBand:opt.coverageBand??.03,tileCount:info?.primitiveTileCount||safe.chosen.length}};}
function stateFromFieldResult(r,ctx,w,h){if(!r||!(r.kind==='field'||r.kind==='cell'||r.kind==='periodic')||r.w!==w||r.h!==h)return null;const occ=new Uint8Array(w*h),chosen=[];for(const rp of r.placements||[]){const p=placementFromAnchor(ctx.oris,Number(rp.ori)||0,Number(rp.anchor?.[0])||0,Number(rp.anchor?.[1])||0,w,h);if(!p||!canPlace(p,occ))return null;place(p,occ,chosen);}return{occ,chosen};}
function betterDensity(a,b){if(!a)return b;if(!b)return a;const ca=a.metrics?.coverage||0,cb=b.metrics?.coverage||0,band=Math.max(.005,Math.min(.08,a.metrics?.coverageBand??b.metrics?.coverageBand??.03)),d=ca-cb;if(Math.abs(d)>band)return d>0?a:b;const va=a.metrics?.voidComponents??999,vb=b.metrics?.voidComponents??999,la=a.metrics?.voidLargestShare??0,lb=b.metrics?.voidLargestShare??0,pa=a.metrics?.voidPerimeter??1e9,pb=b.metrics?.voidPerimeter??1e9;const vqa=(va===0?30:22/va)+la*12-Math.min(12,pa/Math.max(1,a.w*a.h)*18),vqb=(vb===0?30:22/vb)+lb*12-Math.min(12,pb/Math.max(1,b.w*b.h)*18);if(Math.abs(vqa-vqb)>2.2)return vqa>vqb?a:b;const qa=(a.metrics?.score||0)+vqa+(ca>=.999999?12:0),qb=(b.metrics?.score||0)+vqb+(cb>=.999999?12:0);if(Math.abs(qa-qb)>1e-9)return qa>=qb?a:b;return ca>=cb?a:b;}
function latticeDensitySeed(ctx){
  const ck=ctx.shapeSig;if(LATTICE_CACHE.has(ck))return LATTICE_CACHE.get(ck);
  const shape=ctx.oris[0],n=ctx.n,maxDet=Math.min(72,Math.max(n+10,n*4));
  for(let D=n;D<=maxDet;D++)for(let a=1;a<=D;a++)if(D%a===0){const c=D/a;for(let b=0;b<a;b++){
    let valid=true;for(let i=0;i<shape.length&&valid;i++)for(let j=i+1;j<shape.length;j++){const x=shape[i][0]-shape[j][0],y=shape[i][1]-shape[j][1];if(y%c===0&&(x-b*(y/c))%a===0){valid=false;break;}}if(!valid)continue;
    const anchors=[],seen=new Set();for(let q=0;q<D;q++)for(let p0=0;p0<D;p0++){const x=mod(p0*a+q*b,D),y=mod(q*c,D),k=`${x},${y}`;if(!seen.has(k)){seen.add(k);anchors.push([x,y]);}}if(anchors.length!==D)continue;
    const occ=new Uint8Array(D*D),chosen=[];let ok=true;for(const[ax,ay]of anchors){const p=placementFromAnchor(ctx.oris,0,ax,ay,D,D);if(!p||!canPlace(p,occ)){ok=false;break;}place(p,occ,chosen);}if(!ok)continue;
    const r=chosen.length*n===D*D?makePeriodicResult(D,D,chosen,'L','lattice periodic',ctx.opt):makeDensityResult(D,D,{occ,chosen},'near','L','lattice density',ctx.opt);lruSet(LATTICE_CACHE,ck,r,LATTICE_CACHE_LIMIT);return r;
  }}
  lruSet(LATTICE_CACHE,ck,null,LATTICE_CACHE_LIMIT);return null;
}
function maxPackingBranch(data,incumbent,deadline,seed,nodeLimit=180000){
  const A=data.area,n=data.placements[0]?.idx.length||1,target=Math.floor(A/n);if(!data.placements.length||!incumbent||incumbent.chosen.length>=target)return incumbent;const occ=new Uint8Array(A),chosen=[];let best=incumbent.chosen.slice(),nodes=0,s=seed>>>0||1;
  function feasible(p){for(const id of p.idx)if(occ[id])return false;return true;}
  function dfs(uncovered){if(STOP||now()>=deadline||++nodes>nodeLimit)return false;if(chosen.length>best.length){best=chosen.slice();if(best.length>=target)return true;}if(chosen.length+Math.floor(uncovered/n)<=best.length)return false;let c=-1,opts=null,min=1e9;
    for(let id=0;id<A;id++)if(!occ[id]){const a=[];for(const pi of data.cellTo[id]){const p=data.placements[pi];if(feasible(p)){a.push(pi);if(a.length>=min)break;}}if(a.length<min){min=a.length;c=id;opts=a;if(min===0)break;}}
    if(c<0)return false;if(opts?.length){opts=shuffled(opts,s=rngStep(s));for(const pi of opts){const p=data.placements[pi];for(const id of p.idx)occ[id]=1;chosen.push(p);if(dfs(uncovered-n))return true;chosen.pop();for(const id of p.idx)occ[id]=0;if(STOP||now()>=deadline)return false;}}
    // Leaving a cell uncovered is tried last, so the search behaves like a maximum-packing solver.
    occ[c]=2;dfs(uncovered-1);occ[c]=0;return false;
  }
  dfs(A);if(best.length<=incumbent.chosen.length)return incumbent;const out=new Uint8Array(A);for(const p of best)for(const id of p.idx)out[id]=1;return{occ:out,chosen:best};
}

function seedStateForAlgorithm(ctx,data,w,h,seed,round=0){
  const alg=ctx.algorithm||'auto';if(alg==='constructive')return constructivePack(data,w,h,seed,ctx.opt);if(alg==='dense'){const a=mrvPack(data,w,h,seed,Math.max(16,Math.min(40,Math.round(Math.sqrt(w*h))))),b=packFromPlacements(data,w,h,seed^0x85ebca6b,round%3);return a.chosen.length>=b.chosen.length?a:b;}
  const a=constructivePack(data,w,h,seed,ctx.opt),b=mrvPack(data,w,h,seed^0x7f4a7c15,Math.max(14,Math.min(34,Math.round(Math.sqrt(w*h))))),c=packFromPlacements(data,w,h,seed^0x85ebca6b,round%3);return bestOfStates([a,b,c]);
}
function quickDensityField(ctx){
  const dims=densityDimensionPortfolio(ctx,true);let best=latticeDensitySeed(ctx),seed=hashSeed(ctx.shapeSig,0x51ed,ctx.runNonce);for(let di=0;di<dims.length&&!STOP;di++){const d=dims[di],data=torusData(ctx,d.w,d.h);if(!data.placements.length)continue;seed=rngStep(seed);const st=seedStateForAlgorithm(ctx,data,d.w,d.h,seed,di),r=makeDensityResult(d.w,d.h,st,'near',ctx.algorithm==='constructive'?'C':ctx.algorithm==='dense'?'D':'A','density field',ctx.opt);best=betterDensity(best,r);}return best;
}
function densityExactProbe(ctx,deadline,dimsHint=null){
  if(ctx.knownNonTiler||now()>=deadline)return null;const seen=new Set(),dims=[];
  const add=(w,h,bonus=0)=>{w=Math.round(w);h=Math.round(h);const area=w*h,k=`${w}x${h}`,pieceLimit=Math.max(1,Math.min(20,ctx.opt.cellPieces||20));if(w<1||h<1||area<ctx.n||area>520||area%ctx.n||(ctx.mode==='cell'&&area/ctx.n>pieceLimit)||seen.has(k))return;seen.add(k);dims.push({w,h,area,bonus,aspect:Math.abs(Math.log(w/h))});};
  for(const d of dimsHint||[])add(d.w,d.h,3+(d.rank||0));
  // Systematic compact rectangles are cheap enough to probe and catch many tilings that
  // heuristic field sizing would never nominate (e.g. 8×16, 9×18).
  for(let w=1;w<=34;w++)for(let h=1;h<=34;h++)add(w,h,0);
  dims.sort((a,b)=>(a.area-b.area)*.08+(a.aspect-b.aspect)*2-(a.bonus-b.bonus)*.22||a.w-b.w);let seed=hashSeed(ctx.shapeSig,0xdecafbad,ctx.runNonce);
  for(let i=0;i<dims.length;i++){if(STOP||now()>=deadline)break;const d=dims[i],data=torusData(ctx,d.w,d.h);if(!data.placements.length||data.cellTo.some(x=>!x.length))continue;let avg=0,samples=0,step=Math.max(1,Math.floor(data.cellTo.length/32));for(let j=0;j<data.cellTo.length;j+=step){avg+=data.cellTo[j].length;samples++;}avg/=Math.max(1,samples);const remaining=deadline-now(),left=Math.max(1,dims.length-i),slice=Math.min(115,Math.max(28,remaining/Math.min(12,left)));if(slice<9)break;seed=rngStep(seed);const nodes=Math.max(70000,Math.min(520000,Math.round(80000+d.area*Math.max(1,avg)*28))),r=exactCoverDLX(data,now()+slice,seed,nodes,null,false);if(r.solution)return makePeriodicResult(d.w,d.h,r.solution,'DLX','periodic',ctx.opt);}
  return null;
}
function cellResultAllowed(ctx,r){if(ctx.mode!=='cell'||!r)return true;const limit=Math.max(1,Math.min(20,ctx.opt.cellPieces||10));return (r.placements?.length||0)<=limit;}
function solveDensityField(ctx,mode,labelOverride=null,initialResult=null){
  const opt=ctx.opt,target=Math.max(.95,Math.min(1,opt.targetCoverage??.9995)),portfolio=densityDimensionPortfolio(ctx,false),byDim=new Map(),stats=new Map();let seed=hashSeed(ctx.shapeSig,0xabc98388,ctx.runNonce),lattice0=latticeDensitySeed(ctx),best=betterDensity(initialResult,cellResultAllowed(ctx,lattice0)?lattice0:null),iter=0;
  const key=d=>`${d.w}x${d.h}`;const stat=d=>{const k=key(d);if(!stats.has(k))stats.set(k,{visits:0,best:0,gain:0});return stats.get(k);};
  const putDim=(r)=>{if(!r?.w||!r?.h)return false;const k=`${r.w}x${r.h}`,old=byDim.get(k);if(!old||betterDensity(old,r)===r){byDim.set(k,r);const st=stats.get(k)||{visits:0,best:0,gain:0};const oldCov=st.best;st.best=Math.max(st.best,r.metrics?.coverage||0);st.gain=Math.max(st.gain,st.best-oldCov);stats.set(k,st);}const oldBest=best;best=betterDensity(best,r);return best!==oldBest;};
  if(initialResult)putDim(initialResult);const lattice=lattice0;if(lattice&&cellResultAllowed(ctx,lattice))putDim(lattice);for(const resumed of ctx.resumeResults||[])if(resumed&&(resumed.kind==='field'||resumed.kind==='cell'||resumed.kind==='periodic')&&cellResultAllowed(ctx,resumed))putDim(resumed);
  if(best){ctx.reporter.preview(best,{phase:'pack',engine:best.engine||'Λ',bestCoverage:best.metrics?.coverage},true);ctx.reporter.partial([best],{phase:'pack',engine:best.engine||'Λ'},true);}
  // Stage 1: cheap scouting across the whole geometry portfolio. It is intentionally
  // uniform across n, and prevents the solver from committing to the first square-ish torus.
  const scoutEnd=Math.min(ctx.deadline,now()+Math.min(360,Math.max(120,(ctx.deadline-ctx.start)*.12)));for(let di=0;di<portfolio.length&&!STOP&&now()<scoutEnd;di++){const d=portfolio[di],data=torusData(ctx,d.w,d.h);if(!data.placements.length)continue;seed=rngStep(seed);let st=packFromPlacements(data,d.w,d.h,seed,di%3);if(di<4&&now()+8<scoutEnd){const m=mrvPack(data,d.w,d.h,seed^0x7f4a7c15,Math.max(12,Math.min(24,Math.round(Math.sqrt(d.area)))));if(m.chosen.length>st.chosen.length)st=m;}const r=makeDensityResult(d.w,d.h,st,mode,'A',labelOverride,opt);stat(d).visits++;if(putDim(r)){ctx.reporter.preview(best,{phase:'pack',engine:'A',bestCoverage:best.metrics.coverage},di===0);ctx.reporter.partial([best],{phase:'pack',engine:'A'},di===0);}}
  if(best?.metrics.coverage>=.999999)return[best];
  // Exact cover is an opportunistic shortcut selected by observed matrix size, never by n.
  const total=ctx.deadline-ctx.start,agg=Math.max(0,Math.min(1,opt.aggression??.58)),exactFraction=(ctx.algorithm==='constructive'?.24:ctx.algorithm==='dense'?.34:.42)*(.75+agg*.5),exactMs=Math.min(6500,Math.max(180,total*exactFraction));if(!ctx.knownNonTiler&&now()+80<ctx.deadline){const exactEnd=Math.min(ctx.deadline,now()+exactMs),hnf=hnfExactProbe(ctx,Math.min(exactEnd,now()+exactMs*.34));let exact=hnf||densityExactProbe(ctx,exactEnd,portfolio);if(exact){exact=refineExactPeriodic(exact,ctx,Math.min(ctx.deadline,now()+Math.min(700,total*.08)));ctx.reporter.preview(exact,{phase:'exact',engine:exact.engine||'DLX',bestCoverage:1},true);ctx.reporter.partial([exact],{phase:'exact',engine:exact.engine||'DLX'},true);return[exact];}}
  // Stage 2: deepen the best few geometries with the selected personality.
  const ranked=portfolio.slice().sort((a,b)=>(stat(b).best-stat(a).best)||(b.rank-a.rank)).slice(0,Math.min(6,portfolio.length));for(let di=0;di<ranked.length&&!STOP&&now()<ctx.deadline;di++){const d=ranked[di],data=torusData(ctx,d.w,d.h);seed=rngStep(seed);let st=seedStateForAlgorithm(ctx,data,d.w,d.h,seed,di),engine=ctx.algorithm==='constructive'?'C':ctx.algorithm==='dense'?'D':'A';const upper=Math.floor(d.area/ctx.n),gap=upper-st.chosen.length;if(gap>0&&gap<=3&&d.area<=190&&now()+25<ctx.deadline)st=maxPackingBranch(data,st,Math.min(ctx.deadline,now()+Math.min(180,35+d.area*.45)),seed^0x31415926,150000);const r=makeDensityResult(d.w,d.h,st,mode,engine,labelOverride,opt);stat(d).visits++;if(putDim(r)){ctx.reporter.preview(best,{phase:'pack',engine,bestCoverage:best.metrics.coverage},true);ctx.reporter.partial([best],{phase:'pack',engine},true);}}
  if(best?.metrics.coverage>=.999999)return[best];
  const minimumRun=Math.min(1800,Math.max(500,total*.04));
  function chooseDim(){let choice=portfolio[0],score=-1e9;const log=Math.log(iter+2);for(const d of portfolio){const st=stat(d),uncertainty=.028*Math.sqrt(log/(st.visits+1)),base=st.best||.45,scaleBonus=(d.rank||0)*.003,gain=st.gain*.8,v=base+uncertainty+gain+scaleBonus;if(v>score){score=v;choice=d;}}return choice;}
  while(!STOP&&now()<ctx.deadline){iter++;seed=rngStep(seed);const d=chooseDim(),w=d.w,h=d.h,data=torusData(ctx,w,h);if(!data.placements.length){stat(d).visits+=3;continue;}const ds=stat(d);ds.visits++;const k=key(d),baseR=byDim.get(k),base=baseR?stateFromFieldResult(baseR,ctx,w,h):null;let st,engine;
    if(base&&iter%4!==0){st=base;const slice=Math.min(ctx.deadline,now()+48);st=improveOneForTwo(data,w,h,st,seed^0xc2b2ae35,slice);if(now()<ctx.deadline-8)st=localExactRepack(data,w,h,st,seed^0x6d2b79f5,Math.min(ctx.deadline,now()+82),.82);if(now()<ctx.deadline-8)st=ruinRecreate(data,w,h,st,seed^0x27d4eb2d,Math.min(ctx.deadline,now()+66),.72);if(now()<ctx.deadline-8&&ctx.algorithm!=='constructive')st=annealPacking(data,w,h,st,seed^0x165667b1,Math.min(ctx.deadline,now()+34),.46);if(now()<ctx.deadline-8)st=refillLocal(data,w,h,st,seed^0x4cf5ad43,Math.min(ctx.deadline,now()+34));engine=ctx.algorithm==='constructive'?'C·R':'D·LNS';}
    else{st=seedStateForAlgorithm(ctx,data,w,h,seed,iter);engine=ctx.algorithm==='constructive'?'C':ctx.algorithm==='dense'?'D':'A';if(now()<ctx.deadline-10&&iter%2===0)st=localExactRepack(data,w,h,st,seed^0x9e3779b9,Math.min(ctx.deadline,now()+65),.72);const gap=Math.floor((w*h)/ctx.n)-st.chosen.length;if(gap>0&&gap<=2&&w*h<=190&&now()+18<ctx.deadline)st=maxPackingBranch(data,st,Math.min(ctx.deadline,now()+90),seed^0x27182818,90000);}
    const r=makeDensityResult(w,h,st,mode,engine,labelOverride,opt),oldGlobal=best;putDim(r);if(best!==oldGlobal){ctx.reporter.preview(best,{phase:'pack',engine:best.engine,current:iter,bestCoverage:best.metrics.coverage,bestScore:best.metrics.score});ctx.reporter.partial([best],{phase:'pack',engine:best.engine});}
    ctx.reporter.progress({phase:'pack',engine:best?.engine||engine,current:iter,percent:Math.min(.999,(now()-ctx.start)/(ctx.deadline-ctx.start)),w:best?.w||w,h:best?.h||h,bestCoverage:best?.metrics?.coverage,bestScore:best?.metrics?.score});
    const elapsed=now()-ctx.start;if(best?.metrics.coverage>=.999999)break;if(best?.metrics.coverage>=target&&elapsed>=minimumRun)break;
  }
  return best?[best]:[];
}



// --- v9 periodic spectrum + companion --------------------------------------
// Every rank-2 translation lattice is enumerated once in Hermite normal form
// L=< (a,0), (b,c) >, 0<=b<a. This is equivalent to enumerating all integer
// bases ((a,b),(c,d)) modulo unimodular basis changes, without redundant copies.
function gcdInt(a,b){a=Math.abs(a|0);b=Math.abs(b|0);while(b){const t=a%b;a=b;b=t;}return a||1;}
const SPECTRUM_PRIORITY_PERIODS=new Set([1,2,3,4,6,8]);
function spectrumPriority(k){return SPECTRUM_PRIORITY_PERIODS.has(k)?1:0;}
function spectrumVariantQuota(ctx,k){const total=Math.max(1,ctx.deadline-ctx.start),base=total<4000?5:total<15000?8:12;return base+(spectrumPriority(k)?(total<4000?2:total<15000?4:8):0);}
function latticeCandidatesForTarget(ctx,target,gapLo=0,gapHi=null){const n=ctx.n,base=n*target,out=[],hi=gapHi==null?n-1:Math.max(gapLo,gapHi);for(let gap=Math.max(0,gapLo);gap<=hi;gap++){const area=base+gap;for(let a=1;a<=area;a++){if(area%a)continue;const c=area/a;for(let b=0;b<a;b++){const shear=Math.min(b,a-b),u=a,v=Math.hypot(shear,c),aspect=Math.abs(Math.log(Math.max(1e-9,u/v))),longness=Math.max(u,v)/Math.sqrt(area),rect=b===0?-.24:0;out.push({target,area,a,b,c,gap,rank:gap*110+aspect*1.35+longness*.42+shear/Math.max(1,a)*.18+rect,key:`${a},${b},${c}`});}}}out.sort((x,y)=>x.rank-y.rank||x.a-y.a||x.b-y.b||x.c-y.c);return out;}
function stateFromLatticeResult(r,ctx,cand){if(!r?.placements?.length)return null;const lat=r.lattice||{a:r.w,b:0,c:r.h};if(lat.a!==cand.a||(lat.b||0)!==cand.b||lat.c!==cand.c)return null;const occ=new Uint8Array(cand.area),chosen=[];for(const rp of r.placements){const p=placementFromAnchorHNF(ctx.oris,Number(rp.ori)||0,Number(rp.anchor?.[0])||0,Number(rp.anchor?.[1])||0,cand.a,cand.b,cand.c);if(!p||!canPlace(p,occ))return null;place(p,occ,chosen);}return{occ,chosen};}
function projectedExactSubstructureState(ctx,r,cand,deadline=Infinity){
  if(!r?.placements?.length||cand.gap!==0)return null;const target=cand.target,uniq=new Map();
  for(const rp of r.placements){const oi=Number(rp.ori)||0,ax=Number(rp.anchor?.[0])||0,ay=Number(rp.anchor?.[1])||0,p=placementFromAnchorHNF(ctx.oris,oi,ax,ay,cand.a,cand.b,cand.c);if(!p)continue;uniq.set(p.idx.join(','),p);}
  const ps=[...uniq.values()];if(ps.length<target)return null;const cellTo=Array.from({length:cand.area},()=>[]);for(let i=0;i<ps.length;i++)for(const id of ps[i].idx)cellTo[id].push(i);if(cellTo.some(x=>!x.length))return null;
  const occ=new Uint8Array(cand.area),chosen=[];let nodes=0;function dfs(){if(++nodes>6000||((nodes&63)===0&&now()>=deadline))return false;if(chosen.length===target){for(const v of occ)if(!v)return false;return true;}let cell=-1,opts=null,min=1e9;for(let id=0;id<occ.length;id++)if(!occ[id]){const a=[];for(const pi of cellTo[id])if(canPlace(ps[pi],occ))a.push(pi);if(!a.length)return false;if(a.length<min){min=a.length;cell=id;opts=a;if(min===1)break;}}if(cell<0)return false;for(const pi of opts){const p=ps[pi];place(p,occ,chosen);if(dfs())return true;chosen.pop();for(const id of p.idx)occ[id]=0;if(now()>=deadline)return false;}return false;}if(now()>=deadline||!dfs())return null;return{occ,chosen:chosen.slice()};
}
function mineSubstructureResults(ctx,r,limit,deadline=ctx.deadline){
  const cov=r?.metrics?.coverage||0,p=spectrumPeriod(r);if(cov<1-1e-12||p<=1)return[];const targets=[];for(let k=1;k<Math.min(p,limit+1);k++)targets.push(k);targets.sort((a,b)=>Number(p%b===0)-Number(p%a===0)||spectrumPriority(b)-spectrumPriority(a)||a-b);const out=[];
  for(const k of targets){if(now()>=deadline)break;let found=0;const cands=latticeCandidatesForTarget(ctx,k,0,0);for(const cand of cands.slice(0,spectrumPriority(k)?220:140)){if(now()>=deadline)break;const st=projectedExactSubstructureState(ctx,r,cand,deadline);if(!st)continue;const q=makeLatticeSpectrumResult(ctx,cand,st,ctx.mode==='companion'?'companion':'cell','Σ↓');if(q&&q.metrics?.coverage>=1-1e-12){q.metrics.substructureFrom=p;out.push(q);if(++found>=2)break;}}}
  return out;
}
function reduceStateToMinimalLattice(ctx,data,state){const info=periodicPatternInfoHNF(data.a,data.b||0,data.c,state.chosen);if(!info)return null;const L=info.minimalLattice||{a:data.a,b:data.b||0,c:data.c,det:data.area};if(L.det===data.area)return{data,state,info};const d=hnfData(ctx,L.a,L.b,L.c),seen=new Set(),chosen=[],occ=new Uint8Array(d.area);for(const p of state.chosen){const q=placementFromAnchorHNF(ctx.oris,p.ori,p.anchor[0],p.anchor[1],L.a,L.b,L.c);if(!q)continue;const sig=q.idx.join(',');if(seen.has(sig))continue;seen.add(sig);if(!canPlace(q,occ))return null;place(q,occ,chosen);}if(!chosen.length)return null;const reduced={occ,chosen},check=periodicPatternInfoHNF(L.a,L.b,L.c,chosen);if(!check)return null;return{data:d,state:reduced,info:check};}
function periodicMetricsHNF(ctx,data,state,mode){const safe=validateState(data.a,data.c,state);if(!safe)return null;const a=data.a,b=data.b||0,c=data.c,info=periodicPatternInfoHNF(a,b,c,safe.chosen);if(!info)return null;const occ=safe.occ,coverage=info.coverage,owner=info.owner,oriCounts=new Map(),adjCounts=new Map();for(const p of safe.chosen)oriCounts.set(p.ori,(oriCounts.get(p.ori)||0)+1);for(let id=0;id<owner.length;id++){const x=owner[id];if(x<0)continue;for(const[dx,dy]of[[1,0],[0,1]]){const y=owner[hnfNeighborId(id,dx,dy,a,b,c)];if(y<0||y===x)continue;const ox=safe.chosen[x].ori,oy=safe.chosen[y].ori,k=ox<=oy?`${ox}:${oy}`:`${oy}:${ox}`;adjCounts.set(k,(adjCounts.get(k)||0)+1);}}
  const adjEntropy=entropy(adjCounts),oriEntropy=entropy(oriCounts),voids=periodicVoidSummaryHNF(occ,a,b,c),complexity=Math.max(0,Math.min(1,ctx.opt.complexity??.56)),regularity=Math.max(0,Math.min(1,ctx.opt.regularity??.48)),pieces=Math.max(1,info.primitiveTileCount||safe.chosen.length),scaleNorm=Math.max(0,Math.min(1,Math.log2(pieces+1)/Math.log2(31))),structure=complexity*(adjEntropy+oriEntropy)*.5+regularity*(1-(adjEntropy+oriEntropy)*.5),voidQuality=(voids.components===0?1.3:voids.cohesion)-Math.min(1,voids.perimeter/Math.max(1,data.area))*.25-Math.max(0,voids.components-1)*.05;let companion=null,combinedCoverage=coverage;
  if(mode==='companion'){const q=companionFromVoid(occ,a,b,c,ctx.group,ctx.deadline,hashSeed(ctx.shapeSig,a,b,c,safe.chosen.length,ctx.runNonce));if(!q.valid)return null;companion={area:q.area,copies:q.copies,totalCells:q.totalCells,coverage:q.coverage,shapeSig:q.shapeSig,group:ctx.group,components:q.components,placements:q.placements,remaining:q.remaining,patternKind:q.patternKind,componentClasses:q.componentClasses,exactResidual:!!q.exact};combinedCoverage=Math.min(1,(info.covered+q.totalCells)/data.area);}
  const companionBonus=mode==='companion'?combinedCoverage*135+coverage*24+(companion?.exactResidual?22:0)+(Number.isFinite(companion?.area)?10/(1+companion.area):0):0,score=coverage*1000+structure*36+scaleNorm*(complexity*18-regularity*6)+voidQuality*28+companionBonus;
  return{safe,info,voids,companion,metrics:{coverage,combinedCoverage,fundArea:info.fundArea,adjEntropy,oriEntropy,shortestPeriod:info.shortestPeriod,translationSymmetries:info.translationSymmetries,primitiveTileCount:info.primitiveTileCount,tileCount:safe.chosen.length,voidComponents:voids.components,voidLargestShare:voids.largestShare,voidPerimeter:voids.perimeter,companionArea:companion?.area??NaN,companionCopies:companion?.copies??0,companionCoverage:companion?.coverage??NaN,companionPattern:companion?.patternKind||'',companionComponentClasses:companion?.componentClasses??voids.components,unmatchedVoid:companion?.remaining?.length??voids.holes.length,score}};
}
function makeLatticeSpectrumResult(ctx,cand,state,mode,engine){const baseData=hnfData(ctx,cand.a,cand.b,cand.c),red=reduceStateToMinimalLattice(ctx,baseData,state);if(!red)return null;const L={a:red.data.a,b:red.data.b||0,c:red.data.c},z=periodicMetricsHNF(ctx,red.data,red.state,mode);if(!z||!z.safe.chosen.length)return null;const exact=z.metrics.coverage>=.999999,holes=mode==='companion'?(z.companion?.remaining||z.voids.holes):z.voids.holes,r={id:`S${++RESULT_SEQ}`,kind:exact?'periodic':'cell',label:mode==='companion'?'companion':'periodic cell',w:L.a,h:L.c,lattice:{a:L.a,b:L.b,c:L.c,u:[L.a,0],v:[L.b,L.c],det:L.a*L.c,key:`${L.a},${L.b},${L.c}`},engine,placements:z.safe.chosen.map(p=>({cells:p.cells,rawCells:p.rawCells||p.cells,ori:p.ori,anchor:p.anchor})),colorClasses:z.info.colorClasses,holes,metrics:{...z.metrics,periodTarget:cand.target,latticeArea:L.a*L.c,searchLatticeArea:cand.area}};if(mode==='companion')r.companion=z.companion;return r;}
function spectrumPeriod(r){const p=Number(r?.metrics?.primitiveTileCount??r?.metrics?.tileCount??r?.placements?.length);return Number.isFinite(p)?Math.max(0,Math.round(p)):0;}
function betterSpectrum(a,b,mode){if(!a)return b;if(!b)return a;const ca=a.metrics?.coverage||0,cb=b.metrics?.coverage||0;if(mode==='companion'){const xa=a.metrics?.combinedCoverage??ca,xb=b.metrics?.combinedCoverage??cb;if(Math.abs(xa-xb)>1e-10)return xa>xb?a:b;if(Math.abs(ca-cb)>1e-10)return ca>cb?a:b;const aa=Number.isFinite(a.metrics?.companionArea)?a.metrics.companionArea:1e9,ab=Number.isFinite(b.metrics?.companionArea)?b.metrics.companionArea:1e9;if(Math.abs(aa-ab)>1e-10)return aa<ab?a:b;}else if(Math.abs(ca-cb)>1e-10)return ca>cb?a:b;const sa=a.metrics?.score||0,sb=b.metrics?.score||0;if(Math.abs(sa-sb)>1e-9)return sa>sb?a:b;return (a.metrics?.latticeArea||a.w*a.h)<=(b.metrics?.latticeArea||b.w*b.h)?a:b;}
function trimSpectrumAlternatives(arr,mode,cap){
  if(arr.length<=cap)return arr;if(mode!=='companion')return arr.slice(0,cap);
  const keep=[],seen=new Set(),add=r=>{if(!r)return;const k=`${r.lattice?.key||`${r.w},0,${r.h}`}|${r.companion?.shapeSig||''}|${r.companion?.patternKind||''}|${r.metrics?.companionArea??'x'}|${(r.placements||[]).map(p=>`${p.ori||0}@${p.anchor?.[0]||0},${p.anchor?.[1]||0}`).sort().join(';')}`;if(seen.has(k))return;seen.add(k);keep.push(r);};
  for(const r of arr.slice(0,Math.min(7,cap)))add(r);
  // Reserve slots for genuine non-empty companion shapes even when a 100% primary tiling
  // exists and therefore wins the lexicographic objective.
  const nontrivial=arr.filter(r=>r.companion?.exactResidual&&Number.isFinite(r.metrics?.companionArea)&&r.metrics.companionArea>0),shapeSeen=new Set();
  for(const r of nontrivial){const k=`${r.companion?.shapeSig||''}|${r.companion?.patternKind||''}|${r.metrics?.companionArea}`;if(shapeSeen.has(k))continue;shapeSeen.add(k);add(r);if(keep.length>=cap)break;}
  for(const r of arr){if(keep.length>=cap)break;add(r);}return keep.slice(0,cap);
}
function stateByCount(states){let best=null;for(const s of states)if(s&&(!best||s.chosen.length>best.chosen.length))best=s;return best;}
function stateOccSig(st){
  if(!st?.occ)return'';let h=2166136261>>>0;
  for(let i=0;i<st.occ.length;i++)if(st.occ[i]){h^=(i+1);h=Math.imul(h,16777619)>>>0;}
  // Occupancy alone collapses every exact tiling of the same quotient to the same key.
  // Include an order-independent hash of tile boundaries/orientations so 100%-coverage
  // states with genuinely different partitions can survive to the variant portfolio.
  const tiles=(st.chosen||[]).map(p=>{let z=2166136261>>>0;z^=(Number(p.ori)||0)+1;z=Math.imul(z,16777619)>>>0;for(const id of (p.idx||[]).slice().sort((a,b)=>a-b)){z^=(id+1);z=Math.imul(z,16777619)>>>0;}return z>>>0;}).sort((a,b)=>a-b);
  let ph=2166136261>>>0;for(const z of tiles){ph^=z;ph=Math.imul(ph,16777619)>>>0;}return `${st.chosen?.length||0}:${h}:${ph}`;
}
function statePortfolio(states,data,mode,cap=3){
  const uniq=new Map();for(const st of states){if(!st?.chosen?.length)continue;const k=stateOccSig(st),old=uniq.get(k);if(!old||st.chosen.length>old.chosen.length)uniq.set(k,st);}
  const scored=[...uniq.values()].map(st=>{const coverage=(st.chosen.length*ACTIVE_SHAPE_N)/Math.max(1,data.area),v=periodicVoidSummaryHNF(st.occ,data.a,data.b||0,data.c),topology=(v.cohesion||0)-Math.max(0,v.components-1)*.045-Math.min(1,(v.perimeter||0)/Math.max(1,data.area))*.10,companionFit=mode==='companion'?companionTopologyScore(st.occ,data.a,data.b||0,data.c,ACTIVE_GROUP||'D4'):0;return{st,coverage,topology,companionFit};});
  scored.sort((x,y)=>y.coverage-x.coverage||y.companionFit-x.companionFit||y.topology-x.topology);
  if(mode!=='companion')return scored.slice(0,Math.max(1,cap)).map(x=>x.st);
  // Companion quality is not monotone in primary-tile count. Keep a coverage leader,
  // a void-topology leader, and (when distinct) one near-leader with a different hole set.
  const out=[];const add=x=>{if(x&&!out.some(z=>stateOccSig(z)===stateOccSig(x.st)))out.push(x.st);};add(scored[0]);
  add(scored.slice().sort((x,y)=>y.companionFit-x.companionFit||y.topology-x.topology||y.coverage-x.coverage)[0]);
  for(const x of scored){if(out.length>=cap)break;if(scored[0].coverage-x.coverage<=Math.max(.04,ACTIVE_SHAPE_N/Math.max(1,data.area)*1.5))add(x);}
  return out.slice(0,Math.max(1,cap));
}
function evaluateLatticeCandidate(ctx,cand,round,warm=null,fair=false,quick=false){
  const data=hnfData(ctx,cand.a,cand.b,cand.c);if(!data.placements.length)return[];let seed=hashSeed(ctx.shapeSig,cand.a,cand.b,cand.c,cand.target,round,ctx.runNonce),states=[];if(warm)states.push(warm);const area=cand.area,priority=!!spectrumPriority(cand.target),all=fair||priority||area<=180||round%5===0,alg=ctx.algorithm||'auto';
  seed=rngStep(seed);states.push(packFromPlacements(data,cand.a,cand.c,seed,round%3));
  if(quick){
    if(area<=260){seed=rngStep(seed);states.push(mrvPack(data,cand.a,cand.c,seed,Math.max(10,Math.min(24,Math.round(Math.sqrt(area)*1.1)))));}
    // The universal fast pass must stay cheap, but periods 5/7 used to regress when the
    // priority lane for 1/2/3/4/6/8 was added. Give these small non-priority periods
    // one algorithm-specific refinement so priority search does not steal their baseline.
    if(!priority&&cand.target<=8){
      seed=rngStep(seed);
      if(alg==='dense')states.push(mrvPack(data,cand.a,cand.c,seed,Math.max(16,Math.min(30,Math.round(Math.sqrt(area)*1.45)))));
      else states.push(constructivePack(data,cand.a,cand.c,seed,ctx.opt));
    }
    return statePortfolio(states,data,ctx.mode==='companion'?'companion':'cell',ctx.mode==='companion'?2:1);
  }
  if(priority&&(alg==='auto'||alg==='dense')){seed=rngStep(seed);states.push(packFromPlacements(data,cand.a,cand.c,seed,2));}
  if(all||alg!=='constructive'){seed=rngStep(seed);states.push(mrvPack(data,cand.a,cand.c,seed,Math.max(14,Math.min(priority?42:30,Math.round(Math.sqrt(area)*(priority?1.65:1.3))))));if(priority&&alg==='dense'){seed=rngStep(seed);states.push(mrvPack(data,cand.a,cand.c,seed,Math.max(18,Math.min(46,Math.round(Math.sqrt(area)*1.8)))));}}
  if(all||alg!=='dense'){seed=rngStep(seed);states.push(constructivePack(data,cand.a,cand.c,seed,ctx.opt));if(priority&&alg==='constructive'){seed=rngStep(seed);states.push(constructivePack(data,cand.a,cand.c,seed,{...ctx.opt,complexity:Math.min(1,(ctx.opt.complexity??.56)+.12),regularity:Math.max(0,(ctx.opt.regularity??.48)-.08)}));}}
  let st=stateByCount(states);if(!st)return[];const target=Math.floor(area/ctx.n),gap=target-st.chosen.length,remaining=ctx.deadline-now();
  // Small primitive periods benefit more from repairing a nearly-complete packing than
  // from another blind restart. Give each public algorithm a distinct bounded finisher.
  if(priority&&gap>0&&gap<=5&&area<=380&&remaining>24){
    const slice=Math.min(alg==='dense'?34:alg==='constructive'?26:22,Math.max(10,remaining*.045));let q=st;
    if(alg==='dense'){q=improveOneForTwo(data,cand.a,cand.c,q,seed^0x243f6a88,Math.min(ctx.deadline,now()+slice*.48));q=refillLocal(data,cand.a,cand.c,q,seed^0x13198a2e,Math.min(ctx.deadline,now()+slice));}
    else if(alg==='constructive')q=refillLocal(data,cand.a,cand.c,q,seed^0xa4093822,Math.min(ctx.deadline,now()+slice));
    else q=improveOneForTwo(data,cand.a,cand.c,q,seed^0x299f31d0,Math.min(ctx.deadline,now()+slice));
    if(q&&q.chosen.length>=st.chosen.length){states.push(q);st=q;}
  }
  if(cand.gap===0&&gap>0&&data.cellTo.every(x=>x.length)&&remaining>14&&(fair||priority||cand.rank<2.8||gap<=1)){
    const factor=priority?(alg==='dense'?1.55:alg==='constructive'?1.20:1.40):1,slice=Math.min(priority?(fair?72:145):(fair?52:96),Math.max(14,remaining*(fair?(priority ? .075 : .055):(priority ? .11 : .075)))),nodes=Math.min(620000,Math.round((76000+area*(priority?980:620))*factor)),ex=exactCoverDLX(data,now()+slice,seed^0x6a09e667,nodes,null,false);
    if(ex.solution){const occ=new Uint8Array(area),chosen=[];for(const p of ex.solution)place(p,occ,chosen);const q={occ,chosen};states.push(q);st=q;}
  }
  const gap2=target-st.chosen.length;if(gap2>0&&gap2<=(priority?3:2)&&remaining>14&&area<=(priority?380:300)){const q=maxPackingBranch(data,st,Math.min(ctx.deadline,now()+Math.min(priority?145:92,20+area*(priority ? .30 : .22))),seed^0x31415926,priority?145000:85000);if(q)states.push(q);}
  return statePortfolio(states,data,ctx.mode==='companion'?'companion':'cell',ctx.mode==='companion'?(priority?5:4):(priority?4:2));
}
function solvePeriodicSpectrum(ctx,mode='cell',initialResult=null){
  const limit=Math.max(1,Math.min(20,ctx.opt.cellPieces||10)),firstGap=Math.max(0,ctx.n-1),maxGap=Math.max(firstGap,Math.min(ctx.n*4-1,96));
  const makeStream=k=>{const cands=latticeCandidatesForTarget(ctx,k,0,firstGap),buckets=new Map();for(const q of cands){if(!buckets.has(q.gap))buckets.set(q.gap,[]);buckets.get(q.gap).push(q);}return{k,cands,buckets,gapTurn:0,gapCursor:new Map(),visits:0,best:0,gain:0,gapHi:firstGap,maxGap,scoutKeys:new Set(),fairKeys:new Set()};};
  const streams=Array.from({length:limit},(_,i)=>makeStream(i+1)),byPeriod=new Map(),alts=new Map(),warmByLattice=new Map(),minedSource=new Map();let evals=0,lastBest=null;
  const objective=r=>mode==='companion'?(r.metrics?.combinedCoverage??r.metrics?.coverage??0):(r.metrics?.coverage||0);
  const sig=r=>{const L=r.lattice?.key||`${r.w},0,${r.h}`,ps=(r.placements||[]).map(p=>`${p.ori||0}@${p.anchor?.[0]||0},${p.anchor?.[1]||0}`).sort().join('|'),cx=mode==='companion'?`|${Math.round((r.metrics?.combinedCoverage||0)*1e8)}|${r.metrics?.companionArea??'x'}|${r.companion?.shapeSig||''}`:'';return `${L}|${Math.round((r.metrics?.coverage||0)*1e8)}|${ps}${cx}`;};
  const record=r=>{r=validateResultForOutput(r);if(!r)return false;const p=spectrumPeriod(r);if(p<1||p>limit)return false;let arr=alts.get(p)||[],rs=sig(r);if(arr.some(x=>sig(x)===rs))return false;arr.push(r);arr.sort((x,y)=>betterSpectrum(x,y,mode)===x?-1:1);arr=trimSpectrumAlternatives(arr,mode,spectrumPriority(p)?14:10);alts.set(p,arr);const old=byPeriod.get(p),best=arr[0];byPeriod.set(p,best);const st=streams[p-1],v=objective(best),oldv=st.best;st.best=Math.max(st.best,v);st.gain=Math.max(st.gain,st.best-oldv);return best!==old;};
  const output=()=>{const ladder=[...byPeriod.entries()].sort((a,b)=>a[0]-b[0]).map(x=>x[1]),variants=[];for(let k=1;k<=limit;k++){const a=alts.get(k)||[],best=a[0],bv=best?objective(best):0;for(const r of a.slice(1)){if(variants.length>=36)break;if(bv-objective(r)<=.035)variants.push(r);}}return ladder.concat(variants).slice(0,72);};
  const ingest=r=>{if(!r)return;const p=spectrumPeriod(r);if(p<1||p>limit)return;record(r);const lat=r.lattice||{a:r.w,b:0,c:r.h,key:`${r.w},0,${r.h}`};if(lat?.a&&lat?.c)warmByLattice.set(lat.key||`${lat.a},${lat.b||0},${lat.c}`,r);};
  const mine=r=>{if(!r||objective(r)<1-1e-12)return false;const pk=spectrumPeriod(r),seen=minedSource.get(pk)||0;if(seen>=2)return false;minedSource.set(pk,seen+1);const total=Math.max(1,ctx.deadline-ctx.start),slice=total<4000?55:total<15000?140:280,end=Math.min(ctx.deadline-8,now()+slice);let changed=false;for(const q of mineSubstructureResults(ctx,r,limit,end)){if(record(q))changed=true;const lat=q.lattice;if(lat?.key)warmByLattice.set(lat.key,q);}return changed;};
  const periodObjectiveSettled=st=>{const r=byPeriod.get(st.k);if(!r)return false;if(mode==='cell')return (r.metrics?.coverage||0)>=1-1e-12;const plus=r.metrics?.combinedCoverage??r.metrics?.coverage??0;return plus>=1-1e-12&&r.companion?.exactResidual===true;};
  const addStreamCandidates=(st,more)=>{if(!more?.length)return;st.cands.push(...more);for(const q of more){if(!st.buckets.has(q.gap))st.buckets.set(q.gap,[]);st.buckets.get(q.gap).push(q);}};
  const streamHasUnvisited=st=>st.fairKeys.size<st.cands.length;
  const extendStream=st=>{if(streamHasUnvisited(st))return true;const settled=periodObjectiveSettled(st);if(settled&&mode==='companion'){
      const hasNontrivial=(alts.get(st.k)||[]).some(r=>r.companion?.exactResidual&&Number.isFinite(r.metrics?.companionArea)&&r.metrics.companionArea>0),exploreHi=Math.min(st.maxGap,ctx.n*(spectrumPriority(st.k)?3:2)-1);
      if(!hasNontrivial&&st.gapHi<exploreHi&&st.visits<spectrumVariantQuota(ctx,st.k)+8){const lo=st.gapHi+1,hi=Math.min(exploreHi,st.gapHi+Math.max(1,ctx.n));addStreamCandidates(st,latticeCandidatesForTarget(ctx,st.k,lo,hi));st.gapHi=hi;return streamHasUnvisited(st);}
    }
    if(settled||st.gapHi>=st.maxGap)return false;const lo=st.gapHi+1,hi=Math.min(st.maxGap,st.gapHi+Math.max(1,ctx.n));addStreamCandidates(st,latticeCandidatesForTarget(ctx,st.k,lo,hi));st.gapHi=hi;return streamHasUnvisited(st);};
  const nextDeepCandidate=st=>{const gaps=[...st.buckets.keys()].sort((a,b)=>a-b);if(!gaps.length)return null;for(let t=0;t<gaps.length*2;t++){const g=gaps[st.gapTurn++%gaps.length],arr=st.buckets.get(g)||[];let i=st.gapCursor.get(g)||0;while(i<arr.length&&st.fairKeys.has(arr[i].key))i++;st.gapCursor.set(g,i);if(i<arr.length){const q=arr[i];st.gapCursor.set(g,i+1);return q;}}for(const q of st.cands)if(!st.fairKeys.has(q.key))return q;return null;};
  const processCandidate=(st,cand,fair=false,quick=false)=>{const warmR=warmByLattice.get(cand.key),warm=warmR?stateFromLatticeResult(warmR,ctx,cand):null,states=evaluateLatticeCandidate(ctx,cand,++evals,warm,fair,quick);let changed=false,bestHere=null;for(const state of states){if(STOP||now()>=ctx.deadline-5)break;const r=makeLatticeSpectrumResult(ctx,cand,state,mode,ctx.algorithm==='dense'?'D·Σ':ctx.algorithm==='constructive'?'H·Σ':'A·Σ');if(!r)continue;if(!bestHere||betterSpectrum(r,bestHere,mode)===r)bestHere=r;if(record(r))changed=true;const lat=r.lattice;if(lat?.key)warmByLattice.set(lat.key,r);}if(bestHere){if(!quick&&mine(bestHere))changed=true;if(spectrumPeriod(bestHere)===st.k){const v=objective(bestHere),old=st.best;st.best=Math.max(st.best,v);st.gain=.70*st.gain+.30*Math.max(0,st.best-old);}else st.gain*=.88;}else st.gain*=.88;return{changed,bestHere};};
  ingest(initialResult);for(const r of ctx.resumeResults||[])if(r&&(r.kind==='periodic'||r.kind==='cell'))ingest(r);for(const r of [...byPeriod.values()])mine(r);
  if(byPeriod.size){const out=output();lastBest=out.slice().sort((a,b)=>betterSpectrum(a,b,mode)===a?-1:1)[0];ctx.reporter.preview(lastBest,{phase:'prepare',engine:'resume',bestCoverage:lastBest?.metrics?.coverage},true);ctx.reporter.partial(out,{phase:'prepare',engine:'resume'},true);}
  // Ultra-cheap first look across the entire requested spectrum. This protects short budgets
  // from being consumed by proof-oriented work before later period levels have any candidate.
  for(const st of streams){if(STOP||now()>=ctx.deadline-28)break;const cand=st.cands[0];if(!cand)continue;st.scoutKeys.add(cand.key);st.visits++;const z=processCandidate(st,cand,false,true);if(z.changed&&(!lastBest||objective(z.bestHere)>objective(lastBest)))lastBest=z.bestHere;}
  if(byPeriod.size){const out=output(),best=out.slice().sort((a,b)=>betterSpectrum(a,b,mode)===a?-1:1)[0];lastBest=best;ctx.reporter.preview(best,{phase:'scout',engine:best?.engine||'Σ·q',bestCoverage:best?.metrics?.coverage},true);ctx.reporter.partial(out,{phase:'scout',engine:best?.engine||'Σ·q'},true);}
  // Baseline fair scout touches every requested level before any priority-only deepening.
  // This prevents the 1/2/3/4/6/8 acceleration from starving ordinary levels such as 5 or 7.
  const totalBudget=Math.max(1,ctx.deadline-ctx.start),shortBudget=totalBudget<4000,scoutWidth=shortBudget?2:limit<=8?3:limit<=16?2:1;
  for(const st of streams){if(STOP||now()>=ctx.deadline-24)break;let levelChanged=false,tries=scoutWidth,visited=shortBudget?st.scoutKeys:st.fairKeys;for(let q=0;q<tries&&now()<ctx.deadline-20;q++){let cand=null;const wantGap=Math.min(ctx.n-1,q),pick=st.cands.find(x=>x.gap===wantGap&&!visited.has(x.key));cand=pick||st.cands.find(x=>!visited.has(x.key));if(!cand)break;visited.add(cand.key);st.visits++;const z=processCandidate(st,cand,!shortBudget,shortBudget);levelChanged=levelChanged||z.changed;}if(levelChanged){const out=output(),best=out.slice().sort((a,b)=>betterSpectrum(a,b,mode)===a?-1:1)[0];lastBest=best;ctx.reporter.preview(best,{phase:'pack',engine:best?.engine||'A·Σ',current:st.k,total:limit,bestCoverage:best?.metrics?.coverage},st.k===1);ctx.reporter.partial(out,{phase:'pack',engine:best?.engine||'A·Σ'});}ctx.reporter.progress({phase:'pack',engine:lastBest?.engine||'A·Σ',current:st.k,total:limit,percent:Math.min(.999,(now()-ctx.start)/(ctx.deadline-ctx.start)),bestCoverage:lastBest?.metrics?.coverage});}
  // Dedicated small-period pass. The baseline spectrum is already represented, so the
  // remaining budget can safely concentrate on 1/2/3/4/6/8 exact-area HNF candidates.
  const priorityQuota=totalBudget<4000?1:totalBudget<15000?3:6,priorityEnd=Math.min(ctx.deadline-18,now()+Math.min(totalBudget*(totalBudget<4000 ? .18 : .24),Math.max(0,ctx.deadline-now())*.45));
  for(const k of [1,2,3,4,6,8]){if(k>limit||STOP||now()>=priorityEnd)continue;const st=streams[k-1];let done=0;for(const cand of st.cands){if(done>=priorityQuota||STOP||now()>=priorityEnd)break;if(cand.gap!==0||st.fairKeys.has(cand.key))continue;st.fairKeys.add(cand.key);st.visits++;done++;const z=processCandidate(st,cand,true);if(z.changed){const out=output(),best=out.slice().sort((a,b)=>betterSpectrum(a,b,mode)===a?-1:1)[0];lastBest=best;ctx.reporter.preview(best,{phase:'pack',engine:best?.engine||'Σ*',current:k,total:limit,bestCoverage:best?.metrics?.coverage},false);ctx.reporter.partial(out,{phase:'pack',engine:best?.engine||'Σ*'});}}}
  // Adaptive deepening. Exact 100% levels remain searchable for a bounded variant quota instead
  // of disappearing immediately, so non-trivial multi-tile structures survive in the spectrum.
  let rr=0;while(!STOP&&now()<ctx.deadline-12){rr++;let chosen=null,score=-1e9;const log=Math.log(evals+3);for(const st of streams){if(!extendStream(st))continue;const solved=periodObjectiveSettled(st),unc=.20*Math.sqrt(log/(st.visits+1)),deficit=1-(st.best||0),fair=1/(st.visits+1),quota=spectrumVariantQuota(ctx,st.k),variant=solved?(st.visits<quota ? .18+(spectrumPriority(st.k) ? .06 : 0) : -.52):0,gain=st.gain*.55,depth=.035*(st.gapHi/Math.max(1,st.maxGap)),priority=spectrumPriority(st.k) ? .08 : 0,v=deficit*.72+unc+fair*.42+gain+variant+priority-depth;if(v>score){score=v;chosen=st;}}if(!chosen)break;let cand=nextDeepCandidate(chosen);if(!cand){if(extendStream(chosen))cand=nextDeepCandidate(chosen);if(!cand)continue;}chosen.fairKeys.add(cand.key);chosen.visits++;const z=processCandidate(chosen,cand,false);if(z.changed){const out=output(),best=out.slice().sort((a,b)=>betterSpectrum(a,b,mode)===a?-1:1)[0];lastBest=best;ctx.reporter.preview(best,{phase:'pack',engine:best?.engine||'A·Σ',current:evals,bestCoverage:best?.metrics?.coverage,bestScore:best?.metrics?.score});ctx.reporter.partial(out,{phase:'pack',engine:best?.engine||'A·Σ'});}if((rr&3)===0)ctx.reporter.progress({phase:'pack',engine:lastBest?.engine||'A·Σ',current:evals,total:streams.reduce((z,x)=>z+x.cands.length,0),percent:Math.min(.999,(now()-ctx.start)/(ctx.deadline-ctx.start)),w:cand.a,h:cand.c,bestCoverage:lastBest?.metrics?.coverage,bestScore:lastBest?.metrics?.score});}
  return output();
}
// Legacy finite-board experimental path removed; both public v9 modes are periodic.
function staticCellOrder(data,w,h,seed){const a=Array.from({length:w*h},(_,i)=>i);a.sort((x,y)=>data.cellTo[x].length-data.cellTo[y].length||(((x*2654435761)^(seed>>>0))>>>0)-(((y*2654435761)^(seed>>>0))>>>0));return a;}
function scarcityPack(data,w,h,seed){
  const occ=new Uint8Array(w*h),chosen=[],order=staticCellOrder(data,w,h,seed);let s=seed>>>0||1;
  for(const c of order){if(occ[c])continue;let bestPi=-1,best=-Infinity;for(const pi of data.cellTo[c]){const p=data.placements[pi];if(!canPlace(p,occ))continue;let contact=0,edge=0,exposed=0;for(const id of p.idx){const x=id%w,y=(id/w)|0;if(x===0||x===w-1||y===0||y===h-1)edge++;for(const[dx,dy]of DIR4){const nx=x+dx,ny=y+dy;if(nx<0||nx>=w||ny<0||ny>=h)continue;if(occ[ny*w+nx])contact++;else exposed++;}}s=rngStep(s^pi);const sc=contact*5+edge*.6-exposed*.04+(s&1023)/8192;if(sc>best){best=sc;bestPi=pi;}}
    if(bestPi>=0)place(data.placements[bestPi],occ,chosen);
  }
  return{occ,chosen};
}
function greedyPack(data,w,h,seed){const occ=new Uint8Array(w*h),chosen=[];for(const pi of shuffled(Array.from({length:data.placements.length},(_,i)=>i),seed)){const p=data.placements[pi];if(canPlace(p,occ))place(p,occ,chosen);}return{occ,chosen};}
function frontierPack(data,w,h,seed){
  const occ=new Uint8Array(w*h),chosen=[];let s=seed>>>0||1,mode=s%4,order=[];if(mode<2){for(let y=0;y<h;y++){if(mode===1&&y&1)for(let x=w-1;x>=0;x--)order.push(y*w+x);else for(let x=0;x<w;x++)order.push(y*w+x);}}else{for(let x=0;x<w;x++){if(mode===3&&x&1)for(let y=h-1;y>=0;y--)order.push(y*w+x);else for(let y=0;y<h;y++)order.push(y*w+x);}}
  for(const c of order){if(occ[c])continue;let bestPi=-1,best=-Infinity;for(const pi of data.cellTo[c]){const p=data.placements[pi];if(!canPlace(p,occ))continue;let contact=0,edge=0;for(const id of p.idx){const x=id%w,y=(id/w)|0;if(x===0||x===w-1||y===0||y===h-1)edge++;for(const[dx,dy]of DIR4){const nx=x+dx,ny=y+dy;if(nx>=0&&nx<w&&ny>=0&&ny<h&&occ[ny*w+nx])contact++;}}s=rngStep(s^pi);const sc=contact*4+edge*.55+(s&1023)/4096;if(sc>best){best=sc;bestPi=pi;}}if(bestPi>=0)place(data.placements[bestPi],occ,chosen);}
  return{occ,chosen};
}
function repairPacking(data,w,h,state,seed,deadline){
  let occ=state.occ,chosen=state.chosen.slice(),s=seed>>>0||1,bestCount=chosen.length;for(let it=0;it<48&&now()<deadline;it++){const holes=[];for(let c=0;c<occ.length;c++)if(!occ[c])holes.push(c);if(!holes.length)break;s=rngStep(s);const hc=holes[s%holes.length],hx=hc%w,hy=(hc/w)|0,near=[];for(let i=0;i<chosen.length;i++){let d=99;for(const[x,y]of chosen[i].cells){d=Math.min(d,Math.abs(x-hx)+Math.abs(y-hy));if(d<=1)break;}if(d<=2)near.push(i);}if(!near.length)continue;const removeCount=Math.min(1+(s%4),near.length),remove=new Set(shuffled(near,s).slice(0,removeCount)),old=chosen,keep=[];for(let i=0;i<chosen.length;i++){if(remove.has(i)){for(const id of chosen[i].idx)occ[id]=0;}else keep.push(chosen[i]);}const cand=new Set();for(let yy=Math.max(0,hy-5);yy<=Math.min(h-1,hy+5);yy++)for(let xx=Math.max(0,hx-5);xx<=Math.min(w-1,hx+5);xx++)if(Math.abs(xx-hx)+Math.abs(yy-hy)<=6)for(const pi of data.cellTo[yy*w+xx])cand.add(pi);for(const pi of shuffled([...cand],s)){const p=data.placements[pi];if(canPlace(p,occ))place(p,occ,keep);}if(keep.length>=bestCount){chosen=keep;bestCount=keep.length;}else{chosen=old;occ=new Uint8Array(w*h);for(const p of chosen)for(const id of p.idx)occ[id]=1;}}
  return{occ,chosen};
}
function finiteMetrics(w,h,chosen,occ,mode,opt={}){
  const valid=validateState(w,h,{chosen});if(!valid)return{metrics:{coverage:0,score:-1e12,invalid:true},holes:[]};occ=valid.occ;const covered=countBitsOcc(occ),coverage=covered/(w*h),voids=voidConnectivity(occ,w,h,false),oriCounts=new Map(),owner=new Int32Array(w*h);owner.fill(-1);
  for(let i=0;i<chosen.length;i++){const p=chosen[i];oriCounts.set(p.ori,(oriCounts.get(p.ori)||0)+1);for(const id of p.idx)owner[id]=i;}
  const adjCounts=new Map();for(let y=0;y<h;y++)for(let x=0;x<w;x++){const a=owner[y*w+x];if(a<0)continue;for(const[dx,dy]of[[1,0],[0,1]]){const nx=x+dx,ny=y+dy;if(nx>=w||ny>=h)continue;const b=owner[ny*w+nx];if(b<0||a===b)continue;const oa=chosen[a].ori,ob=chosen[b].ori,k=oa<=ob?`${oa}:${ob}`:`${ob}:${oa}`;adjCounts.set(k,(adjCounts.get(k)||0)+1);}}
  const adjEntropy=entropy(adjCounts),oriEntropy=entropy(oriCounts),entropyMix=(adjEntropy+oriEntropy)*.5,cw=opt.coverageWeight??.9,cx=opt.complexity??.5,rg=opt.regularity??.5,complexTerm=(mode==='drift'?1.35:1)*entropyMix,regularTerm=1-entropyMix,voidQuality=voids.cohesion-voids.enclosed*.08-Math.min(1,voids.perimeter/Math.max(1,w*h))*.22,score=coverage*(520+cw*1500)+complexTerm*cx*120+regularTerm*rg*90+voidQuality*150;
  return{metrics:{coverage,fundArea:NaN,adjEntropy,oriEntropy,shortestPeriod:NaN,holeSpread:NaN,voidComponents:voids.components,voidLargestShare:voids.largestShare,voidPerimeter:voids.perimeter,enclosedVoids:voids.enclosed,score},holes:voids.holes};
}
function makeFiniteResult(w,h,state,mode,engine,labelOverride=null,opt={}){const safe=validateState(w,h,state);if(!safe)return null;const{metrics,holes}=finiteMetrics(w,h,safe.chosen,safe.occ,mode,opt);if(metrics.invalid)return null;return{id:`F${++RESULT_SEQ}`,kind:'finite',label:labelOverride||(mode==='near'?'near packing':'aperiodic-looking / finite'),w,h,engine,placements:safe.chosen.map(p=>({cells:p.cells,ori:p.ori,anchor:p.anchor})),holes,metrics};}
function stateFromFiniteResult(r,w,h){if(!r||r.kind!=='finite'||r.w!==w||r.h!==h)return null;const occ=new Uint8Array(w*h),chosen=[];for(const rp of r.placements||[]){const cells=rp.cells||[],idx=[];let ok=true;for(const[x,y]of cells){if(x<0||x>=w||y<0||y>=h){ok=false;break;}const id=y*w+x;if(occ[id]){ok=false;break;}idx.push(id);}if(!ok)return null;const p={cells,idx,ori:Number(rp.ori)||0,anchor:rp.anchor||cells[0]||[0,0]};place(p,occ,chosen);}return{occ,chosen};}

function smallRectangleSeed(ctx,deadline){
  if(ctx.knownNonTiler||ctx.algorithm==='greedy'||ctx.algorithm==='frontier')return null;
  const n=ctx.n,cands=[];for(let w=1;w<=20;w++)for(let h=w;h<=20;h++){const area=w*h;if(area%n||area<n||area>Math.min(200,n*10))continue;const pieces=area/n;if(pieces>10)continue;cands.push({w,h,area,pieces,aspect:Math.abs(Math.log(w/h))});}cands.sort((a,b)=>a.area-b.area||a.aspect-b.aspect);
  let seed=hashSeed(ctx.shapeSig,31337,ctx.runNonce);for(const d of cands.slice(0,14)){if(now()>deadline)break;const data=makeFiniteData(ctx.oris,d.w,d.h);if(!data.placements.length||data.cellTo.some(x=>!x.length))continue;seed=rngStep(seed);const r=exactCoverDLX(data,Math.min(deadline,now()+35),seed,32000,null,false);if(r.solution)return{w:d.w,h:d.h,placements:r.solution};}return null;
}
function repeatBlock(block,w,h){
  const occ=new Uint8Array(w*h),chosen=[];for(let by=0;by+block.h<=h;by+=block.h)for(let bx=0;bx+block.w<=w;bx+=block.w)for(const p of block.placements){const cells=p.cells.map(([x,y])=>[x+bx,y+by]),idx=cells.map(([x,y])=>y*w+x),q={cells,idx,ori:p.ori,anchor:[p.anchor[0]+bx,p.anchor[1]+by]};place(q,occ,chosen);}return{occ,chosen};
}
function fillExisting(data,w,h,state,seed){
  const occ=state.occ,chosen=state.chosen,order=staticCellOrder(data,w,h,seed);let s=seed>>>0||1;for(const c of order){if(occ[c])continue;let bestPi=-1,best=-Infinity;for(const pi of data.cellTo[c]){const p=data.placements[pi];if(!canPlace(p,occ))continue;let contact=0;for(const id of p.idx){const x=id%w,y=(id/w)|0;for(const[dx,dy]of DIR4){const nx=x+dx,ny=y+dy;if(nx>=0&&nx<w&&ny>=0&&ny<h&&occ[ny*w+nx])contact++;}}s=rngStep(s^pi);const sc=contact*4+(s&1023)/4096;if(sc>best){best=sc;bestPi=pi;}}if(bestPi>=0)place(data.placements[bestPi],occ,chosen);}return state;
}
function boardSizeFor(n,opt,fast=false){if(fast)return n>=15?32:n>=10?34:30;const req=opt.boardSize||44;if(n>=15)return Math.min(req,64);if(n>=10)return Math.min(req,68);return Math.min(req,72);}
function bestOfStates(states){let best=states[0];for(const s of states)if(s.chosen.length>best.chosen.length)best=s;return best;}
function quickFallback(ctx){
  const size=boardSizeFor(ctx.n,ctx.opt,true),data=finiteData(ctx,size,size),seed=hashSeed(ctx.shapeSig,size,991,ctx.runNonce),st=bestOfStates([scarcityPack(data,size,size,seed),frontierPack(data,size,size,seed^0x9e3779b9)]),r=makeFiniteResult(size,size,st,'near','F','near fallback',ctx.opt);return r;
}
function solveFinite(ctx,mode,labelOverride=null){
  const size=boardSizeFor(ctx.n,ctx.opt,false),w=size,h=size,data=finiteData(ctx,w,h),totalBudget=Math.max(0,ctx.deadline-now()),algorithm=ctx.algorithm,opt=ctx.opt;let best=null,seed=hashSeed(ctx.shapeSig,ctx.n,w,h,7919,ctx.runNonce),iter=0;
  const resumed=ctx.resumeResults.find(r=>r?.kind==='finite'&&r.w===w&&r.h===h);if(resumed){const st=stateFromFiniteResult(resumed,w,h);if(st){best=makeFiniteResult(w,h,st,mode,'resume',labelOverride,opt);ctx.reporter.preview(best,{phase:'pack',engine:'resume',bestCoverage:best.metrics.coverage,bestScore:best.metrics.score},true);}}
  const aggression=opt.aggression??.5,target=opt.targetCoverage??.995,maxIter=Math.max(80,Math.round(350+aggression*2400)),minImproveMs=mode==='near'?260:650;ctx.reporter.progress({phase:'pack',engine:algorithm==='auto'?'H':algorithm,current:0,total:maxIter,percent:0,w,h,bestCoverage:best?.metrics.coverage},true);
  if(!best&&(algorithm==='auto'||algorithm==='hybrid'||algorithm==='exact')&&totalBudget>420&&!ctx.knownNonTiler){const block=smallRectangleSeed(ctx,Math.min(ctx.deadline,now()+Math.min(180,totalBudget*.12)));if(block){let st=repeatBlock(block,w,h);st=fillExisting(data,w,h,st,seed^0x517cc1b7);if(now()<ctx.deadline-15)st=repairPacking(data,w,h,st,seed,Math.min(ctx.deadline,now()+75));best=makeFiniteResult(w,h,st,mode,'block',labelOverride,opt);ctx.reporter.preview(best,{phase:'pack',engine:'block',current:0,total:maxIter,bestCoverage:best.metrics.coverage,bestScore:best.metrics.score},true);ctx.reporter.partial([best],{phase:'pack',engine:'block'},true);if(best.metrics.coverage>=.999999)return[best];}}
  while(iter<maxIter&&!STOP&&now()<ctx.deadline){iter++;seed=rngStep(seed);let st,engine;if(algorithm==='greedy'){st=greedyPack(data,w,h,seed);engine='greedy';}else if(algorithm==='frontier'){st=frontierPack(data,w,h,seed);engine='frontier';}else{const selector=(seed>>>3)%100,a=scarcityPack(data,w,h,seed),b=frontierPack(data,w,h,seed^0x9e3779b9);let states=[a,b];if(selector<18+aggression*42)states.push(greedyPack(data,w,h,seed^0x85ebca6b));st=bestOfStates(states);if(now()<ctx.deadline-10&&selector<35+aggression*55)st=repairPacking(data,w,h,st,seed,Math.min(ctx.deadline,now()+Math.round(30+aggression*95)));engine='hybrid';}
    const candidate=makeFiniteResult(w,h,st,mode,engine,labelOverride,opt);if(!best||candidate.metrics.score>best.metrics.score){best=candidate;ctx.reporter.preview(best,{phase:'pack',engine,current:iter,total:maxIter,bestCoverage:best.metrics.coverage,bestScore:best.metrics.score},iter===1);ctx.reporter.partial([best],{phase:'pack',engine});}
    ctx.reporter.progress({phase:'pack',engine,current:iter,total:maxIter,percent:Math.min(.999,(now()-ctx.start)/(ctx.deadline-ctx.start)),w,h,bestCoverage:best?.metrics.coverage,bestScore:best?.metrics.score});
    const enough=best&&best.metrics.coverage>=target,elapsed=now()-ctx.start;if(enough&&elapsed>=minImproveMs){if(mode==='near'||(mode==='drift'&&((best.metrics.adjEntropy||0)+(best.metrics.oriEntropy||0))*.5>=(.22+(opt.complexity??.5)*.36)))break;}if(best?.metrics.coverage>=.999999)break;
  }
  return best?[best]:[];
}
