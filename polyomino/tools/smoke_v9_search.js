const fs=require('fs'),vm=require('vm'),path=require('path');
const ROOT=path.dirname(path.dirname(__filename));
global.self={};global.postMessage=()=>{};
vm.runInThisContext(fs.readFileSync(path.join(ROOT,'solver.worker.js'),'utf8'));
function assert(ok,msg){if(!ok)throw new Error(msg);}
function makeOcc(a,c,holes){const occ=new Uint8Array(a*c);occ.fill(1);for(const [x,y] of holes)occ[y*a+x]=0;return occ;}

// Companion semantics: the whole residual branch is the companion; congruent branches stay whole.
let q=companionFromVoid(makeOcc(6,3,[[0,1],[1,1],[3,1],[4,1]]),6,0,3,'D4');
assert(q.exact&&q.patternKind==='congruent-components','congruent residual components not recognized');
assert(q.area===2&&q.copies===2&&q.totalCells===4&&q.remaining.length===0,'congruent companion accounting');
q=companionFromVoid(makeOcc(7,3,[[0,1],[1,1],[3,1],[4,1],[4,2]]),7,0,3,'D4');
assert(!q.exact&&q.patternKind==='mixed-components'&&q.totalCells===0,'mixed components must not be fragmented into a fake companion');
q=companionFromVoid(makeOcc(4,3,[[0,1],[1,1],[2,1],[3,1]]),4,0,3,'D4');
assert(!q.exact&&q.patternKind==='winding-residual','non-contractible residual stripe must not be treated as a finite companion');

// Exact states on one quotient can have the same occupancy but different tile partitions.
// They must remain distinct so 100% coverage does not erase non-trivial structural variants.
const dA={occ:new Uint8Array([1,1,1,1]),chosen:[{idx:[0,1],ori:0},{idx:[2,3],ori:0}]};
const dB={occ:new Uint8Array([1,1,1,1]),chosen:[{idx:[0,2],ori:1},{idx:[1,3],ori:1}]};
assert(stateOccSig(dA)!==stateOccSig(dB),'exact structural variants collapsed by occupancy-only signature');
q=companionFromVoid(makeOcc(4,3,[[0,1],[1,1],[2,1],[3,1]]),4,0,3,'D4');
assert(!q.exact&&q.patternKind==='winding-residual','torus-winding residual must not be reported as a finite companion');

// Substructure mining: a non-translation-symmetric 4-domino supercell contains a 1-domino tiling motif.
const shape=[[0,0],[1,0]],group='D4',oris=orientations(shape,group);ACTIVE_SHAPE_N=2;ACTIVE_ORIS=oris;ACTIVE_GROUP=group;
const ctx={shape,shapeSig:`${encode(shape)}|${group}`,oris,n:2,group,mode:'cell',algorithm:'auto',opt:{complexity:.56,regularity:.48},start:0,deadline:1e12,runNonce:0};
const hi=oris.findIndex(o=>bounds(o).w===2),vi=oris.findIndex(o=>bounds(o).h===2);
const P=(oi,ax,ay)=>placementFromAnchorHNF(oris,oi,ax,ay,4,0,2),ps=[P(vi,0,0),P(vi,1,0),P(hi,2,0),P(hi,2,1)],occ=new Uint8Array(8),chosen=[];
for(const p of ps){assert(p&&canPlace(p,occ),'invalid source domino construction');place(p,occ,chosen);}
const info=periodicPatternInfoHNF(4,0,2,chosen);assert(info.primitiveTileCount===4&&info.coverage===1,'source must genuinely have primitive m=4');
const r={kind:'periodic',w:4,h:2,lattice:{a:4,b:0,c:2,key:'4,0,2'},placements:chosen.map(p=>({cells:p.cells,rawCells:p.rawCells,ori:p.ori,anchor:p.anchor})),metrics:{coverage:1,primitiveTileCount:4}};
const mined=mineSubstructureResults(ctx,r,4);assert(mined.some(x=>x.metrics?.coverage>=1-1e-12&&x.metrics?.primitiveTileCount===1&&x.metrics?.substructureFrom===4),'m=4 -> m=1 substructure feedback missing');

console.log('OK: v9 whole-residual companion semantics + exact substructure mining');
