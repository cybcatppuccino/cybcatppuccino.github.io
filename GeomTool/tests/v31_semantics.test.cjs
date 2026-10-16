'use strict';
const assert=require('node:assert/strict');const fs=require('node:fs'),vm=require('node:vm');
const {scan,registry}=require('../assets/relations/detectors.js');const {matches}=require('../assets/relations/search.js');
const {Kernel}=require('../assets/kernel.js');const lib=require('../assets/library.js');
function scene(points=[],lines=[],circles=[]){return {points:points.map(([id,x,y])=>({id,label:id,x,y})),lines:lines.map(([id,x1,y1,x2,y2])=>({id,label:id,x1,y1,x2,y2})),circles:circles.map(([id,cx,cy,r])=>({id,label:id,cx,cy,r}))};}
let count=0;function yes(value,label){assert(value,label);count++;}
function result(s,kinds){return scan(s,{kinds,tolerance:'loose',perturb:true});}
// A whole direction class and its perpendicular class generate only one record.
const grid=scene([['A',0,0],['B',1,0],['C',0,1],['D',1,1],['E',0,2],['F',1,2],['G',2,0],['H',2,1]]);
const parallel=result(grid,['parallel']);yes(parallel.entries.some(e=>e.marks.filter(m=>m.type==='line').length>=3),'parallel families form a maximal chain');
yes(!parallel.entries.some(e=>e.id.endsWith(':AB:CD')),'parallel output uses groups, not all point pairs');
const perps=result(grid,['perpendicular']);yes(perps.entries.length<12,'perpendicularity compresses Cartesian products of parallel line groups');
// Equal-length clusters, inverse ratio and displayed implied class names.
const lengths=scene([['A',0,0],['B',2,0],['C',8,0],['D',9,0],['E',0,4],['F',2,4]]);
const ratios=result(lengths,['distanceRatio']);
yes(ratios.entries.some(e=>matches(e,'AB:CD=2')),'2:1 matches forward segment notation');
yes(ratios.entries.some(e=>matches(e,'DC:BA=1/2')),'inverse ratio and reversed endpoint names match the same entry');
yes(ratios.entries.some(e=>e.metric?.factor===1),'equal metric chain retained');
const square=result(scene([['A',0,0],['B',2,0],['C',2,2],['D',0,2]]),['square']);
yes(square.count===1&&matches(square.entries[0],'菱形')&&matches(square.entries[0],'平行四边形'),'square queried by implied rectangle/rhombus/parallelogram types');
// Don't manufacture angle / collinearity near-misses, even next to an outlier.
const near=result(scene([['A',0,0],['B',1,0],['C',2,1e-5],['X',1e11,1e11]]),['collinear']);
yes(near.count===0,'remote outlier does not enlarge local distance tolerance');
const concur=result(scene([['A',-2,-2],['B',2,2],['C',-2,2],['D',2,-2],['E',0,-2],['F',0,2]]),['concurrent']);
yes(concur.count===1&&concur.unstableRejected===0,'three virtual lines genuinely concurrent at an uncreated point');
const trivial=result(scene([['A',0,0],['B',1,0],['C',0,1],['D',1,1]]),['concurrent']);
yes(trivial.count===0,'line fans from shared point do not count as three independent concurrent lines');
// Radical axis must be a point of equal nonzero powers, not just a circle crossing.
const radial=result(scene([['P',1,3]],[],[['c1',0,0,2],['c2',2,0,2]]),['radicalAxis']);
yes(radial.count===1&&radial.entries[0].marks.some(m=>m.type==='radicalAxis'),'nontrivial radical-axis point is rendered with actual axis');
const crossing=result(scene([['P',1,Math.sqrt(3)]],[],[['c1',0,0,2],['c2',2,0,2]]),['radicalAxis']);
yes(crossing.count===0,'intersection of both circles excluded from nontrivial radical-axis claims');
const circleLine=result(scene([], [['l',1,-4,1,4]], [['a',0,0,Math.SQRT2],['b',2,0,Math.SQRT2]]),['circleIntersectionLine']);
yes(circleLine.count===1,'two intersections on same line shown as one common chord');
const triple=result(scene([],[],[['a',1,0,1],['b',0,1,1],['c',-1,0,1]]),['multiCircleCommon']);
yes(triple.count===1,'three actual circles have common point without explicit witness point');
const area=result(scene([['A',0,0],['B',2,0],['C',0,2],['D',2,2]]),['equalArea']);
yes(area.entries.some(e=>e.title.includes('=')),'equal-area triangles chained instead of n-choose-2 output');
const tris=scene([['A',0,0],['B',3,0],['C',0,4],['D',10,0],['E',16,0],['F',10,8],['G',20,0],['H',29,0],['I',20,12]]);
const similar=result(tris,['similarTriangles']);
yes(similar.entries.some(e=>e.title.includes('ABC')&&e.title.includes('DEF')&&e.title.includes('GHI')),'three genuinely similar triangles reported as one family');
// Two actual entries of the user-provided library: no false non-strict matches,
// no duplicate ids, stable under two independent deterministic perturbations.
const context={window:{}};vm.createContext(context);vm.runInContext(fs.readFileSync(__dirname+'/../assets/library-data.js','utf8'),context);
for(const name of ['Conway_circle','Malfatti_s_problem']){
 const item=context.window.GeomLibraryData.entries.find(e=>e.title===name);yes(!!item,`library ${name} available`);
 const converted=lib.convert(item,{attempts:1}),k=new Kernel();k.load(converted.graph,false);
 const nodes=k.snapshot().nodes.filter(n=>n.valid&&n.geom),s={points:[],lines:[],circles:[]};
 for(const n of nodes){const g=n.geom;if(g.kind==='point')s.points.push({id:n.id,label:n.label,x:g.x,y:g.y});
  else if(g.kind==='line')s.lines.push({id:n.id,label:n.label,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y});
  else if(g.kind==='circle')s.circles.push({id:n.id,label:n.label,cx:g.c.x,cy:g.c.y,r:g.r});}
 const results=result(s,['coincident','collinear','parallel','perpendicular','concyclic','distanceRatio']);
 yes(results.tolerance==='strict'&&results.perturbations===2,`library ${name}: no non-strict mode`);
 yes(new Set(results.entries.map(e=>e.id)).size===results.count,`library ${name}: unique semantic relation ids`);
 yes(results.entries.every(e=>e.title&&e.marks.length),`library ${name}: drawable witnesses`);
}
console.log(`PASS v3.1 extended: ${count} assertions; ${registry.length} independent categories`);
