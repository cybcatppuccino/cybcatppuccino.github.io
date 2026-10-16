'use strict';
const assert=require('node:assert/strict');
const {scan,registry}=require('../assets/relations/detectors.js');
const {matches}=require('../assets/relations/search.js');
const {makeContext}=require('../assets/relations/core.js');
const {Kernel}=require('../assets/kernel.js');
const scene=(p=[],l=[],c=[])=>({points:p.map(([id,x,y])=>({id,x,y,label:id})),lines:l.map(([id,x1,y1,x2,y2])=>({id,label:id,x1,y1,x2,y2})),circles:c.map(([id,cx,cy,r])=>({id,label:id,cx,cy,r}))});
let n=0;const ok=(x,name)=>{assert.ok(x,name);n++};const run=(s,kinds,perturb=false)=>scan(s,{kinds,perturb});
// 1: optional perturbation, deterministic and no input mutation.
const exact=scene([['A',0,0],['B',2,0],['C',0,2],['D',2,2]]);
const copy=JSON.stringify(exact);
let a=run(exact,['collinear','distanceRatio'],false),b=run(exact,['collinear','distanceRatio'],true);
ok(a.perturbations===0&&b.perturbations===2,'microperturbation is opt-in');
ok(JSON.stringify(exact)===copy,'both scan modes are strictly read-only');
ok(a.entries.every(e=>['construction','discovery'].includes(e.source)),'each candidate has a source type');
ok(new Set(a.entries.map(e=>e.id)).size===a.entries.length,'unique result IDs');
// 2: triangle correspondence, including nontrivial vertex swap.
const triangles=scene([['A',0,0],['B',3,0],['C',0,4],['D',20,0],['E',20,4],['F',23,0]]);
const cong=run(triangles,['congruentTriangles']);
const sameShape=run(triangles,['similarTriangles']);
ok(cong.entries.some(e=>e.title.includes('△ABC')&&e.title.includes('△DFE')),'correct congruent triangle vertex correspondence');
ok(!cong.entries.some(e=>e.title.includes('△ABC ≅ △DEF')),'never state wrong vertex correspondence');
ok(cong.entries.some(e=>e.correspondence?.every(x=>x.alternates.length>=1)),'correspondence metadata preserved');
ok(sameShape.count===0,'pure congruence is more specific than similarity');
const scaled=scene([['A',0,0],['B',3,0],['C',0,4],['D',20,0],['E',20,16],['F',32,0]]);
ok(run(scaled,['similarTriangles']).entries.some(e=>e.title.includes('△ABC')&&e.title.includes('△DFE')),'similar triangle permutation recovered');
// 3: tolerance independent of uniform rotations and scales, and excludes near-misses.
for(const scale of [0.01,1,1e4]){
 const R=scene([['A',0,0],['B',scale*3,0],['C',0,scale*4],['X',scale*6,0],['Y',scale*6,scale*8]]);
 const rr=run(R,['perpendicular','collinear'],false);
 ok(rr.entries.some(e=>e.kind==='perpendicular'),'exact orthogonal relation at scale '+scale);
 const near=scene([['A',0,0],['B',scale*3,0],['C',scale*6,scale*1e-6]]);
 ok(!run(near,['collinear'],false).count,'perturbed off-axis point rejected at scale '+scale);
}
const rad=0.732,turn=([id,x,y])=>[id,x*Math.cos(rad)-y*Math.sin(rad),x*Math.sin(rad)+y*Math.cos(rad)];
ok(run(scene([['A',0,0],['B',4,0],['C',0,3]].map(turn)),['perpendicular'],false).count>=1,'rotational invariance');
const outlier=scene([['A',0,0],['B',2,0],['C',4,1e-6],['Z',1e11,-1e11]]);
ok(!run(outlier,['collinear'],false).count,'remote outlier does not loosen local tolerance');
// 4: quadratic L2 families, dimensional correctness, ratio search, no sign mixing.
const quadratic=scene([['A',0,0],['B',2,0],['C',0,2],['D',4,0],['P',0,4]],[],[['unit',0,0,1]]);
const q=run(quadratic,['quadraticRatio'],false);
ok(q.entries.length>0,'quadratic registry emits relations');
ok(q.entries.some(e=>e.kind==='quadraticRatio'&&e.title.includes('²')),'squared side lengths included');
ok(q.entries.some(e=>e.title.includes('S(△')),'triangle areas included');
ok(q.entries.some(e=>e.title.includes('Pow(')),'signed circle power included');
ok(q.entries.every(e=>!e.metric||e.metric.high.concat(e.metric.low).every(m=>m.type)),'structured metric types preserved');
ok(q.entries.some(e=>matches(e,'AB²:AD²=1/4')),'inverse squared ratio syntax resolves canonical entry');
ok(q.entries.some(e=>matches(e,'AD²:AB²=4')),'forward squared ratio syntax resolves canonical entry');
ok(q.entries.every(e=>{let m=e.metric;if(!m)return true;const vals=m.high.concat(m.low).filter(x=>x.type==='pointPower');return vals.every(x=>x.key.startsWith('W:'));}),'point power classification is explicit');
// 5: provenance facts from actual Kernel commands, not matching by coincidence.
const k=new Kernel();k.execute({op:'free',id:'A',inputs:[],params:{x:0,y:0}});k.execute({op:'free',id:'B',inputs:[],params:{x:2,y:0}});k.execute({op:'midpoint',id:'M',inputs:['A','B'],params:{}});
const snapshot=k.snapshot();const selected={points:[],lines:[],circles:[],constructions:snapshot.nodes.map(x=>({id:x.id,op:x.op,inputs:x.inputs}))};
for(const item of snapshot.nodes){if(item.geom?.kind==='point')selected.points.push({id:item.id,label:item.label,x:item.geom.x,y:item.geom.y});}
const mid=run(selected,['midpoint','distanceRatio','collinear'],false);
ok(mid.entries.some(e=>e.kind==='midpoint'&&e.source==='construction'),'midpoint explicitly proven by its generator');
ok(mid.constructionStats.definitions.some(f=>f.op==='midpoint'),'construction statistic includes midpoint definition');
ok(mid.implications.edges.some(e=>e.rule==='MIDPOINT_DEF'),'typed implication graph contains midpoint definition edge');
ok(mid.entries.some(e=>e.kind==='distanceRatio'&&e.metric?.factor===1&&e.source==='construction'),'equal halves tagged as generator consequence');
ok(mid.entries.some(e=>e.kind==='collinear'&&e.source==='construction'),'midpoint collinearity directly tagged');
ok(mid.entries.some(e=>e.kind==='distanceRatio'&&e.metric?.factor===2&&e.source==='discovery'),'other metric ratios are not automatically tagged');
const forget=run({...selected,constructions:[]},['midpoint','distanceRatio'],false);
ok(forget.entries.every(e=>e.source==='discovery'),'no reconstruction of causal provenance from coordinates alone');
const ext={...selected,points:[...selected.points,{id:'N',label:'N',x:1,y:1}]};
const chained=run(ext,['distanceRatio'],false);
ok(chained.entries.some(e=>e.metric?.factor===1&&e.source==='mixed'),'direct midpoint equality inside longer equal chain is marked as mixed rather than entirely constructed');
const axis=scene([['P',1,3]],[],[['c1',0,0,2],['c2',2,0,2]]);
const withAxis=run(axis,['quadraticRatio','radicalAxis'],false);
ok(withAxis.entries.some(e=>e.kind==='radicalAxis')&&!withAxis.entries.some(e=>e.kind==='quadraticRatio'&&e.metric?.factor===1&&e.metric.high.every(w=>w.type==='pointPower')),'equal powers of same point and two circles imply radical-axis fact without duplicate row');
ok(run(axis,['quadraticRatio'],false).entries.some(e=>e.metric?.factor===1&&e.metric.high.every(w=>w.type==='pointPower')),'quadratic equality retained if radical-axis rule not selected');
// 6: implications must not suppress primitives if parent is unselected.
const square=scene([['A',0,0],['B',2,0],['C',2,2],['D',0,2]]);
ok(run(square,['square'],false).count===1,'square preserved');
ok(run(square,['perpendicular'],false).entries.length>=1,'perpendicular remains searchable without square');
ok(run(square,['distanceRatio'],false).entries.some(e=>e.metric?.factor===1),'natural square equality remains');
// 7: param noise: large degenerate drawings; list identities do not oscillate after 1e-11 perturb.
const dense=scene(Array.from({length:12},(_,i)=>['P'+i,2*Math.floor(i/3),2*(i%3)]),[],[['c',3,2,Math.SQRT2]]);
const kinds=['collinear','parallel','perpendicular','distanceRatio','quadraticRatio','equalArea','congruentTriangles'];
let base=run(dense,kinds,false),plus=run({...dense,points:dense.points.map(p=>p.id==='P5'?{...p,x:p.x+1e-11,y:p.y-1e-11}:p)},kinds,false);
ok(base.count<500,'quadratic candidates manageable on 12-point grid');
ok(Math.abs(base.count-plus.count)<30,'tiny physical movement does not yield a combinatorial avalanche');
ok(new Set(base.entries.map(e=>e.id)).size===base.count,'stress grid stable unique IDs');
ok(registry.some(r=>r.id==='quadraticRatio'),'quadratic category registered');
console.log('PASS v3.3 relations:',n,'assertions', 'grid=',base.count,'perturb=',plus.count);
