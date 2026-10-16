'use strict';
const assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js'),C=require('../assets/pretty-constraints.js'),Pretty=require('../assets/pretty.js'),D=require('../assets/relations/detectors.js'),Compiler=require('../assets/ddar-compiler.js'),DDAR=require('../assets/ddar.js');
let passed=0;
const check=(x,msg)=>{assert.ok(x,msg);passed++;};
function make(){const k=new Kernel();for(const [id,x,y] of [['A',.1,-.2],['B',3.3,.5],['C',1.7,1.1],['D',.3,2.4],['E',-2.1,1.6],['F',1.3,4.1]])k.execute({op:'free',id,params:{x,y}});return k;}
const cases=[
 ['col A B C','perp A B A D'],
 ['col A B C','perp A B A D','dist A B = E F'],
 ['col A B C','perp A B A D','dist A B = E F','dist A C = 2 A B'],
 ['col A B C','perp A B A D','dist A B = E F','dist A C = 2 A B','para A B E F'],
];
function board(k){const out={points:[],lines:[],circles:[],construction:[]};for(const n of k.snapshot().nodes){if(!n.valid||!n.geom)continue;const g=n.geom;out.construction.push({id:n.id,op:n.op,inputs:n.inputs});if(g.kind==='point')out.points.push({id:n.id,x:g.x,y:g.y,label:n.id});if(g.kind==='line')out.lines.push({id:n.id,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y});if(g.kind==='circle')out.circles.push({id:n.id,cx:g.c.x,cy:g.c.y,r:g.r});}return out;}
for(const ss of cases){
 const k=make(),cs=ss.map(x=>C.parse(x,k)),prior=JSON.stringify(k.serialize());
 const plan=Pretty.optimize(k,{constraints:cs,iterations:130,restarts:2,strength:60});
 check(JSON.stringify(k.serialize())===prior,'planning must never modify user geometry');
 Pretty.apply(k,plan,100);const stats=C.assess(cs,k,5e-11);
 const max=Math.max(...stats.map(x=>x.error));
 console.log('precision',ss.length,'max',max.toExponential(5),'solver',plan.feasibility.maxError.toExponential(5),'rounds',plan.rounds,'beauty',plan.beautyEnd-plan.beautyStart);
 check(max<5e-10,ss.length+' constraints reach detector/AG2 precision: '+JSON.stringify(stats));
 check(stats.every(x=>x.satisfied),ss.length+' constraints all true');
 const report=D.scan(board(k),{kinds:['collinear','perpendicular','parallel','distanceRatio'],perturb:false});
 console.log('scan',ss.length, report.perturbations,report.entries.filter(x=>['collinear','perpendicular','parallel'].includes(x.kind)).map(x=>x.kind+':'+x.title).slice(0,8));
 check(report.perturbations===0,'default no perturb');
 check(report.entries.some(x=>x.kind==='collinear'), 'detector finds col');
 check(report.entries.some(x=>x.kind==='perpendicular'), 'detector finds perpendicular');
 if(ss.includes('para A B E F'))check(report.entries.some(x=>x.kind==='parallel'),'detector finds parallel');
 if(ss.some(s=>s.includes(' = E F')))check(report.entries.some(x=>x.kind==='distanceRatio'),'detector finds length ratio');
 const base=Compiler.compile(k.snapshot()),ag=C.ag2(cs,k,base);check(ag.predicates.some(p=>p.name==='coll'),'AG2 col injected');
 check(ag.predicates.some(p=>p.name==='perp'),'AG2 perp injected');
}
const k=make();
for(const text of ['ang(A,B,C)=90°','ang(A,B,C)=90deg','ang(A,B,C)=pi/2','dist A B = sqrt(5) C D','(A B)^2=(C D)^2','collinear(A,B,C)','cyclic(A,B,C,D)','simtri (A B C) (D E F)','subset(A,(B,C))','eqratio A B C D E F A C']){check(!!C.parse(text,k), 'parse '+text);}
for(const text of ['dist A B + area A B C = 2','dist A B = S(A B C)','tan (A B) (C D)','(A B)^(C D) = 3','sqrt(-5) = A B'])assert.throws(()=>C.parse(text,k),text),passed++;
// Hidden generic perturbation is forbidden unless explicitly permitted.
// Construct an AG1 numerical degeneracy and test the opt-in guard if possible.
const rev=JSON.stringify(k.serialize());D.scan(board(k),{kinds:['collinear'],perturb:false});check(rev===JSON.stringify(k.serialize()),'relation scan does not modify kernel');
{
 const k=new Kernel();for(const [id,x,y] of [['A',.2,-.1],['B',2.1,.21],['C',1.73,1.81],['D',-.2,1.34],['E',-1,1.1],['F',.5,3.0],['O',.4,.5],['R',1.7,.7]])k.execute({op:'free',id,params:{x,y}});
 k.execute({op:'circle',id:'c',inputs:['O','R']});k.execute({op:'line',id:'l',inputs:['E','F']});
 const cs=['cyc A B C D','tan c l'].map(s=>C.parse(s,k));const before=JSON.stringify(k.serialize()),plan=Pretty.optimize(k,{constraints:cs,iterations:120});
 check(before===JSON.stringify(k.serialize()),'circle optimization only on scratch kernel');
 Pretty.apply(k,plan,100);
 check(C.assess(cs,k,5e-11).every(s=>s.satisfied),'cyclic and tangent both machine accurate');
 const r=D.scan(board(k),{kinds:['concyclic','circleLineTangent'],perturb:false});
 check(r.entries.some(x=>x.kind==='concyclic'),'unperturbed scan finds cyclic');
 check(r.entries.some(x=>x.kind==='circleLineTangent'),'unperturbed scan finds tangency');
}
{
 const k=new Kernel();for(const [id,x,y] of [['A',0,0],['B',0,0],['C',2,0]])k.execute({op:'free',id,params:{x,y}});
 assert.throws(()=>DDAR.prepare(k.snapshot(),{engine:'ag1',enumerate:false}),/默认禁止静默/);
 passed++;
}
{
 let worst=0;
 for(let seed=1;seed<=60;seed++){
  let v=seed;const rand=()=>{v=(Math.imul(1664525,v)+1013904223)>>>0;return v/4294967296;};
  const k=new Kernel();for(const [id,x,y] of [['A',0,0],['B',3,.3],['C',1,1],['D',.5,2.5],['E',-2,1.5],['F',1,3.7]])
   k.execute({op:'free',id,params:{x:x+1.2*(rand()-.5),y:y+1.2*(rand()-.5)}});
  const constraints=['col A B C','perp A B A D','dist A B = E F','dist A C = 2 A B','para A B E F'].map(s=>C.parse(s,k));
  const plan=Pretty.optimize(k,{constraints,iterations:130});Pretty.apply(k,plan,100);
  const max=Math.max(...C.assess(constraints,k).map(x=>x.error));worst=Math.max(worst,max);
  check(max<1e-9,'seed '+seed+' five goals max '+max);
 }
 console.log('5-goal 60-seed worst',worst.toExponential(4));
}
console.log('PASS v5.4 precision '+passed+' assertions');
