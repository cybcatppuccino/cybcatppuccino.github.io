'use strict';
const assert=require('node:assert/strict');
const {Kernel,METHODS}=require('../assets/kernel.js'),DD=require('../assets/ddar.js'),Goals=require('../assets/ddar-goals.js');
const put=(k,op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
const p=(k,id,x,y)=>put(k,'free',id,[],{x,y});
const kernel=()=>new Kernel();
const has=(r,k,rule)=>r.facts.some(f=>f.kind===k&&(!rule||f.rule===rule));
const proof=(k,kind,args)=>DD.infer(k.snapshot(),{goal:{kind,args}}).goal;
{
 const k=kernel();p(k,'A',0,0);p(k,'B',6,0);p(k,'C',2,5);put(k,'centroid','G',['A','B','C']);
 const r=DD.infer(k.snapshot());assert.equal(r.virtuals.filter(v=>v.op==='midpoint').length,2);
 assert(has(r,'eqratio','CENTROID_RATIO'));assert(has(r,'coll','CENTROID_MEDIAN'));
 const m=r.virtuals.find(v=>v.inputs.includes('B')&&v.inputs.includes('C')).id;
 assert.equal(proof(k,'eqratio',[DD.seg('A','G'),DD.seg('G',m),'2']).status,'proved');
}
{
 const k=kernel();p(k,'A',0,0);p(k,'B',4,0);put(k,'perpBisector','l',['A','B']);
 const r=DD.infer(k.snapshot());assert(r.virtuals.some(v=>v.op==='midpoint'));
 const m=r.virtuals[0].id;assert.equal(proof(k,'onl',[m,'L:l']).status,'proved');
 assert.equal(proof(k,'perp',['L:l','P:A|B']).status,'proved');
}
{
 const k=kernel();p(k,'A',4,0);p(k,'B',0,0);p(k,'C',0,3);
 put(k,'angleBisector','l',['A','B','C']);
 assert.equal(proof(k,'eqangleLines',['P:A|B','L:l','L:l','P:B|C']).status,'proved');
 const r=DD.infer(k.snapshot());assert(has(r,'eqangleLines','ANGLEBISECTOR_DEF'));
}
{
 const k=kernel();p(k,'A',-1,2);p(k,'B',2,1);p(k,'C',4,4);
 put(k,'incenter','I',['A','B','C']);put(k,'excenter','J',['A','B','C']);
 assert.equal(proof(k,'eqangleLines',['P:A|B','P:B|I','P:B|I','P:B|C']).status,'proved');
 assert(has(DD.infer(k.snapshot()),'eqangleLines','EXCENTER_BISECTOR'));
}
{
 const k=kernel();p(k,'A',1,3);p(k,'B',0,0);p(k,'C',2,0);put(k,'line','l',['B','C']);
 put(k,'reflectLine','R',['A','l']);
 const r=DD.infer(k.snapshot());assert(r.virtuals.some(v=>v.op==='foot'));
 assert.equal(proof(k,'perp',['P:A|R','L:l']).status,'proved');
 assert(has(r,'midp','AXIAL_REFLECTION'));
}
{
 const k=kernel();p(k,'P',0,2);p(k,'A',0,0);p(k,'B',3,0);put(k,'translate','Q',['P','A','B']);
 assert.equal(proof(k,'cong',['P|Q','A|B']).status,'proved');
 assert.equal(proof(k,'para',['P:P|Q','P:A|B']).status,'proved');
 put(k,'homothety','R',['P','A'],{factor:2});
 assert.equal(proof(k,'eqratio',['A|R','A|P','2']).status,'proved');
 put(k,'rotateLeft','L',['P','A']);
 assert.equal(proof(k,'perp',['P:A|P','P:A|L']).status,'proved');
 put(k,'rotate','S',['P','A'],{degrees:60});
 assert.equal(proof(k,'cong',['P|S','A|P']).status,'proved');
}
{
 const k=kernel();p(k,'A',-2,0);p(k,'B',2,0);p(k,'C',0,2);
 put(k,'circumcircle','c',['A','B','C']);put(k,'onCircle','D',['c'],{theta:2.4});
 const r=DD.infer(k.snapshot());assert.equal(proof(k,'cong',['@O_c|A','@O_c|B']).status,'proved');
 assert(has(r,'cyclic'));
 put(k,'onCircumcircle','E',['A','B','C'],{theta:1.65});
 const rr=DD.infer(k.snapshot());assert(rr.virtuals.some(v=>v.op==='circumcircle'));
 assert.equal(proof(k,'cyclic',['A','B','C','E']).status,'proved');
}
{
 const k=kernel();p(k,'O',0,0);p(k,'A',2,0);p(k,'P',5,4);
 put(k,'circle','c',['O','A']);put(k,'tangentFrom','t',['P','c'],{branch:0});
 const r=DD.infer(k.snapshot());const h=r.virtuals.find(v=>v.op==='foot'&&v.inputs.includes('t'))?.id;
 assert(h,'tangent contact point should be compiled');
 assert.equal(proof(k,'onc',[h,'c']).status,'proved');
 assert.equal(proof(k,'perp',['P:'+DD.seg('O',h),'L:t']).status,'proved');
}
{
 const k=kernel();p(k,'A',-1,0);p(k,'B',4,0);p(k,'C',1,3);
 put(k,'centroid','G',['A','B','C']);put(k,'circumcenter','O',['A','B','C']);put(k,'orthocenter','H',['A','B','C']);
 assert.equal(proof(k,'coll',['O','G','H']).status,'proved');
 assert.equal(proof(k,'eqratio',['G|H','G|O','2']).status,'proved');
 put(k,'incenter','I',['A','B','C']);put(k,'excenter','J',['A','B','C']);
 assert.equal(proof(k,'perp',['P:B|I','P:B|J']).status,'proved');
 assert.equal(proof(k,'coll',['A','I','J']).status,'proved');
}
{
 const k=kernel();p(k,'O',0,0);p(k,'P',3,1);put(k,'rotate','Q',['P','O'],{degrees:37});
 assert.equal(proof(k,'angconst',['P:O|P','P:O|Q','37']).status,'proved');
 assert.equal(proof(k,'angconst',['P:O|P','P:O|Q','38']).status,'unknown');
 const custom=Goals.parseGoal('angconst O P O Q 37',k.snapshot());
 assert.equal(proof(k,custom.kind,custom.args).status,'proved');
}
{
 const k=kernel();p(k,'A',0,0);p(k,'B',4,0);put(k,'line','l',['A','B']);
 put(k,'onLine','X',['l'],{t:.25});
 assert.equal(proof(k,'eqratio',['A|X','A|B','.25']).status,'proved');
}
{
 const k=kernel();p(k,'A',0,0);p(k,'B',3,0);p(k,'C',0,2);p(k,'D',5,1);p(k,'E',6,5);p(k,'F',7,4);p(k,'G',2,7);p(k,'H',6,8);
 for(const [goal,kind] of [['AB=CD','cong'],['AB ⟂ CD','perp'],['∠ABC=∠DEF','eqangleLines'],['∠(AB,CD)=∠(EF,GH)','eqangleLines']])
  assert.equal(Goals.parseGoal(goal,k.snapshot()).kind,kind);
}
{
 const k=kernel();p(k,'A',0,0);p(k,'B',4,0);p(k,'C',0,3);p(k,'D',4,3);
 const g=Goals.discover(k.snapshot());assert(g.goals.some(x=>x.kind==='cong'));
 assert(g.goals.some(x=>x.kind==='eqangleLines'));
 assert(g.goals.some(x=>x.kind==='para'));
 assert.equal(proof(k,'cong',['A|B','C|D']).status,'unknown','equal numerical distances are not proofs');
 assert.equal(proof(k,'eqangleLines',['P:A|B','P:A|C','P:C|D','P:B|D']).status,'unknown','equal angle coincidences are not proofs');
}
{
 const k=kernel();for(let i=0;i<35;i++)p(k,'P'+i,Math.sin(i*3.15)*10+i*.18,Math.cos(i*2.14)*10);
 const start=Date.now(),r=Goals.discover(k.snapshot(),{maxGoals:600}),d=DD.infer(k.snapshot());
 assert(Date.now()-start<4000,'bounded discovery should not enumerate all 4-segment permutations');
 assert(d.total===0,'35 unconditioned free points should yield no formal theorems');
 assert(r.goals.length<=600);
}
const unsupported=Object.keys(METHODS).filter(x=>!new Set(['referencePoint','free','real','onLine','circleCenter','onCircle','line','segment','ray','circle','circleRadius','midpoint','intersection','perpendicular','parallel','perpBisector','angleBisector','foot','reflectPoint','reflectLine','translate','rotate','homothety','centroid','circumcenter','incenter','orthocenter','circumcircle','tangent','tangentFrom','parallelogram','excenter','outerAngleBisector','diameterCircle','circleBySegment','equilateral','rotateRight','rotateLeft','foot3','parallel3','perpendicular3','onCircumcircle','secondIntersection','polygon']).has(x));
assert.deepEqual(unsupported,[],'new Kernel constructions need a DDAR compilation audit');
console.log('PASS v4.3 construction compiler / angle goals / cyclic linkage / target discovery / formal goal separation');
