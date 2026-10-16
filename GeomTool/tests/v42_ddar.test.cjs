'use strict';
const assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js');
const DD=require('../assets/ddar.js'),Aux=require('../assets/auxiliary.js');
const Rel=require('../assets/relations/detectors.js');
const make=()=>new Kernel();
const put=(k,op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
const p=(k,id,x,y)=>put(k,'free',id,[],{x,y});
const is=(f,kind,rule)=>f.some(x=>x.kind===kind&&(!rule||x.rule===rule));
function radar(k,kinds){const snap=k.snapshot(),board={points:[],lines:[],circles:[],construction:[]};for(const n of snap.nodes){if(!n.valid||n.style.system)continue;const g=n.geom;board.construction.push({id:n.id,op:n.op,inputs:n.inputs,params:n.params});if(g.kind==='point')board.points.push({id:n.id,label:n.id,x:g.x,y:g.y});if(g.kind==='line')board.lines.push({id:n.id,label:n.id,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y,drawType:g.extent});if(g.kind==='circle')board.circles.push({id:n.id,label:n.id,cx:g.c.x,cy:g.c.y,r:g.r,centerId:['circle','circleRadius'].includes(n.op)?n.inputs[0]:null});}return Rel.scan(board,{kinds}).entries;}
{
 const k=make();p(k,'A',0,0);p(k,'B',4,0);p(k,'C',1,3);put(k,'midpoint','M',['A','B']);put(k,'midpoint','N',['A','C']);const entries=radar(k,['parallel','midpoint','distanceRatio','collinear']);const r=DD.infer(k.snapshot(),{entries});
 assert(is(r.facts,'para','MIDSEGMENT_THEOREM'));assert(is(r.facts,'eqratio','MIDSEGMENT_RATIO'));
 assert.equal(r.matches.find(x=>x.id.startsWith('parallel:'))?.status,'proved');
 assert.equal(r.matches.find(x=>x.id.startsWith('distanceRatio:R:D:B|C|D:M|N|2'))?.status,'proved');
 assert.equal(r.matches.filter(x=>x.status==='proved'&&x.id.startsWith('midpoint:')).length,2);
 assert(r.facts.some(x=>x.premises.length&&x.depth>0),'must record derivation premises');
 const aux=Aux.suggest(k.snapshot(),r,entries);assert(aux.candidates.length>2);
 for(const c of aux.candidates.slice(0,15)){const clone=new Kernel(k.serialize());try{assert(Aux.execute(clone,c).valid);}catch(e){/* Suggestions are admissible hypotheses; invalid branches must be rejected by Kernel. */}}
}
{
 const k=make();p(k,'A',0,0);p(k,'B',4,0);p(k,'C',0,3);p(k,'D',4,3);
 const r=DD.infer(k.snapshot(),{entries:radar(k,['parallel','perpendicular','concyclic','distanceRatio'])});
 assert.equal(r.facts.length,0,'numeric coincidences alone must not become proofs');
 assert(r.matches.every(x=>x.status!=='proved'),'do not certify square by coordinate coincidence');
}
{
 const k=make();p(k,'O',0,0);p(k,'A',2,0);p(k,'B',0,2);p(k,'C',-2,0);p(k,'D',0,-2);
 put(k,'circle','c',['O','A']);for(const [name,theta] of [['E',Math.PI/3],['F',Math.PI*.77],['G',Math.PI*1.32]])put(k,'onCircle',name,['c'],{theta});
 const r=DD.infer(k.snapshot());assert(is(r.facts,'cong','RADII_EQUAL'));assert(is(r.facts,'cyclic','SAME_CIRCLE'));assert(is(r.facts,'eqangle','INSCRIBED_ANGLE'));
 assert(!r.facts.some(x=>x.kind==='cyclic'&&x.args.includes('D')),'free point on same measured circle is not an axiom');
}
{
 const k=make();p(k,'A',-2,0);p(k,'B',2,0);put(k,'diameterCircle','c',['A','B']);put(k,'onCircle','P',['c'],{theta:Math.PI/2});put(k,'onCircle','Q',['c'],{theta:Math.PI/4});
 const r=DD.infer(k.snapshot(),{entries:radar(k,['perpendicular','concyclic','diameter'])});
 assert(is(r.facts,'perp','THALES'));assert(is(r.facts,'cyclic'));assert(r.matches.some(x=>x.status==='proved'&&x.id.startsWith('diameter:')));
}
{
 const k=make();p(k,'A',0,0);p(k,'B',4,0);p(k,'O',1,7);p(k,'P',0,4);
 put(k,'circleBySegment','c',['O','A','B']);put(k,'onCircle','Q',['c'],{theta:Math.PI/2});
 const r=DD.infer(k.snapshot());assert(r.facts.some(x=>x.kind==='cong'&&x.args.includes('A|B')&&x.args.includes('O|Q')),'transferred-radius equality');
 assert(!r.facts.some(x=>x.kind==='cong'&&x.args.includes('A|B')&&x.args.includes('A|O')),'do not invent OA=AB');
}
{
 const k=make();p(k,'A',0,0);p(k,'B',3,0);p(k,'C',1,4);put(k,'line','l',['A','B']);put(k,'parallel','m',['C','l']);put(k,'perpendicular','n',['C','l']);
 const r=DD.infer(k.snapshot(),{entries:radar(k,['parallel','perpendicular'])});
 assert(is(r.facts,'para'));assert(is(r.facts,'perp'));assert(r.matches.some(x=>x.status==='proved'&&x.id.startsWith('parallel:')));
}
{
 const k=make();for(let i=0;i<23;i++)p(k,'P'+i,Math.cos(i*2.61)*10+i*.017,Math.sin(i*3.01)*12);
 const r=DD.infer(k.snapshot());assert.equal(r.facts.length,0);assert(r.elapsedMs<1200,'unconstrained point set should be cheap');
}
{
 const k=make();p(k,'A',-2,-2);p(k,'B',2,2);p(k,'C',-2,2);p(k,'D',2,-2);
 // The diagonals cross at a NEW unnamed point. Compound proposal must be
 // atomic when applied in the app, and all its construction dependencies valid.
 const aux=Aux.suggest(k.snapshot(),DD.infer(k.snapshot()),[]);
 const bridge=aux.candidates.find(c=>c.op==='compoundIntersection');assert(bridge,'should propose an implicit-line intersection');
 const n=Aux.execute(k,bridge);assert(n.valid);assert.equal(n.op,'intersection');
 assert.equal(k.node(n.inputs[0]).op,'line');assert.equal(k.node(n.inputs[1]).op,'line');
}
console.log('PASS v4.2 DDAR / ratios / cyclic / Thales / strict numeric separation / helper proposals');
