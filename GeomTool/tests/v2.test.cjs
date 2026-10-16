const assert=require('node:assert/strict');
const {Kernel,intersections}=require('../assets/kernel.js');
const Pretty=require('../assets/pretty.js');
const Guides=require('../assets/tool-guides.js');
let tests=0;
function check(v,m){assert.ok(v,m);tests++;}
function eq(a,b,m){assert.deepEqual(a,b,m);tests++;}
function mk(k,op,id,inputs=[],params={},style={}){return k.execute({op,id,inputs,params,style});}
// Required tooling: every bundled UI tool gets its own hand-authored sketch + guide.
const toolIds=['select','pan','free','onCarrier','intersection','midpoint','foot','centroid','circumcenter','incenter','orthocenter','excenter','parallelogram','line','segment','ray','parallel','perpendicular','perpBisector','angleBisector','tangent','tangentFrom','parallel3','perpendicular3','outerAngleBisector','circle','circleRadius','circumcircle','polygon','circleBySegment','diameterCircle','equilateral','foot3','secondIntersection','reflectPoint','reflectLine','translate','rotate','homothety','rotateLeft','rotateRight'];
for(const id of toolIds){
 check(!!Guides.guides[id],`Guide missing: ${id}`);
 check(Guides.sketch(id).includes('<svg')&&Guides.sketch(id).includes('</svg>'),`SVG missing: ${id}`);
 check(Guides.guide(id).steps.length>=1&&Guides.guide(id).effect.length>3,`Step missing: ${id}`);
}
const svg=new Set(toolIds.map(id=>Guides.sketch(id)));
eq(svg.size,toolIds.length,'All 41 tools must show a distinct sketch');
// Stable circle-circle branch: side relative to the ordered center pair must not flip.
const k=new Kernel();mk(k,'referencePoint','O',[],{x:0,y:0});mk(k,'circleRadius','c1',['O'],{radius:2});mk(k,'free','Q',[],{x:1.5,y:0});mk(k,'circleRadius','c2',['Q'],{radius:2});mk(k,'intersection','X0',['c1','c2'],{branch:0});mk(k,'intersection','X1',['c1','c2'],{branch:1});
for(let i=0;i<=48;i++){
 const a=i*Math.PI/24;k.execute({op:'move',id:'Q',x:1.5*Math.cos(a),y:1.5*Math.sin(a)});
 const q=k.geom('Q'),lo=k.geom('X0'),hi=k.geom('X1');
 const cr=p=>q.x*p.y-q.y*p.x;
 check(!!lo&&!!hi&&cr(lo)<-0.2&&cr(hi)>0.2,`Circle branch flipped at angle ${a}`);
}
// Moving a line's endpoint through a full turn: stable signed line direction.
const l=new Kernel();mk(l,'free','O',[],{x:0,y:0});mk(l,'free','Q',[],{x:3,y:0});mk(l,'line','axis',['O','Q']);mk(l,'circleRadius','c',['O'],{radius:2});mk(l,'intersection','L0',['axis','c'],{branch:0});mk(l,'intersection','L1',['axis','c'],{branch:1});
for(let i=0;i<48;i++){
 const a=i*Math.PI/24;l.execute({op:'move',id:'Q',x:3*Math.cos(a),y:3*Math.sin(a)});
 const q=l.geom('Q'),lo=l.geom('L0'),hi=l.geom('L1');
 check(q.x*lo.x+q.y*lo.y<0&&q.x*hi.x+q.y*hi.y>0,`Line circle order flipped ${a}`);
}
// Reference objects survive round-trip and remain viable operands.
const r=new Kernel();r.execute({op:'referencePoint',id:'GTREF_origin',label:'0',params:{x:0,y:0},system:true,style:{system:true,visible:true}});
r.execute({op:'referencePoint',id:'GTREF_one',label:'1',params:{x:1,y:0},system:true,style:{system:true,visible:true}});
mk(r,'line','GTREF_real',['GTREF_origin','GTREF_one'],{}, {system:true,visible:true});
mk(r,'onLine','P',['GTREF_real'],{t:.3});mk(r,'midpoint','M',['GTREF_origin','P']);
const after=new Kernel(r.serialize());check(after.geom('M')?.x===.15,'Reference loaded and usable');
check(after.geom('GTREF_real')?.kind==='line','System line loaded');
assert.throws(()=>after.execute({op:'move',id:'GTREF_origin',x:4,y:5}));tests++;
// Pretty keyframes preserve dependent construction and all constrained degrees of freedom.
const b=new Kernel();mk(b,'free','A',[],{x:0,y:0});mk(b,'free','B',[],{x:4,y:.1});mk(b,'free','C',[],{x:.15,y:.05});mk(b,'midpoint','M',['A','B']);mk(b,'line','h',['A','B']);mk(b,'onLine','T',['h'],{t:.32});mk(b,'circleRadius','circle',['A'],{radius:2});mk(b,'onCircle','U',['circle'],{theta:1.2});mk(b,'polygon','poly',['A','B','C']);
const before=b.serialize(),plan=Pretty.optimize(b,{strength:82,iterations:70});
check(plan.steps>0,'Pretty has accepted descent steps');
check(plan.energyEnd<plan.energyStart,'Pretty must decrease objective');
check(plan.energies.every((v,i)=>i===0||v<plan.energies[i-1]),'Every stored frame accepted only on descent');
for(const position of [0,3,25,50,75,97,100]){
 Pretty.apply(b,plan,position);const A=b.geom('A'),B=b.geom('B'),M=b.geom('M'),T=b.geom('T'),U=b.geom('U');
 check(Math.abs(M.x-(A.x+B.x)/2)<1e-8,'Midpoint stays constrained');
 check(Math.abs((B.x-A.x)*(T.y-A.y)-(B.y-A.y)*(T.x-A.x))<1e-8,'On-line stays constrained');
 check(Math.abs(Math.hypot(U.x-A.x,U.y-A.y)-2)<1e-8,'On-circle stays constrained');
}
Pretty.apply(b,plan,0);
check(Math.abs(b.geom('A').x-before.nodes.find(x=>x.id==='A').params.x)<1e-8,'Slider 0 is original');
Pretty.apply(b,plan,100);
check(Math.abs(b.geom('A').x-before.nodes.find(x=>x.id==='A').params.x)>1e-5,'Slider 100 is optimized');
console.log(`PASS v2 — ${tests} assertions; 41 unique guides, stable intersections, references, path-based Pretty`);
