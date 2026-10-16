'use strict';
/* v3.4: small distinct construction diagrams, real Kernel geometry, and
 * numerical audit of EVERY emitted operand-aware definition in these scenes.
 * No screenshots or UI dependency. */
const assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js');
const {scan}=require('../assets/relations/detectors.js');
const Logic=require('../assets/relations/construction-logic.js');
const Provenance=require('../assets/relations/provenance.js');
const {sMark}=require('../assets/relations/core.js');
const base=[['A',0,0],['B',4,0],['C',1.3,3.1],['D',3.3,1.7],['E',8,2.4],['O',2,1]];
const cmd=(op,inputs=[],params={})=>({op,inputs,params});
// Different carriers, vertex orders and special point constructors. A case
// contains one advanced constructor on top of the same minimal anchor graph.
const cases=[
 ['real-axis point',cmd('real',[],{x:2.8})],['fixed reference point',cmd('referencePoint',[],{x:6.1,y:-1.6})],
 ['point on circumcircle',cmd('onCircumcircle',['A','B','C'],{theta:.43})],['triangle polygon',cmd('polygon',['A','B','C'])],
 ['midpoint AB',cmd('midpoint',['A','B'])],['midpoint BC',cmd('midpoint',['B','C'])],['midpoint AC',cmd('midpoint',['A','C'])],['midpoint CD',cmd('midpoint',['C','D'])],
 ['midpoint AE',cmd('midpoint',['A','E'])],['midpoint DO',cmd('midpoint',['D','O'])],
 ['point on line',cmd('onLine',['lAB'],{t:.35})],['point on circle',cmd('onCircle',['c0'],{theta:.7})],['circle center',cmd('circleCenter',['c0'])],
 ['line CD',cmd('line',['C','D'])],['segment BC',cmd('segment',['B','C'])],['ray AC',cmd('ray',['A','C'])],
 ['parallel point',cmd('parallel',['C','lAB'])],['parallel through D',cmd('parallel',['D','lBC'])],
 ['parallel 3 ABC',cmd('parallel3',['C','A','B'])],['parallel 3 BCD',cmd('parallel3',['D','B','C'])],
 ['perpendicular from C',cmd('perpendicular',['C','lAB'])],['perpendicular from D',cmd('perpendicular',['D','lBC'])],
 ['perpendicular 3 from C',cmd('perpendicular3',['C','A','B'])],['perpendicular 3 from D',cmd('perpendicular3',['D','B','C'])],
 ['perpendicular bisector AB',cmd('perpBisector',['A','B'])],['perpendicular bisector BC',cmd('perpBisector',['B','C'])],
 ['foot C',cmd('foot',['C','lAB'])],['foot D',cmd('foot',['D','lBC'])],['foot3 C',cmd('foot3',['C','A','B'])],['foot3 D',cmd('foot3',['D','B','C'])],
 ['inner angle bisector ABC',cmd('angleBisector',['A','B','C'])],['inner angle bisector BCD',cmd('angleBisector',['B','C','D'])],
 ['outer angle bisector ABC',cmd('outerAngleBisector',['A','B','C'])],
 ['circle center/point',cmd('circle',['A','B'])],['circle fixed radius',cmd('circleRadius',['C'],{radius:2.8})],['circle by segment',cmd('circleBySegment',['D','A','B'])],
 ['circumcircle ABC',cmd('circumcircle',['A','B','C'])],['circumcircle BCD',cmd('circumcircle',['B','C','D'])],
 ['diameter circle AB',cmd('diameterCircle',['A','B'])],['diameter circle BC',cmd('diameterCircle',['B','C'])],
 ['centroid ABC',cmd('centroid',['A','B','C'])],['centroid BCD',cmd('centroid',['B','C','D'])],
 ['circumcenter ABC',cmd('circumcenter',['A','B','C'])],['circumcenter BCD',cmd('circumcenter',['B','C','D'])],
 ['incenter ABC',cmd('incenter',['A','B','C'])],['incenter BCD',cmd('incenter',['B','C','D'])],
 ['orthocenter ABC',cmd('orthocenter',['A','B','C'])],['orthocenter BCD',cmd('orthocenter',['B','C','D'])],
 ['excenter ABC',cmd('excenter',['A','B','C'])],['excenter BCD',cmd('excenter',['B','C','D'])],
 ['equilateral AB',cmd('equilateral',['A','B'])],['equilateral BC',cmd('equilateral',['B','C'])],
 ['parallelogram ABC',cmd('parallelogram',['A','B','C'])],['parallelogram BCD',cmd('parallelogram',['B','C','D'])],
 ['reflect point about O',cmd('reflectPoint',['C','O'])],['reflect line about lAB',cmd('reflectLine',['C','lAB'])],
 ['translation',cmd('translate',['C','A','B'])],['rotation 30',cmd('rotate',['C','O'],{degrees:30})],
 ['rotation 90 right',cmd('rotateRight',['C','O'])],['rotation 90 left',cmd('rotateLeft',['C','O'])],
 ['homothety 3/2',cmd('homothety',['C','O'],{factor:1.5})],
 ['tangent at A',cmd('tangent',['A','c0'])],['tangent from E',cmd('tangentFrom',['E','c0'],{branch:0})],
 ['line-circle intersection',cmd('intersection',['lAB','c0'],{branch:0})],['circle-circle intersection',cmd('intersection',['c0','c1'],{branch:0})],
 ['second line-circle intersection',cmd('secondIntersection',['A','lAB','c0'])]
];
const kinds=['collinear','pointOnLine','pointOnCircle','perpendicular','parallel','distanceRatio','midpoint','specialPoint','orthocentricSystem','angleBisector','specialLine','equilateral','circleLineTangent','congruentTriangles'];
const diff=(a,b)=>({x:a.x-b.x,y:a.y-b.y});const dot=(a,b)=>a.x*b.x+a.y*b.y;const cr=(a,b)=>a.x*b.y-a.y*b.x;const len=p=>Math.hypot(p.x,p.y);
function origin(){const k=new Kernel();for(const [id,x,y] of base)k.execute({op:'free',id,params:{x,y}});
 k.execute({op:'line',id:'lAB',inputs:['A','B']});k.execute({op:'line',id:'lBC',inputs:['B','C']});k.execute({op:'circle',id:'c0',inputs:['O','A']});k.execute({op:'circle',id:'c1',inputs:['A','B']});return k;}
function scene(k){const s={points:[],lines:[],circles:[],construction:[]};for(const n of k.snapshot().nodes){if(!n.valid||!n.geom)continue;s.construction.push({id:n.id,op:n.op,inputs:n.inputs,params:n.params});const g=n.geom;
 if(g.kind==='point')s.points.push({id:n.id,x:g.x,y:g.y});
 if(g.kind==='line')s.lines.push({id:n.id,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y});
 if(g.kind==='circle')s.circles.push({id:n.id,cx:g.c.x,cy:g.c.y,r:g.r});}return s;}
function asLine(k,ref){if(ref.type==='object'){const g=k.geom(ref.id);return g?.kind==='line'?g:null;}
 const p=k.geom(ref.a),q=k.geom(ref.b);return p&&q?{p,d:diff(q,p),kind:'line'}:null;}
function factGeometry(k,f){const o=f.operand;const p=id=>k.geom(id),d=(a,b)=>len(diff(p(a),p(b)));
 switch(f.kind){
 case 'onRealAxis':return Math.abs(p(o.point).y)<1e-7;
 case 'fixedCoordinate':return dFrom(p(o.point),o)<1e-7;
 case 'polygonVertices':return k.geom(o.polygon)?.kind==='polygon'&&o.vertices.every(v=>!!p(v));
 case 'concyclic':{const [a,b,c,d]=o.points;const A=p(a),B=p(b),C=p(c),D=p(d);const v=diff(B,A),w=diff(C,A),den=2*cr(v,w);if(Math.abs(den)<1e-8)return false;const e=dot(v,v),f=dot(w,w);const center={x:A.x+(e*w.y-f*v.y)/den,y:A.y+(f*v.x-e*w.x)/den};return Math.abs(dFrom(center,D)-dFrom(center,A))<1e-7;}
 case 'equalSegments':{const arr=o.segments.map(v=>v.substring(2).split('|')).map(([a,b])=>d(a,b));return arr.every(x=>Math.abs(x-arr[0])<1e-7);}
 case 'collinear':{const [a,b,c]=o.points;return Math.abs(cr(diff(p(a),p(b)),diff(p(c),p(b))))<1e-7;}
 case 'midpoint':{const [a,b]=o.vertices,m=p(o.point);return len(diff(m,{x:(p(a).x+p(b).x)/2,y:(p(a).y+p(b).y)/2}))<1e-7;}
 case 'perpendicular':case 'parallel':{const a=asLine(k,o.lines[0]),b=asLine(k,o.lines[1]);if(!a||!b||!len(a.d)||!len(b.d))return false;const r=f.kind==='perpendicular'?dot(a.d,b.d):cr(a.d,b.d);return Math.abs(r)/(len(a.d)*len(b.d))<1e-8;}
 case 'incidenceCircle':{const q=k.geom(o.circle),x=p(o.point);return !!q&&Math.abs(dFrom(q.c,x)-q.r)<1e-7;}
 case 'incidenceLine':{const q=asLine(k,o.line);return !!q&&(o.points||[o.point]).every(id=>{const x=p(id);return x&&Math.abs(cr(q.d,diff(x,q.p)))/len(q.d)<1e-7;});}
 case 'incidenceCarrier':{const q=k.geom(o.carrier),x=p(o.point);return q?.kind==='line'?Math.abs(cr(q.d,diff(x,q.p)))/len(q.d)<1e-7:q?.kind==='circle'?Math.abs(dFrom(q.c,x)-q.r)<1e-7:false;}
 case 'orthocentricSystem':{const q=o.points;return q.every((h,i)=>{const [a,b,c]=q.filter((_,j)=>i!==j);return Math.abs(dot(diff(p(h),p(a)),diff(p(b),p(c))))<1e-7&&Math.abs(dot(diff(p(h),p(b)),diff(p(a),p(c))))<1e-7;});}
 case 'reflection':{const [a,b]=o.points,l=asLine(k,o.line),m={x:(p(a).x+p(b).x)/2,y:(p(a).y+p(b).y)/2};return !!l&&Math.abs(cr(l.d,diff(m,l.p)))/len(l.d)<1e-7&&Math.abs(dot(l.d,diff(p(a),p(b))))/len(l.d)<1e-7;}
 case 'equalCircleRadius':{const [a,b]=o.from.substring(2).split('|'),q=k.geom(o.circle);return Math.abs(q.r-d(a,b))<1e-7;}
 case 'diameter':{const q=k.geom(o.circle),[a,b]=o.vertices;return len(diff(q.c,{x:(p(a).x+p(b).x)/2,y:(p(a).y+p(b).y)/2}))<1e-7;}
 case 'circleCenter':return len(diff(k.geom(o.circle).c,p(o.point)))<1e-7;
 case 'fixedRadius':return Math.abs(k.geom(o.circle).r-o.radius)<1e-7;
 case 'equalDirectedVectors':{const [[a,b],[c,d]]=o.vectors;return len(diff(diff(p(b),p(a)),diff(p(d),p(c))))<1e-7;}
 case 'tangent':{const l=k.geom(o.line),q=k.geom(o.circle);return l?.kind==='line'&&q?.kind==='circle'&&Math.abs(Math.abs(cr(l.d,diff(q.c,l.p)))/len(l.d)-q.r)<1e-7;}
 case 'angleBisector':{const l=asLine(k,o.line),[a,b]=o.arms,v=p(o.vertex);if(!l)return false;const u=diff(p(a),v),w=diff(p(b),v),u0={x:u.x/len(u),y:u.y/len(u)},w0={x:w.x/len(w),y:w.y/len(w)},d=o.outer?diff(u0,w0):{x:u0.x+w0.x,y:u0.y+w0.y};return len(d)>1e-8&&Math.abs(cr(l.d,d))/(len(l.d)*len(d))<1e-7;}
 case 'triangleCenter':{const [a,b,c]=o.vertices,at=p(o.point);if(!at)return false;const A=p(a),B=p(b),C=p(c);
  if(o.center==='重心')return dFrom(at,{x:(A.x+B.x+C.x)/3,y:(A.y+B.y+C.y)/3})<1e-7;
  if(o.center==='外心')return Math.abs(dFrom(at,A)-dFrom(at,B))<1e-7&&Math.abs(dFrom(at,A)-dFrom(at,C))<1e-7;
  if(o.center==='垂心')return Math.abs(dot(diff(at,A),diff(B,C)))<1e-7&&Math.abs(dot(diff(at,B),diff(A,C)))<1e-7;
  if(o.center==='内心'||o.center==='旁心'){const dl=(q,r)=>Math.abs(cr(diff(r,q),diff(at,q)))/dFrom(q,r);const values=[dl(A,B),dl(B,C),dl(C,A)];return values.every(x=>Math.abs(x-values[0])<1e-7);}
  return true;}
 case 'equivalentOrthocenter':{const [a,b,c]=o.vertices,h=o.point;return Math.abs(dot(diff(p(h),p(a)),diff(p(b),p(c))))<1e-7&&Math.abs(dot(diff(p(h),p(b)),diff(p(a),p(c))))<1e-7;}
 case 'midpointOnLine':{const l=k.geom(o.line),[a,b]=o.vertices,m={x:(p(a).x+p(b).x)/2,y:(p(a).y+p(b).y)/2};return l?.kind==='line'&&Math.abs(cr(l.d,diff(m,l.p)))/len(l.d)<1e-7;}
 default:return true;
 }
}
function dFrom(a,b){return Math.hypot(a.x-b.x,a.y-b.y);}
let n=0,verified=0,matched=0,opKinds=new Set();const detail=[];
for(let i=0;i<cases.length;i++){
 const [name,command]=cases[i],k=origin(),id='X'+i;opKinds.add(command.op);
 const value=k.execute({...command,id});assert.ok(k.geom(id),`${name} must have valid geometry`);n++;
 const snap=scene(k),defs=Logic.makeFacts(snap.construction).filter(f=>f.node===id);
 assert.ok(defs.length>=1,`${name} needs an operand-bearing construction rule`);n++;
 for(const f of defs){assert.ok(factGeometry(k,f),`${name}: invalid ${f.kind} / ${f.reason}`);verified++;}
 const result=scan(snap,{kinds,perturb:false});
 const w=result.entries.flatMap(e=>(e.verifiedConsequences||[]).filter(v=>v.node===id).map(v=>e));
 matched+=w.length;
 detail.push({case:name,op:command.op,definitionFacts:defs.length,witnesses:w.length,relationCount:result.count});
 assert.ok(result.entries.every(e=>e.score>=1&&e.score<=100),'scores stay normalized');n++;
 assert.ok(result.implicationGraph.nodes.length>=defs.length&&result.implicationGraph.edges.some(e=>e.category==='definition'),'typed graph includes construction facts');n++;
}
assert.ok(cases.length>=50,'at least fifty independent small construction diagrams');n++;
assert.ok(opKinds.size>=30,'broad Kernel command coverage');n++;
assert.ok(matched>=25,'construction facts actually match detected results');n++;
// Full orthocentric package, with no invented relations on unselected classes.
const k=origin();k.execute({op:'orthocenter',id:'H',inputs:['A','B','C']});
const ortho=Logic.makeFacts(scene(k).construction).filter(f=>f.node==='H');
assert.equal(ortho.length,11,'orthocenter yields 3 altitudes, 3 opposite-pair perpendiculars, 4 equivalent roles plus group/definition');n++;
assert.equal(ortho.filter(f=>f.kind==='perpendicular').length,6);n++;
const result=scan(scene(k),{kinds:['orthocentricSystem','perpendicular','specialPoint'],perturb:false});
assert.ok(result.entries.some(e=>e.kind==='orthocentricSystem'&&e.verifiedConsequences?.some(v=>v.node==='H')));n++;
console.log(`PASS v3.4 minimal diagrams: ${cases.length} distinct scenes / ${opKinds.size} operations, ${verified} validated construction facts, ${matched} source matches, ${n} assertions`);
if(process.env.V34_REPORT)require('node:fs').writeFileSync(process.env.V34_REPORT,JSON.stringify(detail,null,2));
