/* GeomTool Kernel Protocol 1.0 — browser & Node compatible, no dependencies.
 * The graph is the source of truth. Every construction is a command referencing stable IDs.
 * Derived coordinates are cached only, and are recomputed in creation order.
 */
(function(root,factory){ const api=factory(); root.GeomKernel=api; if(typeof module==='object'&&module.exports)module.exports=api; })(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const EPS=1e-10, maxN=2000;
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const xy=(x,y)=>({x,y});
const add=(a,b)=>xy(a.x+b.x,a.y+b.y),sub=(a,b)=>xy(a.x-b.x,a.y-b.y),mul=(a,k)=>xy(a.x*k,a.y*k);
const dot=(a,b)=>a.x*b.x+a.y*b.y,cross=(a,b)=>a.x*b.y-a.y*b.x;
const norm=a=>Math.hypot(a.x,a.y),dist=(a,b)=>norm(sub(a,b));
const unit=a=>norm(a)>EPS?mul(a,1/norm(a)):null;
const rot90=a=>xy(-a.y,a.x),isPoint=g=>g&&g.kind==='point';
const midpoint=(a,b)=>mul(add(a,b),.5);
const clean=p=>p&&finite(p.x)&&finite(p.y)?p:null;
const schema='geomtool.ir/v1';
// V4 reserves the five canonical Cartesian reference identifiers and the unit circle.
const RESERVED=new Set(['l_Re','l_Im','p_0','p_1','p_i','c_unit']);
const LEGACY_REFERENCES={GTREF_origin:'p_0',GTREF_one:'p_1',GTREF_i:'p_i',GTREF_real:'l_Re',GTREF_imag:'l_Im',GTREF_unit:'c_unit'};
const customEvaluators=Object.create(null);
const METHODS={
 referencePoint:{kind:'point',inputs:[],group:'reference',label:'固定参考点',syntax:'(系统内置)',rational:'rational'},
 free:{kind:'point',inputs:[],group:'point',label:'自由点',syntax:'point A x y',rational:'independent'},
 real:{kind:'point',inputs:[],group:'point',label:'实轴点',syntax:'real R x',rational:'independent'},
 onLine:{kind:'point',inputs:['line'],group:'point',label:'线上点',syntax:'online P l t',rational:'rational'},
 circleCenter:{kind:'point',inputs:['circle'],group:'point',label:'圆心',syntax:'circlecenter O c',rational:'rational'},
 onCircle:{kind:'point',inputs:['circle'],group:'point',label:'圆上点',syntax:'oncircle P c angle',rational:'algebraic'},
 line:{kind:'line',inputs:['point','point'],group:'line',label:'直线',syntax:'line l A B',rational:'rational'},
 segment:{kind:'line',inputs:['point','point'],group:'line',label:'线段',syntax:'segment s A B',rational:'rational'},
 ray:{kind:'line',inputs:['point','point'],group:'line',label:'射线',syntax:'ray r A B',rational:'rational'},
 circle:{kind:'circle',inputs:['point','point'],group:'circle',label:'圆心及过点',syntax:'circle c O A',rational:'rational'},
 circleRadius:{kind:'circle',inputs:['point'],group:'circle',label:'定半径圆',syntax:'circler c O radius',rational:'rational'},
 complexSquare:{kind:'point',inputs:['point'],group:'point',label:'复数平方点',syntax:'complexsquare A P',rational:'rational'},
 complexNegativeProduct:{kind:'point',inputs:['point','point'],group:'point',label:'单位圆弧中点',syntax:'complexnegativeproduct D Baux Caux',rational:'rational'},
 similarPoint:{kind:'point',inputs:['point','point','point','point','point'],group:'point',label:'相似点',syntax:'similarpoint F A B C D E',rational:'algebraic'},
 equalAngleLine:{kind:'line',inputs:['point','point','point','point','point'],group:'line',label:'等角线',syntax:'equalangleline l A B C D E',rational:'algebraic'},
 midpoint:{kind:'point',inputs:['point','point'],group:'point',label:'中点',syntax:'midpoint M A B',rational:'rational'},
 intersection:{kind:'point',inputs:['carrier','carrier'],group:'point',label:'交点',syntax:'intersect X l m [0|1]',rational:'algebraic'},
 perpendicular:{kind:'line',inputs:['point','line'],group:'line',label:'过点垂线',syntax:'perpendicular h P l',rational:'rational'},
 parallel:{kind:'line',inputs:['point','line'],group:'line',label:'过点平行线',syntax:'parallel h P l',rational:'rational'},
 perpBisector:{kind:'line',inputs:['point','point'],group:'line',label:'中垂线',syntax:'perpbisector h A B',rational:'rational'},
 angleBisector:{kind:'line',inputs:['point','point','point'],group:'line',label:'角平分线',syntax:'bisector h A B C',rational:'algebraic'},
 foot:{kind:'point',inputs:['point','line'],group:'point',label:'垂足',syntax:'foot H P l',rational:'rational'},
 reflectPoint:{kind:'point',inputs:['point','point'],group:'point',label:'中心对称点',syntax:'reflectpoint P2 P O',rational:'rational'},
 reflectLine:{kind:'point',inputs:['point','line'],group:'point',label:'轴对称点',syntax:'reflectline P2 P l',rational:'rational'},
 translate:{kind:'point',inputs:['point','point','point'],group:'point',label:'平移点',syntax:'translate P2 P A B',rational:'rational'},
 rotate:{kind:'point',inputs:['point','point'],group:'point',label:'旋转点',syntax:'rotate P2 P O degrees',rational:'algebraic'},
 homothety:{kind:'point',inputs:['point','point'],group:'point',label:'位似点',syntax:'homothety P2 P O factor',rational:'rational'},
 centroid:{kind:'point',inputs:['point','point','point'],group:'point',label:'重心',syntax:'centroid G A B C',rational:'rational'},
 circumcenter:{kind:'point',inputs:['point','point','point'],group:'point',label:'外心',syntax:'circumcenter O A B C',rational:'rational'},
 incenter:{kind:'point',inputs:['point','point','point'],group:'point',label:'内心',syntax:'incenter I A B C',rational:'algebraic'},
 orthocenter:{kind:'point',inputs:['point','point','point'],group:'point',label:'垂心',syntax:'orthocenter H A B C',rational:'rational'},
 circumcircle:{kind:'circle',inputs:['point','point','point'],group:'circle',label:'三点圆',syntax:'circumcircle c A B C',rational:'rational'},
 tangent:{kind:'line',inputs:['point','circle'],group:'line',label:'切线 (圆上点)',syntax:'tangent t P c',rational:'rational'},
 tangentFrom:{kind:'line',inputs:['point','circle'],group:'line',label:'外点切线',syntax:'tangentfrom t P c [0|1]',rational:'algebraic'},
 parallelogram:{kind:'point',inputs:['point','point','point'],group:'point',label:'平行四边形第四点',syntax:'parallelogram D A B C',rational:'rational'},
 excenter:{kind:'point',inputs:['point','point','point'],group:'point',label:'A 旁心',syntax:'excenter J A B C',rational:'algebraic'},
 outerAngleBisector:{kind:'line',inputs:['point','point','point'],group:'line',label:'外角平分线',syntax:'outerbisector h A B C',rational:'algebraic'},
 diameterCircle:{kind:'circle',inputs:['point','point'],group:'circle',label:'直径圆',syntax:'diametercircle c A B',rational:'rational'},
 circleBySegment:{kind:'circle',inputs:['point','point','point'],group:'circle',label:'转移线段为半径',syntax:'circlebysegment c O A B',rational:'rational'},
 equilateral:{kind:'point',inputs:['point','point'],group:'point',label:'等边三角形顶点',syntax:'equilateral C A B [0|1]',rational:'algebraic'},
 rotateRight:{kind:'point',inputs:['point','point'],group:'point',label:'顺时针旋转 90°',syntax:'rotateright P2 P O',rational:'rational'},
 rotateLeft:{kind:'point',inputs:['point','point'],group:'point',label:'逆时针旋转 90°',syntax:'rotateleft P2 P O',rational:'rational'},
 foot3:{kind:'point',inputs:['point','point','point'],group:'point',label:'三点垂足',syntax:'foot3 H P A B',rational:'rational'},
 parallel3:{kind:'line',inputs:['point','point','point'],group:'line',label:'三点平行线',syntax:'parallel3 h P A B',rational:'rational'},
 perpendicular3:{kind:'line',inputs:['point','point','point'],group:'line',label:'三点垂线',syntax:'perpendicular3 h P A B',rational:'rational'},
 onCircumcircle:{kind:'point',inputs:['point','point','point'],group:'point',label:'三点圆上点',syntax:'oncircum P A B C angle',rational:'algebraic'},
 secondIntersection:{kind:'point',inputs:['point','carrier','carrier'],group:'point',label:'已知交点求另一个',syntax:'secondintersect Q P l c',rational:'algebraic'},
 polygon:{kind:'polygon',inputs:['point','point','point'],variadic:true,group:'polygon',label:'多边形',syntax:'polygon poly A B C [D...]',rational:'rational'},
};
function inputKind(g){return g?.kind==='circle'?'circle':g?.kind==='line'?'line':g?.kind==='point'?'point':null;}
function lgeom(a,d,extent='line'){if(!a||!d||norm(d)<EPS)return null;return {kind:'line',p:a,d,extent};}
function project(p,l){let d=l.d;return add(l.p,mul(d,dot(sub(p,l.p),d)/dot(d,d)));}
function circle3(a,b,c){const ab=sub(b,a),ac=sub(c,a),d=2*cross(ab,ac);if(Math.abs(d)<EPS*Math.max(1,norm(ab)*norm(ac)))return null;const q=dot(ab,ab),r=dot(ac,ac);const center=add(a,xy((q*ac.y-r*ab.y)/d,(r*ab.x-q*ac.x)/d));return {kind:'circle',c:center,r:dist(center,a)};}
function allowedLine(l,t){return l.extent==='segment'?t>=-EPS&&t<=1+EPS:l.extent==='ray'?t>=-EPS:true;}
// A double root is returned twice only when a construction explicitly needs
// two stable algebraic branches at an accidental (not structural) tangency.
// The tolerance here is floating-roundoff-sized, not geometric EPS: real
// negative radicands must never silently turn into spurious real roots.
function intersectionBranches(g,h,duplicateTangent=false){
 if(!g||!h||g.kind==='point'||h.kind==='point')return [];
 let list=[];
 if(g.kind==='line'&&h.kind==='line'){
  const det=cross(g.d,h.d);if(Math.abs(det)<EPS*norm(g.d)*norm(h.d))return [];
  const v=sub(h.p,g.p),t=cross(v,h.d)/det,u=cross(v,g.d)/det;
  if(allowedLine(g,t)&&allowedLine(h,u))list=[add(g.p,mul(g.d,t))];
 }else if(g.kind==='line'||h.kind==='line'){
  const l=g.kind==='line'?g:h,c=g.kind==='circle'?g:h;
  const aa=dot(l.d,l.d);if(!(aa>0)||!(c.r>0))return [];
  const f=sub(l.p,c.c),t0=-dot(f,l.d)/aa;
  const nearest=add(l.p,mul(l.d,t0));
  const d2=dot(sub(nearest,c.c),sub(nearest,c.c));
  let rad=c.r*c.r-d2;
  const tolerance=32*Number.EPSILON*Math.max(c.r*c.r,d2,1);
  if(rad< -tolerance)return [];
  const tangent=Math.abs(rad)<=tolerance;
  if(tangent){list=duplicateTangent?[nearest,nearest]:[nearest];}
  else {const dt=Math.sqrt(rad/aa);list=[add(l.p,mul(l.d,t0-dt)),add(l.p,mul(l.d,t0+dt))];}
  list=list.map((p,i)=>allowedLine(l,tangent?t0:t0+(i===0?-1:1)*Math.sqrt(Math.max(0,rad)/aa))?p:null);
 }else{
  const d=dist(g.c,h.c),r=g.r,s=h.r;
  if(!(r>0&&s>0)||d<=EPS*Math.max(1,r,s))return [];
  const a=(r*r-s*s+d*d)/(2*d);
  // Heron's form reduces cancellation for near-tangencies; unlike a large
  // absolute EPS, this tolerance will not accept a genuinely distant root.
  const h2=r*r-a*a;
  const tol=32*Number.EPSILON*Math.max(1,r*r,a*a,s*s);
  if(h2< -tol)return [];
  const u=mul(sub(h.c,g.c),1/d),base=add(g.c,mul(u,a));
  if(Math.abs(h2)<=tol)list=duplicateTangent?[base,base]:[base];
  else {const perp=mul(rot90(u),Math.sqrt(h2));list=[sub(base,perp),add(base,perp)];}
  // A negative separation test is not replaced by sqrt(0), unless the
  // discrepancy is on the order of roundoff in the circle centers/radii.
 }
 // Preserve the carrier-frame branch order during dragging.
 return list.map(q=>clean(q));
}
// Compact legacy/query view; construction nodes use the non-compacting slots.
// In particular a ray meeting only the FAR root receives branch=1, not 0.
function intersections(g,h){return intersectionBranches(g,h).filter(Boolean);}
function calc(node,of){const a=node.inputs.map(id=>of(id));if(a.some(x=>!x))return null;const p=node.params||{}, get=i=>a[i];let q;
 switch(node.op){
 case 'circleCenter':q=get(0);return q.kind==='circle'?{kind:'point',x:q.c.x,y:q.c.y}:null;
 case 'referencePoint':return clean({x:p.x,y:p.y})?{kind:'point',x:p.x,y:p.y}:null;
 case 'free':return clean({kind:'point',x:p.x,y:p.y})?{kind:'point',x:p.x,y:p.y}:null;
 case 'real':return finite(p.x)?{kind:'point',x:p.x,y:0}:null;
 case 'onLine':q=get(0);return finite(p.t)&&allowedLine(q,p.t)?{kind:'point',...add(q.p,mul(q.d,p.t))}:null;
 case 'onCircle':q=get(0);return finite(p.theta)?{kind:'point',x:q.c.x+q.r*Math.cos(p.theta),y:q.c.y+q.r*Math.sin(p.theta)}:null;
 case 'onCircumcircle':q=circle3(...a);return q&&finite(p.theta)?{kind:'point',x:q.c.x+q.r*Math.cos(p.theta),y:q.c.y+q.r*Math.sin(p.theta)}:null;
 case 'line':case 'segment':case 'ray':return lgeom(get(0),sub(get(1),get(0)),node.op);
 case 'circle':{const r=dist(get(0),get(1));return r>EPS?{kind:'circle',c:get(0),r}:null;}
 case 'circleRadius':return finite(p.radius)&&p.radius>EPS?{kind:'circle',c:get(0),r:p.radius}:null;
 case 'circleBySegment':q=dist(get(1),get(2));return q>EPS?{kind:'circle',c:get(0),r:q}:null;
 case 'diameterCircle':q=midpoint(get(0),get(1));return dist(get(0),get(1))>EPS?{kind:'circle',c:q,r:dist(get(0),get(1))/2}:null;
 case 'midpoint':return {kind:'point',...midpoint(get(0),get(1))};
 case 'intersection':{const x=intersectionBranches(get(0),get(1),!!p.allowTangentDuplicate);return x[p.branch===1?1:0]?{kind:'point',...x[p.branch===1?1:0]}:null;}
 case 'secondIntersection':{const xs=intersections(get(1),get(2));const other=xs.find(v=>dist(v,get(0))>1e-7*Math.max(1,norm(v)));return xs.some(v=>dist(v,get(0))<=1e-7*Math.max(1,norm(v)))&&other?{kind:'point',...other}:null;}
 case 'complexSquare':{const z=get(0);return {kind:'point',x:z.x*z.x-z.y*z.y,y:2*z.x*z.y};}
 case 'complexNegativeProduct':{const z=get(0),w=get(1);return {kind:'point',x:-(z.x*w.x-z.y*w.y),y:-(z.x*w.y+z.y*w.x)};}
 case 'similarPoint':{
   // Oriented similarity ABC -> DEF, complex multiplication by (E-D)/(B-A).
   // A=B and D!=E cannot be related by a similarity. Do not create false data.
   const ab=sub(get(1),get(0)),de=sub(get(4),get(3)),ac=sub(get(2),get(0));
   const den=dot(ab,ab);
   if(den<EPS*EPS){if(norm(de)>EPS)return null;
     // Nonunique when both reference segments collapse: choose an explicit
     // canonical representative. The proof compiler does NOT assert similarity.
     return {kind:'point',...add(get(3),ac)};
   }
   const alpha=dot(ab,de)/den,beta=cross(ab,de)/den;
   return {kind:'point',...add(get(3),xy(alpha*ac.x-beta*ac.y,beta*ac.x+alpha*ac.y))};
 }
 case 'equalAngleLine':{
   // ∠ABC is oriented BA -> BC; ∠DEQ is oriented ED -> EQ.
   const ba=sub(get(0),get(1)),bc=sub(get(2),get(1)),ed=sub(get(3),get(4));
   if(norm(ba)<EPS||norm(bc)<EPS||norm(ed)<EPS)return null;
   const t=Math.atan2(cross(ba,bc),dot(ba,bc));
   return lgeom(get(4),xy(ed.x*Math.cos(t)-ed.y*Math.sin(t),ed.x*Math.sin(t)+ed.y*Math.cos(t)));
 }
 case 'parallel':return lgeom(get(0),get(1).d);
 case 'parallel3':return lgeom(get(0),sub(get(2),get(1)));
 case 'perpendicular3':return lgeom(get(0),rot90(sub(get(2),get(1))));
 case 'foot3':{q=lgeom(get(1),sub(get(2),get(1)));return q?{kind:'point',...project(get(0),q)}:null;}
 case 'perpendicular':return lgeom(get(0),rot90(get(1).d));
 case 'perpBisector':return lgeom(midpoint(get(0),get(1)),rot90(sub(get(1),get(0))));
 case 'angleBisector':{const u=unit(sub(get(0),get(1))),v=unit(sub(get(2),get(1)));return u&&v?lgeom(get(1),add(u,v)):null;}
 case 'outerAngleBisector':{const u=unit(sub(get(0),get(1))),v=unit(sub(get(2),get(1)));return u&&v?lgeom(get(1),sub(u,v)):null;}
 case 'foot':return {kind:'point',...project(get(0),get(1))};
 case 'reflectPoint':return {kind:'point',...sub(mul(get(1),2),get(0))};
 case 'reflectLine':{const h=project(get(0),get(1));return {kind:'point',...sub(mul(h,2),get(0))};}
 case 'translate':return {kind:'point',...add(get(0),sub(get(2),get(1)))};
 case 'parallelogram':return {kind:'point',...add(get(1),sub(get(2),get(0)))};
 case 'rotateRight':return {kind:'point',...sub(get(1),rot90(sub(get(0),get(1))))};
 case 'rotateLeft':return {kind:'point',...add(get(1),rot90(sub(get(0),get(1))))};
 case 'equilateral':{const v=sub(get(1),get(0));return norm(v)>EPS?{kind:'point',...add(midpoint(get(0),get(1)),mul(rot90(v),(p.branch===1?-1:1)*Math.sqrt(3)/2))}:null;}
 case 'rotate':{const v=sub(get(0),get(1)),t=Number(p.degrees)*Math.PI/180;if(!finite(t))return null;return {kind:'point',...add(get(1),xy(v.x*Math.cos(t)-v.y*Math.sin(t),v.x*Math.sin(t)+v.y*Math.cos(t)))};}
 case 'homothety':return finite(p.factor)?{kind:'point',...add(get(1),mul(sub(get(0),get(1)),p.factor))}:null;
 case 'centroid':return {kind:'point',...mul(add(add(get(0),get(1)),get(2)),1/3)};
 case 'circumcenter':return (q=circle3(...a))?{kind:'point',...q.c}:null;
 case 'circumcircle':return circle3(...a);
 case 'orthocenter':{q=circle3(...a);return q?{kind:'point',...sub(add(add(get(0),get(1)),get(2)),mul(q.c,2))}:null;}
 case 'excenter':{const aa=dist(get(1),get(2)),bb=dist(get(0),get(2)),cc=dist(get(0),get(1)),sum=-aa+bb+cc;if(sum<EPS||Math.abs(cross(sub(get(1),get(0)),sub(get(2),get(0))))<EPS)return null;return {kind:'point',...mul(add(add(mul(get(0),-aa),mul(get(1),bb)),mul(get(2),cc)),1/sum)};}
 case 'incenter':{const aa=dist(get(1),get(2)),bb=dist(get(0),get(2)),cc=dist(get(0),get(1)),sum=aa+bb+cc;if(sum<EPS||Math.abs(cross(sub(get(1),get(0)),sub(get(2),get(0))))<EPS)return null;return {kind:'point',...mul(add(add(mul(get(0),aa),mul(get(1),bb)),mul(get(2),cc)),1/sum)};}
 case 'tangent':{q=get(1);const v=sub(get(0),q.c);return Math.abs(norm(v)-q.r)<=1e-7*Math.max(1,q.r)?lgeom(get(0),rot90(v)):null;}
 case 'tangentFrom':{
  const c=get(1),v=sub(get(0),c.c),d2=dot(v,v),r2=c.r*c.r;
  const radial=d2-r2,roundoff=32*Number.EPSILON*Math.max(1,d2,r2);
  // Never take sqrt of a genuinely negative radicand: a point inside
  // the circle does NOT admit a real tangent, even during dragging.
  if(radial < -roundoff||d2<=EPS*EPS)return null;
  if(Math.abs(radial)<=roundoff)return lgeom(get(0),rot90(v));
  const k=r2/d2,h=Math.sqrt(radial)*c.r/d2,sign=p.branch===1?1:-1;
  const touch=add(c.c,add(mul(v,k),mul(rot90(v),sign*h)));
  return lgeom(get(0),sub(touch,get(0)));
 }
 case 'polygon':return Math.abs(a.reduce((s,p,i)=>s+cross(p,a[(i+1)%a.length]),0))>EPS?{kind:'polygon',points:a}:null;
 default:return customEvaluators[node.op]?customEvaluators[node.op](a,{...p},{xy,add,sub,mul,dot,cross,norm,dist,unit,rot90,project,intersections,circle3}):null;
 }
}
const clone=o=>JSON.parse(JSON.stringify(o));
class Kernel{
 constructor(initial){this.nodes=[];this.map=new Map();this.undoStack=[];this.redoStack=[];this.listeners=new Set();this.revision=0;this.counts={point:0,line:0,circle:0,polygon:0};if(initial)this.load(initial,false);}
 subscribe(fn){this.listeners.add(fn);return ()=>this.listeners.delete(fn);}
 _emit(type){this.revision++;for(const fn of this.listeners)try{fn({type,revision:this.revision})}catch(e){console.error(e)}}
 _take(){return this.serialize();}
 _commit(old,type){this.undoStack.push(old);if(this.undoStack.length>120)this.undoStack.shift();this.redoStack=[];this._emit(type);}
 _index(){this.map=new Map(this.nodes.map(n=>[n.id,n]));this.counts={point:0,line:0,circle:0,polygon:0};for(const n of this.nodes)this.counts[METHODS[n.op].kind]++;this.rebuild();}
 rebuild(){const g=new Map();for(const n of this.nodes){try{n.geom=calc(n,id=>g.get(id)||null);n.valid=!!n.geom;if(n.geom?.kind==='point'&&(!finite(n.geom.x)||!finite(n.geom.y)))n.valid=false;}catch(e){n.valid=false;n.geom=null;}g.set(n.id,n.valid?n.geom:null);}return this;}
 // Moving an independent point only invalidates its descendants. The graph is
 // topologically ordered, so geometry elsewhere remains bit-for-bit unchanged.
 rebuildFrom(id){const affected=new Set([id]);for(const n of this.nodes){if(!affected.has(n.id)&&!n.inputs.some(parent=>affected.has(parent)))continue;affected.add(n.id);try{n.geom=calc(n,key=>{const parent=this.map.get(key);return parent?.valid?parent.geom:null;});n.valid=!!n.geom;if(n.geom?.kind==='point'&&(!finite(n.geom.x)||!finite(n.geom.y)))n.valid=false;}catch(e){n.valid=false;n.geom=null;}}return this;}
 node(ref){return this.map.get(ref)||this.nodes.find(n=>n.label===ref)||null;}
 geom(ref){const n=this.node(ref);return n?.valid?n.geom:null;}
 list(){return this.nodes.slice();}
 // The ID is the ONLY public name. All input references use that same ID.
 usedName(id){return RESERVED.has(id)||this.nodes.some(n=>n.id===id);}
 nextId(kind){
  const letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  if(kind==='point'){
   for(const id of letters)if(!this.usedName(id))return id;
   for(let i=1;;i++)if(!this.usedName('P'+i))return 'P'+i;
  }
  const prefix=kind==='line'?'l':kind==='circle'?'c':'poly';
  for(let i=1;;i++)if(!this.usedName(prefix+i))return prefix+i;
 }
 edgeName(a,b){
  const proposed='s'+a+b;
  if(/^[A-Za-z][A-Za-z0-9_]*$/.test(proposed)&&!this.usedName(proposed))return proposed;
  for(let i=1;;i++)if(!this.usedName('s'+i))return 's'+i;
 }
 createPolygonEdges(poly){
  const result=[];
  for(let i=0;i<poly.inputs.length;i++){
   const a=poly.inputs[i],b=poly.inputs[(i+1)%poly.inputs.length];
   const existing=this.nodes.find(n=>n.op==='segment'&&n.params?.polygonParent===poly.id&&n.inputs[0]===a&&n.inputs[1]===b);
   if(existing){result.push(existing);continue;}
   result.push(this._create({op:'segment',id:this.edgeName(a,b),inputs:[a,b],params:{polygonParent:poly.id,edgeAutoName:true}},true));
  }
  return result;
 }
 _create(cmd,allowInvalid=false){const spec=METHODS[cmd.op];if(!spec)throw Error('未知构造：'+cmd.op);
  let inputs=cmd.inputs||[];if(!Array.isArray(inputs)||inputs.length<(spec.variadic?3:spec.inputs.length)||(!spec.variadic&&inputs.length!==spec.inputs.length))throw Error('参数数量不正确：'+spec.syntax);
  if(this.nodes.length>=maxN)throw Error('对象上限：'+maxN);
  inputs.forEach((name,i)=>{const n=this.node(name);if(!n||(!allowInvalid&&!n.valid))throw Error('找不到有效对象：'+name);const required=spec.inputs[Math.min(i,spec.inputs.length-1)],actual=METHODS[n.op].kind;
   if(required!=='carrier'&&required!==actual||required==='carrier'&&!['line','circle'].includes(actual))throw Error('对象类型不匹配：'+name+' (需要 '+required+')');});
  // Signed intersection branches are permanently tied to ordered carriers.
  // v2 documents used a dense list after filtering by ray/segment bounds;
  // migrate the old root by its actual geometric coordinates when loading.
  let normalizedParams=cmd.params||{};
  if(cmd.op==='intersection'){
   const old=inputs.slice(),priority=id=>METHODS[this.node(id).op].kind==='line'?0:1;
   inputs=inputs.slice().sort((a,b)=>priority(a)-priority(b)||this.node(a).id.localeCompare(this.node(b).id));
   let newBranch=normalizedParams.branch===1?1:0;
   if(cmd.__legacy&&!normalizedParams.slotVersion){
    const target=intersections(this.geom(old[0]),this.geom(old[1]))[newBranch];
    const candidates=intersectionBranches(this.geom(inputs[0]),this.geom(inputs[1]));
    if(target){const distances=candidates.map(g=>g?dist(g,target):Infinity);newBranch=distances[1]<distances[0]?1:0;}
    else if(old[0]!==inputs[0]&&priority(old[0])===1&&priority(old[1])===1)newBranch=1-newBranch;
   }else if(old[0]!==inputs[0]&&priority(old[0])===1&&priority(old[1])===1&&
    !normalizedParams.canonicalInputs)newBranch=1-newBranch;
   normalizedParams={...normalizedParams,branch:newBranch,slotVersion:3};
  }
  const system=!!(cmd.system||cmd.style?.system);
  let id=cmd.id==null?this.nextId(spec.kind):String(cmd.id);
  if(!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(id)){
   if(cmd.__legacy)throw Error('旧文件对象名非法：'+id);
   id=this.nextId(spec.kind);
  }
  if(this.usedName(id)&&!(system&&RESERVED.has(id)&&!this.map.has(id))){
   if(cmd.__legacy)throw Error('导入对象名称重复或保留：'+id);
   // The desired point name may already be occupied: always create with the
   // smallest available A..Z, P1..Pn, rather than dropping the click.
   id=this.nextId(spec.kind);
  }
  if(RESERVED.has(id)&&!system)throw Error('系统保留名称不可使用：'+id);
  const params=clone(normalizedParams);if(Object.values(params).some(x=>typeof x==='number'&&!finite(x)))throw Error('参数必须是有限数字');
  const mustNumber={referencePoint:['x','y'],free:['x','y'],real:['x'],onLine:['t'],onCircle:['theta'],onCircumcircle:['theta'],circleRadius:['radius'],rotate:['degrees'],homothety:['factor']};
  if((mustNumber[cmd.op]||[]).some(key=>!finite(params[key])))throw Error('缺少必要数值参数');
  if(['intersection','tangentFrom','equilateral'].includes(cmd.op)&&params.branch!==undefined&&![0,1].includes(params.branch))throw Error('交点分支只能是 0 或 1');
  const resolved=inputs.map(n=>this.node(n).id);
  const node={id,label:id,op:cmd.op,inputs:resolved,params,style:{visible:true,dashed:false,...(cmd.style||{})}};
  // Geometric degeneracy is an error at creation; dragging may invalidate a derived node later.
  const g=inputs.every(x=>!!this.geom(x))?calc(node,name=>this.geom(name)):null;if(!g&&!allowInvalid)throw Error('构造退化或无有效解');
  this.nodes.push(node);this.map.set(id,node);this.rebuild();if(cmd.op==='polygon'&&!cmd.__deferEdges)this.createPolygonEdges(node);return node;
 }
 execute(cmd){if(!cmd||typeof cmd!=='object')throw Error('无效命令');const op=cmd.op;if(op==='undo')return this.undo();if(op==='redo')return this.redo();if(op==='export')return this.serialize();if(op==='import')return this.load(cmd.data);
  const before=this._take();let value;
  if(op==='clear'){this.nodes=[];this._index();value=true;}
  else if(op==='delete'){
   const n=this.node(cmd.id);if(!n)throw Error('对象不存在：'+cmd.id);if(n.style.system)throw Error('系统参考对象不能删除，可在设置中隐藏');const deleted=new Set([n.id]);let dirty=true;while(dirty){dirty=false;for(const x of this.nodes)if(!deleted.has(x.id)&&(x.inputs.some(id=>deleted.has(id))||deleted.has(x.params?.polygonParent))){deleted.add(x.id);dirty=true;}}
   this.nodes=this.nodes.filter(x=>!deleted.has(x.id));this._index();value=[...deleted];
  }else if(op==='move'){
   const n=this.node(cmd.id);if(!n||n.style.system||!['free','real','onLine','onCircle'].includes(n.op))throw Error('只能移动自由点或受约束的活动点');
   if(!finite(cmd.x)||!finite(cmd.y))throw Error('坐标不是有限数字');
   if(n.op==='free'){n.params={x:cmd.x,y:cmd.y}}else if(n.op==='real'){n.params.x=cmd.x}
   else if(n.op==='onLine'){const g=this.geom(n.inputs[0]);let t=dot(sub(cmd,g.p),g.d)/dot(g.d,g.d);if(g.extent==='segment')t=Math.max(0,Math.min(1,t));if(g.extent==='ray')t=Math.max(0,t);n.params.t=t;}
   else{const g=this.geom(n.inputs[0]);n.params.theta=Math.atan2(cmd.y-g.c.y,cmd.x-g.c.x);}
   this.rebuild();value=n;
  }else if(['rename','style'].includes(op)){
   const n=this.node(cmd.id);if(!n)throw Error('对象不存在：'+cmd.id);
   if(op==='rename'){const label=String(cmd.label||'').trim();if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(label)||label.length>32||RESERVED.has(label)||n.style.system||this.nodes.some(x=>x!==n&&x.id===label))throw Error('名称无效、保留或重复');const old=n.id;n.id=label;n.label=label;
    if(n.params?.edgeAutoName)n.params.edgeAutoName=false;
    for(const x of this.nodes){x.inputs=x.inputs.map(id=>id===old?label:id);if(x.params?.polygonParent===old)x.params.polygonParent=label;}
    // Derived polygon sides track their defining vertex names. This is an
    // atomic graph-wide rename: existing references to each side are remapped.
    for(const side of this.nodes.filter(x=>x.op==='segment'&&x.params?.edgeAutoName)){
     const wanted='s'+side.inputs.join('');
     if(side.id===wanted||!/^[A-Za-z][A-Za-z0-9_]*$/.test(wanted)||this.usedName(wanted))continue;
     const beforeSide=side.id;side.id=wanted;side.label=wanted;
     for(const child of this.nodes)child.inputs=child.inputs.map(id=>id===beforeSide?wanted:id);
    }
    this._index();}
   else{n.style={...n.style,...(cmd.style||{})};}value=n;
  }else value=this._create(cmd);
  if(JSON.stringify(before.nodes)!==JSON.stringify(this.serialize().nodes))this._commit(before,op);
  return value;
 }
 // Drag emits cheap previews without pushing hundreds of undo frames.
 previewMove(id,x,y){const n=this.node(id);if(!n||n.style.system||!['free','real','onLine','onCircle'].includes(n.op)||!finite(x)||!finite(y))return false;
  if(n.op==='free'){n.params.x=x;n.params.y=y;}else if(n.op==='real'){n.params.x=x;}else if(n.op==='onLine'){const g=this.geom(n.inputs[0]);if(!g)return false;let t=dot(sub({x,y},g.p),g.d)/dot(g.d,g.d);if(g.extent==='segment')t=Math.max(0,Math.min(1,t));if(g.extent==='ray')t=Math.max(0,t);n.params.t=t;}else{const g=this.geom(n.inputs[0]);if(!g)return false;n.params.theta=Math.atan2(y-g.c.y,x-g.c.x);}this.rebuildFrom(id);this._emit('preview');return true;
 }
 checkpoint(before,type='move'){if(before&&JSON.stringify(before.nodes)!==JSON.stringify(this.serialize().nodes))this._commit(before,type);}
 undo(){if(!this.undoStack.length)return false;const snap=this.undoStack.pop();this.redoStack.push(this._take());this._loadRaw(snap);this._emit('undo');return true;}
 redo(){if(!this.redoStack.length)return false;const snap=this.redoStack.pop();this.undoStack.push(this._take());this._loadRaw(snap);this._emit('redo');return true;}
 serialize(){return {schema,protocol:'geomtool-kernel/1.0',nodes:this.nodes.map(n=>({id:n.id,label:n.label,op:n.op,inputs:[...n.inputs],params:clone(n.params),style:clone(n.style)}))};}
 _loadRaw(data){this.nodes=clone(data.nodes);this._index();}
 load(data,record=true){if(typeof data==='string')data=JSON.parse(data);if(!data||data.schema!==schema||!Array.isArray(data.nodes)||data.nodes.length>maxN)throw Error('不支持的图形文件或对象过多');
  const staging=new Kernel();
  // Convert v3.5's separate display labels and GTREF IDs to a single namespace.
  const mapping=new Map(),occupied=new Set(RESERVED);
  const incoming=data.nodes;
  for(const source of incoming){if(!source||!METHODS[source.op])throw Error('导入包含未知构造');}
  for(const source of incoming){
   const sys=!!source.style?.system;
   let wanted=LEGACY_REFERENCES[source.id]||(sys&&RESERVED.has(source.id)?source.id:(source.label||source.id));
   if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(wanted)||wanted.length>32||(!sys&&RESERVED.has(wanted))||occupied.has(wanted)&&!(sys&&RESERVED.has(wanted)&&!mapping.has(source.id))){
    // Legacy aliases may collide; allocate deterministically, without losing objects.
    const kind=METHODS[source.op].kind,prefix=kind==='point'?'P':kind==='line'?'l':kind==='circle'?'c':'poly';
    if(kind==='point')wanted=[...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'].find(x=>!occupied.has(x))||null;
    if(!wanted||occupied.has(wanted)){for(let i=1;;i++)if(!occupied.has(prefix+i)){wanted=prefix+i;break;}}
   }
   mapping.set(source.id,wanted);occupied.add(wanted);
  }
  for(const source of incoming){
   const id=mapping.get(source.id),inputs=(source.inputs||[]).map(x=>mapping.get(x)||x);
   const params=clone(source.params||{});if(params.polygonParent)params.polygonParent=mapping.get(params.polygonParent)||params.polygonParent;
   staging._create({...source,id,label:id,inputs,params,__legacy:true,__deferEdges:true},true);
  }
  for(const poly of staging.nodes.filter(x=>x.op==='polygon'))staging.createPolygonEdges(poly);
  const before=this._take();this._loadRaw(staging.serialize());if(record)this._commit(before,'import');else this._emit('load');return true;}
 snapshot(){return {revision:this.revision,nodes:this.nodes.map(n=>({...clone({id:n.id,label:n.label,op:n.op,inputs:n.inputs,params:n.params,style:n.style}),kind:METHODS[n.op].kind,valid:n.valid,geom:n.valid?clone(n.geom):null}))};}
 // Strictly labeled as a *numeric* relation snapshot; never a proof.
 relationSnapshot(){const pts=[],lines=[],circles=[];for(const n of this.nodes){if(!n.valid)continue;const g=n.geom;if(g.kind==='point')pts.push({id:n.id,x:g.x,y:g.y});if(g.kind==='line')lines.push({id:n.id,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y,drawType:g.extent==='segment'?'lineSegment':g.extent});if(g.kind==='circle')circles.push({id:n.id,cx:g.c.x,cy:g.c.y,r:g.r,centerId:['circle','circleRadius'].includes(n.op)?n.inputs[0]:null});}const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);const span=Math.max(1,...xs.map(Math.abs),...ys.map(Math.abs))*2;return {span,points:pts,lines,circles};}
}
function commandNumber(input){
 const expression=(typeof globalThis!=='undefined'&&globalThis.GeomExpression)||(typeof require==='function'?require('./expression.js'):null);
 if(expression)return expression.parse(input);const n=Number(input);if(!Number.isFinite(n))throw Error('公式参数无效：'+input);return n;
}
function previewConstruction(op,geometries,params={}){
 const spec=METHODS[op];if(!spec||geometries.length<(spec.variadic?3:spec.inputs.length))return null;
 const inputs=geometries.map((_,i)=>'preview_'+i);
 return calc({op,inputs,params},id=>geometries[Number(id.slice(8))]);
}
function runText(kernel,line){const parts=(line.match(/"[^"]*"|'[^']*'|\S+/g)||[]).map(s=>/^['"]/.test(s)?s.slice(1,-1):s);if(!parts.length||parts[0].startsWith('#'))return null;
 const [cmd,...p]=parts;const a=cmd.toLowerCase();if(a==='help'||a==='?')return Object.values(METHODS).map(s=>s.syntax).join('\n')+'\nmove A x y | hide A | show A | delete A | rename A 新名 | undo | redo | clear | list | export';
 if(a==='list')return kernel.list().map(n=>`${n.id}  ${n.op}(${n.inputs.join(', ')}) ${n.valid?'':'[失效]'}`).join('\n')||'(空)';
 if(a==='export')return JSON.stringify(kernel.serialize(),null,2);
 if(['undo','redo','clear'].includes(a))return kernel.execute({op:a});
 if(a==='delete'||a==='del')return kernel.execute({op:'delete',id:p[0]});
 if(a==='hide'||a==='show')return kernel.execute({op:'style',id:p[0],style:{visible:a==='show'}});
 if(a==='rename')return kernel.execute({op:'rename',id:p[0],label:p.slice(1).join(' ')});
 if(a==='move'){if(p.length!==3)throw Error('move A x y');return kernel.execute({op:'move',id:p[0],x:commandNumber(p[1]),y:commandNumber(p[2])});}
 const aliases={point:'free',real:'real',online:'onLine',oncircle:'onCircle',line:'line',segment:'segment',ray:'ray',circle:'circle',circler:'circleRadius',midpoint:'midpoint',similarpoint:'similarPoint',equalangleline:'equalAngleLine',perpendicular:'perpendicular',parallel:'parallel',perpbisector:'perpBisector',bisector:'angleBisector',foot:'foot',reflectpoint:'reflectPoint',reflectline:'reflectLine',translate:'translate',rotate:'rotate',homothety:'homothety',centroid:'centroid',circumcenter:'circumcenter',circumcircle:'circumcircle',incenter:'incenter',orthocenter:'orthocenter',tangent:'tangent',tangentfrom:'tangentFrom',polygon:'polygon',parallelogram:'parallelogram',excenter:'excenter',outerbisector:'outerAngleBisector',diametercircle:'diameterCircle',circlebysegment:'circleBySegment',equilateral:'equilateral',rotateright:'rotateRight',rotateleft:'rotateLeft',foot3:'foot3',parallel3:'parallel3',perpendicular3:'perpendicular3',oncircum:'onCircumcircle'};
 const op=aliases[a]||Object.keys(METHODS).find(name=>name.toLowerCase()===a);if(['intersection','secondIntersection'].includes(op))throw Error('交点功能已由智能点替代');if(!op)throw Error('未知命令：'+cmd+'，输入 help 查看');const spec=METHODS[op];const id=p[0];let inputs=[],params={};
 if(op==='free'){if(p.length!==3)throw Error(spec.syntax);params={x:commandNumber(p[1]),y:commandNumber(p[2])};}
 else if(op==='real'){params={x:commandNumber(p[1])};}
 else if(op==='onLine'){inputs=[p[1]];params.t=commandNumber(p[2]);}
 else if(op==='onCircle'){inputs=[p[1]];params.theta=commandNumber(p[2])*(String(p[3]||'').toLowerCase()==='rad'?1:Math.PI/180);}
 else if(op==='onCircumcircle'){inputs=p.slice(1,4);params.theta=commandNumber(p[4])*(String(p[5]||'').toLowerCase()==='rad'?1:Math.PI/180);}
 else if(op==='circleRadius'){inputs=[p[1]];params.radius=commandNumber(p[2]);}
 else{const count=spec.variadic?p.length-1:spec.inputs.length;inputs=p.slice(1,1+count);
  const extra=p.slice(1+count);if(op==='intersection'||op==='tangentFrom')params.branch=commandNumber(extra[0]||0);
  if(op==='rotate'){if(!extra.length)throw Error(spec.syntax+' 角度 [deg|rad]');const unit=String(extra[1]||'deg').toLowerCase();if(!['deg','rad'].includes(unit))throw Error('角度单位必须为 deg 或 rad');params.degrees=commandNumber(extra[0])*(unit==='rad'?180/Math.PI:1);params.angleExpr=String(extra[0]);params.angleUnit=unit;}if(op==='homothety'){if(!extra.length)throw Error(spec.syntax+' 倍数');params.factor=commandNumber(extra[0]);params.factorExpr=String(extra[0]);}}
 if(!p[0]||inputs.some(x=>!x))throw Error(spec.syntax);
 return kernel.execute({op,id,inputs,params});
}
function registerMethod(name,spec,evaluator){if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(name)||METHODS[name])throw Error('构造名称重复或无效');if(!spec||!['point','line','circle','polygon'].includes(spec.kind)||!Array.isArray(spec.inputs)||typeof evaluator!=='function')throw Error('扩展构造定义不完整');if(!spec.inputs.every(x=>['point','line','circle','carrier'].includes(x)))throw Error('扩展构造参数类型错误');METHODS[name]={...spec,syntax:spec.syntax||name+' id '+spec.inputs.join(' ')};customEvaluators[name]=evaluator;return METHODS[name];}
return {Kernel,METHODS,EPS,schema,RESERVED,previewConstruction,intersections,intersectionBranches,project,circle3,dist,runText,registerMethod,version:'4.0.0'};
});
