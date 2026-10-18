/* GeomTool v7.7 — one mechanical interpreter for AG2 and Newclid.
 * Does not invent formal constructions for AG2 coordinate-only predicate inputs.
 * Each symbolic clause is applied atomically; unsupported constraints are not discarded.
 */
(function(root,factory){const api=factory(root.GeomKernel||(typeof require==='function'?require('./kernel.js'):null));root.GeomLibrary=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(G){
'use strict';
const legal=id=>/^[a-zA-Z][a-zA-Z0-9_]*$/.test(id)&&id.length<=32;
function hash(s){let h=2166136261;for(const c of s){h=Math.imul(h^c.charCodeAt(0),16777619);}return h>>>0;}
function seed(s,index,attempt){const h=hash(s+'|'+attempt),t=index*2.3999632297+(h%500)/1300,r=1.2+.65*Math.sqrt(index+1);return {x:r*Math.cos(t),y:r*Math.sin(t)};}
// Specialized carrier primitives stay inside the existing Kernel extension API.
// Angle loci remain editable: a dependent point is constrained to the carrier.
if(!G.METHODS.ag47AngleLine)G.registerMethod('ag47AngleLine',
 {kind:'line',inputs:['point','point','point','point','point'],group:'line',label:'定角轨迹'},
 (a,p,h)=>{const [A,B,C,D,E]=a,v=h.sub(B,A),w=h.sub(D,C),z=h.sub(E,D);
  if(h.norm(v)<1e-9||h.norm(w)<1e-9||h.norm(z)<1e-9)return null;
  const t=-Math.atan2(h.cross(w,z),h.dot(w,z)),dx=v.x*Math.cos(t)-v.y*Math.sin(t),dy=v.x*Math.sin(t)+v.y*Math.cos(t);
  return {kind:'line',p:A,d:{x:-dx,y:-dy},extent:'ray'};
 });
if(!G.METHODS.ag47SignedRay)G.registerMethod('ag47SignedRay',
 {kind:'line',inputs:['point','point'],group:'line',label:'定角射线'},
 (a,p,h)=>{const v=h.sub(a[0],a[1]);if(h.norm(v)<1e-9||!Number.isFinite(p.degrees))return null;
  const t=p.degrees*Math.PI/180;return {kind:'line',p:a[1],d:{x:v.x*Math.cos(t)-v.y*Math.sin(t),y:v.x*Math.sin(t)+v.y*Math.cos(t)},extent:'ray'};
 });
if(!G.METHODS.ag47Angle0Line)G.registerMethod('ag47Angle0Line',
 {kind:'line',inputs:['point','point','point','point','point','point','point'],group:'line',label:'AG2 定角平行轨迹'},
 (a,p,h)=>{const [A,B,C,D,E,F,H]=a,u=h.sub(B,A),v=h.sub(D,C),w=h.sub(F,E);
  if(h.norm(u)<1e-9||h.norm(v)<1e-9||h.norm(w)<1e-9)return null;
  const t=Math.atan2(h.cross(u,v),h.dot(u,v));
  return {kind:'line',p:H,d:{x:w.x*Math.cos(t)-w.y*Math.sin(t),y:w.x*Math.sin(t)+w.y*Math.cos(t)},extent:'ray'};
 });
if(!G.METHODS.ag47AngleMirrorLine)G.registerMethod('ag47AngleMirrorLine',
 {kind:'line',inputs:['point','point','point'],group:'line',label:'定角镜像轨迹'},
 (a,p,h)=>{const [A,B,C]=a,u=h.sub(A,B),v=h.sub(C,B),w=h.sub(B,C);
  if(h.norm(u)<1e-9||h.norm(v)<1e-9)return null;
  const t=Math.atan2(h.cross(u,v),h.dot(u,v));
  return {kind:'line',p:B,d:{x:w.x*Math.cos(t)-w.y*Math.sin(t),y:w.x*Math.sin(t)+w.y*Math.cos(t)},extent:'ray'};
 });
if(!G.METHODS.ag47TangentTouch)G.registerMethod('ag47TangentTouch',
 {kind:'point',inputs:['point','point','point','point'],group:'point',label:'双圆公切线切点'},
 (a,p,h)=>{const [O,A,W,B]=a,d=h.sub(W,O),dist=h.norm(d),r=h.dist(O,A),s=h.dist(W,B);
  if(dist<1e-10||r<1e-10||s<1e-10)return null;
  const internal=!!p.internal,k=(r-(internal?-s:s))/dist;if(Math.abs(k)>=1-1e-9)return null;
  const ux=d.x/dist,uy=d.y/dist,perp=Math.sqrt(1-k*k)*(p.side===1?-1:1),
   n={x:k*ux-perp*uy,y:k*uy+perp*ux};
  return p.which===1?{kind:'point',...h.add(W,h.mul(n,internal?-s:s))}:{kind:'point',...h.add(O,h.mul(n,r))};
 });
if(!G.METHODS.ag47ScaledCircle)G.registerMethod('ag47ScaledCircle',
 {kind:'circle',inputs:['point','point'],group:'circle',label:'定倍数半径圆'},
 (a,p,h)=>{const r=h.dist(a[0],a[1])*p.factor;return r>1e-9?{kind:'circle',c:a[0],r}:null;});
if(!G.METHODS.ag47CircleByScaleThree)G.registerMethod('ag47CircleByScaleThree',
 {kind:'circle',inputs:['point','point','point'],group:'circle',label:'三点长度比例圆'},
 (a,p,h)=>{const r=h.dist(a[1],a[2])*p.factor;return r>1e-9?{kind:'circle',c:a[0],r}:null;});
if(!G.METHODS.ag47Apollonius)G.registerMethod('ag47Apollonius',
 {kind:'circle',inputs:['point','point','point','point','point','point'],group:'circle',label:'阿波罗尼斯圆'},
 (a,p,h)=>{const [A,B,C,D,E,F]=a,den=h.dist(E,F);if(den<1e-9)return null;
 const r=h.dist(C,D)/den,k=r*r,div=1-k;
 if(Math.abs(div)<1e-7||h.dist(A,B)<1e-9)return null;
 const center=h.mul(h.sub(A,h.mul(B,k)),1/div);
 return {kind:'circle',c:center,r:Math.abs(r*h.dist(A,B)/div)};
 });
function convert(entry,options={}){
 if(!G)throw Error('Kernel missing');
 const attempts=Math.max(1,Math.min(8,options.attempts||4));
 if(entry.kind==='coordinates'){
  const k=new G.Kernel(),skipped=[];let n=0;
  for(const id of entry.points||Object.keys(entry.coords||{})){
   if(!legal(id)||!entry.coords?.[id]||k.node(id)){skipped.push(id);continue;}
   const [x,y]=entry.coords[id];
   try{k._create({op:'free',id,inputs:[],params:{x,y}});n++;}catch(e){skipped.push(id+' '+e.message);}
  }
  return {graph:k.serialize(),converted:n,total:(entry.points||[]).length,skipped,
   valid:k.list().filter(v=>v.valid).length,seedCount:0,coordinateOnly:true,complete:skipped.length===0};
 }
 let best=null;
 for(let attempt=0;attempt<attempts;attempt++){
  let k=new G.Kernel(),converted=0,skipped=[],seq=0,freeIndex=0;const sample=entry.coords||{};
  const newId=()=>{let id;do{id='libaux_'+(seq++);}while(k.node(id));return id;};
  function node(id){return k.node(id);}
  function add(op,id,args=[],params={}){
   if(!legal(id)||node(id)||!G.METHODS[op])return null;
   try{const prev=k.list();const made=k._create({op,id,inputs:args,params,style:id.startsWith('libaux_')?{visible:false,dashed:true,system:true}:undefined});
    if(made.geom?.kind==='point'&&!id.startsWith('libaux_')&&prev.some(p=>p.geom?.kind==='point'&&!p.id.startsWith('libaux_')&&Math.hypot(p.geom.x-made.geom.x,p.geom.y-made.geom.y)<1e-8))return null;
    return made;}catch(e){return null;}
  }
  function temp(op,args,params={}){return add(op,newId(),args,params)?.id||null;}
  function free(id){if(!legal(id))return false;if(node(id))return true;const pt=sample[id]?{x:sample[id][0],y:sample[id][1]}:seed(entry.id+'|'+id,freeIndex++,attempt);
   return !!add('free',id,[],pt);}
  function line(a,b){return temp('line',[a,b]);}
  function circle(a,b){return temp('circle',[a,b]);}
  function cross(l,r,id){if(!l||!r)return false;
   const ga=node(l)?.geom,gb=node(r)?.geom;if(!ga||!gb)return false;
   const pts=G.intersections(ga,gb);if(!pts.length)return false;
   let branch=0;
   if(sample[id]&&pts.length>1){const t=sample[id];branch=Math.hypot(pts[1].x-t[0],pts[1].y-t[1])<Math.hypot(pts[0].x-t[0],pts[0].y-t[1])?1:0;}
   // Prefer a distinct intersection when one root is a known endpoint. A
   // coincident numeric root is not evidence that two symbolic points coincide.
   else if(pts.length>1){const old=k.list().filter(n=>n.geom?.kind==='point'&&!n.id.startsWith('libaux_'));
    const near=p=>old.some(n=>Math.hypot(n.geom.x-p.x,n.geom.y-p.y)<1e-7);
    if(near(pts[0])&&!near(pts[1]))branch=1;}
   // Kernel has sparse intersection slots; try candidate branches rather than
   // assuming that the filtered intersection array index equals the raw slot.
   return !!add('intersection',id,[l,r],{branch})||!!add('intersection',id,[l,r],{branch:1-branch});
  }
  function parameter(id){return .22+(hash(entry.id+id+attempt)%520)/1000;}
  function locus(kind,a){
   switch(kind){
    case 'on_line':return a.length===2?line(a[0],a[1]):null;
    case 'on_pline':case 'on_pline0':return a.length===3?temp('parallel3',a):null;
    case 'on_tline':return a.length===3?temp('perpendicular3',a):null;
    case 'on_bline':return a.length===2?temp('perpBisector',a):null;
    case 'on_circle':return a.length===2?circle(a[0],a[1]):null;
    case 'on_circum':return a.length===3?temp('circumcircle',a):null;
    case 'on_dia':return a.length===2?temp('diameterCircle',a):null;
    case 'eqdistance':return a.length===3?temp('circleBySegment',a):null;
    case 'angle_bisector':return a.length===3?temp('angleBisector',a):null;
    case 'lc_tangent':return a.length===2?temp('perpendicular3',[a[0],a[0],a[1]]):null;
    case 'on_aline':return a.length===5?temp('ag47AngleLine',a):null;
    case 's_angle':{const degrees=Number((a[3]||'').replace(/o$/,''));return a.length===4&&Number.isFinite(degrees)?temp('ag47SignedRay',[a[0],a[1]],{degrees}):null;}
    case 'between':case 'between_bound':return a.length===2?temp('segment',a):null;
    case 'on_aline0':return a.length===7?temp('ag47Angle0Line',a):null;
    case 'angle_mirror':return a.length===3?temp('ag47AngleMirrorLine',a):null;
    case 'lconst':case 'l2const':{const n=Number(a[1]);const rad=kind==='l2const'?Math.sqrt(n):n;return a.length===2&&Number.isFinite(rad)&&rad>0?temp('circleRadius',[a[0]],{radius:rad}):null;}
    case 'eqratio6':return a.length===6?temp('ag47Apollonius',a):null;

    default:return null;
   }
  }
  function fromLocus(id,carrier){if(!carrier)return false;const g=node(carrier)?.geom;
   if(g?.kind==='line')return !!add('onLine',id,[carrier],{t:parameter(id)});
   if(g?.kind==='circle')return !!add('onCircle',id,[carrier],{theta:parameter(id)*9});
   return false;
  }
  function single(st){const outs=st.outputs||[],ops=st.operations||[],first=ops[0]||{},kind=first.op,a=first.args||[],id=outs[0];
   if(!outs.length)return false;
   if(!ops.length)return outs.every(free);
   const freeGroups=['free','triangle','segment','quadrangle','pentagon','acute_triangle'];
   if(ops.length===1&&freeGroups.includes(kind))return outs.every(free);
   if(ops.length===1&&kind==='between'&&outs.length===3&&a.length===0){
    return !!(free(outs[0])&&free(outs[2])&&fromLocus(outs[1],temp('segment',[outs[0],outs[2]])));
   }
   if(ops.length===1){
    const direct={midpoint:['midpoint',2],foot:['foot3',3],orthocenter:['orthocenter',3],circumcenter:['circumcenter',3],incenter:['incenter',3],excenter:['excenter',3],circle:['circumcenter',3],eq_triangle:['equilateral',2],mirror:['reflectPoint',2],reflect:['reflectLine',3],parallelogram:['parallelogram',3],psquare:['rotateLeft',2],nsquare:['rotateRight',2],shift:['translate',3]};
    if(kind==='circle'&&outs.length===1&&a.length===3)return !!add('circumcenter',id,a);
    if(kind==='ieq_triangle'&&outs.length===3&&a.length===3&&a.every((v,i)=>v===outs[i])){
      return !!(free(outs[0])&&free(outs[1])&&add('equilateral',outs[2],[outs[0],outs[1]],{branch:1}));
    }
    if(kind==='triangle12'&&outs.length===3&&a.length===3&&a.every((v,i)=>v===outs[i])){
      if(!free(outs[0])||!free(outs[1]))return false;
      return fromLocus(outs[2],temp('ag47ScaledCircle',[outs[0],outs[1]],{factor:2}));
    }
    if(kind==='rconst'&&outs.length===1&&a.length===5&&a[3]===id){
      const m=a[4].split('/').map(Number);if(m.length!==2||!m[0]||!m[1])return false;
      // AB : CX = numerator : denominator.
      return fromLocus(id,temp('ag47CircleByScaleThree',[a[2],a[0],a[1]],{factor:m[1]/m[0]}));
    }
    if((kind==='cc_tangent'||kind==='cc_itangent')&&outs.length===4&&a.length===4){
     const intrinsic=kind==='cc_itangent';
     return outs.every((target,j)=>!!add('ag47TangentTouch',target,a,{internal:intrinsic,side:j>=2?1:0,which:j%2}));
    }
    if(kind==='reflect'&&a.length===3){const l=line(a[1],a[2]);return !!(l&&add('reflectLine',id,[a[0],l]));}
    if(kind==='parallelogram'&&a.length===3){return !!add('parallelogram',id,[a[1],a[0],a[2]]);}
    if(kind==='parallelogram'&&a.length===4){ // named final vertex in some source clauses
      if(a[3]!==id)return false;
      return !!add('parallelogram',id,[a[1],a[0],a[2]]);
    }
    if(kind==='shift'&&a.length===3)return !!add('translate',id,[a[1],a[2],a[0]]);
    if(direct[kind]&&outs.length===1&&a.length===direct[kind][1])return !!add(direct[kind][0],id,a,kind==='eq_triangle'?{branch:0}:{});
    if(kind==='intersection_ll'&&a.length===4)return cross(line(a[0],a[1]),line(a[2],a[3]),id);
    if(kind==='intersection_lc'&&a.length===3)return cross(line(a[0],a[2]),circle(a[1],a[2]),id);
    if(kind==='intersection_cc'&&a.length===3)return cross(circle(a[0],a[2]),circle(a[1],a[2]),id);
    if(kind==='intersection_lp'&&a.length===5)return cross(line(a[0],a[1]),temp('parallel3',[a[2],a[3],a[4]]),id);
    if(kind==='intersection_lt'&&a.length===5)return cross(line(a[0],a[1]),temp('perpendicular3',[a[2],a[3],a[4]]),id);
    if(kind==='intersection_pp'&&a.length===6)return cross(temp('parallel3',a.slice(0,3)),temp('parallel3',a.slice(3)),id);
    if(kind==='intersection_tt'&&a.length===6)return cross(temp('perpendicular3',a.slice(0,3)),temp('perpendicular3',a.slice(3)),id);
    if(kind==='incenter2'||kind==='excenter2'||kind==='centroid'||kind==='ninepoints'){
     if(outs.length!==4||a.length!==3)return false;
     const [x,y,z,i]=outs,[A,B,C]=a;
     const center=kind==='incenter2'?'incenter':kind==='excenter2'?'excenter':kind==='centroid'?'centroid':'circumcenter';
     if(kind==='ninepoints'){
      return !!(add('midpoint',x,[B,C])&&add('midpoint',y,[C,A])&&add('midpoint',z,[A,B])&&add('circumcenter',i,[x,y,z]));
     }
     if(kind==='centroid')return !!(add('midpoint',x,[B,C])&&add('midpoint',y,[C,A])&&add('midpoint',z,[A,B])&&add('centroid',i,[A,B,C]));
     return !!(add(center,i,[A,B,C])&&add('foot3',x,[i,B,C])&&add('foot3',y,[i,C,A])&&add('foot3',z,[i,A,B]));
    }
    if(kind==='square'&&outs.length===2&&a.length===4&&a[2]===outs[0]&&a[3]===outs[1]){
     return !!(add('rotateLeft',outs[0],[a[0],a[1]])&&add('parallelogram',outs[1],[a[1],a[0],outs[0]]));
    }
    if(kind==='isquare'&&outs.length===2&&a.length===4&&a[2]===outs[0]&&a[3]===outs[1]){
     return !!(add('rotateLeft',outs[0],[a[0],a[1]])&&add('parallelogram',outs[1],[a[1],a[0],outs[0]]));
    }
    if(kind==='square'||kind==='isquare'||kind==='rectangle'){
     if(outs.length!==4)return false;
     const [A,B,C,D]=outs;if(!free(A)||!free(B))return false;
     if(kind==='rectangle'){const pl=temp('perpendicular3',[B,A,B]);return !!(pl&&add('onLine',C,[pl],{t:.7})&&add('parallelogram',D,[B,A,C]));}
     return !!(add('rotateLeft',C,[A,B])&&add('parallelogram',D,[B,A,C]));
    }
    if(kind==='r_triangle'||kind==='risos'||kind==='iso_triangle'||kind==='iso_triangle_vertex'||kind==='trapezoid'){
     if(kind==='iso_triangle_vertex'&&outs.length===1&&a.length===2){const l=temp('perpBisector',a);return fromLocus(id,l);}
     const [A,B,C,D]=outs;
     if(kind==='trapezoid'){if(outs.length!==4||!free(A)||!free(B)||!free(C))return false;return fromLocus(D,temp('parallel3',[C,A,B]));}
     if(outs.length!==3||!free(A)||!free(B))return false;
     if(kind==='iso_triangle')return fromLocus(C,circle(A,B));
     if(kind==='r_triangle')return fromLocus(C,temp('perpendicular3',[A,A,B]));
     if(kind==='risos')return !!add('rotateLeft',C,[B,A]);
    }
    // Isosceles trapezoid additionally requires equal legs; do not approximate it.
    if(kind==='eq_trapezoid')return false;
    if(kind==='trisegment'&&outs.length===2&&a.length===2){return !!(add('homothety',outs[0],[a[1],a[0]],{factor:1/3})&&add('homothety',outs[1],[a[0],a[1]],{factor:1/3}));}
    if(outs.length===1){const l=locus(kind,a);if(l)return fromLocus(id,l);}
   }
   if(outs.length!==1||ops.length>2)return false;
   // Every constraint must be represented; never silently discard a conjunct.
   const carriers=ops.map(p=>locus(p.op,p.args||[]));
   if(carriers.some(x=>!x))return false;
   if(carriers.length===1)return fromLocus(id,carriers[0]);
   return cross(carriers[0],carriers[1],id);
  }
  for(const st of entry.steps||[]){
   const snap=k.serialize(),oldSeq=seq,oldFree=freeIndex;
   if(single(st)){converted++;continue;}
   // Atomic rollback includes any anonymous construction carriers introduced.
   k=new G.Kernel(snap);seq=oldSeq;freeIndex=oldFree;
   skipped.push(st.raw||st.id);
  }
  const valid=k.list().filter(x=>x.valid).length;
  const result={graph:k.serialize(),converted,total:entry.steps.length,skipped,valid,seedCount:freeIndex,coordinateOnly:false,
   complete:skipped.length===0,score:converted*100+valid};
  if(!best||result.score>best.score)best=result;
  if(result.complete)break;
 }
 return best;
}
function search(data,query='',source='',offset=0,limit=36){
 const terms=String(query).toLowerCase().trim().split(/\s+/).filter(Boolean),matches=[];
 for(const e of data?.entries||[]){if(source&&source!==e.source)continue;
  if(terms.length){const hay=(e.title+' '+e.source+' '+e.sourcePath+' '+e.raw).toLowerCase();if(!terms.every(s=>hay.includes(s)))continue;}
  matches.push(e);
 }
 return {total:matches.length,entries:matches.slice(offset,offset+limit)};
}
return {convert,search,seed};
});
