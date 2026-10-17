/* GeomTool v4.5 — construction-to-predicate boundary.
 * This module DOES NOT perform logical deduction; all proof checks go through
 * unmodified AlphaGeometry2 DDAR Python sources. Never turn coincident screen
 * coordinates into assumptions. Every predicate below has a construction role.
 * License: original GeomTool integration, independent of DeepMind source files.
 */
(function(root,factory){const api=factory();root.GeomDDARCompiler=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const seg=(a,b)=>[a,b].sort().join('|');
const lp=(a,b)=>'P:'+seg(a,b);
function compile(snapshot){
 const nodes=(snapshot.nodes||[]).filter(n=>n.valid&&n.geom),byId=new Map(nodes.map(n=>[n.id,n]));
 const points=new Map(),predicates=[],seen=new Set(),lines=new Map(),circles=new Map(),limitations=[],assumptionReport=[],virtualDescriptions=[];
 for(const n of nodes)if(n.geom.kind==='point'&&Number.isFinite(n.geom.x)&&Number.isFinite(n.geom.y))points.set(n.id,[n.geom.x,n.geom.y]);
 const has=(...a)=>a.every(x=>points.has(x));
 const dist=(a,b)=>{const u=points.get(a),v=points.get(b);return u&&v?Math.hypot(u[0]-v[0],u[1]-v[1]):0;};
 const numericOK=(name,pp,cc)=>{
  const xy=a=>points.get(a), len=(a,b)=>dist(a,b);
  const vector=(a,b)=>{const x=xy(a),y=xy(b);return [y[0]-x[0],y[1]-x[1]]};
  const cross=(u,v)=>u[0]*v[1]-u[1]*v[0],dot=(u,v)=>u[0]*v[0]+u[1]*v[1];
  const orient=(a,b,c,d)=>{const u=vector(a,b),v=vector(c,d);return Math.atan2(cross(u,v),dot(u,v))/Math.PI};
  const modAngle=x=>Math.abs(x-Math.round(x));
  const eps=2e-6;
  if(name==='coll'){const [a,b,c]=pp;return Math.abs(cross(vector(a,b),vector(a,c)))<=eps*Math.max(1,len(a,b)*len(a,c))}
  if(name==='cong')return Math.abs(len(pp[0],pp[1])-len(pp[2],pp[3]))<=eps*Math.max(1,len(pp[0],pp[1]),len(pp[2],pp[3]));
  if(name==='para'||name==='perp'){const u=vector(pp[0],pp[1]),v=vector(pp[2],pp[3]);return Math.abs(name==='para'?cross(u,v):dot(u,v))<=eps*Math.max(1,len(pp[0],pp[1])*len(pp[2],pp[3]));}
  if(name==='eqangle')return modAngle(orient(...pp.slice(0,4))-orient(...pp.slice(4,8)))<=eps;
  if(name==='aconst')return modAngle(orient(...pp)+Number(cc[0])/180)<=eps;
  if(name==='rconst')return Math.abs(len(pp[0],pp[1])/len(pp[2],pp[3])-Number(cc[0]))<=eps*Math.max(1,Number(cc[0]));
  if(name==='eqratio')return Math.abs(len(pp[0],pp[1])*len(pp[6],pp[7])-len(pp[2],pp[3])*len(pp[4],pp[5]))<=eps*Math.max(1,len(pp[0],pp[1])*len(pp[6],pp[7]),len(pp[2],pp[3])*len(pp[4],pp[5]));
  return true;
 };
 const p=(name,pp,cc=[],source='construction')=>{if(!has(...pp))return;const k=JSON.stringify([name,pp,cc]);if(seen.has(k))return;
  if(!numericOK(name,pp,cc)){limitations.push('跳过数值不一致的构造前提: '+source+' '+name+' '+pp.join(' '));return;}
  seen.add(k);predicates.push({name,points:pp,constants:cc,source});};
 const coll=(a,b,c,source)=>{if(new Set([a,b,c]).size===3&&dist(a,b)>1e-10&&dist(a,c)>1e-10)p('coll',[a,b,c],[],source)};
 const cong=(a,b,c,d,source)=>{if(dist(a,b)>1e-10&&dist(c,d)>1e-10)p('cong',[a,b,c,d],[],source)};
 const para=(a,b,c,d,source)=>{if(dist(a,b)>1e-10&&dist(c,d)>1e-10)p('para',[a,b,c,d],[],source)};
 const perp=(a,b,c,d,source)=>{if(dist(a,b)>1e-10&&dist(c,d)>1e-10)p('perp',[a,b,c,d],[],source)};
 const angle=(a,b,c,d,e,f,g,h,source)=>{if([[a,b],[c,d],[e,f],[g,h]].every(([x,y])=>dist(x,y)>1e-10))p('eqangle',[a,b,c,d,e,f,g,h],[],source)};
 const rconst=(a,b,c,d,k,source)=>{if(k>0&&Number.isFinite(k)&&dist(a,b)>1e-10&&dist(c,d)>1e-10)p('rconst',[a,b,c,d],[k],source)};
 const aconst=(a,b,c,d,v,source)=>{if(Number.isFinite(v)&&dist(a,b)>1e-10&&dist(c,d)>1e-10)p('aconst',[a,b,c,d],[v],source)};
 const mid=(m,a,b,s)=>{coll(m,a,b,s);cong(a,m,m,b,s)};
 const same=(a,b,s)=>{if(has(a,b))p('overlap',[a,b],[],s)};
 const addPoint=(id,xy,reason='内部构造见证点')=>{if(xy&&xy.every(Number.isFinite)&&!points.has(id)){
  points.set(id,xy);virtualDescriptions.push({id,reason,coordinates:xy});return id;}return points.has(id)?id:null};
 const get=(x)=>points.get(x),avg=(a,b)=>{const u=get(a),v=get(b);return u&&v?[(u[0]+v[0])/2,(u[1]+v[1])/2]:null};
 // Reuse a provably identical user midpoint (same original constructor), not
 // a visually coincident point. This saves internal witness points and avoids
 // invalid duplicate numeric sketches on simple midpoint-line examples.
 const midpointBySegment=new Map();
 for(const n of nodes)if(n.op==='midpoint'&&n.inputs?.length===2&&points.has(n.id))
  midpointBySegment.set(seg(n.inputs[0],n.inputs[1]),n.id);
 const virtualMid=(a,b)=>midpointBySegment.get(seg(a,b))||addPoint('@mid:'+seg(a,b),avg(a,b),'线段 '+a+b+' 的中点');
 function lineCarrier(id){return lines.get(id)||null}
 // Lines: two witnesses that belong to the CONSTRUCTED line, not free points.
 // Base line uses its defining endpoints; a derived line needs one synthetic
 // witness at one direction-vector step from its construction-defined point.
 for(const n of nodes.filter(n=>n.geom.kind==='line')){
  const t=n.inputs||[],g=n.geom;
  if(['line','segment','ray'].includes(n.op)&&has(t[0],t[1])&&dist(t[0],t[1])>1e-10){lines.set(n.id,[t[0],t[1]]);continue}
  const anchor=({parallel:t[0],perpendicular:t[0],parallel3:t[0],perpendicular3:t[0],perpBisector:null,angleBisector:t[1],outerAngleBisector:t[1],equalAngleLine:t[4],tangent:t[0],tangentFrom:t[0]})[n.op];
  let q=anchor&&get(anchor);
  if(n.op==='perpBisector'){const m=virtualMid(t[0],t[1]);if(m)q=get(m)}
  if(!q&&g.p&&Number.isFinite(g.p.x)&&Number.isFinite(g.p.y))q=[g.p.x,g.p.y];
  const vec=g.d,len=vec&&Math.hypot(vec.x,vec.y);
  if(!q||!len||len<1e-12){limitations.push('直线无法编译: '+n.id);continue}
  const s=has(anchor)?anchor:n.op==='perpBisector'?virtualMid(t[0],t[1]):addPoint('@line:'+n.id+':0',q,'直线 '+n.id+' 的起点'),e=addPoint('@line:'+n.id+':1',[q[0]+vec.x/len,q[1]+vec.y/len],'直线 '+n.id+' 上的见证点');
  if(s&&e)lines.set(n.id,[s,e]);
  if(anchor&&s&&e)coll(anchor,s,e,n.op+':point-on-line');
 }
 const carrier=(key)=>{if(typeof key!=='string')return null;if(key.startsWith('P:')){const a=key.slice(2).split('|');return a.length===2&&has(...a)?a:null}if(key.startsWith('L:'))return lineCarrier(key.slice(2));return lineCarrier(key)};
 const circlePoints=new Map();const circleCenters=new Map();
 const circleRadii=new Map();
 const onc=(pt,id)=>{if(!points.has(pt))return;let s=circlePoints.get(id);if(!s)circlePoints.set(id,s=new Set());s.add(pt)};
 for(const n of nodes.filter(n=>n.geom.kind==='circle')){
  const t=n.inputs||[];circles.set(n.id,{op:n.op,inputs:t});
  if(['circle','circleRadius','circleBySegment'].includes(n.op))circleCenters.set(n.id,t[0]);
  if(n.op==='circle')onc(t[1],n.id);
  if(n.op==='circleRadius'&&has(t[0])){const c=get(t[0]),r=Number(n.params?.radius);if(r>0&&Number.isFinite(r)){const witness=(n.id==='c_unit'&&t[0]==='p_0'&&r===1&&has('p_1'))?'p_1':addPoint('@radius:'+n.id,[c[0]+r,c[1]],'定长圆 '+n.id+' 的半径见证点');if(witness){onc(witness,n.id);circleRadii.set(n.id,witness);}}}
  if(n.op==='circumcircle'){t.forEach(a=>onc(a,n.id));const center=n.geom?.c;
    if(center&&t.length===3){const known=nodes.find(k=>k.op==='circumcenter'&&k.inputs?.length===3&&k.inputs.slice().sort().join('|')===t.slice().sort().join('|'));
    const o=known?.id||addPoint('@circumcenter:'+n.id,[center.x,center.y],'外接圆 '+n.id+' 的外心');
     if(o){circleCenters.set(n.id,o);cong(o,t[0],o,t[1],'circumcircle-center:'+n.id);cong(o,t[1],o,t[2],'circumcircle-center:'+n.id);}
    }
   }
  if(n.op==='diameterCircle'){const m=virtualMid(t[0],t[1]);if(m){mid(m,t[0],t[1],'diameter-center');circleCenters.set(n.id,m)}t.forEach(a=>onc(a,n.id));}
 }
 for(const n of nodes){const t=n.inputs||[],x=n.id,op=n.op;
  const lin=(id)=>lineCarrier(id), lineOn=(pt,id,why)=>{const ab=lin(id);if(ab)coll(pt,...ab,why)};
  if(op==='equalAngleLine'){
   const ab=lin(x);if(ab){const q=ab[0]===t[4]?ab[1]:ab[0];
     // ∠ABC = ∠DEQ: (BA -> BC) and (ED -> EQ).
     angle(t[1],t[0],t[1],t[2],t[4],t[3],t[4],q,op);
   }
  }
  if(op==='similarPoint'){
   const [a,b,c,d,e]=t,f=x;
   // Degenerate cases cannot be certified by the classical AA/SSS/SAS
   // theorems. Record only individually meaningful equalities.
   const ab=dist(a,b),de=dist(d,e),ac=dist(a,c),bc=dist(b,c);
   const ratio=ab>1e-10?de/ab:null;
   if(ratio!==null&&ratio>1e-10){
     // Cross-side equal ratios are exact symbolic constraints: do not
     // convert a possibly irrational scale factor into a huge Fraction.
     for(const [u,v,w,z] of [[a,c,d,f],[b,c,e,f]])
       if(dist(u,v)>1e-10&&dist(w,z)>1e-10&&ab>1e-10&&de>1e-10)
         p('eqratio',[a,b,u,v,d,e,w,z],[],op);
     if(ac>1e-10&&bc>1e-10&&dist(d,f)>1e-10&&dist(e,f)>1e-10)
       p('eqratio',[a,c,b,c,d,f,e,f],[],op);
     for(const parts of [
       [a,b,a,c,d,e,d,f],
       [b,a,b,c,e,d,e,f],
       [c,a,c,b,f,d,f,e]
     ])angle(...parts,op);
     if(ac>1e-10&&bc>1e-10&&dist(d,f)>1e-10&&dist(e,f)>1e-10){
       angle(a,b,a,c,d,e,d,f,op);angle(b,a,b,c,e,d,e,f,op);
     }
   }else limitations.push('相似点 '+f+'：基准边退化或比例为零，经典相似定理不适用；仅保留坐标实例，不注入虚假相似关系');
  }
  if(op==='onLine'){lineOn(x,t[0],op);const source=byId.get(t[0]);
    if(source&&['segment','ray'].includes(source.op))limitations.push('线段/射线 '+t[0]+' 上的点 '+x+'：端点顺序由坐标体现，AG2 目前没有独立的顺序谓词');if(['line','segment','ray'].includes(source?.op)){
   const k=Number(n.params?.t),[a,b]=source.inputs;if(Number.isFinite(k)){if(k!==0)rconst(a,x,a,b,Math.abs(k),op);if(k!==1)rconst(x,b,a,b,Math.abs(1-k),op)}
  }}
  if(op==='real'){
   // Prefer fixed origin/unit witnesses; otherwise choose fresh independent
   // real-axis witnesses away from named points (different named points stay distinct).
   let a=points.has('p_0')?'p_0':null,b=points.has('p_1')?'p_1':null;
   if(!a||!b){let x0=Math.max(2,...[...points.values()].map(v=>v[0]))+Math.SQRT2;
    const occupied=z=>[...points.values()].some(v=>Math.hypot(v[0]-z,v[1])<1e-7);
    while(occupied(x0)||occupied(x0+1))x0+=Math.PI;
    a=a||addPoint('@real:0',[x0,0]);b=b||addPoint('@real:1',[x0+1,0]);}
   if(a&&b)coll(x,a,b,op);
  }
  if(op==='circleCenter'&&circles.has(t[0])){const prev=circleCenters.get(t[0]);if(prev&&prev!==x)same(x,prev,op);circleCenters.set(t[0],x);}
  if(op==='onCircle')onc(x,t[0]);
  if(op==='onCircumcircle'){
   for(const c of nodes.filter(a=>a.op==='circumcircle'&&a.inputs.slice().sort().join('|')===t.slice().sort().join('|')))onc(x,c.id);
   if(has(x,...t)&&t.length===3)p('cyclic',[x,...t],[],op);
  }
  if(op==='secondIntersection')limitations.push('第二交点的排除条件（不同于已知交点）尚不能以 AG 无向正谓词精确描述: '+x);
  if(op==='intersection'||op==='secondIntersection'){
   
   for(const id of (op==='secondIntersection'?t.slice(1):t)){
   const src=byId.get(id);if(src?.geom.kind==='line')lineOn(x,id,op);else if(src?.geom.kind==='circle')onc(x,id);
  }
  }
  if(op==='foot'){lineOn(x,t[1],op);const ab=lin(t[1]);if(ab)perp(t[0],x,...ab,op)}
  if(op==='foot3'){coll(x,t[1],t[2],op);perp(t[0],x,t[1],t[2],op)}
  if(op==='midpoint')mid(x,t[0],t[1],op);
  if(op==='perpBisector'){
   const ab=lin(x);if(ab){perp(...ab,t[0],t[1],op);const m=virtualMid(t[0],t[1]);if(m){mid(m,t[0],t[1],op);coll(m,...ab,op)}}
  }
  if(['parallel','perpendicular'].includes(op)){
   const a=lin(x),b=lin(t[1]);if(a&&b)(op==='parallel'?para:perp)(...a,...b,op);
  }
  if(['parallel3','perpendicular3'].includes(op)){
   const a=lin(x);if(a)(op==='parallel3'?para:perp)(...a,t[1],t[2],op);
  }
  if(op==='angleBisector'||op==='outerAngleBisector'){
   const a=lin(x);if(a){const [u,v,w]=t;const q=a[0]===v?a[1]:a[0];
    // Both bisectors satisfy the directed equality mod 180 degrees.
    // It is crucial that the actual line witness q occurs in both halves.
    angle(v,u,v,q,v,q,v,w,op);}
  }
  if(op==='orthocenter')for(let i=0;i<3;i++)perp(x,t[i],t[(i+1)%3],t[(i+2)%3],op);
  if(op==='circumcenter'){cong(x,t[0],x,t[1],op);cong(x,t[1],x,t[2],op)}
  if(op==='centroid')for(let i=0;i<3;i++){
   const a=t[i],b=t[(i+1)%3],c=t[(i+2)%3],m=virtualMid(b,c);
   if(m){mid(m,b,c,op);coll(a,x,m,op);rconst(a,x,x,m,2,op)}
  }
  if(op==='parallelogram'){
   // Kernel uses X = B + C - A: AB || CX, AC || BX.
   para(t[0],t[1],t[2],x,op);para(t[0],t[2],t[1],x,op);
   cong(t[0],t[1],t[2],x,op);cong(t[0],t[2],t[1],x,op);
  }
  if(op==='equilateral'){
   cong(t[0],t[1],t[0],x,op);cong(t[0],x,t[1],x,op);
   for(const [a,b,c] of [[t[0],t[1],x],[t[1],t[0],x],[x,t[0],t[1]]]){
    if(!has(a,b,c))continue;
    const u=get(a),v=get(b),w=get(c),z=(v[0]-u[0])*(w[1]-u[1])-(v[1]-u[1])*(w[0]-u[0]);
    aconst(a,b,a,c,z>=0?-60:60,op+':60°');
   }
  }
  if(op==='reflectPoint')mid(t[1],t[0],x,op);
  if(op==='translate'){
   para(t[0],x,t[1],t[2],op);para(t[1],t[0],t[2],x,op);
   cong(t[0],x,t[1],t[2],op);cong(t[1],t[0],t[2],x,op);
  }
  if(op==='homothety'){
   if(Number(n.params?.factor)<0)limitations.push('AG 内核的无向长度及模180角语言不能区分负位似的射线朝向: '+x);
   const k=Number(n.params?.factor);if(k===0)same(x,t[1],op);else{coll(t[0],t[1],x,op);rconst(t[1],x,t[1],t[0],Math.abs(k),op);}
  }
  if(['rotate','rotateLeft','rotateRight'].includes(op)){
   const deg=op==='rotateLeft'?90:op==='rotateRight'?-90:Number(n.params?.degrees);
   cong(t[1],t[0],t[1],x,op);if(Number.isFinite(deg))aconst(t[1],t[0],t[1],x,-deg,op);
  }
  if(['incenter','excenter'].includes(op)){
   // In/ex-center lies on two angle-bisector lines, including the external
   // variant: directed equal angles are modulo 180 degrees.
   for(let i=0;i<3;i++){const v=t[i],a=t[(i+1)%3],c=t[(i+2)%3];
    angle(v,a,v,x,v,x,v,c,op);
   }
  }
  if(op==='reflectLine'){
   const a=t[0],h=virtualMid(a,x),ab=lin(t[1]);
   if(ab){cong(a,ab[0],x,ab[0],op);cong(a,ab[1],x,ab[1],op);}
   if(h){mid(h,a,x,op);lineOn(h,t[1],op);if(ab)perp(a,x,...ab,op)}
  }
  if(op==='tangentFrom'){
   const c=circleCenters.get(t[1]),ab=lin(x),g=byId.get(x)?.geom;
   if(c&&ab&&g?.p&&g?.d){const ce=get(c),v=g.d,den=v.x*v.x+v.y*v.y;
    if(ce&&den>1e-12){const tau=((ce[0]-g.p.x)*v.x+(ce[1]-g.p.y)*v.y)/den;
     const h=addPoint('@tangent:'+x,[g.p.x+tau*v.x,g.p.y+tau*v.y],'外点切线 '+x+' 的切点');
     if(h){coll(h,...ab,op);perp(c,h,...ab,op);onc(h,t[1]);}
    }
   }else limitations.push('外点切线中心或直线无法编译: '+x);
  }
  if(op==='tangent'){
   const ab=lin(x),c=circleCenters.get(t[1]);if(ab&&c)perp(...ab,c,t[0],op);
   else if(ab&&circlePoints.has(t[1])){
    const [a,b]=ab,at=t[0],q=a===at?b:a,pts=[...(circlePoints.get(t[1])||[])].filter(u=>u!==at);
    if(pts.length>=2){const [u,v]=pts;angle(at,q,at,u,v,at,v,u,op+':弦切角');}
    else limitations.push('切线缺少圆心及两条弦端点，不能形式化: '+x);
   }
  }
 }
 // The fixed Cartesian reference points carry their own unit-axis axioms.
 if(has('p_0','p_1','p_i')){perp('p_0','p_1','p_0','p_i','reference-coordinate-axes');cong('p_0','p_1','p_0','p_i','reference-coordinate-unit');}
 // Circle axioms: all named members lie on the circle by construction.
 // Use radius congruences and center-aware cyclicity, never computed-circle
 // membership inferred from a coincidental match in numerical coordinates.
 for(const [id,ss] of circlePoints){const ps=[...ss],c=circleCenters.get(id),info=circles.get(id);
  if(c&&points.has(c)){
   for(let i=0;i<ps.length;i++)for(let j=i+1;j<ps.length;j++)cong(c,ps[i],c,ps[j],'radius:'+id);
   if(info?.op==='circleBySegment')for(const pt of ps)cong(c,pt,info.inputs[1],info.inputs[2],'radius:'+id);
   if(ps.length>=3)p('cyclic_with_centers',[c,...ps],[1],'circle:'+id);
  }
  if(ps.length>=4)p('cyclic',ps,[],'circle:'+id);
  // Thales' theorem is the defining condition for a point on a diameter circle.
  if(info?.op==='diameterCircle')for(const pt of ps)if(pt!==info.inputs[0]&&pt!==info.inputs[1])perp(info.inputs[0],pt,info.inputs[1],pt,'diameter-circle:'+id);
 }
 // Attach a stable source audit, but never claim these are derived proofs.
 for(const row of predicates)assumptionReport.push(row.source+': '+row.name+' '+row.points.join(' '));
 const coveredOps=new Set(['free','referencePoint','real','onLine','onCircle','onCircumcircle','line','segment','ray','circle','circleRadius','circleBySegment','diameterCircle','midpoint','intersection','secondIntersection','parallel','parallel3','perpendicular','perpendicular3','perpBisector','angleBisector','outerAngleBisector','foot','foot3','reflectPoint','reflectLine','translate','parallelogram','rotateLeft','rotateRight','equilateral','rotate','homothety','centroid','circumcenter','circumcircle','orthocenter','excenter','incenter','tangent','tangentFrom','polygon','circleCenter','similarPoint','equalAngleLine']);
 for(const n of nodes){if(!coveredOps.has(n.op))limitations.push('没有该自定义构造的形式公理: '+n.op+' '+n.id);}
 return {points:[...points].map(([name,value])=>({name,value,virtual:name.startsWith('@')})),virtualDescriptions,predicates,lines:Object.fromEntries(lines),circles:Object.fromEntries([...circles].map(([id,info])=>[id,{...info,center:circleCenters.get(id)||null,points:[...(circlePoints.get(id)||[])]}])),limitations,assumptionReport,stats:{points:points.size,predicates:predicates.length,virtuals:[...points.keys()].filter(s=>s.startsWith('@')).length}};
}
return {compile,seg,lp};
});
