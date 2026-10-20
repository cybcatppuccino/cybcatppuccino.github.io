/* GeomTool v8.2: model preparation and numeric predicates.
 * Independent of DOM and Kernel. All temporary AB lines keep their endpoint IDs.
 * All numerical discovery results are hypotheses, not symbolic proofs.
 */
(function(root,factory){const api=factory();root.GeomRelationCore=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const groupsApi=(typeof globalThis!=='undefined'&&globalThis.GeomRelationGroups)||(typeof require==='function'?require('./groups.js'):null);
const V={sub:(a,b)=>({x:a.x-b.x,y:a.y-b.y}),add:(a,b)=>({x:a.x+b.x,y:a.y+b.y}),mul:(a,t)=>({x:a.x*t,y:a.y*t}),dot:(a,b)=>a.x*b.x+a.y*b.y,cross:(a,b)=>a.x*b.y-a.y*b.x,dist:(a,b)=>Math.hypot(a.x-b.x,a.y-b.y)};
const RATIOS=[{v:1,label:'1'},{v:2,label:'2'},{v:3,label:'3'},{v:4,label:'4'},{v:1.5,label:'3/2'},{v:4/3,label:'4/3'},{v:Math.SQRT2,label:'√2'},{v:Math.sqrt(3),label:'√3'},{v:(1+Math.sqrt(5))/2,label:'φ'}];
const MODES=Object.freeze({strict:Object.freeze({distance:1e-9,direction:8e-10,angle:5e-8,ratio:1e-9,quadratic:5e-11})});
const finite=n=>Number.isFinite(n),sortIds=a=>a.slice().sort().join('|');
const pMark=id=>({type:'point',id});
const sMark=(a,b)=>({type:'segment',a,b});
const lineMark=l=>({type:'line',ref: l.kind==='implicit'?{type:'pair',a:l.a,b:l.b}:{type:'object',id:l.id}});
const circleMark=c=>({type:'circle',ref:c.kind==='implicit'?{type:'centerThrough',a:c.a,b:c.b}:{type:'object',id:c.id}});
const arcMark=(a,v,b,label)=>({type:'angle',a,v,b,label});
const distanceMark=d=>d.type==='segment'?sMark(d.a,d.b):{type:'distance',point:d.point,line:d.line.kind==='implicit'?{type:'pair',a:d.line.a,b:d.line.b}:{type:'object',id:d.line.id}};
function makeContext(snapshot,options={}){
 const mode=MODES.strict;
 const rawP=(snapshot.points||[]).filter(p=>finite(p.x)&&finite(p.y)).map(p=>({...p})).sort((a,b)=>a.id.localeCompare(b.id));
 const P=[];
 let pMap;
 const coords=[...rawP]; for(const c of snapshot.circles||[])if(finite(c.cx)&&finite(c.cy)&&finite(c.r)&&c.r>0)coords.push({x:c.cx-c.r,y:c.cy-c.r},{x:c.cx+c.r,y:c.cy+c.r});
 const span=coords.length?Math.max(1,Math.hypot(Math.max(...coords.map(p=>p.x))-Math.min(...coords.map(p=>p.x)),Math.max(...coords.map(p=>p.y))-Math.min(...coords.map(p=>p.y)))):1;
 // A single remote outlier must not loosen tolerances throughout a compact
 // construction. Median nearest-neighbour gap is an inexpensive robust local
 // geometry scale; without neighbours fall back to the extents/circle radii.
 const nearest=[];
 for(let i=0;i<rawP.length;i++){let best=Infinity;for(let j=0;j<rawP.length;j++){
  if(i===j)continue;const gap=V.dist(rawP[i],rawP[j]);if(gap>0&&gap<best)best=gap;
 }if(Number.isFinite(best))nearest.push(best);}
 nearest.sort((a,b)=>a-b);
 const radii=(snapshot.circles||[]).map(c=>c.r).filter(x=>Number.isFinite(x)&&x>0).sort((a,b)=>a-b);
 const localScale=nearest.length?nearest[Math.floor((nearest.length-1)*.75)]:radii.length?radii[(radii.length-1)>>1]:span;
 const origins=rawP.map(p=>Math.max(Math.abs(p.x),Math.abs(p.y))).sort((a,b)=>a-b);
 // Scale with the median actual geometry rather than assuming one world unit.
 // Compensate for IEEE-754 coordinate ulps when the entire drawing is
 // translated far from the origin, but never let one extreme point inflate
 // the tolerances for all other objects.
 const originMagnitude=origins.length?origins[(origins.length-1)>>1]:0;
 const eps=Math.max(mode.distance*Math.max(localScale,1e-9),32*Number.EPSILON*Math.max(1,originMagnitude));
 const pointGroups=groupsApi.clusterPoints(rawP,eps);
 P.push(...pointGroups.points);
 pMap=new Map(P.map(p=>[p.id,p]));
 const E=[];for(const l of snapshot.lines||[]){const x=l.x2-l.x1,y=l.y2-l.y1,len=Math.hypot(x,y);if(!(len>eps)||!finite(len))continue;E.push({kind:'explicit',id:l.id,key:'L:'+l.id,name:l.label||l.id,x:l.x1,y:l.y1,dx:x/len,dy:y/len,len,extent:l.drawType||'line'});}
 const VLines=[],segments=[],byPoint=new Map(P.map(p=>[p.id,[]]));
 for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++){
  const a=P[i],b=P[j],len=V.dist(a,b);if(!(len>eps))continue;
  const l={kind:'implicit',a:a.id,b:b.id,key:'P:'+a.id+'|'+b.id,name:a.id+b.id,x:a.x,y:a.y,dx:(b.x-a.x)/len,dy:(b.y-a.y)/len,len,extent:'line'};
  VLines.push(l);const d={type:'segment',a:a.id,b:b.id,key:'D:'+a.id+'|'+b.id,value:len,marks:[sMark(a.id,b.id)]};segments.push(d);byPoint.get(a.id).push(l);byPoint.get(b.id).push(l);
 }
 const rawLines=E.concat(VLines), C=[];
 for(const c of snapshot.circles||[]){if(!finite(c.cx)||!finite(c.cy)||!finite(c.r)||c.r<=eps)continue;C.push({kind:'explicit',id:c.id,key:'C:'+c.id,name:c.label||c.id,x:c.cx,y:c.cy,r:c.r,centerId:c.centerId||null});}
 const distPointLine=(p,l)=>Math.abs((p.x-l.x)*l.dy-(p.y-l.y)*l.dx);
 const signedPointLine=(p,l)=>(p.x-l.x)*l.dy-(p.y-l.y)*l.dx;
 const project=(p,l)=>{const t=(p.x-l.x)*l.dx+(p.y-l.y)*l.dy;return {x:l.x+t*l.dx,y:l.y+t*l.dy,t}};
 const crossLine=(a,b)=>a.dx*b.dy-a.dy*b.dx, dotLine=(a,b)=>a.dx*b.dx+a.dy*b.dy;
 const sameLine=(a,b)=>Math.abs(crossLine(a,b))<=mode.direction&&distPointLine({x:b.x,y:b.y},a)<=eps;
 // Prepass 2: canonical infinite support lines with complete alias and incidence maps.
 const lineGroups=groupsApi.groupLines(rawLines,P,eps,mode.direction,distPointLine);
 const L=lineGroups.lines,lineFamily=lineGroups.byKey;
 const EUnique=E.filter(line=>lineFamily.get(line.key)===line.key);
 // Prepass 3: canonical circles (including virtual circles defined by >=4 distinct points).
 function circumPlaceholder(a,b,c){const x=b.x-a.x,y=b.y-a.y,u=c.x-a.x,v=c.y-a.y,d=2*(x*v-y*u);if(Math.abs(d)<=eps*Math.max(V.dist(a,b),V.dist(a,c)))return null;const e=x*x+y*y,f=u*u+v*v,cx=a.x+(e*v-f*y)/d,cy=a.y+(f*x-e*u)/d,r=Math.hypot(a.x-cx,a.y-cy);return finite(r)?{x:cx,y:cy,r}:null;}
 const circleGroups=groupsApi.groupCircles(P,C,circumPlaceholder,eps,span,options.circlePrepass!==false);
 // Explicit circles are reused by all downstream predicates; preserve raw C for metadata.
 const canonicalC=circleGroups.explicit;
 const aligned=(a,b)=>Math.abs(crossLine(a,b))<=mode.direction;
 const perpendicular=(a,b)=>Math.abs(dotLine(a,b))<=mode.direction;
 const angle=(a,v,b)=>{const x=V.sub(a,v),y=V.sub(b,v),n=Math.hypot(x.x,x.y)*Math.hypot(y.x,y.y);return n<=eps*eps?NaN:Math.atan2(Math.abs(V.cross(x,y)),V.dot(x,y))*180/Math.PI;};
 const triangle=(a,b,c)=>Math.abs(V.cross(V.sub(b,a),V.sub(c,a)))/2;
 const circum=(a,b,c)=>{const x=b.x-a.x,y=b.y-a.y,u=c.x-a.x,v=c.y-a.y,d=2*(x*v-y*u);if(Math.abs(d)<=eps*Math.max(V.dist(a,b),V.dist(a,c)))return null;const e=x*x+y*y,f=u*u+v*v,cx=a.x+(e*v-f*y)/d,cy=a.y+(f*x-e*u)/d,r=Math.hypot(a.x-cx,a.y-cy);return finite(r)?{x:cx,y:cy,r}:null;};
 const ids=xs=>xs.map(id=>pMap.get(id));
 // Semantic roles remain separate: implicit AB and explicit l may coexist.
 const labels=new Map([...rawP,...E,...C].map(x=>[x.id,x.label||x.name||x.id]));
 const display=id=>labels.get(id)||id;
 const lname=l=>{const name=l.kind==='implicit'?display(l.a)+display(l.b):display(l.id);
  const alias=l.aliases?.find(a=>a.kind==='implicit');
  return l.aliases?.length?name+' ('+l.aliases.slice(0,2).map(a=>a.kind==='implicit'?display(a.a)+display(a.b):display(a.id)).join(', ')+(l.aliases.length>2?', …':'')+')':name;};
 const cname=c=>c.kind==='implicit'?`⊙(${display(c.a)},${display(c.b)})`:display(c.id);
 const distances=(options.kinds===undefined||options.kinds.includes('distanceRatio'))?segments.slice():[];
 // Circle radius is an honest length-valued operand. Keep EACH circle's ID,
 // including multiple explicit objects representing the same geometric circle.
 if(options.kinds===undefined||options.kinds.includes('distanceRatio'))for(const q of C){
  distances.push({type:'radius',circle:q,key:'R:'+q.id,value:q.r,marks:[circleMark(q)]});
 }
 // point-to-line distances: no endpoints/zero length; let implicit AB act as the supporting line.
 if(options.kinds===undefined||options.kinds.includes('distanceRatio'))for(const p of P)for(const l of L){if(l.kind==='implicit'&&(l.a===p.id||l.b===p.id))continue;const value=distPointLine(p,l);if(value<=eps)continue;distances.push({type:'pointLine',point:p.id,line:l,key:'H:'+p.id+'|'+l.key,value,marks:[distanceMark({type:'pointLine',point:p.id,line:l})]});}
 distances.sort((a,b)=>a.value-b.value);
 const seen=new Set(),entries=[];
 const lineGroupByKey=new Map(lineGroups.groups.flatMap(g=>g.members.map(x=>[x.key,g])));
 const circleGroupByKey=new Map(circleGroups.groups.flatMap(g=>g.members.map(x=>[x.key,g])));
 const selected=new Set(options.kinds||[]);
 function add(kind,key,title,marks,score=50,detail='',extraAliases=[]){
  if(selected.size&&!selected.has(kind))return;
  const id=kind+':'+key;if(seen.has(id))return;
  seen.add(id);
  const tokens=new Set();
  const pointAlias=(pid)=>{for(const p of pointGroups.aliases.get(pointGroups.identity.get(pid)||pid)||[])tokens.add(p.label||p.id);};
  const inspectRef=ref=>{if(!ref)return;if(ref.type==='pair'){pointAlias(ref.a);pointAlias(ref.b);const key='P:'+sortIds([ref.a,ref.b]);const g=lineGroupByKey.get(key);if(g)for(const x of g.members){if(x.id)tokens.add(labels.get(x.id)||x.id);else {pointAlias(x.a);pointAlias(x.b);tokens.add(display(x.a)+display(x.b));}}}else if(ref.type==='centerThrough'){pointAlias(ref.a);pointAlias(ref.b);}else if(ref.type==='object'){tokens.add(labels.get(ref.id)||ref.id);const l=lineGroupByKey.get('L:'+ref.id);if(l)for(const x of l.members){if(x.id)tokens.add(labels.get(x.id)||x.id);else{pointAlias(x.a);pointAlias(x.b);tokens.add(display(x.a)+display(x.b));}}const circle=circleGroupByKey.get('C:'+ref.id);if(circle)for(const x of circle.members)tokens.add(labels.get(x.id)||x.id);}};
  for(const m of marks||[]){for(const field of ['id','a','b','v','point'])if(typeof m[field]==='string')pointAlias(m[field]);if(m.ref)inspectRef(m.ref);if(m.line)inspectRef(m.line);if(m.circleRef)inspectRef(m.circleRef);if(m.lineRef)inspectRef(m.lineRef);if(m.ids)for(const pid of m.ids)pointAlias(pid);}
  for(const alias of extraAliases)tokens.add(alias);
  entries.push({id,kind,title,marks:marks||[],score,detail,searchText:[...tokens].join(' ')});
 }
 function near(a,b,scale=Math.max(Math.abs(a),Math.abs(b),localScale*.01)){return Math.abs(a-b)<=Math.max(eps,mode.ratio*scale);}
 function nearRatio(a,b,v){return b>eps&&Math.abs(a/b-v)<=mode.ratio*Math.max(1,v)+eps/b;}
 // Squared-distance, area and point-power comparisons must never reuse a
 // length-valued absolute tolerance. Estimate their error using local operand
 // magnitudes (not the scene-wide bounding box, which may have outliers).
 function nearQuadratic(a,b,scale=Math.max(Math.abs(a),Math.abs(b))){
  return Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=mode.quadratic*Math.max(scale,eps*eps)+32*Number.EPSILON*Math.max(Math.abs(a),Math.abs(b));
 }
 function nearQuadraticRatio(a,b,r){
  return Math.abs(b)>eps*eps&&nearQuadratic(a,b*r,Math.max(Math.abs(a),Math.abs(b*r)));
 }
 function binaryLower(n){let a=0,b=distances.length;while(a<b){const m=(a+b)>>1;if(distances[m].value<n)a=m+1;else b=m;}return a;}
 function binaryUpper(n){let a=0,b=distances.length;while(a<b){const m=(a+b)>>1;if(distances[m].value<=n)a=m+1;else b=m;}return a;}
 return {P,rawP,pMap,pointGroups,lineGroups,circleGroups,E,EUnique,VLines,L,lineFamily,C:canonicalC,CAll:C,segments,distances,byPoint,mode,eps,span,localScale,options,entries,selected,add,near,nearRatio,nearQuadratic,nearQuadraticRatio,binaryLower,binaryUpper,display,lname,cname,pMark,sMark,lineMark,circleMark,arcMark,distanceMark,ids,distPointLine,signedPointLine,project,crossLine,dotLine,sameLine,aligned,perpendicular,angle,triangle,circum};
}
return {V,RATIOS,MODES,makeContext,sortIds,pMark,sMark,lineMark,circleMark,arcMark,distanceMark};
});
