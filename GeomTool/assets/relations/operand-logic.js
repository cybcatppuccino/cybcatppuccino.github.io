/* v4.1 operand-grounded facts. Every subfact names actual vertices/metrics;
 * type-name implications alone never erase a numerical observation. */
(function(root,factory){const api=factory();root.GeomRelationOperandLogic=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const pair=(a,b)=>[a,b].sort().join('|');
const distance=(a,b)=>'D:'+pair(a,b);
const quadKinds=new Set(['square','rectangle','rhombus','parallelogram','kite']);
function quadrilateralFacts(e){
 if(!quadKinds.has(e.kind))return [];
 const edges=e.marks.filter(m=>m.type==='segment');if(edges.length<4)return [];
 const v=[edges[0].a,edges[0].b,edges[1].b,edges[2].b];
 if(new Set(v).size!==4||edges[3].a!==v[3]||edges[3].b!==v[0])return [];
 const [a,b,c,d]=v,side=[distance(a,b),distance(b,c),distance(c,d),distance(d,a)];
 const facts=[],add=(kind,operands,alias)=>facts.push({kind,operands,alias});
 const eq=(i,j)=>add('equalLength',[side[i],side[j]],`${v[i]}${v[(i+1)%4]}=${v[j]}${v[(j+1)%4]}`);
 const parallel=(i,j)=>add('parallel',[side[i],side[j]],`${v[i]}${v[(i+1)%4]}∥${v[j]}${v[(j+1)%4]}`);
 const cong=(lhs,rhs)=>add('congruentTriangles',[lhs,rhs],`△${lhs}≅△${rhs}`);
 const bisect=()=>add('diagonalsBisect',[distance(a,c),distance(b,d)],`${a}${c}与${b}${d}互相平分`);
 const angles=()=>{for(let i=0;i<4;i++)add('rightAngle',[v[(i+3)%4],v[i],v[(i+1)%4]],`∠${v[(i+3)%4]}${v[i]}${v[(i+1)%4]}=90°`);};
 if(e.kind!=='kite'){
  parallel(0,2);parallel(1,3);eq(0,2);eq(1,3);bisect();
  cong(`${a}${b}${c}`,`${c}${d}${a}`);cong(`${b}${c}${d}`,`${d}${a}${b}`);
  // A diagonal of a parallelogram bisects its area, and opposite interior
  // angles are equal; neither conclusion requires that the sides be equal.
  add('equalArea',[`${a}${b}${c}`,`${a}${c}${d}`],`[${a}${b}${c}]=[${a}${c}${d}]`);
  add('equalArea',[`${a}${b}${d}`,`${b}${c}${d}`],`[${a}${b}${d}]=[${b}${c}${d}]`);
  add('equalAngle',[`${d}${a}${b}`,`${b}${c}${d}`],`∠${d}${a}${b}=∠${b}${c}${d}`);
  add('equalAngle',[`${a}${b}${c}`,`${c}${d}${a}`],`∠${a}${b}${c}=∠${c}${d}${a}`);
  for(let i=0;i<4;i++){
   const u=v[(i+3)%4],x=v[i],y=v[(i+1)%4],z=v[(i+2)%4];
   add('supplementaryAngles',[`${u}${x}${y}`,`${x}${y}${z}`],`∠${u}${x}${y}+∠${x}${y}${z}=180°`);
  }
 }
 if(['square','rectangle'].includes(e.kind)){
  angles();add('equalLength',[distance(a,c),distance(b,d)],`${a}${c}=${b}${d}`);
  add('concyclic',v.slice(),`${v.join('')}共圆`);
 }
 if(['square','rhombus'].includes(e.kind)){
  eq(0,1);eq(1,2);eq(2,3);eq(3,0);
  add('perpendicular',[distance(a,c),distance(b,d)],`${a}${c}⊥${b}${d}`);
  // Both diagonals bisect the opposite vertex angles in a rhombus.
  for(const [u,x,z,y] of [[b,a,d,c],[b,c,d,a],[a,b,c,d],[a,d,c,b]]){
   add('angleBisector',[`${u}${x}${y}`,`${y}${x}${z}`],`${x}${y}平分∠${u}${x}${z}`);
  }
 }
 if(e.kind==='square'){
  add('equalLength',[distance(a,c),distance(b,d)],`${a}${c}=${b}${d}`);
 }
 if(e.kind==='kite'){
  const first=distance(a,b),second=distance(b,c),third=distance(c,d),fourth=distance(d,a);
  // The two possible axes differ: connecting the vertices where the
  // equal adjacent sides meet. Detect from actual equal-length witnesses.
  // Geometry of a valid kite uniquely identifies the two pairs by lengths.
  const E=e.kiteAxis==='bd'?'bd':'ac';
  if(E==='bd'){
   add('equalLength',[first,second],`${a}${b}=${b}${c}`);add('equalLength',[third,fourth],`${c}${d}=${d}${a}`);
   add('perpendicular',[distance(a,c),distance(b,d)],`${a}${c}⊥${b}${d}`);
   add('diagonalBisector',[distance(b,d),distance(a,c)],`${b}${d}平分${a}${c}`);
   cong(`${b}${a}${d}`,`${b}${c}${d}`);
   add('equalAngle',[`${b}${a}${d}`,`${b}${c}${d}`],`∠${b}${a}${d}=∠${b}${c}${d}`);
   add('equalArea',[`${b}${a}${d}`,`${b}${c}${d}`],`[${b}${a}${d}]=[${b}${c}${d}]`);
   add('angleBisector',[`${a}${b}${d}`,`${d}${b}${c}`],`${b}${d}平分∠${a}${b}${c}`);
   add('angleBisector',[`${a}${d}${b}`,`${b}${d}${c}`],`${d}${b}平分∠${a}${d}${c}`);
  }else{
   add('equalLength',[first,fourth],`${a}${b}=${d}${a}`);add('equalLength',[second,third],`${b}${c}=${c}${d}`);
   add('perpendicular',[distance(a,c),distance(b,d)],`${a}${c}⊥${b}${d}`);
   add('diagonalBisector',[distance(a,c),distance(b,d)],`${a}${c}平分${b}${d}`);
   cong(`${a}${b}${c}`,`${a}${d}${c}`);
   add('equalAngle',[`${a}${b}${c}`,`${a}${d}${c}`],`∠${a}${b}${c}=∠${a}${d}${c}`);
   add('equalArea',[`${a}${b}${c}`,`${a}${d}${c}`],`[${a}${b}${c}]=[${a}${d}${c}]`);
   add('angleBisector',[`${b}${a}${c}`,`${c}${a}${d}`],`${a}${c}平分∠${b}${a}${d}`);
   add('angleBisector',[`${b}${c}${a}`,`${a}${c}${d}`],`${c}${a}平分∠${b}${c}${d}`);
  }
 }
 // Exact operand duplicates are a single fact, not extra discoveries.
 const seen=new Set();return facts.filter(f=>{const k=f.kind+':'+f.operands.slice().sort().join('|');if(seen.has(k))return false;seen.add(k);return true;});
}
function directLengthGraph(snapshot){
 const nodes=new Map((snapshot.construction||snapshot.constructions||[]).map(n=>[n.id,n]));
 const parent=new Map(),reasons=[];
 const find=x=>{if(!parent.has(x))parent.set(x,x);const p=parent.get(x);if(p!==x)parent.set(x,find(p));return parent.get(x);};
 const join=(a,b,by)=>{if(!a||!b||a===b)return;const x=find(a),y=find(b);if(x!==y)parent.set(x,y);reasons.push({a,b,node:by});};
 const center=new Map();for(const n of nodes.values())if(['circle','circleRadius','circleBySegment'].includes(n.op))center.set(n.id,n.inputs?.[0]);
 for(const q of snapshot.circles||[])if(q.centerId)center.set(q.id,q.centerId);
 for(const n of nodes.values()){
  const [a,b,c]=n.inputs||[];
  if(n.op==='midpoint'&&a&&b)join(distance(n.id,a),distance(n.id,b),n.id);
  if(n.op==='circumcenter'&&a&&b&&c){join(distance(n.id,a),distance(n.id,b),n.id);join(distance(n.id,a),distance(n.id,c),n.id);}
  if(n.op==='circle'&&a&&b)join('R:'+n.id,distance(a,b),n.id);
  if(n.op==='circleBySegment'&&b&&c)join('R:'+n.id,distance(b,c),n.id);
  if(n.op==='onCircle'&&a&&center.get(a))join('R:'+a,distance(center.get(a),n.id),n.id);
  if(n.op==='diameterCircle'&&a&&b){ // Diameter, not radius, is AB: keep its 1/2 ratio outside equality DSU.
  }
  // A circle through a named constructed intersection has the same radius
  // as the center-to-intersection segment, irrespective of branch.
  if(['intersection','secondIntersection'].includes(n.op))for(const carrier of n.inputs||[]){
   if(center.get(carrier))join('R:'+carrier,distance(center.get(carrier),n.id),n.id);
  }
 }
 return {same:(a,b)=>parent.has(a)&&parent.has(b)&&find(a)===find(b),reasons};
}
function parallelAreaFact(c,triA,triB){
 const aa=triA.split(/[|,]/),bb=triB.split(/[|,]/);if(aa.length!==3||bb.length!==3)return false;
 const shared=aa.filter(x=>bb.includes(x));if(shared.length!==2)return false;
 const p=c.pMap.get(shared[0]),q=c.pMap.get(shared[1]),u=c.pMap.get(aa.find(x=>!shared.includes(x))),v=c.pMap.get(bb.find(x=>!shared.includes(x)));
 if(!p||!q||!u||!v)return false;
 const base={x:q.x-p.x,y:q.y-p.y},far={x:v.x-u.x,y:v.y-u.y};
 const denom=Math.hypot(base.x,base.y)*Math.hypot(far.x,far.y);
 return denom>c.eps*c.eps&&Math.abs(base.x*far.y-base.y*far.x)<=c.mode.direction*denom;
}
function annotate(entries,snapshot,c){
 const direct=directLengthGraph(snapshot),nodes=snapshot.construction||snapshot.constructions||[];
 for(const e of entries){
  if(e.kind==='harmonic'&&e.harmonicPairing?.length===2){
   const [[a,b],[d,z]]=e.harmonicPairing;
   const product=(x,y,u,v)=>'M:'+[distance(x,y),distance(u,v)].sort().join('*');
   const first=product(a,d,b,z),second=product(a,z,b,d);
   const alias=`${a}${d}·${b}${z}=${a}${z}·${b}${d}`;
   e.derivedFacts=[...(e.derivedFacts||[]),{kind:'equalProduct',operands:[first,second],alias}];
   e.searchTags=[...(e.searchTags||[]),alias,'二次量连等','乘积相等'];
  }
  if(quadKinds.has(e.kind)){
   if(e.kind==='kite'){
    const es=e.marks.filter(m=>m.type==='segment'),v=[es[0]?.a,es[0]?.b,es[1]?.b,es[2]?.b];
    const s=v.map((p,i)=>{const a=c.pMap.get(p),b=c.pMap.get(v[(i+1)%4]);return a&&b?Math.hypot(a.x-b.x,a.y-b.y):NaN;});
    e.kiteAxis=c.near(s[0],s[1])&&c.near(s[2],s[3])?'bd':'ac';
   }
   e.derivedFacts=[...(e.derivedFacts||[]),...quadrilateralFacts(e)];
   e.searchTags=[...(e.searchTags||[]),...e.derivedFacts.map(f=>f.alias),...e.derivedFacts.map(f=>f.kind)];
   const quadPts=new Set(e.marks.filter(m=>m.type==='point').map(m=>m.id));
   if(nodes.some(n=>n.op==='parallelogram'&&[n.id,...(n.inputs||[])].every(id=>quadPts.has(id)))){
    e.source='construction';e.origin='construction';e.constructionWitnesses=[...(e.constructionWitnesses||[]),{op:'parallelogram',reason:'按平行四边形构造定义'}];
   }
  }
  if(!e.metric)continue;
  if(e.metric.factor===1&&['distanceRatio','quadraticRatio'].includes(e.kind)){
   const all=(e.metric.high||[]).filter(m=>!m.redundant);
   const translate=k=>k.startsWith('S:')?k.slice(2):k.startsWith('R2:')?'R:'+k.slice(3):k;
   const matched=[];
   for(let i=0;i<all.length;i++)for(let j=i+1;j<all.length;j++)if(direct.same(translate(all[i].key),translate(all[j].key)))matched.push([all[i].key,all[j].key]);
   if(matched.length){
    e.directMetricPairs=matched;
    const fullyCovered=all.length>=2&&all.every((m,i)=>i===0||direct.same(translate(m.key),translate(all[0].key)));
    if(fullyCovered){e.source='construction';e.origin='construction';}
    else {e.source='mixed';e.origin='construction';}
    e.searchTags=[...(e.searchTags||[]),'半径构造等长','作图自带关系'];
    (e.verifiedConsequences??=[]).push({node:'length-definition',kind:'equalLengths',operand:{pairs:matched},level:'consequence',rule:'构造半径/中点等长闭包'});
   }
  }
  if(e.kind==='quadraticRatio'&&e.algebraicSubstitutions>0&&e.algebraicOnly&&e.source!=='construction'){
   e.source=e.source==='mixed'?'mixed':'entailed';e.searchTags=[...(e.searchTags||[]),'一次等长关系的乘法代换'];
  }
  if(e.kind==='quadraticRatio'&&e.algebraicSubstitutions>0&&!e.algebraicOnly&&e.source==='discovery')e.source='mixed';
  if(e.kind==='quadraticRatio'&&e.source==='discovery'&&
     e.quadraticTypes?.every(t=>['lengthSquared','product','dot'].includes(t))){
   const ids=new Set(e.marks.filter(m=>m.type==='segment').flatMap(m=>[m.a,m.b]));
   // In a one-dimensional 3/4-point construction, dot products collapse to
   // signed length products. The numeric equalities are searchable, but must
   // not outrank the underlying linear proportions/harmonic condition.
   if(ids.size>=3&&ids.size<=4&&c.lineGroups.groups.some(g=>[...ids].every(id=>g.pointIds.includes(id)))){
    e.source='entailed';e.searchTags=[...(e.searchTags||[]),'共线后的一维代数重写'];
   }
  }

 }
 // Equal-area groups with common bases whose opposite vertices lie on a
 // line parallel to that base. Suppress no result if just ONE pair fits.
 for(const e of entries)if(e.kind==='equalArea'){
  const tris=e.id.slice('equalArea:'.length).split(';');if(tris.length<2)continue;
  const matched=[];for(let i=0;i<tris.length;i++)for(let j=i+1;j<tris.length;j++)if(parallelAreaFact(c,tris[i],tris[j]))matched.push([tris[i],tris[j]]);
  if(matched.length){e.entailedByParallel=matched;e.searchTags=[...(e.searchTags||[]),'平行线同底等高'];
   // The parallelism itself can be a discovery: do not call this a direct
   // CONSTRUCTION unless that geometric constraint is constructor-known.
   if(matched.length===tris.length*(tris.length-1)/2)e.source=e.source==='construction'?'construction':'entailed';
   else if(e.source==='discovery')e.source='mixed';
  }
 }
}
function reduce(entries,c){
 const removed=new Set(),strong=entries.filter(e=>quadKinds.has(e.kind)&&e.derivedFacts?.length);
 // A harmonic quadruple implies precisely one cross-multiplied absolute
 // product equality. Suppress only that WHOLE two-member equation; if other
 // independent metrics join the equality chain it must remain visible.
 for(const h of entries.filter(e=>e.kind==='harmonic'&&e.derivedFacts?.length)){
  const operands=h.derivedFacts.find(f=>f.kind==='equalProduct')?.operands;
  if(!operands)continue;
  for(const w of entries.filter(e=>e.kind==='quadraticRatio'&&e.metric?.factor===1)){
   const metrics=w.metric.high||[],matched=operands.every(k=>metrics.some(m=>m.key===k));if(!matched)continue;
   (h.entails??=[]).push({id:w.id,kind:w.kind,reason:'调和四点交比等于负一蕴含交叉线段乘积相等'});
   if(metrics.length===2){h.searchTags=[...(h.searchTags||[]),w.title,...(w.searchTags||[])];removed.add(w.id);}
   else if(w.source==='discovery')w.source='mixed';
  }
 }

 for(const weak of entries){if(weak.kind!=='distanceRatio'||weak.metric?.factor!==1)continue;
  const metrics=weak.metric.high||[];if(metrics.length<2||metrics.some(x=>x.type!=='segment'))continue;
  for(const parent of strong){const adjacency=new Map(),add=(a,b)=>{if(!adjacency.has(a))adjacency.set(a,new Set());adjacency.get(a).add(b);};
   for(const f of parent.derivedFacts)if(f.kind==='equalLength'){const [a,b]=f.operands;add(a,b);add(b,a);}
   const key=metrics[0].key,visited=new Set([key]),todo=[key];while(todo.length){for(const nxt of adjacency.get(todo.pop())||[])if(!visited.has(nxt)){visited.add(nxt);todo.push(nxt);}}
   if(!metrics.every(x=>visited.has(x.key)))continue;
   (parent.entails??=[]).push({id:weak.id,kind:weak.kind,reason:'四边形同一顶点组的实际边长蕴含'});
   parent.searchTags=[...(parent.searchTags||[]),weak.title,...(weak.searchTags||[])];
   removed.add(weak.id);break;
  }
 }
 return {entries:entries.filter(e=>!removed.has(e.id)),suppressed:removed.size};
}
function decorateGraph(g,entries){
 for(const e of entries){if(!e.derivedFacts?.length)continue;
  e.derivedFacts.forEach((f,i)=>{
   const id=`F:v41:${e.id}:${i}`;
   g.nodes.push({id,kind:f.kind,origin:'logical-implication',operands:f.operands,alias:f.alias});
   g.edges.push({from:'R:'+e.id,to:id,rule:'OPERAND_SPECIFIC_QUADRILATERAL',category:'implication'});
  });
 }
 return g;
}
return {quadrilateralFacts,directLengthGraph,parallelAreaFact,annotate,reduce,decorateGraph};
});
