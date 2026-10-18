/* GeomTool v8 relational detector registry.
 * Each rule is independently selectable and returns semantic witnesses + drawable marks.
 */
(function(root,factory){const core=root.GeomRelationCore||(typeof require==='function'?require('./core.js'):null);const api=factory(core);root.GeomRelationEngine=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(core){
'use strict';
const {V,RATIOS,sortIds,pMark,sMark,lineMark,circleMark,arcMark,distanceMark,makeContext}=core;
const registry=[];
const provenance=(typeof globalThis!=='undefined'&&globalThis.GeomRelationProvenance)||(typeof require==='function'?require('./provenance.js'):null);
const semantic=(typeof globalThis!=='undefined'&&globalThis.GeomRelationSemantic)||(typeof require==='function'?require('./semantic.js'):null);
const operandLogic=(typeof globalThis!=='undefined'&&globalThis.GeomRelationOperandLogic)||(typeof require==='function'?require('./operand-logic.js'):null);
const quadratic=(typeof globalThis!=='undefined'&&globalThis.GeomRelationQuadratic)||(typeof require==='function'?require('./quadratic.js'):null);
const triviality=(typeof globalThis!=='undefined'&&globalThis.GeomRelationTriviality)||(typeof require==='function'?require('./triviality.js'):null);
function rule(id,label,group,help,run,enabled=true){registry.push({id,label,group,help,run,enabled});}
const points=ids=>ids.map(pMark),pname=(c,p)=>c.display(p.id),pid=(c,p)=>c.display(p),triName=(c,t)=>t.ids.map(c.display).join('');
const same=(a,b)=>a.id===b.id;
const lineRefs=(a,b)=>[lineMark(a),lineMark(b)];
const triMarks=t=>[sMark(t.ids[0],t.ids[1]),sMark(t.ids[1],t.ids[2]),sMark(t.ids[2],t.ids[0]),...points(t.ids)];
const pairs=(arr,fn)=>{for(let i=0;i<arr.length;i++)for(let j=i+1;j<arr.length;j++)fn(arr[i],arr[j],i,j);};
const cmp=(a,b)=>a-b;
function makeTriangles(c){const out=[],P=c.P;
 for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++)for(let k=j+1;k<P.length;k++){
  const a=P[i],b=P[j],d=P[k],area=c.triangle(a,b,d);if(area<=c.mode.direction*V.dist(a,b)*V.dist(a,d)*.5)continue;
  const sides=[V.dist(b,d),V.dist(a,d),V.dist(a,b)],sorted=sides.slice().sort(cmp),length=sorted[2];
  out.push({ids:[a.id,b.id,d.id],pts:[a,b,d],sides,sorted,area,circ:c.circum(a,b,d)});
 }
 return out;
}
function circleCandidates(c){
 const out=c.C.slice(),byCenter=new Map();
 for(const a of c.P)for(const b of c.P){if(a===b)continue;
  const r=V.dist(a,b);if(r<=c.eps)continue;
  const list=byCenter.get(a.id)||[];
  let existing=list.find(g=>Math.abs(g.r-r)<=c.eps);
  if(existing){existing.aliases.push({a:a.id,b:b.id});continue;}
  existing=out.find(g=>Math.hypot(g.x-a.x,g.y-a.y)<=c.eps&&Math.abs(g.r-r)<=c.eps);
  if(existing){(existing.virtualAliases??=[]).push({a:a.id,b:b.id});continue;}
  const candidate={kind:'implicit',a:a.id,b:b.id,key:'K:'+a.id+'|'+b.id,x:a.x,y:a.y,r,aliases:[]};
  list.push(candidate);byCenter.set(a.id,list);out.push(candidate);
 }
 return out;
}
function targetInsideLine(c,l,p){if(l.kind==='implicit'||l.extent==='line')return true;const t=(p.x-l.x)*l.dx+(p.y-l.y)*l.dy;return l.extent==='ray'?t>=-c.eps:t>=-c.eps&&t<=l.len+c.eps;}
// ---------- incidence and line geometry ----------
rule('coincident','重合点簇','基础位置','将同一位置的全部不同点 ID 合并为一条事实。',c=>{
 for(const pts of c.pointGroups.groups){if(pts.length<2)continue;
 const ids=pts.map(p=>p.id);c.add('coincident',ids.join('|'),`${ids.map(c.display).join('、')} 重合`,points(ids),51,`共 ${ids.length} 个等价点`);
 }});
rule('collinear','最大共线点集','基础位置','先归并全部等价支撑直线，然后每条直线只报告一次最大共线点集。',c=>{
 for(const group of c.lineGroups.groups){const ids=group.pointIds;if(ids.length<3)continue;
 const names=ids.map(c.display).join('、');
 c.add('collinear',group.rep.key,`${names} 共线`,[lineMark(group.rep),...points(ids)],61,`共 ${ids.length} 个点；同线表示 ${group.members.length} 种`);
 }});
rule('pointOnLine','点在线上','基础位置','点在已有直线、线段或射线上（严格遵循对象范围）。',c=>{for(const p of c.P)for(const l of c.E){if(c.distPointLine(p,l)>c.eps||!targetInsideLine(c,l,p))continue;c.add('pointOnLine',p.id+'|'+c.lineFamily.get(l.key),`${c.display(p.id)} ∈ ${c.lname(l)}`,[pMark(p.id),lineMark(l)],36);}});
rule('pointOnCircle','点在圆上','基础位置','点与显式圆的半径相等。',c=>{for(const p of c.P)for(const q of c.C)if(c.near(V.dist(p,{x:q.x,y:q.y}),q.r))c.add('pointOnCircle',p.id+'|'+q.key,`${c.display(p.id)} 在圆 ${c.cname(q)} 上`,[pMark(p.id),circleMark(q)],36);});
rule('concyclic','最大共圆点集','基础位置','先归并现有圆及所有四点以上隐含圆，再按圆输出最大共圆点集。',c=>{
 for(const group of c.circleGroups.groups){const ids=group.pointIds;if(ids.length<4)continue;
 const cc=group.rep,marks=[];
 if(cc.kind==='explicit')marks.push(circleMark(cc));
 else {const triple=cc.key.slice(2).split('|');marks.push({type:'circumcircle',ids:triple});}
 marks.push(...points(ids));
 c.add('concyclic',cc.key,`${ids.map(c.display).join('、')} 共圆`,marks,78,`共 ${ids.length} 个点；同圆对象 ${group.members.length} 个`);
 }});
function directionClasses(c){
 if(c._directionClasses)return c._directionClasses;
 const fam=[];for(const l of c.L){let g=fam.find(group=>group.every(x=>c.aligned(x,l)));
  if(!g){g=[];fam.push(g);}g.push(l);
 }
 return c._directionClasses=fam;
}
const combinedNames=(c,lines)=>lines.length<=7?lines.map(l=>c.lname(l)).join(' ∥ '):lines.slice(0,7).map(l=>c.lname(l)).join(' ∥ ')+` …（${lines.length}条）`;
rule('parallel','平行线簇','线与角','把同方向的全部独立支撑线做一个最大簇，保留所有 AB/线对象表示供检索。',c=>{
 for(const fam of directionClasses(c))if(fam.length>1){
  c.add('parallel',sortIds(fam.map(l=>l.key)),combinedNames(c,fam),fam.map(lineMark),72,
   `同一方向共 ${fam.length} 条不同支撑线`,fam.map(l=>c.lname(l)));
 }
});
rule('perpendicular','垂直线簇','线与角','先归并同向线簇，再搜索垂直方向簇对；一个结果表达组内全部垂直关系。',c=>{
 const families=directionClasses(c);
 pairs(families,(a,b)=>{if(!c.perpendicular(a[0],b[0]))return;
  c.add('perpendicular',sortIds([...a,...b].map(l=>l.key)),`${combinedNames(c,a)} ⟂ ${combinedNames(c,b)}`,
   [...a,...b].map(lineMark),78,`两侧各 ${a.length}、${b.length} 条平行支撑线`,[...a,...b].map(l=>c.lname(l)));
 });
});
function lineIntersection(a,b){const den=V.cross({x:a.dx,y:a.dy},{x:b.dx,y:b.dy});if(Math.abs(den)<core.MODES.strict.direction)return null;
 const w={x:b.x-a.x,y:b.y-a.y},t=V.cross(w,{x:b.dx,y:b.dy})/den;
 return {x:a.x+a.dx*t,y:a.y+a.dy*t};}
function circleIntersections(a,b,eps){const dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);
 if(d<=eps||d>a.r+b.r+eps||d<Math.abs(a.r-b.r)-eps)return [];
 const along=(a.r*a.r-b.r*b.r+d*d)/(2*d),h2=a.r*a.r-along*along;
 if(h2< -eps*Math.max(1,a.r))return [];
 const mx=a.x+along*dx/d,my=a.y+along*dy/d,h=Math.sqrt(Math.max(0,h2));
 return h<=eps?[{x:mx,y:my}]:[{x:mx-h*dy/d,y:my+h*dx/d},{x:mx+h*dy/d,y:my-h*dx/d}];
}
rule('concurrent','非平凡多线共点','线与角','三条或更多独立支撑线在同一点交会：点可以尚未画出；排除同一端点生成的任意连线。',c=>{
 // At named points, a virtual line A-B counts only if neither endpoint
 // is the candidate point, preventing ubiquitous fabricated concurrency.
 const reported=new Set();const candidates=[];
 for(const p of c.P){const lines=c.lineGroups.groups.filter(g=>c.distPointLine(p,g.rep)<=c.eps&&
  g.members.some(l=>l.kind==='explicit'||(l.a!==p.id&&l.b!==p.id))).map(g=>g.rep);
  if(lines.length>=3){const key=sortIds(lines.map(l=>l.key));reported.add(key);
   c.add('concurrent','P:'+p.id+'|'+key,`${lines.map(l=>c.lname(l)).join('、')} 共点 ${c.display(p.id)}`,[pMark(p.id),...lines.map(lineMark)],88,`${lines.length} 条独立直线`);}
 }
 // Arbitrary line intersections need not coincide with a constructed point.
 // Bucket all pairwise intersection witnesses before considering a relation.
 const map=new Map(),pitch=c.eps*4,grid=(x,y)=>`${x},${y}`;
 for(let i=0;i<c.L.length;i++)for(let j=i+1;j<c.L.length;j++){
  const pos=lineIntersection(c.L[i],c.L[j]);if(!pos||!Number.isFinite(pos.x)||!Number.isFinite(pos.y))continue;
  if(c.P.some(p=>V.dist(p,pos)<=c.eps))continue;
  const bx=Math.floor(pos.x/pitch),by=Math.floor(pos.y/pitch);let g=null;
  for(let dx=-1;dx<=1&&!g;dx++)for(let dy=-1;dy<=1&&!g;dy++)
   g=(map.get(grid(bx+dx,by+dy))||[]).find(t=>V.dist(t.pos,pos)<=c.eps)||null;
  if(!g){g={pos,keys:new Set()};const key=grid(bx,by),list=map.get(key)||[];list.push(g);map.set(key,list);candidates.push(g);}
  g.keys.add(c.L[i].key);g.keys.add(c.L[j].key);
 }
 const byKey=new Map(c.L.map(l=>[l.key,l]));
 for(const g of candidates){const lines=[...g.keys].map(k=>byKey.get(k)).filter(l=>c.distPointLine(g.pos,l)<=c.eps);
  if(lines.length<3)continue;const key=sortIds(lines.map(l=>l.key));if(reported.has(key))continue;reported.add(key);
  c.add('concurrent','X:'+key,`${lines.map(l=>c.lname(l)).join('、')} 共点`,[...lines.map(lineMark),{type:'coordinate',x:g.pos.x,y:g.pos.y}],90,'交点不必预先创建');
 }
});
function rayIdentity(c,v,a){const key='P:'+sortIds([v.id,a.id]),fam=c.lineFamily.get(key)||key;
 const line=c.L.find(l=>l.key===fam);const delta=V.sub(a,v);
 const sign=line&&delta.x*line.dx+delta.y*line.dy<0?'-':'+';return fam+sign;
}
function pointAngleKey(c,a,v,b){return v.id+'|'+sortIds([rayIdentity(c,v,a),rayIdentity(c,v,b)]);}
function supportAngleKey(c,a,v,b){return sortIds([c.lineFamily.get('P:'+sortIds([a.id,v.id])),c.lineFamily.get('P:'+sortIds([v.id,b.id]))]);}
rule('fixedAngle','特殊角','线与角','点构成的有向射线夹角：15°、30°、45°、60°、75°、90°、120°、135°、150°。',c=>{const A=[15,30,45,60,75,90,120,135,150];for(const v of c.P)for(let i=0;i<c.P.length;i++)for(let j=i+1;j<c.P.length;j++){
 const a=c.P[i],b=c.P[j];if(a===v||b===v)continue;const deg=c.angle(a,v,b);for(const target of A)if(Math.abs(deg-target)<=c.mode.angle){
 if(target===90&&c.selected.has('perpendicular'))continue; // identical AB ⟂ BC fact
 const key=pointAngleKey(c,a,v,b);c.add('fixedAngle',key,`∠${pid(c,a.id)}${pid(c,v.id)}${pid(c,b.id)} = ${target}°`,[sMark(v.id,a.id),sMark(v.id,b.id),arcMark(a.id,v.id,b.id,target+'°')],66,target+'°');break;
 }
 }});
rule('equalAngle','等角簇','线与角','先归并平行方向对应的双线夹角，再把等角组成最大连等式；保留各表示的检索别名。',c=>{
 // All pairs of parallel lines describe the SAME unoriented angle. Using
 // representative direction pairs avoids O(m^4) redundant comparisons.
 const families=directionClasses(c);
 const classes=[];
 for(let i=0;i<families.length;i++)for(let j=i+1;j<families.length;j++){
  const a=families[i][0],b=families[j][0],value=acuteLineAngle(c,a,b);
  if(value<=c.mode.angle||value>90+c.mode.angle)continue;
  classes.push({a,b,value,key:'DIR:'+i+'|'+j,aliases:families[i].concat(families[j]).map(x=>c.lname(x))});
 }
 // Vertex angles above 90° preserve the two directed rays.
 const vertex=pointAngles(c).filter(a=>a.value>90+c.mode.angle).map(a=>({...a,aliases:[angleName(c,a)]}));
 function emit(values,detail){
  values.sort((a,b)=>a.value-b.value||a.key.localeCompare(b.key));
  const clusters=[];for(const x of values){let g=clusters[clusters.length-1];
   if(!g||x.value-g[0].value>c.mode.angle)clusters.push(g=[]);g.push(x);
  }
  for(const g of clusters)if(g.length>=2){
   const names=g.map(a=>angleName(c,a)),query=g.flatMap(a=>[...a.aliases,angleName(c,a)]);
   const head=names.length<=5?names.join(' = '):names.slice(0,5).join(' = ')+` …（共 ${names.length} 类等价角）`;
   c.add('equalAngle',g.map(a=>a.key).sort().join(';'),head,
    [...angleMarks(c,g[0]),...angleMarks(c,g[1])],75,detail,query);
   const entry=c.entries[c.entries.length-1];if(entry?.kind==='equalAngle'){
    entry.angleEqual=names;
    if(detail==='双线夹角'&&g.some((a,i)=>g.some((b,j)=>j>i&&[a.a.key,a.b.key].some(x=>x===b.a.key||x===b.b.key)))){
     (entry.searchTags??=[]).push('角平分线','角平分','等角');
    }
   }
  }
 }
 emit(classes,'双线夹角');emit(vertex,'顶点钝角');
});
function acuteLineAngle(c,a,b){const d=Math.abs(c.dotLine(a,b));return Math.acos(Math.max(0,Math.min(1,d)))*180/Math.PI;}
function lineAngles(c){const out=[];pairs(c.L,(a,b)=>{
 const deg=acuteLineAngle(c,a,b);
 if(Number.isFinite(deg)&&deg>c.mode.angle&&deg<=90+c.mode.angle)out.push({a,b,value:deg,key:sortIds([a.key,b.key]),support:sortIds([a.key,b.key])});
 });return out.sort((a,b)=>a.value-b.value);
}
function pointAngles(c){const out=[];for(const v of c.P)for(let i=0;i<c.P.length;i++)for(let j=i+1;j<c.P.length;j++){
 const a=c.P[i],b=c.P[j];if(a===v||b===v)continue;
 const value=c.angle(a,v,b);if(Number.isFinite(value)&&value>c.mode.angle&&value<180-c.mode.angle)
  out.push({a:a.id,v:v.id,b:b.id,value,key:pointAngleKey(c,a,v,b),support:supportAngleKey(c,a,v,b)});
 }return [...new Map(out.map(a=>[a.key,a])).values()];
}
function angleMarks(c,a){if(a.v)return [sMark(a.v,a.a),sMark(a.v,a.b),arcMark(a.a,a.v,a.b)];
 const x=lineMark(a.a),y=lineMark(a.b);return [x,y,{type:'lineAngle',a:x.ref,b:y.ref,label:Math.round(a.value*100)/100+'°'}];}
function angleName(c,a){return a.v?`∠${pid(c,a.a)}${pid(c,a.v)}${pid(c,a.b)}`:`∠(${c.lname(a.a)}, ${c.lname(a.b)})`;}
function angleSupportsSame(c,x,y){return !x.v&&!y.v&&sortIds([x.a.key,x.b.key])===sortIds([y.a.key,y.b.key]);}
rule('complementaryAngle','互余角簇','线与角','按互异方向的夹角值归并；排除任意直角被第三条线分拆的恒等式。',c=>{
 const fam=directionClasses(c),anglePairs=[];
 for(let i=0;i<fam.length;i++)for(let j=i+1;j<fam.length;j++){
  const a=fam[i][0],b=fam[j][0],value=acuteLineAngle(c,a,b);
  if(value>c.mode.angle&&value<90-c.mode.angle)anglePairs.push({a,b,key:'DIR:'+i+'|'+j,value});
 }
 anglePairs.sort((a,b)=>a.value-b.value||a.key.localeCompare(b.key));
 const groups=[];for(const q of anglePairs){let g=groups[groups.length-1];if(!g||q.value-g[0].value>c.mode.angle)groups.push(g=[]);g.push(q);}
 const titleGroup=g=>g.length<=3?g.map(x=>angleName(c,x)).join(' = '):g.slice(0,3).map(x=>angleName(c,x)).join(' = ')+` …（${g.length} 类）`;
 for(let i=0;i<groups.length;i++)for(let j=i;j<groups.length;j++){
  if(Math.abs(groups[i][0].value+groups[j][0].value-90)>c.mode.angle)continue;
  if([groups[i][0].value,groups[j][0].value].every(v=>[15,30,45,60,75].some(z=>Math.abs(z-v)<=c.mode.angle)))continue;
  let witness=null;
  for(const x of groups[i]){if(witness)break;for(const y of groups[j]){
   if(x.key===y.key||triviality.intrinsicComplement(c,x,y))continue;
   witness=[x,y];break;
  }}
  if(!witness)continue;
  const [a,b]=witness;
  c.add('complementaryAngle',groups[i].map(x=>x.key).sort().join(';')+'|'+groups[j].map(x=>x.key).sort().join(';'),
   `${titleGroup(groups[i])} 与 ${titleGroup(groups[j])} 互余（和为90°）`,
   [...angleMarks(c,a),...angleMarks(c,b)],76,'互余角连等簇',groups[i].concat(groups[j]).map(x=>angleName(c,x)));
 }
});
// Shared angle-value prepass for supplementary and rational angle ratios.
// A chain is one fact, not a Cartesian product of equivalent renderings.
function angleValueGroups(c){
 if(c._angleValueGroups)return c._angleValueGroups;
 const all=pointAngles(c).sort((x,y)=>x.value-y.value||x.key.localeCompare(y.key)),out=[];
 for(const t of all){let g=out[out.length-1];if(!g||t.value-g[0].value>c.mode.angle)out.push(g=[]);g.push(t);}
 return c._angleValueGroups=out;
}
function groupLabel(c,g){const labels=g.map(x=>angleName(c,x));return labels.length<=4?labels.join(' = '):labels.slice(0,4).join(' = ')+` …（${labels.length} 个等价角）`;}
function angleFixed(c,g){return [15,30,45,60,75,90,120,135,150].some(v=>Math.abs(g[0].value-v)<=c.mode.angle);}
function groupKey(g){return g.map(x=>x.key).sort().join(';');}
rule('supplementaryAngle','互补角簇','线与角','先归并所有表示同一度数的顶点角，再识别独立互补角簇；过滤平行射线同义反复。',c=>{
 const gs=angleValueGroups(c);
 for(let i=0;i<gs.length;i++)for(let j=i;j<gs.length;j++){
  if(Math.abs(gs[i][0].value+gs[j][0].value-180)>c.mode.angle)continue;
  if(angleFixed(c,gs[i])&&angleFixed(c,gs[j]))continue;
  let chosen=null;
  for(const a of gs[i]){if(chosen)break;for(const b of gs[j]){
   if(a.key===b.key||a.support===b.support||triviality.intrinsicSupplement(c,a,b))continue;
   chosen=[a,b];break;
  }}
  if(!chosen)continue;
  const [a,b]=chosen; c.add('supplementaryAngle',groupKey(gs[i])+'|'+groupKey(gs[j]),
   `${groupLabel(c,gs[i])} 与 ${groupLabel(c,gs[j])} 互补（和为180°）`,
   [...angleMarks(c,a),...angleMarks(c,b)],71,'互补角连等簇',gs[i].concat(gs[j]).map(x=>angleName(c,x)));
 }
});
rule('angleRatio','角度常见比例簇','线与角','角按值合并后搜索 3/2、2、3 倍，排除两个固定角之间的数值同义反复。',c=>{
 const gs=angleValueGroups(c);
 for(let i=0;i<gs.length;i++)for(const mult of [1.5,2,3]){
  const target=gs[i][0].value*mult;if(target>=180)continue;
  for(let j=i+1;j<gs.length;j++){
   if(gs[j][0].value>target+c.mode.angle)break;
   if(Math.abs(gs[j][0].value-target)>c.mode.angle)continue;
   if(angleFixed(c,gs[i])&&angleFixed(c,gs[j]))continue;
   const a=gs[i][0],b=gs[j][0];
   c.add('angleRatio',groupKey(gs[i])+'|'+groupKey(gs[j])+'|'+mult,
    `${groupLabel(c,gs[j])} = ${mult===1.5?'3/2':mult} × (${groupLabel(c,gs[i])})`,
    [...angleMarks(c,a),...angleMarks(c,b)],65,'角度连比簇',gs[i].concat(gs[j]).map(x=>angleName(c,x)));
  }
 }
});
rule('angleBisector','角平分','线与角','直线或两点隐含线段是角的内/外角平分线。',c=>{
 for(const v of c.P){const rays=c.P.filter(p=>p!==v);for(let i=0;i<rays.length;i++)for(let j=i+1;j<rays.length;j++){
 const a=rays[i],b=rays[j],x=V.sub(a,v),y=V.sub(b,v),nx=Math.hypot(x.x,x.y),ny=Math.hypot(y.x,y.y);if(nx<=c.eps||ny<=c.eps)continue;
 const u={x:x.x/nx,y:x.y/nx},w={x:y.x/ny,y:y.y/ny};
 for(const l of c.L){
  if(c.distPointLine(v,l)>c.eps)continue;
  const t={x:l.dx,y:l.dy};const v1=Math.abs(V.cross(u,t)),v2=Math.abs(V.cross(w,t));if(Math.abs(v1-v2)>c.mode.direction)continue;
  if(Math.abs(V.cross(u,w))<=c.mode.direction||Math.abs(V.dot(u,t))>=1-c.mode.direction||Math.abs(V.dot(w,t))>=1-c.mode.direction)continue;
  const angle=Math.acos(Math.max(-1,Math.min(1,V.dot(u,w))))*180/Math.PI;
  const internal=Math.abs(Math.abs(V.dot(u,t))-Math.cos(angle*Math.PI/360))<=c.mode.direction*2;
  // A different triangle's special line must NOT suppress this bisector.
  const kind=internal?'内':'外';c.add('angleBisector',pointAngleKey(c,a,v,b)+'|'+l.key,`${c.lname(l)} 是 ∠${pid(c,a.id)}${pid(c,v.id)}${pid(c,b.id)} 的${kind}角平分线`,[sMark(v.id,a.id),sMark(v.id,b.id),lineMark(l),arcMark(a.id,v.id,b.id)],75);
 }
 }}
});
// ---------- metric and ratios ----------
rule('distanceRatio','距离连等 / 连比','长度与面积','统一比较线段长度、全部显式圆半径与点到直线距离；对所有真实长度建立等长簇，等长只报告一条连等；常见倍率按簇比较，支持逆比检索。',c=>{
 const d=c.distances;if(d.length<2)return;
 const clusters=[];
 // Sorted complete-link metric clusters: not the transitive closure of an
 // approximate equality. All comparisons use the same fixed strict mode.
 for(const metric of d){let g=clusters[clusters.length-1];
  if(!g||!c.nearRatio(metric.value,g.members[0].value,1))clusters.push(g={members:[]});
  g.members.push(metric);
 }
 const label=x=>x.type==='radius'?`r(${c.display(x.circle.id)})`:x.type==='segment'?`${pid(c,x.a)}${pid(c,x.b)}`:`d(${pid(c,x.point)},${x.line.kind==='explicit'?c.display(x.line.id):x.line.kind==='implicit'?pid(c,x.line.a)+pid(c,x.line.b):c.lname(x.line)})`;
 // d(P,l)=PQ is merely another notation for PQ when Q is the perpendicular
 // foot on l. Keep the alias searchable, but do not print it as extra evidence.
 const visibleMetrics=g=>{
  const members=g.members.filter(m=>{
   if(m.type!=='pointLine')return true;
   const p=c.pMap.get(m.point);if(!p)return true;
   return !g.members.some(s=>{
    if(s.type!=='segment'||(s.a!==m.point&&s.b!==m.point))return false;
    const q=c.pMap.get(s.a===m.point?s.b:s.a);if(!q||c.distPointLine(q,m.line)>c.eps)return false;
    const vx=q.x-p.x,vy=q.y-p.y,len=Math.hypot(vx,vy);
    return len>c.eps&&Math.abs(vx*m.line.dx+vy*m.line.dy)<=c.mode.direction*len;
   });
  });
  return members;
 };
 const key=x=>x.members.map(m=>m.key).sort().join(';');
 const names=x=>visibleMetrics(x).map(label);
 const chain=x=>{const names0=names(x);return names0.length<=7?names0.join(' = '):names0.slice(0,7).join(' = ')+` … (共${names0.length}段)`;};
 const metrics=x=>x.members.map(m=>({name:label(m),key:m.key,type:m.type,redundant:!visibleMetrics(x).includes(m),aliases:m.type==='segment'?[`${pid(c,m.a)}${pid(c,m.b)}`,`${pid(c,m.b)}${pid(c,m.a)}`]:m.type==='radius'?[label(m),`半径${c.display(m.circle.id)}`,`${c.display(m.circle.id)}半径`]:[label(m)]}));
 const addRatio=(a,b,f)=>{const allMarks=[...visibleMetrics(a),...(b?visibleMetrics(b):[])].flatMap(m=>m.marks);
  const title=f===1?chain(a)+'（等长）':`${chain(a)} = ${f.label} × (${chain(b)})`;
  const tokens=[...names(a),...(b?names(b):[])];
  c.add('distanceRatio',f===1?'EQ:'+key(a):`R:${key(a)}|${key(b)}|${f.label}`,title,allMarks,f===1?58:72,
   f===1?'长度连等，含所有等价记号':`${f.label} 倍 · 高长度 ${a.members.length} 种、低长度 ${b.members.length} 种`,tokens);
  const e=c.entries[c.entries.length-1];
  // Detailed metric witness metadata lets searching recognize every implicit
  // A:B or B:A variation without materializing O(n²) alias strings.
  if(e?.kind==='distanceRatio')e.metric={high:metrics(a),low:b?metrics(b):[],factor:f===1?1:f.v,factorLabel:f===1?'1':f.label};
 };
 function isTrivialSameSide(g){
  if(g.members.length<2||g.members.some(m=>m.type!=='pointLine'))return false;
  const support=g.members[0].line.key;
  if(!g.members.every(m=>m.line.key===support))return false;
  const signs=g.members.map(m=>Math.sign(c.signedPointLine(c.pMap.get(m.point),m.line)));
  return signs.every(v=>v===signs[0]); // equivalent to same-side parallelism
 }
 for(const g of clusters)if(visibleMetrics(g).length>1&&!(c.selected.has('parallel')&&isTrivialSameSide(g)))addRatio(g,null,1);
 // Groups are strictly ordered by length; each factor >1 has only one
 // canonical direction. Reverse forms are semantic search aliases, not rows.
 for(let i=0;i<clusters.length;i++)for(const factor of RATIOS){if(factor.v===1)continue;
  const lo=clusters[i],target=lo.members[0].value*factor.v;
  let left=i+1,right=clusters.length;while(left<right){const mid=(left+right)>>1;if(clusters[mid].members[0].value<target-c.eps*2-c.mode.ratio*target)left=mid+1;else right=mid;}
  for(let j=left;j<clusters.length&&clusters[j].members[0].value<=target+c.eps*2+c.mode.ratio*target;j++){
   const hi=clusters[j];if(c.nearRatio(hi.members[0].value,lo.members[0].value,factor.v))addRatio(hi,lo,factor);
  }
 }
});
rule('quadraticRatio','二次量连等 / 连比','长度与面积','按需扫描：比较长度平方、半径平方、平方和差、线段乘积、三角形面积、圆面积、有符号点幂和点积；对代数替换类连等降权，支持常见正负倍率及倒数查询。',c=>quadratic.scan(c),false);
rule('midpoint','中点','长度与面积','用三个点识别任意 AB 的中点 M。',c=>{for(const a of c.P)for(const b of c.P){if(a===b)continue;for(const m of c.P){if(m===a||m===b)continue;
 const mid=V.mul(V.add(a,b),.5);if(V.dist(mid,m)<=c.eps)c.add('midpoint',m.id+'|'+sortIds([a.id,b.id]),`${pid(c,m.id)} 是 ${pid(c,a.id)}${pid(c,b.id)} 的中点`,[sMark(a.id,b.id),pMark(m.id)],83);
 }}});
rule('goldenDivision','黄金分割点','长度与面积','C 在 AB 上，并使 AB/AC 或 AB/CB 等于 φ。',c=>{const phi=(1+Math.sqrt(5))/2;for(const seg of c.segments){const a=c.pMap.get(seg.a),b=c.pMap.get(seg.b);for(const p of c.P){if(p===a||p===b)continue;
 const t=V.dot(V.sub(p,a),V.sub(b,a))/(seg.value*seg.value);if(t<=0||t>=1||Math.abs(V.cross(V.sub(b,a),V.sub(p,a)))/seg.value>c.eps)continue;
 const x=V.dist(p,a),y=V.dist(p,b);if(c.nearRatio(seg.value,Math.max(x,y),phi))c.add('goldenDivision',p.id+'|'+seg.key,`${pid(c,p.id)} 黄金分割 ${pid(c,a.id)}${pid(c,b.id)}`,[sMark(a.id,b.id),pMark(p.id)],85);
 }}});
rule('perpFoot','垂足','长度与面积','PQ 垂直于已有直线或 BC，Q 位于该直线上。',c=>{for(const l of c.L)for(const q of c.P){if(c.distPointLine(q,l)>c.eps)continue;for(const p of c.P){if(p===q||c.distPointLine(p,l)<=c.eps)continue;
 const x=V.sub(p,q);if(Math.abs(x.x*l.dx+x.y*l.dy)<=c.mode.direction*Math.hypot(x.x,x.y))c.add('perpFoot',p.id+'|'+q.id+'|'+l.key,`${pid(c,q.id)} 是 ${pid(c,p.id)} 到 ${c.lname(l)} 的垂足`,[lineMark(l),sMark(p.id,q.id),pMark(q.id)],77);
 }}});
function areaClasses(c){
 if(c._areaClasses)return c._areaClasses;
 const sorted=c.triangles.slice().sort((a,b)=>a.area-b.area),families=[];
 for(const t of sorted){let g=families[families.length-1];
  if(!g||!c.nearRatio(t.area,g[0].area,1))families.push(g=[]);
  g.push(t);
 }
 return c._areaClasses=families;
}
rule('equalArea','三角形等面积簇','长度与面积','同一面积的三角形合并成一条连等，保留各三角形组合供搜索。',c=>{
 for(const group of areaClasses(c)){if(group.length<2)continue;
  const names=group.map(t=>`[${triName(c,t)}]`);
  c.add('equalArea',group.map(t=>t.ids.join(',')).sort().join(';'),`${names.slice(0,10).join(' = ')}${names.length>10?` …（${names.length}个）`:''}`,
   group.flatMap(triMarks),70,`${group.length} 个等面积三角形`,names);
 }
});
// ---------- circles: explicit circles plus O(A,B) circles centered at A through B ----------
function virtualCircleAliases(c,q){return [...(q.aliases||[]),...(q.virtualAliases||[])].map(a=>`⊙(${c.display(a.a)},${c.display(a.b)})`);}
function extraCircleLabel(c,q){const names=virtualCircleAliases(c,q);return names.length?' · 等价圆：'+names.slice(0,3).join('、')+(names.length>3?' 等':''):'';}
rule('circleLineTangent','圆与线相切','圆与切线','检测现有圆、以 A 为圆心经过 B 的隐含圆，和显式线或 CD 直线相切。',c=>{
 for(const q of c.allCircles)for(const l of c.L){const foot=c.project({x:q.x,y:q.y},l);if(!c.near(c.distPointLine({x:q.x,y:q.y},l),q.r)||triviality.intrinsicLineTangent(c,q,l,foot))continue;
 if(q.kind==='implicit'&&l.kind==='implicit'&&sortIds([q.a,q.b])===sortIds([l.a,l.b]))continue;
 // This describes tangent to the supporting line; show the true tangent contact.
 c.add('circleLineTangent',q.key+'|'+l.key,`圆 ${c.cname(q)} 与 ${c.lname(l)} 相切`,[circleMark(q),lineMark(l),{type:'tangentFoot',circleRef:circleMark(q).ref,lineRef:lineMark(l).ref}],88,(q.kind==='implicit'?'隐含作圆 · 以首点为圆心':'现有圆')+extraCircleLabel(c,q),virtualCircleAliases(c,q));
 }});
rule('circleCircleTangent','两圆相切','圆与切线','两圆内切或外切，允许以 A 为圆心过 B 的隐含圆。',c=>{pairs(c.allCircles,(a,b)=>{
 if(a.key===b.key) return;const d=Math.hypot(a.x-b.x,a.y-b.y);if(d<=c.eps)return;
 const external=c.near(d,a.r+b.r),internal=c.near(d,Math.abs(a.r-b.r));if(!external&&!internal||triviality.intrinsicCircleTangent(c,a,b,external))return;
 c.add('circleCircleTangent',sortIds([a.key,b.key]),`${c.cname(a)} 与 ${c.cname(b)} ${external?'外切':'内切'}`,[circleMark(a),circleMark(b)],86,'',virtualCircleAliases(c,a).concat(virtualCircleAliases(c,b)));
 });});
rule('concentric','同心圆','圆与切线','已有圆之间的同心关系。',c=>{pairs(c.C,(a,b)=>{if(Math.hypot(a.x-b.x,a.y-b.y)<=c.eps)c.add('concentric',sortIds([a.key,b.key]),`${c.cname(a)}、${c.cname(b)} 同心`,[circleMark(a),circleMark(b)],73);});});
rule('equalRadius','等半径圆','圆与切线','两个已有圆的半径相等。',c=>{pairs(c.C,(a,b)=>{if(c.near(a.r,b.r))c.add('equalRadius',sortIds([a.key,b.key]),`${c.cname(a)}、${c.cname(b)} 等半径`,[circleMark(a),circleMark(b)],58);});});
rule('orthogonalCircles','正交圆','圆与切线','两个已有圆交点处的切线相互垂直。',c=>{pairs(c.C,(a,b)=>{const d2=(a.x-b.x)**2+(a.y-b.y)**2;if(c.nearQuadratic(d2,a.r*a.r+b.r*b.r))c.add('orthogonalCircles',sortIds([a.key,b.key]),`${c.cname(a)} ⟂ ${c.cname(b)}（正交圆）`,[circleMark(a),circleMark(b)],81);});});
rule('diameter','直径所对直角','圆与切线','A、B 是现有圆直径的端点。',c=>{for(const q of c.C)for(const seg of c.segments){const a=c.pMap.get(seg.a),b=c.pMap.get(seg.b);if(c.near(V.dist(a,{x:q.x,y:q.y}),q.r)&&c.near(V.dist(b,{x:q.x,y:q.y}),q.r)&&Math.hypot((a.x+b.x)/2-q.x,(a.y+b.y)/2-q.y)<=c.eps)
 c.add('diameter',q.key+'|'+seg.key,`${pid(c,a.id)}${pid(c,b.id)} 是 ${c.cname(q)} 的直径`,[circleMark(q),sMark(a.id,b.id)],76);
 }});
// ---------- triangles and triangle centers ----------
// "right triangle" is represented by perpendicular supporting lines;
// "isosceles triangle" is represented by the corresponding length equality.
// These names remain available in semantic search rather than double-counted.
rule('equilateral','等边三角形','三角形','三条边相等的三角形。',c=>{for(const t of c.triangles){const [a,b,d]=t.sorted;if(c.near(a,b)&&c.near(b,d))c.add('equilateral',t.ids.join('|'),`△${triName(c,t)} 是等边三角形`,triMarks(t),86);}});
rule('specialPoint','三角形特殊点','三角形','同一顶点同时担任多个三角形中心时合并为一条复合角色关系。',c=>{
 for(const t of c.triangles){const [a,b,d]=t.pts,[sa,sb,sc]=t.sides,p=t.circ,perimeter=sa+sb+sc;
  const centroid=V.mul(V.add(V.add(a,b),d),1/3),circ=p?{x:p.x,y:p.y}:null;
  const orth=circ?{x:a.x+b.x+d.x-2*circ.x,y:a.y+b.y+d.y-2*circ.y}:null;
  const incenter={x:(a.x*sa+b.x*sb+d.x*sc)/perimeter,y:(a.y*sa+b.y*sb+d.y*sc)/perimeter};
  const nine=orth&&circ?V.mul(V.add(orth,circ),.5):null;
  const ex=[];for(const [i,s] of [[0,sa],[1,sb],[2,sc]]){const z=2*s-perimeter;if(Math.abs(z)<=c.eps)continue;ex.push({i,point:{x:(2*t.pts[i].x*s-(a.x*sa+b.x*sb+d.x*sc))/z,y:(2*t.pts[i].y*s-(a.y*sa+b.y*sb+d.y*sc))/z}});}
  const centers=[['重心',centroid,96],['外心',circ,95],['垂心',orth,96],['内心',incenter,96],['九点圆心',nine,93],...ex.map(o=>[`${pid(c,t.pts[o.i].id)} 对旁心`,o.point,94])];
  for(const pt of c.P){if(t.ids.includes(pt.id))continue;
   const roles=centers.filter(([,at])=>at&&V.dist(pt,at)<=c.eps);
   if(roles.length)c.add('specialPoint',pt.id+'|'+t.ids.join('|'),`${pid(c,pt.id)} 是 △${triName(c,t)} 的${roles.map(r=>r[0]).join('、')}`,[...triMarks(t),pMark(pt.id)],Math.max(...roles.map(r=>r[2])),roles.map(r=>r[0]).join('、'));
  }
 }});
rule('specialLine','三角形特殊线','三角形','相同三角形、支撑线的中线/高线/角平分线/中垂线角色合并展示。',c=>{
 for(const t of c.triangles)for(let i=0;i<3;i++){
  const v=t.pts[i],b=t.pts[(i+1)%3],d=t.pts[(i+2)%3],side=V.sub(d,b),len=V.dist(b,d),mid=V.mul(V.add(b,d),.5);
  for(const l of c.L){const through=c.distPointLine(v,l)<=c.eps,midOn=c.distPointLine(mid,l)<=c.eps;
   const perp=Math.abs(l.dx*side.x+l.dy*side.y)<=c.mode.direction*len;
   if(through){const roles=[];
    if(midOn)roles.push('中线');if(perp)roles.push('高线');
    const x=V.sub(b,v),y=V.sub(d,v),lx=Math.hypot(x.x,x.y),ly=Math.hypot(y.x,y.y);
    const half=V.add(V.mul(x,1/lx),V.mul(y,1/ly)),nl=Math.hypot(half.x,half.y);
    if(nl>c.eps&&Math.abs(l.dx*half.y-l.dy*half.x)/nl<=c.mode.direction)roles.push('角平分线');
    if(midOn&&perp)roles.push('中垂线');
    if(roles.length===1&&roles[0]==='高线'&&c.selected.has('perpendicular'))continue; // entire altitude is just a perpendicular line through the named vertex
    if(roles.length===1&&roles[0]==='中垂线'&&c.selected.has('perpendicular')&&c.selected.has('midpoint')&&c.P.some(m=>V.dist(m,mid)<=c.eps))continue;
    if(roles.length===1&&roles[0]==='高线'&&(c.distPointLine(b,l)<=c.eps||c.distPointLine(d,l)<=c.eps))continue;
    if(roles.length===1&&roles[0]==='中线'&&c.P.some(m=>m.id!==v.id&&V.dist(m,mid)<=c.eps)&&c.selected.has('midpoint'))continue; // implied by named midpoint and connecting line // leg of a right triangle is not a new altitude condition
    if(roles.length===1&&roles[0]==='高线'&&c.selected.has('perpendicular'))continue; // entire altitude is just a perpendicular line through the named vertex
    if(roles.length===1&&roles[0]==='中垂线'&&c.selected.has('perpendicular')&&c.selected.has('midpoint')&&c.P.some(m=>V.dist(m,mid)<=c.eps))continue;
    if(roles.length===1&&roles[0]==='高线'&&(c.distPointLine(b,l)<=c.eps||c.distPointLine(d,l)<=c.eps))continue;
    if(roles.length===1&&roles[0]==='中线'&&c.P.some(m=>m.id!==v.id&&V.dist(m,mid)<=c.eps)&&c.selected.has('midpoint'))continue; // implied by named midpoint and connecting line // already expressed by the incident sides' perpendicularity
    if(roles.length)c.add('specialLine',v.id+'|'+sortIds([b.id,d.id])+'|'+l.key,`${c.lname(l)} 是 △${triName(c,t)} 从 ${pid(c,v.id)} 引出的${roles.join('、')}`,[...triMarks(t),lineMark(l)],roles.length>1?88:84,roles.join('、'));
   }else if(midOn&&perp)c.add('specialLine','bisector|'+sortIds([b.id,d.id])+'|'+l.key,`${c.lname(l)} 是 ${pid(c,b.id)}${pid(c,d.id)} 的中垂线`,[sMark(b.id,d.id),lineMark(l)],82,'中垂线');
  }
 }});
function triangleClasses(c,mode){
 const ts=c.triangles.map(t=>({ ...t, shape:[t.sorted[0]/t.sorted[2],t.sorted[1]/t.sorted[2]] }));
 ts.sort((a,b)=>(mode==='congruent'?a.sorted[2]-b.sorted[2]:a.shape[0]-b.shape[0]));
 const families=[];
 const equal=(a,b)=>mode==='congruent'?a.sorted.every((v,i)=>c.near(v,b.sorted[i])):
  a.shape.every((v,i)=>Math.abs(v-b.shape[i])<=c.mode.ratio);
 // A family is a complete-link equivalence class, not arbitrary transitive
 // chains of near matches.  A~B, B~C alone does not imply A~C.
 for(const t of ts){let group=null;
  for(let i=families.length-1;i>=0;i--){const g=families[i];
   const dx=mode==='congruent'?t.sorted[2]-g[0].sorted[2]:t.shape[0]-g[0].shape[0];
   if(dx>2*c.eps+2*c.mode.ratio)break;
   if(g.every(u=>equal(u,t))){group=g;break;}
  }
  if(group)group.push(t);else families.push([t]);
 }
 return families.filter(g=>g.length>=2);
}
// Recover a real vertex correspondence (including reflected order). Sorting
// three SIDE LENGTHS is sufficient for testing similarity, but not for
// labelling triangle ABC ~ DEF: corresponding vertices must line up.
const TRI_PERMS=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
function triangleCorrespondence(c,anchor,candidate,similar){
 const len=(t,i,j)=>V.dist(t.pts[i],t.pts[j]);
 const base=anchor.sorted[2],other=candidate.sorted[2];
 const k=similar?other/base:1;
 if(!(base>c.eps&&other>c.eps))return null;
 const valid=TRI_PERMS.filter(perm=>[[0,1],[1,2],[2,0]].every(([i,j])=>
  c.near(len(candidate,perm[i],perm[j]),len(anchor,i,j)*k,Math.max(base,other))));
 // Isosceles/equilateral triangles can have several valid mappings; prefer
 // identical IDs when possible, then use lexicographic canonical order.
 valid.sort((a,b)=>{
  const score=perm=>perm.filter((v,i)=>candidate.ids[v]===anchor.ids[i]).length;
  return score(b)-score(a)||a.map(v=>candidate.ids[v]).join('|').localeCompare(b.map(v=>candidate.ids[v]).join('|'));
 });
 return valid[0]||null;
}
function alignedTriangleGroup(c,g,similar){
 const anchor=g[0];return g.map(t=>{
  const perm=triangleCorrespondence(c,anchor,t,similar);
  return {ids:perm?perm.map(i=>t.ids[i]):t.ids.slice(),original:t.ids.slice(),scale:similar?t.sorted[2]/anchor.sorted[2]:1,valid:!!perm};
 });
}
rule('similarTriangles','相似三角形簇','三角形','保留严格边比例与一一顶点对应关系，并返回所有相似三角形。',c=>{
 for(const g of triangleClasses(c,'similar')){
  if(g.every(t=>t.sorted.every((v,i)=>c.near(v,g[0].sorted[i]))))continue;
  g.sort((a,b)=>a.ids.join('|').localeCompare(b.ids.join('|')));
  const ordered=alignedTriangleGroup(c,g,true);
  if(ordered.some(t=>!t.valid))continue;
  const names=ordered.map(t=>'△'+t.ids.map(c.display).join(''));
  const key=g.map(t=>t.ids.join(',')).join(';');
  c.add('similarTriangles',key,`${names.join(' ∽ ')}（对应顶点同序；同一相似簇）`,g.flatMap(triMarks),90,
   `${g.length} 个三角形，严格对应；可检索任意一组`,names);
  const e=c.entries[c.entries.length-1];if(e?.id==='similarTriangles:'+key){e.correspondences=ordered;e.searchTags=[...(e.searchTags||[]),...g.map(t=>'△'+triName(c,t))];}
 }
});
rule('congruentTriangles','全等三角形簇','三角形','以全边比较确定真实的对应顶点，规范展示同序全等三角形。',c=>{
 for(const g of triangleClasses(c,'congruent')){
  g.sort((a,b)=>a.ids.join('|').localeCompare(b.ids.join('|')));
  const ordered=alignedTriangleGroup(c,g,false);
  if(ordered.some(t=>!t.valid))continue;
  const names=ordered.map(t=>'△'+t.ids.map(c.display).join(''));
  const key=g.map(t=>t.ids.join(',')).join(';');
  c.add('congruentTriangles',key,`${names.join(' ≅ ')}（对应顶点同序；同一全等簇）`,g.flatMap(triMarks),91,
   `${g.length} 个三角形，严格对应；可检索任意一组`,names);
  const e=c.entries[c.entries.length-1];if(e?.id==='congruentTriangles:'+key){e.correspondences=ordered;e.searchTags=[...(e.searchTags||[]),...g.map(t=>'△'+triName(c,t))];}
 }
});
// ---------- quadrilaterals: ordered polygon evaluation, not just 4 chosen unordered points ----------
function quads(c,fn){const P=c.P;for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++)for(let k=j+1;k<P.length;k++)for(let l=k+1;l<P.length;l++){
 const pts=[P[i],P[j],P[k],P[l]],mx=pts.reduce((s,p)=>s+p.x,0)/4,my=pts.reduce((s,p)=>s+p.y,0)/4;
 pts.sort((a,b)=>Math.atan2(a.y-my,a.x-mx)-Math.atan2(b.y-my,b.x-mx));
 const cross=(a,b,d)=>V.cross(V.sub(b,a),V.sub(d,b));const s=pts.map((p,n)=>cross(p,pts[(n+1)%4],pts[(n+2)%4]));
 if(s.some(v=>Math.abs(v)<=c.eps*c.localScale)||!(s.every(x=>x>0)||s.every(x=>x<0)))continue;
 fn(pts);
 }}
const quadMarks=q=>[sMark(q[0].id,q[1].id),sMark(q[1].id,q[2].id),sMark(q[2].id,q[3].id),sMark(q[3].id,q[0].id),...points(q.map(p=>p.id))];
const qName=(c,q)=>q.map(p=>c.display(p.id)).join('');
function isPar(c,q){const [a,b,d,e]=q;return V.dist(V.add(a,d),V.add(b,e))<=2*c.eps;}
function isRect(c,q){return isPar(c,q)&&Math.abs(V.dot(V.sub(q[1],q[0]),V.sub(q[2],q[1])))<=c.mode.direction*V.dist(q[1],q[0])*V.dist(q[2],q[1]);}
function isRhomb(c,q){return isPar(c,q)&&c.near(V.dist(q[0],q[1]),V.dist(q[1],q[2]));}
rule('parallelogram','平行四边形','四边形','两组对边平行的凸四边形，排除矩形、菱形和正方形。',c=>{quads(c,q=>{if(isPar(c,q)&&!isRect(c,q)&&!isRhomb(c,q))c.add('parallelogram',sortIds(q.map(p=>p.id)),`${qName(c,q)} 为平行四边形`,quadMarks(q),78);});});
rule('rectangle','矩形','四边形','四个直角且对边平行；排除正方形。',c=>{quads(c,q=>{if(isRect(c,q)&&!isRhomb(c,q))c.add('rectangle',sortIds(q.map(p=>p.id)),`${qName(c,q)} 为矩形`,quadMarks(q),84);});});
rule('rhombus','菱形','四边形','四边等长；排除正方形。',c=>{quads(c,q=>{if(isRhomb(c,q)&&!isRect(c,q))c.add('rhombus',sortIds(q.map(p=>p.id)),`${qName(c,q)} 为菱形`,quadMarks(q),84);});});
rule('square','正方形','四边形','同时为矩形及菱形。',c=>{quads(c,q=>{if(isRect(c,q)&&isRhomb(c,q))c.add('square',sortIds(q.map(p=>p.id)),`${qName(c,q)} 为正方形`,quadMarks(q),95);});});
rule('kite','筝形','四边形','两组相邻边分别等长，排除菱形/正方形。',c=>{quads(c,q=>{if(isRhomb(c,q))return;const a=q.map((v,i)=>V.dist(v,q[(i+1)%4]));if((c.near(a[0],a[1])&&c.near(a[2],a[3]))||(c.near(a[0],a[3])&&c.near(a[1],a[2])))c.add('kite',sortIds(q.map(p=>p.id)),`${qName(c,q)} 是筝形`,quadMarks(q),75);});});
rule('harmonic','调和点列','四边形','四个互异共线点的无向调和四点组：枚举三种配对，交比 −1；避免因字母排序遗漏。',c=>{
 const P=c.P;
 for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++)for(let k=j+1;k<P.length;k++)for(let h=k+1;h<P.length;h++){
  const four=[P[i],P[j],P[k],P[h]];
  if(four.some((p,t)=>four.slice(t+1).some(q=>V.dist(p,q)<=c.eps)))continue;
  const x=V.sub(four[1],four[0]),len=V.dist(four[0],four[1]);
  if(!(len>c.eps)||four.slice(2).some(p=>Math.abs(V.cross(x,V.sub(p,four[0])))/len>c.eps))continue;
  // For four positions there are three unordered pair partitions.
  // Swap C/D as well as the chosen primary pair to test naming-invariant
  // cross ratios, without emitting six equivalent entries.
  const partitions=[[0,1,2,3],[0,2,1,3],[0,3,1,2]];
  for(const [ai,bi,ci,di] of partitions){const [a,b,d,e]=[four[ai],four[bi],four[ci],four[di]],ab=V.sub(b,a),l2=V.dot(ab,ab);
   const t=p=>V.dot(V.sub(p,a),ab)/l2,tc=t(d),te=t(e),den=(tc-1)*te;
   if(Math.abs(den)<1e-12)continue;
   const ratio=tc*(te-1)/den;
   if(Math.abs(ratio+1)<=c.mode.ratio){const id=sortIds(four.map(p=>p.id));
    c.add('harmonic',id,`(${pid(c,a.id)}${pid(c,b.id)};${pid(c,d.id)}${pid(c,e.id)}) = −1`,[sMark(a.id,b.id),sMark(d.id,e.id),...points(four.map(p=>p.id))],91,'调和交比（带符号）');
    const entry=c.entries[c.entries.length-1];if(entry?.id==='harmonic:'+id){entry.harmonicPairing=[[a.id,b.id],[d.id,e.id]];entry.searchTags=[...(entry.searchTags||[]),'交比 -1','调和四点','调和点列'];}
    break;
   }
  }
 }
});
// ---------- transformations ----------
rule('reflection','轴对称点对','变换','两个点关于某条已存在直线对称。',c=>{for(const l of c.EUnique)for(const seg of c.segments){const a=c.pMap.get(seg.a),b=c.pMap.get(seg.b),m=V.mul(V.add(a,b),.5),dir=V.sub(b,a);
 if(c.distPointLine(m,l)<=c.eps&&Math.abs(dir.x*l.dx+dir.y*l.dy)<=c.mode.direction*seg.value)c.add('reflection',seg.key+'|'+l.key,`${pid(c,a.id)}、${pid(c,b.id)} 关于 ${c.lname(l)} 对称`,[sMark(a.id,b.id),lineMark(l)],86);
 }});
rule('rotation','定角旋转','变换','两个点相对于中心构成 60°、90°、120° 或 180° 旋转且距中心相等。',c=>{for(const center of c.P)for(let i=0;i<c.P.length;i++)for(let j=i+1;j<c.P.length;j++){
 const a=c.P[i],b=c.P[j];if(a===center||b===center)continue;const da=V.dist(a,center),db=V.dist(b,center);if(!c.near(da,db)||da<=c.eps)continue;
 const angle=c.angle(a,center,b);for(const t of [60,90,120,180])if(Math.abs(t-angle)<=c.mode.angle){if(t===180&&c.selected.has('midpoint'))continue;c.add('rotation',center.id+'|'+sortIds([a.id,b.id])+'|'+t,`${pid(c,a.id)}、${pid(c,b.id)} 绕 ${pid(c,center.id)} 旋转 ${t}° 对应`,[sMark(center.id,a.id),sMark(center.id,b.id),arcMark(a.id,center.id,b.id,t+'°')],t===90?82:74);}
 }});
// ---------- additional independent discovery families ----------
rule('radicalAxis','非平凡根轴点','圆','识别在两非同心圆根轴上的点，但不把两圆公共交点重复当作新关系。',c=>{
 pairs(c.C,(a,b)=>{const delta=V.sub({x:b.x,y:b.y},{x:a.x,y:a.y}),len=Math.hypot(delta.x,delta.y);if(len<=c.eps)return;
  for(const p of c.P){const pa=(p.x-a.x)**2+(p.y-a.y)**2-a.r*a.r,pb=(p.x-b.x)**2+(p.y-b.y)**2-b.r*b.r;
   if(!c.nearQuadratic(pa,pb,Math.max(Math.abs(pa),Math.abs(pb),(a.r+b.r)*c.eps)))continue;
   if(c.nearQuadratic(pa,0,Math.max(a.r*a.r,c.localScale*c.localScale))&&c.nearQuadratic(pb,0,Math.max(b.r*b.r,c.localScale*c.localScale)))continue;
   c.add('radicalAxis',p.id+'|'+sortIds([a.key,b.key]),`${pid(c,p.id)} 在 ${c.cname(a)}、${c.cname(b)} 的根轴上`,[pMark(p.id),circleMark(a),circleMark(b),{type:'radicalAxis',a:circleMark(a).ref,b:circleMark(b).ref}],88,'两圆幂相等但 P 不是公共交点');
  }
 });
});
rule('areaRatio','三角形面积连比','长度与面积','不同等面积簇之间的面积比，合并重复的三角形配对。',c=>{
 const groups=areaClasses(c),kinds=[{v:4/3,label:'4/3'},{v:1.5,label:'3/2'},{v:2,label:'2'},{v:3,label:'3'}];
 const names=g=>g.map(t=>`[${triName(c,t)}]`);
 for(let i=0;i<groups.length;i++)for(const f of kinds){const aim=groups[i][0].area*f.v,margin=c.eps*c.localScale*3+c.mode.ratio*aim;
  let lo=i+1,hi=groups.length;while(lo<hi){const m=(lo+hi)>>1;if(groups[m][0].area<aim-margin)lo=m+1;else hi=m;}
  for(let j=lo;j<groups.length&&groups[j][0].area<=aim+margin;j++){
   if(!c.nearRatio(groups[j][0].area,groups[i][0].area,f.v))continue;
   const a=groups[j],b=groups[i],na=names(a),nb=names(b);
   c.add('areaRatio',a.map(t=>t.ids.join(',')).sort().join(';')+'|'+b.map(t=>t.ids.join(',')).sort().join(';')+'|'+f.label,
    `${na.slice(0,6).join(' = ')} = ${f.label} × (${nb.slice(0,6).join(' = ')})`,[...a,...b].flatMap(triMarks),75,`面积比 ${f.label}；${a.length} 比 ${b.length} 个三角形`,na.concat(nb));
  }
 }
});
rule('orthocentricSystem','垂心组','三角形','四点中每个点都是其余三点所成三角形的垂心。',c=>{
 const P=c.P;for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++)for(let k=j+1;k<P.length;k++)for(let h=k+1;h<P.length;h++){
  const q=[P[i],P[j],P[k],P[h]];
  if(q.every((p,z)=>{const [a,b,d]=q.filter((_,r)=>r!==z);if(c.triangle(a,b,d)<=c.eps*c.localScale)return false;return Math.abs(V.dot(V.sub(p,a),V.sub(b,d)))<=c.mode.direction*V.dist(p,a)*V.dist(b,d)+c.eps*c.localScale&&Math.abs(V.dot(V.sub(p,b),V.sub(a,d)))<=c.mode.direction*V.dist(p,b)*V.dist(a,d)+c.eps*c.localScale;}))
   c.add('orthocentricSystem',sortIds(q.map(p=>p.id)),`${q.map(p=>pid(c,p.id)).join('、')} 构成垂心组`,[...points(q.map(p=>p.id)),...triMarks({ids:q.slice(0,3).map(p=>p.id)})],91);
 }
});
rule('eulerLine','欧拉线比例','三角形','某三角形的外心 O、重心 G 与垂心 H 共线且 HG=2·GO。',c=>{
 for(const t of c.triangles){const [a,b,d]=t.pts,circ=t.circ;if(!circ)continue;
  const g=V.mul(V.add(V.add(a,b),d),1/3),o={x:circ.x,y:circ.y},h=V.sub(V.add(V.add(a,b),d),V.mul(o,2));
  if(V.dist(g,o)<=c.eps)continue;
  const G=c.P.filter(p=>V.dist(p,g)<=c.eps),O=c.P.filter(p=>V.dist(p,o)<=c.eps),H=c.P.filter(p=>V.dist(p,h)<=c.eps);
  for(const pg of G)for(const po of O)for(const ph of H)if(new Set([pg.id,po.id,ph.id]).size===3)
   c.add('eulerLine',t.ids.join('|')+'|'+sortIds([pg.id,po.id,ph.id]),`${pid(c,po.id)}、${pid(c,pg.id)}、${pid(c,ph.id)} 为 △${triName(c,t)} 欧拉线三心（GH=2·GO）`,[sMark(po.id,ph.id),pMark(pg.id),...triMarks(t)],94);
 }
});
rule('ninePointCircle','九点圆','三角形','显式圆经过三角形三边中点及三高垂足。',c=>{
 for(const t of c.triangles){const [a,b,d]=t.pts,mid=[V.mul(V.add(a,b),.5),V.mul(V.add(b,d),.5),V.mul(V.add(d,a),.5)];
  const foot=(p,u,v)=>{const w=V.sub(v,u),k=V.dot(V.sub(p,u),w)/V.dot(w,w);return V.add(u,V.mul(w,k));};
  const nine=mid.concat([foot(a,b,d),foot(b,a,d),foot(d,a,b)]);
  for(const cc of c.C)if(nine.every(q=>c.near(V.dist(q,{x:cc.x,y:cc.y}),cc.r)))
   c.add('ninePointCircle',t.ids.join('|')+'|'+cc.key,`${c.cname(cc)} 是 △${triName(c,t)} 的九点圆`,[circleMark(cc),...triMarks(t)],97);
 }
});
rule('incircle','三角形内切圆','圆','某已存在圆位于三角形内侧，并分别与三边相切。',c=>{
 const foot=(p,u,v)=>{const w=V.sub(v,u),k=V.dot(V.sub(p,u),w)/V.dot(w,w);return {q:V.add(u,V.mul(w,k)),t:k};};
 for(const t of c.triangles){const [a,b,d]=t.pts;
  for(const cc of c.C){const center={x:cc.x,y:cc.y},signs=[V.cross(V.sub(b,a),V.sub(center,a)),V.cross(V.sub(d,b),V.sub(center,b)),V.cross(V.sub(a,d),V.sub(center,d))];
   if(!signs.every(z=>z>c.eps||z< -c.eps)||!(signs.every(z=>z>0)||signs.every(z=>z<0)))continue;
   if([[a,b],[b,d],[d,a]].every(([x,y])=>{const {q,t:tt}=foot(center,x,y);return tt>-c.eps&&tt<1+c.eps&&c.near(V.dist(center,q),cc.r);}))
    c.add('incircle',t.ids.join('|')+'|'+cc.key,`${c.cname(cc)} 是 △${triName(c,t)} 的内切圆`,[circleMark(cc),...triMarks(t)],94);
  }
 }
});
rule('circlePower','点的幂（割线积）','圆','圆外点 P 经两条割线的交点满足 PA·PB=PC·PD。',c=>{
 for(const cc of c.C)for(const p of c.P){const cp=V.dist(p,{x:cc.x,y:cc.y});if(cp<=cc.r+c.eps)continue;
  const circumference=c.P.filter(x=>x!==p&&c.near(V.dist(x,{x:cc.x,y:cc.y}),cc.r));
  const secants=[];pairs(circumference,(a,b)=>{
   const v=V.sub(a,p),w=V.sub(b,p),len=V.dist(p,a),ll=V.dist(p,b);
   if(len<=c.eps||ll<=c.eps||Math.abs(V.cross(v,w))>c.mode.direction*len*ll||V.dot(v,w)<=0)return;
   secants.push({a,b,value:len*ll});
  });pairs(secants,(a,b)=>{if(new Set([a.a.id,a.b.id,b.a.id,b.b.id]).size!==4)return;
    if(c.nearQuadratic(a.value,b.value))c.add('circlePower',p.id+'|'+cc.key+'|'+sortIds([sortIds([a.a.id,a.b.id]),sortIds([b.a.id,b.b.id])]),
    `${pid(c,p.id)}${pid(c,a.a.id)}·${pid(c,p.id)}${pid(c,a.b.id)} = ${pid(c,p.id)}${pid(c,b.a.id)}·${pid(c,p.id)}${pid(c,b.b.id)}`,[circleMark(cc),sMark(p.id,a.a.id),sMark(p.id,a.b.id),sMark(p.id,b.a.id),sMark(p.id,b.b.id)],87);
  });
 }
});
// Named and virtual incidence configurations: nontrivial intersection facts.
rule('lineIntersectionCircle','两线交点在圆上','圆与切线','两条不同支撑线的交点恰在已显示圆上，无需先创建交点对象。',c=>{
 pairs(c.L,(a,b)=>{const p=lineIntersection(a,b);if(!p)return;
  for(const q of c.C)if(c.near(V.dist(p,{x:q.x,y:q.y}),q.r)&&!triviality.isNamedIncidence(c,p,[q]))
   c.add('lineIntersectionCircle',sortIds([a.key,b.key])+'|'+q.key,`${c.lname(a)} 与 ${c.lname(b)} 的交点在 ${c.cname(q)} 上`,[lineMark(a),lineMark(b),circleMark(q),{type:'coordinate',x:p.x,y:p.y}],89);
 });
});
rule('circleIntersectionLine','两圆交点在线上','圆与切线','两圆的公共点位于某支撑直线；同一公共弦的两个交点并为一条。',c=>{
 pairs(c.C,(a,b)=>{const ps=circleIntersections(a,b,c.eps);if(!ps.length)return;
  for(const l of c.L){const matches=ps.filter(p=>c.distPointLine(p,l)<=c.eps&&!triviality.isNamedIncidence(c,p,[a,b]));
   if(!matches.length)continue;
   c.add('circleIntersectionLine',sortIds([a.key,b.key])+'|'+l.key,`${c.cname(a)}、${c.cname(b)} 的${matches.length===2?'公共弦':'交点'}在 ${c.lname(l)} 上`,
    [circleMark(a),circleMark(b),lineMark(l),...matches.map(p=>({type:'coordinate',x:p.x,y:p.y}))],89);
  }
 });
});
rule('multiCircleCommon','三圆及更多圆共点','圆与切线','把同一点上交会的三圆或更多圆并成一条记录，支持未画出的公共交点。',c=>{
 const seen=new Set();pairs(c.C,(a,b)=>{
  for(const pos of circleIntersections(a,b,c.eps)){
   const circles=c.C.filter(q=>c.near(V.dist(pos,{x:q.x,y:q.y}),q.r));if(circles.length<3||triviality.isNamedIncidence(c,pos,circles))continue;
   const key=sortIds(circles.map(q=>q.key));
   // A shared group may have two genuinely distinct common points.
   // Instead of positional floating keys (unstable under micro-perturbation),
   // identify the branch using the canonical first two centers and orientation.
   const q=circles[0],r=circles[1],cross=(r.x-q.x)*(pos.y-q.y)-(r.y-q.y)*(pos.x-q.x),branch=Math.abs(cross)<=c.eps*10?'T':cross>0?'+':'-';
   const id=key+'|'+branch;if(seen.has(id))continue;seen.add(id);
   c.add('multiCircleCommon',id,`${circles.map(q=>c.cname(q)).join('、')} 共点`,[...circles.map(circleMark),{type:'coordinate',x:pos.x,y:pos.y}],95,`${circles.length} 圆共点（${branch==='T'?'相切':branch==='+'?'正向':'反向'}分支）`);
  }
 });
});
// v3.3: strict geometric predicates; bilateral perturbation is optional and operates only on disposable cloned snapshots.
function perturbed(snapshot,sign,amplitude){
 // Independent deterministic variation of the input snapshots. No PRNG and no
 // mutation: identical scans are reproducible. Magnitude is far below strict eps.
 const hash=s=>{let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return (h>>>0)/4294967295*2-1;};
 const alter=(id,tag)=>sign*amplitude*hash(String(id)+':'+tag);
 return {...snapshot,
  points:(snapshot.points||[]).map(p=>({...p,x:p.x+alter(p.id,'px'),y:p.y+alter(p.id,'py')})),
  lines:(snapshot.lines||[]).map(l=>({...l,x1:l.x1+alter(l.id,'lx'),y1:l.y1+alter(l.id,'ly'),x2:l.x2+alter(l.id,'lx2'),y2:l.y2+alter(l.id,'ly2')})),
  circles:(snapshot.circles||[]).map(c=>({...c,cx:c.cx+alter(c.id,'cx'),cy:c.cy+alter(c.id,'cy'),r:c.r+alter(c.id,'r')}))};
}
function runPass(snapshot,options,active){
 const ctx=makeContext(snapshot,{...options,kinds:active.map(r=>r.id),tolerance:'strict',circlePrepass:active.some(r=>r.id==='concyclic')});
 if(active.some(r=>['rightTriangle','isosceles','equilateral','specialPoint','specialLine','similarTriangles','congruentTriangles','equalArea','areaRatio','quadraticRatio','eulerLine','ninePointCircle','incircle'].includes(r.id)))ctx.triangles=makeTriangles(ctx);else ctx.triangles=[];
 if(active.some(r=>['circleLineTangent','circleCircleTangent'].includes(r.id)))ctx.allCircles=circleCandidates(ctx);else ctx.allCircles=ctx.C;
 const breakdown={};for(const r of active){const before=ctx.entries.length;r.run(ctx);breakdown[r.id]=ctx.entries.length-before;}
 return {ctx,breakdown};
}
function scan(snapshot,options={}){
 const start=typeof performance!=='undefined'?performance.now():Date.now();
 const active=registry.filter(r=>options.kinds?options.kinds.includes(r.id):r.enabled);
 const {ctx,breakdown}=runPass(snapshot,options,active);
 // Perturbation cannot change any relation we report. Rescan only families
 // producing results; reject unstable IDs rather than silently reclassify them.
 let entries=ctx.entries,removed=0;
 if(options.perturb===true&&entries.length){const kinds=new Set(entries.map(e=>e.kind)),stableRules=active.filter(r=>kinds.has(r.id));
  const amplitude=ctx.eps*0.012;
  const plus=new Set(runPass(perturbed(snapshot,1,amplitude),options,stableRules).ctx.entries.map(x=>x.id));
  const minus=new Set(runPass(perturbed(snapshot,-1,amplitude),options,stableRules).ctx.entries.map(x=>x.id));
  entries=entries.filter(e=>plus.has(e.id)&&minus.has(e.id));removed=ctx.entries.length-entries.length;
 }
 const source=provenance.annotate(entries,snapshot,ctx);
 operandLogic.annotate(entries,snapshot,ctx);
 const simplified=semantic.reduce(entries,ctx);entries=simplified.entries;
 const entailments=provenance.consolidate(entries,ctx,snapshot);entries=entailments.entries;
 const operandReduction=operandLogic.reduce(entries,ctx);entries=operandReduction.entries;
 provenance.score(entries,snapshot);
 entries.sort((a,b)=>b.score-a.score||a.kind.localeCompare(b.kind)||a.id.localeCompare(b.id));
 return {entries,count:entries.length,breakdown,quadraticTruncatedOperands:active.includes('quadraticRatio')?Math.max(0,ctx.segments.length-(Number.isInteger(options.quadraticProductLimit)?Math.max(0,options.quadraticProductLimit):66)):0,quadraticTruncatedExpressionOperands:active.includes('quadraticRatio')?Math.max(0,ctx.segments.length+ctx.CAll.length-(Number.isInteger(options.quadraticExpressionLimit)?Math.max(0,options.quadraticExpressionLimit):20)):0,unstableRejected:removed,semanticRemoved:simplified.suppressed+entailments.suppressed+operandReduction.suppressed,constructionFacts:source.directCount,constructionConsequences:source.definitionFacts,implicationGraph:operandLogic.decorateGraph(provenance.buildImplicationGraph(entries,snapshot),entries),span:ctx.span,revision:snapshot.revision||null,elapsedMs:Math.round((typeof performance!=='undefined'?performance.now():Date.now())-start),objects:{points:ctx.P.length,originalPoints:ctx.rawP.length,pointClusters:ctx.pointGroups.groups.length,explicitLines:ctx.E.length,implicitLines:ctx.VLines.length,supportLines:ctx.L.length,circles:ctx.C.length,circleGroups:ctx.circleGroups.groups.length},tolerance:'strict',perturbations:options.perturb===true?2:0};
}
return {registry:registry.map(({id,label,group,help,enabled})=>({id,label,group,help,enabled})),scan};
});
