/* v3.1 logical entailment reducer. Numeric witnesses are not changed. */
(function(root,factory){const api=factory();root.GeomRelationSemantic=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const keyPair=(a,b)=>[a,b].sort().join('|');
function reduce(entries,c){
 const lineKey=ref=>{if(!ref)return null;const key=ref.type==='pair'?'P:'+keyPair(ref.a,ref.b):'L:'+ref.id;
  return c.lineFamily.get(key)||key;};
 const supportIds=entry=>(entry.marks||[]).filter(m=>m.type==='line').map(m=>lineKey(m.ref));
 const perpendicular=entries.filter(e=>e.kind==='perpendicular').map(e=>new Set(supportIds(e)));
 const specialBisectors=entries.filter(e=>e.kind==='specialLine'&&e.detail?.includes('角平分线'));
 const suppress=new Set(),aliases=entry=>{if(!entry.searchTags)entry.searchTags=[];return entry.searchTags;};
 // Geometric implications are independent of UI checkboxes: reduce only if
 // the stronger witness is actually present, and keep its search synonyms.
 const orthFamilies=entries.filter(e=>e.kind==='orthocentricSystem').map(e=>({entry:e,ids:new Set(e.marks.filter(m=>m.type==='point').map(m=>m.id))}));
 const congFamilies=entries.filter(e=>e.kind==='congruentTriangles').map(e=>({entry:e,triangles:new Set(e.id.split(':').slice(1).join(':').split(';'))}));
 const squareFamilies=entries.filter(e=>e.kind==='square').map(e=>({entry:e,ids:new Set(e.marks.filter(m=>m.type==='segment').flatMap(m=>[m.a,m.b]))}));
 const linePoints=l=>{const g=c.lineGroups.groups.find(g=>g.rep.key===lineKey(l));return g?.pointIds||[];};
 const collinear=c.lineGroups.groups.filter(g=>g.pointIds.length>=3);
 const circleGroup=c.circleGroups.groups.filter(g=>g.pointIds.length>=4);
 for(const e of entries){
  if(e.kind==='specialPoint'&&e.detail==='垂心'){
   const ids=new Set(e.marks.filter(m=>m.type==='point').map(m=>m.id));
   for(const o of orthFamilies)if(ids.size===4&&[...ids].every(x=>o.ids.has(x))){suppress.add(e.id);aliases(o.entry).push('垂心','三角形特殊点');break;}
  }
  if(e.kind==='reflection'){
   const seg=e.marks.find(m=>m.type==='segment'),line=e.marks.find(m=>m.type==='line');
   if(seg&&line){const pts=linePoints(line.ref).filter(x=>x!==seg.a&&x!==seg.b);
    for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){
     const A=[seg.a,pts[i],pts[j]].sort().join(','),B=[seg.b,pts[i],pts[j]].sort().join(',');
     const witness=congFamilies.find(g=>g.triangles.has(A)&&g.triangles.has(B));
     if(witness){suppress.add(e.id);aliases(witness.entry).push('轴对称','关于'+c.display(line.ref.id||line.ref.a||'')+'对称',e.title);}
    }
   }
  }
  if(e.kind==='equalArea'){
   const triangles=e.id.split(':').slice(1).join(':').split(';');
   const witness=congFamilies.find(g=>triangles.length>=2&&triangles.every(t=>g.triangles.has(t)));
   if(witness){suppress.add(e.id);aliases(witness.entry).push('等面积','面积相等',e.title);}
  }
  if(e.kind==='concyclic'){
   const ids=new Set(e.marks.filter(m=>m.type==='point').map(m=>m.id));
   if(ids.size===4)for(const s of squareFamilies)if([...ids].every(x=>s.ids.has(x))){suppress.add(e.id);aliases(s.entry).push('四点共圆','共圆',e.title);break;}
  }
  if(e.kind==='rotation'){
   const angle=e.marks.find(m=>m.type==='angle');
   if(angle?.label==='90°')for(const s of squareFamilies){
    if([angle.a,angle.v,angle.b].every(id=>s.ids.has(id))){suppress.add(e.id);aliases(s.entry).push('90度旋转','定角旋转');break;}
   }
  }

  if(e.kind==='pointOnLine'){
   const p=e.marks.find(m=>m.type==='point'),l=e.marks.find(m=>m.type==='line');
   const obj=c.E.find(x=>x.id===l?.ref?.id);
   if(p&&obj?.extent==='line'&&collinear.some(g=>g.pointIds.includes(p.id)&&g.members.some(x=>x.key===obj.key)))suppress.add(e.id);
  }
  if(e.kind==='pointOnCircle'){
   const p=e.marks.find(m=>m.type==='point'),q=e.marks.find(m=>m.type==='circle');
   if(p&&q?.ref?.type==='object'&&circleGroup.some(g=>g.pointIds.includes(p.id)&&g.members.some(x=>x.id===q.ref.id)))suppress.add(e.id);
  }
  if(e.kind==='perpFoot'){
   const seg=e.marks.find(m=>m.type==='segment'),line=e.marks.find(m=>m.type==='line');
   if(seg&&line){const a=lineKey({type:'pair',a:seg.a,b:seg.b}),b=lineKey(line.ref);
    if(perpendicular.some(s=>s.has(a)&&s.has(b)))suppress.add(e.id);
   }
  }
  if(e.kind==='angleBisector'){
   const a=e.marks.find(m=>m.type==='angle'),l=e.marks.find(m=>m.type==='line');
   if(!a||!l)continue;
   const groupLine=lineKey(l.ref),triangleKey=[a.a,a.v,a.b].sort().join('|');
   if(specialBisectors.some(s=>{
    const triangle=[...new Set(s.marks.filter(m=>m.type==='segment').flatMap(m=>[m.a,m.b]))].sort().join('|');
    return triangle===triangleKey&&supportIds(s).includes(groupLine);
   }))suppress.add(e.id);
  }
  if(e.kind==='perpendicular'){
   const groups=supportIds(e).map(k=>c.lineGroups.groups.find(g=>g.rep.key===k));
   if(groups.some(g=>!g))continue;
   for(let i=0;i<groups.length;i++)for(let j=i+1;j<groups.length;j++){
    const common=groups[i].pointIds.filter(id=>groups[j].pointIds.includes(id));
    if(common.some(id=>groups[i].pointIds.some(x=>x!==id)&&groups[j].pointIds.some(x=>x!==id))){aliases(e).push('直角三角形','垂足','特殊角90°');i=groups.length;break;}
   }
  }
  if(e.kind==='distanceRatio'&&e.metric?.factor===1){
   const arr=e.metric.high.filter(x=>x.type==='segment');
   const ps=arr.map(x=>{const k=x.key.split(':')[1];return k?.split('|')||[];});
   let found=false;for(let i=0;i<ps.length&&!found;i++)for(let j=i+1;j<ps.length&&!found;j++){
    const common=ps[i].filter(x=>ps[j].includes(x));if(common.length!==1)continue;
    const p=c.pMap.get(common[0]),a=c.pMap.get(ps[i].find(x=>x!==common[0])),b=c.pMap.get(ps[j].find(x=>x!==common[0]));
    if(p&&a&&b&&Math.abs((a.x-p.x)*(b.y-p.y)-(a.y-p.y)*(b.x-p.x))>c.eps*c.span)found=true;
   }
   if(found)aliases(e).push('等腰三角形','等腰','两边相等');
  }
 }
 const remaining=entries.filter(e=>!suppress.has(e.id));
 return {entries:remaining,suppressed:suppress.size};
}
return {reduce};
});
