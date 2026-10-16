/* v3.1 geometric identity layer: point clusters, supporting lines and circles.
 * These are geometric equivalence classes; drawing object IDs remain intact as aliases.
 */
(function(root,factory){const api=factory();root.GeomRelationGroups=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const d=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const slots=[-1,0,1];
function clusterPoints(P,eps){
 // Greedy complete-link grouping: every pair within each cluster must satisfy
 // the strict tolerance. A~B and B~C alone never force A~C.
 const grid=new Map(),groups=[];
 const cell=p=>[Math.floor(p.x/eps),Math.floor(p.y/eps)];
 for(const p of P){const [x,y]=cell(p),candidates=new Set();
  for(const dx of slots)for(const dy of slots)for(const index of grid.get((x+dx)+','+(y+dy))||[])candidates.add(index);
  let target=-1;for(const index of candidates)if(groups[index].every(q=>d(p,q)<=eps)){target=index;break;}
  if(target<0){target=groups.length;groups.push([]);}
  groups[target].push(p);
  const key=x+','+y,arr=grid.get(key)||[];arr.push(target);grid.set(key,arr);
 }
 for(const g of groups)g.sort((a,b)=>a.id.localeCompare(b.id));
 groups.sort((a,b)=>a[0].id.localeCompare(b[0].id));
 return {groups,points:groups.map(g=>({...g[0],aliases:g.map(p=>p.id)})),identity:new Map(groups.flatMap(g=>g.map(p=>[p.id,g[0].id]))),aliases:new Map(groups.map(g=>[g[0].id,g]))};
}
function groupLines(rawLines,P,eps,direction,distPointLine){
 const groups=[];
 const same=(a,b)=>Math.abs(a.dx*b.dy-a.dy*b.dx)<=direction&&distPointLine({x:b.x,y:b.y},a)<=eps;
 // Three-coordinate normalized support buckets, rather than O(number of lines²).
 const grid=new Map(),pitchAngle=Math.max(direction*4,1e-13),pitchOffset=Math.max(eps*4,1e-13);
 const coord=l=>{let x=l.dx,y=l.dy;if(x<0||(x===0&&y<0)){x=-x;y=-y;}return [Math.floor(x/pitchAngle),Math.floor(y/pitchAngle),Math.floor((-y*l.x+x*l.y)/pitchOffset)];};
 for(const l of rawLines){const p=coord(l),candidate=new Set();for(const i of slots)for(const j of slots)for(const k of slots)for(const idx of grid.get((p[0]+i)+','+(p[1]+j)+','+(p[2]+k))||[])candidate.add(idx);
  // Close to the direction-chart discontinuity, search the opposite orientation.
  if(Math.abs(l.dx)<direction*2)for(const g of groups)if(same(l,g.rep))candidate.add(groups.indexOf(g));
  let match=-1;for(const idx of candidate)if(same(l,groups[idx].rep)){match=idx;break;}
  if(match<0){match=groups.length;groups.push({rep:l,members:[],pointIds:[]});const key=p.join(',');const a=grid.get(key)||[];a.push(match);grid.set(key,a);}
  groups[match].members.push(l);
 }
 const byKey=new Map(),representatives=[];
 for(const g of groups){g.rep.aliases=g.members.filter(l=>l!==g.rep);for(const l of g.members)byKey.set(l.key,g.rep.key);
  g.pointIds=P.filter(p=>distPointLine(p,g.rep)<=eps).map(p=>p.id);
  g.rep.pointIds=g.pointIds;representatives.push(g.rep);
 }
 return {groups,lines:representatives,byKey};
}
function groupCircles(P,explicit,circum,eps,span,includeImplicit){
 const groups=[],grid=new Map(),pitch=Math.max(eps*4,1e-14);
 const same=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y)<=eps&&Math.abs(a.r-b.r)<=eps;
 function put(c,pts){if(!Number.isFinite(c.r)||!(c.r>eps)||c.r>span*1e7)return;
  const x=Math.floor(c.x/pitch),y=Math.floor(c.y/pitch),r=Math.floor(c.r/pitch);let match=-1;
  for(const i of slots){if(match>=0)break;for(const j of slots){if(match>=0)break;for(const k of slots){for(const idx of grid.get((x+i)+','+(y+j)+','+(r+k))||[])if(same(c,groups[idx].rep)){match=idx;break;}if(match>=0)break;}}}
  if(match<0){match=groups.length;groups.push({rep:c,members:[],pointIds:new Set()});const key=x+','+y+','+r,arr=grid.get(key)||[];arr.push(match);grid.set(key,arr);}
  const g=groups[match];if(c.kind==='explicit')g.members.push(c);for(const p of pts)g.pointIds.add(p.id);
 }
 for(const c of explicit)put(c,P.filter(p=>Math.abs(d(p,{x:c.x,y:c.y})-c.r)<=eps));
 if(includeImplicit)for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++)for(let k=j+1;k<P.length;k++){
  const [a,b,c]=[P[i],P[j],P[k]],ab=d(a,b),ac=d(a,c),bc=d(b,c);
  if(Math.min(ab,ac,bc)<=eps*100)continue;
  const signed=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
  if(Math.abs(signed)<=1e-7*ab*ac)continue; // unstable circumcenter
  const circ=circum(a,b,c);if(circ)put({kind:'implicit',key:'G:'+a.id+'|'+b.id+'|'+c.id,x:circ.x,y:circ.y,r:circ.r},[a,b,c]);
 }
 const merged=groups.filter(g=>g.rep.kind==='explicit'||g.pointIds.size>=4);
 const explicitCanonical=[],byKey=new Map();for(const g of merged){g.pointIds=[...g.pointIds].sort();if(g.members.length){g.rep=g.members[0];g.rep.aliases=g.members.slice(1);g.rep.pointIds=g.pointIds;explicitCanonical.push(g.rep);for(const e of g.members)byKey.set(e.key,g.rep.key);}else{g.rep.pointIds=g.pointIds;g.rep.aliases=[];}}
 return {groups:merged,explicit:explicitCanonical,byKey};
}
return {clusterPoints,groupLines,groupCircles};
});
