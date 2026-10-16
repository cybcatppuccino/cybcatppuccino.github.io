/* GeomTool 4.2 — conclusion-guided local auxiliary constructor.
 * Candidate generation is conjectural. Only a successful Kernel construction
 * may be applied; formal DDAR consequences are evaluated separately.
 */
(function(root,factory){const api=factory();root.GeomAuxiliary=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const uniq=a=>[...new Set(a)];
const pair=(a,b)=>[a,b].sort().join('|');
function suggest(snapshot,inference,entries=[],opts={}){
 const t0=Date.now(),budget=Math.min(1000,Math.max(25,opts.budget||380));
 const N=(snapshot.nodes||[]).filter(n=>n.valid&&!n.style?.system),byId=new Map(N.map(n=>[n.id,n]));
 const points=N.filter(n=>n.geom?.kind==='point'),lines=N.filter(n=>n.geom?.kind==='line'),circles=N.filter(n=>n.geom?.kind==='circle');
 const P=new Map(points.map(n=>[n.id,n.geom]));const candidates=[],seen=new Set(),occupied=new Set();
 for(const n of N){const ids=n.inputs||[],o=n.op;if(o==='midpoint'||o==='diameterCircle'||o==='circumcircle'){occupied.add(o+':'+ids.slice().sort().join('|'));}else occupied.add(o+':'+ids.join('|'));}
 // Proof-conclusion incidence is more useful than simply enumerating all triples.
 const score=new Map(points.map(n=>[n.id,0]));
 for(const f of inference?.facts||[])for(const id of f.args)if(P.has(id))score.set(id,score.get(id)+(f.depth===0?2.2:1.4));
 const unproved=(inference?.matches||[]).filter(m=>m.status==='unknown');const missing=new Set(unproved.map(m=>m.id));
 const titleById=new Map(entries.map(e=>[e.id,e]));
 for(const x of unproved){const e=titleById.get(x.id);if(!e)continue;for(const m of e.marks||[]){for(const name of [m.a,m.b,m.v,m.id,m.point,...(m.ids||[])])if(score.has(name))score.set(name,score.get(name)+2.7);if(m.ref?.type==='pair')for(const name of [m.ref.a,m.ref.b])if(score.has(name))score.set(name,score.get(name)+2.3);}}
 const ranked=points.slice().sort((a,b)=>(score.get(b.id)||0)-(score.get(a.id)||0)||a.id.localeCompare(b.id));
 const focal=ranked.slice(0,Math.min(points.length,opts.maxPoints||16)).map(n=>n.id);
 const radius=points.length?Math.max(1,Math.hypot(Math.max(...points.map(p=>p.geom.x))-Math.min(...points.map(p=>p.geom.x)),Math.max(...points.map(p=>p.geom.y))-Math.min(...points.map(p=>p.geom.y)))):1;
 const distance=(a,b)=>{const p=P.get(a),q=P.get(b);return p&&q?Math.hypot(p.x-q.x,p.y-q.y):Infinity;};
 const avoidPoint=(p)=>points.some(n=>Math.hypot(n.geom.x-p.x,n.geom.y-p.y)<=1e-7*radius);
 const semantic=(ids)=>ids.reduce((sum,id)=>sum+(score.get(id)||0),0);
 function offer(op,ids,title,why,initial,category,bonus=0,params={}){
  if(candidates.length>=budget*3||ids.some(id=>!byId.has(id)))return;
  const sym=['midpoint','diameterCircle','circumcircle'].includes(op);
  const signature=op+':'+(sym?ids.slice().sort().join('|'):ids.join('|'));
  if(occupied.has(signature)||seen.has(signature))return;
  seen.add(signature);
  const coords=initial||null;if(coords&&coords.kind==='point'&&avoidPoint(coords))return;
  const refs=uniq(ids.flatMap(id=>[id,...(byId.get(id)?.inputs||[])]));
  candidates.push({op,inputs:ids,params,title,why,category,score:Math.round(10+Math.min(56,semantic(refs)*.30)+bonus),preview:coords});
 }
 const mid=(a,b)=>{const p=P.get(a),q=P.get(b);return {kind:'point',x:(p.x+q.x)/2,y:(p.y+q.y)/2}};
 const foot=(p,a,b)=>{const z=P.get(p),x=P.get(a),y=P.get(b),dx=y.x-x.x,dy=y.y-x.y,t=((z.x-x.x)*dx+(z.y-x.y)*dy)/(dx*dx+dy*dy);return {kind:'point',x:x.x+t*dx,y:x.y+t*dy};};
 const linePointDistance=(p,a,b)=>{const u=P.get(p),x=P.get(a),y=P.get(b);return Math.abs((y.x-x.x)*(u.y-x.y)-(y.y-x.y)*(u.x-x.x))/distance(a,b);};
 const goodPair=(a,b)=>a!==b&&distance(a,b)>radius*1e-7;
 const supporting=new Map();for(const f of inference?.facts||[]){if(f.kind==='isos'){
  const [v,a,b]=f.args;if(P.has(v)&&P.has(a)&&P.has(b)){
   supporting.set(pair(a,b),{vertex:v,reason:`已推得 ${v}${a} = ${v}${b}，考虑等腰三角形的底边中点、底边高及对称轴`});
  }
 }}
 // Known equal sides -> midpoint & altitude. Midpoint of a chord / mid-segment
 // can also tie together proved collinearities and parallel families.
 for(const [base,hit] of supporting){const [a,b]=base.split('|');const v=hit.vertex;
  if(!goodPair(a,b))continue;
  offer('midpoint',[a,b],`取 ${a}${b} 的中点`,hit.reason,mid(a,b),'中点 · 等腰',21);
  if(linePointDistance(v,a,b)>radius*1e-7)offer('foot3',[v,a,b],`作 ${v} 到 ${a}${b} 的垂足`,hit.reason,foot(v,a,b),'垂足 · 等腰',25);
  offer('perpBisector',[a,b],`作 ${a}${b} 中垂线`,`等腰底边的垂直平分线将两个等距条件勾连`,null,'垂直平分线',12);
  offer('circle',[v,a],`以 ${v} 为圆心作过 ${a} 的圆`,`已知 ${v}${a}=${v}${b}，辅助圆可同时穿过 ${a} 与 ${b}`,null,'辅助圆 · 等腰',24);
 }
 // Midpoints from cited facts and unmatched high-scoring relational predicates.
 for(let i=0;i<focal.length;i++)for(let j=i+1;j<focal.length;j++){
  const [a,b]=[focal[i],focal[j]];if(!goodPair(a,b))continue;
  offer('midpoint',[a,b],`添加 ${a}${b} 的中点`,`尝试中位线定理、中心对称或把等长/平行结论转化为新的构造桥梁`,mid(a,b),'中点 · 桥接',-6);
 }
 // Form new exact line-line intersections only for existing carriers. Never
 // infer an intersection by intersecting arbitrary implicit point pairs here.
 const carrier=lines.concat(circles).slice(0,Math.min(58,lines.length+circles.length));
 function hits(a,b){const u=a.geom,v=b.geom;
  if(u.kind==='line'&&v.kind==='line'){
   const det=u.d.x*v.d.y-u.d.y*v.d.x;if(Math.abs(det)<=1e-11)return [];
   const dx=v.p.x-u.p.x,dy=v.p.y-u.p.y,t=(dx*v.d.y-dy*v.d.x)/det,s=(dx*u.d.y-dy*u.d.x)/det;
   if(u.extent==='segment'&&(t<0||t>1)||v.extent==='segment'&&(s<0||s>1)||u.extent==='ray'&&t<0||v.extent==='ray'&&s<0)return [];
   return [{x:u.p.x+t*u.d.x,y:u.p.y+t*u.d.y,branch:0}];
  }
  const line=u.kind==='line'?u:v.kind==='line'?v:null,circle=u.kind==='circle'?u:v.kind==='circle'?v:null;
  if(line&&circle){const d=line.d,f={x:line.p.x-circle.c.x,y:line.p.y-circle.c.y},aa=d.x*d.x+d.y*d.y,bb=2*(f.x*d.x+f.y*d.y),cc=f.x*f.x+f.y*f.y-circle.r*circle.r,disc=bb*bb-4*aa*cc;if(disc<-1e-10)return [];
   const ts=disc<1e-10?[-bb/(2*aa)]:[(-bb-Math.sqrt(disc))/(2*aa),(-bb+Math.sqrt(disc))/(2*aa)];return ts.map((t,branch)=>({x:line.p.x+t*d.x,y:line.p.y+t*d.y,branch,t})).filter(h=>line.extent==='line'||(line.extent==='ray'?h.t>=-1e-10:h.t>=-1e-10&&h.t<=1+1e-10));
  }
  if(u.kind==='circle'&&v.kind==='circle'){
   const dx=v.c.x-u.c.x,dy=v.c.y-u.c.y,d=Math.hypot(dx,dy);if(d<1e-9||d>u.r+v.r||d<Math.abs(u.r-v.r))return [];
   const r=(u.r*u.r-v.r*v.r+d*d)/(2*d),h=Math.sqrt(Math.max(0,u.r*u.r-r*r)),x=u.c.x+r*dx/d,y=u.c.y+r*dy/d;
   return h<1e-9?[{x,y,branch:0}]:[{x:x+h*dy/d,y:y-h*dx/d,branch:0},{x:x-h*dy/d,y:y+h*dx/d,branch:1}];
  }return [];
 }
 for(let i=0;i<carrier.length;i++)for(let j=i+1;j<carrier.length;j++){
  const a=carrier[i],b=carrier[j];for(const h of hits(a,b)){
   if(avoidPoint(h)||!Number.isFinite(h.x)||!Number.isFinite(h.y))continue;
   const connector=uniq([...(a.inputs||[]),...(b.inputs||[])]).filter(z=>P.has(z));
   offer('intersection',[a.id,b.id],`${a.id} ∩ ${b.id}（分支 ${h.branch}）`,`将两条已存在的载体关联到同一新点；可触发新的共线、共圆及角度传递`,{kind:'point',x:h.x,y:h.y},'交点 · 闭合',16+Math.min(22,semantic(connector)*.16),{branch:h.branch});
  }
 }
 // When no explicit carriers exist, make a meaningful two-line construction
 // plan from two high-incidence point pairs, then take their intersection.
 // It is previewed as a point, but committed atomically via Kernel.load.
 const edges=[];for(let i=0;i<Math.min(10,focal.length);i++)for(let j=i+1;j<Math.min(10,focal.length);j++){
  const a=focal[i],b=focal[j];if(goodPair(a,b))edges.push({a,b,importance:semantic([a,b])});
 }
 edges.sort((a,b)=>b.importance-a.importance);
 let bridges=0;const strong=edges.slice(0,18);
 for(let i=0;i<strong.length&&bridges<14;i++)for(let j=i+1;j<strong.length&&bridges<14;j++){
  const u=strong[i],v=strong[j],ids=[u.a,u.b,v.a,v.b];if(new Set(ids).size<4)continue;
  const a=P.get(u.a),b=P.get(u.b),c=P.get(v.a),d=P.get(v.b),ux=b.x-a.x,uy=b.y-a.y,vx=d.x-c.x,vy=d.y-c.y;
  const delta=ux*vy-uy*vx;if(Math.abs(delta)<1e-7*Math.hypot(ux,uy)*Math.hypot(vx,vy))continue;
  const t=((c.x-a.x)*vy-(c.y-a.y)*vx)/delta,xy={kind:'point',x:a.x+t*ux,y:a.y+t*uy};
  if(!Number.isFinite(xy.x)||!Number.isFinite(xy.y)||Math.hypot(xy.x,xy.y)>radius*15+Math.hypot(a.x,a.y)||avoidPoint(xy))continue;
  offer('compoundIntersection',ids,`补作 ${u.a}${u.b} ∩ ${v.a}${v.b}`,
   `自动先建立 ${u.a}${u.b} 与 ${v.a}${v.b} 两条辅助直线，再求交点，以联通高关联度的证明点集`,xy,'交点 · 自动补线',21);
  bridges++;
 }
 const tri=new Map();
 for(const f of inference?.facts||[]){
  if(f.kind==='cyclic')for(let i=0;i<4;i++){const a=f.args.filter((x,k)=>k!==i);tri.set(a.slice().sort().join('|'),a);}
  if(f.kind==='isos'){const [v,a,b]=f.args;tri.set([v,a,b].sort().join('|'),[v,a,b]);}
 }
 // Capped high-relevance triangles, not cubic brute force on all points.
 const heads=focal.slice(0,Math.min(9,focal.length));for(let i=0;i<heads.length;i++)for(let j=i+1;j<heads.length;j++)for(let k=j+1;k<heads.length;k++){
  const ids=[heads[i],heads[j],heads[k]];tri.set(ids.slice().sort().join('|'),ids);
 }
 let nc=0;for(const ids of tri.values()){if(nc++>90)break;const [a,b,c]=ids;if(![a,b,c].every(z=>P.has(z)))continue;
  const x=P.get(a),y=P.get(b),z=P.get(c),area=Math.abs((y.x-x.x)*(z.y-x.y)-(y.y-x.y)*(z.x-x.x));if(area<1e-7*radius*radius)continue;
  offer('circumcircle',ids,`作 ${a}${b}${c} 的外接圆`,`建立新的四点共圆/同弧所对角桥梁；优先选择已涉及推理结论的三个点`,null,'辅助圆 · 四点共圆',tri.has(ids.slice().sort().join('|'))?2:0);
  if(score.get(a)>3&&score.get(b)>3)offer('diameterCircle',[a,b],`作以 ${a}${b} 为直径的圆`,`利用泰勒斯定理将圆周结论与直角/垂足结论连接`,null,'辅助圆 · 泰勒斯',5);
 }
 // Through prominent vertex: geometric line creation guided by existing
 // line objects; additional directions become DDAR dependencies after applying.
 const shortLines=lines.filter(n=>n.inputs.length===2&&P.has(n.inputs[0])&&P.has(n.inputs[1])).slice(0,18);
 for(const p of focal.slice(0,8))for(const l of shortLines){const [a,b]=l.inputs;if([a,b].includes(p)||linePointDistance(p,a,b)<radius*1e-6)continue;
  offer('parallel3',[p,a,b],`过 ${p} 作 ${a}${b} 的平行线`,`通过平行线实现同位角、内错角或相似三角形的关系桥接`,null,'平行线 · 桥接',-2);
  offer('perpendicular3',[p,a,b],`过 ${p} 作 ${a}${b} 的垂线`,`引入直角、垂足及潜在的圆周角结构`,null,'垂线 · 桥接',-3);
 }
 // Reject any candidate whose reference ids are not valid graph nodes.
 candidates.sort((a,b)=>b.score-a.score||a.category.localeCompare(b.category)||a.title.localeCompare(b.title));
 const diversified=[],categoryCount=new Map();for(const c of candidates){const n=categoryCount.get(c.category)||0;if(n>=7)continue;categoryCount.set(c.category,n+1);diversified.push(c);if(diversified.length>=Math.min(32,opts.limit||25))break;}
 return {candidates:diversified,generated:candidates.length,elapsedMs:Date.now()-t0};
}
function execute(kernel,candidate){
 if(candidate.op!=='compoundIntersection')return kernel.execute({op:candidate.op,inputs:candidate.inputs,params:candidate.params||{}});
 const [a,b,c,d]=candidate.inputs;
 const l=kernel.execute({op:'line',inputs:[a,b]}),m=kernel.execute({op:'line',inputs:[c,d]});
 return kernel.execute({op:'intersection',inputs:[l.id,m.id],params:{branch:0}});
}
return {suggest,execute};
});
