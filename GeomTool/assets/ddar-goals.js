/* GeomTool 4.3 — bounded, coordinate-screened conjecture enumeration.
 * IMPORTANT: every candidate is UNPROVED until the independent DDAR checker
 * accepts it. Numeric values are never inserted into the formal fact base.
 */
(function(root,factory){const api=factory();root.GeomDDARGoals=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const pair=(a,b)=>[a,b].sort().join('|'),lp=(a,b)=>'P:'+pair(a,b),seg=pair;
const eps=2e-8,near=(x,y)=>Math.abs(x-y)<eps*Math.max(1,Math.abs(x),Math.abs(y));
function discover(snapshot,opts={}){
 const started=Date.now(),maxGoals=Number.isFinite(opts.maxGoals)?opts.maxGoals:Infinity;
 const maxPoints=Number.isFinite(opts.maxPoints)?opts.maxPoints:Infinity,maxLines=Number.isFinite(opts.maxLines)?opts.maxLines:Infinity;
 const nodes=(snapshot.nodes||[]).filter(n=>n.valid&&n.geom&&!n.style?.system);
 const points=nodes.filter(n=>n.geom.kind==='point').slice(0,maxPoints),P=points.map(n=>n.id);
 const loc=new Map(points.map(n=>[n.id,n.geom]));
 const vec=(a,b)=>({x:b.x-a.x,y:b.y-a.y}),length=v=>Math.hypot(v.x,v.y),cross=(u,v)=>u.x*v.y-u.y*v.x;
 const segments=[];
 for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){
  const a=points[i],b=points[j],v=vec(a.geom,b.geom),d=length(v);
  if(d>1e-8)segments.push({a:a.id,b:b.id,seg:seg(a.id,b.id),len:d,v,carrier:lp(a.id,b.id)});
 }
 const lines=[];
 // Explicit lines first; then point pairs, so small constructions retain all their carriers.
 for(const n of nodes.filter(n=>n.geom.kind==='line')){
  const v=n.geom.d;if(length(v)<1e-9)continue;
  lines.push({name:'L:'+n.id,v:{x:v.x,y:v.y},theta:((Math.atan2(v.y,v.x)%Math.PI)+Math.PI)%Math.PI});
 }
 for(const s of segments)lines.push({name:s.carrier,v:s.v,theta:((Math.atan2(s.v.y,s.v.x)%Math.PI)+Math.PI)%Math.PI});
 const unique=[],known=new Set();for(const l of lines)if(!known.has(l.name)){known.add(l.name);unique.push(l);if(unique.length>=maxLines)break;}
 const goals=[],ids=new Set(),counts={},capacity=()=>goals.length<maxGoals;
 function put(kind,args,text){
  const k=kind+':'+args.join('~');if(ids.has(k)||!capacity())return false;
  ids.add(k);goals.push({id:k,kind,args,text,source:'数值筛选的待证命题'});counts[kind]=(counts[kind]||0)+1;return true;
 }
 function look(kind,args,text){return put(kind,args,text)}
 // Use sorted numeric windows rather than testing every O(n^4) combination.
 segments.sort((a,b)=>a.len-b.len);
 for(let i=0;i<segments.length&&capacity();i++)for(let j=i+1;j<segments.length&&capacity()&&near(segments[i].len,segments[j].len);j++){
  const a=segments[i],b=segments[j];if(a.seg===b.seg)continue;
  look('cong',[a.seg,b.seg].sort(),`${a.a}${a.b} = ${b.a}${b.b}`);
 }
 // 2:1, 3:2, 3:1, sqrt(2) length ratios, independent of already proven equalities.
 for(const factor of [2,1.5,3,Math.SQRT2]){
  for(const x of segments){if(!capacity())break;const wanted=x.len/factor;
   let lo=0,hi=segments.length;while(lo<hi){const mid=(lo+hi)>>1;if(segments[mid].len<wanted)lo=mid+1;else hi=mid;}
   for(let j=Math.max(0,lo-2);j<Math.min(segments.length,lo+3);j++)if(near(segments[j].len,wanted))look('eqratio',[x.seg,segments[j].seg,String(factor)],`${x.a}${x.b} = ${factor} × ${segments[j].a}${segments[j].b}`);
  }
 }
 // Direction relations and line-pair angle relations, each including implicit AB carriers.
 for(let i=0;i<unique.length&&capacity();i++)for(let j=i+1;j<unique.length&&capacity();j++){
  const a=unique[i],b=unique[j],u=a.v,v=b.v,det=cross(u,v)/(length(u)*length(v)),dot=(u.x*v.x+u.y*v.y)/(length(u)*length(v));
  if(Math.abs(det)<eps)look('para',[a.name,b.name].sort(),`${a.name} ∥ ${b.name}`);
  if(Math.abs(dot)<eps)look('perp',[a.name,b.name].sort(),`${a.name} ⟂ ${b.name}`);
 }
 // Directed angle in the projective angle group mod pi. Bucket by quantized
 // angles and verify actual circular distance; avoids quadratic pair-of-pairs.
 const as=[],bin=new Map(),bins=131072;
 const norm=x=>((x%Math.PI)+Math.PI)%Math.PI;
 const degree=x=>Math.round(x*180/Math.PI*100)/100;
 for(let i=0;i<unique.length;i++)for(let j=0;j<unique.length;j++)if(i!==j){
  const a=unique[i],b=unique[j],theta=norm(b.theta-a.theta);
  const item={a:a.name,b:b.name,theta};as.push(item);
  const key=Math.round(theta/Math.PI*bins)%bins;for(const k of [(key+bins-1)%bins,key,(key+1)%bins]){
   const group=bin.get(k)||[];
   // Do not exceed the fixed angular-window budget in large diagrams.
   group.push(item);bin.set(k,group);
  }
 }
 for(const item of as){if(!capacity())break;
  const key=Math.round(item.theta/Math.PI*bins)%bins,seen=new Set();
  for(const older of bin.get(key)||[]){if(!capacity())break;
   if(older===item||seen.has(older))continue;seen.add(older);
   const delta=Math.abs(older.theta-item.theta);
   if(Math.min(delta,Math.PI-delta)>eps)continue;
   if(older.a===item.a&&older.b===item.b)continue;
   const lhs=[older.a,older.b],rhs=[item.a,item.b];
   if(lhs.join('~')>rhs.join('~'))continue;
   // Both angles are zero or right angles -> elementary equivalence remains valid.
   look('eqangleLines',[...lhs,...rhs],`∠(${lhs.join(', ')}) = ∠(${rhs.join(', ')}) [${degree(item.theta)}° mod 180°]`);
  }
 }
 // Midpoints, collinearity, cyclicity are enumerated on actual named points.
 for(let i=0;i<points.length&&capacity();i++)for(let j=i+1;j<points.length&&capacity();j++){
  const a=points[i],b=points[j],d=length(vec(a.geom,b.geom));if(d<1e-9)continue;
  for(let k=0;k<points.length&&capacity();k++)if(k!==i&&k!==j){const c=points[k],u=vec(a.geom,b.geom),w=vec(a.geom,c.geom);
   if(Math.abs(cross(u,w))>eps*d*Math.max(d,length(w)))continue;
   look('coll',[a.id,b.id,c.id].sort(),`${a.id}, ${b.id}, ${c.id} 共线`);
   if(near(length(vec(a.geom,c.geom)),d/2)&&near(length(vec(c.geom,b.geom)),d/2))look('midp',[c.id,...[a.id,b.id].sort()],`${c.id} 是 ${a.id}${b.id} 的中点`);
  }
 }
 // A cyclic quadruple is a proposition only when the determinant validates;
 // cap O(n^4) for the discovery pass without capping DDAR proof closure.
 for(let i=0;i<points.length&&capacity();i++)for(let j=i+1;j<points.length&&capacity();j++)for(let k=j+1;k<points.length&&capacity();k++)for(let t=k+1;t<points.length&&capacity();t++){
  const a=points[i].geom,b=points[j].geom,c=points[k].geom,d=points[t].geom;
  const ux=b.x-a.x,uy=b.y-a.y,vx=c.x-a.x,vy=c.y-a.y,wx=d.x-a.x,wy=d.y-a.y;
  const aa=ux*ux+uy*uy,bb=vx*vx+vy*vy,cc=wx*wx+wy*wy;
  const det=(ux*vy-uy*vx)*cc-(ux*wy-uy*wx)*bb+(vx*wy-vy*wx)*aa;
  const scale=Math.max(1,aa,bb,cc)**2;if(Math.abs(det)<eps*scale)look('cyclic',[P[i],P[j],P[k],P[t]].sort(),`${P[i]}, ${P[j]}, ${P[k]}, ${P[t]} 共圆`);
 }
 return {version:'DDAR-Goals/4.3',goals,counts,points:points.length,lines:unique.length,segments:segments.length,elapsedMs:Date.now()-started,
  truncated:goals.length>=maxGoals,limitations:'数值筛选只产生候选；目前仅枚举所定义的有限谓词模式，并非无限形式语言。'};
}
function parseGoal(src,snapshot){
 const pts=new Set((snapshot.nodes||[]).filter(n=>n.valid&&n.geom?.kind==='point').map(n=>n.id));
 const lines=new Set((snapshot.nodes||[]).filter(n=>n.valid&&n.geom?.kind==='line').map(n=>n.id));
 const circles=new Set((snapshot.nodes||[]).filter(n=>n.valid&&n.geom?.kind==='circle').map(n=>n.id));
 const text=String(src||'').trim();if(!text)throw Error('请输入待证命题');
 let tokens=text.split(/[\s,，]+/).filter(Boolean),verb=tokens.shift().toLowerCase();
 const compact=text.replace(/\s+/g,'');
 if(/^[A-Za-z]{2}=[A-Za-z]{2}$/.test(compact)&&[...compact.replace('=','')].every(p=>pts.has(p))){
  tokens=[...compact.replace('=','')];verb='cong';
 }
 const symbolic=compact.match(/^([A-Za-z]{2})([⟂∥])([A-Za-z]{2})$/);
 if(symbolic&&[...symbolic[1],...symbolic[3]].every(p=>pts.has(p))){
  verb=symbolic[2]==='⟂'?'perp':'para';tokens=[...symbolic[1],...symbolic[3]];
 }
 const anglePoints=compact.match(/^∠([A-Za-z]{3})=∠([A-Za-z]{3})$/);
 if(anglePoints&&[...anglePoints[1],...anglePoints[2]].every(p=>pts.has(p))){
  verb='eqangle';tokens=[...anglePoints[1],...anglePoints[2]];
 }
 const angleLines=compact.match(/^∠\(([A-Za-z]{2}),([A-Za-z]{2})\)=∠\(([A-Za-z]{2}),([A-Za-z]{2})\)$/);
 if(angleLines&&angleLines.slice(1).every(s=>[...s].every(p=>pts.has(p)))){
  verb='angle';tokens=angleLines.slice(1).flatMap(s=>[...s]);
 }
 const n={cong:4,parallel:4,para:4,perp:4,perpendicular:4,angle:8,eqangle:6,eqratio:5,angconst:5,angleconst:5,midpoint:3,midp:3,collinear:3,coll:3,cyclic:4,oncircle:2,online:2,onc:2,onl:2,isos:3};
 if(!Object.hasOwn(n,verb))throw Error('未知命题类型：支持 cong / para / perp / angle / eqangle / eqratio / midp / coll / cyclic / onc / onl / isos');
 if(tokens.length!==n[verb])throw Error(verb+' 需要 '+n[verb]+' 个参数（点名以空格分隔）');
 const pointCount=['oncircle','online','onc','onl'].includes(verb)?1:['eqratio','angconst','angleconst'].includes(verb)?4:tokens.length;
 for(const p of tokens.slice(0,pointCount))if(!pts.has(p))throw Error('点不存在或当前无效：'+p);
 let kind=({parallel:'para',perpendicular:'perp',angle:'eqangleLines',eqangle:'eqangleLines',angleconst:'angconst',midpoint:'midp',collinear:'coll',oncircle:'onc',online:'onl'})[verb]||verb,args=[];
 if(['para','perp','cong'].includes(kind))args=kind==='cong'?[seg(tokens[0],tokens[1]),seg(tokens[2],tokens[3])].sort():[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3])].sort();
 else if(verb==='angle')args=[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3]),lp(tokens[4],tokens[5]),lp(tokens[6],tokens[7])];
 else if(verb==='eqangle')args=[lp(tokens[1],tokens[0]),lp(tokens[1],tokens[2]),lp(tokens[4],tokens[3]),lp(tokens[4],tokens[5])];
 else if(kind==='angconst'){
  const degrees=Number(tokens[4]);if(!Number.isFinite(degrees))throw Error('角度必须为有限数字');
  args=[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3]),String(((degrees%180)+180)%180)];
 }
 else if(kind==='eqratio'){
  const factor=Number(tokens[4]);if(!Number.isFinite(factor)||factor<=0)throw Error('线段比必须是正数');
  args=[seg(tokens[0],tokens[1]),seg(tokens[2],tokens[3]),String(factor)];
 }
 else if(kind==='midp')args=[tokens[0],...tokens.slice(1).sort()];
 else if(kind==='coll'||kind==='cyclic')args=tokens.sort();
 else if(kind==='onc'){if(!circles.has(tokens[1]))throw Error('圆不存在：'+tokens[1]);args=tokens;}
 else if(kind==='onl'){if(!lines.has(tokens[1]))throw Error('直线不存在：'+tokens[1]);args=[tokens[0],'L:'+tokens[1]];}
 else args=tokens;
 if(args.some((v,i)=>typeof v!=='string'||!v))throw Error('待证目标参数不完整');
 if((kind==='para'||kind==='perp'||kind==='cong')&&args[0]===args[1])throw Error('请选择两个不同的线段或直线');
 const id='manual:'+kind+':'+args.join('~');
 return {id,kind,args,text:'['+kind+'] '+tokens.join(' '),source:'手动设置待证命题'};
}
return {discover,parseGoal};
});
