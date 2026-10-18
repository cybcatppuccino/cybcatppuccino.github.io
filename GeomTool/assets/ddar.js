/* GeomTool v7.7 — AG2 only. No candidate discovery or relationship-scanner input. */
(function(root,factory){const api=factory(root);root.GeomDDAR=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(root){
'use strict';
const compiler=()=>root.GeomDDARCompiler||(typeof require==='function'?require('./ddar-compiler.js'):null);
const pair=(a,b)=>[a,b].sort().join('|');
function splitSeg(s){if(typeof s!=='string')return null;const v=s.split('|');return v.length===2?v:null}
function transformGoal(goal,compiled){
 if(!goal)return null;
 const a=goal.args||[],k=goal.kind;
 const line=s=>{if(typeof s!=='string')return null;if(s.startsWith('P:'))return splitSeg(s.slice(2));if(s.startsWith('L:'))return compiled.lines[s.slice(2)]||null;return null};
 const segment=s=>splitSeg(s);
 const group=[],put=(name,pp,constants=[])=>{if(pp.every(x=>typeof x==='string'&&x))group.push({name,points:pp,constants})};
 if(k==='cong'&&a.length===2){const u=segment(a[0]),v=segment(a[1]);if(u&&v)put('cong',[...u,...v]);}
 else if(k==='eqratio'&&a.length===3){const u=segment(a[0]),v=segment(a[1]),f=Number(a[2]);if(u&&v&&f>0&&Number.isFinite(f))put('rconst',[...u,...v],[f]);}
 else if(k==='para'||k==='perp'){const u=line(a[0]),v=line(a[1]);if(u&&v)put(k,[...u,...v]);}
 else if(k==='eqangleLines'&&a.length===4){const u=a.map(line);if(u.every(Boolean))put('eqangle',u.flat());}
 else if(k==='angconst'&&a.length===3){const u=line(a[0]),v=line(a[1]),deg=Number(a[2]);if(u&&v&&Number.isFinite(deg))put('aconst',[...u,...v],[-deg]);}
 else if(k==='midp'&&a.length===3){const [m,b,c]=a;put('coll',[m,b,c]);put('cong',[m,b,m,c]);}
 else if(k==='coll'&&a.length>=3){for(let i=2;i<a.length;i++)put('coll',[a[0],a[1],a[i]]);}
 else if(k==='cyclic'&&a.length>=4)put('cyclic',a);
 else if(k==='onl'&&a.length===2){const u=line(a[1]);if(u)put('coll',[a[0],...u]);}
 else if(k==='onc'&&a.length===2){const c=compiled.circles[a[1]];if(c){const others=c.points.filter(x=>x!==a[0]);if(c.center&&others.length){put('cong',[c.center,a[0],c.center,others[0]]);}else if(others.length>=3)put('cyclic',[a[0],...others.slice(0,3)]);}}
 else if(k==='isos'&&a.length===3)put('cong',[a[0],a[1],a[0],a[2]]);
 // Never classify reflexive/degenerate angle or length relations as proven.
 const coordinates=new Map(compiled.points.map(p=>[p.name,p.value]));
 for(const pred of group){for(const x of pred.points)if(!coordinates.has(x))return {id:goal.id||k+':'+a.join('~'),goal,predicates:[]};
  const name=pred.name,ps=pred.points;
  if(['cong','rconst','para','perp','eqangle','aconst'].includes(name)){
   for(let i=0;i<ps.length;i+=2){const x=coordinates.get(ps[i]),y=coordinates.get(ps[i+1]);if(!x||!y||Math.hypot(x[0]-y[0],x[1]-y[1])<1e-10)return {id:goal.id||k+':'+a.join('~'),goal,predicates:[]};}
  }
 }
 return {id:goal.id||k+':'+a.join('~'),goal,predicates:group};
}

// Geometrical plausibility check applies ONLY to manually entered goals.
// Passing this numerical test is NOT a formal proof; only AG2 decides that.
function validateGoal(goal,compiled){
 const q=transformGoal(goal,compiled);
 if(!q.predicates.length)throw Error('命题无法翻译为 AG2 支持的非退化谓词');
 const pos=new Map(compiled.points.map(p=>[p.name,p.value]));
 const xy=k=>{const v=pos.get(k);if(!v)throw Error('找不到点 '+k);return v};
 const dist=(a,b)=>Math.hypot(xy(a)[0]-xy(b)[0],xy(a)[1]-xy(b)[1]);
 const theta=(a,b)=>Math.atan2(xy(b)[1]-xy(a)[1],xy(b)[0]-xy(a)[0]);
 const mod=x=>Math.abs(Math.atan2(Math.sin(x),Math.cos(x)))%Math.PI;
 const errPi=x=>{const r=((x%Math.PI)+Math.PI)%Math.PI;return Math.min(r,Math.PI-r)};
 const near=(a,b)=>Math.abs(a-b)<=2e-6*Math.max(1,Math.abs(a),Math.abs(b));
 const segmentOK=(a,b)=>dist(a,b)>1e-9;
 for(const pr of q.predicates){const z=pr.points,n=pr.name;
  if(['para','perp','eqangle','cong','rconst','aconst'].includes(n))
   for(let i=0;i<z.length;i+=2)if(!segmentOK(z[i],z[i+1]))throw Error('命题包含零长度线段');
  let valid=false;
  if(n==='coll'){let a=xy(z[0]),b=xy(z[1]),c=xy(z[2]);valid=Math.abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))<2e-6*Math.max(1,dist(z[0],z[1])*dist(z[0],z[2]));}
  else if(n==='para'||n==='perp'){const delta=theta(z[0],z[1])-theta(z[2],z[3]);valid=n==='para'?errPi(delta)<2e-6:errPi(delta-Math.PI/2)<2e-6;}
  else if(n==='eqangle'){const left=theta(z[0],z[1])-theta(z[2],z[3]),right=theta(z[4],z[5])-theta(z[6],z[7]);valid=errPi(left-right)<2e-6;}
  else if(n==='aconst'){valid=errPi(theta(z[0],z[1])-theta(z[2],z[3])-Number(pr.constants[0])*Math.PI/180)<2e-6;}
  else if(n==='cong'){valid=near(dist(z[0],z[1]),dist(z[2],z[3]));}
  else if(n==='rconst'){valid=near(dist(z[0],z[1])/dist(z[2],z[3]),Number(pr.constants[0]));}
  else if(n==='cyclic'){const [a,b,c]=z.slice(0,3).map(xy),det=2*((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]));
   if(Math.abs(det)>1e-9){const v=([x,y])=>x*x+y*y,aa=v(a),bb=v(b),cc=v(c),cx=((bb-aa)*(c[1]-a[1])-(cc-aa)*(b[1]-a[1]))/det,cy=((cc-aa)*(b[0]-a[0])-(bb-aa)*(c[0]-a[0]))/det;const r=Math.hypot(a[0]-cx,a[1]-cy);valid=z.every(name=>near(Math.hypot(xy(name)[0]-cx,xy(name)[1]-cy),r));}
  }
  else throw Error('该类型暂不能进行数值几何校验: '+n);
  if(!valid)throw Error('命题当前不符合画板几何关系：'+n+' '+z.join(' '));
 }
 return q;
}
let worker=null,seq=0,active=null;const pending=new Map();
function workerInstance(){
 if(typeof Worker==='undefined')throw Error('需要 HTTP 环境中的 Worker / Pyodide');
 if(!worker){worker=new Worker('assets/ddar-py.worker.js');worker.onmessage=e=>{
  const m=e.data;
  if(m.command==='audit'||m.command==='proof'){const task=pending.get(m.requestId);if(task){pending.delete(m.requestId);task.resolve(m.result);}return;}
  if(!active||m.requestId!==active.id)return;
  if(m.command==='error'){active.reject(Error(m.error));active=null;return;}
  if(['progress','complete'].includes(m.command)){
   const state=unpack(active.prepared,m.result);state.progress=m.result;
   active.onProgress?.(state,m.command);
   if(m.command==='complete'){active.resolve(state);active=null;}
  }
 };worker.onerror=e=>{active?.reject(Error(e.message||'Pyodide Worker 错误'));active=null;worker?.terminate();worker=null;};}
 return worker;
}
// In goal mode, include the full connected construction trees of all goal
// objects: both ancestors and downstream constructions.  Do not infer extra
// premises from coordinates; the compiler still admits only defined facts.
const referenceIds=new Set(['p_0','p_1','p_i','l_Re','l_Im','c_unit']);
function inferenceSnapshot(snapshot,goal,userPretty,ancestorsOnly=false){
 const nodes=(snapshot.nodes||[]).filter(n=>n.valid&&n.geom),byId=new Map(nodes.map(n=>[n.id,n]));
 const needed=new Set(),children=new Map();
 let enqueue=null;
 for(const n of nodes)for(const parent of n.inputs||[]){
  if(!byId.has(parent))continue;
  if(!children.has(parent))children.set(parent,[]);
  children.get(parent).push(n.id);
 }
 const include=id=>{
  if(!byId.has(id)||needed.has(id))return;
  needed.add(id);
  if(enqueue)enqueue(id);
  const n=byId.get(id);
  for(const parent of n.inputs||[])include(parent);
  // The compiler implicitly uses these reference construction objects.
  if(n.op==='real')include('l_Re');
  if(id==='c_unit')include('p_1');
 };
 const includePoint=name=>{
  if(typeof name!=='string')return;
  if(byId.has(name)){include(name);return;}
  let match;
  if((match=/^@line:(.+):[01]$/.exec(name))||(match=/^@radius:(.+)$/.exec(name))||
     (match=/^@circumcenter:(.+)$/.exec(name))||(match=/^@tangent:(.+)$/.exec(name)))include(match[1]);
  else if((match=/^@mid:(.+)\|(.+)$/.exec(name))){include(match[1]);include(match[2]);}
 };
 if(goal){
  for(const arg of goal.args||[]){
   if(typeof arg!=='string')continue;
   const id=arg.startsWith('L:')?arg.slice(2):arg.startsWith('P:')?arg.slice(2):arg;
   if(byId.has(id))include(id);
   else if(id.includes('|'))for(const part of id.split('|'))includePoint(part);
  }
  if(!ancestorsOnly){
   // Traverse each dependency edge in both directions to a fixed point.
   // Every newly reached object also contributes all of its own prerequisites.
   const queue=[...needed],queued=new Set(queue);
   enqueue=id=>{if(!queued.has(id)){queued.add(id);queue.push(id);}};
   for(let i=0;i<queue.length;i++)
    for(const child of children.get(queue[i])||[])include(child);
  }
 }else{
  for(const n of nodes)if(!referenceIds.has(n.id))include(n.id);
  for(const pred of userPretty?.predicates||[])for(const name of pred.points||[])includePoint(name);
 }
 return {...snapshot,nodes:nodes.filter(n=>needed.has(n.id))};
}
const predicateKey=p=>JSON.stringify([p.name,p.points,p.constants||[]]);
function prepare(snapshot,opts={}){
 const compiled=compiler().compile(inferenceSnapshot(snapshot,opts.goal,opts.userPretty));
 if(opts.goal){
  // Presentation-only provenance. AG2 receives exactly the same unified
  // points and predicates arrays, so auxiliary facts are not second-class.
  const base=compiler().compile(inferenceSnapshot(snapshot,opts.goal,opts.userPretty,true));
  const basePoints=new Set(base.points.map(p=>p.name));
  const basePredicates=new Set(base.predicates.map(predicateKey));
  for(const p of compiled.points)p.auxiliary=!basePoints.has(p.name);
  for(const p of compiled.predicates)p.auxiliary=!basePredicates.has(predicateKey(p));
 }
 if(opts.userPretty){const extra=opts.userPretty,allowed=new Set(compiled.points.map(p=>p.name)),seen=new Set(compiled.predicates.map(p=>JSON.stringify([p.name,p.points,p.constants||[]])));
  for(const pred of extra.predicates||[]){if(!pred.points?.every(name=>allowed.has(name)))continue;const key=JSON.stringify([pred.name,pred.points,pred.constants||[]]);if(!seen.has(key)){compiled.predicates.push({...pred,auxiliary:!!(opts.goal&&pred.points.some(x=>compiled.points.find(p=>p.name===x)?.auxiliary))});seen.add(key);}}
  compiled.limitations.push(...(extra.limitations||[]));
 }
 const queries=opts.goal?[transformGoal(opts.goal,compiled)]:[];
 if(queries.length&&!queries[0].predicates.length)throw Error('该命题无法转换为 AG2 谓词');
 return {compiled,backend:'ag2',queries,goal:opts.goal||null,started:Date.now()};
}
function unpack(prepared,response){
 const found=new Map((response.queries||[]).map(x=>[x.id,x]));
 const goal=prepared.goal,query=prepared.queries[0],r=query&&found.get(query.id);
 return {version:'AlphaGeometry2-DDAR/v7.7',goal:goal?{kind:goal.kind,args:goal.args,text:query.predicates.map(p=>p.name+' '+p.points.join(' ')).join(' ; '),status:r?.status||'unknown',reason:r?.reason||'未证明'}:null,
  facts:[],matches:[],counts:{},progress:response,rounds:response.depth,elapsedMs:response.elapsedMs,
  points:response.pointCount,premiseCount:response.premiseCount,auditCount:response.auditCount||0,limitations:prepared.compiled.limitations};
}
function cancel(){if(worker){worker.terminate();worker=null;}active?.reject(Error('图形改变或会话取消'));active=null;for(const x of pending.values())x.reject(Error('会话取消'));pending.clear();}
function startSession(snapshot,opts={}){
 if(active)cancel();const prepared=prepare(snapshot,opts),id=++seq;
 const admitted=new Set(prepared.compiled.points.map(p=>p.name));
 const auxiliaryPoints=(snapshot.nodes||[]).filter(n=>n.geom?.kind==='point'&&!['free','real'].includes(n.op)&&admitted.has(n.id)).map(n=>n.id);
 const completion=new Promise((resolve,reject)=>{active={id,prepared,resolve,reject,onProgress:opts.onProgress};workerInstance().postMessage({command:'start',requestId:id,payload:{engine:'ag2',compiled:prepared.compiled,queries:prepared.queries,mode:opts.mode||'closure',targetId:prepared.queries[0]?.id||null,auxiliaryPoints,extendedSinRatio:!!opts.extendedSinRatio}});});
 completion.catch(()=>{});
 const request=command=>new Promise((resolve,reject)=>{const requestId=++seq;pending.set(requestId,{resolve,reject});workerInstance().postMessage({command,requestId});});
 return {prepared,completion,audit(){return request('audit')},proof(){return request('proof')},stop:cancel};
}
async function infer(snapshot,opts={}){return startSession(snapshot,{...opts,mode:opts.goal?'goal':'closure'}).completion;}
return {infer,cancel,prepare,unpack,transformGoal,validateGoal,startSession,compile:s=>compiler().compile(s),engines:['ag2'],engine:'AlphaGeometry2-official'};
});
