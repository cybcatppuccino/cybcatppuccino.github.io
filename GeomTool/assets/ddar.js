/* GeomTool 4.5 — compatibility API on top of ORIGINAL AlphaGeometry2 DDAR.
 * Returns Promise<DDARResult>. Real inference lives in Python in Pyodide;
 * NO JS rule engine, no numerical proof fallback. Goal discovery and
 * auxiliary-construction modules are deliberately left unchanged.
 */
(function(root,factory){const api=factory(root);root.GeomDDAR=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(root){
'use strict';
const compiler=()=>root.GeomDDARCompiler||(typeof require==='function'?require('./ddar-compiler.js'):null);
const goals=()=>root.GeomDDARGoals||(typeof require==='function'?require('./ddar-goals.js'):null);
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
let worker=null,seq=0,active=null;const pending=new Map();
function workerInstance(){
 if(typeof Worker==='undefined')throw Error('需要 Worker / HTTP 环境来启动 Pyodide');
 if(!worker){worker=new Worker('assets/ddar-py.worker.js');worker.onmessage=e=>{
  const m=e.data;
  if(m.command==='proof'||m.command==='audit'){const p=pending.get(m.requestId);if(p){pending.delete(m.requestId);p.resolve(m.result);}return;}
  if(!active||m.requestId!==active.id)return;
  if(m.command==='error'){active.reject?.(Error(m.error));active.onError?.(Error(m.error));active=null;return;}
  if(m.command==='progress'||m.command==='paused'||m.command==='complete'){
   active.onProgress?.(m.result,m.command);
   if(m.command==='complete'){active.resolve?.(m.result);active=null;}
  }
 };worker.onerror=e=>{const error=Error(e.message||'Pyodide Worker 错误');active?.reject?.(error);active?.onError?.(error);active=null;worker?.terminate();worker=null;};}
 return worker;
}
function startSession(snapshot,opts={}){
 if(active)cancel();
 const prepared=prepare(snapshot,{...opts,enumerate:opts.enumerate??false});
 const id=++seq;
 const completion=new Promise((resolve,reject)=>{
  active={id,prepared,resolve,reject,onError:opts.onError,onProgress:(raw,event)=>{
   const result=unpack(prepared,raw);result.progress={...raw};opts.onProgress?.(result,event);
  }};
  workerInstance().postMessage({command:'start',requestId:id,payload:{engine:prepared.backend,compiled:prepared.compiled,ag1:prepared.ag1,queries:prepared.queries,
    mode:opts.mode||'closure',targetId:opts.goal?prepared.queries.find(q=>q.goal?.id===opts.goal.id)?.id:null,maxLevel:opts.maxLevel,levelTimeout:opts.levelTimeout}});
 });
 completion.catch(()=>{});
 return {completion,pause(){worker?.postMessage({command:'pause'})},resume(){worker?.postMessage({command:'resume'})},
  proof(queryId){return new Promise((resolve,reject)=>{const requestId=++seq;pending.set(requestId,{resolve,reject});workerInstance().postMessage({command:'proof',requestId,queryId});})},
  audit(){return new Promise((resolve,reject)=>{const requestId=++seq;pending.set(requestId,{resolve,reject});workerInstance().postMessage({command:'audit',requestId});})},
  stop:cancel,prepared};
}
// AG1's upstream numerical sketch can retry indefinitely if two distinct
// symbolic points happen to occupy identical screen coordinates.  A generic
// realization of the SAME construction program avoids a false identity and
// preserves every definition-implied incidence/ratio.  No perturbation is
// admitted as a theorem premise, and the user's canvas is never modified.
function ag1GenericRealization(snapshot, compiled, ag1Compiler, allowGenericRealization=false){
 const collides=ag1=>{const coords=Object.entries(ag1.pointCoordinates);
  for(let i=0;i<coords.length;i++)for(let j=i+1;j<coords.length;j++)
   if(Math.hypot(coords[i][1][0]-coords[j][1][0],coords[i][1][1]-coords[j][1][1])<1e-8)return [coords[i][0],coords[j][0]];
  return null;
 };
 let ag1=ag1Compiler.compile(snapshot,compiled),problem=collides(ag1);
 if(!problem)return {compiled,ag1};
 // v5.4: reproducible AG1 alternate sketches change numerical coordinates
 // (even though they never touch the canvas). Such a substitution MUST NOT
 // happen without explicit opt-in, especially when the drawing has Pretty goals.
 if(!allowGenericRealization)throw Error('AG1 检测到两符号点数值重合 ('+problem.join(', ')+')。已停止：默认禁止静默生成扰动替代图。请修改作图，或明确设置 allowGenericRealization:true 以允许仅在副本中尝试。');
 const kernelModule=root.GeomKernel||(typeof require==='function'?require('./kernel.js'):null);
 if(!kernelModule?.Kernel)throw Error('AG1 无法生成通用数值示意：Kernel 缺失');
 const freedom=snapshot.nodes.filter(n=>n.valid&&!n.style?.system&&['free','real'].includes(n.op));
 if(!freedom.length)throw Error('AG1 中有未证同一点关系，且没有可扰动自由点：'+problem.join(', '));
 for(let attempt=1;attempt<=8;attempt++){
  const size=[.003,.009,.027,.08,.16,.35,.7,1.2][attempt-1];
  const data={schema:'geomtool.ir/v1',nodes:snapshot.nodes.map(n=>({id:n.id,label:n.label,op:n.op,inputs:n.inputs,params:{...n.params},style:{...n.style}}))};
  for(const n of data.nodes)if(freedom.some(f=>f.id===n.id)){
   const seed=[...n.id].reduce((a,c)=>a*31+c.charCodeAt(0),17),
    angle=(seed%6283)/1000+attempt*1.7;
   if(n.op==='free'){n.params.x+=size*Math.cos(angle);n.params.y+=size*Math.sin(angle);}
   else if(n.op==='real')n.params.x+=size*Math.cos(angle);
  }
  try{
   const sketch=new kernelModule.Kernel(data),generic=sketch.snapshot();
   if(generic.nodes.some((n,i)=>snapshot.nodes[i]?.valid&&!n.valid))continue;
   const translated=compiler().compile(generic), candidate=ag1Compiler.compile(generic,translated);
   if(!collides(candidate)){
    translated.limitations.push('AG1 使用保持构造定义的通用数值示意（未改变原图或增加逻辑前提）');
    return {compiled:translated,ag1:candidate};
   }
  }catch(e){/* invalid numerical branch; try a different generic sketch */}
 }
 throw Error('AG1 无法找到非退化、满足构造关系的通用数值示意：'+problem.join(', '));
}
function prepare(snapshot,opts={}){
 const cmp=compiler();if(!cmp)throw Error('Missing GeomDDARCompiler');
 let compiled=cmp.compile(snapshot);
 // USER-PRETTY hypotheses are deliberate user assertions, not discovered theorems.
 // Only the AG2 submission includes them; invalid/unsupported declarations stay in the audit.
 if(opts.engine!=='ag1'&&opts.userPretty){
  const extra=opts.userPretty,seen=new Set(compiled.predicates.map(p=>JSON.stringify([p.name,p.points,p.constants||[]])));
  for(const pred of extra.predicates||[]){const key=JSON.stringify([pred.name,pred.points,pred.constants||[]]);if(!seen.has(key)){compiled.predicates.push({...pred});seen.add(key);}}
  compiled.limitations.push(...(extra.limitations||[]));
  compiled.assumptionReport.push(...(extra.predicates||[]).map(p=>p.source+': '+p.name+' '+p.points.join(' ')));
  if(compiled.stats)compiled.stats.predicates=compiled.predicates.length;
 }
 const queries=[],ids=new Set();
 const add=(g)=>{if(!g)return;const q=transformGoal(g,compiled);if(!ids.has(q.id)){queries.push(q);ids.add(q.id)}};
 add(opts.goal);
 // Reuse the existing numerical *hypothesis discovery*, but NEVER treat it
 // as inference. Every displayed positive result is separately checked in AG2.
 if(opts.goals)for(const g of opts.goals)add(g);
 if(!opts.goals&&goals()&&opts.enumerate!==false){for(const g of goals().discover(snapshot).goals)add(g);}
 const entries=opts.entries||[];
 for(const e of entries){
  const marks=e.marks||[],ll=marks.filter(m=>m.type==='line').map(m=>m.ref?.type==='pair'?'P:'+pair(m.ref.a,m.ref.b):'L:'+m.ref?.id),pp=marks.filter(m=>m.type==='point').map(m=>m.id),ss=marks.filter(m=>m.type==='segment');
  if(e.kind==='parallel'&&ll.length>=2){
   const q=transformGoal({id:'match:'+e.id,kind:'para',args:ll.slice(0,2)},compiled);
   if(!ids.has(q.id)){
    // An aggregate is proved only when EVERY carrier is parallel.
    q.predicates=[];
    for(const l of ll.slice(1))q.predicates.push(...transformGoal({id:'part',kind:'para',args:[ll[0],l]},compiled).predicates);
    queries.push(q);ids.add(q.id);
   }
  }
  else if(e.kind==='perpendicular'&&ll.length===2)add({id:'match:'+e.id,kind:'perp',args:ll.slice(0,2)});
  else if(e.kind==='collinear'&&pp.length>=3)add({id:'match:'+e.id,kind:'coll',args:pp});
  else if(e.kind==='concyclic'&&pp.length>=4)add({id:'match:'+e.id,kind:'cyclic',args:pp});
  else if(e.kind==='midpoint'&&pp.length&&ss.length)add({id:'match:'+e.id,kind:'midp',args:[pp[0],ss[0].a,ss[0].b]});
  else if(e.kind==='pointOnLine'&&ll.length&&pp.length)add({id:'match:'+e.id,kind:'onl',args:[pp[0],ll[0]]});
  else if(e.kind==='distanceRatio'&&e.metric){
   const metric=e.metric,hi=metric.high||[],lo=metric.low||[];
   if(hi.length&&lo.length&&[...hi,...lo].every(t=>t.type==='segment'&&t.key?.startsWith('D:'))){
    const q={id:'match:'+e.id,goal:null,predicates:[]},s1=hi.map(t=>t.key.slice(2)),s2=lo.map(t=>t.key.slice(2));
    for(let j=1;j<s1.length;j++)q.predicates.push(...transformGoal({id:'part',kind:'cong',args:[s1[0],s1[j]]},compiled).predicates);
    for(let j=1;j<s2.length;j++)q.predicates.push(...transformGoal({id:'part',kind:'cong',args:[s2[0],s2[j]]},compiled).predicates);
    q.predicates.push(...transformGoal({id:'part',kind:'eqratio',args:[s1[0],s2[0],String(metric.factor)]},compiled).predicates);
    if(!ids.has(q.id)){queries.push(q);ids.add(q.id)}
   }
  }
  else if(e.kind==='pointOnCircle'&&pp.length){const c=marks.find(m=>m.type==='circle')?.ref;if(c?.type==='object')add({id:'match:'+e.id,kind:'onc',args:[pp[0],c.id]});}
 }
 const backend=opts.engine==='ag1'?'ag1':'ag2';
 let ag1=null;if(backend==='ag1'){const ag1Compiler=root.GeomAG1Compiler||(typeof require==='function'?require('./ddar-ag1.js'):null);if(!ag1Compiler)throw Error('AG1 compiler missing');const generic=ag1GenericRealization(snapshot,compiled,ag1Compiler,opts.allowGenericRealization===true);compiled=generic.compiled;ag1=generic.ag1;for(const q of queries){if(q.goal)q.predicates=transformGoal(q.goal,compiled).predicates;q.ag1Goal=ag1Compiler.goal(q.goal,ag1);}}
 return {compiled,ag1,backend,queries,entries,goal:opts.goal||null,started:Date.now()};
}
function unpack(prepared,response){
 const found=new Map(response.queries.map(x=>[x.id,x]));
 const textOf=q=>prepared.backend==='ag1'?(q.ag1Goal||''):(q.predicates||[]).map(p=>p.name+' '+p.points.join(' ')+(p.constants?.length?' '+p.constants.join(' '):'')).join(' ; ');
 const answer=g=>{if(!g)return null;const q=transformGoal(g,prepared.compiled),r=found.get(q.id)||{status:'unsupported',reason:'未执行查询'};return {kind:g.kind,args:g.args,text:textOf(q)||g.text||g.kind+' '+g.args.join(' '),status:r.status,reason:r.reason,evidence:r.premises||[]};};
 const facts=[];const seen=new Set();
 for(const q of prepared.queries){const s=found.get(q.id);if(s?.status!=='proved'||!q.goal||q.id.startsWith('match:'))continue;
  const g=q.goal,k=g.kind+':'+(g.args||[]).join('~');if(seen.has(k))continue;seen.add(k);
  facts.push({id:k,kind:g.kind,args:g.args,rule:s.rule||'DDAR',premises:s.premises||[],source:prepared.backend+'-closure',depth:s.depth||0,queryId:q.id,text:textOf(q)||k});
 }
 // Structural facts are enumerated by the official graph itself, not from
 // coordinate similarity. The candidate check list complements this stream.
 const signatures=new Set(facts.map(f=>f.text));
 for(const f of response.nativeFacts||[]){if(signatures.has(f.text))continue;signatures.add(f.text);facts.push({...f});}
 const counts={};for(const f of facts)counts[f.kind]=(counts[f.kind]||0)+1;
 const matches=prepared.entries.map(e=>{const r=found.get('match:'+e.id);return {id:e.id,status:r?.status||'unknown'}});
 return {version:(prepared.backend==='ag1'?'AlphaGeometry1-DDAR/Pyodide-v4.6':'AlphaGeometry2-DDAR/Pyodide-v4.6'),goal:answer(prepared.goal),facts,counts,total:facts.length,rounds:'saturation',elapsedMs:Math.round(response.elapsedMs),truncated:false,points:response.pointCount,lines:Object.keys(prepared.compiled.lines).length,circles:Object.keys(prepared.compiled.circles).length,virtuals:prepared.compiled.points.filter(x=>x.name.startsWith('@')).map(x=>({id:x.name})),matches,nuclei:[],limitations:prepared.compiled.limitations,premiseCount:response.premiseCount,proofTraceAvailable:prepared.backend==='ag1',auditCount:response.auditCount||0,auditTruncated:!!response.auditTruncated};
}
function cancel(){
 if(worker){worker.terminate();worker=null;}
 active?.reject?.(Error('DDAR 场景变化，搜索已取消'));active=null;
 for(const p of pending.values())p.reject(Error('DDAR 搜索已取消'));pending.clear();
}
async function infer(snapshot,opts={}){const task=startSession(snapshot,{...opts,mode:opts.goal?'goal':'closure'});const result=await task.completion;return unpack(task.prepared,result);}
return {infer,cancel,startSession,prepare,unpack,transformGoal,compile:s=>compiler().compile(s),engines:['ag2','ag1'],engine:'AlphaGeometry2-official'};
});
