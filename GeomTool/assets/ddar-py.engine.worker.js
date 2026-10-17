/* v4.5 persistent Pyodide worker. A resumable Python generator is advanced
 * between main-thread message opportunities. No proof is decided in JS.
 */
'use strict';
let pyPromise=null,current=null,lastSession=null,job=0,advanceRunning=false;
async function bootstrap(){
 const urls=['https://cdn.jsdelivr.net/pyodide/v0.26.4/full/','https://pyodide.org/pyodide/v0.26.4/full/'];
 if(typeof self.loadPyodide!=='function'){
  let err;for(const url of urls){try{importScripts(url+'pyodide.js');break}catch(e){err=e}}
  if(!self.loadPyodide)throw Error('Pyodide 未能加载: '+String(err));
 }
 let py,err;for(const url of urls){try{py=await loadPyodide({indexURL:url});break}catch(e){err=e}}
 if(!py)throw Error('Pyodide 无法初始化: '+err);
 await py.loadPackage('numpy');
 await install(py,'/home/py/ag2','../vendor/alphageometry2/', ['ddar.py','elimination.py','numericals.py','parse.py','geomtool_v6_session.py','geomtool_v64_proof.py','b_theorems.py','sine_ratios.py']);
 await py.runPythonAsync("import sys\nsys.path.insert(0,'/home/py/ag2')\nimport geomtool_v6_session as geomtool_ag2_session\nv6_ag2_start_json=geomtool_ag2_session.start_json\nv6_ag2_step_json=geomtool_ag2_session.step_json\nv6_ag2_audit_json=geomtool_ag2_session.audit_json\nv6_ag2_proof_json=geomtool_ag2_session.proof_json");
 return py;
}
async function install(py,path,base,files){py.FS.mkdirTree(path);for(const file of files){const r=await fetch(base+file);if(!r.ok)throw Error('缺少引擎源码 '+base+file);const dst=path+'/'+file;py.FS.mkdirTree(dst.slice(0,dst.lastIndexOf('/')));py.FS.writeFile(dst,await r.text());}}
function invoke(py,engine,fn,arg){
 const func=py.globals.get('v6_'+engine+'_'+fn);
 try{return JSON.parse(func(...(arg===undefined?[]:[arg])));}finally{func.destroy();}
}
function send(command,requestId,result,error){self.postMessage({command,requestId,result,error});}
async function initialize(requestId,payload){
 const mine=++job;current=null;lastSession=null;
 try{
  if(!pyPromise)pyPromise=bootstrap().catch(err=>{pyPromise=null;throw err});
  const py=await pyPromise;if(mine!==job)return;
  if(payload.engine!=='ag2')throw Error('只允许 AlphaGeometry2 内核');
  const result=invoke(py,payload.engine,'start_json',JSON.stringify(payload));
  current={py,engine:payload.engine,requestId,latest:result};
  send('progress',requestId,result);advance();
 }catch(err){send('error',requestId,null,String(err?.stack||err));}
}
async function advance(){
 if(advanceRunning)return;
 advanceRunning=true;
 try{
  while(current){
   const {py,engine,requestId}=current;
   let result;
   try{result=invoke(py,engine,'step_json');}catch(err){send('error',requestId,null,String(err?.stack||err));current=null;break;}
   if(!current||current.requestId!==requestId)break;
   current.latest=result;
   // Emit progress at every completed stage. The main thread throttles DOM
   // rendering to 1Hz. A long single native stage is not preemptible.
   send('progress',requestId,result);
   if(result.done){lastSession=current;send('complete',requestId,result);current=null;break;}
   await new Promise(resolve=>setTimeout(resolve,0));
  }
 }finally{advanceRunning=false;}
}
self.onmessage=async e=>{
 const d=e.data||{};
 if(d.command==='start'){initialize(d.requestId,d.payload);return;}
 if(d.command==='status'){if(current)send('progress',current.requestId,current.latest);return;}
 if(d.command==='audit'||d.command==='proof'){
  const c=current||lastSession;
  if(!c||c.engine!=='ag2'){send(d.command,d.requestId,{available:false,reason:'尚无 AG2 推理会话'});return;}
  try{const fn=c.py.globals.get(d.command==='proof'?'v6_ag2_proof_json':'v6_ag2_audit_json');let data;try{data=JSON.parse(fn());}finally{fn.destroy();}send(d.command,d.requestId,data);}
  catch(err){send(d.command,d.requestId,{available:false,reason:String(err)});}return;
 }
};
