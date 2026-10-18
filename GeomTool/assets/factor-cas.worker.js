/* One polynomial-expression factorization at a time. Outer worker enforces
 * wall-clock deadline; a nested worker enforces EACH Nerdamer check deadline. */
'use strict';
importScripts('rational-cas.js','nerdamer-sparse-v2.js');
let checker=null,nextId=0,pending=null,sessionNames=null;
let sparseWorker=null,sparseId=0,sparsePending=null;
function stopSparse(){if(sparseWorker){sparseWorker.terminate();sparseWorker=null;}}
function sparseCall(task,names,poly,other,deadline){
 const time=Math.floor(deadline-performance.now());
 if(time<=0)return Promise.resolve(task==='gcd'?new Map([['',{re:1n,im:0n,den:1n}]]):[]);
 return new Promise(resolve=>{
  if(!sparseWorker){sparseWorker=new Worker('nerdamer-sparse.worker.js');
   sparseWorker.onmessage=e=>{const d=e.data||{};
    if(!sparsePending||d.id!==sparsePending.id)return;
    const pending=sparsePending;sparsePending=null;clearTimeout(pending.timer);
    pending.resolve(d.answer??null);};
   sparseWorker.onerror=()=>{if(sparsePending){const pending=sparsePending;sparsePending=null;
    clearTimeout(pending.timer);pending.resolve(null);}stopSparse();};
  }
  const id=++sparseId;
  const timer=setTimeout(()=>{if(!sparsePending||sparsePending.id!==id)return;
   sparsePending=null;stopSparse();resolve(null);},time);
  sparsePending={id,timer,resolve};
  sparseWorker.postMessage({id,task,names,poly,other});
 });
}

function closeChecker(){if(checker){checker.terminate();checker=null;}}
function queryNerdamer(expr,ms){
 return new Promise(resolve=>{
  if(!checker){checker=new Worker('nerdamer-check.worker.js');
   checker.onmessage=e=>{if(!pending||e.data?.id!==pending.id)return;
    const request=pending;pending=null;clearTimeout(request.timer);request.resolve(e.data.output||'');};
   checker.onerror=()=>{if(pending){const request=pending;pending=null;clearTimeout(request.timer);request.resolve('');}closeChecker();};
  }
  const id=++nextId;
  const timer=setTimeout(()=>{if(pending?.id!==id)return;
   pending=null;closeChecker();resolve('');},ms);
  pending={id,timer,resolve};
  checker.postMessage({id,expr,ms});
 });
}
self.onmessage=async e=>{
 try{
  const {id,payload,auxiliaryMs,totalMs=10000}=e.data;
  const names=payload.names.join('\0');
  if(names!==sessionNames){self.GeomRationalCAS.reset();sessionNames=names;stopSparse();}
  const begin=performance.now(),total=Math.max(1,totalMs);
  // Reserve the last 70% of the expression's deadline for factorization.
  const gcdDeadline=begin+total*0.30,allDeadline=begin+total;
  const gcdRequest=async(a,b)=>{
   const answer=await sparseCall('gcd',payload.names,a,b,gcdDeadline);
   return answer instanceof Map?answer:new Map([['',{re:1n,im:0n,den:1n}]]);
  };
  const factorRequest=async p=>{
   const answer=await sparseCall('factor',payload.names,p,null,allDeadline);
   return Array.isArray(answer)?answer:[];
  };
  const result=await self.GeomRationalCAS.factorFromPayload(payload,{
   auxiliaryMs,gcdRequest,factorRequest,
   nerdamerRequest:auxiliaryMs>0?expr=>queryNerdamer(expr,Math.min(auxiliaryMs,2000)):null,
   onPartial:partial=>self.postMessage({id,partial})});
  self.postMessage({id,result});
 }catch(error){self.postMessage({id:e.data?.id,error:String(error?.message||error)});}
};
