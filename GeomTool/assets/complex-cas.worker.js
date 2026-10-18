/* GeomTool v8 — exact symbolic construction and budgeted isolated factoring. */
'use strict';
importScripts('rational-cas.js','complex-cas.js');
let activeFactor=null,activeLineGcd=null,lineGcdRequestId=0;
let nextFactorId=0;
let cachedGeometry=null,cachedGraphKey=null;
let epoch=0,interruptFactor=null;
const formattedExpressions=new Map();
async function factorWithBudget(algebra,alias,budgets){
 const start=performance.now();
 const limit=Math.max(1,Number(budgets?.factorMs)||10000);
 const auxiliaryMs=Number.isFinite(Number(budgets?.nerdamerMs))?Math.max(0,Number(budgets.nerdamerMs)):0;
 const memoKey=limit+'|'+auxiliaryMs+'|'+self.GeomRationalCAS.string(algebra);
 if(formattedExpressions.has(memoKey))return formattedExpressions.get(memoKey);
 // Payload preparation also consumes the expression's wall-clock budget.
 // Do not start a child job if that work already used the available time.
 if(performance.now()-start>=limit)return self.GeomRationalCAS.string(algebra,alias);
 const payload=self.GeomRationalCAS.factorPayload(algebra,alias);
 if(performance.now()-start>=limit)return self.GeomRationalCAS.string(algebra,alias);
 return new Promise((resolve,reject)=>{
  const worker=activeFactor||new Worker('factor-cas.worker.js');activeFactor=worker;
  const requestId=++nextFactorId;
  let settled=false;
  let best=self.GeomRationalCAS.string(algebra,alias);
  const finish=(result,error,terminate=false,cache=true)=>{if(settled)return;settled=true;
   if(interruptFactor===cancel)interruptFactor=null;clearTimeout(watchdog);
   if(terminate){worker.terminate();if(activeFactor===worker)activeFactor=null;}
   if(error)reject(new Error(error));else{if(cache)formattedExpressions.set(memoKey,result);resolve(result);}
  };
  const watchdog=setTimeout(()=>finish(best,null,true),Math.max(0,limit-(performance.now()-start)));
  const cancel=()=>finish(best,null,true,false);interruptFactor=cancel;
  worker.onmessage=e=>{const data=e.data||{};
   if(data.id!==requestId)return;
   if(typeof data.partial==='string')best=data.partial;
   if(typeof data.result==='string')finish(data.result);
   else if(data.error)finish(best,null,true);
  };
  worker.onerror=e=>{e.preventDefault?.();finish(best,null,true);};
  worker.onmessageerror=()=>finish(best,null,true);
  try{worker.postMessage({id:requestId,payload,auxiliaryMs,totalMs:Math.max(1,limit-(performance.now()-start))});}
  catch(_){finish(best,null,true);}
 });
}
async function simplifyLineCoefficients(coefficients,alias){
 const payload=self.GeomRationalCAS.lineGcdPayload(coefficients,alias);
 return new Promise(resolve=>{
  const worker=activeLineGcd||new Worker('line-gcd.worker.js');activeLineGcd=worker;
  const id=++lineGcdRequestId;
  let finished=false,best=null;
  const done=(terminate=false)=>{if(finished)return;finished=true;clearTimeout(timer);
   if(terminate){worker.terminate();if(activeLineGcd===worker)activeLineGcd=null;}
   resolve(best);
  };
  const timer=setTimeout(()=>done(true),10000);
  worker.onmessage=e=>{const data=e.data||{};if(data.id!==id)return;
   if(data.coefficients)best=data.coefficients;
   if(data.unchanged||data.error||data.complete)done(!!data.error);
   else if(data.coefficients&&Object.keys(data.coefficients).length===3){
    // Refinements arrive incrementally; the 10s watchdog returns the best one.
   }
  };
  worker.onerror=e=>{e.preventDefault?.();done(true);};
  worker.onmessageerror=()=>done(true);
  try{worker.postMessage({id,payload});}catch(_){done(true);}
 });
}
self.onmessage=async event=>{
 const {id,graph,graphKey,changes,propositions=[],factorSettings={}}=event.data||{};
 const started=performance.now();
 const ownEpoch=++epoch;
 if(interruptFactor)interruptFactor();
 try{
  if(!cachedGeometry||(!changes&&cachedGraphKey!==graphKey)){
   if(!graph)throw Error('缺少可用的复坐标缓存');
   cachedGeometry=null;cachedGraphKey=null;formattedExpressions.clear();
   if(activeFactor){activeFactor.terminate();activeFactor=null;}
   cachedGeometry=self.GeomComplexCAS.prepare(graph,{
    onProgress:(progress,total,last)=>self.postMessage({id,progress,total,last})});
   cachedGraphKey=graphKey;
  }
  else if(changes){
   cachedGeometry.syncGraph(changes);
   const renames=changes.renames||{};
   if(Object.keys(renames).length){
    const translate=text=>text.replace(/[A-Za-z][A-Za-z0-9_]*/g,name=>renames[name]||name);
    for(const [key,formatted] of formattedExpressions)formattedExpressions.set(key,translate(formatted));
   }
  }
  cachedGraphKey=graphKey;
  const result=await cachedGeometry.evaluate(propositions,{
   formatExpression:(a,alias)=>factorWithBudget(a,alias,factorSettings),
   simplifyLine:simplifyLineCoefficients,
   isCurrent:()=>ownEpoch===epoch,
   onReady:ready=>self.postMessage({id,ready}),
   onStatusProgress:statusProgress=>self.postMessage({id,statusProgress}),
   onGoalProgress:goalProgress=>self.postMessage({id,goalProgress})});
  if(result&&ownEpoch===epoch){result.elapsedMs=performance.now()-started;self.postMessage({id,result});}
 }catch(error){if(ownEpoch===epoch)self.postMessage({id,error:String(error?.message||error)});}
 finally{/* Retain factor caches alongside the unchanged exact geometry session. */}
};
