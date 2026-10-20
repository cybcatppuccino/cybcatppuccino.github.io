/* GeomTool v8.2 — one exact, factored Nerdamer 2.0 construction pass.
 * Each finished node is immediately normalized and posted to the UI.
 * No old CAS/factor workers, deferred rewriting, or end-of-graph rebuild.
 */
'use strict';
importScripts('nerdamer-sparse-v2.js','nerdamer-rational.js','complex-cas.js');
let cachedGeometry=null,cachedGraphKey=null,epoch=0;
self.onmessage=async event=>{
 const {id,graph,graphKey,changes,propositions=[]}=event.data||{};
 const started=performance.now(),myEpoch=++epoch;
 self.GeomNerdamerRational.setFactorBudget(event.data?.factorSettings?.factorMs);
 try{
  if(!cachedGeometry||(!changes&&cachedGraphKey!==graphKey)){
   if(!graph)throw Error('缺少复坐标构图');
   cachedGeometry=null;
   cachedGeometry=self.GeomComplexCAS.prepare(graph,{
    onProgress:(progress,total,last)=>self.postMessage({id,progress,total,last}),
    onNode:(nodeProgress,progress,total)=>self.postMessage({id,nodeProgress,progress,total})});
  }else if(changes){cachedGeometry.syncGraph(changes);}
  cachedGraphKey=graphKey;
  const result=await cachedGeometry.evaluate(propositions,{
   formatExpression:(expr,alias)=>self.GeomNerdamerRational.string(expr,alias),
   isCurrent:()=>myEpoch===epoch,
   onReady:ready=>self.postMessage({id,ready}),
   onStatusProgress:statusProgress=>self.postMessage({id,statusProgress}),
   onGoalProgress:goalProgress=>self.postMessage({id,goalProgress})});
  if(result&&myEpoch===epoch){result.elapsedMs=performance.now()-started;self.postMessage({id,result});}
 }catch(error){if(myEpoch===epoch)self.postMessage({id,error:String(error?.message||error)});}
};
