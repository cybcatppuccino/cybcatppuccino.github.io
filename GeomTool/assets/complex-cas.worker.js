/* GeomTool v7.9 patched: standalone offline exact JS worker. No Pyodide,
 * SymPy, Algebrite or CDN loading on the complex-coordinate code path. */
'use strict';
importScripts('nerdamer-factor.min.js','rational-cas.js','complex-cas.js');
self.onmessage=event=>{
 const {id,graph}=event.data||{};
 const started=performance.now();
 try{
  const result=self.GeomComplexCAS.compute(graph,{onProgress:(progress,total,last)=>self.postMessage({id,progress,total,last})});
  result.elapsedMs=performance.now()-started;
  self.postMessage({id,result});
 }catch(error){self.postMessage({id,error:String(error?.message||error)});}
};
