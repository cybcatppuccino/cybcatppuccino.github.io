/* v4.5: numerical GOAL discovery only. Inference uses ddar-py.worker.js. */
'use strict';
importScripts('./ddar-goals.js');
self.onmessage=e=>{
 const {token,snapshot,discoverGoals}=e.data;
 try{if(!discoverGoals)throw Error('Inference runs exclusively in the Pyodide DDAR worker');
  self.postMessage({token,goals:self.GeomDDARGoals.discover(snapshot)});
 }catch(err){self.postMessage({token,error:String(err?.stack||err)})}
};
