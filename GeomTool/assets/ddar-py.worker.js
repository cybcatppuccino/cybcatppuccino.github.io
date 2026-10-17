/* v4.5 Worker supervisor.
 * The native Python step may block its own worker for >1 second. This separate
 * supervisor remains responsive and publishes an honest 1s heartbeat with the
 * last confirmed counters; it never invents work within an unfinished step.
 */
'use strict';
let child=null,active=null,last=null,paused=false,epoch=0;
function spawn(){
 if(child)return child;
 child=new Worker('ddar-py.engine.worker.js');
 child.onmessage=e=>{
  const d=e.data||{};
  if(d.command==='audit'){self.postMessage(d);return;}
  if(!active||d.requestId!==active.id)return;
  if(d.command==='progress'||d.command==='paused'||d.command==='complete'){
   last=d.result;
   if(d.command==='paused')paused=true;
   self.postMessage(d);
  }else self.postMessage(d);
  if(d.command==='complete'||d.command==='error')active=null;
 };
 child.onerror=e=>{if(active)self.postMessage({command:'error',requestId:active.id,error:e.message||'Pyodide worker failed'});child.terminate();child=null;active=null;};
 return child;
}
self.onmessage=e=>{
 const d=e.data||{};
 if(d.command==='start'){active={id:d.requestId,start:performance.now()};last=null;paused=false;epoch++;
   spawn().postMessage(d);return;}
 if(d.command==='pause'){paused=true;child?.postMessage(d);return;}
 if(d.command==='resume'){paused=false;child?.postMessage(d);return;}
 if(d.command==='audit'){spawn().postMessage(d);return;}
 child?.postMessage(d);
};
setInterval(()=>{
 if(!active||paused)return;
 const snapshot=last||{engine:'Pyodide',phase:'启动内核',depth:0,attempts:0,proved:0,checked:0,totalCandidates:0,queries:[],elapsedMs:0};
 const elapsedMs=Math.round(performance.now()-active.start);
 self.postMessage({command:'progress',requestId:active.id,result:{...snapshot,elapsedMs,heartbeat:true}});
},1000);
