'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const win={window:{}};vm.runInNewContext(fs.readFileSync(require('path').join(__dirname,'../assets/library-data.js'),'utf8'),win);
const lib=require('../assets/library.js'),G=require('../assets/kernel.js');
const records=win.window.GeomLibraryData.entries;const issues=[],counts={};let tested=0;
const diff=(a,b)=>Math.abs(a-b)/(1+Math.abs(a)+Math.abs(b));
const sub=(a,b)=>({x:a.x-b.x,y:a.y-b.y});const cross=(a,b)=>a.x*b.y-a.y*b.x;
const dot=(a,b)=>a.x*b.x+a.y*b.y;const norm=a=>Math.hypot(a.x,a.y);
const angle=(a,b)=>Math.atan2(cross(a,b),dot(a,b));const dist=(a,b)=>norm(sub(a,b));
function good(cond,e,step,op){tested++;counts[op]=(counts[op]||0)+1;if(!cond&&issues.length<100)issues.push(e.source+' '+e.title+' '+step.raw+' '+op);}
for(const entry of records.filter(e=>e.kind==='construction')){
 const r=lib.convert(entry,{attempts:5});const k=new G.Kernel(r.graph);if(!r.graph.nodes.length)continue;
 const valid=new Set(k.list().filter(n=>n.valid).map(n=>n.id));
 for(const st of entry.steps){if(!st.outputs.every(x=>valid.has(x)))continue;
 const O=st.operations||[];for(const item of O){const a=item.args||[],id=st.outputs[0],P=k.node(id)?.geom;
  const g=n=>k.node(n)?.geom;const op=item.op;if(!P||!g(a[0]))continue;
  switch(op){
  case 'midpoint':if(a.length!==2)break;good(diff(dist(P,g(a[0])),dist(P,g(a[1])))<1e-7 && norm(sub(P,{x:(g(a[0]).x+g(a[1]).x)/2,y:(g(a[0]).y+g(a[1]).y)/2}))<1e-7,entry,st,op);break;
  case 'on_line':if(a.length!==2)break;good(Math.abs(cross(sub(P,g(a[0])),sub(g(a[1]),g(a[0]))))/(1+dist(g(a[0]),g(a[1])))/ (1+dist(P,g(a[0])))<1e-7,entry,st,op);break;
  case 'on_tline':if(a.length!==3)break;good(Math.abs(dot(sub(P,g(a[0])),sub(g(a[2]),g(a[1]))))/(1+norm(sub(P,g(a[0])))*norm(sub(g(a[2]),g(a[1]))))<1e-7,entry,st,op);break;
  case 'on_pline':if(a.length!==3)break;good(Math.abs(cross(sub(P,g(a[0])),sub(g(a[2]),g(a[1]))))/(1+norm(sub(P,g(a[0])))*norm(sub(g(a[2]),g(a[1]))))<1e-7,entry,st,op);break;
  case 'on_bline':if(a.length!==2)break;good(diff(dist(P,g(a[0])),dist(P,g(a[1])))<1e-7,entry,st,op);break;
  case 'on_circle':case 'eqdistance':if(a.length!==2&&a.length!==3)break;good(diff(dist(P,g(a[0])),dist(g(a[1]),g(a[2]||a[0])))<1e-7,entry,st,op);break;
  case 'lc_tangent':if(a.length!==2)break;good(Math.abs(dot(sub(P,g(a[0])),sub(g(a[0]),g(a[1]))))<1e-7*(1+dist(P,g(a[0]))*dist(g(a[0]),g(a[1]))),entry,st,op);break;
  case 'on_aline':if(a.length!==5)break;
    {const theta=angle(sub(g(a[0]),P),sub(g(a[1]),g(a[0]))),target=angle(sub(g(a[3]),g(a[2])),sub(g(a[4]),g(a[3])));
     good(Math.abs(Math.sin(theta-target))<1e-7&&Math.cos(theta-target)>0,entry,st,op);}break;
  }
 }
 }
}
console.log(JSON.stringify({tested,counts,issueCount:issues.length,issues:issues.slice(0,30)},null,2));
if(issues.length)process.exit(1);
