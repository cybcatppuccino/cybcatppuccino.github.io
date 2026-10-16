'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.join(__dirname,'..');
const G=require('../assets/kernel.js'),Library=require('../assets/library.js');
const sandbox={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'assets/library-data.js'),'utf8'),sandbox);
const data=sandbox.window.GeomLibraryData;
assert.equal(data.schema,'geomtool.library/v4.7');
assert.equal(data.entries.length,1170);
assert.deepEqual(Object.fromEntries(Object.entries(data.summary)),{'AlphaGeometry 1':265,Newclid:879,'AlphaGeometry 2':26});
for(const e of data.entries){assert.ok(e.goals.length&&e.goals[0].ag1&&e.goals[0].ag2);assert.ok(e.id&&e.sourcePath);}
const midline=data.entries.find(e=>e.source==='Newclid'&&e.title==='r06');
assert.ok(midline&&midline.goals[0].ag1==='para e f b c');
const midgraph=Library.convert(midline,{attempts:5});
assert.equal(midgraph.complete,true);
const checkMid=new G.Kernel(midgraph.graph);
assert.equal(checkMid.node('e').op,'midpoint');
assert.equal(checkMid.node('f').op,'midpoint');
const ag2sample=data.entries.find(e=>e.source==='AlphaGeometry 2'&&e.title==='2000_p1');
assert.ok(ag2sample&&ag2sample.coords.a&&ag2sample.goals[0].ag2==='cong e p e q');
const ag2graph=Library.convert(ag2sample);
assert.equal(ag2graph.coordinateOnly,true);
assert.equal(new G.Kernel(ag2graph.graph).geom('a').x,ag2sample.coords.a[0]);

const stats={sources:{},complete:0,partial:0,empty:0,invalid:0,converted:0,clauses:0,bad:[],unmapped:{}};
const only=parseInt(process.env.LIMIT||'0',10);
for(const e of (only?data.entries.slice(0,only):data.entries)){
 let res;
 try{res=Library.convert(e,{attempts:5});const trial=new G.Kernel(res.graph);assert.equal(trial.list().length,res.graph.nodes.length);}
 catch(err){stats.invalid++;stats.bad.push(e.title+': '+err.stack);continue;}
 const s=stats.sources[e.source]||(stats.sources[e.source]={complete:0,partial:0,empty:0});
 stats.converted+=res.converted;stats.clauses+=res.total;
 if(!res.graph.nodes.length){stats.empty++;s.empty++;}
 if(res.complete){stats.complete++;s.complete++;}else{stats.partial++;s.partial++;}
 for(const clause of res.skipped){const op=clause.match(/=\s*(\w+)/)?.[1]||'<other>';stats.unmapped[op]=(stats.unmapped[op]||0)+1;}
}
console.log(JSON.stringify(stats,null,2));
if(stats.invalid)process.exit(1);
