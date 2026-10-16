'use strict';
/* v4 dumb-diagram audit: definition identities vs independent concurrency.
 * Probe small constructions with coordinate changes, not just one picture.
 */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js');
const {scan}=require('../assets/relations/detectors.js');
function add(k,op,id,inputs=[],params={}){return k.execute({op,id,inputs,params});}
function point(k,id,x,y){return add(k,'free',id,[],{x,y});}
function snapshot(k){const s={points:[],lines:[],circles:[],construction:[]};for(const n of k.snapshot().nodes){if(!n.valid)continue;const g=n.geom;s.construction.push({id:n.id,op:n.op,inputs:n.inputs,params:n.params});if(g.kind==='point')s.points.push({id:n.id,x:g.x,y:g.y});if(g.kind==='line')s.lines.push({id:n.id,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y,drawType:g.extent==='segment'?'lineSegment':g.extent});if(g.kind==='circle')s.circles.push({id:n.id,cx:g.c.x,cy:g.c.y,r:g.r});}return s;}
const report=[];
for(const scale of [0.75,1,3])for(const shift of [-6,0,7]){
 const k=new Kernel(),xy=(x,y)=>[x*scale+shift,y*scale-shift];
 for(const [id,x,y] of [['A',0,0],['B',4,0],['C',-1,3],['D',-3,-2]])point(k,id,...xy(x,y));
 for(const p of ['B','C','D'])add(k,'line','l'+p,['A',p]);
 const radar=scan(snapshot(k),{kinds:['concurrent'],perturb:false});
 const rows=radar.entries.filter(e=>e.kind==='concurrent'&&e.title.includes('A'));
 assert(rows.length,'fan concurrency omitted');assert(rows.every(e=>e.source==='construction'&&e.score<=12),'fan must be a low-scored condition');
 report.push({case:'three_line_fan',scale,shift,conditions:rows.map(e=>({score:e.score,origin:e.source,title:e.title}))});
 }
for(const scale of [0.6,1,2.5]){
 const k=new Kernel(), xy=(x,y)=>[scale*x,scale*y];
 for(const [id,x,y] of [['A',0,0],['B',4,0],['C',1,3]])point(k,id,...xy(x,y));
 add(k,'midpoint','M',['A','B']);
 const radar=scan(snapshot(k),{kinds:['midpoint','distanceRatio','quadraticRatio','collinear'],perturb:false});
 const mids=radar.entries.filter(e=>e.kind==='midpoint'&&e.title.includes('M'));
 assert(mids.length&&mids.every(e=>e.source==='construction'&&e.score<=22),'midpoint is a direct definition');
 const eq=radar.entries.filter(e=>(e.kind==='distanceRatio'||e.kind==='quadraticRatio')&&e.metric?.factor===1&&e.metric?.high?.some(v=>v.key==='D:A|M')&&e.metric?.high?.some(v=>v.key==='D:B|M'));
 assert(eq.length&&eq.some(e=>e.origin==='construction'&&e.score<=42),'MA=MB must be reported as construction property');
 report.push({case:'midpoint_equal_legs',scale,midpoint:mids.map(e=>({score:e.score,origin:e.source})),equalLength:eq.map(e=>({score:e.score,origin:e.source,source:e.source}))});
 }
for(const scale of [.7,1,4]){
 const k=new Kernel(),xy=(x,y)=>[x*scale,y*scale];
 for(const [id,x,y] of [['A',0,0],['B',4,0],['C',1,3]])point(k,id,...xy(x,y));
 add(k,'orthocenter','H',['A','B','C']);
 for(const [id,arr] of [['hA',['A','B','C']],['hB',['B','A','C']],['hC',['C','A','B']]])add(k,'perpendicular3',id,arr);
 const radar=scan(snapshot(k),{kinds:['concurrent'],perturb:false});
 const rows=radar.entries.filter(e=>e.kind==='concurrent'&&e.title.includes('H'));
 assert(rows.length&&rows.every(e=>e.source!=='construction'&&e.score>12),'altitude concurrence must not be mistaken for shared input point');
 report.push({case:'nontrivial_altitudes',scale,discoveries:rows.map(e=>({score:e.score,origin:e.source,title:e.title}))});
 }
const out={schema:'geomtool.v4.dummy-audit/v1',scenes:report.length,results:report};
const target=path.join(__dirname,'../docs/V4_DUMMY_SCENE_AUDIT.json');fs.writeFileSync(target,JSON.stringify(out,null,2));
console.log(`PASS v4 dumb diagrams: ${report.length} translated/scaled scenes; construction-only fan capped 12; midpoint equal lengths marked; true altitude concurrency retained`);
