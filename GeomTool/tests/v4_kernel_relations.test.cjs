'use strict';
const assert=require('node:assert/strict');
const {Kernel,previewConstruction}=require('../assets/kernel.js');
const Relation=require('../assets/relations/detectors.js');
const k=new Kernel();
const pt=(id,x,y)=>k.execute({op:'free',id,params:{x,y}});
for(let i=0;i<26;i++)pt(String.fromCharCode(65+i),i,Math.sin(i));
assert.equal(k.nextId('point'),'P1');
pt('P1',0,1);assert.equal(k.nextId('point'),'P2');
const collision=pt('H',3,3);assert.equal(collision.id,'P2');
const A=k.node('A'),B=k.node('B'),C=k.node('C');
assert.equal(k.node('P2').label,'P2');
const line=k.execute({op:'line',id:'l1',inputs:['A','B']});
assert.equal(k.execute({op:'line',id:'l1',inputs:['A','C']}).id,'l2');
assert.equal(k.execute({op:'circleRadius',id:'c1',inputs:['A'],params:{radius:1}}).id,'c1');
assert.equal(k.execute({op:'circleRadius',id:'c1',inputs:['B'],params:{radius:1}}).id,'c2');
assert.equal(k.execute({op:'free',id:'p_0',params:{x:0,y:0}}).id,'P3');
const rename=k.execute({op:'rename',id:'A',label:'Z2'});
assert.equal(rename.id,'Z2');assert.equal(rename.label,'Z2');assert.equal(k.node('A'),null);assert.equal(k.node('l1').inputs[0],'Z2');
assert.throws(()=>k.execute({op:'rename',id:'Z2',label:'l_Re'}),/保留/);
k.undo();assert.equal(k.node('A').label,'A');assert.equal(k.node('l1').inputs[0],'A');k.redo();
const poly=k.execute({op:'polygon',id:'poly1',inputs:['Z2','B','C']});
const sides=k.list().filter(n=>n.params?.polygonParent===poly.id);
assert.equal(sides.length,3);assert.deepEqual(sides.map(n=>n.id),['sZ2B','sBC','sCZ2']);
assert(sides.every(n=>n.valid&&n.geom.kind==='line'&&n.geom.extent==='segment'));
const mid=k.execute({op:'midpoint',inputs:['Z2','B']});
assert(mid.valid);const segment=sides[0];
const part=k.execute({op:'onLine',inputs:[segment.id],params:{t:.4}});
assert(part.valid);const sec=k.execute({op:'circleRadius',inputs:['Z2'],params:{radius:Math.hypot(segment.geom.d.x,segment.geom.d.y)/2}});
const branches=require('../assets/kernel.js').intersectionBranches(segment.geom,sec.geom);
assert(branches.some(Boolean),'polygon edge must really intersect the circle');
const branch=branches.findIndex(Boolean);
const cross=k.execute({op:'intersection',inputs:[segment.id,sec.id],params:{branch}});
assert(cross.valid,'smart intersection must also be constructible from polygon edges');
const predicted=previewConstruction('circumcircle',[k.geom('Z2'),k.geom('B'),k.geom('C')]);
assert.equal(predicted.kind,'circle');
const expected=k.execute({op:'circumcircle',inputs:['Z2','B','C']}).geom;
assert(Math.abs(predicted.r-expected.r)<1e-10);
assert.equal(previewConstruction('perpendicular',[k.geom('Z2'),k.geom('l1')]).kind,'line');
const serial=k.serialize(),loaded=new Kernel(serial);
assert.equal(loaded.list().filter(n=>n.params.polygonParent===poly.id).length,3);
assert(loaded.list().every(n=>n.label===n.id));
const count=k.list().length;
k.execute({op:'delete',id:poly.id});assert.equal(k.list().filter(n=>n.params.polygonParent===poly.id).length,0);
k.undo();assert.equal(k.list().length,count);
// Loading original v3.5 metadata converts GTREF and divergent display labels.
const old={schema:'geomtool.ir/v1',nodes:[
 {id:'GTREF_origin',label:'0',op:'referencePoint',inputs:[],params:{x:0,y:0},style:{system:true}},
 {id:'oldpoint',label:'T9',op:'free',inputs:[],params:{x:2,y:0}},
 {id:'oldline',label:'lold',op:'line',inputs:['GTREF_origin','oldpoint'],params:{}}
]};
const upgraded=new Kernel(old);
assert(upgraded.node('p_0')?.style.system);assert.equal(upgraded.node('T9')?.label,'T9');
assert.deepEqual(upgraded.node('lold').inputs,['p_0','T9']);
function sc(kernel,kinds){const source={points:[],lines:[],circles:[],construction:[]};
 for(const n of kernel.snapshot().nodes){if(!n.valid)continue;const g=n.geom;source.construction.push({id:n.id,label:n.label,op:n.op,inputs:n.inputs,params:n.params});
 if(g.kind==='point')source.points.push({id:n.id,x:g.x,y:g.y});
 if(g.kind==='line')source.lines.push({id:n.id,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y,drawType:g.extent==='segment'?'lineSegment':g.extent});
 if(g.kind==='circle')source.circles.push({id:n.id,cx:g.c.x,cy:g.c.y,r:g.r});}
 return Relation.scan(source,{kinds,perturb:false});}
const scene=new Kernel();for(const [id,x,y] of [['A',0,0],['B',3,0],['C',0,3],['D',-3,1]])scene.execute({op:'free',id,params:{x,y}});
for(const p of ['B','C','D'])scene.execute({op:'line',inputs:['A',p]});
let rel=sc(scene,['concurrent']);const trivial=rel.entries.filter(e=>e.kind==='concurrent'&&e.title.includes('A'));
assert(trivial.length,'should report concurrency as a condition');assert(trivial.every(e=>e.source==='construction'&&e.score<=12),JSON.stringify(trivial));
scene.execute({op:'midpoint',id:'M',inputs:['A','B']});
rel=sc(scene,['midpoint','distanceRatio','quadraticRatio']);
assert(rel.entries.some(e=>e.kind==='midpoint'&&e.source==='construction'&&e.score<=22));
const equality=rel.entries.filter(e=>['distanceRatio','quadraticRatio'].includes(e.kind)&&e.metric?.factor===1&&
 (e.metric?.high||[]).some(x=>x.key==='D:A|M')&&(e.metric?.high||[]).some(x=>x.key==='D:B|M'));
assert(equality.length,'midpoint MA=MB should be present');
assert(equality.some(e=>e.origin==='construction'&&e.score<=42),'midpoint equality must not count as new theorem');
console.log('PASS v4 names, reference migration, polygon edges, undo/roundtrip, previews, construction relation provenance');
