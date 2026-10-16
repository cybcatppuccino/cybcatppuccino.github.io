const assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js');
const {scan,registry}=require('../assets/relations/detectors.js');
const {makeResolver}=require('../assets/relations/overlay.js');
function scene(spec){const k=new Kernel();for(const [id,x,y] of spec)k.execute({op:'free',id,params:{x,y}});return k;}
function snap(k){const nodes=k.snapshot().nodes;return {points:nodes.filter(n=>n.kind==='point').map(n=>({id:n.id,x:n.geom.x,y:n.geom.y})),lines:nodes.filter(n=>n.kind==='line').map(n=>({id:n.id,x1:n.geom.p.x,y1:n.geom.p.y,x2:n.geom.p.x+n.geom.d.x,y2:n.geom.p.y+n.geom.d.y,drawType:n.geom.extent})),circles:nodes.filter(n=>n.kind==='circle').map(n=>({id:n.id,cx:n.geom.c.x,cy:n.geom.c.y,r:n.geom.r}))};}
function get(k,kinds){return scan(snap(k),{kinds,tolerance:'normal'}).entries;}
function add(k,op,id,inputs){k.execute({op,id,inputs});}
const k=scene([['A',0,0],['B',4,0],['C',0,3],['D',4,3],['M',2,0],['P',4,1]]);add(k,'line','l',['C','D']);
const para=get(k,['parallel']);assert(para.some(e=>e.title.includes('AB')&&e.title.includes('CD')),'AB ∥ CD without explicit AB/CD lines');
assert(para.some(e=>e.title.includes('AB')&&e.title.includes('l')),'implicit AB ∥ explicit l');
const ratio=get(k,['distanceRatio']);assert(ratio.some(e=>[...(e.metric?.high||[]),...(e.metric?.low||[])].some(m=>m.type==='pointLine'&&m.name.includes('d(P,'))),'point-to-line and point-to-BC distances retained, including semantic aliases');
assert(ratio.some(e=>e.title.includes('2')),'simple segment ratios');
const mids=get(k,['midpoint']);assert(mids.some(e=>e.title.includes('M')&&e.title.includes('AB')),'midpoint');
const square=scene([['A',0,0],['B',2,0],['C',2,2],['D',0,2],['O',1,1]]);
assert(get(square,['square']).length===1,'square once');
assert(get(square,['specialPoint']).some(e=>e.title.includes('外心')),'circumcenter');
const tan=scene([['A',0,0],['B',0,2],['C',2,3],['D',2,5]]);
assert(get(tan,['circleLineTangent']).some(e=>e.title.includes('⊙(A,B)')&&e.title.includes('CD')),'virtual circle tangent at an uncreated foot independent of its radius witness');
const angle=scene([['A',1,0],['B',0,0],['C',0,1]]);
const found=get(angle,['fixedAngle']);const right=found.find(e=>e.title.includes('90°')&&e.title.includes('ABC'));
assert(right&&right.marks.some(m=>m.type==='angle'&&m.v==='B'&&m.a==='A'&&m.b==='C'),'arc uses actual BA and BC rays');
const resolve=makeResolver(id=>angle.geom(id));assert(resolve.line({type:'pair',a:'A',b:'B'}),'implicit line resolved from points');
assert(registry.length>=30,'wide independent detector registry');
console.log('PASS v2.2 targeted relation tests: 11 assertions, '+registry.length+' detector categories');
