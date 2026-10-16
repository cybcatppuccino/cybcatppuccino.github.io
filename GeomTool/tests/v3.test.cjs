'use strict';
/* A small, logic-centered v3 suite; full interactive acceptance is left to the user. */
const assert=require('node:assert/strict');
const {Kernel,intersections,intersectionBranches}=require('../assets/kernel.js');
const {scan,registry}=require('../assets/relations/detectors.js');
const {parse}=require('../assets/expression.js');
let checks=0;function ok(test,msg){assert(test,msg);checks++;}
function equal(a,b,eps=1e-8){return Math.abs(a-b)<eps;}
const k=new Kernel();function mk(op,id,inputs=[],params={}){return k.execute({op,id,inputs,params});}
mk('free','O1',[],{x:0,y:0});mk('free','O2',[],{x:4,y:0});mk('circleRadius','c1',['O1'],{radius:3});mk('circleRadius','c2',['O2'],{radius:3});
mk('intersection','lower',['c1','c2'],{branch:0});mk('intersection','upper',['c2','c1'],{branch:0});
ok(k.node('upper').inputs[0]==='c1' && k.node('upper').params.branch===1,'reversed circle selection maps to opposite chirality');
ok(k.geom('lower').y<0&&k.geom('upper').y>0,'circle-circle roots are different and signed');
mk('intersection','upper2',['c1','c2'],{branch:1});ok(equal(k.geom('upper').y,k.geom('upper2').y),'same mathematical branch stable with reordered carriers');
k.execute({op:'move',id:'O2',x:6,y:0});ok(k.geom('upper')===null,'second root temporarily invalid at tangency');
k.execute({op:'move',id:'O2',x:4,y:1});ok(k.geom('upper')&&k.geom('lower')&&Math.hypot(k.geom('upper').x-k.geom('lower').x,k.geom('upper').y-k.geom('lower').y)>1,'distinct intersections reappear after tangency');
const v={x:k.geom('O2').x-k.geom('O1').x,y:k.geom('O2').y-k.geom('O1').y},b=k.geom('upper');
ok(v.x*b.y-v.y*b.x>0,'circle chirality is fixed by ordered centers');
const ray=new Kernel();function m(op,id,inputs=[],params={}){return ray.execute({op,id,inputs,params});}
m('free','O',[],{x:0,y:0});m('free','Q',[],{x:0,y:0});m('free','R',[],{x:1,y:0});m('circleRadius','c',['O'],{radius:2});m('ray','r',['Q','R']);
const slots=intersectionBranches(ray.geom('r'),ray.geom('c'));
ok(slots[0]===null&&equal(slots[1].x,2),'ray retains a hole in the lower root slot');
m('intersection','X',['r','c'],{branch:1});
m('free','Q2',[],{x:-3,y:0});ray.execute({op:'delete',id:'Q2'}); // incidental topological changes
ray.execute({op:'move',id:'Q',x:-3,y:0});
ok(equal(ray.geom('X').x,2),'ray far-branch intersection cannot swap to the near branch');
ok(intersections(ray.geom('r'),ray.geom('c')).length===2,'two valid intersections available on ray');
const legacy=ray.serialize();legacy.nodes.find(n=>n.id==='Q').params.x=0;
const x=legacy.nodes.find(n=>n.id==='X');delete x.params.slotVersion;x.params.branch=0;delete x.params.canonicalInputs;
const reloaded=new Kernel();reloaded.load(legacy,false);
ok(reloaded.node('X').params.branch===1&&equal(reloaded.geom('X').x,2),'legacy dense branch indexing migrates to stable slots');
for(const [expr,answer] of [['1+sqrt2',1+Math.sqrt(2)],['-sqrt(2)',-Math.SQRT2],['5/7',5/7],['2^3+pi',8+Math.PI],['-(1+sqrt(3))',-1-Math.sqrt(3)]])ok(equal(parse(expr),answer),expr+' parsed');
for(const expr of ['eval(1)','3/0','sqrt(-1)','sqrt(2','unknown']){assert.throws(()=>parse(expr));checks++;}
function snap(points,lines=[]){return {points:points.map(([id,x,y])=>({id,x,y,label:id})),lines:lines.map(([id,a,b])=>({id,x1:a[0],y1:a[1],x2:b[0],y2:b[1],label:id}))};}
const square=snap([['A',0,0],['B',4,0],['C',0,3],['D',4,3],['E',0,5],['F',4,5]], [['bottom',[0,0],[4,0]],['b2',[0,0],[1,0]],['top',[0,3],[4,3]]]);
const parallel=scan(square,{kinds:['parallel']}).entries;
const horizontal=parallel.filter(t=>t.title.includes('bottom')&&t.title.includes('top'));
ok(horizontal.length===1,'coincident representations of same supporting line are deduplicated');
const points=snap([['A',0,0],['B',3,0],['C',0,1],['D',3,2],['E',0,5],['F',3,5],['G',0,6],['H',3,7]]);
const equals=scan(points,{kinds:['equalAngle']}).entries;
ok(equals.some(e=>e.detail==='双线夹角'&&e.marks.filter(m=>m.type==='line').length===4),'equal angle supports AB/CD versus EF/GH');
const complementary=scan(snap([['A',1,0],['O',0,0],['B',1,1],['C',1,1],['P',0,2],['D',1,3]]),{kinds:['complementaryAngle']}).entries;
ok(complementary.length>0,'complementary angles found');
ok(registry.length>=45,'v3.1 retains a broad detector registry after intentional consolidation');
console.log(`PASS v3 targeted logic: ${checks} assertions, ${registry.length} categories`);
