'use strict';
// Focused semantic & numerical v3.1 contract checks, not exhaustive acceptance.
const assert=require('node:assert/strict');
const {scan,registry}=require('../assets/relations/detectors.js');
const {makeContext}=require('../assets/relations/core.js');
const kinds=registry.map(r=>r.id);
function scene(points,lines=[],circles=[]){return {points:points.map(([id,x,y])=>({id,x,y,label:id})),lines:lines.map(([id,a,b])=>({id,label:id,x1:a[0],y1:a[1],x2:b[0],y2:b[1]})),circles:circles.map(([id,x,y,r])=>({id,label:id,cx:x,cy:y,r}))};}
function run(s,ks){return scan(s,{kinds:ks||undefined,tolerance:'loose',perturb:true});}
let checks=0;function ok(x,m){assert(x,m);checks++;}
for(const obsolete of ['equalVectors','orthodiagonal','equidistant','equalChords'])ok(!kinds.includes(obsolete),obsolete+' removed from registry');
const line=scene([['A',0,0],['B',1,0],['C',2,0],['D',3,0],['E',4,0],['A2',0,0],['A3',0,0]]);
const lc=run(line,['coincident','collinear']);
ok(lc.entries.filter(x=>x.kind==='coincident').length===1,'coincident cluster displayed once');
ok(lc.entries.filter(x=>x.kind==='collinear').length===1,'five collinear points reported once, not ten triples');
ok(lc.entries.find(x=>x.kind==='collinear').title.includes('E'),'maximal point set, including E');
ok(lc.entries.find(x=>x.kind==='collinear').searchText.includes('A2'),'equivalent point aliases are searchable');
ok(lc.objects.points===5&&lc.objects.originalPoints===7,'geometry operates on merged points');
ok(lc.tolerance==='strict'&&lc.perturbations===2,'strict regardless of caller and bilateral stability check');
const circlePts=[['A',1,0],['B',0,1],['C',-1,0],['D',0,-1],['E',Math.SQRT1_2,Math.SQRT1_2]];
const cs=run(scene(circlePts,[],[['unit',0,0,1],['unitCopy',0,0,1]]),['concyclic','pointOnCircle']);
ok(cs.entries.filter(x=>x.kind==='concyclic').length===1,'maximal five-point circle, no five quadruple results');
ok(cs.entries.filter(x=>x.kind==='pointOnCircle').length===0,'circle-point incidence implied by maximal concyclicity suppressed');
ok(cs.entries.find(x=>x.kind==='concyclic').searchText.includes('unitCopy'),'circle aliases searchable');
const offs=run(scene([['A',0,0],['B',3,0],['C',5,0.0000005]]),['collinear']);
ok(offs.count===0,'strict scanner rejects near collinearity at 5e-7');
const doubleLine=scene([['A',0,0],['B',2,0],['C',3,0],['D',0,2],['E',2,2],['F',3,2]], [['l1',[0,0],[1,0]],['l2',[1,0],[4,0]]]);
const par=run(doubleLine,['parallel']);
ok(par.entries.filter(e=>e.title.includes('l1')&&e.title.includes('DE')).length===1,'same two horizontal supporting lines reported once');
ok(par.entries.find(e=>e.title.includes('l1')).searchText.includes('l2')&&par.entries.find(e=>e.title.includes('l1')).searchText.includes('A'),'all explicit/implicit aliases can be searched');
const eq=run(scene([['A',0,0],['B',2,0],['C',0,2],['D',2,2],['M',1,0],['N',1,2]]),['distanceRatio']);
ok(eq.entries.some(e=>e.metric?.factor===1 && e.title.includes('等长')),'equal-distance cases are retained as one metric equality chain');
// One altitude/median/bisector/bisector-of-opposite-side relationship, not four.
const special=run(scene([['V',0,2],['L',-2,0],['R',2,0],['M',0,0]]),['specialLine']);
const role=special.entries.filter(e=>e.title.includes('△')&&e.title.includes('中线')&&e.title.includes('高线'));
ok(role.length>0&&role.some(e=>e.title.includes('角平分线')),'special-line roles merged');
// Virtual circle center O through B and C has the same radius, hence one circle.
const tan=run(scene([['O',0,0],['B',0,2],['C',0,-2],['D',2,-3],['E',2,3]]),['circleLineTangent']);
const tangents=tan.entries.filter(e=>e.marks.some(m=>m.type==='circle'&&m.ref?.type==='centerThrough'&&m.ref.a==='O'));
const uniqueTangent=new Set(tangents.map(e=>e.id));
ok(uniqueTangent.size===1&&tangents.length===1,'O through B and O through C define one circle tangent away from its defining radius endpoints');
ok(tangents[0].detail.includes('O,C')&&tangents[0].searchText.includes('O,C'),'equivalent center-through-point circle remains visible and searchable');
console.log(`PASS v3.1 semantic strict & merge tests: ${checks} assertions; ${registry.length} independent categories`);
