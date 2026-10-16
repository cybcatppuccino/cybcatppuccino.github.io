'use strict';
const assert=require('node:assert/strict');
const provenance=require('../assets/relations/provenance.js');
const logic=require('../assets/relations/construction-logic.js');
const {scan}=require('../assets/relations/detectors.js');
const {Kernel}=require('../assets/kernel.js');
const s=(a,b)=>({type:'segment',a,b});
const nodes=['A','B','C','D','P','Q'].map(id=>({id,op:'free',inputs:[]}));
for(const [id,inputs] of [['M1',['A','B']],['M2',['A','M1']],['M3',['A','M2']],['Z',['A','M3']]])nodes.push({id,op:'midpoint',inputs});
const relation=(id,a,b,c,d)=>({id,kind:'distanceRatio',marks:[s(a,b),s(c,d)],score:72,origin:'discovered',metric:{factor:2,high:[{key:'D:'+ [a,b].sort().join('|')}],low:[{key:'D:'+ [c,d].sort().join('|')}]}});
const entries=[relation('shallow','A','B','C','D'),relation('deep','Z','P','C','D')];
for(let i=0;i<12;i++)entries.push(relation('common'+i,'A','B',i%2?'P':'Q',i%3?'C':'D'));
provenance.score(entries,{construction:nodes});
const deep=entries.find(x=>x.id==='deep'),shallow=entries.find(x=>x.id==='shallow');
assert.ok(deep.score>shallow.score,`deep and sparse result must outrank shallow frequent result: ${deep.score}>${shallow.score}`);
assert.ok(deep.scoreComponents.dependencyDepth>=4);
assert.ok(deep.scoreComponents.graphNovelty>shallow.scoreComponents.graphNovelty);
assert.ok(deep.scoreComponents.independentInformation>shallow.scoreComponents.independentInformation);
// More consequences are not automatically more *interesting* when they are
// just a definition of the same node.
const defin=relation('definition','M1','A','M1','B');defin.origin='construction';
const discover=relation('discovered','Z','A','P','Q');provenance.score([defin,discover],{construction:nodes});
assert.ok(discover.score>defin.score,'construction definition penalized versus nontrivial sparse discovery');
// Verify that weaker grouped relation is not swallowed if it includes an
// additional independent object, while a complete primitive orthogonality
// is allowed to be explained by an orthocenter fact.
const k=new Kernel();for(const [id,x,y] of [['A',0,0],['B',4,0],['C',1.3,3.1],['X',7,2]])k.execute({op:'free',id,params:{x,y}});k.execute({op:'orthocenter',id:'H',inputs:['A','B','C']});
const shot={construction:[],points:[],lines:[],circles:[]};for(const n of k.snapshot().nodes){shot.construction.push({id:n.id,op:n.op,inputs:n.inputs});if(n.geom?.kind==='point')shot.points.push({id:n.id,x:n.geom.x,y:n.geom.y});}
const {entries:found,constructionConsequences}=scan(shot,{kinds:['orthocentricSystem','perpendicular','specialPoint'],perturb:false});
assert.ok(constructionConsequences>=11);
assert.ok(found.some(e=>e.kind==='orthocentricSystem'));
const facts=logic.makeFacts(shot.construction).filter(f=>f.node==='H');
assert.ok(facts.some(f=>f.kind==='equivalentOrthocenter')&&facts.filter(f=>f.kind==='perpendicular').length===6);
console.log('PASS v3.4 implication/graph scoring: deep rare vs shallow frequent, direct facts vs discoveries, orthocenter operands');
