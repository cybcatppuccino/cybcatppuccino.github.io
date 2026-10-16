'use strict';
const assert=require('assert');
const {Kernel}=require('../assets/kernel.js');
const cmp=require('../assets/ddar-compiler.js');
const make=(dict)=>{const k=new Kernel();for(const [id,xy] of Object.entries(dict))k.execute({op:'free',id,params:{x:xy[0],y:xy[1]}});return k;};
function test(name,dict,op,id,inputs,expect){
 const k=make(dict);k.execute({op,id,inputs});const c=cmp.compile(k.snapshot());
 const mine=c.predicates.filter(p=>p.source===op),a=new Set(mine.map(p=>p.name));
 for(const prop of expect)assert(a.has(prop),`${name}: missing ${prop}; ${JSON.stringify(c.limitations)}`);
 console.log(name,'point',k.geom(id),'predicates',mine.length,'limitations',c.limitations.length);
 return {k,c};
}
let x=test('similar nondegenerate',{A:[0,0],B:[2,0],C:[.5,2],D:[3,1],E:[3,5]},'similarPoint','F',['A','B','C','D','E'],['eqratio','eqangle']);
assert.deepStrictEqual([x.k.geom('F').x,x.k.geom('F').y],[-1,2]);
x=test('similar collinear',{A:[0,0],B:[2,0],C:[6,0],D:[-1,1],E:[1,4]},'similarPoint','F',['A','B','C','D','E'],['eqratio','eqangle']);
assert(Math.abs(x.k.geom('F').x-5)<1e-9&&Math.abs(x.k.geom('F').y-10)<1e-9);
x=test('similar duplicate source',{A:[0,0],B:[0,0],C:[1,0],D:[2,0],E:[2,0]},'similarPoint','F',['A','B','C','D','E'],[]);
assert(x.c.limitations.some(s=>s.includes('退化')));
x=test('equal-angle line',{A:[0,0],B:[1,0],C:[1,1],D:[4,4],E:[4,1]},'equalAngleLine','l',['A','B','C','D','E'],['eqangle']);
assert(x.c.lines.l?.includes('E'));
assert(x.c.virtualDescriptions.some(v=>v.id.includes('@line:l')));
console.log('v5.2 JS constructors PASS');
