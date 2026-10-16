'use strict';
const assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js');
const C=require('../assets/pretty-constraints.js'),Pretty=require('../assets/pretty.js');
const Compiler=require('../assets/ddar-compiler.js'), DDAR=require('../assets/ddar.js');
let tested=0;
function check(value,message){assert.ok(value,message);tested++;}
const k=new Kernel();
for(const [id,x,y] of [['A',0,0],['B',2,0],['C',.8,.7],['D',1,3],['E',4,2],['F',2,1],['G',1.5,1],['H',2.3,3],['I',-1,1],['J',.2,-1],['R',1,-2],['S',2,1],['T',4,4],['U',1,4],['V',-1,2],['W',3,-1]])
 k.execute({op:'free',id,params:{x,y}});
k.execute({op:'line',id:'l',inputs:['A','B']});
k.execute({op:'circle',id:'c',inputs:['A','B']});
k.execute({op:'circle',id:'c2',inputs:['C','D']});
const syntax=[
 'col A B C D E','collinear(A B C)','collinear A B C D','cyc A B C D E','cyclic(A,B,C,D)',
 'dist A B = d(C D) = 2 E F = sqrt5 *G H = (sqrt5)G H',
 'distance A B = sqrt(5) * G H','A B / D E = C F / I J','A B / D E = C F / I J + 1',
 'd(A,B)^2 = (C D)^2','(A B)^2 + (C D)^2 = 10',
 'ang(A,B,C)=ang A1 B1 C1', // intentionally unknown object, handled below
 'ang(A,B,C)=ang D E F G','ang D E F G = 2 ang(R S T U)',
 'ang(A B C)+angle(D,E,F,G)=90','ang A B C D = pi/2','ang(A B C)=90','angle A B C D = 90',
 'perp A B C D','perp l (C D)','perp(c, c2)','perp l c',
 'para A B C D','para l (A,B)','para(c,c2)','para c (A B)',
 'tangent(c, c2)','tan c l','tan c (A,B)','A in c','A in l',
 'sub A c','subset A c(B C D)','subset(A, l)',
 'cong (A B C) (D, E, F)','cong A B C D E F',
 'sim (A B C) (D E F)','simtri A B C D E F',
 'S(A,B,C) = area D E F','area A B C = 2 S(D,E,F)',
];
for(const x of syntax){
 if(x.includes('A1')){assert.throws(()=>C.parse(x,k));tested++;continue;}
 const p=C.parse(x,k);check(p&&Array.isArray(C.residuals(p,k)),x);
 check(C.residuals(p,k).every(Number.isFinite),'finite '+x);
}
for(const x of ['wat A B C','col A B','dist A B =','subset(A,unknown)','perp A B C','sin 1 = 0','dist A UNKNOWN = 1']){
 assert.throws(()=>C.parse(x,k),x);tested++;
}
function small(){const x=new Kernel();for(const [id,a,b] of [['A',0,0],['B',4,.5],['C',2,2.1],['D',3,2]])x.execute({op:'free',id,params:{x:a,y:b}});return x;}
{
 const x=small(),goals=['perp A B A C','dist A B = 2 C D','col A B D'];
 // Arbitrary example might be overconstrained, just assert improvement and reporting.
 const constraints=goals.map(s=>C.parse(s,x)),before=C.assess(constraints,x);
 const plan=Pretty.optimize(x,{constraints,iterations:90,restarts:2});
 check(plan.constrained&&plan.feasibility.total===3,'constrained planning');
 Pretty.apply(x,plan,100);const after=C.assess(constraints,x);
 check(after.filter(d=>d.satisfied).length>=before.filter(d=>d.satisfied).length,'improve satisfied count');
 check(plan.frames.length===plan.steps+1,'valid keyframes');
}
{
 const x=small(),constraints=['col A B C','dist A B = 3','perp A B C D'].map(s=>C.parse(s,x));
 const plan=Pretty.optimize(x,{constraints,iterations:100,restarts:1});
 Pretty.apply(x,plan,100);
 const status=C.assess(constraints,x);
 check(status.every(x=>x.satisfied),'simple simultaneous constraints '+JSON.stringify(status));
 check(plan.feasibility.satisfied===constraints.length,'all feasible');
 check(plan.beautyEnd<=plan.beautyStart+1e-7,'beauty does not worsen');
 const compiled=Compiler.compile(x.snapshot());const extra=C.ag2(constraints,x,compiled);
 check(extra.predicates.some(p=>p.name==='coll'&&p.source.startsWith('USER-PRETTY:')),'AG2 col');
 check(extra.predicates.some(p=>p.name==='perp'),'AG2 perpendicular');
 const ag2=DDAR.prepare(x.snapshot(),{engine:'ag2',enumerate:false,userPretty:extra});
 check(ag2.compiled.predicates.some(p=>p.source.startsWith('USER-PRETTY:')),'injection into AG2');
 const ag1=DDAR.prepare(x.snapshot(),{engine:'ag1',enumerate:false,userPretty:extra});
 check(!ag1.compiled.predicates.some(p=>p.source.startsWith('USER-PRETTY:')),'AG1 unaffected');
}
{
 const constraints=['dist A B = 2 C D','ang A B C D = pi/2', 'A in c'];
 const cs=constraints.map(s=>C.parse(s,k));const compiled=Compiler.compile(k.snapshot());
 const result=C.ag2(cs,k,compiled);
 check(result.predicates.length===0,'never submit unfulfilled goals as axioms');
 check(result.limitations.length>=1,'limitations state explicitly');
}
console.log('PASS v5.3 — '+tested+' checks for condition syntax, solver, and DDAR boundary');
