const {Kernel}=require('../assets/kernel.js'),cmp=require('../assets/ddar-compiler.js');const fs=require('fs');
const operations=[
 ['midpoint','M',['A','B']],['foot','H',['P','l']],['foot3','H',['P','B','C']],
 ['centroid','G',['A','B','C']],['circumcenter','U',['A','B','C']],
 ['orthocenter','H',['A','B','C']],['incenter','I',['A','B','C']],['excenter','J',['A','B','C']],
 ['parallelogram','X',['A','B','C']],['equilateral','E',['A','B']],
 ['rotateLeft','X',['P','O']],['rotateRight','X',['P','O']],['rotate','X',['P','O'],{degrees:35}],
 ['homothety','X',['P','O'],{factor:-2}],['reflectPoint','X',['P','O']],['reflectLine','X',['P','l']],
 ['translate','X',['P','A','B']],['onLine','X',['l'],{t:.4}],['onCircle','X',['c'],{theta:.45}],
 ['onCircumcircle','X',['A','B','C'],{theta:.8}],['intersection','X',['l','m']],
 ['parallel','h',['P','l']],['perpendicular','h',['P','l']],
 ['parallel3','h',['P','B','C']],['perpendicular3','h',['P','B','C']],
 ['perpBisector','h',['A','B']],['angleBisector','h',['A','B','C']],['outerAngleBisector','h',['A','B','C']],
 ['tangentFrom','h',['P','c']],['diameterCircle','c2',['A','B']],
 ['circleBySegment','c2',['O','A','B']],['circleRadius','c2',['O'],{radius:2.25}],
 ['circumcircle','c2',['A','B','C']],['circleCenter','U',['c']],['real','X',[],{x:3.6}],
 ['polygon','poly',['A','B','C']]
];
let arr=[];for(const [op,id,inputs,params={}] of operations){
 const k=new Kernel();for(const [x,a,b] of [['A',0,0],['B',4,0],['C',1,3],['D',-1,2],['O',0,.5],['R',2,.5],['P',4,4]])k.execute({op:'free',id:x,params:{x:a,y:b}});
 k.execute({op:'line',id:'l',inputs:['B','C']});k.execute({op:'line',id:'m',inputs:['A','D']});k.execute({op:'circle',id:'c',inputs:['O','R']});
 k.execute({op,id,inputs,params});const compiled=cmp.compile(k.snapshot());arr.push({op,compiled});}
if(process.argv[2])fs.writeFileSync(process.argv[2],JSON.stringify(arr));console.log('v51 compile:',arr.length,'constructors; skipped',arr.filter(x=>x.compiled.limitations.length).map(x=>[x.op,x.compiled.limitations]));
