// Rebuild the 36 v4.5 construction fixtures and exercise v4.6's preparation.
const fs=require('fs'), {Kernel}=require('../assets/kernel.js'), ddar=require('../assets/ddar.js');
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
const rows=[];
for(const [op,id,inputs,params={}] of operations){
 const k=new Kernel();const make=(op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
 for(const [id,x,y] of [['A',0,0],['B',4,0],['C',1,3],['D',-1,2],['O',0,.5],['R',2,.5],['P',4,4]])make('free',id,[],{x,y});
 make('line','l',['B','C']);make('line','m',['A','D']);make('circle','c',['O','R']);
 make(op,id,inputs,params);
 const snapshot=k.snapshot(),prepared=ddar.prepare(snapshot,{engine:'ag1',goals:[]});
 rows.push({op,ag1:prepared.ag1,virtualCount:prepared.compiled.stats.virtuals,limitations:prepared.compiled.limitations});
}
fs.writeFileSync(process.argv[2],JSON.stringify(rows));
console.log('Prepared',rows.length,'constructors; genericized:',rows.filter(x=>x.limitations.some(x=>x.includes('通用数值示意'))).map(x=>x.op));
