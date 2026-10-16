const fs=require('fs'),{Kernel}=require('../assets/kernel.js'),comp=require('../assets/ddar-compiler.js'),AG1=require('../assets/ddar-ag1.js');
const basic=()=>{const k=new Kernel();const make=(op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
 make('free','A',[],{x:0,y:0});make('free','B',[],{x:4,y:0});make('free','C',[],{x:1,y:3});make('free','D',[],{x:-1,y:2});make('free','O',[],{x:0,y:0.5});make('free','R',[],{x:2,y:.5});make('free','P',[],{x:4,y:4});
 make('line','l',['B','C']);make('line','m',['A','D']);make('circle','c',['O','R']);
 return [k,make];};
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
 ['tangentFrom','h',['P','c']],
 ['diameterCircle','c2',['A','B']],['circleBySegment','c2',['O','A','B']],['circleRadius','c2',['O'],{radius:2.25}],
 ['circumcircle','c2',['A','B','C']],['circleCenter','U',['c']],
 ['real','X',[],{x:3.6}],['polygon','poly',['A','B','C']]
];
const data=[];
for(const [op,id,inputs,params={}] of operations){try{
 const [k,make]=basic();make(op,id,inputs,params);const node=k.node(id);if(!node.valid){data.push({op,valid:false});continue;}
 const output=comp.compile(k.snapshot()),ag1=AG1.compile(k.snapshot(),output);
 data.push({op,valid:true,output,ag1,synthetic:output.stats.virtuals});
}catch(e){data.push({op,error:String(e.stack||e)})}}
fs.writeFileSync(process.argv[2],JSON.stringify(data));
console.log('valid',data.filter(x=>x.valid).length,'invalid',data.filter(x=>x.valid===false).map(x=>x.op),'exceptions',data.filter(x=>x.error).map(x=>x.op));
