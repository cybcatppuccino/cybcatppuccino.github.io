const fs=require('fs'),{Kernel}=require('../assets/kernel.js'),DD=require('../assets/ddar-compiler.js');
const cases=[];
function run(type){const k=new Kernel(),put=(op,id,inputs=[],params={})=>k.execute({op,id,inputs,params}),p=(id,x,y)=>put('free',id,[],{x,y});
 p('A',0,0);p('B',4,0);p('C',1,3);p('D',3,4);
 const safe=(op,id,ids,param={})=>{try{put(op,id,ids,param)}catch(e){console.log('SKIPPED',op,e.message)}};
 if(type==='incenter')safe('incenter','I',['A','B','C']);
 if(type==='excenter')safe('excenter','J',['A','B','C']);
 if(type==='angles'){safe('angleBisector','l',['A','B','C']);safe('outerAngleBisector','r',['A','B','C']);}
 if(type==='center'){safe('circumcenter','O',['A','B','C']);safe('circumcircle','c',['A','B','C']);safe('onCircle','E',['c'],{theta:1.2});}
 if(type==='reflect'){safe('line','l',['B','C']);safe('reflectLine','R',['D','l']);safe('reflectPoint','S',['D','B']);}
 if(type==='tangent'){safe('circle','c',['A','B']);safe('tangentFrom','t',['D','c'],{branch:0});}
 if(type==='rotate'){safe('rotate','R',['D','B'],{degrees:37});safe('rotateLeft','T',['C','B']);}
 if(type==='parallelogram'){safe('parallelogram','E',['A','B','C']);safe('centroid','G',['A','B','C']);}
 const compiled=DD.compile(k.snapshot());cases.push({type,compiled});}
for(const t of ['incenter','excenter','angles','center','reflect','tangent','rotate','parallelogram'])run(t);
fs.writeFileSync('/mnt/data/work/v44_audit_cases.json',JSON.stringify(cases));
