const fs=require('node:fs'),{Kernel}=require('../assets/kernel.js'),cmp=require('../assets/ddar-compiler.js'),dd=require('../assets/ddar.js');require('../assets/ddar-goals.js');
function make(type){const k=new Kernel(),put=(op,id,inputs=[],params={})=>k.execute({op,id,inputs,params}),p=(id,x,y)=>put('free',id,[],{x,y});
 if(type==='mid'){p('A',0,0);p('B',4,0);p('C',0,3);put('midpoint','M',['A','B']);put('midpoint','N',['A','C']);}
 if(type==='free'){p('A',0,0);p('B',4,0);p('C',0,3);p('D',4,3);}
 if(type==='foot'){p('A',0,2);p('B',0,0);p('C',4,0);put('line','l',['B','C']);put('foot','H',['A','l']);}
 if(type==='circle'){p('O',0,0);p('A',2,0);put('circle','c',['O','A']);put('onCircle','B',['c'],{theta:Math.PI/3});put('onCircle','C',['c'],{theta:1.2});put('onCircle','D',['c'],{theta:2.1});}
 return k.snapshot();}
const goals={mid:[{kind:'para',args:['P:A|B','P:M|N']},{kind:'para',args:['P:B|C','P:M|N']},{kind:'cong',args:['A|M','M|B']}],free:[{kind:'cong',args:['A|B','C|D']},{kind:'para',args:['P:A|B','P:C|D']}],foot:[{kind:'perp',args:['P:A|H','P:B|C']},{kind:'coll',args:['H','B','C']}],circle:[{kind:'cong',args:['O|A','O|B']},{kind:'cyclic',args:['A','B','C','D']}]};
const cases=Object.keys(goals).map(name=>{const s=make(name);const prep=dd.prepare(s,{goal:null,enumerate:false});const queries=goals[name].map((g,i)=>{g.id=name+':'+i;return dd.transformGoal(g,prep.compiled)});return {name,compiled:prep.compiled,queries}});
fs.writeFileSync(process.argv[2]||'/mnt/data/v44_cases.json',JSON.stringify(cases,null,2));
