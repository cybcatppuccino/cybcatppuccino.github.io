const fs=require('fs'),{Kernel}=require('../assets/kernel.js'),ag2=require('../assets/ddar-compiler.js'),AG1=require('../assets/ddar-ag1.js'),DD=require('../assets/ddar.js');
const put=(k,op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
const p=(k,id,x,y)=>put(k,'free',id,[],{x,y});
const cases=[];
for(const type of ['mid','free','foot','circle','orthocenter']){
 const k=new Kernel();
 if(type==='mid'){p(k,'A',0,0);p(k,'B',4,0);p(k,'C',0,3);put(k,'midpoint','M',['A','B']);put(k,'midpoint','N',['A','C']);}
 if(type==='free'){p(k,'A',0,0);p(k,'B',4,0);p(k,'C',0,3);p(k,'D',4,3);}
 if(type==='foot'){p(k,'A',2,3);p(k,'B',0,0);p(k,'C',4,0);put(k,'line','l',['B','C']);put(k,'foot','H',['A','l']);}
 if(type==='circle'){p(k,'O',0,0);p(k,'A',2,0);put(k,'circle','c',['O','A']);put(k,'onCircle','B',['c'],{theta:Math.PI/3});put(k,'onCircle','C',['c'],{theta:1.2});}
 if(type==='orthocenter'){p(k,'A',0,0);p(k,'B',4,0);p(k,'C',1,3);put(k,'orthocenter','H',['A','B','C']);}
 const compiled=ag2.compile(k.snapshot()),ag1=AG1.compile(k.snapshot(),compiled);
 const wants={mid:[{kind:'cong',args:['A|M','M|B']},{kind:'para',args:['P:B|C','P:M|N']}],free:[{kind:'cong',args:['A|B','C|D']}],foot:[{kind:'perp',args:['P:A|H','P:B|C']}],circle:[{kind:'cong',args:['O|A','O|B']}],orthocenter:[{kind:'perp',args:['P:A|H','P:B|C']}]}[type];
 const queries=wants.map((g,i)=>({id:type+':'+i,ag1Goal:AG1.goal(g,ag1)}));
 cases.push({name:type,ag1,queries,maxLevel:7,levelTimeout:1});
}
fs.writeFileSync(process.argv[2],JSON.stringify(cases,null,2));
