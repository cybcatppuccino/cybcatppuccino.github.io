const fs=require('fs'),{Kernel}=require('../assets/kernel.js'),c=require('../assets/ddar-compiler.js'),a=require('../assets/ddar-ag1.js'),d=require('../assets/ddar.js');
const k=new Kernel();for(const [id,x,y] of [['p_0',0,0],['p_1',1,0],['p_i',0,1]]) k._create({op:'referencePoint',id,params:{x,y},style:{system:true}});
k._create({op:'line',id:'l_Re',inputs:['p_0','p_1'],style:{system:true}});k._create({op:'line',id:'l_Im',inputs:['p_0','p_i'],style:{system:true}});k._create({op:'circleRadius',id:'c_unit',inputs:['p_0'],params:{radius:1},style:{system:true}});
const put=(op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
put('free','A',[],{x:2,y:3});put('free','B',[],{x:6,y:3});put('free','C',[],{x:3,y:7});
put('midpoint','M',['A','B']);put('midpoint','N',['A','C']);
const snap=k.snapshot(),compiled=c.compile(snap),ag1=a.compile(snap,compiled);const goal={id:'midline',kind:'para',args:['P:M|N','P:B|C']};const query=d.transformGoal(goal,compiled);query.ag1Goal=a.goal(goal,ag1);fs.writeFileSync(process.argv[2],JSON.stringify({compiled,ag1,queries:[query],mode:'goal',targetId:goal.id,maxLevel:6,levelTimeout:20},null,2));
console.log('query',query,'AG1 points',Object.keys(ag1.pointCoordinates).length,'AG2 points',compiled.points.length,'virtual',compiled.stats.virtuals, 'ag1 limitations',ag1.limitations);
