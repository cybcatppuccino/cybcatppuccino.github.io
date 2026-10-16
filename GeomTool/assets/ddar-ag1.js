/* v4.5: Graph-native AG1 construction compiler. Retains the original AG1
 * theorem engine, but injects exactly the construction's formal premises via
 * Graph.add_piece rather than downgrading unsupported constructions to free.
 * No numerical conjecture is admitted as a premise.
 */
(function(root,factory){const api=factory();root.GeomAG1Compiler=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const CAN=new Set(['coll','cong','para','perp','eqangle','cyclic','rconst','aconst']);
function compile(snapshot,ag2){
 const pointCoordinates={},nameMap={},nodes=[],representatives=[],limitations=[...ag2.limitations],aliased=[];
 // Omit disconnected virtual points. They cannot contribute to any kernel
 // premise or goal and may accidentally overlap another numeric witness.
 const required=new Set();
 for(const row of ag2.predicates)for(const n of row.points)required.add(n);
 for(const n of Object.values(ag2.lines||{}))for(const x of n)required.add(x);
 for(const n of ag2.points)if(!n.name.startsWith('@'))required.add(n.name);
 const parent=new Map(ag2.points.map(p=>[p.name,p.name]));
 const find=x=>{while(parent.get(x)!==x)x=parent.get(x);return x};
 for(const pred of ag2.predicates)if(pred.name==='overlap'&&pred.points.length===2){const [a,b]=pred.points;parent.set(find(a),find(b));}
 for(const item of ag2.points){
  if(!required.has(item.name))continue;
  // AG1's Graph.build_problem rejects coincident named points. Unify them as
  // identical point objects; AG2 handles the same equality through 'overlap'.
  const found=representatives.find(x=>find(x.id)===find(item.name));
  if(!found&&representatives.some(x=>Math.hypot(x.xy[0]-item.value[0],x.xy[1]-item.value[1])<1e-9))limitations.push('AG1 有数值重合但未有形式相等证明，保留为独立点: '+item.name);
  if(found){nameMap[item.name]=found.name;aliased.push([item.name,found.id]);continue;}
  const name='p'+representatives.length;representatives.push({id:item.name,name,xy:item.value});
  nameMap[item.name]=name;pointCoordinates[item.name]=item.value;
  nodes.push(name+'@'+item.value[0]+'_'+item.value[1]+' = free '+name);
 }
 const predicates=[],seen=new Set();
 for(const item of ag2.predicates){
  if(item.name==='overlap'||item.name==='cyclic_with_centers'){
   // 'overlap' is handled by alias unification; cyclic-with-center expands to
   // concrete radius congruences in the AG2 compiler.
   continue;
  }
  if(!CAN.has(item.name)){limitations.push('AG1 无法编码的前提 '+item.name);continue;}
  const mapped=item.points.map(x=>nameMap[x]);
  if(mapped.some(x=>!x)){limitations.push('AG1 点名缺失: '+item.source);continue;}
  if(['coll','cyclic'].includes(item.name)&&new Set(mapped).size<Math.min(mapped.length,3))continue;
  if(['cong','para','perp','eqangle','rconst','aconst'].includes(item.name)){
   let zero=false;for(let i=0;i<mapped.length;i+=2)if(mapped[i]===mapped[i+1])zero=true;
   if(zero)continue;
  }
  const code=JSON.stringify([item.name,mapped,item.constants]);if(seen.has(code))continue;seen.add(code);
  predicates.push({name:item.name,points:mapped,constants:item.constants||[],source:item.source});
 }
 return {problem:nodes.join('; '),predicates,nameMap,pointCoordinates,lines:ag2.lines,limitations,aliases:aliased};
}
function goal(goal,comp){
 if(!goal)return null;const map=comp.nameMap,args=goal.args||[],p=id=>map[id],seg=s=>s?.split('|'),line=s=>s?.startsWith('P:')?seg(s.slice(2)):s?.startsWith('L:')?comp.lines[s.slice(2)]:null;
 let points=null,kind=goal.kind;
 if(kind==='cong')points=[...seg(args[0]),...seg(args[1])];
 else if(kind==='para'||kind==='perp'){const u=line(args[0]),v=line(args[1]);if(u&&v)points=[...u,...v];}
 else if(kind==='eqangleLines'){const u=args.map(line);if(u.every(Boolean))points=u.flat();kind='eqangle';}
 else if(kind==='midp'||kind==='coll'||kind==='cyclic')points=args;
 else if(kind==='isos'){points=[args[0],args[1],args[0],args[2]];kind='cong';}
 else if(kind==='eqratio'){
  const a=seg(args[0]),b=seg(args[1]),f=Number(args[2]);
  if(a&&b&&Number.isFinite(f)&&f>0){let d=1;while(d<10000&&Math.abs(f*d-Math.round(f*d))>1e-9)d++;points=[...a,...b,Math.round(f*d)+'/'+d];kind='rconst';}
 }
 else if(kind==='angconst'){
  const u=line(args[0]),v=line(args[1]),deg=Number(args[2]);
  if(u&&v&&Number.isFinite(deg)){let d=1;while(d<360&&Math.abs(deg*d/180-Math.round(deg*d/180))>1e-9)d++;points=[...u,...v,Math.round(deg*d/180)+'pi/'+d];kind='aconst';}
 }
 else if(kind==='onl'){const u=line(args[1]);if(u){points=[args[0],...u];kind='coll';}}
 else if(kind==='onc')return null;
 if(!points||points.some(x=>!(x in map)&&!/^(\d+\/\d+|\-?\d+pi\/\d+)$/.test(x)))return null;
 return kind+' '+points.map(x=>map[x]||x).join(' ');
}
return {compile,goal};
});
