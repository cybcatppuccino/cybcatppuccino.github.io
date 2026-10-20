/* GeomTool v8.2 — parse one user-specified goal; never enumerate candidates. */
(function(root,factory){const api=factory();root.GeomDDARGoals=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const pair=(a,b)=>[a,b].sort().join('|'),lp=(a,b)=>'P:'+pair(a,b),seg=pair;
const eps=2e-8,near=(x,y)=>Math.abs(x-y)<eps*Math.max(1,Math.abs(x),Math.abs(y));
function parseGoal(src,snapshot){
 const pts=new Set((snapshot.nodes||[]).filter(n=>n.valid&&n.geom?.kind==='point').map(n=>n.id));
 const lines=new Set((snapshot.nodes||[]).filter(n=>n.valid&&n.geom?.kind==='line').map(n=>n.id));
 const circles=new Set((snapshot.nodes||[]).filter(n=>n.valid&&n.geom?.kind==='circle').map(n=>n.id));
 const text=String(src||'').trim();if(!text)throw Error('请输入待证命题');
 let tokens=text.split(/[\s,，]+/).filter(Boolean),verb=tokens.shift().toLowerCase();
 const compact=text.replace(/\s+/g,'');
 if(/^[A-Za-z]{2}=[A-Za-z]{2}$/.test(compact)&&[...compact.replace('=','')].every(p=>pts.has(p))){
  tokens=[...compact.replace('=','')];verb='cong';
 }
 const symbolic=compact.match(/^([A-Za-z]{2})([⟂∥])([A-Za-z]{2})$/);
 if(symbolic&&[...symbolic[1],...symbolic[3]].every(p=>pts.has(p))){
  verb=symbolic[2]==='⟂'?'perp':'para';tokens=[...symbolic[1],...symbolic[3]];
 }
 const anglePoints=compact.match(/^∠([A-Za-z]{3})=∠([A-Za-z]{3})$/);
 if(anglePoints&&[...anglePoints[1],...anglePoints[2]].every(p=>pts.has(p))){
  verb='eqangle';tokens=[...anglePoints[1],...anglePoints[2]];
 }
 const angleLines=compact.match(/^∠\(([A-Za-z]{2}),([A-Za-z]{2})\)=∠\(([A-Za-z]{2}),([A-Za-z]{2})\)$/);
 if(angleLines&&angleLines.slice(1).every(s=>[...s].every(p=>pts.has(p)))){
  verb='angle';tokens=angleLines.slice(1).flatMap(s=>[...s]);
 }
 const n={cong:4,parallel:4,para:4,perp:4,perpendicular:4,angle:8,eqangle:6,eqratio:5,angconst:5,angleconst:5,midpoint:3,midp:3,collinear:3,coll:3,cyclic:4,oncircle:2,online:2,onc:2,onl:2,isos:3};
 if(!Object.hasOwn(n,verb))throw Error('未知命题类型：支持 cong / para / perp / angle / eqangle / eqratio / midp / coll / cyclic / onc / onl / isos');
 if(!(verb==='eqangle'&&[6,8].includes(tokens.length))&&tokens.length!==n[verb])throw Error(verb+' 需要 '+(verb==='eqangle'?'6 或 8':n[verb])+' 个参数（点名以空格分隔）');
 const pointCount=['oncircle','online','onc','onl'].includes(verb)?1:['eqratio','angconst','angleconst'].includes(verb)?4:tokens.length;
 for(const p of tokens.slice(0,pointCount))if(!pts.has(p))throw Error('点不存在或当前无效：'+p);
 let kind=({parallel:'para',perpendicular:'perp',angle:'eqangleLines',eqangle:'eqangleLines',angleconst:'angconst',midpoint:'midp',collinear:'coll',oncircle:'onc',online:'onl'})[verb]||verb,args=[];
 if(['para','perp','cong'].includes(kind))args=kind==='cong'?[seg(tokens[0],tokens[1]),seg(tokens[2],tokens[3])].sort():[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3])].sort();
 else if(verb==='angle')args=[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3]),lp(tokens[4],tokens[5]),lp(tokens[6],tokens[7])];
 else if(verb==='eqangle')args=tokens.length===8?[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3]),lp(tokens[4],tokens[5]),lp(tokens[6],tokens[7])]:[lp(tokens[1],tokens[0]),lp(tokens[1],tokens[2]),lp(tokens[4],tokens[3]),lp(tokens[4],tokens[5])];
 else if(kind==='angconst'){
  const degrees=Number(tokens[4]);if(!Number.isFinite(degrees))throw Error('角度必须为有限数字');
  args=[lp(tokens[0],tokens[1]),lp(tokens[2],tokens[3]),String(((degrees%180)+180)%180)];
 }
 else if(kind==='eqratio'){
  const factor=Number(tokens[4]);if(!Number.isFinite(factor)||factor<=0)throw Error('线段比必须是正数');
  args=[seg(tokens[0],tokens[1]),seg(tokens[2],tokens[3]),String(factor)];
 }
 else if(kind==='midp')args=[tokens[0],...tokens.slice(1).sort()];
 else if(kind==='coll'||kind==='cyclic')args=tokens.sort();
 else if(kind==='onc'){if(!circles.has(tokens[1]))throw Error('圆不存在：'+tokens[1]);args=tokens;}
 else if(kind==='onl'){if(!lines.has(tokens[1]))throw Error('直线不存在：'+tokens[1]);args=[tokens[0],'L:'+tokens[1]];}
 else args=tokens;
 if(args.some((v,i)=>typeof v!=='string'||!v))throw Error('待证目标参数不完整');
 if((kind==='para'||kind==='perp'||kind==='cong')&&args[0]===args[1])throw Error('请选择两个不同的线段或直线');
 const id='manual:'+kind+':'+args.join('~');
 return {id,kind,args,text:'['+kind+'] '+tokens.join(' '),source:'手动设置待证命题'};
}
return {parseGoal};
});
