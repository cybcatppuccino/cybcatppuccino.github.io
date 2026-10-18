/* GeomTool v7.9 — exact Q(i) complex-coordinate geometry on the local JS CAS.
 * Only rational constructions are represented. No numeric geometry is inserted
 * into symbolic expressions; coordinates are read solely for intersection branch matching.
 */
(function(root,factory){const api=factory(root.GeomRationalCAS||(typeof require==='function'?require('./rational-cas.js'):null));root.GeomComplexCAS=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(Q){
'use strict';
if(!Q)throw Error('Missing exact rational algebra core');
const {add,sub,mul,div,neg,pow,eq,isZero,symbol,value,string:rawString,one:ONE,zero:ZERO,i:I}=Q;
const R=n=>value(n),two=R(2),three=R(3),flat=x=>[].concat(x);
class Unavailable extends Error{}
const POINT_OPS=new Set(['referencePoint','free','real','onLine','onCircle','onCircumcircle','circleCenter','intersection','secondIntersection','midpoint','foot','foot3','reflectPoint','reflectLine','translate','parallelogram','rotateLeft','rotateRight','rotate','homothety','similarPoint','centroid','circumcenter','orthocenter','complexSquare','complexNegativeProduct','incenter','excenter','equilateral']);
const plus=(...a)=>a.reduce(add,ZERO),times=(...a)=>a.reduce(mul,ONE);
const sub2=(a,b)=>sub(a,b);
function compute(graph,config={}){
 const nodes=graph?.nodes||[];if(!Array.isArray(nodes))throw Error('无效作图数据');
 Q.reset();
 const values=new Map(),statuses=[],aliases=new Map(),conjugates=new Map(),positions=graph.positions||{};
 let symbolSerial=0;
 let invalid=0;
 function newSymbol(id,label,namespace='point'){
  // Unique opaque ASCII keys also support user-visible Unicode object IDs.
  const key='gt_'+namespace+'_'+(symbolSerial++);
  if(!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(key))throw new Unavailable('无法符号化对象标识');
  const s=symbol(key);aliases.set(key,label);return s;
 }
 function symLabel(n){const name=String(n.params?.complexSymbol||n.id);return /^[a-zA-Z][a-zA-Z0-9_]*$/.test(name)?name:n.id;}
 const alias=s=>aliases.get(s)||s;
 // Factor and clear all rational coefficient denominators entirely inside Q(i).
 const format=x=>Q.factor(x,alias);
 function parameterNumber(input,label){
  const s=String(input);
  if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s))throw new Unavailable(label+'不是有理数');
  const [base,exponent='0']=s.toLowerCase().split('e');
  const exp=Number(exponent),digits=(base.replace(/^\+/, '').replace('.',''));
  if(!Number.isSafeInteger(exp))throw new Unavailable(label+'的科学记数指数无法精确表示');
  const dot=(base.split('.')[1]||'').length,shift=exp-dot;
  const n=BigInt(digits),a=shift>=0?n*10n**BigInt(shift):n;
  return shift>=0?value(a):div(value(a),value(10n**BigInt(-shift)));
 }
 const pt=(z,bar)=>({kind:'point',z,bar});
 function lineThrough(p,d,db){if(isZero(d)||isZero(db))throw new Unavailable('直线方向退化');return {kind:'line',a:db,b:neg(d),c:sub(mul(d,p.bar),mul(db,p.z))};}
 const lineBetween=(p,q)=>lineThrough(p,sub(q.z,p.z),sub(q.bar,p.bar));
 function ll(l,m){const den=sub(mul(l.a,m.b),mul(m.a,l.b));if(isZero(den))throw new Unavailable('平行或重合直线');return pt(div(sub(mul(l.b,m.c),mul(m.b,l.c)),den),div(sub(mul(m.a,l.c),mul(l.a,m.c)),den));}
 const midpoint=(p,q)=>pt(div(add(p.z,q.z),two),div(add(p.bar,q.bar),two));
 function foot(p,l){return pt(div(sub(sub(mul(l.a,p.z),mul(l.b,p.bar)),l.c),mul(two,l.a)),div(sub(sub(mul(l.b,p.bar),mul(l.a,p.z)),l.c),mul(two,l.b)));}
 function perpBisector(p,q){const m=midpoint(p,q),d=mul(I,sub(q.z,p.z)),db=neg(mul(I,sub(q.bar,p.bar)));return lineThrough(m,d,db);}
 const circumcenter=(p,q,r)=>ll(perpBisector(p,q),perpBisector(p,r));
 function circle(o,r2){return {kind:'circle',a:neg(o.bar),b:neg(o.z),c:sub(mul(o.z,o.bar),r2)};}
 const circleCenter=c=>pt(neg(c.b),neg(c.a));
 function onCarrier(p,c){const sum=c.kind==='line'?plus(mul(c.a,p.z),mul(c.b,p.bar),c.c):plus(mul(p.z,p.bar),mul(c.a,p.z),mul(c.b,p.bar),c.c);return isZero(sum);}
 function second(p,c,d){if(!onCarrier(p,c)||!onCarrier(p,d))throw new Unavailable('已知点未能符号验证为两个载体的公共点');if(c.kind==='line'&&d.kind==='line')return ll(c,d);
  if(c.kind==='line'||d.kind==='line'){const l=c.kind==='line'?c:d,cir=c.kind==='line'?d:c,h=foot(circleCenter(cir),l);return pt(sub(mul(two,h.z),p.z),sub(mul(two,h.bar),p.bar));}
  if(eq(c.b,d.b))throw new Unavailable('同心圆不产生唯一的第二交点');const axis=lineBetween(circleCenter(c),circleCenter(d)),h=foot(p,axis);return pt(sub(mul(two,h.z),p.z),sub(mul(two,h.bar),p.bar));
 }
 function free(n,real=false){const label=symLabel(n),z=newSymbol(n.id,label),bar=real?z:newSymbol(n.id,label+'\u0305','conjugate');conjugates.set(z,bar);conjugates.set(bar,z);return pt(z,bar);}
 function onLine(n,l){const z=newSymbol(n.id,symLabel(n));const bar=div(neg(add(mul(l.a,z),l.c)),l.b);conjugates.set(z,bar);return pt(z,bar);}
 function onCircle(n,c){const z=newSymbol(n.id,symLabel(n));const bar=div(neg(add(mul(c.a,z),c.c)),add(z,c.b));conjugates.set(z,bar);return pt(z,bar);}
 const addpoint=(p,q)=>pt(add(p.z,q.z),add(p.bar,q.bar));
 const scalePt=(p,k,kb=k)=>pt(mul(k,p.z),mul(kb,p.bar));
 function process(n){const nid=n.id,op=n.op,inp=n.inputs||[],p=n.params||{};
  if(inp.some(id=>!values.has(id)))throw new Unavailable('依赖点或载体尚不可用有理复坐标表示');
  const a=inp.map(id=>values.get(id));let out;
  switch(op){
  case 'referencePoint':out=nid==='p_0'?pt(ZERO,ZERO):nid==='p_1'?pt(ONE,ONE):nid==='p_i'?pt(I,neg(I)):(()=>{throw new Unavailable('未定义的坐标参考点')})();break;
  case 'free':case 'real':out=free(n,op==='real');break;
  case 'onLine':out=onLine(n,a[0]);break;
  case 'onCircle':out=onCircle(n,a[0]);break;
  case 'onCircumcircle':{const cen=circumcenter(...a);out=onCircle(n,circle(cen,mul(sub(a[0].z,cen.z),sub(a[0].bar,cen.bar))));break;}
  case 'circleCenter':out=circleCenter(a[0]);break;
  case 'line':case 'segment':case 'ray':out=lineBetween(...a);
   if(nid==='l_Re')out={kind:'line',a:ONE,b:neg(ONE),c:ZERO};
   if(nid==='l_Im')out={kind:'line',a:ONE,b:ONE,c:ZERO};break;
  case 'parallel':out=lineThrough(a[0],neg(a[1].b),a[1].a);break;
  case 'perpendicular':out=lineThrough(a[0],neg(mul(I,a[1].b)),neg(mul(I,a[1].a)));break;
  case 'parallel3':out=lineThrough(a[0],sub(a[2].z,a[1].z),sub(a[2].bar,a[1].bar));break;
  case 'perpendicular3':out=lineThrough(a[0],mul(I,sub(a[2].z,a[1].z)),neg(mul(I,sub(a[2].bar,a[1].bar))));break;
  case 'perpBisector':out=perpBisector(...a);break;
  case 'equalAngleLine':{const d=div(mul(sub(a[3].z,a[4].z),sub(a[2].z,a[1].z)),sub(a[0].z,a[1].z)),db=div(mul(sub(a[3].bar,a[4].bar),sub(a[2].bar,a[1].bar)),sub(a[0].bar,a[1].bar));out=lineThrough(a[4],d,db);break;}
  case 'tangent':{const c=a[1],q=a[0];if(!onCarrier(q,c))throw new Unavailable('切点不能符号验证在圆上');out=lineThrough(q,mul(I,add(q.z,c.b)),neg(mul(I,add(q.bar,c.a))));break;}
  case 'angleBisector':case 'outerAngleBisector':case 'tangentFrom':throw new Unavailable('该构造通常需要非有理平方根');
  case 'circle':out=circle(a[0],mul(sub(a[1].z,a[0].z),sub(a[1].bar,a[0].bar)));break;
  case 'circleRadius':{const r=nid==='c_unit'?ONE:parameterNumber(p.radius,'半径');out=circle(a[0],pow(r,2));break;}
  case 'circleBySegment':out=circle(a[0],mul(sub(a[2].z,a[1].z),sub(a[2].bar,a[1].bar)));break;
  case 'diameterCircle':{const c=midpoint(...a);out=circle(c,mul(sub(a[0].z,c.z),sub(a[0].bar,c.bar)));break;}
  case 'circumcircle':{const c=circumcenter(...a);out=circle(c,mul(sub(a[0].z,c.z),sub(a[0].bar,c.bar)));break;}
  case 'intersection':{
    if(a[0].kind==='line'&&a[1].kind==='line'){out=ll(...a);break;}
    let witness=null,wid=null;
    for(const [id,v] of values){if(v.kind!=='point')continue;if(onCarrier(v,a[0])&&onCarrier(v,a[1])){witness=v;wid=id;break;}}
    if(!witness)throw new Unavailable('线圆/双圆没有已构造且符号验证的公共交点，无法有理化');
    const q=second(witness,...a),pos=positions[wid],g=n.geometry||{},slots=g.slots||[];
    if(!pos||!g.point)throw new Unavailable('缺少交点分支匹配信息');
    if(slots.length===1)out=witness;
    else if(slots.length===2){const distance=s=>s?(pos.x-s.x)**2+(pos.y-s.y)**2:Infinity;const closest=distance(slots[0])<=distance(slots[1])?0:1;
     if(distance(slots[closest])>1e-10*Math.max(1,pos.x**2+pos.y**2))throw new Unavailable('已有公共点与几何交点分支不匹配');
     out=(Number(p.branch||0)===closest)?witness:q;
    }else throw new Unavailable('无法识别稳定的交点分支');
    break;
  }
  case 'secondIntersection':out=second(a[0],a[1],a[2]);break;
  case 'midpoint':out=midpoint(...a);break;
  case 'foot':case 'foot3':out=foot(a[0],op==='foot'?a[1]:lineBetween(a[1],a[2]));break;
  case 'reflectPoint':out=addpoint(scalePt(a[1],two),scalePt(a[0],neg(ONE)));break;
  case 'reflectLine':{const h=foot(...a);out=addpoint(scalePt(h,two),scalePt(a[0],neg(ONE)));break;}
  case 'translate':out=addpoint(a[0],addpoint(a[2],scalePt(a[1],neg(ONE))));break;
  case 'parallelogram':out=addpoint(a[1],addpoint(a[2],scalePt(a[0],neg(ONE))));break;
  case 'rotateLeft':{const v=addpoint(a[0],scalePt(a[1],neg(ONE)));out=addpoint(a[1],scalePt(v,I,neg(I)));break;}
  case 'rotateRight':{const v=addpoint(a[0],scalePt(a[1],neg(ONE)));out=addpoint(a[1],scalePt(v,neg(I),I));break;}
  case 'rotate':{const deg=Number(p.degrees||0),quadrant=Math.round(deg/90);if(!Number.isFinite(deg)||Math.abs(deg-90*quadrant)>1e-10)throw new Unavailable('非直角旋转通常不属于有理复坐标构造');
   const u=pow(I,((quadrant%4)+4)%4),ub=pow(neg(I),((quadrant%4)+4)%4);out=addpoint(a[1],scalePt(addpoint(a[0],scalePt(a[1],neg(ONE))),u,ub));break;}
  case 'homothety':{const k=parameterNumber(p.factor,'位似比');out=addpoint(a[1],scalePt(addpoint(a[0],scalePt(a[1],neg(ONE))),k));break;}
  case 'similarPoint':{const k=div(sub(a[4].z,a[3].z),sub(a[1].z,a[0].z)),kb=div(sub(a[4].bar,a[3].bar),sub(a[1].bar,a[0].bar));out=addpoint(a[3],scalePt(addpoint(a[2],scalePt(a[0],neg(ONE))),k,kb));break;}
  case 'centroid':out=scalePt(a.reduce(addpoint,pt(ZERO,ZERO)),div(ONE,three));break;
  case 'circumcenter':out=circumcenter(...a);break;
  case 'orthocenter':{const c=circumcenter(...a);out=addpoint(a.reduce(addpoint,pt(ZERO,ZERO)),scalePt(c,neg(two)));break;}
  case 'complexSquare':out=pt(pow(a[0].z,2),pow(a[0].bar,2));break;
  case 'complexNegativeProduct':out=pt(neg(mul(a[0].z,a[1].z)),neg(mul(a[0].bar,a[1].bar)));break;
  case 'incenter':case 'excenter':case 'equilateral':throw new Unavailable('一般情形需要非有理长度或根式');
  case 'polygon':out={kind:'polygon'};break;
  default:throw new Unavailable('当前构造尚无有理复坐标实现');
  }
  return out;
 }
 for(const n of nodes){let out;try{
   out=process(n);values.set(n.id,out);
   if(out.kind==='point')statuses.push({id:n.id,kind:'point',ok:true,expr:format(out.z),conjugate:format(out.bar)});
   else if(['line','circle'].includes(out.kind))statuses.push({id:n.id,kind:out.kind,ok:true,coefficients:Object.fromEntries(['a','b','c'].map(k=>[k,format(out[k])]))});
   else statuses.push({id:n.id,kind:out.kind,ok:true});
  }catch(e){invalid++;statuses.push({id:n.id,kind:n.kind||n.op,ok:false,reason:e instanceof Unavailable?e.message:'符号化失败：'+String(e.message||e).slice(0,180)});}
  if(typeof config.onProgress==='function'&&(statuses.length%3===0||statuses.length===nodes.length))config.onProgress(statuses.length,nodes.length,n.id);
 }
 return {points:statuses.filter(s=>POINT_OPS.has((nodes.find(n=>n.id===s.id)||{}).op)),carriers:statuses.filter(s=>s.ok&&(s.kind==='line'||s.kind==='circle')),statuses,total:nodes.length,failed:invalid,engine:'原生 JS 精确 Q(i) + 多元 GCD + Nerdamer factor 复验 v7.9'};
}
return {compute,Unavailable};
});
