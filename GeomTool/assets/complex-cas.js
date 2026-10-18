/* GeomTool v8 — exact Q(i) complex-coordinate geometry on the local JS CAS.
 * Only rational constructions are represented. No numeric geometry is inserted
 * into symbolic expressions; coordinates are read solely for intersection branch matching.
 */
(function(root,factory){const api=factory(root.GeomRationalCAS||(typeof require==='function'?require('./rational-cas.js'):null));root.GeomComplexCAS=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(Q){
'use strict';
if(!Q)throw Error('Missing exact rational algebra core');
const {add,sub,subGoal,addGoal,mulGoal,mul,div,neg,pow,eq,isZero,symbol,value,string:rawString,one:ONE,zero:ZERO,i:I}=Q;
const R=n=>value(n),two=R(2),three=R(3),flat=x=>[].concat(x);
class Unavailable extends Error{}
const POINT_OPS=new Set(['referencePoint','free','real','onLine','onCircle','onCircumcircle','circleCenter','intersection','secondIntersection','midpoint','foot','foot3','reflectPoint','reflectLine','translate','parallelogram','rotateLeft','rotateRight','rotate','homothety','similarPoint','centroid','circumcenter','orthocenter','complexSquare','complexNegativeProduct','incenter','excenter','equilateral']);
const plus=(...a)=>a.reduce(add,ZERO),times=(...a)=>a.reduce(mul,ONE);
const sub2=(a,b)=>sub(a,b);
function prepare(graph,config={}){
 const nodes=graph?.nodes||[];if(!Array.isArray(nodes))throw Error('无效作图数据');
 Q.reset();
 const values=new Map(),statuses=[],aliases=new Map(),conjugates=new Map(),positions={...(graph.positions||{})},symbolOwners=new Map();
 let symbolSerial=0;
 let invalid=0;
 function newSymbol(id,label,namespace='point'){
  // Unique opaque ASCII keys also support user-visible Unicode object IDs.
  const key='gt_'+namespace+'_'+(symbolSerial++);
  if(!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(key))throw new Unavailable('无法符号化对象标识');
  const s=symbol(key);aliases.set(key,label);symbolOwners.set(key,{id,namespace});return s;
 }
 function symLabel(n){const name=String(n.params?.complexSymbol||n.id);return /^[a-zA-Z][a-zA-Z0-9_]*$/.test(name)?name:n.id;}
 const alias=s=>aliases.get(s)||s;
 // Factor and clear all rational coefficient denominators entirely inside Q(i).
 const format=x=>x; // Defer expensive factorization until symbolic geometry is complete.
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
 function lineThrough(p,d,db){if(isZero(d)||isZero(db))throw new Unavailable('直线方向退化');return {kind:'line',...Q.normalizeLineCoefficients(db,neg(d),sub(mul(d,p.bar),mul(db,p.z)))};}
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
 function free(n,real=false){const label=symLabel(n),z=newSymbol(n.id,label),bar=real?z:newSymbol(n.id,[...label].map(ch=>ch+'\u0305').join(''),'conjugate');conjugates.set(z,bar);conjugates.set(bar,z);return pt(z,bar);}
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

 // Translate only the listed AlphaGeometry-style predicates into exact
 // rational-complex identities. This is symbolic verification, not a numeric
 // substitute based on the current draggable coordinates.
 function evaluateGoal(raw){
  const tokens=String(raw||'').trim().split(/\s+/).filter(Boolean);
  const op=(tokens.shift()||'').toLowerCase();
  const arities={cong:[4],para:[4],perp:[4],eqangle:[8],eqratio:[8],coll:[3],cyclic:[4],onl:[2,3],onc:[2,4]};
  if(!Object.hasOwn(arities,op))throw Error('仅支持 cong / para / perp / eqangle / eqratio / coll / cyclic / onl / onc');
  if(!arities[op].includes(tokens.length))throw Error(op+' 的参数数目不正确：允许 '+arities[op].join(' 或 ')+' 个对象');
  const point=name=>{const p=values.get(name);if(!p||p.kind!=='point')throw Error('未找到可符号计算的点：'+name);return p;};
  const carrier=(name,kind)=>{const v=values.get(name);if(!v||v.kind!==kind)throw Error('未找到可符号计算的'+(kind==='line'?'直线':'圆')+'：'+name);return v;};
  const bar=name=>[...name].map(ch=>ch+'\u0305').join('');
  const diff=(i,j,k='z')=>subGoal(point(tokens[i])[k],point(tokens[j])[k]);
  const txt=(i,j,k='z')=>'('+ (k==='bar'?bar(tokens[i]):tokens[i])+'-'+(k==='bar'?bar(tokens[j]):tokens[j])+')';
  const norm2=(i,j)=>mulGoal(diff(i,j),diff(i,j,'bar'));
  const items=[];
  function record(label,expression,algebra,check=false){
   const zero=isZero(algebra);
   items.push({label,expression,result:format(algebra),zero:check?zero:null});
   return zero;
  }
  function addRatio(num,den,numText,denText){
   if(isZero(den))items.push({label:'复比',expression:numText+'/'+denText,result:'分母恒等于 0，复比未定义',zero:null});
   else record('复比','('+numText+')/('+denText+')',div(num,den));
  }
  let verdict;
  if(op==='cong'){
   const left=norm2(0,1),right=norm2(2,3);
   const lt=txt(0,1)+txt(0,1,'bar'),rt=txt(2,3)+txt(2,3,'bar');
   verdict=record('等长判定',lt+' - '+rt,subGoal(left,right),true);
   record('第一段模长平方',lt,left);record('第二段模长平方',rt,right);
  }else if(op==='para'||op==='perp'||op==='coll'){
   let indices=op==='coll'?[0,1,0,2]:[0,1,2,3];
   const [a,b,c,d]=indices;
   const u=diff(a,b),v=diff(c,d),ub=diff(a,b,'bar'),vb=diff(c,d,'bar');
   const ut=txt(a,b),vt=txt(c,d),ubt=txt(a,b,'bar'),vbt=txt(c,d,'bar');
   const left=mulGoal(u,vb),right=mulGoal(v,ub);
   const sign=op==='perp'?'+':'-';
   verdict=record(op==='perp'?'垂直判定':op==='coll'?'共线判定':'平行判定',
    ut+vbt+' '+sign+' '+vt+ubt,op==='perp'?addGoal(left,right):subGoal(left,right),true);
   addRatio(u,v,ut,vt);
  }else if(op==='eqangle'||op==='eqratio'){
   const [ab,cd,ef,gh]=[[0,1],[2,3],[4,5],[6,7]];
   const u=mulGoal(diff(...ab),diff(...gh)),v=mulGoal(diff(...cd),diff(...ef));
   const ub=mulGoal(diff(...ab,'bar'),diff(...gh,'bar')),
    vb=mulGoal(diff(...cd,'bar'),diff(...ef,'bar'));
   const ut=txt(...ab)+txt(...gh),vt=txt(...cd)+txt(...ef),
    ubt=txt(...ab,'bar')+txt(...gh,'bar'),vbt=txt(...cd,'bar')+txt(...ef,'bar');
   if(op==='eqangle'){
    verdict=record('等角判定',ut+vbt+' - '+vt+ubt,subGoal(mulGoal(u,vb),mulGoal(v,ub)),true);
    addRatio(u,v,ut,vt);
   }else{
    const left=mulGoal(u,ub),right=mulGoal(v,vb);
    verdict=record('比例模平方判定',ut+ubt+' - '+vt+vbt,subGoal(left,right),true);
    record('第一组模平方乘积',ut+ubt,left);
    record('第二组模平方乘积',vt+vbt,right);
    addRatio(left,right,ut+ubt,vt+vbt);
   }
  }else if(op==='cyclic'){
   const u=mulGoal(diff(0,2),diff(1,3)),v=mulGoal(diff(0,3),diff(1,2));
   const ub=mulGoal(diff(0,2,'bar'),diff(1,3,'bar')),
    vb=mulGoal(diff(0,3,'bar'),diff(1,2,'bar'));
   const ut=txt(0,2)+txt(1,3),vt=txt(0,3)+txt(1,2);
   verdict=record('共圆判定',ut+txt(0,3,'bar')+txt(1,2,'bar')+' - '+vt+txt(0,2,'bar')+txt(1,3,'bar'),subGoal(mulGoal(u,vb),mulGoal(v,ub)),true);
   addRatio(u,v,ut,vt);
  }else if(op==='onl'||op==='onc'){
   const z=point(tokens[0]);
   const l=op==='onl'?(tokens.length===2?carrier(tokens[1],'line'):lineBetween(point(tokens[1]),point(tokens[2]))):null;
   let c=null;
   if(op==='onc')c=tokens.length===2?carrier(tokens[1],'circle'):(()=>{const p=point(tokens[1]),q=point(tokens[2]),r=point(tokens[3]),o=circumcenter(p,q,r);return circle(o,mul(sub(p.z,o.z),sub(p.bar,o.bar)));})();
   const o=l||c,objectName=tokens.slice(1).join('');
   const expression=op==='onl'?'a_'+objectName+'·'+tokens[0]+' + b_'+objectName+'·'+bar(tokens[0])+' + c_'+objectName:
    tokens[0]+'·'+bar(tokens[0])+' + a_'+objectName+'·'+tokens[0]+' + b_'+objectName+'·'+bar(tokens[0])+' + c_'+objectName;
   const sumGoal=(...terms)=>terms.reduce(addGoal,ZERO);
   const value=op==='onl'?sumGoal(mulGoal(l.a,z.z),mulGoal(l.b,z.bar),l.c):sumGoal(mulGoal(z.z,z.bar),mulGoal(c.a,z.z),mulGoal(c.b,z.bar),c.c);
   verdict=record(op==='onl'?'点在线上判定':'点在圆上判定',expression,value,true);
  }
  return {text:String(raw).trim(),kind:op,items,verified:verdict};
 }
 function processNode(n){let out;try{
   out=process(n);values.set(n.id,out);
   if(out.kind==='point')statuses.push({id:n.id,kind:'point',ok:true,expr:format(out.z),conjugate:format(out.bar)});
   else if(['line','circle'].includes(out.kind))statuses.push({id:n.id,kind:out.kind,ok:true,coefficients:Object.fromEntries(['a','b','c'].map(k=>[k,format(out[k])]))});
   else statuses.push({id:n.id,kind:out.kind,ok:true});
  }catch(e){invalid++;statuses.push({id:n.id,kind:n.kind||n.op,ok:false,reason:e instanceof Unavailable?e.message:'符号化失败：'+String(e.message||e).slice(0,180)});}
  if(typeof config.onProgress==='function'&&(statuses.length%3===0||statuses.length===nodes.length))config.onProgress(statuses.length,nodes.length,n.id);
 }
 for(const n of nodes)processNode(n);
 const cachedGoals=new Map();
 let formattedStatuses=false;
 // Keep the exact Rat objects in the persistent geometry session. UI snapshots
 // stringify only unfinished values; they never replace the cached algebra.
 function statusView(status){
  if(!status.ok)return status;
  const view={...status};
  if(view.kind==='point'){
   view.expr=typeof view.expr==='string'?view.expr:rawString(view.expr,alias);
   view.conjugate=typeof view.conjugate==='string'?view.conjugate:rawString(view.conjugate,alias);
  }else if(view.coefficients)view.coefficients=Object.fromEntries(Object.entries(view.coefficients).map(([k,v])=>[k,typeof v==='string'?v:rawString(v,alias)]));
  return view;
 }
 function goalView(goal){return {...goal,items:(goal.items||[]).map(item=>({...item,result:typeof item.result==='string'?item.result:rawString(item.result,alias)}))};}
 function snapshot(goals){
  const all=statuses.map(statusView);
  return {points:all.filter(s=>POINT_OPS.has((nodes.find(n=>n.id===s.id)||{}).op)),carriers:all.filter(s=>s.ok&&(s.kind==='line'||s.kind==='circle')),statuses:all,total:nodes.length,failed:invalid,goals:goals.map(goalView),engine:'精确 Q(i) + Nerdamer 2.0 稀疏多项式主 GCD/factor v8'};
 }
 async function evaluate(propositions=[],options={}){
  const goals=[];
  const formatter=options.formatExpression||((r)=>Q.factor(r,alias));
  const current=()=>!options.isCurrent||options.isCurrent();
  // Compute and send exact propositions BEFORE optional factoring. In particular,
  // a slow factor must never withhold the conclusion or the point coordinates.
  for(const raw of propositions){
   if(!current())return null;
   let goal=cachedGoals.get(raw);
   if(!goal){try{goal=evaluateGoal(raw);}catch(error){goal={text:String(raw),error:String(error.message||error)};}}
   goals.push(goal);cachedGoals.set(raw,goal);
   if(options.onGoalProgress)options.onGoalProgress({index:goals.length-1,text:goal.text,kind:goal.kind,verified:goal.verified,error:goal.error,items:goalView(goal).items});
  }
  if(options.onReady)options.onReady(snapshot(goals));
  for(let index=0;index<goals.length;index++){
   const goal=goals[index];
   for(let itemIndex=0;itemIndex<(goal.items||[]).length;itemIndex++){
    if(!current())return null;
    const item=goal.items[itemIndex];
    if(typeof item.result!=='string')item.result=await formatter(item.result,alias);
    if(!current())return null;
    if(options.onGoalProgress)options.onGoalProgress({index,itemIndex,item:goalView(goal).items[itemIndex]});
   }
  }
  if(!formattedStatuses){
   for(const status of statuses){
    if(!current())return null;
    if(!status.ok)continue;
    if(status.kind==='point'){
     if(typeof status.expr!=='string')status.expr=await formatter(status.expr,alias);
     if(!current())return null;
     if(typeof status.conjugate!=='string')status.conjugate=await formatter(status.conjugate,alias);
    }else if(status.coefficients){
     const rawCoefficients={...status.coefficients};
     for(const k of ['a','b','c']){
      if(!current())return null;
      if(typeof status.coefficients[k]!=='string')status.coefficients[k]=await formatter(status.coefficients[k],alias);
     }
     // Projective line equations permit cancellation of a shared rational
     // coefficient. Run the expensive three-way GCD only AFTER factoring;
     // it has its own independent 10s watchdog and never blocks earlier data.
     if(status.kind==='line'&&Object.values(rawCoefficients).every(v=>typeof v!=='string')&&options.simplifyLine){
      const reduced=await options.simplifyLine(rawCoefficients,alias);
      if(reduced&&current())status.coefficients={...status.coefficients,...reduced};
     }
    }
    if(!current())return null;
    if(options.onStatusProgress)options.onStatusProgress(statusView(status));
   }
   formattedStatuses=true;
  }
  return snapshot(goals);
 }
 function syncGraph(changes={}){
  const renames=changes.renames||{};
  if(Object.keys(renames).length){
   for(const n of nodes){
    n.id=renames[n.id]||n.id;
    n.inputs=n.inputs.map(id=>renames[id]||id);
    if(n.params?.polygonParent)n.params.polygonParent=renames[n.params.polygonParent]||n.params.polygonParent;
   }
   for(const [oldId,newId] of Object.entries(renames)){
    if(values.has(oldId)){const exact=values.get(oldId);values.delete(oldId);values.set(newId,exact);}
   }
   for(const status of statuses)status.id=renames[status.id]||status.id;
   const translate=str=>String(str).replace(/[A-Za-z][A-Za-z0-9_]*/g,x=>renames[x]||x);
   for(const [name,owner] of symbolOwners){
    if(renames[owner.id]){
     const oldId=owner.id,newId=renames[oldId];owner.id=newId;
     const former=owner.namespace==='conjugate'?[...oldId].map(c=>c+'\u0305').join(''):oldId;
     if(aliases.get(name)===former)aliases.set(name,owner.namespace==='conjugate'?[...newId].map(c=>c+'\u0305').join(''):newId);
    }
   }
   for(const status of statuses){
    if(typeof status.expr==='string')status.expr=translate(status.expr);
    if(typeof status.conjugate==='string')status.conjugate=translate(status.conjugate);
    for(const key of ['a','b','c'])if(typeof status.coefficients?.[key]==='string')status.coefficients[key]=translate(status.coefficients[key]);
   }
   const renamed=[];
   for(const goal of cachedGoals.values()){
    goal.text=translate(goal.text);
    for(const item of goal.items||[]){item.expression=translate(item.expression);if(typeof item.result==='string')item.result=translate(item.result);}
    renamed.push(goal);
   }
   cachedGoals.clear();for(const goal of renamed)cachedGoals.set(goal.text,goal);
  }
  if(changes.positions)Object.assign(positions,changes.positions);
  const added=changes.appended||[];
  if(added.length){for(const n of added){nodes.push(n);processNode(n);}formattedStatuses=false;}
 }
 return {evaluate,syncGraph};
}

async function compute(graph,config={}){return prepare(graph,config).evaluate(config.propositions||[],config);}
return {compute,prepare,Unavailable};
});
