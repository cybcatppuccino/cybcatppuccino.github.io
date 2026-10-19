/* GeomTool v8.1 — exact Q(i) complex-coordinate geometry on the local JS CAS.
 * Only rational constructions are represented. No numeric geometry is inserted
 * into symbolic expressions; coordinates are read solely for intersection branch matching.
 */
(function(root,factory){const api=factory(root.GeomNerdamerRational||(typeof require==='function'?require('./nerdamer-rational.js'):null));root.GeomComplexCAS=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(Q){
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
 const format=x=>Q.string(x,alias); // v8.1: store the factored result immediately, in construction order.
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
 // Expand the collinearity determinant only once: (q̄-p̄)z-(q-p)z̄+(q-p)p̄-(q̄-p̄)p=0.
 const lineBetween=(p,q)=>{
  const dz=sub(q.z,p.z),db=sub(q.bar,p.bar);
  return {kind:'line',...Q.normalizeLineCoefficients(db,neg(dz),sub(mul(dz,p.bar),mul(db,p.z)))};
 };
 function ll(l,m){const den=sub(mul(l.a,m.b),mul(m.a,l.b));if(isZero(den))throw new Unavailable('平行或重合直线');return pt(div(sub(mul(l.b,m.c),mul(m.b,l.c)),den),div(sub(mul(m.a,l.c),mul(l.a,m.c)),den));}
 const midpoint=(p,q)=>pt(div(add(p.z,q.z),two),div(add(p.bar,q.bar),two));
 function foot(p,l){return pt(div(sub(sub(mul(l.a,p.z),mul(l.b,p.bar)),l.c),mul(two,l.a)),div(sub(sub(mul(l.b,p.bar),mul(l.a,p.z)),l.c),mul(two,l.b)));}
 // |z-p|²=|z-q|² expanded directly: no midpoint or rotation is needed.
 function perpBisector(p,q){
  return {kind:'line',...Q.normalizeLineCoefficients(
   sub(q.bar,p.bar),sub(q.z,p.z),sub(mul(p.z,p.bar),mul(q.z,q.bar)))};
 }
 const circumcenter=(p,q,r)=>ll(perpBisector(p,q),perpBisector(p,r));
 // Expand the concyclicity determinant directly, with r as origin.  Unlike
 // intersecting perpendicular bisectors, this uses just ONE denominator.
 // For x=p-r, y=q-r: D=x*bar(y)-y*bar(x),
 // u=bar(x)*bar(y)*(y-x)/D, v=x*y*(bar(x)-bar(y))/D;
 // w=z-r obeys w*bar(w)+u*w+v*bar(w)=0.
 function circumcircle(p,q,r){
  const x=sub(p.z,r.z),y=sub(q.z,r.z),xb=sub(p.bar,r.bar),yb=sub(q.bar,r.bar);
  const det=sub(mul(x,yb),mul(y,xb));
  if(isZero(det))throw new Unavailable('三点共线，无法确定圆');
  const u=div(mul(mul(xb,yb),sub(y,x)),det);
  const v=div(mul(mul(x,y),sub(xb,yb)),det);
  return {kind:'circle',a:sub(u,r.bar),b:sub(v,r.z),c:sub(sub(mul(r.z,r.bar),mul(u,r.z)),mul(v,r.bar))};
 }
 function circle(o,r2){return {kind:'circle',a:neg(o.bar),b:neg(o.z),c:sub(mul(o.z,o.bar),r2)};}
 const circleCenter=c=>pt(neg(c.b),neg(c.a));
 function onCarrier(p,c){const sum=c.kind==='line'?plus(mul(c.a,p.z),mul(c.b,p.bar),c.c):plus(mul(p.z,p.bar),mul(c.a,p.z),mul(c.b,p.bar),c.c);return isZero(sum);}
 // For lc and cc, eliminate zbar to obtain the following quadratic.
 // lc: -A1*z²+(A2*B1-A1*B2-C1)*z+(B1*C2-B2*C1)=0.
 // cc: (A2-A1)*z²+(A2*B1-A1*B2-C1+C2)*z+(B1*C2-B2*C1)=0.
 // In both cases L=A*z+B*zbar+C is the line/radical axis and
 // O=z*zbar+u*z+v*zbar+w is one incident circle. Vieta gives
 // z1*z2=(v*C-w*B)/A and bar1*bar2=(u*C-w*A)/B.
 // Since z1 already lies on L and O, substitution of C=-A*z1-B*bar1
 // into the product and cancellation of z1 gives the equivalent exact
 // rational forms below. Crucially, cancel BEFORE constructing the large
 // (v*C-w*B)/(A*z1) CAS expression: otherwise the IMO2019P6 Q explodes.
 function second(p,c,d,alreadyKnown=false){
  if(!alreadyKnown&&(!onCarrier(p,c)||!onCarrier(p,d)))throw new Unavailable('已知点未能符号验证为两个载体的公共点');
  if(c.kind==='line'&&d.kind==='line')return ll(c,d);
  const line=c.kind==='line'?c:d.kind==='line'?d:null;
  const cir=line?(c.kind==='circle'?c:d):c;
  const A=line?line.a:sub(c.a,d.a);
  const B=line?line.b:sub(c.b,d.b);
  if(isZero(A)||isZero(B))throw new Unavailable('重合/同心圆或退化直线没有唯一的第二交点');
  // Vieta-product quotient after cancelling the KNOWN common root.
  // Unlike the unreduced product, this also works when the known z1=0.
  return pt(sub(div(mul(B,add(p.bar,cir.a)),A),cir.b),
            sub(div(mul(A,add(p.z,cir.b)),B),cir.a));
 }
 // A point incident by construction is an exact witness; checking every
 // earlier point against complicated circle equations is needlessly costly.
 const nodeById=new Map(nodes.map(n=>[n.id,n]));
 function knownOn(pointId,carrierId){
  const p=nodeById.get(pointId),c=nodeById.get(carrierId);
  if(!p||!c)return false;
  if(p.op==='intersection'&&p.inputs.includes(carrierId))return true;
  if(p.op==='secondIntersection'&&p.inputs.slice(1).includes(carrierId))return true;
  if(p.op==='onCircle'&&p.inputs[0]===carrierId)return true;
  if(p.op==='onLine'&&p.inputs[0]===carrierId)return true;
  switch(c.op){
  case 'line':case 'segment':case 'ray':case 'circumcircle':case 'diameterCircle':return c.inputs.includes(pointId);
  case 'circle':return c.inputs[1]===pointId;
  case 'parallel':case 'perpendicular':case 'parallel3':case 'perpendicular3':case 'tangent':return c.inputs[0]===pointId;
  case 'circleRadius':return carrierId==='c_unit'&&(pointId==='p_1'||pointId==='p_i');
  default:return false;
  }
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
  case 'onCircumcircle':out=onCircle(n,circumcircle(...a));break;
  case 'circleCenter':out=circleCenter(a[0]);break;
  case 'line':case 'segment':case 'ray':{
   // The chord through two known points of the same pair of circles is
   // their radical axis.  Reusing it avoids expanding the Vieta root again.
   const axisPoint=inp.find(id=>{const v=nodeById.get(id);return v?.op==='intersection'&&v.inputs.every(cid=>values.get(cid)?.kind==='circle');});
   const origin=axisPoint&&nodeById.get(axisPoint),other=inp.find(id=>id!==axisPoint);
   if(origin&&knownOn(other,origin.inputs[0])&&knownOn(other,origin.inputs[1])){
    const c=values.get(origin.inputs[0]),d=values.get(origin.inputs[1]);
    out={kind:'line',a:sub(c.a,d.a),b:sub(c.b,d.b),c:sub(c.c,d.c)};

   }else out=lineBetween(...a);
  }
   if(nid==='l_Re')out={kind:'line',a:ONE,b:neg(ONE),c:ZERO};
   if(nid==='l_Im')out={kind:'line',a:ONE,b:ONE,c:ZERO};break;
  case 'parallel':{const [p,l]=a;out={kind:'line',...Q.normalizeLineCoefficients(l.a,l.b,neg(add(mul(l.a,p.z),mul(l.b,p.bar))))};break;}
  case 'perpendicular':{const [p,l]=a,b=neg(l.b);out={kind:'line',...Q.normalizeLineCoefficients(l.a,b,neg(add(mul(l.a,p.z),mul(b,p.bar))))};break;}
  case 'parallel3':{const p=a[0],d=sub(a[2].z,a[1].z),db=sub(a[2].bar,a[1].bar);out={kind:'line',...Q.normalizeLineCoefficients(db,neg(d),sub(mul(d,p.bar),mul(db,p.z)))};break;}
  case 'perpendicular3':{const p=a[0],d=sub(a[2].z,a[1].z),db=sub(a[2].bar,a[1].bar);out={kind:'line',...Q.normalizeLineCoefficients(db,d,neg(add(mul(db,p.z),mul(d,p.bar))))};break;}
  case 'perpBisector':out=perpBisector(...a);break;
  case 'equalAngleLine':{const d=div(mul(sub(a[3].z,a[4].z),sub(a[2].z,a[1].z)),sub(a[0].z,a[1].z)),db=div(mul(sub(a[3].bar,a[4].bar),sub(a[2].bar,a[1].bar)),sub(a[0].bar,a[1].bar));out=lineThrough(a[4],d,db);break;}
  case 'tangent':{const c=a[1],q=a[0];if(!knownOn(inp[0],inp[1])&&!onCarrier(q,c))throw new Unavailable('切点不能符号验证在圆上');out=lineThrough(q,mul(I,add(q.z,c.b)),neg(mul(I,add(q.bar,c.a))));break;}
  case 'angleBisector':case 'outerAngleBisector':case 'tangentFrom':throw new Unavailable('该构造通常需要非有理平方根');
  case 'circle':out=circle(a[0],mul(sub(a[1].z,a[0].z),sub(a[1].bar,a[0].bar)));break;
  case 'circleRadius':{const r=nid==='c_unit'?ONE:parameterNumber(p.radius,'半径');out=circle(a[0],pow(r,2));break;}
  case 'circleBySegment':out=circle(a[0],mul(sub(a[2].z,a[1].z),sub(a[2].bar,a[1].bar)));break;
  case 'diameterCircle':{const c=midpoint(...a);out=circle(c,mul(sub(a[0].z,c.z),sub(a[0].bar,c.bar)));break;}
  case 'circumcircle':out=circumcircle(...a);break;
  case 'intersection':{
    if(a[0].kind==='line'&&a[1].kind==='line'){out=ll(...a);break;}
    const g=n.geometry||{},slots=g.slots||[];
    if(!g.point||!(slots.length===1||slots.length===2))throw new Unavailable('缺少交点分支匹配信息');
    const distance=(v,s)=>v&&s?(v.x-s.x)**2+(v.y-s.y)**2:Infinity;
    const tolerance=v=>1e-10*Math.max(1,v.x**2+v.y**2);
    let witness=null,witnessId=null,closest=-1;
    // First resolve incidence through the construction graph, without invoking CAS.
    for(let pass=0;pass<2&&!witness;pass++){
     for(const [id,v] of values){
      if(v.kind!=='point')continue;
      const pos=positions[id];if(!pos)continue;
      const slot=slots.length===1?0:distance(pos,slots[0])<=distance(pos,slots[1])?0:1;
      if(distance(pos,slots[slot])>tolerance(pos))continue;
      const constructionKnown=knownOn(id,inp[0])&&knownOn(id,inp[1]);
      if(pass===0&&!constructionKnown||pass===1&&constructionKnown)continue;
      if(!constructionKnown&&(!onCarrier(v,a[0])||!onCarrier(v,a[1])))continue;
      witness=v;witnessId=id;closest=slot;break;
     }
    }
    if(!witness)throw new Unavailable('线圆/双圆没有已构造且符号验证的公共交点，无法有理化');
    const q=second(witness,...a,true);
    out=slots.length===1||Number(p.branch||0)===closest?witness:q;
 
    break;
  }
  case 'secondIntersection':out=second(a[0],a[1],a[2],knownOn(inp[0],inp[1])&&knownOn(inp[0],inp[2]));break;
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
   Q.beginStep();
   out=process(n);values.set(n.id,out);

   if(out.kind==='point')statuses.push({id:n.id,kind:'point',ok:true,expr:format(out.z),conjugate:format(out.bar)});
   else if(['line','circle'].includes(out.kind))statuses.push({id:n.id,kind:out.kind,ok:true,coefficients:Object.fromEntries(['a','b','c'].map(k=>[k,format(out[k])]))});
   else statuses.push({id:n.id,kind:out.kind,ok:true});
  }catch(e){invalid++;statuses.push({id:n.id,kind:n.kind||n.op,ok:false,reason:e instanceof Unavailable?e.message:'符号化失败：'+String(e.message||e).slice(0,180)});}
  if(typeof config.onProgress==='function')config.onProgress(statuses.length,nodes.length,n.id);
  if(typeof config.onNode==='function')config.onNode(statusView(statuses[statuses.length-1]),statuses.length,nodes.length);
 }
 for(const n of nodes)processNode(n);
 const cachedGoals=new Map();
 
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
  return {points:all.filter(s=>POINT_OPS.has((nodes.find(n=>n.id===s.id)||{}).op)),carriers:all.filter(s=>s.ok&&(s.kind==='line'||s.kind==='circle')),statuses:all,total:nodes.length,failed:invalid,goals:goals.map(goalView),engine:'Nerdamer 2.0 精确多项式 · 逐步因式化 v8.1'};
 }
 async function evaluate(propositions=[],options={}){
  const goals=[];
  const formatter=options.formatExpression||((r)=>Q.string(r,alias));
  const current=()=>!options.isCurrent||options.isCurrent();
  // Geometry is complete and stepwise factored: publish every valid coordinate
  // BEFORE evaluating potentially expensive additional proposition identities.
  if(options.onReady)options.onReady(snapshot(goals));
  for(const raw of propositions){
   if(!current())return null;
   let goal=cachedGoals.get(raw);
   if(!goal){try{goal=evaluateGoal(raw);}catch(error){goal={text:String(raw),error:String(error.message||error)};}}
   goals.push(goal);cachedGoals.set(raw,goal);
   if(options.onGoalProgress)options.onGoalProgress({index:goals.length-1,text:goal.text,kind:goal.kind,verified:goal.verified,error:goal.error,items:goalView(goal).items});
  }
  for(let index=0;index<goals.length;index++){ 
   const goal=goals[index];
   for(let itemIndex=0;itemIndex<(goal.items||[]).length;itemIndex++){
    if(!current())return null;
    const item=goal.items[itemIndex];
    if(typeof item.result!=='string'){
     if(options.onFactorStart)options.onFactorStart(goal.text,'命题表达式 '+(itemIndex+1));
     item.result=await formatter(item.result,alias);
    }
    if(!current())return null;
    if(options.onGoalProgress)options.onGoalProgress({index,itemIndex,item:goalView(goal).items[itemIndex]});
   }
  }
  return snapshot(goals);
 }
 function syncGraph(changes={}){
  const symbolChanges=changes.symbolChanges||{};
  if(Object.keys(symbolChanges).length){
   const translate=str=>String(str).replace(/[A-Za-z_][A-Za-z0-9_]*/g,x=>symbolChanges[x]||x);
   for(const [internal,name] of aliases)if(symbolChanges[name])aliases.set(internal,symbolChanges[name]);
   for(const n of nodes){const symbol=n.params?.complexSymbol||n.id;if(symbolChanges[symbol]&&['free','real','onLine','onCircle','onCircumcircle'].includes(n.op))n.params={...n.params,complexSymbol:symbolChanges[symbol]};}
   for(const status of statuses){
    if(typeof status.expr==='string')status.expr=translate(status.expr);
    if(typeof status.conjugate==='string')status.conjugate=translate(status.conjugate);
    for(const k of ['a','b','c'])if(typeof status.coefficients?.[k]==='string')status.coefficients[k]=translate(status.coefficients[k]);
   }
   for(const goal of cachedGoals.values())for(const item of goal.items||[])if(typeof item.result==='string')item.result=translate(item.result);
  }
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
  if(added.length){for(const n of added){nodes.push(n);nodeById.set(n.id,n);processNode(n);}}
 }
 return {evaluate,syncGraph};
}

async function compute(graph,config={}){return prepare(graph,config).evaluate(config.propositions||[],config);}
return {compute,prepare,Unavailable};
});
