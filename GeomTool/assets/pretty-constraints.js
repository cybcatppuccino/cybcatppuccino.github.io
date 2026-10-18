/* GeomTool v7.7 — typed geometric conditions, numeric residuals and AG2 hypotheses.
 * This is a declaration/optimization layer, NOT an automatic theorem prover.
 * Deliberately no eval(), no network, and no inferred mathematical axioms.
 */
(function(root,factory){const api=factory();root.GeomPrettyConstraints=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const aliases={col:'col',coll:'col',collinear:'col',cyc:'cyc',cyclic:'cyc',concyclic:'cyc',perp:'perp',perpendicular:'perp',para:'para',parallel:'para',tan:'tan',tangent:'tan',in:'in',sub:'in',subset:'in',cong:'cong',congruent:'cong',sim:'sim',similar:'sim',simtri:'sim',similartri:'sim'};
const err=s=>{throw Error(s)};
const tokenize=s=>{const v=String(s).replace(/π/g,'pi').replace(/√\s*(\d+)/g,'sqrt($1)').replace(/°/g,' deg');const re=/\s*([A-Za-z_@][A-Za-z_0-9:@]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[()+\-*/^=,;])/gy, out=[];let i=0;while(i<v.length){if(/^\s*$/.test(v.slice(i)))break;re.lastIndex=i;const m=re.exec(v);if(!m)err('不支持的符号：'+v.slice(i,i+12));out.push(m[1]);i=re.lastIndex;}return out;};
const isNumber=s=>/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(s||'');
const isName=s=>/^[A-Za-z_@][A-Za-z_0-9:@]*$/.test(s||'');
const normalize=s=>aliases[String(s||'').toLowerCase()]||String(s||'').toLowerCase();
function parser(raw,kernel){
 const ts=tokenize(raw);if(!ts.length)err('请输入几何条件');let i=0;
 const peek=()=>ts[i],take=()=>ts[i++],eat=s=>peek()===s?(i++,true):false,need=s=>{if(!eat(s))err('需要 '+s+'，当前为 '+(peek()||'末尾'))};
 const named=name=>{const n=kernel.node(name);if(!n||!n.valid)err('未知或无效的几何对象：'+name);return n};
 const point=()=>{const name=take();if(!isName(name))err('这里需要点名');const n=named(name);if(n.geom.kind!=='point')err(name+' 不是点');return n.id};
 const separated=()=>{while(peek()===',')take()};
 const compact=()=>{const name=peek();if(!/^[A-Z]{2,8}$/.test(name||'')||kernel.node(name))return null;
  const chars=[...name];return chars.every(a=>kernel.node(a)?.valid&&kernel.geom(a)?.kind==='point')?chars.map(a=>kernel.node(a).id):null;};
 const groupPoints=()=>{need('(');const p=[];while(peek()&&peek()!==')'){
  separated();if(peek()===')')break;const packed=compact();if(packed){take();p.push(...packed);}else p.push(point());separated();
 }need(')');return p};
 const pts=(count,min=count)=>{const p=[];if(peek()==='('){p.push(...groupPoints());}else{while(peek()&&peek()!=='='&&peek()!=='+'&&peek()!=='-'&&peek()!=='*'&&peek()!=='/'&&peek()!==')'&&peek()!==';'&&p.length<count){separated();if(!peek()||peek()===')')break;p.push(point());separated();}}if(p.length<min||p.length>count)err('这里需要 '+(min===count?count:min+'–'+count)+' 个点');return p};
 const carrier=()=>{
  if(peek()==='('){const p=groupPoints();if(p.length===2)return {kind:'line',points:p};if(p.length===3)return {kind:'circle',points:p};err('括号内应为两个点（直线）或三个点（圆）');}
  if(/^(?:c|circle)$/i.test(peek()||'')&&ts[i+1]==='('){
   const closing=ts.indexOf(')',i+2),items=ts.slice(i+2,closing).filter(z=>z!==',');
   if(items.length===3){take();const p=groupPoints();return {kind:'circle',points:p};}
  }
  const first=take();if(!isName(first))err('需要直线或圆的名称');const n=named(first);if(!['line','circle'].includes(n.geom.kind))err(first+' 不是直线或圆');return {kind:n.geom.kind,id:n.id};
 };
 const finish=()=>{if(i<ts.length)err('无法识别末尾：'+ts.slice(i).join(' '))};
 const build=(kind,props={})=>{finish();return {text:String(raw).trim(),kind,...props};};
 let head=normalize(peek());
 if(['eqangle','eqratio','aconst','rconst','midp','onl','onc'].includes(head)){
  take();
  if(['onl','onc'].includes(head)){
   const p=point();separated();const c=carrier();if(c.kind!==(head==='onl'?'line':'circle'))err('归属对象类型与 '+head+' 不符');return build('in',{point:p,carrier:c});
  }
  const count={eqangle:8,eqratio:8,aconst:4,rconst:4,midp:3}[head],p=[];
  if(peek()==='(')p.push(...groupPoints());else while(p.length<count&&peek()){
   separated();if(!peek())break;const compressed=compact();if(compressed&&p.length+compressed.length<=count){take();p.push(...compressed);}else p.push(point());
  }
  if(p.length!==count)err(head+' 需要 '+count+' 个点');
  const num=v=>({op:'num',value:v}),scalarD=(a,b)=>({op:'dist',points:[a,b]});
  if(head==='midp')return build('and',{conditions:[{kind:'col',points:p},{kind:'eq',expressions:[scalarD(p[0],p[1]),scalarD(p[0],p[2])]}]});
  if(head==='eqangle')return build('eq',{expressions:[{op:'ang',points:p.slice(0,4)},{op:'ang',points:p.slice(4)}]});
  if(head==='eqratio')return build('eq',{expressions:[{op:'/',args:[scalarD(p[0],p[1]),scalarD(p[2],p[3])]},{op:'/',args:[scalarD(p[4],p[5]),scalarD(p[6],p[7])]}]});
  const tokens=ts.slice(i).filter(x=>x!==',');
  if(!tokens.length)err('缺少常量');
  // Sandboxed numeric expression grammar for native aconst / rconst syntax.
  let z=0;const peekN=()=>tokens[z],popN=()=>tokens[z++];
  const numberAtom=()=>{
   const x=popN();if(x==='-')return -numberAtom();if(x==='+')return numberAtom();
   if(x==='('){const n=numberExpr();if(popN()!==')')err('常量缺少右括号');return n;}
   if(x?.toLowerCase()==='pi')return Math.PI;
   if(x?.toLowerCase()==='sqrt'){if(popN()!=='(')err('sqrt 后需要括号');const n=numberExpr();if(popN()!==')')err('sqrt 缺少右括号');return Math.sqrt(n);}
   if(/^sqrt\d+$/i.test(x||''))return Math.sqrt(Number(x.slice(4)));
   if(isNumber(x))return Number(x);
   err('未知常量：'+x);
  };
  const numberPow=()=>{let x=numberAtom();while(peekN()==='^'){popN();x=x**numberAtom();}return x;};
  const numberMul=()=>{let x=numberPow();while(['*','/'].includes(peekN())){const op=popN(),v=numberPow();x=op==='*'?x*v:x/v;}return x;};
  function numberExpr(){let x=numberMul();while(['+','-'].includes(peekN())){const op=popN(),v=numberMul();x=op==='+'?x+v:x-v;}return x;}
  let value=numberExpr();
  if(tokens[z]==='deg'){z++;value=value*Math.PI/180;}
  if(z!==tokens.length||!Number.isFinite(value))err('常量必须是有效实数算式');
  i=ts.length;
  return build('eq',{expressions:head==='aconst'?[{op:'ang',points:p},{...num(value),pi:tokens.includes('pi')||tokens.includes('deg')||Math.abs(value)<=2*Math.PI}]:[{op:'/',args:[scalarD(p[0],p[1]),scalarD(p[2],p[3])]},num(value)]});
 }
 if(head==='col'||head==='cyc'){
  take();let p=[];if(peek()==='(')p=groupPoints();else while(peek()){separated();if(peek()){const packed=compact();if(packed){take();p.push(...packed);}else p.push(point());}}
  const min=head==='col'?3:4;if(p.length<min)err(head+' 至少需要 '+min+' 个点');if(new Set(p).size!==p.length)err('该条件中的点必须各不相同');return build(head,{points:p});
 }
 if(head==='cong'||head==='sim'){
  take();let p=[];if(peek()==='('){p.push(...groupPoints());separated();if(peek()==='(')p.push(...groupPoints());else while(peek()){separated();if(peek())p.push(point());}}else while(peek()){separated();if(peek()){const packed=compact();if(packed){take();p.push(...packed);}else p.push(point());}}
  if(head==='cong'&&p.length===4)return build('eq',{expressions:[{op:'dist',points:p.slice(0,2)},{op:'dist',points:p.slice(2)}]});
  if(p.length!==6)err(head+' 需要两组三角形顶点，共 6 点');if(new Set(p.slice(0,3)).size!==3||new Set(p.slice(3)).size!==3)err('三角形不能有重复顶点');return build(head,{points:p});
 }
 if(head==='perp'||head==='para'||head==='tan'){
  take();let a,b;if(peek()==='('){ // distinguish tangent(c1,c2) and grouped lines (A B)
   const mark=i;try{need('(');a=carrier();separated();b=carrier();need(')');finish();return {text:String(raw).trim(),kind:head,carriers:[a,b]};}catch(e){i=mark;}
  }
  // Four bare points denote two lines, not four independent carriers.
  if(ts.slice(i).filter(t=>isName(t)).length===4&&ts.slice(i).every(t=>isName(t)||t===',')){
   const p=[];while(peek()){separated();if(peek())p.push(point());}return build(head,{carriers:[{kind:'line',points:p.slice(0,2)},{kind:'line',points:p.slice(2)}]});
  }
  a=carrier();separated();b=carrier();if(head==='tan'&&a.kind==='line'&&b.kind==='line')err('两条直线不定义相切，请使用 para');return build(head,{carriers:[a,b]});
 }
 // "A in c", "sub A c", "subset(A, l)", "subset A c(B C D)"
 if(['sub','subset'].includes(String(peek()).toLowerCase())){
  take();const outer=eat('(');const p=point();separated();const c=carrier();if(outer)need(')');return build('in',{point:p,carrier:c});
 }
 if(isName(peek())&&normalize(ts[i+1])==='in'){
  const p=point();take();return build('in',{point:p,carrier:carrier()});
 }
 // Arithmetic expression parser; all variable references are geometric quantities.
 const isSegmentStart=()=>isName(peek())&&kernel.node(peek())?.geom?.kind==='point';
 const pointsExact=(min,max)=>{
  if(peek()==='('){const p=groupPoints();if(p.length<min||p.length>max)err('需要 '+min+'–'+max+' 个点');return p;}
  const p=[];
  if(compact()&&compact().length>=min&&compact().length<=max){const found=compact();take();return found;}
  while(p.length<max&&isSegmentStart()){p.push(point());separated();}
  if(p.length<min)err('缺少角/距离/面积的点参数');return p;
 };
 const ast=(op,...args)=>({op,args});
 const atom=()=>{
  const v=peek();if(!v)err('表达式不完整');
  if(eat('+'))return atom();if(eat('-'))return ast('neg',atom());
  if(isNumber(v)){take();const degree=eat('deg');return {op:'num',value:Number(v)*(degree?Math.PI/180:1),pi:degree};}
  const kw=String(v).toLowerCase();
  if(kw==='pi'){take();return {op:'num',value:Math.PI,pi:true};}
  if(kw==='deg')err('度数单位应紧跟数值：例如 90deg 或 90°');
  if(/^sqrt\d+(?:_\d+)?$/i.test(v)){take();return ast('sqrt',{op:'num',value:Number(v.slice(4).replace('_','.'))});}
  if(kw==='sqrt'){take();if(eat('(')){const q=expr(0);need(')');return ast('sqrt',q);}return ast('sqrt',atom());}
  if(['dist','distance','d','len','length'].includes(kw)&&!(v==='D'&&isSegmentStart())){take();return {op:'dist',points:pointsExact(2,2)};}
  if(['ang','angle'].includes(kw)){take();return {op:'ang',points:pointsExact(3,4)};}
  if(['s','area'].includes(kw)&&!(v==='S'&&ts[i+1]!=='('&&isSegmentStart())){take();return {op:'area',points:pointsExact(3,3)};}
  if(v==='('){
   // Treat (A B) as a segment scalar and (A B C) as a group only for geometry statements.
   if(kernel.node(ts[i+1])?.geom?.kind==='point'){
    const mark=i;try{const p=groupPoints();if(p.length===2)return {op:'dist',points:p};}catch(e){}i=mark;
   }
   take();const out=expr(0);need(')');return out;
  }
  if(isSegmentStart())return {op:'dist',points:pointsExact(2,2)};
  err('未知表达式项目：'+v);
 };
 const prec={'+':1,'-':1,'*':2,'/':2,'^':4};
 function expr(min){let lhs=atom();while(peek()){
   let symbol=peek(),implicit=false;
   if(!(symbol in prec)){
    if(isNumber(symbol)||symbol==='('||isSegmentStart()||/^(sqrt\d*|sqrt|pi|ang|angle|dist|distance|d|area|s)$/i.test(symbol)){symbol='*';implicit=true;}
    else break;
   }
   const p=prec[symbol];if(p<min)break;if(!implicit)take();const rhs=expr(p+(symbol==='^'?0:1));lhs=ast(symbol,lhs,rhs);
  }return lhs;}
  const equations=[expr(0)];while(eat('='))equations.push(expr(0));if(equations.length<2)err('需要等式（例如 dist A B = 2 C D）');
  // A homogeneous geometric quantity carries units (length / area / angle).
  // Bare constants may declare an absolute numerical value in the units of
  // the other side. Mixed addition such as AB + area(A,B,C) is forbidden.
  const dimension=e=>{
   if(e.op==='num')return 0;
   if(e.op==='dist')return 1;
   if(e.op==='area')return 2;
   if(e.op==='ang')return 100;
   if(e.op==='neg')return dimension(e.args[0]);
   if(['+','-'].includes(e.op)){
    const [a,b]=e.args.map(dimension);if(a!==b&&a!==0&&b!==0)err('加减法两边必须是相同量纲（距离、面积或角度）');return a||b;
   }
   if(e.op==='*'||e.op==='/'){
    const [a,b]=e.args.map(dimension);if((a>=100&&b!==0)||(b>=100&&a!==0))err('角度只能乘除无量纲的数');
    if(e.op==='/'&&b===100)err('不能以角度作为分母');return e.op==='*'?a+b:a-b;
   }
   if(e.op==='sqrt'){
    const d=dimension(e.args[0]);
    if(!d){const pureNumeric=q=>q.op==='num'||(q.args?.length&&q.args.every(pureNumeric));
     if(pureNumeric(e.args[0])&&!(scalar(e.args[0],{geom:()=>null})>=0))err('sqrt 的常数参数必须非负');}
    return d/2;
   }
   if(e.op==='^'){
    const pureNumeric=q=>q.op==='num'||(q.args?.length&&q.args.every(pureNumeric));
    if(!pureNumeric(e.args[1]))err('幂指数必须是常数算式，不能是几何量或比例');
    const exponent=scalar(e.args[1],{geom:()=>null});
    if(!Number.isFinite(exponent)||Math.abs(exponent)>30)err('幂指数必须是有限且合理的数值');
    return dimension(e.args[0])*exponent;
   }
   return 0;
  };
  const dimensions=equations.map(dimension);
  const fixed=dimensions.filter(d=>d!==0);
  if(fixed.length&&fixed.some(d=>d!==fixed[0]))err('等式两边的量纲不同：请使用同类几何量');
  return build('eq',{expressions:equations});
}
const cross=(a,b)=>a.x*b.y-a.y*b.x,dot=(a,b)=>a.x*b.x+a.y*b.y,vec=(a,b)=>({x:b.x-a.x,y:b.y-a.y}),norm=a=>Math.hypot(a.x,a.y),dist=(a,b)=>norm(vec(a,b)),clamp=(v,l,h)=>Math.min(h,Math.max(l,v));
function scalar(expr,k){
 const a=expr.args||[];switch(expr.op){
  case 'num':return expr.value;
  case 'dist':return dist(...expr.points.map(p=>k.geom(p)));
  case 'area':{const [p,q,r]=expr.points.map(x=>k.geom(x));return .5*Math.abs(cross(vec(p,q),vec(p,r)));}
  case 'ang':{const ps=expr.points.map(x=>k.geom(x));const [u,v]=ps.length===3?[vec(ps[1],ps[0]),vec(ps[1],ps[2])]:[vec(ps[0],ps[1]),vec(ps[2],ps[3])];const x=Math.atan2(cross(u,v),dot(u,v));return ps.length===3?Math.abs(x):x;}
  case 'neg':return -scalar(a[0],k);
  case '+':return scalar(a[0],k)+scalar(a[1],k);
  case '-':return scalar(a[0],k)-scalar(a[1],k);
  case '*':return scalar(a[0],k)*scalar(a[1],k);
  case '/':return scalar(a[0],k)/scalar(a[1],k);
  case '^':return scalar(a[0],k)**scalar(a[1],k);
  case 'sqrt':return Math.sqrt(scalar(a[0],k));
 }return NaN;
}
function carrierGeom(c,k){
 if(c.id)return k.geom(c.id);
 const pts=c.points.map(x=>k.geom(x));if(!pts.every(Boolean))return null;
 if(c.kind==='line')return {kind:'line',p:pts[0],d:vec(pts[0],pts[1])};
 const [a,b,d]=pts,ab=vec(a,b),ac=vec(a,d),det=2*cross(ab,ac);if(Math.abs(det)<1e-12)return null;
 const ab2=dot(ab,ab),ac2=dot(ac,ac),center={x:a.x+(ac.y*ab2-ab.y*ac2)/det,y:a.y+(ab.x*ac2-ac.x*ab2)/det};return {kind:'circle',c:center,r:dist(center,a)};
}
function cv(c,k){const g=carrierGeom(c,k);if(!g)throw Error('直线或圆退化');return g;}
const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
function residuals(c,k){
 try{
  if(c.kind==='and')return c.conditions.flatMap(q=>residuals(q,k));
  if(c.kind==='eq'){
   return c.expressions.slice(1).map((q,i)=>{
    const left=c.expressions[i],x=scalar(left,k),y=scalar(q,k);
    // A bare angle constant 90/180/45 is in degrees, while pi/2 is radians.
    const hasAng=v=>v.op==='ang'||(v.args||[]).some(hasAng);
    const hasPi=v=>v.pi||(v.args||[]).some(hasPi);
    let a=x,b=y;if(hasAng(left)&&!hasAng(q)&&!hasPi(q)&&Math.abs(y)>2*Math.PI+1e-8)b=y*Math.PI/180;
    if(hasAng(q)&&!hasAng(left)&&!hasPi(left)&&Math.abs(x)>2*Math.PI+1e-8)a=x*Math.PI/180;
    const angle=hasAng(left)||hasAng(q);
    const value=angle?wrap(a-b):(a-b)/Math.max(1,Math.abs(a),Math.abs(b));
    return value;
   });
  }
  if(c.kind==='col'){
   const ps=c.points.map(p=>k.geom(p)),a=ps[0],b=ps[1],v=vec(a,b),len=norm(v);
   if(len<1e-9)return [1,...ps.slice(3).map(()=>1)];
   return ps.slice(2).map(p=>cross(v,vec(a,p))/(len*Math.max(1,dist(a,p))));
  }
  if(c.kind==='cyc'){
   // Translation-invariant, division-free concyclicity determinant. Unlike
   // circumcenter subtraction this stays smooth at near-collinear triples.
   const ps=c.points.map(id=>k.geom(id)),a=ps[0],b=ps[1],d=ps[2];
   const ab=vec(a,b),ad=vec(a,d),length=Math.max(1,...ps.map(v=>dist(a,v)));
   const area=cross(ab,ad)/(length*length);
   const degeneracy=Math.max(0,1e-5-Math.abs(area));
   return [degeneracy,...ps.slice(3).map(p=>{
    const ap=vec(a,p),n=cross(ab,ad)*dot(ap,ap)-cross(ap,ad)*dot(ab,ab)-cross(ab,ap)*dot(ad,ad);
    return n/(length**4);
   })];
  }
  if(c.kind==='cong'||c.kind==='sim'){
   const p=c.points.map(s=>k.geom(s)),sides=t=>[dist(t[0],t[1]),dist(t[1],t[2]),dist(t[2],t[0])],a=sides(p.slice(0,3)),b=sides(p.slice(3));
   const bad=t=>Math.max(0,1e-7-Math.abs(cross(vec(t[0],t[1]),vec(t[0],t[2])))/Math.max(1,...sides(t))**2);
   const nondegenerate=[bad(p.slice(0,3)),bad(p.slice(3))];
   if(c.kind==='cong')return [...a.map((x,i)=>(x-b[i])/Math.max(1,x,b[i])),...nondegenerate];
   return [...nondegenerate,...[1,2].map(i=>(a[i]*b[0]-a[0]*b[i])/Math.max(1,a[i]*b[0],a[0]*b[i]))];
  }
  if(c.kind==='in'){
   const p=k.geom(c.point),g=cv(c.carrier,k);
   if(g.kind==='circle')return [(dist(p,g.c)-g.r)/Math.max(1,g.r)];
   return [cross(g.d,vec(g.p,p))/Math.max(1e-9,norm(g.d),dist(g.p,p))];
  }
  if(['perp','para','tan'].includes(c.kind)){
   const [a,b]=c.carriers.map(z=>cv(z,k));
   if(a.kind==='line'&&b.kind==='line'){
    const denom=norm(a.d)*norm(b.d);if(denom<1e-10)return [1e6];
    if(c.kind==='tan')throw Error('两条直线不能互相相切');
    return [(c.kind==='perp'?dot(a.d,b.d):cross(a.d,b.d))/denom];
   }
   if(a.kind==='circle'&&b.kind==='circle'){
    const d=dist(a.c,b.c);
    if(c.kind==='perp')return [(d*d-a.r*a.r-b.r*b.r)/Math.max(1,d*d,a.r*a.r+b.r*b.r)];
    if(d<1e-9)return [1]; // Coincident circles are not tangent.
    return [Math.min(Math.abs(d-(a.r+b.r)),Math.abs(d-Math.abs(a.r-b.r)))/Math.max(1,a.r+b.r)];
   }
   const l=a.kind==='line'?a:b,o=a.kind==='circle'?a:b,ln=norm(l.d);
   if(ln<1e-10)return [1e6];const separation=cross(l.d,vec(l.p,o.c))/ln;
   if(c.kind==='perp')return [separation/Math.max(1,o.r)];
   return [(Math.abs(separation)-o.r)/Math.max(1,o.r)];
  }
 }catch(e){return [1e6];}return [1e6];
}
function assess(conditions,k,tolerance=2e-10){return (conditions||[]).map(c=>{
 const rs=residuals(c,k),error=Math.max(0,...rs.map(x=>Number.isFinite(x)?Math.abs(x):1e6));return {text:c.text,kind:c.kind,error,satisfied:error<=tolerance,residuals:rs};
});}
function validate(raw,k){const c=parser(raw,k);const sample=residuals(c,k);if(sample.some(x=>!Number.isFinite(x)))err('初始图形退化，无法计算该条件');return c;}
// Only supported declarative hypotheses become AG2 predicates. Unrepresentable
// conditions remain visible as limitations; no numerical coincidence is silently
// promoted to an axiom. Every emitted hypothesis is accuracy-checked first.
function ag2(conditions,k,compiled,tolerance=2e-10){
 const preds=[],notes=[],seen=new Set((compiled.predicates||[]).map(p=>JSON.stringify([p.name,p.points,p.constants||[]])));
 const lookup=key=>compiled.lines[key]||null;
 const ln=c=>c.kind==='line'?(c.points||lookup(c.id)):null;
 const push=(name,points,constants,source)=>{if(!points||points.some(x=>!x)||points.some(x=>!compiled.points.some(p=>p.name===x)))return false;
  const key=JSON.stringify([name,points,constants||[]]);if(!seen.has(key)){seen.add(key);preds.push({name,points,constants:constants||[],source});}return true;};
 const numeric=e=>{
  if(!e||['dist','ang','area'].includes(e.op))return undefined;
  if(e.op==='num')return e.value;
  const a=(e.args||[]).map(numeric);if(a.some(x=>!Number.isFinite(x)))return undefined;
  const v=e.op==='neg'?-a[0]:e.op==='sqrt'?Math.sqrt(a[0]):e.op==='+'?a[0]+a[1]:e.op==='-'?a[0]-a[1]:e.op==='*'?a[0]*a[1]:e.op==='/'?a[0]/a[1]:e.op==='^'?a[0]**a[1]:NaN;
  return Number.isFinite(v)?v:undefined;
 };
 const rational=v=>{
  if(!Number.isFinite(v))return false;
  for(let q=1;q<=3600;q++)if(Math.abs(v-Math.round(v*q)/q)<1e-10)return true;
  return false;
 };
 const anglePoints=e=>{
  if(e?.op!=='ang')return null;
  if(e.points.length===4)return e.points;
  const [a,b,c]=e.points,[pa,pb,pc]=[a,b,c].map(id=>k.geom(id));
  return cross(vec(pb,pa),vec(pb,pc))<0?[b,c,b,a]:[b,a,b,c];
 };
 const seg=e=>e?.op==='dist'?e.points:null;
 const convert=(c,source)=>{
  const p=c.points||[];
  if(c.kind==='and')return c.conditions.map(q=>convert(q,source)).every(Boolean);
  if(c.kind==='col')return p.slice(2).every(x=>push('coll',[p[0],p[1],x],[],source));
  if(c.kind==='cyc')return p.length===4?push('cyclic',p,[],source):p.slice(3).every((x,i)=>push('cyclic',[p[0],p[1],p[2],x],[],source));
  if(['para','perp'].includes(c.kind)){
   const ab=ln(c.carriers[0]),cd=ln(c.carriers[1]);if(ab&&cd)return push(c.kind,[...ab,...cd],[],source);
   if(c.kind==='perp'){const line=ab||cd, circle=c.carriers[ab?1:0],center=circle.id&&compiled.circles[circle.id]?.center;
    if(line&&center)return push('coll',[center,...line],[],source);}
   return false;
  }
  if(c.kind==='in'){
   const ab=ln(c.carrier);if(ab)return push('coll',[c.point,...ab],[],source);
   const circle=c.carrier.id&&compiled.circles[c.carrier.id];if(!circle)return false;
   const others=circle.points.filter(x=>x!==c.point);if(circle.center&&others.length)return push('cong',[circle.center,c.point,circle.center,others[0]],[],source);
   if(others.length>=3)return push('cyclic',[c.point,...others.slice(0,3)],[],source);return false;
  }
  if(c.kind==='cong'){const [a,b,d,e,f,g]=p;return [[a,b,e,f],[b,d,f,g],[d,a,g,e]].every(s=>push('cong',s,[],source));}
  if(c.kind==='sim'){const [a,b,d,e,f,g]=p;return [ [a,b,b,d,e,f,f,g],[b,d,d,a,f,g,g,e] ].every(s=>push('eqratio',s,[],source));}
  if(c.kind==='eq'){
   // A chain is a set of simultaneous equalities; preserve every translatable
   // part, and report unsupported subexpressions instead of silently dropping.
   if(c.expressions.length>2){let full=true;for(let i=1;i<c.expressions.length;i++)
    full=convert({kind:'eq',expressions:c.expressions.slice(i-1,i+1)},source)&&full;
    return full;}
   const [a,b]=c.expressions,sa=seg(a),sb=seg(b);
   if(sa&&sb)return push('cong',[...sa,...sb],[],source);
   const squared=e=>e?.op==='^'&&seg(e.args[0])&&numeric(e.args[1])===2?seg(e.args[0]):null;
   if(squared(a)&&squared(b))return push('cong',[...squared(a),...squared(b)],[],source);
   const ratio=(lhs,rhs)=>{
    if(lhs.op==='/'&&seg(lhs.args[0])&&seg(lhs.args[1])&&rational(numeric(rhs))&&numeric(rhs)>0)
     return push('rconst',[...seg(lhs.args[0]),...seg(lhs.args[1])],[numeric(rhs)],source);
    if(seg(lhs)&&rhs.op==='*'){
     const segment=seg(rhs.args[0])||seg(rhs.args[1]),v=numeric(seg(rhs.args[0])?rhs.args[1]:rhs.args[0]);
     if(segment&&rational(v)&&v>0)return push('rconst',[...seg(lhs),...segment],[v],source);
    }
    return false;
   };
   if(ratio(a,b)||ratio(b,a))return true;
   const quot=e=>e?.op==='/'&&seg(e.args[0])&&seg(e.args[1])?[seg(e.args[0]),seg(e.args[1])]:null;
   if(quot(a)&&quot(b))return push('eqratio',[...quot(a)[0],...quot(a)[1],...quot(b)[0],...quot(b)[1]],[],source);
   const product=e=>e?.op==='*'&&seg(e.args[0])&&seg(e.args[1])?[seg(e.args[0]),seg(e.args[1])]:null;
   if(product(a)&&product(b))return push('eqratio',[...product(a)[0],...product(b)[0],...product(b)[1],...product(a)[1]],[],source);
   if(a.op==='ang'&&b.op==='ang')return push('eqangle',[...anglePoints(a),...anglePoints(b)],[],source);
   const angle=(u,v)=>{
    const value=numeric(v);if(u.op!=='ang'||!Number.isFinite(value))return false;
    const deg=Math.abs(value)>2*Math.PI+1e-8?value:value*180/Math.PI;
    if(!rational(deg))return false; // AG2's aconst is rational degrees.
    return push('aconst',anglePoints(u),[-deg],source);
   };
   return angle(a,b)||angle(b,a);
  }
  return false;
 };
 for(const c of conditions||[]){const stat=assess([c],k,tolerance)[0],source='USER-PRETTY:'+c.text;
  if(!stat.satisfied){notes.push('未达到 AG2 提交精度，未作为前提：'+c.text+' (误差 '+stat.error.toExponential(2)+')');continue;}
  if(!convert(c,source))notes.push('AG2 原生谓词不支持完整表达，仍参与 Pretty 数值优化：'+c.text);
 }
 return {predicates:preds,limitations:notes};
}
return {parse:validate,tokenize,residuals,assess,ag2,scalar};
});
