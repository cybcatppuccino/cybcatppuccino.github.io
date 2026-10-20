/* GeomTool v8.1 — factored rational expressions over Nerdamer 2.0
 * Polynomial arithmetic, exact division, GCD, and factorization all delegate
 * to the small, upstream Nerdamer SparsePolynomial modules (Apache-2.0).
 * No rational-cas or legacy factorizer dependencies.
 */
(function(root,factory){const api=factory(root.GeomNerdamerSparse2);root.GeomNerdamerRational=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(N){
'use strict';
if(!N?.Polynomial||!N.factorMultivariateInteger||!N.sparsePolynomialGcd)throw Error('Nerdamer 2.0 sparse polynomial engine missing');
const Poly=N.Polynomial;
const abs=x=>x<0n?-x:x;
const intGcd=(a,b)=>{a=abs(a);b=abs(b);while(b){[a,b]=[b,a%b];}return a;};
let names=['gt_imaginary_unit'],nameToIndex=new Map([['gt_imaginary_unit',0]]),factorMemo=new Map(),cache=new Map();
let factorMs=10000,stepDeadline=Infinity;
function setFactorBudget(ms){factorMs=Number.isFinite(Number(ms))?Math.max(100,Number(ms)):10000;}
function beginStep(){stepDeadline=Date.now()+factorMs;}
function reset(){names=['gt_imaginary_unit'];nameToIndex=new Map([['gt_imaginary_unit',0]]);factorMemo=new Map();cache=new Map();}
function polynomial(terms){return new Poly(names.length,terms);}
const zeroPoly=()=>polynomial([]),unitPoly=()=>polynomial([{coefficient:1n,exponents:Array(names.length).fill(0n)}]);
function constantPoly(n){return n===0n?zeroPoly():polynomial([{coefficient:n,exponents:Array(names.length).fill(0n)}]);}
function variablePoly(s){let k=nameToIndex.get(s);if(k===undefined){k=names.length;nameToIndex.set(s,k);names.push(s);}const exponents=Array(names.length).fill(0n);exponents[k]=1n;return polynomial([{coefficient:1n,exponents}]);}
function lift(p){if(p.variableCount===names.length)return p;return polynomial(p.terms().map(t=>({coefficient:t.coefficient,exponents:t.exponents.concat(Array(names.length-p.variableCount).fill(0n))})));}
function reduceImaginary(p){p=lift(p);if(p.terms().every(t=>t.exponents[0]<2n))return p;return polynomial(p.terms().map(t=>{const e=t.exponents[0];const powers=t.exponents.slice();powers[0]=e%2n;return {coefficient:e/2n%2n===0n?t.coefficient:-t.coefficient,exponents:powers};}));}
function termKey(p){return p.terms().map(t=>t.coefficient+':'+t.exponents.map((e,i)=>e?i+'='+e:'').filter(Boolean).join(',')).sort().join(';');}
function product(ps){let out=unitPoly();for(const p of ps)out=reduceImaginary(lift(out).multiply(lift(p)));return out;}
const isOnePoly=p=>{const t=p.terms();return t.length===1&&t[0].coefficient===1n&&t[0].exponents.every(x=>x===0n);};
const isMinusOnePoly=p=>{const t=p.terms();return t.length===1&&t[0].coefficient===-1n&&t[0].exponents.every(x=>x===0n);};
const constNum=p=>{const t=p.terms();return t.length===0?0n:t.length===1&&t[0].exponents.every(x=>x===0n)?t[0].coefficient:null;};
function primeFactors(p){
 p=lift(p);if(p.isZero())return [zeroPoly()];
 const key=termKey(p),hit=factorMemo.get(key);if(hit)return hit.map(lift);
 let out=[p];const t=p.terms();
 try{
  // Polynomials of high degree/large support are left as exact atoms;
  // no uncontrolled expansion or unbounded recursive factor attempt.
  const used=p.variables();
  if(Date.now()<stepDeadline&&t.length>1&&t.length<=550&&p.totalDegree()<=16n&&used.length<=6){
   N.setOperationDeadline?.(stepDeadline);
   const result=used.length===1?N.factorUnivariateInteger(p,used[0]):N.factorMultivariateInteger(p,used);
   const found=[];
   if(result.content!==1n)found.push(constantPoly(result.content));
   for(const entry of result.factors){for(let i=0n;i<entry.multiplicity;i++)found.push(entry.factor);}
   if(found.length)out=found;
  }
 }catch(_){out=[p];}
 finally{N.setOperationDeadline?.(Infinity);}
 if(Date.now()<stepDeadline&&factorMemo.size<10000)factorMemo.set(key,out);
 return out.map(lift);
}
function splitPolys(polys){let k=1n,parts=[];for(const p of polys){const n=constNum(p);if(n!==null)k*=n;else if(!isOnePoly(p))parts.push(p);}return {k,parts};}
function rational(upper,lower=[],scalarN=1n,scalarD=1n){
 const u=splitPolys(upper),v=splitPolys(lower);let num=scalarN*u.k,den=scalarD*v.k;
 // Work in Z[i], with i²=-1. Never let the Gaussian unit become a free variable.
 const imagKey=termKey(variablePoly('gt_imaginary_unit'));
 const imN=u.parts.filter(p=>termKey(p)===imagKey).length,imD=v.parts.filter(p=>termKey(p)===imagKey).length;
 let imPower=((imN-imD)%4+4)%4;if(imPower>=2){num=-num;imPower-=2;}
 if(imPower===1)u.parts=[...u.parts.filter(p=>termKey(p)!==imagKey),variablePoly('gt_imaginary_unit')];else u.parts=u.parts.filter(p=>termKey(p)!==imagKey);
 v.parts=v.parts.filter(p=>termKey(p)!==imagKey);
 if(den===0n)throw Error('零分母');if(num===0n)return ZERO;
 if(den<0n){num=-num;den=-den;}
 const gg=intGcd(num,den);num/=gg;den/=gg;
 let nu=u.parts,dd=v.parts;
 // Cancel entire common factors before any polynomial multiplication.
 const db=new Map();for(const p of dd){const k=termKey(p);let a=db.get(k);if(!a)db.set(k,a=[]);a.push(p);}
 const remains=[];for(const p of nu){const k=termKey(p),existing=db.get(k);if(existing?.length)existing.pop();else remains.push(p);}
 dd=[...db.values()].flat();nu=remains;
 // Only unresolved large polynomial atoms need a structural GCD.
 for(let i=0;i<nu.length;i++)for(let j=0;j<dd.length;j++){
  const a=lift(nu[i]),b=lift(dd[j]);
  if(a.termCount>110||b.termCount>110)continue;
  if(a.termCount<=3&&b.termCount<=3)continue;
  if(!a.variables().some(k=>b.variables().includes(k)))continue;
  // The GCD needs the union of active variables; an intersection can silently
  // disable cancellation when the two expressions have different other variables.
  try{N.setOperationDeadline?.(stepDeadline);
   const g=N.sparsePolynomialGcd(a,b,[...new Set([...a.variables(),...b.variables()])].sort((x,y)=>x-y));
   if(g.isConstant())continue;const aa=a.divideExact(g),bb=b.divideExact(g);
   if(aa&&bb){nu[i]=aa;dd[j]=bb;}
  }catch(_){}finally{N.setOperationDeadline?.(Infinity);}
 }
 return {n:nu,d:dd,p:num,q:den};
}
const ZERO={n:[],d:[],p:0n,q:1n},ONE={n:[],d:[],p:1n,q:1n};
function value(a){if(a&&typeof a==='object'&&Array.isArray(a.n))return a;if(typeof a==='bigint')return rational([],[],a);if(typeof a==='number'){if(!Number.isSafeInteger(a))throw Error('非精确整数');return rational([],[],BigInt(a));}if(typeof a==='string'){if(/^[+-]?\d+$/.test(a))return rational([],[],BigInt(a));return parse(a);}throw Error('不支持的 Nerdamer 有理数类型');}
function symbol(s){return rational([variablePoly(s)]);}
const eqp=(a,b)=>termKey(a)===termKey(b);
const neg=a=>{a=value(a);return a.p===0n?a:{...a,p:-a.p};};
function mul(a,b){a=value(a);b=value(b);if(!a.p||!b.p)return ZERO;return rational([...a.n,...b.n],[...a.d,...b.d],a.p*b.p,a.q*b.q);}
function div(a,b){a=value(a);b=value(b);if(!b.p)throw Error('除以零');if(!a.p)return ZERO;return rational([...a.n,...b.d],[...a.d,...b.n],a.p*b.q,a.q*b.p);}
function commonFactors(x,y){const map=new Map();for(const f of y){const key=termKey(f);let items=map.get(key);if(!items)map.set(key,items=[]);items.push(f);}
 const common=[],rest=[];for(const f of x){const arr=map.get(termKey(f));if(arr?.length){arr.pop();common.push(f);}else rest.push(f);}
 return [common,rest,[...map.values()].flat()];
}
function add(a,b){a=value(a);b=value(b);if(!a.p)return b;if(!b.p)return a;
 // LCM denominators, preserving shared factors, rather than repeated full expansion.
 const [sharedDen,onlyA,onlyB]=commonFactors(a.d,b.d);
 const [sharedNum,leftNum,rightNum]=commonFactors(a.n,b.n);
 const coeffA=a.p*b.q,coeffB=b.p*a.q;
 // Quotient sum after removing common factors in both numerator and denominator.
 const A=product([...leftNum,...onlyB]),B=product([...rightNum,...onlyA]);
 const mixed=reduceImaginary(lift(A).scale(coeffA).add(lift(B).scale(coeffB)));
 if(mixed.isZero())return ZERO;
 const factors=primeFactors(mixed);
 return rational([...sharedNum,...factors],[...sharedDen,...onlyA,...onlyB],1n,a.q*b.q);
}
const sub=(a,b)=>add(a,neg(b));
function pow(a,n){a=value(a);if(!Number.isSafeInteger(n))throw Error('非整数幂');if(n===0)return ONE;if(n<0)return pow(div(ONE,a),-n);let r=ONE;while(n){if(n%2)r=mul(r,a);n=Math.floor(n/2);if(n)a=mul(a,a);}return r;}
function isZero(a){return value(a).p===0n;}
function eq(a,b){return isZero(sub(a,b));}
function normalizeLineCoefficients(a,b,c){
 a=value(a);b=value(b);c=value(c);
 // Projective line coefficients: only cancel factors common to all three,
 // without cross-multiplying their denominators into huge polynomials.
 const factors=[a,b,c],ns=factors.map(x=>x.n),ds=factors.map(x=>x.d);
 let common=ns[0];for(const row of ns.slice(1))common=commonFactors(common,row)[0];
 if(!common.length)return {a,b,c};
 return Object.fromEntries(['a','b','c'].map((k,i)=>[k,div(factors[i],rational(common))]));
}
function atomString(p,alias){
 const terms=p.terms();if(!terms.length)return '0';
 const mapped=terms.map(t=>{const v=t.exponents.map((power,i)=>power?((names[i]==='gt_imaginary_unit'?'i':alias?alias(names[i]):names[i])+(power===1n?'':'^'+power)):'').filter(Boolean).join('*');const num=abs(t.coefficient);return {minus:t.coefficient<0n,text:(v?(num===1n?'':num+'*')+v:String(num)),degree:t.exponents.reduce((s,e)=>s+e,0n)};});
 mapped.sort((x,y)=>x.degree===y.degree?x.text.localeCompare(y.text):x.degree>y.degree?-1:1);
 return mapped.map((t,i)=>(t.minus?'-':i?'+':'')+t.text).join('');
}
function factorStringParts(ps,alias){
 const counts=new Map();for(const p of ps){const key=termKey(p),item=counts.get(key);if(item)item.power++;else counts.set(key,{p,power:1});}
 return [...counts.values()].map(({p,power})=>{const s=atomString(p,alias),wrap=/^[A-Za-z_][A-Za-z0-9_]*$/.test(s)?s:'('+s+')';return wrap+(power>1?'^'+power:'');}).join('*');
}
function string(a,alias){a=value(a);if(!a.p)return '0';
 const n=[...(a.p!==1n||!a.n.length?[String(a.p)]:[]),factorStringParts(a.n,alias)].filter(Boolean).join('*');
 if(a.q===1n&&!a.d.length)return n||'1';
 const d=[...(a.q!==1n?[String(a.q)]:[]),factorStringParts(a.d,alias)].filter(Boolean).join('*')||'1';
 return '('+(n||'1')+')/('+d+')';
}
// For large expanded integer polynomials, parse in the sparse ring once.
// The ordinary factored-rational parser remains the default for short inputs,
// negative powers, divisions, and expressions where expansion is undesirable.
function tryParseExpanded(tokens) {
 let additions=0,depth=0;
 for(const t of tokens){if(t==='(')depth++;else if(t===')')depth--;else if((t==='+'||t==='-')&&depth===0)additions++;}
 if(additions<10||tokens.includes('/'))return null;
 let index=0;
 const atom=()=>{const t=tokens[index++];if(t==='('){const p=sum();if(tokens[index++]!==')')throw Error('缺少右括号');return p;}
  if(!t)throw Error('表达式缺少项');
  return /^\d+$/.test(t)?constantPoly(BigInt(t)):variablePoly(t);};
 const power=()=>{let p=atom();if(tokens[index]==='^'){
  index++;const exponent=tokens[index++];if(!/^\d+$/.test(exponent||''))throw Error('指数需非负整数');
  const k=BigInt(exponent);if(k>16n)throw Error('展开次数超过快速解析限制');
  p=lift(p).pow(k);
 }return p;};
 const unary=()=>tokens[index]==='-'?(index++,lift(unary()).negate()):tokens[index]==='+'?(index++,unary()):power();
 const term=()=>{let p=unary();while(tokens[index]==='*'){index++;const b=unary();if((p.totalDegree()??0n)+(b.totalDegree()??0n)>16n)throw Error('快速解析乘积次数超限');p=reduceImaginary(lift(p).multiply(lift(b)));if(p.termCount>1600)throw Error('快速解析规模超限');}return p;};
 const sum=()=>{let p=term();while(tokens[index]==='+'||tokens[index]==='-'){
  const op=tokens[index++],right=term();p=op==='+'?lift(p).add(lift(right)):lift(p).subtract(lift(right));if(p.termCount>1600)throw Error('快速解析规模超限');
 }return p;};
 const poly=sum();
 if(index!==tokens.length)throw Error('快速多项式解析未消耗全部输入');
 if(poly.termCount>1200||poly.totalDegree()>16n)return null;
 return rational(primeFactors(reduceImaginary(lift(poly))));
}
function parse(source){const src=String(source).replace(/\s+/g,'');const tokens=src.match(/\d+|[A-Za-z_][A-Za-z0-9_]*|[()+*\/^\-]/g)||[];if(tokens.join('')!==src)throw Error('Nerdamer 输入含不支持字符');
 if(src.length>80){try{const accelerated=tryParseExpanded(tokens);if(accelerated!==null)return accelerated;}catch(_){/* fallback to the established rational parser */}}
 let i=0;
 function primary(){if(tokens[i]==='('){i++;const p=sum();if(tokens[i++]!==')')throw Error('缺少右括号');return p;}const t=tokens[i++];if(!t)throw Error('公式不完整');return /^\d+$/.test(t)?value(t):symbol(t);}
 function power(){let a=primary();if(tokens[i]==='^'){i++;let sign=1;if(tokens[i]==='-'){i++;sign=-1;}const e=tokens[i++];if(!/^\d+$/.test(e||''))throw Error('指数必须是整数');a=pow(a,sign*Number(e));}return a;}
 function unary(){return tokens[i]==='-'?(i++,neg(unary())):power();}
 function term(){let v=unary();while(tokens[i]==='*'||tokens[i]==='/'){const op=tokens[i++],r=unary();v=op==='*'?mul(v,r):div(v,r);}return v;}
 function sum(){let v=term();while(tokens[i]==='+'||tokens[i]==='-'){const op=tokens[i++],r=term();v=op==='+'?add(v,r):sub(v,r);}return v;}
 const result=sum();if(i!==tokens.length)throw Error('公式解析尾部无效');return result;
}
function factor(a,alias){return string(a,alias);}
function stats(){return {symbols:names.length,polynomialFactors:factorMemo.size};}
return {setFactorBudget,beginStep,value,symbol,add,sub,addGoal: add,subGoal:sub,mul,mulGoal:mul,div,neg,pow,isZero,eq,string,parse,factor,reset,normalizeLineCoefficients,stats,zero:ZERO,one:ONE,i:symbol('gt_imaginary_unit')};
});
