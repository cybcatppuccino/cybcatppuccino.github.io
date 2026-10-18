/* GeomTool v7.9 exact rational-complex algebra. No SymPy, eval, network or floating arithmetic.
 * Sparse Q(i)[variables] rational functions, immutable polynomials, no application-level size or time caps.
 * All failed normalizations are reported, never silently approximated.
 * License: original GeomTool implementation.
 */
(function(root,factory){const api=factory();root.GeomRationalCAS=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';

const abs=n=>n<0n?-n:n;
function gcd(a,b){a=abs(a);b=abs(b);while(b){const r=a%b;a=b;b=r;}return a;}
function gc3(a,b,c){return gcd(gcd(a,b),c);}
function ga(re,im=0n,den=1n){re=BigInt(re);im=BigInt(im);den=BigInt(den);if(!den)throw Error('零分母');if(den<0n){re=-re;im=-im;den=-den;}if(!re&&!im)return G0;const d=gc3(re,im,den);re/=d;im/=d;den/=d;return {re,im,den};}
const G0={re:0n,im:0n,den:1n},G1={re:1n,im:0n,den:1n},GI={re:0n,im:1n,den:1n};
const gz=a=>a.re===0n&&a.im===0n;
const eqg=(a,b)=>a.re===b.re&&a.im===b.im&&a.den===b.den;
const gn=a=>ga(-a.re,-a.im,a.den),gj=a=>ga(a.re,-a.im,a.den);
const ggadd=(a,b)=>gz(a)?b:gz(b)?a:ga(a.re*b.den+b.re*a.den,a.im*b.den+b.im*a.den,a.den*b.den);
const ggsub=(a,b)=>ggadd(a,gn(b));
const gmul=(a,b)=>gz(a)||gz(b)?G0:eqg(a,G1)?b:eqg(b,G1)?a:ga(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re,a.den*b.den);
const gdiv=(a,b)=>{const t=b.re*b.re+b.im*b.im;if(!t)throw Error('除以零');return ga((a.re*b.re+a.im*b.im)*b.den,(a.im*b.re-a.re*b.im)*b.den,a.den*t);};
const isone=a=>eqg(a,G1),symbolNames=[],symbolIndex=new Map(),monomialCache=new Map([['',[]]]),mmCache=new Map();
function monarr(key){let ar=monomialCache.get(key);if(!ar){ar=key.split(',').map(s=>s.split(':').map(Number));monomialCache.set(key,ar);}return ar;}
function mkey(arr){return arr.filter(p=>p[1]>0).map(p=>p[0]+':'+p[1]).join(',');}
function mmul(a,b){if(!a)return b;if(!b)return a;const tag=a+'|'+b,cache=mmCache.get(tag);if(cache!==undefined)return cache;const aa=monarr(a),bb=monarr(b),c=[];let i=0,j=0;while(i<aa.length||j<bb.length){if(j>=bb.length||i<aa.length&&aa[i][0]<bb[j][0])c.push(aa[i++]);else if(i>=aa.length||bb[j][0]<aa[i][0])c.push(bb[j++]);else{c.push([aa[i][0],aa[i][1]+bb[j][1]]);i++;j++;}}const key=mkey(c);if(mmCache.size<120000)mmCache.set(tag,key);return key;}
function mdiv(a,b){const aa=monarr(a),bb=monarr(b),c=[];let i=0,j=0;while(i<aa.length){let e=aa[i][1];if(j<bb.length&&aa[i][0]===bb[j][0]){e-=bb[j][1];j++;}else if(j<bb.length&&bb[j][0]<aa[i][0])return null;if(e<0)return null;if(e)c.push([aa[i][0],e]);i++;}return j===bb.length?mkey(c):null;}
function mcompare(a,b){const aa=monarr(a),bb=monarr(b);let da=0,db=0;for(const v of aa)da+=v[1];for(const v of bb)db+=v[1];if(da!==db)return da-db;let i=0,j=0;while(i<aa.length||j<bb.length){const ix=Math.min(i<aa.length?aa[i][0]:Infinity,j<bb.length?bb[j][0]:Infinity);const ea=i<aa.length&&aa[i][0]===ix?aa[i++][1]:0,eb=j<bb.length&&bb[j][0]===ix?bb[j++][1]:0;if(ea!==eb)return ea-eb;}return 0;}
const P0=new Map(),P1=new Map([['',G1]]);
const barNames=new Map();
function pconst(g){return gz(g)?P0:isone(g)?P1:new Map([['',g]]);}
function varpoly(i){return new Map([[i+':1',G1]]);}
function cp(p){return p.size===1&&p.has('')?p.get(''):null;}
function chk(p){return p;}
function padd(a,b){if(!a.size)return b;if(!b.size)return a;const c=new Map(a);for(const [k,v] of b){const t=c.has(k)?ggadd(c.get(k),v):v;if(gz(t))c.delete(k);else c.set(k,t);}return chk(c);}
function pneg(a){if(!a.size)return a;const b=new Map();for(const [k,v] of a)b.set(k,gn(v));return b;}
function psub(a,b){return padd(a,pneg(b));}
function pscale(a,g){if(gz(g))return P0;if(isone(g))return a;const b=new Map();for(const [k,v] of a)b.set(k,gmul(v,g));return b;}
function pmul(a,b){if(!a.size||!b.size)return P0;if(a===P1)return b;if(b===P1)return a;const ca=cp(a),cb=cp(b);if(ca)return pscale(b,ca);if(cb)return pscale(a,cb);const c=new Map();for(const [ka,va] of a)for(const [kb,vb] of b){const k=mmul(ka,kb),v=gmul(va,vb),t=c.has(k)?ggadd(c.get(k),v):v;if(gz(t))c.delete(k);else c.set(k,t);}return chk(c);}
function pident(a,b){if(a===b)return true;if(a.size!==b.size)return false;for(const [k,v] of a)if(!b.has(k)||!eqg(v,b.get(k)))return false;return true;}
function leading(a){let k=null,v=null;for(const [mk,cv] of a)if(k===null||mcompare(mk,k)>0){k=mk;v=cv;}return [k,v];}
function pdivide(a,b,needRem=false){
 if(!b.size)throw Error('除以零多项式');
 if(!a.size)return needRem?[P0,P0]:P0;
 const scalar=cp(b);
 if(scalar){const out=pscale(a,gdiv(G1,scalar));return needRem?[out,P0]:out;}
 let work=new Map(a),quot=P0,rem=P0;const [bk,bv]=leading(b);
 while(work.size){
  const [ak,av]=leading(work),key=mdiv(ak,bk);
  if(key===null){const term=new Map([[ak,av]]);rem=padd(rem,term);work.delete(ak);continue;}
  const step=new Map([[key,gdiv(av,bv)]]);
  quot=padd(quot,step);work=psub(work,pmul(step,b));
 }
 return needRem?[quot,rem]:(!rem.size?quot:null);
}
function variables(p){const set=new Set();for(const k of p.keys())for(const [i] of monarr(k))set.add(i);return set;}
function univariateGcd(a,b){let v=new Set([...variables(a),...variables(b)]);if(v.size!==1)return null;while(b.size){const [,r]=pdivide(a,b,true);if(r.size>=b.size&&pident(r,a))return null;a=b;b=r;}if(!a.size)return null;const [,lc]=leading(a);return pscale(a,gdiv(G1,lc));}
function monomialGcd(a,b){if(!a.size||!b.size)return '';let min=null;for(const k of [...a.keys(),...b.keys()]){const d=new Map(monarr(k));if(min===null)min=d;else for(const [i,e] of min){const v=Math.min(e,d.get(i)||0);if(v)min.set(i,v);else min.delete(i);}}return mkey([...min].sort((x,y)=>x[0]-y[0]));}

// Exact multivariate polynomial GCD via recursive primitive pseudo-remainder
// sequences over Q(i). No evaluation, heuristically guessed divisors, or CAS.
// Pseudo division takes place in Q(i)[other variables][pivot]; each nonzero
// remainder is made primitive by a recursive coefficient GCD. The bounds only
// disable an expensive attempt, never accept a non-divisor.

function dropVar(p,id){const out=new Map();for(const [k,c] of p){const nk=mkey(monarr(k).filter(([j])=>j!==id));out.set(nk,c);}return out;}
function coefficients(p,id){const out=new Map();for(const [k,c] of p){let exp=0;const parts=[];for(const [j,e] of monarr(k)){if(j===id)exp=e;else parts.push([j,e]);}const key=mkey(parts);let poly=out.get(exp);if(!poly){poly=new Map();out.set(exp,poly);}poly.set(key,c);}return out;}
function toUnivar(p,id){return coefficients(p,id);}
function degOf(cs){return cs.size?Math.max(...cs.keys()):-1;}
function embedCoeff(p,id,e){if(e===0)return p;const x=mkey([[id,e]]);const out=new Map();for(const [k,c] of p)out.set(mmul(k,x),c);return out;}
function primitiveGcd(a,b,state,depth=0){
 if(!a.size)return normalizePoly(b);
 if(!b.size)return normalizePoly(a);

 const ids=[...new Set([...variables(a),...variables(b)])];
 if(!ids.length)return P1;
 if(ids.length===1){while(b.size){const rem=pdivide(a,b,true)[1];a=b;b=rem;}return normalizePoly(a);}
 // A low-degree pivot minimizes pseudodivision intermediate growth.
 const id=ids.map(v=>({id:v,da:degOf(toUnivar(a,v)),db:degOf(toUnivar(b,v))}))
  .filter(v=>v.da>0||v.db>0).sort((x,y)=>Math.max(x.da,x.db)-Math.max(y.da,y.db))[0]?.id;
 if(id===undefined)return P1;
 const ca=toUnivar(a,id),cb=toUnivar(b,id);
 function content(cs){let g=P0;for(const poly of cs.values()){

  g=g.size?primitiveGcd(g,poly,state,depth+1):poly;
  if(cp(g))return P1;
 }return normalizePoly(g);
 }
 function primitive(cs){const c=content(cs);if(cp(c)||!c.size)return cs;
  const result=new Map();for(const [ex,p] of cs){const q=pdivide(p,c);if(q===null)throw Error('Inexact primitive division');if(q.size)result.set(ex,q);}return result;
 }
 const contA=content(ca),contB=content(cb),contG=primitiveGcd(contA,contB,state,depth+1);
 let A=primitive(ca),B=primitive(cb);if(degOf(A)<degOf(B))[A,B]=[B,A];
 while(B.size){

  let R=new Map(A),db=degOf(B),bc=B.get(db);
  while(R.size&&degOf(R)>=db){
   
   const dr=degOf(R),rc=R.get(dr),next=new Map();
   // R <- lc(B)*R - lc(R)*x^(dr-db)*B, coefficientwise.
   for(const [e,p] of R)next.set(e,pmul(p,bc));
   for(const [e,p] of B){const k=e+dr-db,v=pmul(p,rc),difference=next.has(k)?psub(next.get(k),v):pneg(v);if(difference.size)next.set(k,difference);else next.delete(k);}

   R=next;
  }
  A=B;B=primitive(R);
 }

 let candidate=P0;for(const [e,coeff] of A)candidate=padd(candidate,embedCoeff(coeff,id,e));
 candidate=pmul(candidate,contG);
 return normalizePoly(candidate);
}
function normalizePoly(p){if(!p.size)return P0;const [,lc]=leading(p);return isone(lc)?p:pscale(p,gdiv(G1,lc));}
function polynomialGcd(a,b){
 if(!a.size)return normalizePoly(b);if(!b.size)return normalizePoly(a);
 if(pident(a,b))return normalizePoly(a);
 if(cp(a)||cp(b))return P1;
 const mon=monomialGcd(a,b);
 // A polynomial must be normalized BEFORE testing exact divisibility.
 {
  try{
   const q=pdivide(a,b);if(q!==null)return normalizePoly(b);
   if(pdivide(b,a)!==null)return normalizePoly(a);
   const g=primitiveGcd(a,b,{steps:0});
   if(g.size>1&&pdivide(a,g)!==null&&pdivide(b,g)!==null)return g;
  }catch(e){/* Reject failed, unproven GCD; the original expression remains exact. */}
 }
 return mon?new Map([[mon,G1]]):P1;
}
function cancelGcd(n,d){
 if(n.size===0)return [P0,P1];if(d.size===0)throw Error('除以零');
 const g=polynomialGcd(n,d);
 if(g.size>1||g.size===1&&!g.has('')){const nn=pdivide(n,g),dd=pdivide(d,g);if(nn!==null&&dd!==null){n=nn;d=dd;}}
 return [n,d];
}
function commonDivisors(a,b){const g=polynomialGcd(a,b);if(g.size===1&&g.has(''))return [a,b];const aa=pdivide(a,g),bb=pdivide(b,g);return aa&&bb?[aa,bb]:[a,b];}


// Algebraic square recognition and coefficient-primitive / quadratic splitting.
// A proposed factor is NEVER emitted without an exact polynomial division.
function bigSqrt(n){if(n<0n)return null;if(n<2n)return n;let a=1n<<BigInt((n.toString(2).length+1)>>1);for(;;){const b=(a+n/a)>>1n;if(b>=a)return a*a===n?a:null;a=b;}}
function sqrtFraction(n,d){if(n<0n||d<=0n)return null;const a=bigSqrt(n),b=bigSqrt(d);return a===null||b===null?null:{a,b};}
function sqrtGaussian(z){
 const len=sqrtFraction(z.re*z.re+z.im*z.im,z.den*z.den);if(!len)return null;
 const rr=ga(len.a,0n,len.b),plus=ggadd(rr,ga(z.re,0n,z.den)),minus=ggsub(rr,ga(z.re,0n,z.den));
 const real=sqrtFraction(plus.re,2n*plus.den),imag=sqrtFraction(minus.re,2n*minus.den);
 if(!real||!imag)return null;
 // Different component denominators must be taken through ga separately.
 const corrected=ggadd(ga(real.a,0n,real.b),ga(0n,z.im<0n?-imag.a:imag.a,imag.b));
 return eqg(gmul(corrected,corrected),z)?corrected:null;
}
function squareRootPolynomial(p){
 if(!p.size)return P0;
 const [k,c]=leading(p),arr=monarr(k);if(arr.some(([,e])=>e%2))return null;
 const sc=sqrtGaussian(c);if(!sc)return null;
 let r=new Map([[mkey(arr.map(([i,e])=>[i,e/2])),sc]]);
 for(;;){
  let remainder;try{remainder=psub(p,pmul(r,r));}catch(e){return null;}
  if(!remainder.size)return r;
  const [rk,rc]=leading(remainder),[lk,lc]=leading(r),term=mdiv(rk,lk);
  if(term===null)return null;
  const coeff=gdiv(rc,gmul(ga(2n),lc));
  r=padd(r,new Map([[term,coeff]]));
 }
 return null;
}
function derivative(p,id){const r=new Map();for(const [k,v] of p){const aa=monarr(k),e=aa.find(([i])=>i===id)?.[1]||0;if(e)r.set(mkey(aa.map(([j,n])=>[j,j===id?n-1:n])),gmul(v,ga(BigInt(e))));}return r;}
function knownFactorCandidates(p){
 const ids=[...variables(p)].sort((x,y)=>x-y);
 const out=[];
 {
  // Extract polynomial coefficient content in each variable. This detects
  // products with nonlinear multivariate factors, not just linear forms.
  for(const id of ids){const cs=toUnivar(p,id);if(cs.size<2)continue;
   let content=P0;for(const poly of cs.values()){content=content.size?polynomialGcd(content,poly):poly;if(cp(content))break;}
   if(content.size>1)out.push(content);
  }
  // Polynomial-square discriminant test with EVERY variable as the main variable.
  for(const id of ids){const cs=toUnivar(p,id);if(degOf(cs)!==2)continue;
   const a=cs.get(2),b=cs.get(1)||P0,c=cs.get(0)||P0;
   let diff,root;try{diff=psub(pmul(b,b),pscale(pmul(a,c),ga(4n)));root=squareRootPolynomial(diff);}catch(e){continue;}
   if(!root)continue;
   for(const sign of [1,-1]){
    const expr=padd(pscale(embedCoeff(a,id,1),ga(2n)),sign>0?padd(b,root):psub(b,root));
    if(expr.size<=1)continue;
    out.push(expr);
    const coeff=toUnivar(expr,id);let cont=P0;
    for(const poly of coeff.values()){cont=cont.size?polynomialGcd(cont,poly):poly;if(cp(cont))break;}
    if(cont.size>1){const pure=pdivide(expr,cont);if(pure)out.push(pure);}
   }
  }
 }
 return out;
}
// Small exact multivariate linear-factor search. Every proposed factor is
// checked by polynomial long division, so it can never introduce false equalities.
// This catches (x^2-y^2)/(x-y), (x^2+1)/(x-i) and their geometry analogues.
function candidateFactors(a,b=null){
 const ids=[...new Set([...variables(a),...(b?variables(b):[])])].sort((x,y)=>x-y);

 const forms=[];
 // Univariate roots in Q(i), then all possible affine combinations of 2/3
 // variables. Only candidates provably dividing the input are retained.
 for(const id of ids){const x=varpoly(id);
   for(const c of [G1,gn(G1),GI,gn(GI),ga(2n),ga(-2n)])forms.push(padd(x,pconst(c)));
 }
 const max=ids.length>5?2:3;
 const coefficients=[G1,gn(G1),GI,gn(GI),ga(2n),ga(-2n)];
 for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
   const x=varpoly(ids[i]),y=varpoly(ids[j]);
   for(const coeff of coefficients)forms.push(padd(x,pscale(y,coeff)));
   if(max>=3)for(let k=j+1;k<ids.length;k++)for(const c of [G1,gn(G1),GI,gn(GI)])for(const d of [G1,gn(G1)]){
     forms.push(padd(padd(x,pscale(y,c)),pscale(varpoly(ids[k]),d)));
   }
 }
 return forms;
}
function cancelLinear(n,d){

 for(const candidate of candidateFactors(n,d)){
  for(;;){
   const qn=pdivide(n,candidate),qd=pdivide(d,candidate);
   if(qn===null||qd===null)break;
   n=qn;d=qd;
  }
 }
 return [n,d];
}
function cleanup(n,d){
 if(!d.size)throw Error('除以零');if(!n.size)return {n:P0,d:P1};if(pident(n,d))return {n:P1,d:P1};
 const scalar=cp(d);if(scalar)return {n:pscale(n,gdiv(G1,scalar)),d:P1};
 [n,d]=cancelGcd(n,d);
 const [,lc]=leading(d);if(!isone(lc)){const inv=gdiv(G1,lc);n=pscale(n,inv);d=pscale(d,inv);}
 return {n,d};
}
class Rat{
 constructor(n,d=P1){const normalized=cleanup(n,d);this.n=normalized.n;this.d=normalized.d;}
}
const ZERO=new Rat(P0),ONE=new Rat(P1),IMAG=new Rat(pconst(GI));
function value(x){if(x instanceof Rat)return x;if(typeof x==='bigint')return new Rat(pconst(ga(x)));if(typeof x==='number'&&Number.isSafeInteger(x))return new Rat(pconst(ga(BigInt(x))));if(typeof x==='string'&&/^-?\d+$/.test(x))return new Rat(pconst(ga(BigInt(x))));throw Error('仅支持整数/有理数常量');}
function symbol(name){name=String(name);let i=symbolIndex.get(name);if(i===undefined){i=symbolNames.length;symbolIndex.set(name,i);symbolNames.push(name);}return new Rat(varpoly(i));}
function neg(x){x=value(x);return new Rat(pneg(x.n),x.d);}
function eq(a,b){a=value(a);b=value(b);return !psub(pmul(a.n,b.d),pmul(b.n,a.d)).size;}
function add(a,b){a=value(a);b=value(b);if(!a.n.size)return b;if(!b.n.size)return a;
 const g=polynomialGcd(a.d,b.d),ad=pdivide(a.d,g),bd=pdivide(b.d,g);
 if(ad===null||bd===null)return new Rat(padd(pmul(a.n,b.d),pmul(b.n,a.d)),pmul(a.d,b.d));
 return new Rat(padd(pmul(a.n,bd),pmul(b.n,ad)),pmul(ad,b.d));
}
function sub(a,b){return add(a,neg(b));}
function mul(a,b){a=value(a);b=value(b);if(!a.n.size||!b.n.size)return ZERO;
 if(a===ONE)return b;if(b===ONE)return a;
 let [an,bd]=commonDivisors(a.n,b.d),[bn,ad]=commonDivisors(b.n,a.d);
 return new Rat(pmul(an,bn),pmul(ad,bd));
}
function div(a,b){a=value(a);b=value(b);if(!b.n.size)throw Error('除以零');if(!a.n.size)return ZERO;if(eq(a,b))return ONE;
 let [an,bn]=commonDivisors(a.n,b.n),[bd,ad]=commonDivisors(b.d,a.d);
 return new Rat(pmul(an,bd),pmul(ad,bn));
}
function pow(a,n){a=value(a);if(!Number.isSafeInteger(n))throw Error('幂指数必须是安全整数');if(n<0)return pow(div(ONE,a),-n);let result=ONE;while(n){if(n&1)result=mul(result,a);n=Math.floor(n/2);if(n)a=mul(a,a);}return result;}
function conjugate(a,mapping=new Map()){a=value(a);const cache=new Map();const conv=p=>{let sum=ZERO;for(const [k,c] of p){let term=new Rat(pconst(gj(c)));for(const [i,e] of monarr(k)){let v=cache.get(i);if(!v){const name=symbolNames[i],mapped=mapping.get(name);v=mapped===undefined?symbol(barNames.get(name)||('conj_'+name)):value(mapped);if(mapped===undefined){barNames.set(name,'conj_'+name);barNames.set('conj_'+name,name);}cache.set(i,v);}term=mul(term,pow(v,e));}sum=add(sum,term);}return sum;};return div(conv(a.n),conv(a.d));}
function isZero(a){return !value(a).n.size;}
function rationalString(g){const re=g.re,im=g.im,d=g.den;let parts=[];if(re)parts.push((d===1n?re:re+'/'+d).toString());if(im){const sign=im<0n?'-':'+',v=abs(im),s=(v===d?'':(d===1n?v:v+'/'+d).toString()+'*')+'i';parts.push((re?sign:im<0n?'-':'')+s);}return parts.length?parts.join(''):'0';}
function polynomialString(p,alias=(s)=>s){if(!p.size)return '0';const sorted=[...p].sort((x,y)=>mcompare(y[0],x[0]));return sorted.map(([k,c],index)=>{const name=monarr(k).map(([i,e])=>{const str=alias(symbolNames[i]);return (/[+*/\s\-]/.test(str)?'('+str+')':str)+(e>1?'^'+e:'');}).join('*');const coeff=rationalString(c);if(!name)return (index&&!coeff.startsWith('-')?'+':'')+coeff;const c1=eqg(c,G1),cm=eqg(c,gn(G1));let term=c1?name:cm?'-'+name:/^-?\d+$/.test(coeff)?coeff+'*'+name:'('+coeff+')*'+name;if(index&&!term.startsWith('-'))term='+'+term;return term;}).join('');}
function string(a,alias){a=value(a);const n=polynomialString(a.n,alias),d=polynomialString(a.d,alias);return a.d===P1||pident(a.d,P1)?n:'('+n+')/('+d+')';}
function evalAt(a,values){a=value(a);const pe=p=>{let v=G0;for(const [key,c] of p){let t=c;for(const [i,e] of monarr(key)){const x=values[symbolNames[i]];if(!x)throw Error('缺少符号数值 '+symbolNames[i]);let term=G1;for(let j=0;j<e;j++)term=gmul(term,x);t=gmul(t,term);}v=ggadd(v,t);}return v;};return gdiv(pe(a.n),pe(a.d));}
// A safe, arithmetic-only grammar for test cases and plug-in consumers.
function parse(source){const s=String(source).replace(/\s+/g,'');const tokens=s.match(/\d+|[A-Za-z_][A-Za-z_0-9]*|[()+*\/^\-]/g)||[];if(tokens.join('')!==s)throw Error('不支持的输入字符');let i=0;const primary=()=>{if(tokens[i]==='('){i++;const r=sum();if(tokens[i++]!==')')throw Error('缺少右括号');return r;}const t=tokens[i++];if(!t)throw Error('公式不完整');return /^\d+$/.test(t)?value(t):t==='i'?IMAG:symbol(t);};const power=()=>{let p=primary();if(tokens[i]==='^'){i++;let sign=1;if(tokens[i]==='-'){i++;sign=-1;}let n=tokens[i++];if(!/^\d+$/.test(n||''))throw Error('幂必须为整数');p=pow(p,sign*Number(n));}return p;};const unary=()=>tokens[i]==='-'?(i++,neg(unary())):power();const product=()=>{let p=unary();while(tokens[i]==='*'||tokens[i]==='/'){const op=tokens[i++],t=unary();p=op==='*'?mul(p,t):div(p,t);}return p;};const sum=()=>{let p=product();while(tokens[i]==='+'||tokens[i]==='-'){const op=tokens[i++],t=product();p=op==='+'?add(p,t):sub(p,t);}return p;};const out=sum();if(i!==tokens.length)throw Error('公式语法错误');return out;}
// Exact, browser-native factoring and fraction normalization for display.
// A shared denominator is cleared across BOTH sides, and any common integer
// content is cancelled before factoring. All factors come from proven exact
// polynomial divisions; no external CAS or floating approximations are used.
const lcm=(a,b)=>a/gcd(a,b)*b;
function integralPair(n,d){
 let commonDen=1n,commonGcd=0n;
 for(const p of [n,d])for(const [,c] of p)commonDen=lcm(commonDen,c.den);
 const scale=p=>{
  const out=new Map();
  for(const [k,c] of p){
   const v=commonDen/c.den;
   const next=ga(c.re*v,c.im*v,1n);
   if(!gz(next)){out.set(k,next);commonGcd=gcd(commonGcd,gcd(next.re,next.im));}
  }
  return out;
 };
 let nn=scale(n),dd=scale(d);
 if(commonGcd>1n){const compact=p=>new Map([...p].map(([k,c])=>[k,ga(c.re/commonGcd,c.im/commonGcd)]));nn=compact(nn);dd=compact(dd);}
 // A unit rotation is harmless; choose the principal orientation of the
 // denominator so a visible leading negative sign is not unnecessarily kept.
 const [,lead]=leading(dd);
 const unit=lead.re<0n||(lead.re===0n&&lead.im<0n)?gn(G1):G1;
 if(!isone(unit)){nn=pscale(nn,unit);dd=pscale(dd,unit);}
 return [nn,dd];
}
function integralPrimitive(p){
 let denominator=1n,content=0n;
 for(const [,c] of p)denominator=lcm(denominator,c.den);
 const out=new Map();
 for(const [k,c] of p){const z=ga(c.re*(denominator/c.den),c.im*(denominator/c.den));
  if(!gz(z)){out.set(k,z);content=gcd(content,gcd(z.re,z.im));}}
 if(content>1n)for(const [k,c] of out)out.set(k,ga(c.re/content,c.im/content));
 const [,lc]=leading(out);
 if(lc&&(lc.re<0n||(lc.re===0n&&lc.im<0n)))return pscale(out,gn(G1));
 return out;
}
// Two independent factorization passes: exact Q(i) arithmetic is authoritative;
// Nerdamer is a source of *untrusted candidate factors only*. No proposal is
// accepted without polynomial GCD and an exact quotient in the native kernel.
const factorCache=new Map();
const resultFactorCache=new Map();
function factoredCandidatesFromNerdamer(p){
 const engine=(typeof globalThis!=='undefined'?globalThis.nerdamer:null);
 if(!engine||typeof engine.factor!=='function')return [];
 if([...p.values()].some(c=>c.im!==0n))return []; // i-containing coefficients are native-only.
 let output;
 try{
  const core=engine.getCore();core.Settings.TIMEOUT=Infinity;
  output=engine.factor(polynomialString(p)).toString();
 }catch(e){return [];}
 // The Nerdamer surface notation may include radicals. Such suggestions are
 // ignored unless expressible as a polynomial over our exact coefficient field.
 // An unfactored sum is NOT a product: top-level '*' would otherwise
 // split individual monomials and cause expensive, meaningless GCD searches.
 let level=0;
 for(let i=0;i<output.length;i++){
  const c=output[i];if(c==='(')level++;else if(c===')')level--;
  else if(level===0&&i>0&&(c==='+'||c==='-')&&!['^','*','/','+','-'].includes(output[i-1]))return [];
 }
 const terms=[];let start=0,depth=0;
 for(let i=0;i<=output.length;i++){
  const c=output[i];
  if(c==='(')depth++;else if(c===')')depth--;
  if((c==='*'&&depth===0)||i===output.length){terms.push(output.slice(start,i));start=i+1;}
 }
 const unwrap=t=>{
  while(t.startsWith('(')&&t.endsWith(')')){
   let level=0,closeAt=-1;
   for(let j=0;j<t.length;j++){if(t[j]==='(')level++;else if(t[j]===')'&&--level===0){closeAt=j;break;}}
   if(closeAt!==t.length-1)break;t=t.slice(1,-1);
  }
  return t;
 };
 const out=[];
 for(let term of terms){term=term.trim();term=term.replace(/\^\d+$/,'');term=unwrap(term);
  if(!term||term==='1')continue;
  try{const parsed=parse(term);if(pident(parsed.d,P1)&&parsed.n.size>1)out.push(parsed.n);}catch(e){/* Nonpolynomial Nerdamer proposal: ignore. */}
 }
 return out;
}
function factorString(a,alias){
 a=value(a);alias=typeof alias==='function'?alias:s=>s;
 const [numerator,denominator]=integralPair(a.n,a.d);
 function splitAll(poly){
  const cacheKey=polynomialString(poly);
  if(resultFactorCache.has(cacheKey))return resultFactorCache.get(cacheKey);
  let constant=G1,parts=[];
  const queue=[poly];
  while(queue.length){
   let current=queue.pop();
   if(!current.size){return {constant:G0,parts:[]};}
   // Normalize each candidate up to a Gaussian scalar. Scalar extraction is
   // explicitly accounted for, including opposite signs and rational content.
   const primitive=integralPrimitive(current);
   const scalar=cp(pdivide(current,primitive));
   if(scalar===null)throw Error('非整数本原多项式的精确商验证失败');
   constant=gmul(constant,scalar);current=primitive;
   const coefficient=cp(current);
   if(coefficient){constant=gmul(constant,coefficient);continue;}
   if(current.size===1){
    const [key,c]=[...current][0];constant=gmul(constant,c);
    for(const [i,e] of monarr(key))parts.push({p:varpoly(i),exponent:e});
    continue;
   }
   const key=polynomialString(current);
   let proposals=factorCache.get(key);
   if(!proposals){
    // Both engines inspect the very same primitive polynomial, independently.
    const a=knownFactorCandidates(current);
    const b=factoredCandidatesFromNerdamer(current);
    const c=candidateFactors(current);
    proposals=[...a.map(p=>({p,external:false})),...b.map(p=>({p,external:true})),...c.map(p=>({p,external:false}))];
    // Exact tests may be costly, so memoize successful split decisions rather
    // than caching entire symbolic expressions between different documents.
   }
   let found=null;
   for(const {p:candidate,external} of proposals){
    if(!candidate.size||cp(candidate))continue;
    const base=integralPrimitive(candidate);
    let divisor=base,quotient=pdivide(current,base);
    if(quotient===null){
     // Proposals from different factorizers may overlap partially (e.g. ab
     // against bc): compare by multivariate GCD before discarding either.
     if(external){
      const common=polynomialGcd(current,base);
      if(common.size>1){divisor=integralPrimitive(common);quotient=pdivide(current,divisor);}
     }
    }
    if(quotient===null||cp(quotient))continue;
    found=[divisor,quotient];break;
   }
   if(found){
    // Re-run both engines on every newly derived factor until fixed point.
    queue.push(found[0],found[1]);
   }else{
    factorCache.set(key,[]);
    parts.push({p:current,exponent:1});
   }
  }
  // A multivariate GCD refinement across all accepted factors removes
  // overlap if two engines chose different composite partitions.
  let changed=true;
  while(changed){
   changed=false;
   outer:for(let i=0;i<parts.length;i++)for(let j=i+1;j<parts.length;j++){
    if(pident(parts[i].p,parts[j].p))continue;
    const g=polynomialGcd(parts[i].p,parts[j].p);
    if(cp(g)||!g.size)continue;
    const d=integralPrimitive(g);
    const qi=pdivide(parts[i].p,d),qj=pdivide(parts[j].p,d);
    if(qi===null||qj===null||cp(qi)&&cp(qj))continue;
    const ei=parts[i].exponent,ej=parts[j].exponent;
    parts.splice(j,1);parts.splice(i,1);
    for(const [p,e] of [[d,ei+ej],[qi,ei],[qj,ej]]){
     const c=cp(p);if(c)constant=gmul(constant,powGaussian(c,e));
     else parts.push({p:integralPrimitive(p),exponent:e});
    }
    changed=true;break outer;
   }
   if(changed){
    // Refinement by pairwise GCD may expose new factors; send every new
    // piece back through both factorization engines, preserving multiplicity.
    const refined=[];
    for(const part of parts){
     const more=splitAll(part.p);
     constant=gmul(constant,powGaussian(more.constant,part.exponent));
     for(const f of more.parts)refined.push({p:f.p,exponent:f.exponent*part.exponent});
    }
    parts=refined;
   }
  }
  // Normalize associates differing by a unit or integer content and combine
  // identical factors as exact powers (not repeated string multiplication).
  const merged=new Map();
  for(const {p,exponent} of parts){
   const canonical=integralPrimitive(p),s=cp(pdivide(p,canonical));
   if(s===null)throw Error('因式标准化不一致');
   constant=gmul(constant,powGaussian(s,exponent));
   const k=polynomialString(canonical);
   const prior=merged.get(k);if(prior)prior.exponent+=exponent;
   else merged.set(k,{p:canonical,exponent});
  }
  const final={constant,parts:[...merged.values()]};
  resultFactorCache.set(cacheKey,final);
  return final;
 }
 function powGaussian(g,e){let x=G1;while(e){if(e%2)x=gmul(x,g);e=Math.floor(e/2);if(e)g=gmul(g,g);}return x;}
 const top=splitAll(numerator),bottom=splitAll(denominator);
 let c=gdiv(top.constant,bottom.constant);
 // Merge scalar factors into primitive numerator and denominator integers.
 let nr=c.re,ni=c.im,dr=c.den;
 const scalarTop=[],scalarBottom=[];
 if(ni===0n){if(nr<0n){scalarTop.push('-1');nr=-nr;}
  const d=gcd(nr,dr);nr/=d;dr/=d;
  if(nr!==1n||!top.parts.length)scalarTop.push(String(nr));
  if(dr!==1n)scalarBottom.push(String(dr));
 }else{scalarTop.push('('+rationalString(c)+')');}
 function tokens(factors,scalars){
  const sorted=[...factors].sort((a,b)=>{
   const ax=a.p.size===1?0:1,bx=b.p.size===1?0:1;
   return ax-bx||polynomialString(a.p).localeCompare(polynomialString(b.p));
  });
  const out=[...scalars];
  for(const {p,exponent} of sorted){
   const f=polynomialString(p,alias),wrapped=p.size>1?'('+f+')':f;
   out.push(wrapped+(exponent>1?'^'+exponent:''));
  }
  return out.length?out:['1'];
 }
 const negative=scalarTop[0]==='-1';if(negative)scalarTop.shift();
 const nt=tokens(top.parts,scalarTop),dt=tokens(bottom.parts,scalarBottom);
 if(negative)nt[0]='-'+nt[0];
 const product=t=>t.join(' * ');
 const dn=product(dt);
 if(dt.length===1&&dt[0]==='1')return nt.length===1&&nt[0].startsWith('(')&&nt[0].endsWith(')')?nt[0].slice(1,-1):product(nt);
 return '('+product(nt)+')/('+dn+')';
}

function stats(){return {symbols:symbolNames.length,monomials:monomialCache.size};}
function reset(){symbolNames.length=0;symbolIndex.clear();monomialCache.clear();monomialCache.set('',[]);mmCache.clear();barNames.clear();factorCache.clear();resultFactorCache.clear();}
return {polynomialGcd,squareRootPolynomial,value,symbol,ga,gcd,add,sub,mul,div,neg,pow,conjugate,simplify:value,factor:factorString,factorString,reset,eq,isZero,string,parse,evalAt,stats,zero:ZERO,one:ONE,i:IMAG,gaussian:{zero:G0,one:G1,i:GI,add:ggadd,mul:gmul,div:gdiv,conjugate:gj}};
});
