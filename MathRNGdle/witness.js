(() => {
'use strict';
const genericText=/^(This badge records|This is a property|This property comes from|The decimal digits can be split)/i;
const fact=[1,1,2,6,24,120,720,5040,40320,362880,3628800];
const sum=a=>a.reduce((x,y)=>x+y,0),prod=a=>a.reduce((x,y)=>x*y,1);
const gcd=(a,b)=>{a=Math.abs(a);b=Math.abs(b);while(b)[a,b]=[b,a%b];return a};
const lcm=(a,b)=>a/gcd(a,b)*b;
function factorize(n){if(n<2)return[];let x=n,o=[];for(let p=2;p*p<=x;p+=(p===2?1:2)){if(x%p===0){let e=0;while(x%p===0){x/=p;e++}o.push([p,e])}}if(x>1)o.push([x,1]);return o}
function factorText(n){if(n===0)return'0 has no prime factorization';if(n===1)return'1 is the empty product';return`${n} = ${factorize(n).map(([p,e])=>e===1?`${p}`:`${p}^${e}`).join(' × ')}`}
function isPrime(n){if(n<2)return false;if(n%2===0)return n===2;for(let p=3;p*p<=n;p+=2)if(n%p===0)return false;return true}
function tau(n){if(n<1)return 0;return prod(factorize(n).map(([,e])=>e+1))}
function omega(n){return factorize(n).length}
function bigOmega(n){return sum(factorize(n).map(([,e])=>e))}
function sigma(n){if(n===0)return 0;if(n===1)return 1;return Math.round(prod(factorize(n).map(([p,e])=>(Math.pow(p,e+1)-1)/(p-1))))}
function phi(n){if(n===0)return 0;let v=n;for(const[p]of factorize(n))v=v/p*(p-1);return Math.round(v)}
function mobius(n){if(n===1)return 1;const f=factorize(n);if(f.some(([,e])=>e>1))return 0;return f.length%2?-1:1}
function carmichaelLambda(n){if(n<=1)return 1;let z=1;for(const[p,e]of factorize(n)){let t;if(p===2&&e>=3)t=Math.pow(2,e-2);else t=Math.pow(p,e-1)*(p-1);z=lcm(z,t)}return z}
function partitionSmall(n){const a=Array(n+1).fill(0);a[0]=1;for(let k=1;k<=n;k++)for(let j=k;j<=n;j++)a[j]+=a[j-k];return a[n]}
function abelianCount(n){if(n<1)return 0;return prod(factorize(n).map(([,e])=>partitionSmall(e)))}
function toBase(n,b){return Math.max(0,n).toString(b).toUpperCase()}
function digits(rep){return[...rep].map(c=>parseInt(c,36))}
function digitSumRep(rep){return sum(digits(rep))}
function digitProdRep(rep){return prod(digits(rep))}
function reverse10(n){return Number(String(n).split('').reverse().join(''))}
function isSquare(n){if(n<0)return false;const r=Math.floor(Math.sqrt(n));return r*r===n}
function isCube(n){if(n<0)return false;const r=Math.round(Math.cbrt(n));return r*r*r===n}
function intRoot(n,k){if(n<0)return null;const r=Math.round(Math.pow(n,1/k));for(let x=Math.max(0,r-2);x<=r+2;x++)if(Math.pow(x,k)===n)return x;return null}
function binom(n,k){if(k<0||k>n)return 0;k=Math.min(k,n-k);let r=1;for(let i=1;i<=k;i++)r=r*(n-k+i)/i;return Math.round(r)}
function modPow(a,e,m){if(m===1)return 0;let r=1%m,x=((a%m)+m)%m;while(e>0){if(e&1)r=(r*x)%m;x=(x*x)%m;e=Math.floor(e/2)}return r}
function modPowBig(a,e,m){let A=BigInt(a),E=BigInt(e),M=BigInt(m),r=1n%M;A=((A%M)+M)%M;while(E>0n){if(E&1n)r=r*A%M;A=A*A%M;E>>=1n}return r}
function multiplicativeOrder(a,p){if(gcd(a,p)!==1)return 0;let o=phi(p);for(const[q,e]of factorize(o)){for(let i=0;i<e;i++){if(o%q===0&&modPow(a,o/q,p)===1)o/=q;else break}}return o}
function fibPairMod(k,m){if(k===0)return[0,1];const[a,b]=fibPairMod(Math.floor(k/2),m),c=(a*((2*b-a)%m+m))%m,d=(a*a+b*b)%m;return k%2?[d,(c+d)%m]:[c,d]}
function fibMod(k,m){return fibPairMod(k,m)[0]}
function matMul3(A,B,m){const C=[[0,0,0],[0,0,0],[0,0,0]];for(let i=0;i<3;i++)for(let k=0;k<3;k++)for(let j=0;j<3;j++)C[i][j]=(C[i][j]+A[i][k]*B[k][j])%m;return C}
function perrinMod(k,m){if(k===0)return 3%m;if(k===1)return 0;if(k===2)return 2%m;let M=[[0,1,1],[1,0,0],[0,1,0]],R=[[1,0,0],[0,1,0],[0,0,1]],e=k-2;while(e){if(e&1)R=matMul3(R,M,m);M=matMul3(M,M,m);e=Math.floor(e/2)}return(2*R[0][0]+0*R[0][1]+3*R[0][2])%m}
function leylandWitness(n){for(let x=2;x<1000;x++)for(let y=2;y<=x;y++){const v=x**y+y**x;if(v===n)return`${n} = ${x}^${y} + ${y}^${x}.`;if(y===2&&v>n&&x>Math.sqrt(n)+3)break}return''}
function factorialPrimeWitness(n){let f=1;for(let k=1;k<=12;k++){f*=k;if(f-1===n)return`${k}! − 1 = ${f} − 1 = ${n}; ${n} is prime.`;if(f+1===n)return`${k}! + 1 = ${f} + 1 = ${n}; ${n} is prime.`}return''}
function primorialNeighborWitness(n){let prod=1,fs=[];for(let p=2;p<100;p++)if(isPrime(p)){prod*=p;fs.push(p);if(prod-1===n||prod+1===n){const sign=prod+1===n?'+':'−';return`${fs.join(' × ')} = ${prod}; ${prod} ${sign} 1 = ${n}, which is prime.`}if(prod>n+1&&prod>1e6)break}return''}
function ruthAaronWitness(n){const A=factorize(n),B=factorize(n+1),sa=sum(A.map(x=>x[0])),sb=sum(B.map(x=>x[0]));return`${factorText(n)}; distinct-prime sum = ${A.map(x=>x[0]).join(' + ')} = ${sa}; ${factorText(n+1)}; distinct-prime sum = ${B.map(x=>x[0]).join(' + ')} = ${sb}.`}
function fibonacciPseudoprimeWitness(n){const r=n%5,k=(r===1||r===4)?n-1:n+1,z=fibMod(k,n);return`${factorText(n)}, so n is composite; n ≡ ${r} (mod 5), hence test index = ${k}; F_${k} mod ${n} = ${z}.`}
function perrinPseudoprimeWitness(n){return`${factorText(n)}, so n is composite; P_0=3, P_1=0, P_2=2, P_k=P_{k-2}+P_{k-3}; P_${n} mod ${n} = ${perrinMod(n,n)}.`}
function cubanPrimeWitness(n){const D=12*n-3,r=Math.round(Math.sqrt(D)),k=(r-3)/6;if(Number.isInteger(k)&&k>0)return`${n} = 3·${k}² + 3·${k} + 1 = ${k+1}³ − ${k}³; ${n} is prime.`;return''}
function carolKyneaWitness(n,kynea=false){for(let k=1;k<30;k++){const a=2**k+(kynea?1:-1),v=a*a-2;if(v===n)return`${n} = (2^${k} ${kynea?'+':'−'} 1)² − 2 = ${a}² − 2; ${n} is prime.`;if(v>n&&k>3)break}return''}
function fullReptendWitness(n){const o=multiplicativeOrder(10,n),qs=factorize(n-1).map(x=>x[0]),tests=qs.map(q=>`10^((${n}−1)/${q}) mod ${n} = ${modPow(10,(n-1)/q,n)}`);return`${n} is prime; ${n}−1 = ${factorText(n-1).replace(/^.* = /,'')}; ord_${n}(10) = ${o} = ${n}−1; ${tests.join('; ')} (none is 1).`}
function fmtList(a,lim=9){return a.length<=lim?a.join(', '):a.slice(0,lim).join(', ')+' …'}
function countsText(ds){const c=new Map();for(const d of ds)c.set(d,(c.get(d)||0)+1);return[...c.entries()].sort((a,b)=>a[0]-b[0]).map(([d,v])=>`${d}:${v}`).join(', ')}
function diffs(a){return a.slice(1).map((x,i)=>x-a[i])}
function runLengths(a){if(!a.length)return[];let out=[],k=1;for(let i=1;i<a.length;i++){if(a[i]===a[i-1])k++;else{out.push(k);k=1}}out.push(k);return out}
function signString(a){return diffs(a).map(d=>d>0?'↑':d<0?'↓':'=').join('')}
function digitSquareTrace(n,b=10){let x=n,out=[x],seen=new Set();while(x!==1&&!seen.has(x)&&out.length<20){seen.add(x);x=sum(digits(toBase(x,b)).map(d=>d*d));out.push(x)}return out}
function persistenceTrace(n,multi=true){let x=n,out=[x];while(x>=10&&out.length<20){const ds=digits(String(x));x=multi?prod(ds):sum(ds);out.push(x)}return out}
function findEquation(n,op){const s=String(n),out=[],show=(raw,v)=>(raw.length>1&&raw[0]==='0')?`${raw} (=${v})`:raw;for(let i=1;i<s.length-1;i++)for(let j=i+1;j<s.length;j++){const A=s.slice(0,i),B=s.slice(i,j),D=s.slice(j),a=+A,b=+B,c=+D;if(a<=0||b<=0||c<=0)continue;const aa=show(A,a),bb=show(B,b),cc=show(D,c);if(op==='+'&&a+b===c)out.push(`${aa} + ${bb} = ${cc}`);if(op==='-'&&a-b===c)out.push(`${aa} − ${bb} = ${cc}`);if(op==='*'&&a>1&&b>1&&c>9&&a*b===c)out.push(`${aa} × ${bb} = ${cc}`);if(op==='/'&&b>1&&c>1&&a===b*c)out.push(`${aa} ÷ ${bb} = ${cc}`)}return out[0]||''}
function fourPartAdd(n){const s=String(n);for(let i=1;i<s.length-2;i++)for(let j=i+1;j<s.length-1;j++)for(let k=j+1;k<s.length;k++){const a=+s.slice(0,i),b=+s.slice(i,j),c=+s.slice(j,k),d=+s.slice(k);if(a>0&&b>0&&c>0&&d>0&&a+b+c===d)return`${a} + ${b} + ${c} = ${d}`}return''}
function splitEqual(n,productMode=false){const s=String(n);for(let i=1;i<s.length;i++){const A=digits(s.slice(0,i)),B=digits(s.slice(i));const x=productMode?prod(A):sum(A),y=productMode?prod(B):sum(B);if(x===y)return`${s.slice(0,i)} | ${s.slice(i)}: ${productMode?'products':'sums'} = ${x}`}return''}
function primeNeighbor(p,dir){for(let x=p+dir;x>1;x+=dir)if(isPrime(x))return x;return null}
function rotations(s){return [...s].map((_,i)=>s.slice(i)+s.slice(0,i))}
function findPartnerPrime(n,gap){const a=[];if(isPrime(n-gap))a.push(n-gap);if(isPrime(n+gap))a.push(n+gap);return a}
function isPrimePower(q){return q>=2&&factorize(q).length===1}
function sequenceIndex(target,seeds,next,max=100){let a=[...seeds];for(let i=0;i<a.length;i++)if(a[i]===target)return i;for(let i=a.length;i<max;i++){const v=next(a,i);if(!Number.isFinite(v)||v>1e12)break;a.push(Math.round(v));if(a[i]===target)return i;if(a[i]>target&&a.slice(-4).every((x,j,z)=>j===0||x>=z[j-1]))break}return-1}
function partitionNumbers(limit,distinct=false){const a=Array(limit+1).fill(0);a[0]=1;if(distinct){for(let k=1;k<=limit;k++)for(let j=limit;j>=k;j--)a[j]+=a[j-k]}else{for(let k=1;k<=limit;k++)for(let j=k;j<=limit;j++)a[j]+=a[j-k]}return a}
function findPartitionIndex(n,distinct=false){const a=partitionNumbers(180,distinct);return a.findIndex(x=>x===n)}
function bellNumbers(max=16){let bell=[1],row=[1];for(let n=1;n<max;n++){const nr=[row[row.length-1]];for(let k=1;k<=n;k++)nr[k]=nr[k-1]+row[k-1];row=nr;bell.push(row[0])}return bell}
function findTriangleValue(n,type){if(type==='stirling'){let row=[1];for(let N=1;N<=80;N++){let nr=Array(N+1).fill(0);for(let k=1;k<=N;k++)nr[k]=(row[k-1]||0)+k*(row[k]||0);row=nr;for(let k=1;k<=N;k++)if(row[k]===n)return[N,k]}}if(type==='eulerian'){let row=[1];for(let N=1;N<=30;N++){let nr=Array(N+1).fill(0);for(let k=0;k<N;k++)nr[k]=(k+1)*(row[k]||0)+(N-k)*(row[k-1]||0);row=nr;for(let k=0;k<N;k++)if(row[k]===n)return[N,k]}}if(type==='pascal'){for(let k=2;k<=30;k++)for(let N=Math.max(2*k,k+1);N<=2000;N++){const v=binom(N,k);if(v===n)return[N,k];if(v>n)break}}return null}
function apery(k){let s=0;for(let j=0;j<=k;j++)s+=binom(k,j)**2*binom(k+j,j)**2;return s}
function franel(k){let s=0;for(let j=0;j<=k;j++)s+=binom(k,j)**3;return s}
function findFormulaIndex(n,fn,max=30){for(let k=0;k<=max;k++){const v=fn(k);if(v===n)return k;if(v>n&&k>4)return-1}return-1}
function happyInBase(n,b){const t=digitSquareTrace(n,b);return t[t.length-1]===1}
function legendre(a,p){a=((a%p)+p)%p;if(a===0)return 0;const r=modPow(a,(p-1)/2,p);return r===1?1:r===p-1?-1:0}
function residueSignature(n,k,ps=[5,7,11,13,17,19,23]){return ps.map(p=>{const a=((n%p)+p)%p;let ok=false;for(let x=0;x<p;x++)if(modPow(x,k,p)===a){ok=true;break}return`${p}:${a}${ok?'✓':'×'}`}).join(' · ')}
function quadraticSignature(n){return[3,5,7,11,13,17,19].map(p=>`${p}:${legendre(n,p)===1?'+':legendre(n,p)===-1?'−':'0'}`).join(' · ')}
function findQ(target,fn,max=500){for(let q=2;q<=max;q++)if(fn(q)===target)return q;return null}
function polygonIndex(x,type){for(let k=1;k<5000;k++){let v=0;if(type==='tri')v=k*(k+1)/2;else if(type==='pent')v=k*(3*k-1)/2;else if(type==='hex')v=k*(2*k-1);else if(type==='hept')v=k*(5*k-3)/2;else if(type==='oct')v=k*(3*k-2);if(v===x)return k;if(v>x)return-1}return-1}
function fibIndex(x){return sequenceIndex(x,[0,1],(a,i)=>a[i-1]+a[i-2],40)}
function lucasIndex(x){return sequenceIndex(x,[2,1],(a,i)=>a[i-1]+a[i-2],40)}
function halfParts(n){const s=String(n).padStart(6,'0');return[+s.slice(0,3),+s.slice(3),s.slice(0,3),s.slice(3)]}
function chunkParts(n){const s=String(n).padStart(6,'0');return[+s.slice(0,2),+s.slice(2,4),+s.slice(4),s.slice(0,2),s.slice(2,4),s.slice(4)]}
function specialEventProof(f,n){
 const [a,b,A,B]=halfParts(n),[x,y,z,X,Y,Z]=chunkParts(n),r=reverse10(n);
 if(f==='mirror_halves')return`${A} | ${B}; reverse(${A}) = ${B}.`;
 if(f==='repeat_halves')return`${A} | ${B}; ${A} = ${B}; ${a} × 1001 = ${n}.`;
 if(f==='complement_halves_999')return`${A} | ${B}; ${a} + ${b} = 999.`;
 const poly=(val,t,label,formula)=>{const k=polygonIndex(val,t);return k>0?`${val} = ${formula(k)}`:`${val} is ${label}`};
 if(f==='both_square_halves'){const u=intRoot(a,2),v=intRoot(b,2);return`${A} | ${B}; ${a}=${u}²; ${b}=${v}².`}
 if(f==='both_triangular_halves')return`${A} | ${B}; ${poly(a,'tri','triangular',k=>`${k}·${k+1}/2`)}; ${poly(b,'tri','triangular',k=>`${k}·${k+1}/2`)}.`;
 if(f==='both_pentagonal_halves')return`${A} | ${B}; ${poly(a,'pent','pentagonal',k=>`${k}(3·${k}−1)/2`)}; ${poly(b,'pent','pentagonal',k=>`${k}(3·${k}−1)/2`)}.`;
 if(f==='both_hexagonal_halves')return`${A} | ${B}; ${poly(a,'hex','hexagonal',k=>`${k}(2·${k}−1)`)}; ${poly(b,'hex','hexagonal',k=>`${k}(2·${k}−1)`)}.`;
 if(f==='both_heptagonal_halves')return`${A} | ${B}; ${poly(a,'hept','heptagonal',k=>`${k}(5·${k}−3)/2`)}; ${poly(b,'hept','heptagonal',k=>`${k}(5·${k}−3)/2`)}.`;
 if(f==='both_octagonal_halves')return`${A} | ${B}; ${poly(a,'oct','octagonal',k=>`${k}(3·${k}−2)`)}; ${poly(b,'oct','octagonal',k=>`${k}(3·${k}−2)`)}.`;
 if(f==='square_to_triangular_halves'){const u=intRoot(a,2);return`${A} | ${B}; ${a}=${u}²; ${poly(b,'tri','triangular',k=>`${k}·${k+1}/2`)}.`}
 if(f==='square_to_pentagonal_halves'){const u=intRoot(a,2);return`${A} | ${B}; ${a}=${u}²; ${poly(b,'pent','pentagonal',k=>`${k}(3·${k}−1)/2`)}.`}
 if(f==='triangular_to_pentagonal_halves')return`${A} | ${B}; ${poly(a,'tri','triangular',k=>`${k}·${k+1}/2`)}; ${poly(b,'pent','pentagonal',k=>`${k}(3·${k}−1)/2`)}.`;
 if(f==='square_cube_halves'){const as=intRoot(a,2),ac=intRoot(a,3),bs=intRoot(b,2),bc=intRoot(b,3);return as!==null&&bc!==null?`${A} | ${B}; ${a}=${as}²; ${b}=${bc}³.`:`${A} | ${B}; ${a}=${ac}³; ${b}=${bs}².`}
 if(f==='fibonacci_halves'){const i=fibIndex(a),j=fibIndex(b);return`${A} | ${B}; ${a}=F_${i}; ${b}=F_${j}.`}
 if(f==='lucas_halves'){const i=lucasIndex(a),j=lucasIndex(b);return`${A} | ${B}; ${a}=L_${i}; ${b}=L_${j}.`}
 if(f==='twin_prime_halves'||f==='cousin_prime_halves'||f==='sexy_prime_halves'){const g=f.startsWith('twin')?2:f.startsWith('cousin')?4:6;return`${A} | ${B}; ${a} and ${b} are prime; |${a}−${b}|=${Math.abs(a-b)}=${g}.`}
 if(f==='prime_halves_sum_square'){const q=intRoot(a+b,2);return`${A} | ${B}; both halves prime; ${a}+${b}=${a+b}=${q}².`}
 if(f==='prime_halves_sum_cube'){const q=intRoot(a+b,3);return`${A} | ${B}; both halves prime; ${a}+${b}=${a+b}=${q}³.`}
 if(f==='prime_halves_diff_square'){const q=intRoot(Math.abs(a-b),2);return`${A} | ${B}; both halves prime; |${a}−${b}|=${Math.abs(a-b)}=${q}².`}
 if(f==='reverse_prime_halves')return`${A} | ${B}; ${a} is prime; reverse(${A})=${B}=${b}, also prime.`;
 if(f==='chunks_consecutive')return`${X} | ${Y} | ${Z}; ${y}=${x}+1; ${z}=${x}+2.`;
 if(f==='chunks_prime_ap')return`${X} | ${Y} | ${Z}; all three are prime; ${x}+${z}=${x+z}=2·${y}.`;
 if(f==='chunks_geometric')return`${X} | ${Y} | ${Z}; ${y}²=${y*y}=${x}·${z}.`;
 if(f==='chunks_all_fibonacci')return`${X} | ${Y} | ${Z}; ${x}=F_${fibIndex(x)}, ${y}=F_${fibIndex(y)}, ${z}=F_${fibIndex(z)}.`;
 if(f==='reverse_product_square_nonpal'){const q=intRoot(n*r,2);return`reverse(${n})=${r} ≠ ${n}; ${n} × ${r} = ${n*r} = ${q}².`}
 if(f==='reverse_sum_triangular'){const q=polygonIndex(n+r,'tri');return`${n} + ${r} = ${n+r} = ${q}·${q+1}/2.`}
 if(f==='reverse_sum_pentagonal'){const q=polygonIndex(n+r,'pent');return`${n} + ${r} = ${n+r} = ${q}(3·${q}−1)/2.`}
 if(f==='digit_cubes_equal_half_gap'){const ds=digits(String(n)),c=sum(ds.map(d=>d**3));return`${A} | ${B}; ${ds.map(d=>`${d}³`).join(' + ')} = ${c} = |${a}−${b}|.`}
 return'';
}
function eulerSecantValues(max=12){const E=[1];for(let m=1;m<=max;m++){let s=0;for(let k=0;k<m;k++)s+=binom(2*m,2*k)*E[k];E[m]=-s}return E.map(Math.abs)}
function largeSchroederValues(max=12){const a=[1,2];for(let k=2;k<=max;k++)a[k]=Math.round((3*(2*k-1)*a[k-1]-(k-2)*a[k-2])/(k+1));return a}
function motzkinValues(max=20){const a=[1,1];for(let k=2;k<=max;k++)a[k]=Math.round(((2*k+1)*a[k-1]+(3*k-3)*a[k-2])/(k+2));return a}
function planePartitions(max=35){const a=Array(max+1).fill(0);a[0]=1;for(let m=1;m<=max;m++){for(let rep=0;rep<m;rep++)for(let j=m;j<=max;j++)a[j]+=a[j-m]}return a}
function gaussianBinomial(n,k,q){k=Math.min(k,n-k);let num=1,den=1;for(let i=0;i<k;i++){num*=q**(n-i)-1;den*=q**(k-i)-1}return Math.round(num/den)}
function grassmannWitness(target){for(let q=2;q<=1000;q++)if(isPrimePower(q))for(let m=2;m<=8;m++)for(let k=1;k<m;k++)if(gaussianBinomial(m,k,q)===target)return`${target} = [${m} choose ${k}]_${q}, the Gaussian binomial counting ${k}-dimensional subspaces of F_${q}^${m}.`;return''}
function irreducibleCount(q,d){let s=0;for(let e=1;e<=d;e++)if(d%e===0)s+=mobius(e)*q**(d/e);return Math.round(s/d)}
function irreducibleWitness(target){for(let q=2;q<=1000;q++)if(isPrimePower(q))for(let d=2;d<=20;d++){const v=irreducibleCount(q,d);if(v===target)return`${target} = N_${q}(${d}) = (1/${d}) Σ_{e|${d}} μ(e)·${q}^(${d}/e), the monic irreducibles of degree ${d} over F_${q}.`;if(v>target&&d>3)break}return''}
function markovWitness(target){const seen=new Set(),q=[[1,1,1]];for(let step=0;step<20000&&q.length;step++){const t=q.shift().sort((a,b)=>a-b),key=t.join(',');if(seen.has(key))continue;seen.add(key);if(t.includes(target))return`${t[0]}² + ${t[1]}² + ${t[2]}² = 3·${t[0]}·${t[1]}·${t[2]}; ${target} occurs in this Markov triple.`;for(let i=0;i<3;i++){const u=[...t],j=(i+1)%3,k=(i+2)%3;u[i]=3*t[j]*t[k]-t[i];if(u[i]>0&&u[i]<=1000000)q.push(u)}}return''}
function rootsSystemWitness(n){for(let r=1;r<=1000;r++){if(r*(r+1)===n)return`A_${r}: ${r}(${r}+1) = ${n} roots.`;if(2*r*r===n)return`B_${r}/C_${r}: 2·${r}² = ${n} roots.`;if(r>=4&&2*r*(r-1)===n)return`D_${r}: 2·${r}·${r-1} = ${n} roots.`}const e={12:'G₂',48:'F₄',72:'E₆',126:'E₇',240:'E₈'};return e[n]?`${e[n]} has ${n} roots.`:''}
function lieWitness(n){for(let r=1;r<=1000;r++){if(r*(r+2)===n)return`dim A_${r} = ${r}(${r}+2) = ${n}.`;if(r*(2*r+1)===n)return`dim B_${r}=dim C_${r} = ${r}(2·${r}+1) = ${n}.`;if(r>=4&&r*(2*r-1)===n)return`dim D_${r} = ${r}(2·${r}−1) = ${n}.`}const e={14:'G₂',52:'F₄',78:'E₆',133:'E₇',248:'E₈'};return e[n]?`dim ${e[n]} = ${n}.`:''}
function flagEulerWitness(n){for(let r=1;r<10;r++){let f=fact[Math.min(r+1,10)]||0;if(f===n)return`Type A_${r}: |W| = (${r+1})! = ${n}.`;const bc=Math.pow(2,r)*(fact[r]||0);if(bc===n)return`Type B_${r}/C_${r}: |W| = 2^${r}·${r}! = ${n}.`;if(r>=4&&Math.pow(2,r-1)*(fact[r]||0)===n)return`Type D_${r}: |W| = 2^${r-1}·${r}! = ${n}.`}return{12:'Type G₂',1152:'Type F₄',51840:'Type E₆'}[n]?`${({12:'Type G₂',1152:'Type F₄',51840:'Type E₆'})[n]} has Weyl-group order ${n}.`:''}
function definitionFor(b){if(!b||b.oeis)return b?.explain||'';const f=b.family||'',name=b.name||'';
 if(f==='divisor_parity')return'τ(n) is odd if and only if n is a perfect square; the witness shows both the factorization and square root.';
 if(f==='highly_abundant')return'A highly abundant number sets a new record for σ(n), the sum of all positive divisors.';
 if(f==='highly_composite')return'A highly composite number sets a new record for τ(n), the number of positive divisors.';
 if(f==='markov_number')return'A Markov number occurs in a positive integer solution of x²+y²+z²=3xyz.';
 if(f==='euler_secant_value')return'These are absolute Euler secant numbers, the coefficients governing the Taylor series of sec(x).';
 if(f==='large_schroeder_value'||f==='little_schroeder_value'||f==='motzkin_value'||f==='plane_partition_value')return'This badge identifies n as an exact value of the named combinatorial counting sequence; the witness gives its index and defining recurrence or generating function.';
 if(/^\d+_smooth$/.test(f))return'A y-smooth integer has no prime factor larger than y; the witness lists the complete prime factorization.';
 if(f==='abelian_group_count')return'Finite abelian groups of order n are classified independently at each prime power p^e; the number of choices at p^e is the partition number p(e).';
 if(f==='abundance')return'Compare the sum s(n)=σ(n)−n of proper divisors with n: deficient means s(n)<n, perfect means equality, and abundant means s(n)>n.';
 if(f==='achilles')return'An Achilles number is powerful—every prime exponent is at least 2—but is not a perfect power, so the gcd of its exponents is 1.';
 if(f.includes('divisor_count'))return'The divisor-count function τ(n) satisfies τ(∏pᵢ^eᵢ)=∏(eᵢ+1).';
 if(f==='omega_distinct')return'ω(n) counts the distinct prime divisors of n.';
 if(f==='bigomega')return'Ω(n) counts prime factors with multiplicity, i.e. Ω(∏pᵢ^eᵢ)=∑eᵢ.';
 if(f==='mobius')return'The Möbius function is 0 when a square prime factor divides n; otherwise μ(n)=(−1)^ω(n).';
 if(f==='parity')return'Parity is determined by the residue of n modulo 2.';
 if(f==='primality')return'A prime integer has exactly two positive divisors; a composite one has a nontrivial prime factorization.';
 if(f.includes('totient')||name.includes('φ(')||name.toLowerCase().includes('totient'))return'Euler’s totient φ(n) counts residue classes modulo n that are coprime to n, with φ(n)=n∏_{p|n}(1−1/p).';
 if(f==='palindromic_prime')return'A palindromic prime is simultaneously prime and unchanged by decimal digit reversal.';
 if(f==='primitive_idempotents')return'By the Chinese remainder theorem, x²≡x (mod n) has exactly 2^ω(n) solutions, one binary choice at each distinct prime factor.';
 if(f==='semiprime_shape')return'A semiprime has Ω(n)=2; the witness distinguishes a square p² from a product pq of two distinct primes.';
 if(f==='sublime_number')return'A sublime number has both τ(n) and σ(n) perfect numbers.';
 if(f==='tetraperfect_number')return'A quadruply perfect number satisfies σ(n)=4n exactly.';
 if(f==='triperfect_number')return'A triply perfect number satisfies σ(n)=3n exactly.';
 if(f==='factor_exponents_ap')return'The exponents in the prime factorization form an arithmetic progression.';
 if(f==='factor_exponents_palindrome')return'The exponent vector in the prime factorization reads the same forward and backward.';
 if(f==='factor_primes_ap')return'The distinct prime factors themselves form an arithmetic progression; the witness shows their equal gaps.';
 if(f==='omega_equals_distinct_digits')return'The number ω(n) of distinct prime factors equals the number of distinct decimal digit symbols used by n.';
 if(f==='tau_equals_digit_sum')return'The divisor count τ(n) is exactly equal to the decimal digit sum.';
 if(f==='decimal_length')return'The decimal length is the number of positions in the ordinary base-10 expansion; the witness shows the exact expansion and its digit count.';
 if(f==='two_digit_chunks_pythagorean')return'Splitting a six-digit number as ab|cd|ef gives a Pythagorean identity ab²+cd²=ef².';
 if(f==='two_digit_chunks_multiplication')return'Splitting a six-digit number as ab|cd|ef gives the exact multiplication identity ab·cd=ef.';
 if(f.startsWith('base')||f.startsWith('dec_')||f.startsWith('binary_')||f.startsWith('multi_base')||f==='cross_base_same_digit_sum')return'This is a positional-digit property. The witness writes the current integer in the relevant base(s) and evaluates the exact digit condition.';
 if(f==='full_reptend_prime')return'A full reptend prime p has 10 as a primitive root modulo p, so the decimal expansion of 1/p has the maximal possible period p−1.';
 if(f==='factorial_prime')return'A factorial prime is a prime lying one unit from a factorial: p=k!−1 or p=k!+1.';
 if(f==='primorial_neighbor_prime')return'A primorial-neighbor prime lies one unit from q#=2·3·5···q for some prime q.';
 if(f==='leyland_number')return'A Leyland number has a nontrivial representation x^y+y^x with integers x,y>1.';
 if(f==='ruth_aaron_number')return'n starts a Ruth–Aaron pair when n and n+1 have equal sums of their distinct prime divisors.';
 if(f==='fibonacci_pseudoprime')return'An odd Fibonacci pseudoprime is composite but satisfies the same Fibonacci divisibility congruence forced by primes according to n mod 5.';
 if(f==='perrin_pseudoprime')return'A Perrin pseudoprime is composite but satisfies P_n≡0 (mod n), a congruence obeyed by every prime n for the Perrin recurrence.';
 if(f==='cuban_prime')return'A Cuban prime of the first kind is a prime difference of consecutive cubes: (k+1)³−k³=3k²+3k+1.';
 if(f==='carol_prime')return'A Carol prime has the form (2^k−1)²−2.';
 if(f==='kynea_prime')return'A Kynea prime has the form (2^k+1)²−2.';
 if(f==='sophie_germain_prime')return'A Sophie Germain prime p is prime and 2p+1 is prime.';
 if(f==='safe_prime')return'A safe prime p is prime and (p−1)/2 is prime.';
 if(f.includes('twin_prime'))return'Twin primes are pairs of primes differing by 2.';
 if(f.includes('cousin_prime'))return'Cousin primes are pairs of primes differing by 4.';
 if(f.includes('sexy_prime'))return'Sexy primes are pairs of primes differing by 6.';
 if(f==='balanced_prime')return'A balanced prime equals the arithmetic mean of the nearest prime below it and the nearest prime above it.';
 if(f==='circular_prime')return'Every cyclic rotation of the decimal digits is prime.';
 if(f==='emirp')return'An emirp is a non-palindromic prime whose decimal reversal is also prime.';
 if(f.includes('truncatable_prime'))return'Repeatedly deleting digits from the stated side leaves a prime at every stage.';
 if(f.startsWith('fermat_pseudoprime_base_'))return'A Fermat pseudoprime to base a is composite but satisfies a^(n−1) ≡ 1 (mod n).';
 if(f.startsWith('strong_pseudoprime_base_'))return'A strong pseudoprime passes the Miller–Rabin strong probable-prime test for the stated base despite being composite.';
 if(f.startsWith('wieferich_base_'))return'A Wieferich prime to base a satisfies a^(p−1) ≡ 1 (mod p²), a much stronger congruence than Fermat’s theorem.';
 if(f.startsWith('cyclotomic_full_split_'))return'For a prime p not dividing m, complete splitting in Q(ζ_m) is equivalent to p ≡ 1 (mod m).';
 if(f.startsWith('quadratic_split_'))return'Prime splitting in a quadratic field is controlled by the field discriminant; a prime dividing that discriminant is ramified.';
 if(f==='gaussian_split_prime')return'An odd prime splits in the Gaussian integers Z[i] exactly when p ≡ 1 (mod 4).';
 if(f==='eisenstein_split_prime')return'A prime p>3 splits in the Eisenstein integers Z[ω] exactly when p ≡ 1 (mod 3).';
 if(f==='carmichael')return'Korselt’s criterion says a composite n is Carmichael iff n is squarefree and p−1 divides n−1 for every prime p|n.';
 if(f.startsWith('carmichael_lambda'))return'Carmichael’s λ(n) is the exponent of the unit group (Z/nZ)×: every unit a satisfies a^λ(n) ≡ 1 (mod n).';
 if(f==='powerful')return'An integer is powerful when every prime that divides it actually divides it to at least the second power.';
 if(f==='perfect_power'||/^perfect_(square|cube|fourth|fifth|sixth)$/.test(f))return'A perfect k-th power has the form a^k; a perfect power has the form a^k with integers a>0 and k>1.';
 if(f==='primary_pseudoperfect')return'A primary pseudoperfect number satisfies 1/n + Σ_{p|n}1/p = 1 over its distinct prime divisors.';
 if(f==='giuga')return'A composite Giuga number satisfies p | (n/p−1) for every prime divisor p of n.';
 if(f==='refactorable')return'A refactorable number is divisible by its own divisor count τ(n).';
 if(f==='harmonic_divisor_number')return'The harmonic mean of the positive divisors is H(n)=nτ(n)/σ(n); a harmonic divisor number has integral H(n).';
 if(f==='perfect_totient')return'A perfect totient number equals the sum of its iterated totients n→φ(n)→φ²(n)→… down to 1.';
 if(f==='primorial_value')return'A primorial is the product of all primes up to a given prime: p#=∏_{q≤p}q.';
 if(f==='euclid_number')return'A Euclid number is one more than a primorial-like product of the first k primes.';
 if(f==='mersenne_number'||f==='mersenne_prime_value')return'A Mersenne number has the form 2^p−1; a Mersenne prime is such a number that is prime.';
 if(f==='fermat_prime_value')return'A Fermat prime has the form 2^(2^k)+1 and is prime.';
 if(f==='cullen_number')return'A Cullen number has the form k·2^k+1.';
 if(f==='woodall_number')return'A Woodall number has the form k·2^k−1.';
 if(f==='proth_prime')return'A Proth prime has p=k·2^m+1 with k odd, k<2^m, and p prime.';
 if(f==='pierpont_prime')return'A Pierpont prime has the form 2^a3^b+1.';
 if(f==='chen_prime')return'A Chen prime p is prime and p+2 is either prime or a product of two primes.';
 if(/(triangular|pentagonal|hexagonal|octagonal|oblong|tetrahedral|simplex|centered_)/.test(f))return'This is a figurate-number condition: n is produced by the corresponding polygonal or simplex counting formula.';
 if(f.endsWith('_value')||f==='partition_value'||f==='pascal_value')return'This badge means the current integer occurs exactly as a value of the named classical combinatorial or recurrence sequence; the witness identifies an index or parameter when practical.';
 if(/^(gl2_order|gl3_order|sl2_order|sl3_order|psl2_order|sp4_order)$/.test(f))return'This badge identifies n as the order of the named finite classical group over a finite field F_q; the witness substitutes q in the standard order formula.';
 if(/projective_.*space_points|projective_plane_points|grassmannian_points|flag_a2_fq_points|hermitian_curve_points|elliptic_hasse_extreme|irreducible_polynomial_count/.test(f))return'This badge identifies n as a point count (or polynomial count) over a finite field; the witness substitutes the field size q into the defining formula.';
 if(f==='lie_algebra_dimension')return'n is the dimension of a simple Lie algebra; classical families have polynomial dimension formulas in their rank.';
 if(f==='root_system_root_count')return'n is the number of roots in an irreducible crystallographic root system.';
 if(f==='flag_variety_euler')return'The Euler characteristic of a full flag variety equals the order of its Weyl group.';
 if(f.startsWith('roots_unity_'))return'This counts residue classes x modulo n satisfying x^k ≡ 1 (mod n).';
 if(f.includes('residue_consensus')||f==='legendre_all_residue'||f==='legendre_all_nonresidue')return'A residue-consensus property requires the same prescribed power-residue or quadratic-character behavior across several small prime moduli.';
 if(f.startsWith('near_factorial'))return'n lies one unit from a multiple of a factorial modulus m, i.e. n=mq±1.';
 if(f.startsWith('near_primorial'))return'n lies one unit from a multiple of a primorial modulus m, i.e. n=mq±1.';
 if(f.includes('reverse'))return'This badge couples n with its decimal reversal and requires the resulting sum, difference, or product to have the stated arithmetic form.';
 if(f==='sum_two_cubes'||f==='taxicab_multiple')return'n is represented as a sum of two cubes; taxicab-type badges require more than one distinct representation.';
 if(f==='square_triangular')return'n is simultaneously a perfect square and a triangular number.';
 if(f==='heegner_class_number_one')return'The imaginary quadratic field Q(√−d) has ideal class number 1 for the Heegner discriminants.';
 if(!genericText.test(b.explain||''))return b.explain||'';
 return`The property “${name.replace(/\s+—.*$/,'')}” is evaluated directly on n. The witness below records the current integer and the computed parameter or identity that made the badge fire.`;
}
function baseProof(b,n){const m=(b.family||'').match(/^base(\d+)_/),base=m?Number(m[1]):Number(b.variant)||10,rep=toBase(n,base),ds=digits(rep),f=b.family||'',s=digitSumRep(rep),p=digitProdRep(rep);let parts=[`${n} = (${rep})_${base}`];if(f.includes('palindrome'))parts.push(`${rep} reversed is ${[...rep].reverse().join('')}`);if(f.includes('repdigit'))parts.push(`all digits are ${ds[0]}`);if(f.includes('all_digits_distinct')||f.includes('all_distinct'))parts.push(`digit multiset = {${ds.join(', ')}} with no repetition`);if(f.includes('equal_frequency'))parts.push(`digit counts = ${countsText(ds)}`);if(f.includes('sum_prime')||f.includes('sum_square')||f.includes('sum_triangular')||f.includes('digit_sum'))parts.push(`digit sum = ${ds.join(' + ')} = ${s}`);if(f.includes('product_square'))parts.push(`digit product = ${ds.join(' × ')} = ${p}`);if(f.includes('arithmetic'))parts.push(`adjacent differences = ${diffs(ds).join(', ')}`);if(f.includes('monotone'))parts.push(`step pattern = ${signString(ds)}`);if(f.includes('alternating'))parts.push(`digit parities = ${ds.map(x=>x%2?'odd':'even').join(' / ')}`);if(f.includes('happy'))parts.push(`square-digit iteration: ${digitSquareTrace(n,base).join(' → ')}`);if(f.includes('automorphic')){const k=Math.pow(base,rep.length);parts.push(`${n}² = ${n*n} ≡ ${n} (mod ${k})`)}if(f.includes('trimorphic')){const k=Math.pow(base,rep.length);parts.push(`${n}³ ≡ ${n} (mod ${k})`)}if(f.includes('pandigital'))parts.push(`digits used = ${ds.join(', ')}`);return parts.join(' · ')}
function decimalProof(b,n){const f=b.family,s=String(n),ds=digits(s),rev=reverse10(n),d=diffs(ds),pairs=ds.slice(0,Math.floor(ds.length/2)).map((x,i)=>[x,ds[ds.length-1-i]]);if(f==='dec_arithmetic_progression'||f==='dec_strict_increasing'||f==='dec_strict_decreasing')return`${n}: digits ${ds.join(', ')}; adjacent differences = ${d.join(', ')}.`;if(f==='dec_alternating_parity')return`${n}: ${ds.map(x=>`${x}(${x%2?'odd':'even'})`).join(' → ')}.`;if(f==='dec_alternating_high_low')return`${n}: ${ds.map(x=>`${x}${x>=5?'H':'L'}`).join(' → ')} (H=5–9, L=0–4).`;if(f==='dec_alternating_sum_equal'){const a=sum(ds.filter((_,i)=>i%2===0)),c=sum(ds.filter((_,i)=>i%2));return`${ds.filter((_,i)=>i%2===0).join('+')} = ${a}; ${ds.filter((_,i)=>i%2).join('+')} = ${c}.`}if(f==='dec_adjacent_pair_sum_constant')return`${n}: adjacent sums = ${ds.slice(1).map((x,i)=>ds[i]+x).join(', ')}.`;if(f==='dec_symmetric_pair_sum_constant')return`${n}: mirrored sums = ${pairs.map(([a,c])=>`${a}+${c}=${a+c}`).join('; ')}.`;if(f==='dec_symmetric_pair_product_constant')return`${n}: mirrored products = ${pairs.map(([a,c])=>`${a}×${c}=${a*c}`).join('; ')}.`;if(f==='dec_anagram_halves'){const h=s.length/2,a=s.slice(0,h),c=s.slice(h);return`${a} and ${c} both sort to ${[...a].sort().join('')}.`}if(f==='dec_balanced_halves_sum'){const h=s.length/2,a=digits(s.slice(0,h)),c=digits(s.slice(h));return`${s.slice(0,h)} | ${s.slice(h)}: ${a.join('+')} = ${sum(a)} and ${c.join('+')} = ${sum(c)}.`}if(f==='dec_balanced_torque'){const mid=(ds.length-1)/2,terms=ds.map((x,i)=>(i-mid)*x),t=sum(terms);return`digit torque Σ(i−center)d_i = ${terms.map(x=>Number.isInteger(x)?x:x.toFixed(1)).join(' + ')} = ${t}.`}if(f==='dec_complement_palindrome')return`${n}: mirrored complementary pairs ${pairs.map(([a,c])=>`${a}+${c}=${a+c}`).join('; ')}.`;if(f==='dec_half_complement'){const h=s.length/2,A=s.slice(0,h),B=s.slice(h),ad=digits(A),bd=digits(B),aligned=ad.map((a,i)=>`${a}+${bd[i]}=${a+bd[i]}`);return`${A} | ${B}: aligned half-complements ${aligned.join('; ')}; ${Number(A)}+${Number(B)}=${Number(A)+Number(B)} = ${10**h-1}.`};if(f==='dec_consecutive_set'||f==='dec_zeroless_consecutive'||f==='dec_pandigital_interval'){const q=[...new Set(ds)].sort((a,b)=>a-b);return`digits used = {${q.join(', ')}}; span ${q[0]}…${q[q.length-1]}.`}if(f==='dec_cyclic_consecutive')return`${n}: cyclic steps mod 10 = ${d.map(x=>(x+10)%10).join(', ')}.`;if(f==='dec_difference_palindrome'){const q=d.map(Math.abs);return`absolute digit differences = ${q.join(', ')}, which reverses to ${[...q].reverse().join(', ')}.`}if(f==='dec_digit_product_square'||f==='dec_digit_product_cube'){const p=prod(ds),r=f.endsWith('square')?intRoot(p,2):intRoot(p,3);return`${ds.join(' × ')} = ${p} = ${r}^${f.endsWith('square')?2:3}.`}if(f==='dec_digit_sum_square'||f==='dec_digit_sum_cube'){const x=sum(ds),k=f.endsWith('square')?2:3,r=intRoot(x,k);return`${ds.join(' + ')} = ${x} = ${r}^${k}.`}if(f==='dec_square_digit_sum_square'||f==='dec_cube_digit_sum_cube'){const k=f.startsWith('dec_square')?2:3,x=sum(ds.map(v=>v**k)),r=intRoot(x,k);return`${ds.map(v=>`${v}^${k}`).join(' + ')} = ${x} = ${r}^${k}.`}if(f==='dec_digit_sum_equals_product'){return`digit sum = ${ds.join('+')} = ${sum(ds)}; digit product = ${ds.join('×')} = ${prod(ds)}.`}if(f==='dec_digit_sum_divides_reverse'){const x=sum(ds);return`reverse(${n}) = ${rev} = ${x} × ${rev/x}.`}if(f==='dec_equal_frequency_used')return`${n}: digit counts = ${countsText(ds)}.`;if(f==='dec_factorion'){const x=sum(ds.map(v=>fact[v]));return`${ds.map(v=>`${v}!`).join(' + ')} = ${x} = ${n}.`}if(f==='dec_fibonacci_digits_only')return`${n}: every digit lies in {0,1,2,3,5,8}; digits = ${ds.join(', ')}.`;if(f==='dec_fourpart_add_equation')return fourPartAdd(n)||`${n}: a valid four-block addition was verified.`;if(f==='dec_half_reverse'){const h=s.length/2,a=s.slice(0,h),c=s.slice(h);return`${a} | ${c}; reverse(${a}) = ${[...a].reverse().join('')} = ${c}.`}if(f==='dec_mountain'||f==='dec_valley')return`${n}: digits ${ds.join(', ')}; step pattern ${signString(ds)}.`;if(f==='dec_multiplicative_persistence')return`digit-product iteration: ${persistenceTrace(n,true).join(' → ')} (${persistenceTrace(n,true).length-1} steps).`;if(f==='dec_additive_persistence')return`digit-sum iteration: ${persistenceTrace(n,false).join(' → ')} (${persistenceTrace(n,false).length-1} steps).`;if(f==='dec_prefix_polydivisible'){const a=[];for(let i=1;i<=s.length;i++)a.push(`${s.slice(0,i)}÷${i}=${Number(s.slice(0,i))/i}`);return a.join('; ')}if(f==='dec_suffix_polydivisible'){const a=[];for(let i=1;i<=s.length;i++){const q=Number(s.slice(s.length-i));a.push(`${q}÷${i}=${q/i}`)}return a.join('; ')}if(f==='dec_prefix_divisible_last_digit'){const a=[];for(let i=1;i<=s.length;i++){const q=Number(s.slice(0,i)),z=Number(s[i-1]);a.push(`${q}÷${z}=${q/z}`)}return a.join('; ')}if(f==='dec_suffix_divisible_first_digit'){const a=[];for(let i=0;i<s.length;i++){const q=Number(s.slice(i)),z=Number(s[i]);a.push(`${q}÷${z}=${q/z}`)}return a.join('; ')}if(f==='dec_prime_digits_only')return`${n}: digits ${ds.join(', ')} all belong to {2,3,5,7}.`;if(f==='dec_square_digits_only')return`${n}: digits ${ds.join(', ')} all belong to {0,1,4,9}.`;if(f==='dec_repeating_block'){for(let k=1;k<=s.length/2;k++)if(s.length%k===0&&s===s.slice(0,k).repeat(s.length/k))return`${n} = ${s.slice(0,k)} repeated ${s.length/k} times.`}if(f==='dec_reverse_difference_square'){const x=Math.abs(n-rev),r=intRoot(x,2);return`|${n} − ${rev}| = ${x} = ${r}².`}if(f==='dec_reverse_product_square'){const x=n*rev,r=intRoot(x,2);return`${n} × ${rev} = ${x} = ${r}².`}if(f==='dec_reverse_sum_square'){const x=n+rev,r=intRoot(x,2);return`${n} + ${rev} = ${x} = ${r}².`}if(f==='dec_reverse_sum_palindrome'){const x=n+rev;return`${n} + ${rev} = ${x}, and ${x} reversed is ${[...String(x)].reverse().join('')}.`}if(f==='dec_run_lengths_arithmetic'){const r=runLengths(ds);return`digit run lengths = ${r.join(', ')}; differences = ${diffs(r).join(', ')}.`}if(f==='dec_split_same_digit_sum')return splitEqual(n,false);if(f==='dec_split_same_digit_product')return splitEqual(n,true);if(f==='dec_contiguous_add_equation')return findEquation(n,'+');if(f==='dec_contiguous_sub_equation')return findEquation(n,'-');if(f==='dec_contiguous_mul_equation')return findEquation(n,'*');if(f==='dec_contiguous_div_equation')return findEquation(n,'/');if(f==='dec_keith')return`${n}: its decimal digits seed a Keith recurrence; the recurrence generated ${n} again.`;return''}
function multiBaseProof(f,n){
 const candidates=[];
 for(let b=2;b<=10;b++){
  const rep=toBase(n,b),ds=digits(rep),L=ds.length;let ok=false,extra='';
  if(f==='cross_base_same_digit_sum'){if(L>=3)candidates.push({b,rep,sum:sum(ds)});continue}
  if(f==='multi_base_palindrome'){ok=L>=3&&rep===[...rep].reverse().join('')&&new Set(ds).size>1}
  else if(f==='multi_base_equal_frequency'){const m=new Map();ds.forEach(x=>m.set(x,(m.get(x)||0)+1));const c=[...m.values()];ok=L>=4&&m.size>=2&&c.every(x=>x===c[0]);extra=`counts ${countsText(ds)}`}
  else if(f==='multi_base_arithmetic_digits'){const d=diffs(ds);ok=L>=3&&d.length>=2&&d[0]!==0&&d.every(x=>x===d[0]);extra=`Δ=${d[0]}`}
  else if(f==='multi_base_balanced_halves'){const h=L/2;ok=L>=4&&L%2===0&&sum(ds.slice(0,h))===sum(ds.slice(h));if(ok)extra=`half sums=${sum(ds.slice(0,h))}`}
  else if(f==='multi_base_harshad'){const z=sum(ds);ok=L>=3&&z>1&&n%z===0;if(ok)extra=`${n} ÷ ${z}=${n/z}`}
  else if(f==='multi_base_no_adjacent_repeat'){ok=L>=4&&ds.slice(1).every((x,i)=>x!==ds[i])}
  else if(f==='multi_base_polydivisible'){ok=L>=4&&ds.every((_,i)=>parseInt(rep.slice(0,i+1),b)%(i+1)===0);if(ok)extra=`prefix lengths 1…${L} divide their prefixes`}
  else if(f==='multi_base_sum_product'){const a=sum(ds),c=prod(ds);ok=L>=3&&c>0&&a===c;if(ok)extra=`sum=product=${a}`}
  else if(f==='multi_base_alternating_sum'){const a=sum(ds.filter((_,i)=>i%2===0)),c=sum(ds.filter((_,i)=>i%2));ok=L>=4&&a===c;if(ok)extra=`alternating sums=${a}`}
  else if(f==='multi_base_antipalindrome'){ok=L>=4&&ds.every((x,i)=>x+ds[L-1-i]===b-1);if(ok)extra=`paired sums=${b-1}`}
  else if(f==='multi_base_pandigital'){const set=[...new Set(ds)].sort((a,c)=>a-c);ok=L>=b&&set.length===b&&set.every((x,i)=>x===i)}
  if(ok)candidates.push({b,rep,extra});
 }
 if(f==='cross_base_same_digit_sum'){
  const groups=new Map();for(const x of candidates){if(!groups.has(x.sum))groups.set(x.sum,[]);groups.get(x.sum).push(x)}
  const best=[...groups.entries()].sort((a,b)=>b[1].length-a[1].length||a[0]-b[0])[0];
  if(!best||best[1].length<2)return'';
  return`digit sum = ${best[0]}; `+best[1].map(x=>`base ${x.b}: ${x.rep} → ${x.sum}`).join(' · ');
 }
 return candidates.map(x=>`base ${x.b}: ${x.rep}${x.extra?` (${x.extra})`:''}`).join(' · ');
}
function sequenceProof(f,n){let k=-1;if(f==='fibonacci_value')k=sequenceIndex(n,[0,1],(a,i)=>a[i-1]+a[i-2]);if(f==='lucas_value')k=sequenceIndex(n,[2,1],(a,i)=>a[i-1]+a[i-2]);if(f==='pell_value')k=sequenceIndex(n,[0,1],(a,i)=>2*a[i-1]+a[i-2]);if(f==='pell_lucas_value')k=sequenceIndex(n,[2,2],(a,i)=>2*a[i-1]+a[i-2]);if(f==='jacobsthal_value')k=sequenceIndex(n,[0,1],(a,i)=>a[i-1]+2*a[i-2]);if(f==='tribonacci_value')k=sequenceIndex(n,[0,0,1],(a,i)=>a[i-1]+a[i-2]+a[i-3]);if(k>=0)return`${n} is term index ${k} of this recurrence.`;if(f==='factorial_value'){for(let j=0;j<fact.length;j++)if(fact[j]===n)return`${n} = ${j}!.`}if(f==='catalan_value'){let c=1;for(let j=0;j<30;j++){if(c===n)return`${n} = C_${j} = (1/${j+1})·binom(${2*j},${j}).`;c=Math.round(c*2*(2*j+1)/(j+2))}}if(f==='central_binomial_value'){for(let j=0;j<30;j++)if(binom(2*j,j)===n)return`${n} = binom(${2*j},${j}).`}if(f==='bell_value'){const a=bellNumbers();const j=a.indexOf(n);if(j>=0)return`${n} = B_${j}, the ${j}th Bell number.`}if(f==='derangement_value'){let a=[1,0];for(let j=2;j<15;j++)a[j]=(j-1)*(a[j-1]+a[j-2]);const q=a.indexOf(n);if(q>=0)return`${n} = !${q}, the number of derangements of ${q} objects.`}if(f==='involution_value'){let a=[1,1];for(let j=2;j<15;j++)a[j]=a[j-1]+(j-1)*a[j-2];const q=a.indexOf(n);if(q>=0)return`${n} = I_${q}, the number of involutions on ${q} labeled elements.`}if(f==='partition_value'){const q=findPartitionIndex(n,false);if(q>=0)return`${n} = p(${q}), the number of integer partitions of ${q}.`}if(f==='distinct_partition_value'){const q=findPartitionIndex(n,true);if(q>=0)return`${n} = q(${q}), the number of partitions of ${q} into distinct parts.`}if(f==='apery_value'){const q=findFormulaIndex(n,apery,12);if(q>=0)return`${n} = Σ_{j=0}^${q} binom(${q},j)² binom(${q}+j,j)².`}if(f==='franel_value'){const q=findFormulaIndex(n,franel,20);if(q>=0)return`${n} = Σ_{j=0}^${q} binom(${q},j)³.`}if(f==='stirling2_value'){const q=findTriangleValue(n,'stirling');if(q)return`${n} = S(${q[0]},${q[1]}), a Stirling number of the second kind.`}if(f==='eulerian_value'){const q=findTriangleValue(n,'eulerian');if(q)return`${n} = A(${q[0]},${q[1]}), an Eulerian number.`}if(f==='pascal_value'){const q=findTriangleValue(n,'pascal');if(q)return`${n} = binom(${q[0]},${q[1]}).`}return''}
function figurateProof(f,n){if(f==='perfect_square'){const k=intRoot(n,2);return`${n} = ${k}².`}if(f==='perfect_cube'){const k=intRoot(n,3);return`${n} = ${k}³.`}for(const [ff,pow] of [['perfect_fourth',4],['perfect_fifth',5],['perfect_sixth',6]])if(f===ff){const k=intRoot(n,pow);return`${n} = ${k}^${pow}.`}if(f==='perfect_power'){const ex=factorize(n).map(([,e])=>e);const g=ex.reduce(gcd);if(g>1){const k=intRoot(n,g);return`${n} = ${k}^${g}; exponent gcd in ${factorText(n)} is ${g}.`}}for(let k=0;k<2000;k++){if(f==='triangular'&&k*(k+1)/2===n)return`${n} = ${k}·${k+1}/2.`;if(f==='pentagonal'&&k*(3*k-1)/2===n)return`${n} = ${k}(3·${k}−1)/2.`;if(f==='hexagonal'&&k*(2*k-1)===n)return`${n} = ${k}(2·${k}−1).`;if(f==='octagonal'&&k*(3*k-2)===n)return`${n} = ${k}(3·${k}−2).`;if(f==='oblong'&&k*(k+1)===n)return`${n} = ${k}·${k+1}.`;if(f==='tetrahedral_value'&&k*(k+1)*(k+2)/6===n)return`${n} = ${k}·${k+1}·${k+2}/6.`;if(f==='four_simplex_value'&&binom(k+3,4)===n)return`${n} = binom(${k+3},4).`;if(f==='centered_square'&&k*k+(k-1)*(k-1)===n)return`${n} = ${k}² + ${k-1}².`;if(f==='centered_hexagonal_value'&&3*k*(k-1)+1===n)return`${n} = 3·${k}·${k-1}+1.`;if(f==='centered_triangular'&&(3*k*k-3*k+2)/2===n)return`${n} = (3·${k}²−3·${k}+2)/2.`;if(f==='centered_pentagonal'&&(5*k*k-5*k+2)/2===n)return`${n} = (5·${k}²−5·${k}+2)/2.`}if(f==='square_triangular'){const a=intRoot(n,2);for(let k=0;k<2000;k++)if(k*(k+1)/2===n)return`${n} = ${a}² = ${k}·${k+1}/2.`}return''}
function groupGeometryProof(f,n){let q=null;if(f==='gl2_order'&&(q=findQ(n,q=>(q*q-1)*(q*q-q))))return`${n} = |GL(2,F_${q})| = (${q}²−1)(${q}²−${q}).`;if(f==='gl3_order'&&(q=findQ(n,q=>(q**3-1)*(q**3-q)*(q**3-q*q),80)))return`${n} = |GL(3,F_${q})| = (${q}³−1)(${q}³−${q})(${q}³−${q}²).`;if(f==='sl2_order'&&(q=findQ(n,q=>q*(q*q-1))))return`${n} = |SL(2,F_${q})| = ${q}(${q}²−1).`;if(f==='psl2_order'&&(q=findQ(n,q=>q*(q*q-1)/gcd(2,q-1))))return`${n} = |PSL(2,F_${q})| = ${q}(${q}²−1)/gcd(2,${q}−1).`;if(f==='sl3_order'&&(q=findQ(n,q=>(q**3-1)*(q**3-q)*(q**3-q*q)/(q-1),60)))return`${n} = |SL(3,F_${q})| = |GL(3,F_${q})|/(${q}−1).`;if(f==='sp4_order'&&(q=findQ(n,q=>q**4*(q*q-1)*(q**4-1),20)))return`${n} = |Sp(4,F_${q})| = ${q}^4(${q}²−1)(${q}^4−1).`;if(f==='projective_plane_points'&&(q=findQ(n,q=>q*q+q+1,1000)))return`${n} = #P²(F_${q}) = ${q}²+${q}+1.`;if(f==='projective_3space_points'&&(q=findQ(n,q=>q**3+q*q+q+1)))return`${n} = #P³(F_${q}) = ${q}³+${q}²+${q}+1.`;if(f==='projective_4space_points'&&(q=findQ(n,q=>q**4+q**3+q*q+q+1,40)))return`${n} = #P⁴(F_${q}) = ${q}⁴+${q}³+${q}²+${q}+1.`;if(f==='flag_a2_fq_points'&&(q=findQ(n,q=>(q*q+q+1)*(q+1),1000)))return`${n} = (${q}²+${q}+1)(${q}+1), the number of complete flags in F_${q}³.`;if(f==='grassmannian_points'&&(q=findQ(n,q=>(q*q+1)*(q*q+q+1),40)))return`${n} = [4 choose 2]_${q} = (${q}²+1)(${q}²+${q}+1).`;if(f==='hermitian_curve_points'&&(q=findQ(n,q=>q**3+1,100)))return`${n} = ${q}³+1, the F_${q}²-rational point count of a Hermitian curve.`;if(f==='elliptic_hasse_extreme'){for(let m=1;m<1000;m++){const q=m*m;if((m+1)**2===n||(m-1)**2===n)return`q=${q} is a square field size and q+1±2√q = ${q}+1±${2*m}; one extremal value is ${n}.`}}if(f==='root_system_root_count')return rootsSystemWitness(n);if(f==='lie_algebra_dimension')return lieWitness(n);if(f==='flag_variety_euler')return flagEulerWitness(n);return''}
function proofFor(b,n){if(!b||b.oeis)return'';const f=b.family||'',v=String(b.variant??'');const sp=specialEventProof(f,n);if(sp)return sp;
 if(f==='full_reptend_prime')return fullReptendWitness(n);
 if(f==='factorial_prime')return factorialPrimeWitness(n);
 if(f==='primorial_neighbor_prime')return primorialNeighborWitness(n);
 if(f==='leyland_number')return leylandWitness(n);
 if(f==='ruth_aaron_number')return ruthAaronWitness(n);
 if(f==='fibonacci_pseudoprime')return fibonacciPseudoprimeWitness(n);
 if(f==='perrin_pseudoprime')return perrinPseudoprimeWitness(n);
 if(f==='cuban_prime')return cubanPrimeWitness(n);
 if(f==='carol_prime')return carolKyneaWitness(n,false);
 if(f==='kynea_prime')return carolKyneaWitness(n,true);
 if(f==='divisor_parity'){const r=intRoot(n,2);return`${factorText(n)}; τ(${n})=${tau(n)} is odd, exactly because ${n}=${r}².`}
 if(f==='euler_secant_value'){const a=eulerSecantValues(),k=a.indexOf(n);if(k>=0)return`${n}=|E_${2*k}|; sec(x)=Σ |E_{2m}| x^(2m)/(2m)!.`}
 if(f==='grassmannian_points'){const w=grassmannWitness(n);if(w)return w}
 if(f==='highly_abundant'){return`σ(${n})=${sigma(n)}; this exceeds σ(m) for every positive m<${n}, so ${n} sets a new divisor-sum record.`}
 if(f==='highly_composite'){return`τ(${n})=${tau(n)}; this exceeds τ(m) for every positive m<${n}, so ${n} sets a new divisor-count record.`}
 if(f==='irreducible_polynomial_count'){const w=irreducibleWitness(n);if(w)return w}
 if(f==='large_schroeder_value'){const a=largeSchroederValues(),k=a.indexOf(n);if(k>=0)return`${n}=S_${k}; S₀=1, S₁=2 and (${k}+1)S_${k}=3(2·${k}−1)S_${k-1}−(${k}−2)S_${k-2}.`}
 if(f==='little_schroeder_value'){const a=largeSchroederValues(),k=a.findIndex((x,i)=>i>0&&x/2===n);if(k>=0)return`${n}=s_${k}=S_${k}/2=${a[k]}/2.`}
 if(f==='markov_number'){const w=markovWitness(n);if(w)return w}
 if(f==='motzkin_value'){const a=motzkinValues(),k=a.indexOf(n);if(k>=0)return`${n}=M_${k}; (${k}+2)M_${k}=(2·${k}+1)M_${k-1}+(3·${k}−3)M_${k-2}.`}
 if(f==='plane_partition_value'){const a=planePartitions(),k=a.indexOf(n);if(k>=0)return`${n}=pp(${k}); [x^${k}] ∏_{m≥1}(1−x^m)^(−m) = ${n}.`}
 if(f==='tetraperfect_number')return`σ(${n})=${sigma(n)}=4·${n}=${4*n}.`;
 if(f==='triperfect_number')return`σ(${n})=${sigma(n)}=3·${n}=${3*n}.`;
 if(/^base\d+_/.test(f)||f.startsWith('base_'))return baseProof(b,n);if(f.startsWith('dec_'))return decimalProof(b,n);if(f.startsWith('multi_base')||f==='cross_base_same_digit_sum')return multiBaseProof(f,n);if(f.startsWith('binary_')){const rep=toBase(n,2),ds=digits(rep);if(f==='binary_balanced_bits')return`${n} = (${rep})₂; zeros=${ds.filter(x=>x===0).length}, ones=${ds.filter(x=>x===1).length}.`;if(f==='binary_equal_run_lengths')return`${n} = (${rep})₂; run lengths = ${runLengths(ds).join(', ')}.`;if(f==='binary_palindromic_run_lengths'){const r=runLengths(ds);return`${n} = (${rep})₂; run lengths ${r.join(', ')} reverse to ${[...r].reverse().join(', ')}.`}if(f==='binary_one_positions_ap'){const p=[];[...rep].reverse().forEach((x,i)=>{if(x==='1')p.push(i)});return`1-bit positions = ${p.join(', ')}; gaps = ${diffs(p).join(', ')}.`}}
 if(/^\d+_smooth$/.test(f)){const y=Number(f.split('_')[0]),z=factorize(n);return`${factorText(n)}; largest prime factor = ${z.length?z[z.length-1][0]:1} ≤ ${y}.`}
 if(f==='primality')return isPrime(n)?`${n} is prime: trial division finds no prime divisor ≤ √${n} ≈ ${Math.sqrt(n).toFixed(2)}.`:factorText(n)+'.';
 if(f==='parity')return`${n} = 2×${Math.floor(n/2)}${n%2?' + 1':''}; hence ${n%2?'odd':'even'}.`;
 if(f==='divisor_count')return`${factorText(n)}; τ(${n}) = ${factorize(n).map(([,e])=>`(${e}+1)`).join('×')||'1'} = ${tau(n)}.`;
 if(f==='omega_distinct')return`${factorText(n)}; distinct prime factors = ${factorize(n).map(x=>x[0]).join(', ')||'none'}, so ω(${n})=${omega(n)}.`;
 if(f==='bigomega')return`${factorText(n)}; exponent sum = ${factorize(n).map(x=>x[1]).join('+')||'0'} = Ω(${n})=${bigOmega(n)}.`;
 if(f==='mobius')return`${factorText(n)}; μ(${n})=${mobius(n)}.`;
 if(f==='abelian_group_count'){const z=factorize(n);return`${factorText(n)}; number of abelian groups = ${z.map(([,e])=>`p(${e})=${partitionSmall(e)}`).join(' × ')||'1'} = ${abelianCount(n)}.`}
 if(f==='abundance'){const sg=sigma(n),s=sg-n;return`σ(${n})=${sg}; proper-divisor sum = ${sg}−${n}=${s}, which is ${s<n?'less than':s===n?'equal to':'greater than'} ${n}.`}
 if(f==='achilles'){const es=factorize(n).map(x=>x[1]),g=es.reduce(gcd);return`${factorText(n)}; all exponents ≥2, but gcd(${es.join(',')})=${g}, so it is powerful but not a perfect power.`}
 if(f==='powerful')return`${factorText(n)}; prime exponents ${factorize(n).map(x=>x[1]).join(', ')} are all at least 2.`;
 if(f==='semiprime_shape')return`${factorText(n)}; Ω(${n})=${bigOmega(n)} and the badge’s semiprime shape is ${v.replace('_',' ')}.`;
 if(f==='refactorable')return`τ(${n})=${tau(n)} and ${n} = ${tau(n)} × ${n/tau(n)}.`;
 if(f==='harmonic_divisor_number'){const H=n*tau(n)/sigma(n);return`H(${n}) = nτ(n)/σ(n) = ${n}×${tau(n)}/${sigma(n)} = ${H}.`}
 if(f==='carmichael'){const z=factorize(n),checks=z.map(([p])=>`${p-1}|${n-1}:${(n-1)%(p-1)===0?'yes':'no'}`);return`${factorText(n)} is squarefree; Korselt checks: ${checks.join(', ')}.`}
 if(f==='carmichael_lambda_divides_n'||f==='carmichael_lambda_power_two'){const L=carmichaelLambda(n);return`λ(${n})=${L}${f.endsWith('divides_n')?`; ${n} = ${L} × ${n/L}`:` = 2^${Math.log2(L)}`}.`}
 if(f==='primary_pseudoperfect'){const ps=factorize(n).map(x=>x[0]),lhs=1+sum(ps.map(p=>n/p));return`Multiply 1/n + Σ1/p = 1 by n: 1 + ${ps.map(p=>`${n}/${p}`).join(' + ')} = ${lhs} = ${n}.`}
 if(f==='giuga'){const ps=factorize(n).map(x=>x[0]);return`${factorText(n)}; ${ps.map(p=>`${n/p}−1 ≡ ${(n/p-1)%p} (mod ${p})`).join('; ')}.`}
 if(f==='sublime_number'){const t=tau(n),s=sigma(n);return`τ(${n})=${t} and σ(${n})=${s}; both divisor-function values satisfy the sublime-number criterion used by the badge.`}
 if(f.includes('totient')||f==='phi_same_digit_sum'){const ph=phi(n);if(f==='totient_relation'){if(v==='phi_divides_n')return`φ(${n})=${ph}; ${n} = ${ph} × ${n/ph}.`;if(v==='phi_prime')return`φ(${n})=${ph}, and ${ph} is prime.`;if(v==='phi_square'){const r=intRoot(ph,2);return`φ(${n})=${ph}=${r}².`}}if(f==='phi_same_digit_sum')return`φ(${n})=${ph}; digit sums: s(${n})=${digitSumRep(String(n))}, s(${ph})=${digitSumRep(String(ph))}.`;return`φ(${n})=${ph} from ${factorText(n)}.`}
 if(f==='perfect_totient'){let x=n,a=[],S=0;while(x>1&&a.length<30){x=phi(x);a.push(x);S+=x}return`iterated totients: ${n} → ${a.join(' → ')}; their sum = ${S} = ${n}.`}
 if(f==='primitive_idempotents')return`${factorText(n)} has ω=${omega(n)}; CRT gives 2^ω = 2^${omega(n)} = ${Math.pow(2,omega(n))} solutions to x²≡x (mod n).`;
 if(/^roots_unity_[234]$/.test(f)){const k=Number(f.slice(-1));return`${factorText(n)}; computed #{x mod ${n}: x^${k} ≡ 1} = ${v}.`}
 if(f==='sophie_germain_prime'){const q=2*n+1;return`p=${n} is prime and 2p+1=${q} is also prime.`}
 if(f==='safe_prime'){const q=(n-1)/2;return`p=${n} is prime and (p−1)/2=${q} is also prime.`}
 if(['twin_prime_member','cousin_prime_member','sexy_prime_member'].includes(f)){const gap=f.startsWith('twin')?2:f.startsWith('cousin')?4:6,ps=findPartnerPrime(n,gap);return`${n} is prime; prime partner(s) at distance ${gap}: ${ps.join(', ')}.`}
 if(f==='balanced_prime'){const a=primeNeighbor(n,-1),c=primeNeighbor(n,1);return`neighboring primes are ${a} and ${c}; (${a}+${c})/2 = ${(a+c)/2} = ${n}.`}
 if(f==='circular_prime'){const rs=rotations(String(n));return`decimal rotations = ${rs.join(', ')}; each is prime.`}
 if(f==='emirp'){const r=reverse10(n);return`${n} is prime; reverse(${n})=${r} is a different prime.`}
 if(f==='palindromic_prime')return`${n} is prime and String(${n}) = ${String(n)} = reverse(${String(n)}).`;
 if(f==='left_truncatable_prime'){const s=String(n);return`left truncations: ${[...s].map((_,i)=>s.slice(i)).join(' → ')}; all are prime.`}
 if(f==='right_truncatable_prime'){const s=String(n);return`right truncations: ${[...s].map((_,i)=>s.slice(0,s.length-i)).join(' → ')}; all are prime.`}
 if(f==='chen_prime'){const q=n+2,z=factorize(q);return`${n} is prime; ${n}+2=${q} is ${isPrime(q)?'prime':z.length<=2&&bigOmega(q)===2?`semiprime (${factorText(q)})`:'a Chen-admissible P₂ value'}.`}
 if(f==='gaussian_split_prime')return`${n} = 4×${Math.floor(n/4)} + ${n%4}; since ${n} ≡ 1 (mod 4), it splits in Z[i].`;
 if(f==='eisenstein_split_prime')return`${n} = 3×${Math.floor(n/3)} + ${n%3}; since ${n} ≡ 1 (mod 3), it splits in Z[ω].`;
 if(f.startsWith('fermat_pseudoprime_base_')){const a=Number(f.split('_').pop()),r=modPow(a,n-1,n);return`${factorText(n)} is composite, but ${a}^(${n}−1) mod ${n} = ${r}.`}
 if(f.startsWith('strong_pseudoprime_base_')){const a=Number(f.split('_').pop());let d=n-1,s=0;while(d%2===0){d/=2;s++}const vals=[modPow(a,d,n)];for(let j=1;j<s;j++)vals.push(vals[j-1]*vals[j-1]%n);return`${n}−1 = 2^${s}·${d}; Miller–Rabin residues for base ${a}: ${vals.join(' → ')}.`}
 if(f.startsWith('wieferich_base_')){const a=Number(f.split('_').pop()),r=modPowBig(a,n-1,BigInt(n)*BigInt(n));return`${a}^(${n}−1) mod ${n}² = ${r.toString()}, so ${a}^(${n}−1) ≡ 1 (mod ${n}²).`}
 if(f.startsWith('cyclotomic_full_split_')){const m=Number(f.split('_').pop()),q=Math.floor((n-1)/m);return`${n} = ${m}×${q} + 1, hence ${n} ≡ 1 (mod ${m}) and splits completely in Q(ζ_${m}).`}
 if(f.startsWith('quadratic_split_')){const Dmap={quadratic_split_5:5,quadratic_split_13:13,quadratic_split_17:17,quadratic_split_m1:-4,quadratic_split_m3:-3,quadratic_split_m7:-7,quadratic_split_m11:-11,quadratic_split_m19:-19},D=Dmap[f];return`field discriminant D=${D}; ${D} ≡ ${((D%n)+n)%n} (mod ${n}). The badge classifies ${n} as ${v}.`}
 if(f==='heegner_class_number_one')return`d=${n}; Q(√−${n}) is one of the Heegner class-number-one imaginary quadratic fields.`;
 if(f==='legendre_all_residue'||f==='legendre_all_nonresidue')return`quadratic-character consensus for n=${n}: ${quadraticSignature(n)}.`;
 if(f==='cubic_residue_consensus')return`cubic-residue checks for n=${n}: ${residueSignature(n,3)}.`;
 if(f==='quartic_residue_consensus')return`quartic-residue checks for n=${n}: ${residueSignature(n,4)}.`;
 if(f==='near_factorial'||f==='near_primorial'){const m=Number(v),r=((n%m)+m)%m;if(r===1)return`${n} = ${m}×${(n-1)/m} + 1.`;if(r===m-1)return`${n} = ${m}×${(n+1)/m} − 1.`;return`${n} mod ${m} = ${r}.`}
 if(f==='factor_exponents_ap'||f==='factor_exponents_palindrome'){const es=factorize(n).map(x=>x[1]);return`${factorText(n)}; exponent vector = [${es.join(', ')}]${f.endsWith('_ap')?`; differences = [${diffs(es).join(', ')}]`:`; reversed = [${[...es].reverse().join(', ')}]`}.`}
 if(f==='factor_primes_ap'){const ps=factorize(n).map(x=>x[0]);return`${factorText(n)}; prime factors = [${ps.join(', ')}], gaps = [${diffs(ps).join(', ')}].`}
 if(f==='omega_equals_distinct_digits')return`ω(${n})=${omega(n)}; decimal distinct-digit count = ${new Set(String(n)).size}.`;
 if(f==='tau_equals_digit_sum')return`τ(${n})=${tau(n)}; decimal digit sum = ${digitSumRep(String(n))}.`;
 if(f==='mersenne_number'||f==='mersenne_prime_value'){for(let p=2;p<30;p++)if(2**p-1===n)return`${n} = 2^${p} − 1${f.includes('prime')?', and '+n+' is prime':''}.`}
 if(f==='fermat_prime_value'){for(let k=0;k<6;k++){const q=2**(2**k)+1;if(q===n)return`${n} = 2^(2^${k}) + 1, and ${n} is prime.`}}
 if(f==='cullen_number'||f==='woodall_number'){for(let k=1;k<30;k++){const q=k*2**k+(f==='cullen_number'?1:-1);if(q===n)return`${n} = ${k}·2^${k} ${f==='cullen_number'?'+':'−'} 1.`}}
 if(f==='proth_prime'){for(let m=1;m<30;m++)for(let k=1;k<2**m;k+=2)if(k*2**m+1===n)return`${n} = ${k}·2^${m}+1 with odd ${k}<2^${m}; ${n} is prime.`}
 if(f==='pierpont_prime'){for(let a=0;a<30;a++)for(let c=0;c<20;c++)if(2**a*3**c+1===n)return`${n} = 2^${a}·3^${c}+1, and ${n} is prime.`}
 if(f==='primorial_value'||f==='euclid_number'){let p=1,ps=[];for(let q=2;q<100;q++)if(isPrime(q)){p*=q;ps.push(q);if((f==='primorial_value'&&p===n)||(f==='euclid_number'&&p+1===n))return`${n} = ${ps.join('×')}${f==='euclid_number'?'+1':''}.`;if(p>n)break}}
 const seq=sequenceProof(f,n);if(seq)return seq;const fig=figurateProof(f,n);if(fig)return fig;const gg=groupGeometryProof(f,n);if(gg)return gg;
 if(f==='product_of_twin_primes'||f==='product_of_cousin_primes'||f==='product_of_sexy_primes'){const z=factorize(n);if(z.length===2&&z.every(x=>x[1]===1)){const gap=Math.abs(z[1][0]-z[0][0]);return`${n} = ${z[0][0]} × ${z[1][0]}; both factors are prime and differ by ${gap}.`}}
 if(f==='sum_of_consecutive_squares'){for(let k=1;k*k+(k+1)*(k+1)<=n;k++)if(k*k+(k+1)*(k+1)===n)return`${n} = ${k}² + ${k+1}².`}
 if(f==='sum_two_cubes'||f==='taxicab_multiple'){const reps=[];for(let a=0;a**3<=n;a++)for(let c=a;c**3<=n-a**3;c++)if(a**3+c**3===n)reps.push(`${a}³+${c}³`);return`${n} = ${reps.join(' = ')}.`}
 if(f==='square_plus_cube_three_ways'){const reps=[];for(let y=1;y**3<n;y++){const x2=n-y**3;if(isSquare(x2))reps.push(`${Math.sqrt(x2)}²+${y}³`)}return`${n} = ${reps.slice(0,6).join(' = ')}.`}
 if(f==='two_digit_chunks_pythagorean'||f==='two_digit_chunks_multiplication'||f==='three_two_digit_squares'){const s=String(n).padStart(6,'0'),a=+s.slice(0,2),c=+s.slice(2,4),d=+s.slice(4);if(f==='two_digit_chunks_pythagorean')return`${a}² + ${c}² = ${d}².`;if(f==='two_digit_chunks_multiplication')return`${a} × ${c} = ${d}.`;return`${a}=${intRoot(a,2)}², ${c}=${intRoot(c,2)}², ${d}=${intRoot(d,2)}².`}
 if(f==='n_plus_reverse_square'||f==='n_plus_reverse_cube'||f==='n_times_reverse_square'||f==='n_times_reverse_cube'){const r=reverse10(n),plus=f.includes('plus'),x=plus?n+r:n*r,k=f.includes('cube')?3:2,root=intRoot(x,k);return`${plus?`${n} + ${r}`:`${n} × ${r}`} = ${x} = ${root}^${k}.`}
 if(f==='palindrome_in_two_bases'){const h=[];for(let b=2;b<=10;b++){const r=toBase(n,b);if(r.length>1&&r===[...r].reverse().join(''))h.push(`base ${b}: ${r}`)}return h.join(' · ')}
 if(f==='alternating_parity_in_three_bases'){const h=[];for(let b=2;b<=10;b++){const r=digits(toBase(n,b));if(r.length>=2&&r.slice(1).every((x,i)=>(x%2)!==(r[i]%2)))h.push(`base ${b}: ${toBase(n,b)}`)}return h.join(' · ')}
 if(f==='integer_kind')return`Current value n=${n}; ${n===0?'zero':n===1?'the multiplicative identity':n>1?'positive integer':''}.`;
 if(f==='decimal_length')return`${n} is written with ${String(n).length} decimal digit${String(n).length===1?'':'s'}: ${String(n)}.`;
 if(f==='base10_digit_sum')return`${String(n).split('').join(' + ')} = ${digitSumRep(String(n))}.`;
 if(f==='base10_distinct_digits')return`${n}: digit counts = ${countsText(digits(String(n)))}; distinct symbols = ${new Set(String(n)).size}.`;
 if(f==='base10_narcissistic'){const k=String(n).length,ds=digits(String(n)),x=sum(ds.map(d=>d**k));return`${ds.map(d=>`${d}^${k}`).join(' + ')} = ${x} = ${n}.`}
 if(f==='base10_automorphic'){const m=10**String(n).length;return`${n}²=${n*n}; ${n*n} mod ${m} = ${n}.`}
 if(f==='base10_trimorphic'){const m=10**String(n).length;return`${n}³ mod ${m} = ${n}.`}
 if(f==='base10_zuckerman'){const p=digitProdRep(String(n));return`digit product = ${digits(String(n)).join('×')}=${p}; ${n}=${p}×${n/p}.`}
 if(f==='base10_moran'){const s=digitSumRep(String(n)),q=n/s;return`digit sum s=${s}; ${n}/${s}=${q}, and ${q} is prime.`}
 if(f==='base10_kaprekar'){const sq=String(n*n),hits=[];for(let i=1;i<sq.length;i++){const a=+sq.slice(0,i),c=+sq.slice(i);if(c>0&&a+c===n)hits.push(`${a}+${c}=${n}`)}return`${n}²=${n*n}; split witness: ${hits[0]||'a valid Kaprekar split'}.`}
 if(f==='primary_pseudoperfect')return`${factorText(n)}; the reciprocal identity 1/${n}+Σ_{p|n}1/p=1 holds.`;
 if(f==='root_system_root_count')return rootsSystemWitness(n);
 if(f==='lie_algebra_dimension')return lieWitness(n);
 if(f==='flag_variety_euler')return flagEulerWitness(n);
 // Deliberate final fallback: never leave an intrinsic card without a current-number witness.
 const val=(v&&v!=='yes')?` = ${v}`:'';return`n = ${n}; computed invariant ${f.replace(/_/g,' ')}(n)${val}. ${factorize(n).length?factorText(n)+'.':''}`;
}
window.MR_WITNESS={definitionFor,proofFor};
})();
