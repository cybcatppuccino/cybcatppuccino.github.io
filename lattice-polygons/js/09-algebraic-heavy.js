'use strict';
/* v33 — on-demand algebraic-geometry computations for lattice polygons. */

/* ---------- shared exact combinatorics ---------- */
function agInt(x,min,max,fallback){
  x=Math.trunc(Number(x));
  return Number.isFinite(x)?Math.max(min,Math.min(max,x)):fallback
}
function agPause(){return new Promise(resolve=>setTimeout(resolve,0))}
function agPointKey(p){return p.join(',')}
function agSortedPoints(points){
  return points.map(p=>p.slice()).sort((a,b)=>{
    for(let i=0;i<Math.max(a.length,b.length);i++){
      const d=(a[i]||0)-(b[i]||0);
      if(d)return d
    }
    return 0
  })
}
function agExpKey(e){return e.join(',')}
function agAddExp(a,b){return a.map((x,i)=>x+b[i])}
function agUnitExp(n,i){const e=Array(n).fill(0);e[i]=1;return e}
function agExpFromIndices(n,idxs){const e=Array(n).fill(0);for(const i of idxs)e[i]++;return e}
function agMonomialLatex(e,varName='X'){
  const terms=[];
  e.forEach((a,i)=>{
    if(!a)return;
    const x=`${varName}_{${i+1}}`;
    terms.push(a===1?x:`${x}^{${a}}`)
  });
  return terms.join('')||'1'
}
function agDifferentialMonomialLatex(e){
  const terms=[];
  e.forEach((a,i)=>{
    if(!a)return;
    const x=`\\partial_{c_{${i+1}}}`;
    terms.push(a===1?x:`${x}^{${a}}`)
  });
  return terms.join('')||'1'
}
function agBinomialLatex(g,varName='X'){
  return `${agMonomialLatex(g.a,varName)}-${agMonomialLatex(g.b,varName)}`
}
function agVectorLatex(v){return `(${v.join(',')})`}
function agBigIntPow(a,n){
  a=BigInt(a);let r=1n;
  while(n-->0)r*=a;
  return r
}
function agEscapeText(x){
  return String(x).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))
}
function agDisplayList(items,limit=20){
  if(!items.length)return '<p class="subtle">none</p>';
  const shown=items.slice(0,limit);
  return `<ol class="ag-equation-list">${shown.map(x=>`<li>${x}</li>`).join('')}</ol>${items.length>shown.length?`<p class="small-note">Showing ${shown.length} of ${items.length} entries.</p>`:''}`
}
function agError(out,error){out.innerHTML=`<p class="empty">${agEscapeText(error&&error.message?error.message:error)}</p>`}

function agEnumerateMonomials(n,d,cap=200000){
  const out=[],idx=Array(d).fill(0);
  let stopped=false;
  function rec(pos,start){
    if(stopped)return;
    if(pos===d){
      out.push({idxs:idx.slice(),exp:agExpFromIndices(n,idx)});
      if(out.length>cap)stopped=true;
      return
    }
    for(let i=start;i<n;i++){
      idx[pos]=i;rec(pos+1,i);
      if(stopped)return
    }
  }
  if(d===0)return [{idxs:[],exp:Array(n).fill(0)}];
  rec(0,0);
  if(stopped)throw new Error(`Monomial cap ${cap.toLocaleString()} exceeded in degree ${d}. Increase it only for a manageable support.`);
  return out
}
function agWeightedSum(A,exp){
  const s=Array(A[0]?.length||0).fill(0);
  exp.forEach((a,i)=>{
    if(!a)return;
    for(let j=0;j<s.length;j++)s[j]+=a*A[i][j]
  });
  return s
}
function agFiberData(A,d,cap){
  const monomials=agEnumerateMonomials(A.length,d,cap),fibers=new Map();
  for(const m of monomials){
    m.sum=agWeightedSum(A,m.exp);
    const key=agPointKey(m.sum);
    if(!fibers.has(key))fibers.set(key,[]);
    fibers.get(key).push(m)
  }
  return {monomials,fibers}
}
function agUnionFind(n){
  const parent=Array.from({length:n},(_,i)=>i),rank=Array(n).fill(0);
  return {
    find(a){while(parent[a]!==a){parent[a]=parent[parent[a]];a=parent[a]}return a},
    join(a,b){
      a=this.find(a);b=this.find(b);if(a===b)return;
      if(rank[a]<rank[b])[a,b]=[b,a];
      parent[b]=a;if(rank[a]===rank[b])rank[a]++
    }
  }
}

/* A two-dimensional lattice polygon has a toric ideal generated in degrees 2 and 3.
   Degree-two fibers are handled exactly; degree-three generators are computed modulo
   variable multiples of the chosen quadratic basis by connected components. */
function agToricGenerators(points,cap=200000){
  const A=agSortedPoints(points),n=A.length;
  if(n<2)return {support:A,quadrics:[],cubics:[],all:[],fiber2:null,fiber3:null};

  const fiber2=agFiberData(A,2,cap),quadrics=[];
  for(const [fiber,monomials] of fiber2.fibers){
    if(monomials.length<2)continue;
    const root=monomials[0];
    for(let i=1;i<monomials.length;i++)quadrics.push({degree:2,a:root.exp.slice(),b:monomials[i].exp.slice(),fiber})
  }

  const fiber3=agFiberData(A,3,cap),index3=new Map(fiber3.monomials.map((m,i)=>[agExpKey(m.exp),i]));
  const uf=agUnionFind(fiber3.monomials.length);
  for(const q of quadrics)for(let i=0;i<n;i++){
    const left=agAddExp(q.a,agUnitExp(n,i)),right=agAddExp(q.b,agUnitExp(n,i));
    const a=index3.get(agExpKey(left)),b=index3.get(agExpKey(right));
    if(a!==undefined&&b!==undefined)uf.join(a,b)
  }

  const cubics=[];
  for(const [fiber,monomials] of fiber3.fibers){
    const components=new Map();
    for(const m of monomials){
      const component=uf.find(index3.get(agExpKey(m.exp)));
      if(!components.has(component))components.set(component,m)
    }
    const reps=[...components.values()];
    if(reps.length<2)continue;
    const root=reps[0];
    for(let i=1;i<reps.length;i++)cubics.push({degree:3,a:root.exp.slice(),b:reps[i].exp.slice(),fiber})
  }

  const all=quadrics.concat(cubics);
  all.forEach((g,i)=>g.id=i);
  return {support:A,quadrics,cubics,all,fiber2,fiber3}
}

/* ---------- affine circuits and GKZ data ---------- */
function agGcdArray(values){let g=0;for(const x of values)g=gcd(g,x);return Math.abs(g)||1}
function agNormalizeRelation(coeff){
  const g=agGcdArray(coeff);coeff=coeff.map(x=>x/g);
  const first=coeff.find(x=>x!==0)||1;
  return first<0?coeff.map(x=>-x):coeff
}
function agCross3(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function agDet3(a,b,c){
  return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
}
function agCircuitRelation(A,idx){
  if(idx.length===3){
    const one=[1,1,1],x=idx.map(i=>A[i][0]),y=idx.map(i=>A[i][1]);
    let coeff=agCross3(one,x);
    if(coeff.every(z=>z===0))coeff=agCross3(one,y);
    if(coeff.every(z=>z===0))coeff=agCross3(x,y);
    if(coeff.every(z=>z===0)||coeff.some(z=>z===0))return null;
    coeff=agNormalizeRelation(coeff);
    if(coeff.reduce((s,z)=>s+z,0)!==0)return null;
    if(coeff.reduce((s,z,i)=>s+z*A[idx[i]][0],0)!==0)return null;
    if(coeff.reduce((s,z,i)=>s+z*A[idx[i]][1],0)!==0)return null;
    return coeff
  }
  if(idx.length===4){
    const columns=idx.map(i=>[1,A[i][0],A[i][1]]),coeff=[];
    for(let i=0;i<4;i++){
      const q=columns.filter((_,j)=>j!==i);
      coeff.push((i%2?-1:1)*agDet3(q[0],q[1],q[2]))
    }
    if(coeff.some(z=>z===0))return null; // then a proper dependent subset exists
    return agNormalizeRelation(coeff)
  }
  return null
}
function agMod(a,p){a%=p;return a<0?a+p:a}
function agModPow(a,n,p){
  let r=1;a=agMod(a,p);
  while(n){if(n&1)r=r*a%p;a=a*a%p;n=Math.floor(n/2)}
  return r
}
function agModInv(a,p){if(!agMod(a,p))throw new Error('Non-invertible modular pivot.');return agModPow(a,p-2,p)}
function agRankMod(rows,p=1000003){
  if(!rows.length)return 0;
  const M=rows.map(r=>Int32Array.from(r,x=>agMod(x,p))),m=M.length,n=M[0].length;
  let rank=0;
  for(let col=0;col<n&&rank<m;col++){
    let pivot=rank;while(pivot<m&&!M[pivot][col])pivot++;
    if(pivot===m)continue;
    [M[rank],M[pivot]]=[M[pivot],M[rank]];
    const inv=agModInv(M[rank][col],p);
    for(let j=col;j<n;j++)M[rank][j]=M[rank][j]*inv%p;
    for(let i=0;i<m;i++)if(i!==rank&&M[i][col]){
      const factor=M[i][col];
      for(let j=col;j<n;j++)M[i][j]=agMod(M[i][j]-factor*M[rank][j],p)
    }
    rank++
  }
  return rank
}
function agCircuits(points,maxOutput=120,subsetCap=200000){
  const A=agSortedPoints(points),circuits=[],independent=[];
  let rank=0,tested=0,truncated=false,totalCircuits=0;
  const target=Math.max(0,A.length-3);

  function accept(idx){
    tested++;
    if(tested>subsetCap){truncated=true;return false}
    const coeff=agCircuitRelation(A,idx);
    if(!coeff)return true;
    totalCircuits++;
    const full=Array(A.length).fill(0);
    idx.forEach((j,k)=>full[j]=coeff[k]);
    const newRank=agRankMod(independent.map(x=>x.full).concat([full]));
    if(newRank>rank){rank=newRank;independent.push({idx:idx.slice(),coeff:coeff.slice(),full});}
    if(circuits.length<maxOutput)circuits.push({idx:idx.slice(),coeff:coeff.slice(),full});
    return true
  }

  outer3:for(let i=0;i<A.length;i++)for(let j=i+1;j<A.length;j++)for(let k=j+1;k<A.length;k++)if(!accept([i,j,k]))break outer3;
  if(!truncated)outer4:for(let i=0;i<A.length;i++)for(let j=i+1;j<A.length;j++)for(let k=j+1;k<A.length;k++)for(let l=k+1;l<A.length;l++)if(!accept([i,j,k,l]))break outer4;
  return {support:A,circuits,independent,rank,target,tested,truncated,totalCircuits,outputTruncated:totalCircuits>circuits.length}
}
function agCircuitBinomial(c,varName='X'){
  const pos=Array(c.full.length).fill(0),neg=Array(c.full.length).fill(0);
  c.full.forEach((x,i)=>{if(x>0)pos[i]=x;else if(x<0)neg[i]=-x});
  return `${agMonomialLatex(pos,varName)}-${agMonomialLatex(neg,varName)}`
}
/* Exact circuit discriminant up to a non-zero scalar/sign. */
function agCircuitDiscriminant(c,varName='c'){
  const pos=Array(c.full.length).fill(0),neg=Array(c.full.length).fill(0);
  let positiveWeight=1n,negativeWeight=1n;
  c.full.forEach((x,i)=>{
    if(x>0){pos[i]=x;positiveWeight*=agBigIntPow(x,x)}
    else if(x<0){neg[i]=-x;negativeWeight*=agBigIntPow(-x,-x)}
  });
  const positiveMonomial=agMonomialLatex(pos,varName),negativeMonomial=agMonomialLatex(neg,varName);
  const left=negativeWeight===1n?positiveMonomial:`${negativeWeight}${positiveMonomial}`;
  const right=positiveWeight===1n?negativeMonomial:`${positiveWeight}${negativeMonomial}`;
  return `${left}-${right}`
}
function agBoxOperator(c,label='\\ell'){
  const pos=Array(c.full.length).fill(0),neg=Array(c.full.length).fill(0);
  c.full.forEach((x,i)=>{if(x>0)pos[i]=x;else if(x<0)neg[i]=-x});
  return `\\square_{${label}}=${agDifferentialMonomialLatex(pos)}-${agDifferentialMonomialLatex(neg)}`
}
function agRelationVectorLatex(c,label='\\ell'){return `${label}=${agVectorLatex(c.full)}`}

/* ---------- low-degree first syzygies over exact finite fields ---------- */
function agBuildF1Columns(generators,n,totalDegree,monomialCap){
  const cache=new Map(),columns=[];
  for(const g of generators){
    const multiplierDegree=totalDegree-g.degree;
    if(multiplierDegree<0)continue;
    if(!cache.has(multiplierDegree))cache.set(multiplierDegree,agEnumerateMonomials(n,multiplierDegree,monomialCap));
    for(const m of cache.get(multiplierDegree))columns.push({generator:g.id,multiplier:m.exp,key:`${g.id}|${agExpKey(m.exp)}`})
  }
  return columns
}
function agBuildSyzygyMatrix(generators,n,totalDegree,monomialCap,columnCap){
  const columns=agBuildF1Columns(generators,n,totalDegree,monomialCap);
  if(columns.length>columnCap)throw new Error(`Syzygy module has ${columns.length.toLocaleString()} columns in total degree ${totalDegree}, above the cap ${columnCap.toLocaleString()}.`);
  const rowMap=new Map();
  function add(rowKey,column,value){
    if(!rowMap.has(rowKey))rowMap.set(rowKey,new Map());
    const row=rowMap.get(rowKey);row.set(column,(row.get(column)||0)+value)
  }
  columns.forEach((column,j)=>{
    const g=generators[column.generator];
    add(agExpKey(agAddExp(g.a,column.multiplier)),j,1);
    add(agExpKey(agAddExp(g.b,column.multiplier)),j,-1)
  });
  const rows=[];
  for(const sparse of rowMap.values()){
    const row=Array(columns.length).fill(0);
    for(const [j,x] of sparse)row[j]=x;
    rows.push(row)
  }
  return {columns,rows}
}
function agRrefNullspace(rows,p){
  if(!rows.length)return {rank:0,basis:[]};
  const M=rows.map(r=>Int32Array.from(r,x=>agMod(x,p))),m=M.length,n=M[0].length,pivots=[];
  let rank=0;
  for(let col=0;col<n&&rank<m;col++){
    let pivot=rank;while(pivot<m&&!M[pivot][col])pivot++;
    if(pivot===m)continue;
    [M[rank],M[pivot]]=[M[pivot],M[rank]];
    const inv=agModInv(M[rank][col],p);
    for(let j=col;j<n;j++)M[rank][j]=M[rank][j]*inv%p;
    for(let i=0;i<m;i++)if(i!==rank&&M[i][col]){
      const factor=M[i][col];
      for(let j=col;j<n;j++)M[i][j]=agMod(M[i][j]-factor*M[rank][j],p)
    }
    pivots.push(col);rank++
  }
  const pivotSet=new Set(pivots),free=[];
  for(let col=0;col<n;col++)if(!pivotSet.has(col))free.push(col);
  const basis=[];
  for(const f of free){
    const vector=new Int32Array(n);vector[f]=1;
    for(let i=0;i<pivots.length;i++)vector[pivots[i]]=agMod(-M[i][f],p);
    basis.push(vector)
  }
  return {rank,basis,pivots}
}
function agSpanRank(vectors,p){return vectors.length?agRankMod(vectors.map(v=>Array.from(v)),p):0}
function agMultiplyKernel(previous,currentColumns,n,p,vectorCap=2500){
  if(!previous||!previous.basis.length)return [];
  const currentIndex=new Map(currentColumns.map((c,i)=>[c.key,i])),out=[];
  for(const vector of previous.basis)for(let variable=0;variable<n;variable++){
    if(out.length>=vectorCap)throw new Error(`Multiplied-syzygy span exceeded ${vectorCap.toLocaleString()} vectors.`);
    const image=new Int32Array(currentColumns.length);
    for(let j=0;j<previous.columns.length;j++)if(vector[j]){
      const column=previous.columns[j],multiplier=column.multiplier.slice();multiplier[variable]++;
      const target=currentIndex.get(`${column.generator}|${agExpKey(multiplier)}`);
      if(target===undefined)throw new Error('Internal syzygy-degree indexing failure.');
      image[target]=agMod(image[target]+vector[j],p)
    }
    out.push(image)
  }
  return out
}
function agSyzygyStrands(generators,n,maxDegree,monomialCap,columnCap){
  if(!generators.length)return {rows:[],consistent:true,primes:[1000003,1000033]};
  const primes=[1000003,1000033],states=[null,null],rows=[];
  const start=Math.min(...generators.map(g=>g.degree));
  for(let degree=start;degree<=maxDegree;degree++){
    const data=agBuildSyzygyMatrix(generators,n,degree,monomialCap,columnCap),values=[];
    for(let z=0;z<primes.length;z++){
      const p=primes[z],kernel=agRrefNullspace(data.rows,p);
      const products=agMultiplyKernel(states[z],data.columns,n,p);
      const generated=agSpanRank(products,p);
      values.push({kernel:kernel.basis.length,generated,minimal:kernel.basis.length-generated});
      states[z]={basis:kernel.basis,columns:data.columns}
    }
    rows.push({
      degree,columns:data.columns.length,polynomialRows:data.rows.length,
      kernel:values[0].kernel,generatedFromPrevious:values[0].generated,minimal:values[0].minimal,
      consistent:values.every(x=>x.kernel===values[0].kernel&&x.generated===values[0].generated&&x.minimal===values[0].minimal)
    })
  }
  return {rows,consistent:rows.every(r=>r.consistent),primes}
}

/* ---------- canonical pencils and scrollar invariants ---------- */
function agMinWidthDirections(v){return latticeWidthData(v).directions}
function agWidthInvariantData(v,direction){
  const pointData=latticePoints(v),values=v.map(p=>dot(direction,p));
  const lo=Math.min(...values),hi=Math.max(...values),rows=[];
  for(let level=lo+1;level<hi;level++){
    const count=pointData.inn.filter(p=>dot(direction,p)===level).length;
    rows.push({level,count,value:count-1})
  }
  const widthInvariants=rows.map(x=>x.value),scrollar=widthInvariants.filter(x=>x>=0).sort((a,b)=>a-b);
  return {direction:direction.slice(),degree:hi-lo,lo,hi,rows,widthInvariants,scrollar,complete:widthInvariants.every(x=>x>=0)}
}
function agDualAction(direction,T){
  const A=T.A,d=T.det;
  const inverseTranspose=[[A[1][1]/d,-A[1][0]/d],[-A[0][1]/d,A[0][0]/d]];
  const q=[inverseTranspose[0][0]*direction[0]+inverseTranspose[0][1]*direction[1],inverseTranspose[1][0]*direction[0]+inverseTranspose[1][1]*direction[1]];
  return canonicalDualDirection(q.map(Math.round))
}
function agDirectionOrbits(directions,maps){
  const allowed=new Set(directions.map(x=>x.join(','))),seen=new Set(),orbits=[];
  for(const direction of directions){
    const key=direction.join(',');if(seen.has(key))continue;
    const orbit=new Set([key]),queue=[direction];
    while(queue.length){
      const u=queue.pop();
      for(const T of maps){
        const w=agDualAction(u,T),k=w.join(',');
        if(allowed.has(k)&&!orbit.has(k)){orbit.add(k);queue.push(w)}
      }
    }
    orbit.forEach(k=>seen.add(k));
    orbits.push([...orbit].map(k=>k.split(',').map(Number)))
  }
  return orbits
}

/* ---------- universal family, GKZ Euler operators ---------- */
function agLaurentPower(variable,exponent){
  if(exponent===0)return '';
  if(exponent===1)return variable;
  return `${variable}^{${exponent}}`
}
function agUniversalLaurentLatex(A,limit=18){
  const terms=A.slice(0,limit).map((p,i)=>`c_{${i+1}}${agLaurentPower('x',p[0])}${agLaurentPower('y',p[1])}`);
  return `f_{\\mathbf c}(x,y)=${terms.join('+')}${A.length>limit?'+\\cdots':''}`
}
function agSignedTerm(coefficient,body){
  if(coefficient===0)return null;
  if(coefficient===1)return body;
  if(coefficient===-1)return `-${body}`;
  return `${coefficient}${body}`
}
function agJoinSignedTerms(terms){
  const clean=terms.filter(Boolean);if(!clean.length)return '0';
  return clean.join('+').replace(/\+\-/g,'-')
}
function agEulerOperators(A){
  const e0=agJoinSignedTerms(A.map((_,i)=>`c_{${i+1}}\\partial_{c_{${i+1}}}`));
  const e1=agJoinSignedTerms(A.map((p,i)=>agSignedTerm(p[0],`c_{${i+1}}\\partial_{c_{${i+1}}}`)));
  const e2=agJoinSignedTerms(A.map((p,i)=>agSignedTerm(p[1],`c_{${i+1}}\\partial_{c_{${i+1}}}`)));
  return [`E_0=${e0}-\\beta_0`,`E_1=${e1}-\\beta_1`,`E_2=${e2}-\\beta_2`]
}

/* ---------- equivariant section-ring characters ---------- */
function agPermutationForMap(A,T){
  const index=new Map(A.map((p,i)=>[agPointKey(p),i])),permutation=[];
  for(const p of A){
    const q=[T.A[0][0]*p[0]+T.A[0][1]*p[1]+T.t[0],T.A[1][0]*p[0]+T.A[1][1]*p[1]+T.t[1]];
    const j=index.get(agPointKey(q));
    if(j===undefined)throw new Error('An affine automorphism failed to permute the lattice-point support.');
    permutation.push(j)
  }
  return permutation
}
function agCycleLengths(permutation){
  const seen=Array(permutation.length).fill(false),out=[];
  for(let i=0;i<permutation.length;i++)if(!seen[i]){
    let j=i,length=0;
    while(!seen[j]){seen[j]=true;length++;j=permutation[j]}
    out.push(length)
  }
  return out.sort((a,b)=>a-b)
}
/* Coefficients of product over cycles (1-t^length)^(-1). */
function agSymmetricPowerTrace(cycles,maxDegree){
  const coeff=Array(maxDegree+1).fill(0);coeff[0]=1;
  for(const length of cycles)for(let d=length;d<=maxDegree;d++)coeff[d]+=coeff[d-length];
  return coeff
}
function agDilate(v,d){return v.map(p=>[d*p[0],d*p[1]])}
function agFixedSections(v,T,d){
  const points=latticePoints(agDilate(v,d)).all;
  let fixed=0;
  for(const p of points){
    const q=[T.A[0][0]*p[0]+T.A[0][1]*p[1]+d*T.t[0],T.A[1][0]*p[0]+T.A[1][1]*p[1]+d*T.t[1]];
    if(q[0]===p[0]&&q[1]===p[1])fixed++
  }
  return fixed
}
function agEquivariantCharacters(v,maxDegree){
  const support=agSortedPoints(latticePoints(v).all),aut=automorphisms(v),elements=[];
  for(const T of aut.maps){
    const permutation=agPermutationForMap(support,T),cycles=agCycleLengths(permutation);
    const symmetric=agSymmetricPowerTrace(cycles,maxDegree),ring=Array(maxDegree+1).fill(0);ring[0]=1;
    for(let d=1;d<=maxDegree;d++)ring[d]=agFixedSections(v,T,d);
    elements.push({T,cycles,symmetric,ring,ideal:symmetric.map((x,d)=>x-ring[d])})
  }
  const invariants=[];
  for(let d=0;d<=maxDegree;d++){
    const average=field=>elements.reduce((sum,e)=>sum+e[field][d],0)/elements.length;
    invariants.push({degree:d,symmetric:average('symmetric'),ring:average('ring'),ideal:average('ideal')})
  }
  return {support,aut,elements,invariants}
}

/* ---------- verified Minkowski decompositions and Cayley configurations ---------- */
function agLatticePointsAny(poly){
  const h=hull(poly);
  if(!h.length)return [];
  if(h.length===1)return [h[0].slice()];
  if(h.length===2||area2(h)===0){
    const a=h[0],b=h[h.length-1],dx=b[0]-a[0],dy=b[1]-a[1],steps=gcd(dx,dy);
    return Array.from({length:steps+1},(_,i)=>[a[0]+dx*i/steps,a[1]+dy*i/steps])
  }
  return latticePoints(h).all
}
function agCayleySupport(Q,R){
  const support=[];
  for(const p of agSortedPoints(agLatticePointsAny(Q)))support.push([p[0],p[1],0]);
  for(const p of agSortedPoints(agLatticePointsAny(R)))support.push([p[0],p[1],1]);
  return support
}
function agMinkowskiConvolutionEquations(Q,R){
  const qPoints=agSortedPoints(agLatticePointsAny(Q));
  const rPoints=agSortedPoints(agLatticePointsAny(R));
  const fibers=new Map();
  for(let i=0;i<qPoints.length;i++)for(let j=0;j<rPoints.length;j++){
    const key=agPointKey([qPoints[i][0]+rPoints[j][0],qPoints[i][1]+rPoints[j][1]]);
    if(!fibers.has(key))fibers.set(key,[]);
    fibers.get(key).push([i,j])
  }
  const sumPolygon=hull(Q.flatMap(q=>R.map(r=>[q[0]+r[0],q[1]+r[1]])));
  const pPoints=agSortedPoints(agLatticePointsAny(sumPolygon));
  const entries=pPoints.map(point=>({point,pairs:fibers.get(agPointKey(point))||[]}));
  const equations=entries.map((entry,k)=>{
    const convolution=entry.pairs.length?entry.pairs.map(([i,j])=>`a_{${i+1}}b_{${j+1}}`).join('+'):'0';
    return `c_{${k+1}}-\\left(${convolution}\\right)=0`
  });
  return {qPoints,rPoints,pPoints,entries,equations,missing:entries.filter(entry=>entry.pairs.length===0).length}
}
function agScalePoly(poly,k){return poly.map(p=>[k*p[0],k*p[1]])}
function agZonotopePair(v,pointData){
  const generators=zonotopeGenerators(v,pointData);
  if(generators.length<2)return null;
  const R=[[0,0],generators[0]],rest=generators.slice(1);
  let Q=[[0,0]];
  for(const g of rest)Q=minkowskiSumPoly(Q,[[0,0],g]);
  Q=hull(Q);
  return keyOf(minkowskiSumPoly(Q,R))===keyOf(v)?{Q,R,type:'zonotope split'}:null
}
function agHomotheticPair(v){
  const data=primitivePolygonData(v);
  if(!data.changed||data.content<2)return null;
  const Q=data.poly,R=agScalePoly(Q,data.content-1);
  return keyOf(minkowskiSumPoly(Q,R))===keyOf(v)?{Q,R,type:`homothetic ${data.content}-fold split`}:null
}

function agCayleyQuadrics(A,cap){
  const fiber=agFiberData(A,2,cap),out=[];
  for(const [key,monomials] of fiber.fibers){
    if(monomials.length<2)continue;
    const root=monomials[0];
    for(let i=1;i<monomials.length;i++)out.push({a:root.exp,b:monomials[i].exp,fiber:key})
  }
  return out
}


/* ---------- automatic resource profiles and Koszul cohomology ---------- */
function agChooseNumber(n,k){
  if(k<0||k>n)return 0;
  k=Math.min(k,n-k);let r=1;
  for(let i=1;i<=k;i++)r=r*(n-k+i)/i;
  return Math.round(r)
}
function agAutomaticProfile(n){
  const degree3=agChooseNumber(n+2,3);
  return {
    monomialCap:Math.min(500000,Math.max(5000,degree3+32)),
    circuitDisplay:Math.max(30,Math.min(140,4*n)),
    circuitSubsetCap:Math.min(600000,Math.max(3000,agChooseNumber(n,3)+agChooseNumber(n,4)+16)),
    koszulColumnCap:n<=9?30000:n<=12?18000:n<=16?9000:4500,
    koszulOperationCap:n<=9?18000000:n<=12?11000000:n<=16?6500000:3500000
  }
}
function agCombinations(n,k){
  if(k<0||k>n)return [];
  if(k===0)return [[]];
  const out=[],a=Array(k);
  function rec(pos,start){
    if(pos===k){out.push(a.slice());return}
    for(let i=start;i<=n-(k-pos);i++){a[pos]=i;rec(pos+1,i+1)}
  }
  rec(0,0);return out
}
function agSemigroupDegreeBasis(A,d,cap,cache){
  if(cache&&cache.has(d))return cache.get(d);
  let out;
  if(d===0)out=[Array(A[0]?.length||0).fill(0)];
  else{
    const unique=new Map();
    for(const m of agEnumerateMonomials(A.length,d,cap)){
      const sum=agWeightedSum(A,m.exp);unique.set(agPointKey(sum),sum)
    }
    out=agSortedPoints([...unique.values()])
  }
  if(cache)cache.set(d,out);return out
}
function agSparseRankMod(columns,p,operationCap=8000000){
  const pivots=new Map();let rank=0,operations=0;
  for(const source of columns){
    const vector=new Map();
    for(const [row,value] of source){
      const v=agMod((vector.get(row)||0)+value,p);
      if(v)vector.set(row,v);else vector.delete(row)
    }
    while(vector.size){
      let pivot=null;
      for(const row of vector.keys())if(pivot===null||row<pivot)pivot=row;
      const basis=pivots.get(pivot);
      if(!basis){
        const inv=agModInv(vector.get(pivot),p);
        for(const [row,value] of [...vector]){
          const v=value*inv%p;if(v)vector.set(row,v);else vector.delete(row)
        }
        pivots.set(pivot,vector);rank++;break
      }
      const factor=vector.get(pivot);
      for(const [row,value] of basis){
        const v=agMod((vector.get(row)||0)-factor*value,p);
        if(v)vector.set(row,v);else vector.delete(row)
      }
      operations+=basis.size;
      if(operations>operationCap)throw new Error(`Koszul sparse-rank operation guard ${operationCap.toLocaleString()} reached.`)
    }
  }
  return {rank,operations}
}
function agKoszulDifferentialColumns(A,p,q,basisCache,columnCap){
  if(p<=0||q<0)return {columns:[],sourceDimension:0,targetRows:0};
  const basis=agSemigroupDegreeBasis(A,q,500000,basisCache),subsetCount=agChooseNumber(A.length,p),sourceDimension=subsetCount*basis.length;
  if(!Number.isFinite(sourceDimension)||sourceDimension>columnCap)return {skipped:true,sourceDimension,reason:`${Number.isFinite(sourceDimension)?sourceDimension.toLocaleString():'Too many'} columns exceed the automatic ${columnCap.toLocaleString()}-column guard`};
  const subsets=agCombinations(A.length,p);
  const rowIds=new Map(),columns=[];
  function rowId(key){if(!rowIds.has(key))rowIds.set(key,rowIds.size);return rowIds.get(key)}
  for(const subset of subsets)for(const weight of basis){
    const column=[];
    for(let r=0;r<subset.length;r++){
      const i=subset[r],targetSubset=subset.slice(0,r).concat(subset.slice(r+1));
      const targetWeight=weight.map((x,j)=>x+A[i][j]);
      column.push([rowId(`${targetSubset.join('.')}|${agPointKey(targetWeight)}`),r%2?-1:1])
    }
    columns.push(column)
  }
  return {columns,sourceDimension,targetRows:rowIds.size}
}
function agAffineConfigurationRank(A){
  if(!A.length)return 0;
  const rows=[A.map(()=>1)];
  for(let j=0;j<A[0].length;j++)rows.push(A.map(p=>p[j]));
  return agRankMod(rows,1000003)
}
function agKoszulBetti(A,options={}){
  A=agSortedPoints(A);const n=A.length,profile=agAutomaticProfile(n),basisCache=new Map();
  const affineRank=agAffineConfigurationRank(A),projectiveDimension=Math.max(0,n-affineRank);
  const maxQ=options.maxQ??2,primes=options.primes||[1000003,1000033];
  const columnCap=options.columnCap||profile.koszulColumnCap,operationCap=options.operationCap||profile.koszulOperationCap;
  const differentialCache=new Map(),cells=[];
  function differential(p,q){
    const key=`${p}|${q}`;if(differentialCache.has(key))return differentialCache.get(key);
    if(p<=0||q<0){const z={available:true,ranks:primes.map(()=>0),consistent:true,sourceDimension:0};differentialCache.set(key,z);return z}
    const built=agKoszulDifferentialColumns(A,p,q,basisCache,columnCap);
    if(built.skipped){const z={available:false,...built};differentialCache.set(key,z);return z}
    const ranks=[];
    try{for(const prime of primes)ranks.push(agSparseRankMod(built.columns,prime,operationCap).rank)}
    catch(error){const z={available:false,sourceDimension:built.sourceDimension,reason:error.message};differentialCache.set(key,z);return z}
    const z={available:true,ranks,consistent:ranks.every(x=>x===ranks[0]),sourceDimension:built.sourceDimension,targetRows:built.targetRows};
    differentialCache.set(key,z);return z
  }
  for(let q=1;q<=maxQ;q++)for(let p=1;p<=projectiveDimension;p++){
    const basis=agSemigroupDegreeBasis(A,q,profile.monomialCap,basisCache),chainDimension=agChooseNumber(n,p)*basis.length;
    const outgoing=differential(p,q),incoming=differential(p+1,q-1);
    if(!outgoing.available||!incoming.available){cells.push({p,q,available:false,chainDimension,reason:outgoing.reason||incoming.reason});continue}
    const values=primes.map((_,i)=>chainDimension-outgoing.ranks[i]-incoming.ranks[i]);
    const valid=values.every(x=>Number.isInteger(x)&&x>=0);
    cells.push({p,q,available:valid,values,consistent:valid&&values.every(x=>x===values[0]),value:valid?values[0]:null,chainDimension,reason:valid?'':'rank computation produced an invalid homology dimension'})
  }
  return {support:A,n,affineRank,projectiveDimension,maxQ,primes,cells,columnCap,operationCap,basisCache}
}
function agKoszulCell(table,p,q){return table.cells.find(c=>c.p===p&&c.q===q)}
function agKoszulTableHtml(table,title='Koszul Betti table'){
  const cols=Array.from({length:table.projectiveDimension},(_,i)=>i+1);
  if(!cols.length)return '<p class="subtle">The configuration has no nontrivial projective syzygies.</p>';
  const head=cols.map(p=>`<th>${p}</th>`).join('');
  const rows=[];
  for(let q=1;q<=table.maxQ;q++)rows.push(`<tr><th>${mathInline(`q=${q}`)}</th>${cols.map(p=>{const c=agKoszulCell(table,p,q);return `<td title="${agEscapeText(c?.reason||'')}">${c&&c.available?(c.consistent?c.value:`${c.values.join('/')}`):'—'}</td>`}).join('')}</tr>`);
  const computed=table.cells.filter(c=>c.available).length,total=table.cells.length;
  return `<table class="small-table ag-betti-table"><thead><tr><th>${mathInline('\\beta_{p,p+q}')}</th>${head}</tr></thead><tbody>${rows.join('')}</tbody></table><p class="small-note">${computed} of ${total} cells computed exactly over ${table.primes.map(p=>mathInline(`\\mathbf F_{${p}}`)).join(' and ')}. A dash means that the automatic sparse-matrix guard stopped that cell; it does not mean vanishing.</p>`
}
function agPropertyNReport(table){
  if(table.maxP===0)return 'the toric ideal has no nontrivial homological column; the requested syzygy conditions are vacuous';
  let verified=0,complete=true,firstNonzero=null;
  for(let p=1;p<=table.projectiveDimension;p++){
    const cell=agKoszulCell(table,p,2);
    if(!cell||!cell.available){complete=false;break}
    if(cell.value!==0){firstNonzero=p;break}
    verified=p
  }
  if(firstNonzero===1)return `${mathInline('N_0')} only; cubic generators obstruct ${mathInline('N_1')}`;
  if(firstNonzero!==null)return `${mathInline(`N_{${firstNonzero-1}}`)} verified, while ${mathInline(`\\beta_{${firstNonzero},${firstNonzero+2}}\\ne0`)} obstructs ${mathInline(`N_{${firstNonzero}}`)}`;
  if(complete)return `${mathInline(`N_{${table.projectiveDimension}}`)} verified through the whole computed resolution`;
  return verified?`${mathInline(`N_{${verified}}`)} verified; later cells exceeded the automatic guard`:'not certified because an early quadratic-strand cell exceeded the automatic guard'
}

/* ---------- saturated Gale dual and Horn--Kapranov chart ---------- */
function agBigAbs(x){return x<0n?-x:x}
function agExtendedGcdBig(a,b){
  let oldR=agBigAbs(a),r=agBigAbs(b),oldS=1n,s=0n,oldT=0n,t=1n;
  while(r!==0n){const q=oldR/r;[oldR,r]=[r,oldR-q*r];[oldS,s]=[s,oldS-q*s];[oldT,t]=[t,oldT-q*t]}
  if(a<0n)oldS=-oldS;if(b<0n)oldT=-oldT;
  return {g:oldR,s:oldS,t:oldT}
}
function agIdentityBig(n){return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1n:0n))}
function agSwapRows(M,i,j){if(i!==j)[M[i],M[j]]=[M[j],M[i]]}
function agSwapCols(M,i,j){if(i===j)return;for(const row of M)[row[i],row[j]]=[row[j],row[i]]}
function agCombineRowsToGcd(M,k,i,col){
  const a=M[k][col],b=M[i][col];if(b===0n)return;
  const {g,s,t}=agExtendedGcdBig(a,b),u=-b/g,v=a/g,oldK=M[k].slice(),oldI=M[i].slice();
  for(let j=0;j<M[0].length;j++){M[k][j]=s*oldK[j]+t*oldI[j];M[i][j]=u*oldK[j]+v*oldI[j]}
}
function agCombineColsToGcd(M,V,k,j,row){
  const a=M[row][k],b=M[row][j];if(b===0n)return;
  const {g,s,t}=agExtendedGcdBig(a,b),u=-b/g,v=a/g;
  for(let i=0;i<M.length;i++){const x=M[i][k],y=M[i][j];M[i][k]=s*x+t*y;M[i][j]=u*x+v*y}
  for(let i=0;i<V.length;i++){const x=V[i][k],y=V[i][j];V[i][k]=s*x+t*y;V[i][j]=u*x+v*y}
}
function agSaturatedIntegerKernel(matrix){
  const M=matrix.map(row=>row.map(BigInt)),m=M.length,n=M[0]?.length||0,V=agIdentityBig(n);
  let row=0,pivotCol=0;
  while(row<m&&pivotCol<n){
    let pivot=null;
    for(let i=row;i<m&&!pivot;i++)for(let j=pivotCol;j<n;j++)if(M[i][j]!==0n){pivot={i,j};break}
    if(!pivot)break;
    agSwapRows(M,row,pivot.i);agSwapCols(M,pivotCol,pivot.j);agSwapCols(V,pivotCol,pivot.j);
    /* The unprocessed columns are zero in all previously processed rows.
       Unimodular gcd operations among those columns therefore preserve the
       earlier echelon steps while clearing the current row exactly. */
    for(let j=pivotCol+1;j<n;j++)if(M[row][j]!==0n)agCombineColsToGcd(M,V,pivotCol,j,row);
    if(M[row][pivotCol]<0n){
      for(let i=0;i<m;i++)M[i][pivotCol]=-M[i][pivotCol];
      for(let i=0;i<n;i++)V[i][pivotCol]=-V[i][pivotCol]
    }
    for(let j=pivotCol+1;j<n;j++)if(M[row][j]!==0n)throw new Error('Integer Gale column reduction failed to clear a row.');
    row++;pivotCol++
  }
  const rank=pivotCol,basis=[];
  for(let j=rank;j<n;j++)basis.push(V.map(r=>r[j]));
  for(const vector of basis)for(const sourceRow of matrix){
    const sum=sourceRow.reduce((acc,x,i)=>acc+BigInt(x)*vector[i],0n);
    if(sum!==0n)throw new Error('Internal Gale-kernel verification failed.')
  }
  /* Because V is unimodular, its zero-column block is the full integral
     kernel, hence a saturated lattice rather than merely a rational basis. */
  return {rank,basis,transformed:M,columnTransform:V}
}
function agGaleDual(A){
  const homogenized=[A.map(()=>1),A.map(p=>p[0]),A.map(p=>p[1])],kernel=agSaturatedIntegerKernel(homogenized);
  const rows=A.map((_,i)=>kernel.basis.map(v=>v[i]));
  return {homogenized,kernel,rows,rank:kernel.basis.length}
}
function agLinearFormLatex(coeff,variable='u'){
  const terms=[];
  coeff.forEach((a,j)=>{const x=BigInt(a);if(x===0n)return;const sign=x<0n?'-':'+';const abs=agBigAbs(x),body=abs===1n?`${variable}_{${j+1}}`:`${abs}${variable}_{${j+1}}`;terms.push({sign,body})});
  if(!terms.length)return '0';
  return terms.map((t,i)=>(i===0?(t.sign==='-'?'-':''):t.sign)+t.body).join('')
}
function agHornCoordinates(gale,limit=12){
  const r=gale.rank;if(!r)return {coordinates:[],zeroRows:[]};
  const forms=gale.rows.map(row=>agLinearFormLatex(row)),zeroRows=[];
  gale.rows.forEach((row,i)=>{if(row.every(x=>x===0n))zeroRows.push(i)});
  const coordinates=[];
  for(let j=0;j<Math.min(r,limit);j++){
    const numerator=[],denominator=[];
    gale.rows.forEach((row,i)=>{const exponent=row[j];if(exponent>0n)numerator.push(exponent===1n?`L_{${i+1}}`:`L_{${i+1}}^{${exponent}}`);else if(exponent<0n){const e=-exponent;denominator.push(e===1n?`L_{${i+1}}`:`L_{${i+1}}^{${e}}`)}});
    const value=denominator.length?`\\frac{${numerator.join('')||'1'}}{${denominator.join('')}}`:(numerator.join('')||'1');
    coordinates.push(`y_{${j+1}}=${value}`)
  }
  return {coordinates,forms,zeroRows,truncated:r>limit}
}
function agGaleRowsHtml(gale,limit=30){
  const rows=gale.rows.slice(0,limit).map((row,i)=>`<tr><td>${i+1}</td><td>${mathInline(agVectorLatex(row.map(String)))}</td><td>${mathInline(`L_{${i+1}}=${agLinearFormLatex(row)}`)}</td></tr>`).join('');
  return `<table class="small-table"><thead><tr><th>${mathInline('i')}</th><th>${mathInline('b_i')}</th><th>${mathInline('L_i(u)=b_i\\cdot u')}</th></tr></thead><tbody>${rows}</tbody></table>${gale.rows.length>limit?`<p class="small-note">Showing ${limit} of ${gale.rows.length} Gale rows.</p>`:''}`
}

/* ---------- on-demand algebraic-geometry display ---------- */
function agModuleShell(id,title,lead){
  return `<details class="formula ag-module" id="${id}"><summary>${helpLabel(title)}</summary><div class="ag-module-body"><p class="ag-module-lead">${lead}</p><div class="compute-panel ag-compute-panel"><button type="button" class="compute-btn" data-ag-compute="${id}">Compute</button></div><div class="compute-result ag-result" data-ag-result="${id}"><p class="subtle">No calculation has been run. This module is computed on demand.</p></div></div></details>`
}
function agLegacyDropdowns(s){
  return `<details class="formula"><summary>Adjoint tower and subcurve polygons</summary><div class="figure-box adjoint-tower-inline">${svgAdjointTower(s.v)}</div></details><details class="formula"><summary>Curve-counting and mirror-symmetry computations</summary>${curveCountingRows(s)}<div class="compute-panel"><label>Mirror period support<select id="period-support"><option value="all" selected>all lattice points</option><option value="vertices">vertices only</option><option value="boundary">boundary lattice points</option></select></label><label>max order<input id="period-order" type="number" min="1" value="16"></label><button type="button" id="compute-period" class="compute-btn">Compute CT period</button><button type="button" id="compute-pf" class="compute-btn">Compute PF / mirror map</button><button type="button" id="stop-pf" class="compute-btn stop-btn">Stop PF</button></div><div id="curve-compute-result" class="compute-result"></div></details>${detail('Background: curves, adjunction, toric MMP and periods',`The adjoint tower repeatedly replaces a polygon by the convex hull of its strict interior lattice points. Width-spectrum rows record the actual primitive directions and slice sizes, not just a direct rewrite of one basic count. The blow-down button enumerates toric ${mathInline('(-1)')}-curve contractions only when the normal fan is smooth; it is a toric minimal-model diagnostic. The period button computes exact constant terms ${mathInline('\\operatorname{CT}(f^k)')} after translating by the selected interior origin.`)}`
}
function algebraicSection(s){
  const modules=`<div class="ag-module-grid">
    ${agModuleShell('ag-module-a','Equations, toric ideals and syzygies','Constructs minimal toric equations and computes graded Koszul homology, giving a finite-characteristic Betti table rather than only low-degree relation counts.')}
    ${agModuleShell('ag-module-b','Canonical curve, pencils and scrolls','Computes the canonical toric envelope, its Koszul syzygies, and every exact minimum-width monomial pencil with scrollar data.')}
    ${agModuleShell('ag-module-c','GKZ system and discriminant strata','Constructs primitive circuits, a saturated integer Gale dual, GKZ operators, circuit discriminants, and a Horn–Kapranov chart for the reduced discriminant.')}
  </div>`;
  return `<div class="card"><h3>Algebraic geometry and curve-counting data</h3>${modules}${agLegacyDropdowns(s)}</div>`
}
function agLinearStrandSummary(table){
  const known=table.cells.filter(c=>c.q===1&&c.available&&c.value>0);
  if(!known.length)return 'no nonzero linear-strand entry was found among computed cells';
  const last=Math.max(...known.map(c=>c.p));
  return `${mathInline(`\\beta_{${last},${last+1}}`)} is the last nonzero computed linear-strand entry${table.cells.some(c=>c.q===1&&!c.available&&c.p>last)?'; later cells were guarded':''}`
}

/* ---------- Equations, toric ideals and syzygies ---------- */
async function computeAGModuleA(v,root){
  const out=root.querySelector('[data-ag-result="ag-module-a"]');if(!out)return;
  out.innerHTML='<p class="empty">Enumerating semigroup fibers and computing Koszul homology over two finite fields...</p>';
  await agPause();
  try{
    const A=agSortedPoints(latticePoints(v).all),profile=agAutomaticProfile(A.length);
    const generators=agToricGenerators(A,profile.monomialCap),circuits=agCircuits(A,profile.circuitDisplay,profile.circuitSubsetCap);
    await agPause();
    const betti=agKoszulBetti(A,{maxQ:2,columnCap:profile.koszulColumnCap,operationCap:profile.koszulOperationCap});
    const equations=generators.all.map(g=>mathInline(`${agBinomialLatex(g,'X')}=0`));
    const circuitRows=circuits.circuits.map((c,i)=>`${mathInline(`\\ell^{(${i+1})}=${agVectorLatex(c.full)}`)};&nbsp; ${mathInline(`${agCircuitBinomial(c,'X')}=0`)}`);
    out.innerHTML=`${kvTable([
      ['Minimal toric equations',`${generators.quadrics.length} quadratic and ${generators.cubics.length} essential cubic generator(s)`],
      ['Graded Koszul cohomology',`${agLinearStrandSummary(betti)}; ${agPropertyNReport(betti)}`],
      ['Primitive affine circuits',`${circuits.totalCircuits.toLocaleString()} circuit(s) found${circuits.truncated?' before the automatic subset guard':''}`],
      ['Automatic resource profile',`${betti.cells.filter(c=>c.available).length}/${betti.cells.length} Betti cells completed; computation guards adapt to the ${A.length}-point support`]
    ])}<details class="formula"><summary>${helpLabel('Minimal toric equations')}</summary>${agDisplayList(equations,40)}<p class="subtle">Degree-two fibers give a basis of quadratic relations. Essential cubics are degree-three relations remaining after quotienting by variable multiples of the quadrics; for a lattice polygon these degrees generate the homogeneous toric ideal.</p></details><details class="formula"><summary>${helpLabel('Koszul Betti table')}</summary>${agKoszulTableHtml(betti)}<p class="subtle">The entry in column ${mathInline('p')} and row ${mathInline('q')} is ${mathInline('\\beta_{p,p+q}=\\dim K_{p,q}(R_P)')}. The differential is built directly from multiplication of semigroup monomials in the Koszul complex.</p></details><details class="formula"><summary>${helpLabel('Primitive affine circuits')}</summary>${agDisplayList(circuitRows,profile.circuitDisplay)}${circuits.outputTruncated?'<p class="small-note">Only the first circuits are displayed; the count reports all circuits visited before the automatic guard.</p>':''}</details><p class="small-note">Variable order: ${A.map((p,i)=>`${mathInline(`X_{${i+1}}`)}↔${mathInline(agVectorLatex(p))}`).join(', ')}.</p>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}

/* ---------- Canonical curve, pencils and scrolls ---------- */
async function computeAGModuleB(v,root){
  const out=root.querySelector('[data-ag-result="ag-module-b"]');if(!out)return;
  out.innerHTML='<p class="empty">Computing the canonical support, its syzygies, and all minimum-width pencils...</p>';
  await agPause();
  try{
    const pointData=latticePoints(v),canonicalSupport=agSortedPoints(pointData.inn),interiorDimension=interiorPolygonDimensionValue(v);
    const profile=agAutomaticProfile(canonicalSupport.length),envelope=canonicalSupport.length>=2?agToricGenerators(canonicalSupport,profile.monomialCap):null;
    const canonicalBetti=canonicalSupport.length>=2?agKoszulBetti(canonicalSupport,{maxQ:2,columnCap:profile.koszulColumnCap,operationCap:profile.koszulOperationCap}):null;
    const directions=agMinWidthDirections(v),aut=automorphisms(v),orbits=agDirectionOrbits(directions,aut.maps),pencils=directions.map(u=>agWidthInvariantData(v,u));
    const envelopeText=!canonicalSupport.length?'empty canonical support':canonicalSupport.length===1?'one canonical monomial':`${envelope.quadrics.length} quadratic and ${envelope.cubics.length} essential cubic equation(s) for the canonical toric envelope`;
    const pencilRows=pencils.map((p,i)=>`<tr><td>${i+1}</td><td>${mathInline(agVectorLatex(p.direction))}</td><td>${mathInline(`\\pi_{${i+1}}=[1:x^{${p.direction[0]}}y^{${p.direction[1]}}]`)}</td><td>${p.degree}</td><td>${mathInline(`(${p.widthInvariants.join(',')})`)}</td><td>${mathInline(`(${p.scrollar.join(',')})`)}</td><td>${interiorDimension===2?(p.complete?'complete':'incomplete'):'not asserted'}</td></tr>`).join('');
    const equations=envelope?envelope.all.map(g=>mathInline(`${agBinomialLatex(g,'Z')}=0`)):[];
    out.innerHTML=`${kvTable([
      ['Canonical toric envelope',envelopeText],
      ['Canonical-envelope Koszul syzygies',canonicalBetti?`${agLinearStrandSummary(canonicalBetti)}; ${agPropertyNReport(canonicalBetti)}`:'not defined for an empty or one-point canonical support'],
      ['Minimum-width pencil package',`${pencils.length} exact minimum-width direction(s), grouped into ${orbits.length} automorphism orbit(s)`],
      ['Canonical monomial coordinates',canonicalSupport.length?canonicalSupport.map((p,i)=>mathInline(`Z_{${i+1}}=x^{${p[0]}}y^{${p[1]}}`)).join(', '):'empty']
    ])}<details class="formula"><summary>${helpLabel('Canonical-envelope equations')}</summary>${agDisplayList(equations,30)}<p class="subtle">These equations define the toric envelope determined by the strict interior support. A coefficient-specific canonical curve has additional equations depending on its Laurent polynomial.</p></details>${canonicalBetti?`<details class="formula"><summary>${helpLabel('Canonical-envelope Koszul table')}</summary>${agKoszulTableHtml(canonicalBetti,'Canonical-envelope Koszul table')}<p class="subtle">This table belongs to the canonical toric envelope, not automatically to the full canonical curve. It nevertheless records genuine ambient syzygies constraining the canonical model.</p></details>`:''}<details class="formula"><summary>${helpLabel('Minimum-width pencils')}</summary><table class="small-table"><thead><tr><th>#</th><th>direction</th><th>monomial map</th><th>degree</th><th>width invariants</th><th>scrollar invariants</th><th>completeness</th></tr></thead><tbody>${pencilRows||'<tr><td colspan="7">none</td></tr>'}</tbody></table><p class="subtle">For a Newton-nondegenerate curve with two-dimensional interior polygon, ${mathInline('E_\\ell=\\#\\{m\\in P^\\circ\\cap M:\\langle u,m\\rangle=a+\\ell\\}-1')}. The nonnegative ${mathInline('E_\\ell')} form the scrollar-invariant multiset; outside this hypothesis only the exact combinatorial sequence is asserted.</p></details><details class="formula"><summary>${helpLabel('Pencil orbits under polygon automorphisms')}</summary>${agDisplayList(orbits.map((orbit,i)=>`${i+1}. ${orbit.map(agVectorLatex).map(mathInline).join(', ')}`),30)}</details>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}

/* ---------- GKZ system and discriminant strata ---------- */
async function computeAGModuleC(v,root){
  const out=root.querySelector('[data-ag-result="ag-module-c"]');if(!out)return;
  out.innerHTML='<p class="empty">Computing circuits, a saturated Gale dual, GKZ operators, and the Horn–Kapranov chart...</p>';
  await agPause();
  try{
    const A=agSortedPoints(latticePoints(v).all),profile=agAutomaticProfile(A.length);
    const circuits=agCircuits(A,profile.circuitDisplay,profile.circuitSubsetCap),gale=agGaleDual(A),horn=agHornCoordinates(gale,12);
    const operators=circuits.independent.map((c,i)=>`${mathInline(agRelationVectorLatex(c,`\\ell^{(${i+1})}`))}<br>${mathInline(agBoxOperator(c,`\\ell^{(${i+1})}`))}`);
    const discriminants=circuits.circuits.map((c,i)=>mathInline(`\\Delta_{C_{${i+1}}}=${agCircuitDiscriminant(c)}`)),eulers=agEulerOperators(A).map(mathInline);
    const hornForms=horn.forms?horn.forms.map((f,i)=>mathInline(`L_{${i+1}}=${f}`)):[];
    const hornCoordinates=horn.coordinates.map(mathInline);
    const hornStatus=!gale.rank?'The support is affinely independent, so there are no reduced discriminant coordinates.':horn.zeroRows.length?`A zero Gale row occurs at support index ${horn.zeroRows.map(i=>i+1).join(', ')}; the unsimplified Horn chart degenerates along a pyramid factor.`:`${horn.coordinates.length} reduced coordinate formula(s) displayed${horn.truncated?' from a larger chart':''}`;
    out.innerHTML=`${kvTable([
      ['Saturated Gale dual',`${mathInline('A B=0')} verified after unimodular integer column reduction; ${gale.rows.length} Gale row vector(s)`],
      ['Horn–Kapranov reduced-discriminant chart',hornStatus],
      ['Independent circuit box operators',`${operators.length} modularly independent primitive circuit operator(s) found${circuits.truncated?' before the automatic subset guard':''}`],
      ['Circuit-discriminant strata',`${circuits.totalCircuits.toLocaleString()} exact circuit discriminant(s) found; ${discriminants.length} retained for display`],
      ['Universal Laurent family',mathInline(agUniversalLaurentLatex(A))]
    ])}<details class="formula"><summary>${helpLabel('Saturated Gale dual')}</summary>${agGaleRowsHtml(gale)}<p class="subtle">The displayed columns form the integral kernel of the homogenized configuration as a saturated sublattice, because they are columns of a unimodular transformation sending the configuration matrix to integral column-echelon form.</p></details><details class="formula"><summary>${helpLabel('Horn–Kapranov reduced-discriminant chart')}</summary>${agDisplayList(hornForms,30)}${agDisplayList(hornCoordinates,20)}<p class="subtle">With Gale rows ${mathInline('b_i')} and ${mathInline('L_i(u)=b_i\\cdot u')}, the reduced coordinates are ${mathInline('y_j=\\prod_iL_i(u)^{b_{ij}}')}. This parametrizes the reduced ${mathInline('A')}-discriminant on the torus where the indicated linear forms do not vanish.</p></details><details class="formula"><summary>${helpLabel('GKZ box operators')}</summary>${agDisplayList(operators,operators.length||1)}</details><details class="formula"><summary>${helpLabel('GKZ Euler operators')}</summary>${agDisplayList(eulers,10)}</details><details class="formula"><summary>${helpLabel('Circuit discriminants')}</summary>${agDisplayList(discriminants,profile.circuitDisplay)}<p class="subtle">Each equation is the exact discriminant of a minimally affinely dependent subconfiguration, up to a nonzero scalar. The collection is not mislabeled as the complete ${mathInline('A')}-discriminant.</p></details><p class="small-note">Coefficient order: ${A.map((p,i)=>`${mathInline(`c_{${i+1}}`)}↔${mathInline(agVectorLatex(p))}`).join(', ')}. ${circuits.truncated?`Enumeration stopped after ${circuits.tested.toLocaleString()} tested subsets.`:`Tested ${circuits.tested.toLocaleString()} candidate subsets.`}</p>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}

/* ---------- Minkowski, Cayley and deformation ---------- */
function agConvolutionJacobianRanks(convolution,primes=[1000003,1000033]){
  const ranks=[];
  for(const prime of primes){
    let best=0;
    for(let sample=0;sample<3;sample++){
      const a=convolution.qPoints.map((_,i)=>agMod((i+2)*(sample+3)+1,prime)||1);
      const b=convolution.rPoints.map((_,j)=>agMod((j+3)*(j+sample+4)+1,prime)||1);
      const rows=convolution.entries.map(entry=>{
        const row=Array(a.length+b.length).fill(0);
        for(const [i,j] of entry.pairs){row[i]=agMod(row[i]+b[j],prime);row[a.length+j]=agMod(row[a.length+j]+a[i],prime)}
        return row
      });
      best=Math.max(best,agRankMod(rows,prime))
    }
    ranks.push(best)
  }
  return {primes,ranks,consistent:ranks.every(x=>x===ranks[0]),expected:convolution.qPoints.length+convolution.rPoints.length-1}
}
function minkowskiSection(s,P){
  return `<div class="card"><h3>Minkowski, Cayley and deformation</h3><details class="formula"><summary>${helpLabel('Minkowski decomposition')}</summary><input id="minkowski-limit" type="hidden" value="12"><div class="compute-panel"><button type="button" id="compute-minkowski" class="compute-btn">Compute Minkowski decompositions</button></div><div id="minkowski-result" class="compute-result"><p class="subtle">No decomposition search has been run.</p></div></details><details class="formula ag-module" id="minkowski-equivariant"><summary>${helpLabel('Equivariant algebra and deformation presentations')}</summary><div class="ag-module-body"><p class="ag-module-lead">Computes automorphism characters and attaches coefficient-convolution, Cayley-ideal, Jacobian-rank, and low-degree Koszul data to verified Minkowski sums.</p><div class="compute-panel ag-compute-panel"><button type="button" class="compute-btn" data-ag-compute="minkowski-equivariant">Compute</button></div><div class="compute-result ag-result" data-ag-result="minkowski-equivariant"><p class="subtle">No calculation has been run. This module is computed on demand.</p></div></div></details></div>`
}
async function computeMinkowskiEquivariant(v,root){
  const out=root.querySelector('[data-ag-result="minkowski-equivariant"]');if(!out)return;
  out.innerHTML='<p class="empty">Computing equivariant characters and verified Minkowski–Cayley deformation data...</p>';
  await agPause();
  try{
    const supportSize=latticePoints(v).all.length,maxDegree=supportSize<=14?5:supportSize<=28?4:3;
    const limit=agInt(root.querySelector('#minkowski-limit')?.value,1,40,12),equivariant=agEquivariantCharacters(v,maxDegree);
    const invariantRows=equivariant.invariants.map(r=>`<tr><td>${r.degree}</td><td>${r.symmetric}</td><td>${r.ring}</td><td>${r.ideal}</td></tr>`).join('');
    const characterRows=equivariant.elements.map((e,i)=>`<tr><td>${i+1}</td><td>${e.T.det}</td><td>${mathInline(agVectorLatex(e.cycles))}</td><td>${e.symmetric[2]}</td><td>${e.ring[2]}</td><td>${e.ideal[2]}</td></tr>`).join('');
    const pointData=latticePoints(v),decompositions=[],seen=new Set();
    function add(item){if(!item||decompositions.length>=limit||keyOf(minkowskiSumPoly(item.Q,item.R))!==keyOf(v))return;const key=[keyOf(item.Q),keyOf(item.R)].sort().join('||');if(seen.has(key))return;seen.add(key);decompositions.push(item)}
    add(agZonotopePair(v,pointData));add(agHomotheticPair(v));
    for(const item of segmentSummands(v,pointData,limit))add({Q:item.quotientVertices,R:[[0,0],item.u],type:'segment summand'});
    const remaining=Math.max(0,limit-decompositions.length),split=remaining?edgeSplitDecompositions(v,remaining,350000):{out:[],stopped:false,visited:0};
    for(const item of split.out)add({Q:item.Q,R:item.R,type:'full-dimensional split'});
    const cards=[];
    for(let i=0;i<decompositions.length;i++){
      const d=decompositions[i],support=agCayleySupport(d.Q,d.R),profile=agAutomaticProfile(support.length),lowGenerators=agToricGenerators(support,profile.monomialCap);
      const cayleyEquations=lowGenerators.all.slice(0,12).map(g=>mathInline(`${agMonomialLatex(g.a,'U')}-${agMonomialLatex(g.b,'U')}=0`));
      const convolution=agMinkowskiConvolutionEquations(d.Q,d.R),convolutionEquations=convolution.equations.slice(0,14).map(mathInline),jacobian=agConvolutionJacobianRanks(convolution);
      let cayleyBetti='';
      if(i<3){
        const table=agKoszulBetti(support,{maxQ:2,columnCap:Math.min(9000,profile.koszulColumnCap),operationCap:Math.min(6500000,profile.koszulOperationCap)});
        cayleyBetti=`<details class="formula"><summary>${helpLabel('Cayley low-degree Koszul strands')}</summary>${agKoszulTableHtml(table)}<p class="subtle">Only rows ${mathInline('q=1,2')} are computed. A higher-dimensional Cayley semigroup ring may have further rows, so this is deliberately not called a complete Betti table.</p></details>`
      }
      cards.push(`<div class="decomp-card"><h5>${i+1}. ${d.type}: ${mathInline('P=Q+R')}</h5>${kvTable([
        ['Coefficient-convolution presentation',`${mathInline('f_{\\mathbf c}=g_{\\mathbf a}h_{\\mathbf b}')}; ${convolution.entries.length} exact coefficient equation(s)`],
        ['Sampled differential rank',`${jacobian.ranks.join('/')} over ${jacobian.primes.map(p=>mathInline(`\\mathbf F_{${p}}`)).join(', ')}; expected generic value ${jacobian.expected}${jacobian.consistent&&jacobian.ranks[0]===jacobian.expected?' attained':' not certified'}`],
        ['Cayley low-degree ideal',`${lowGenerators.quadrics.length} quadratic and ${lowGenerators.cubics.length} cubic relation(s) modulo the quadrics; higher generator degrees are not excluded`]
      ])}<details class="formula"><summary>${helpLabel('Coefficient convolution equations')}</summary>${agDisplayList(convolutionEquations,14)}</details><details class="formula"><summary>${helpLabel('Cayley low-degree equations')}</summary>${agDisplayList(cayleyEquations,12)}</details>${cayleyBetti}<p class="small-note">Factor coordinates: ${convolution.qPoints.map((p,j)=>`${mathInline(`a_{${j+1}}`)}↔${mathInline(agVectorLatex(p))}`).join(', ')}; ${convolution.rPoints.map((p,j)=>`${mathInline(`b_{${j+1}}`)}↔${mathInline(agVectorLatex(p))}`).join(', ')}.</p></div>`)
    }
    out.innerHTML=`${kvTable([
      ['Equivariant section-ring character',`${equivariant.aut.order} affine lattice automorphism(s); traces on ${mathInline('\\operatorname{Sym}^dH^0(L)')}, ${mathInline('H^0(dL)')}, and ${mathInline('I_d')} through degree ${maxDegree}`],
      ['Invariant equation spaces',mathInline('\\dim I_d^G=|G|^{-1}\\sum_{g\\in G}\\chi_{I_d}(g)')],
      ['Verified deformation presentations',`${decompositions.length} Minkowski decomposition(s) converted into coefficient-convolution and Cayley packages${split.stopped?' before the automatic split-search guard':''}`]
    ])}<details class="formula"><summary>${helpLabel('Invariant Hilbert coefficients')}</summary><table class="small-table"><thead><tr><th>${mathInline('d')}</th><th>${mathInline('\\dim(\\operatorname{Sym}^dH^0(L))^G')}</th><th>${mathInline('\\dim H^0(dL)^G')}</th><th>${mathInline('\\dim I_d^G')}</th></tr></thead><tbody>${invariantRows}</tbody></table></details><details class="formula"><summary>${helpLabel('Automorphism character rows in degree two')}</summary><table class="small-table"><thead><tr><th>element</th><th>det</th><th>cycles on ${mathInline('P\\cap M')}</th><th>${mathInline('\\chi_{\\operatorname{Sym}^2}')}</th><th>${mathInline('\\chi_{R_2}')}</th><th>${mathInline('\\chi_{I_2}')}</th></tr></thead><tbody>${characterRows}</tbody></table></details><details class="formula"><summary>${helpLabel('Minkowski–Cayley deformation presentations')}</summary>${cards.join('')||'<p class="subtle">No nontrivial exact Minkowski decomposition was found within the automatic search guard.</p>'}<p class="subtle">The coefficient equations and Cayley syzygies are exact for each verified sum. The sampled Jacobian rank is explicitly labeled as a finite-field generic-rank check, and none of these packages is identified with ${mathInline('T^1')} or a miniversal base without cotangent-cohomology computation.</p></details>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}
function attachAlgebraicHeavyControls(v,root=document){
  root.querySelectorAll('[data-ag-compute]').forEach(button=>{
    if(button.dataset.agBound)return;button.dataset.agBound='1';
    button.onclick=()=>{
      const id=button.dataset.agCompute;
      if(id==='ag-module-a')computeAGModuleA(v,root);
      else if(id==='ag-module-b')computeAGModuleB(v,root);
      else if(id==='ag-module-c')computeAGModuleC(v,root);
      else if(id==='minkowski-equivariant')computeMinkowskiEquivariant(v,root)
    }
  })
}


/* ========================================================================== */
/* v33 strengthening: Markov-fiber rigidity, truncated Graver bases,          */
/* determinantal scroll equations, Horn-rank certificates, and user-visible   */
/* homological truncation parameters.                                          */
/* ========================================================================== */

Object.assign(HELP_TEXT,{
  "Betti columns": "<p><strong>What it is.</strong> The largest homological index \\(p\\) requested in the displayed Koszul table.</p><p><strong>How it is calculated.</strong> The app constructs the differentials around \\(K_{p,q}\\) only through the requested column. The default receives a computation budget of at least ten seconds; asking for more columns increases the internal sparse-matrix and time budgets.</p>",
  "Primitive-binomial degree bound": "<p><strong>What it is.</strong> The largest standard degree through which primitive toric binomials are enumerated.</p><p><strong>How it is calculated.</strong> Fiber pairs are reduced by their monomial gcd and tested for conformal decomposability. Completion through degree \\(d\\) certifies the entire Graver basis in degrees at most \\(d\\), but says nothing about higher degrees.</p>",
  "Markov-fiber rigidity": "<p><strong>What it is.</strong> The generating fibers of the toric ideal after quotienting by lower-degree moves, together with the exact number of minimal Markov bases.</p><p><strong>How it is calculated.</strong> If a generating fiber has component sizes \\(s_1,\\ldots,s_k\\), minimal generators are spanning trees of the complete multigraph having \\(s_i s_j\\) edges between components \\(i\\) and \\(j\\). The weighted Matrix-Tree theorem gives the number of choices.</p>",
  "Truncated Graver basis": "<p><strong>What it is.</strong> Primitive binomials \\(X^{u^+}-X^{u^-}\\) up to the requested degree. Primitive means that the lattice relation \\(u\\) has no nonzero conformal subrelation.</p><p><strong>How it is calculated.</strong> The app enumerates equal-semigroup-degree monomial pairs, removes common factors, and processes relations by increasing degree. A relation is discarded exactly when a previously certified primitive relation lies componentwise in the same orthant.</p>",
  "Determinantal scroll presentations": "<p><strong>What it is.</strong> Explicit rational-normal-scroll equations forced by a minimum-width monomial pencil on a Newton-nondegenerate canonical curve.</p><p><strong>How it is calculated.</strong> Interior lattice points are ordered in parallel slices. Consecutive points form the columns of a two-row scroll matrix, and all \\(2\\times2\\) minors are verified as equal-exponent binomials in the canonical coordinates.</p>",
  "Scroll Eagon–Northcott strand": "<p><strong>What it is.</strong> The linear Betti strand of the determinantal scroll containing the canonical model.</p><p><strong>How it is calculated.</strong> For a two-row scroll matrix with \\(r\\) columns, the Eagon–Northcott resolution gives \\(\\beta_{p,p+1}=p\\binom{r}{p+1}\\) for \\(1\\le p\\le r-1\\).</p>",
  "Horn rank samples": "<p><strong>What it is.</strong> The number of deterministic finite-field points used to seek a maximal-rank logarithmic Jacobian of the Horn–Kapranov map.</p><p><strong>How it is calculated.</strong> For Gale matrix \\(B\\), the logarithmic Jacobian is \\(J(u)=B^{\\mathsf T}\\operatorname{diag}((Bu)^{-1})B\\). A nonzero \\((r-1)\\)-minor modulo one prime is an exact certificate that the reduced discriminant has the expected hypersurface dimension over characteristic zero. Failure to find one is reported only as inconclusive.</p>",
  "Horn–Kapranov rank certificate": "<p><strong>What it is.</strong> A rigorous non-defect certificate obtained from the generic differential of the reduced Horn map.</p><p><strong>How it is calculated.</strong> Since the source is projective, the expected rank is \\(r-1\\). Attaining this rank modulo a prime proves that an integer numerator minor is nonzero, hence the same rank occurs generically over \\(\\mathbb Q\\). Lower sampled ranks do not prove defect.</p>",
  "Invariant degree bound": "<p><strong>What it is.</strong> The highest graded degree used for automorphism characters and invariant equation spaces.</p><p><strong>How it is calculated.</strong> Traces are computed on \\(\\operatorname{Sym}^dH^0(L)\\), \\(H^0(dL)\\), and their kernel \\(I_d\\), then averaged over the affine lattice automorphism group.</p>",
  "Cayley Betti columns": "<p><strong>What it is.</strong> The largest homological column requested for each displayed Cayley Koszul strand.</p><p><strong>How it is calculated.</strong> Cayley rings are not silently assumed Cohen–Macaulay. The app computes only the requested columns of \\(K_{p,q}\\), with the degree rows explicitly labelled as a truncation.</p>",
  "Maximum decompositions": "<p><strong>What it is.</strong> The maximum number of exact Minkowski decompositions retained and displayed.</p><p><strong>How it is calculated.</strong> Segment quotients and full-dimensional edge-length splittings are checked by reconstructing the summands and verifying the exact convex-hull equality \\(P=Q+R\\). Raising the limit also raises the search-state and time budgets.</p>"
});

function agNow(){return (typeof performance!=='undefined'&&performance.now)?performance.now():Date.now()}
function agBudgetMs(base,extras=[]){
  let total=base;
  for(const [value,baseline,step] of extras)total+=Math.max(0,value-baseline)*step;
  return Math.min(120000,Math.max(10000,total))
}
function agCheckDeadline(deadline,message='The requested computation reached its time budget.'){
  if(deadline&&agNow()>deadline)throw new Error(message)
}
function agExpDegree(e){return e.reduce((a,b)=>a+b,0)}
function agExpLeq(a,b){for(let i=0;i<a.length;i++)if(a[i]>b[i])return false;return true}
function agCompareExp(a,b){for(let i=0;i<a.length;i++){if(a[i]!==b[i])return a[i]-b[i]}return 0}
function agCanonicalRelationPair(a,b){
  return agCompareExp(a,b)<=0?{a:a.slice(),b:b.slice()}:{a:b.slice(),b:a.slice()}
}
function agRelationKey(a,b){const c=agCanonicalRelationPair(a,b);return `${agExpKey(c.a)}|${agExpKey(c.b)}`}

/* Exact Bareiss determinant over Z, used by the weighted Matrix-Tree theorem. */
function agBareissDetBig(matrix){
  const n=matrix.length;if(!n)return 1n;if(n===1)return BigInt(matrix[0][0]);
  const A=matrix.map(row=>row.map(BigInt));let sign=1n,den=1n;
  for(let k=0;k<n-1;k++){
    let pivot=k;while(pivot<n&&A[pivot][k]===0n)pivot++;
    if(pivot===n)return 0n;
    if(pivot!==k){[A[pivot],A[k]]=[A[k],A[pivot]];sign=-sign}
    const p=A[k][k];
    for(let i=k+1;i<n;i++)for(let j=k+1;j<n;j++)A[i][j]=(A[i][j]*p-A[i][k]*A[k][j])/den;
    den=p;
    for(let i=k+1;i<n;i++)A[i][k]=0n
  }
  return sign*A[n-1][n-1]
}
function agWeightedTreeCount(componentSizes){
  const sizes=componentSizes.map(BigInt),k=sizes.length;if(k<=1)return 1n;
  const total=sizes.reduce((a,b)=>a+b,0n),L=Array.from({length:k},()=>Array(k).fill(0n));
  for(let i=0;i<k;i++)for(let j=0;j<k;j++)L[i][j]=i===j?sizes[i]*(total-sizes[i]):-sizes[i]*sizes[j];
  return agBigAbs(agBareissDetBig(L.slice(0,k-1).map(row=>row.slice(0,k-1))))
}

/* v33 override: retain exact generating-fiber component data. */
function agToricGenerators(points,cap=200000){
  const A=agSortedPoints(points),n=A.length;
  if(n<2)return {support:A,quadrics:[],cubics:[],all:[],fiber2:null,fiber3:null,markovFibers:[]};
  const fiber2=agFiberData(A,2,cap),quadrics=[],markovFibers=[];
  for(const [fiber,monomials] of fiber2.fibers){
    if(monomials.length<2)continue;
    const components=monomials.map(m=>[m]),root=monomials[0];
    for(let i=1;i<monomials.length;i++)quadrics.push({degree:2,a:root.exp.slice(),b:monomials[i].exp.slice(),fiber});
    markovFibers.push({degree:2,fiber,components,required:components.length-1,choices:agWeightedTreeCount(components.map(c=>c.length))})
  }
  const fiber3=agFiberData(A,3,cap),index3=new Map(fiber3.monomials.map((m,i)=>[agExpKey(m.exp),i])),uf=agUnionFind(fiber3.monomials.length);
  for(const q of quadrics)for(let i=0;i<n;i++){
    const left=agAddExp(q.a,agUnitExp(n,i)),right=agAddExp(q.b,agUnitExp(n,i));
    const a=index3.get(agExpKey(left)),b=index3.get(agExpKey(right));if(a!==undefined&&b!==undefined)uf.join(a,b)
  }
  const cubics=[];
  for(const [fiber,monomials] of fiber3.fibers){
    const componentMap=new Map();
    for(const m of monomials){const c=uf.find(index3.get(agExpKey(m.exp)));if(!componentMap.has(c))componentMap.set(c,[]);componentMap.get(c).push(m)}
    const components=[...componentMap.values()];if(components.length<2)continue;
    const root=components[0][0];for(let i=1;i<components.length;i++)cubics.push({degree:3,a:root.exp.slice(),b:components[i][0].exp.slice(),fiber});
    markovFibers.push({degree:3,fiber,components,required:components.length-1,choices:agWeightedTreeCount(components.map(c=>c.length))})
  }
  const all=quadrics.concat(cubics);all.forEach((g,i)=>g.id=i);
  return {support:A,quadrics,cubics,all,fiber2,fiber3,markovFibers}
}
function agMarkovAnalysis(generators){
  let totalBases=1n,indispensable=[];
  for(const f of generators.markovFibers){
    totalBases*=f.choices;
    if(f.components.length===2&&f.components[0].length===1&&f.components[1].length===1){
      indispensable.push({degree:f.degree,a:f.components[0][0].exp.slice(),b:f.components[1][0].exp.slice(),fiber:f.fiber})
    }
  }
  return {fibers:generators.markovFibers,totalBases,unique:totalBases===1n,indispensable}
}
function agMarkovFiberTable(analysis){
  const rows=analysis.fibers.map(f=>`<tr><td>${f.degree}</td><td>${mathInline(agVectorLatex(f.fiber.split(',').map(Number)))}</td><td>${mathInline(agVectorLatex(f.components.map(c=>c.length)))}</td><td>${f.required}</td><td>${f.choices.toString()}</td></tr>`).join('');
  return `<table class="small-table"><thead><tr><th>degree</th><th>${mathInline('A')}–degree</th><th>lower-move component sizes</th><th>minimal moves</th><th>basis choices</th></tr></thead><tbody>${rows||'<tr><td colspan="5">No generating fibers.</td></tr>'}</tbody></table>`
}

async function agTruncatedGraverBasis(points,maxDegree,options={}){
  const A=agSortedPoints(points),n=A.length,deadline=options.deadline||0,monomialCap=options.monomialCap||500000,pairCap=options.pairCap||2000000;
  const candidates=new Map();let completeThrough=1,pairs=0,stopped=false,reason='';
  for(let d=2;d<=maxDegree;d++){
    await agPause();
    if(deadline&&agNow()>deadline){stopped=true;reason='time budget reached';break}
    const count=agChooseNumber(n+d-1,d);
    if(!Number.isFinite(count)||count>monomialCap){stopped=true;reason=`degree-${d} monomial count ${Number.isFinite(count)?count.toLocaleString():'overflow'} exceeds the adaptive cap`;break}
    const fiber=agFiberData(A,d,monomialCap),degreeCandidates=new Map();let degreeComplete=true;
    outer:for(const monomials of fiber.fibers.values())if(monomials.length>1){
      for(let i=0;i<monomials.length;i++)for(let j=i+1;j<monomials.length;j++){
        if((++pairs&2047)===0&&deadline&&agNow()>deadline){degreeComplete=false;reason='time budget reached';break outer}
        if(pairs>pairCap){degreeComplete=false;reason=`fiber-pair cap ${pairCap.toLocaleString()} reached`;break outer}
        const a=monomials[i].exp,b=monomials[j].exp,g=a.map((x,k)=>Math.min(x,b[k])),ra=a.map((x,k)=>x-g[k]),rb=b.map((x,k)=>x-g[k]),rd=agExpDegree(ra);
        if(rd<2||rd>maxDegree)continue;
        const c=agCanonicalRelationPair(ra,rb);degreeCandidates.set(agRelationKey(c.a,c.b),{degree:rd,a:c.a,b:c.b})
      }
    }
    if(!degreeComplete){stopped=true;break}
    for(const [key,value] of degreeCandidates)if(!candidates.has(key))candidates.set(key,value);
    completeThrough=d
  }
  const ordered=[...candidates.values()].filter(c=>c.degree<=completeThrough).sort((x,y)=>x.degree-y.degree||agRelationKey(x.a,x.b).localeCompare(agRelationKey(y.a,y.b))),primitive=[];
  for(const c of ordered){
    let decomposable=false;
    for(const g of primitive){
      if(g.degree>=c.degree)break;
      if((agExpLeq(g.a,c.a)&&agExpLeq(g.b,c.b))||(agExpLeq(g.b,c.a)&&agExpLeq(g.a,c.b))){decomposable=true;break}
    }
    if(!decomposable)primitive.push(c)
  }
  return {support:A,requestedDegree:maxDegree,completeThrough,primitive,stopped,reason,pairs}
}

/* v33 time-aware Koszul routines and explicit homological truncation. */
function agSparseRankMod(columns,p,operationCap=8000000,deadline=0){
  const pivots=new Map();let rank=0,operations=0,columnNumber=0;
  for(const source of columns){
    if((columnNumber++&255)===0)agCheckDeadline(deadline,'Koszul rank computation reached its time budget.');
    const vector=new Map();
    for(const [row,value] of source){const v=agMod((vector.get(row)||0)+value,p);if(v)vector.set(row,v);else vector.delete(row)}
    while(vector.size){
      let pivot=null;for(const row of vector.keys())if(pivot===null||row<pivot)pivot=row;
      const basis=pivots.get(pivot);
      if(!basis){const inv=agModInv(vector.get(pivot),p);for(const [row,value] of [...vector]){const v=value*inv%p;if(v)vector.set(row,v);else vector.delete(row)}pivots.set(pivot,vector);rank++;break}
      const factor=vector.get(pivot);for(const [row,value] of basis){const v=agMod((vector.get(row)||0)-factor*value,p);if(v)vector.set(row,v);else vector.delete(row)}
      operations+=basis.size;if(operations>operationCap)throw new Error(`Koszul sparse-rank operation guard ${operationCap.toLocaleString()} reached.`);
      if((operations&16383)===0)agCheckDeadline(deadline,'Koszul rank computation reached its time budget.')
    }
  }
  return {rank,operations}
}
function agKoszulDifferentialColumns(A,p,q,basisCache,columnCap,deadline=0,monomialCap=500000){
  if(p<=0||q<0)return {columns:[],sourceDimension:0,targetRows:0};
  agCheckDeadline(deadline,'Koszul matrix construction reached its time budget.');
  const basis=agSemigroupDegreeBasis(A,q,monomialCap,basisCache),subsetCount=agChooseNumber(A.length,p),sourceDimension=subsetCount*basis.length;
  if(!Number.isFinite(sourceDimension)||sourceDimension>columnCap)return {skipped:true,sourceDimension,reason:`${Number.isFinite(sourceDimension)?sourceDimension.toLocaleString():'Too many'} columns exceed the adaptive ${columnCap.toLocaleString()}-column guard`};
  const subsets=agCombinations(A.length,p),rowIds=new Map(),columns=[];let built=0;
  function rowId(key){if(!rowIds.has(key))rowIds.set(key,rowIds.size);return rowIds.get(key)}
  for(const subset of subsets)for(const weight of basis){
    if((built++&511)===0)agCheckDeadline(deadline,'Koszul matrix construction reached its time budget.');
    const column=[];for(let r=0;r<subset.length;r++){const i=subset[r],targetSubset=subset.slice(0,r).concat(subset.slice(r+1)),targetWeight=weight.map((x,j)=>x+A[i][j]);column.push([rowId(`${targetSubset.join('.')}|${agPointKey(targetWeight)}`),r%2?-1:1])}columns.push(column)
  }
  return {columns,sourceDimension,targetRows:rowIds.size}
}
function agKoszulBetti(A,options={}){
  A=agSortedPoints(A);const n=A.length,profile=agAutomaticProfile(n),basisCache=new Map(),affineRank=agAffineConfigurationRank(A),codimension=Math.max(0,n-affineRank);
  const maxQ=options.maxQ??2,primes=options.primes||[1000003,1000033],assumeCM=options.assumeCM!==false;
  const homologicalCeiling=assumeCM?codimension:Math.max(0,n-1),maxP=Math.min(homologicalCeiling,Math.max(0,options.maxP??homologicalCeiling));
  const columnCap=options.columnCap||profile.koszulColumnCap,operationCap=options.operationCap||profile.koszulOperationCap,deadline=options.deadline||0,monomialCap=options.monomialCap||profile.monomialCap;
  const differentialCache=new Map(),cells=[];
  function differential(p,q){
    const key=`${p}|${q}`;if(differentialCache.has(key))return differentialCache.get(key);
    if(p<=0||q<0){const z={available:true,ranks:primes.map(()=>0),consistent:true,sourceDimension:0};differentialCache.set(key,z);return z}
    let built;try{built=agKoszulDifferentialColumns(A,p,q,basisCache,columnCap,deadline,monomialCap)}catch(error){const z={available:false,sourceDimension:0,reason:error.message};differentialCache.set(key,z);return z}
    if(built.skipped){const z={available:false,...built};differentialCache.set(key,z);return z}
    const ranks=[];try{for(const prime of primes)ranks.push(agSparseRankMod(built.columns,prime,operationCap,deadline).rank)}catch(error){const z={available:false,sourceDimension:built.sourceDimension,reason:error.message};differentialCache.set(key,z);return z}
    const z={available:true,ranks,consistent:ranks.every(x=>x===ranks[0]),sourceDimension:built.sourceDimension,targetRows:built.targetRows};differentialCache.set(key,z);return z
  }
  for(let q=1;q<=maxQ;q++)for(let p=1;p<=maxP;p++){
    let basis;try{basis=agSemigroupDegreeBasis(A,q,monomialCap,basisCache)}catch(error){cells.push({p,q,available:false,chainDimension:0,reason:error.message});continue}
    const chainDimension=agChooseNumber(n,p)*basis.length,outgoing=differential(p,q),incoming=differential(p+1,q-1);
    if(!outgoing.available||!incoming.available){cells.push({p,q,available:false,chainDimension,reason:outgoing.reason||incoming.reason});continue}
    const values=primes.map((_,i)=>chainDimension-outgoing.ranks[i]-incoming.ranks[i]),valid=values.every(x=>Number.isInteger(x)&&x>=0);
    cells.push({p,q,available:valid,values,consistent:valid&&values.every(x=>x===values[0]),value:valid?values[0]:null,chainDimension,reason:valid?'':'rank computation produced an invalid homology dimension'})
  }
  return {support:A,n,affineRank,codimension,projectiveDimension:codimension,homologicalCeiling,maxP,maxQ,primes,cells,columnCap,operationCap,basisCache,assumeCM}
}
function agKoszulTableHtml(table,title='Koszul Betti table'){
  const cols=Array.from({length:table.maxP},(_,i)=>i+1);if(!cols.length)return '<p class="subtle">No nontrivial homological column was requested.</p>';
  const rows=[];for(let q=1;q<=table.maxQ;q++)rows.push(`<tr><th>${mathInline(`q=${q}`)}</th>${cols.map(p=>{const c=agKoszulCell(table,p,q);return `<td title="${agEscapeText(c?.reason||'')}">${c&&c.available?(c.consistent?c.value:`${c.values.join('/')}`):'—'}</td>`}).join('')}</tr>`);
  const computed=table.cells.filter(c=>c.available).length,total=table.cells.length,scope=table.maxP<table.homologicalCeiling?` through requested column ${table.maxP}`:` through column ${table.maxP}`;
  return `<table class="small-table ag-betti-table"><thead><tr><th>${mathInline('\\beta_{p,p+q}')}</th>${cols.map(p=>`<th>${p}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table><p class="small-note">${computed} of ${total} requested cells computed over ${table.primes.map(p=>mathInline(`\\mathbf F_{${p}}`)).join(' and ')}${scope}. A dash is an uncomputed cell, never a claimed zero. Agreement between the displayed characteristics is a consistency check, not a characteristic-zero proof.</p>`
}
function agPropertyNReport(table){
  let verified=0,firstNonzero=null,fieldDependent=false,unavailable=false;
  for(let p=1;p<=table.maxP;p++){
    const cell=agKoszulCell(table,p,2);
    if(!cell||!cell.available){unavailable=true;break}
    if(!cell.consistent){fieldDependent=true;break}
    if(cell.value!==0){firstNonzero=p;break}
    verified=p
  }
  const overFields=' over the displayed finite fields';
  if(fieldDependent)return 'not certified: the requested quadratic strand is characteristic-dependent in the tested fields';
  if(firstNonzero===1)return `${mathInline('N_0')} only${overFields}; cubic generators obstruct ${mathInline('N_1')}`;
  if(firstNonzero!==null)return `${mathInline(`N_{${firstNonzero-1}}`)} verified${overFields}, while ${mathInline(`\\beta_{${firstNonzero},${firstNonzero+2}}\\ne0`)} obstructs ${mathInline(`N_{${firstNonzero}}`)}`;
  if(verified===table.maxP)return `${mathInline(`N_{${verified}}`)} verified through the requested range${overFields}${table.maxP<table.homologicalCeiling?'; later columns were not requested':''}`;
  return verified?`${mathInline(`N_{${verified}}`)} verified${overFields} before ${unavailable?'an unavailable cell':'the requested range ended'}`:'not certified because the first requested quadratic-strand cell was unavailable'
}
function agLinearStrandSummary(table){
  const strand=table.cells.filter(c=>c.q===1);
  if(strand.some(c=>c.available&&!c.consistent))return 'the requested linear strand is characteristic-dependent in the tested fields';
  const known=strand.filter(c=>c.available&&c.consistent&&c.value>0);if(!known.length)return 'no consistent nonzero linear-strand entry was found in the requested range';
  const last=Math.max(...known.map(c=>c.p));return `${mathInline(`\\beta_{${last},${last+1}}`)} is the last consistent nonzero entry found through requested column ${table.maxP}`
}

function agScrollPresentation(canonicalSupport,direction){
  const A=agSortedPoints(canonicalSupport),index=new Map(A.map((p,i)=>[agPointKey(p),i])),e=[-direction[1],direction[0]],groups=new Map();
  for(const p of A){const level=dot(direction,p);if(!groups.has(level))groups.set(level,[]);groups.get(level).push(p)}
  const blocks=[],columns=[];
  for(const [level,pts0] of [...groups.entries()].sort((a,b)=>a[0]-b[0])){
    const pts=pts0.slice().sort((a,b)=>dot(e,a)-dot(e,b));
    for(let j=1;j<pts.length;j++){const diff=[pts[j][0]-pts[j-1][0],pts[j][1]-pts[j-1][1]];if(diff[0]!==e[0]||diff[1]!==e[1])throw new Error('Interior slice is not consecutive in the primitive kernel direction.')}
    const block=[];for(let j=0;j+1<pts.length;j++){const col={top:index.get(agPointKey(pts[j])),bottom:index.get(agPointKey(pts[j+1])),level,position:j};columns.push(col);block.push(col)}
    if(block.length)blocks.push(block)
  }
  const equations=[],seen=new Set();
  for(let i=0;i<columns.length;i++)for(let j=i+1;j<columns.length;j++){
    const a=Array(A.length).fill(0),b=Array(A.length).fill(0);a[columns[i].top]++;a[columns[j].bottom]++;b[columns[i].bottom]++;b[columns[j].top]++;
    if(agPointKey(agWeightedSum(A,a))!==agPointKey(agWeightedSum(A,b)))throw new Error('A proposed scroll minor failed exponent verification.');
    const c=agCanonicalRelationPair(a,b),key=agRelationKey(c.a,c.b);if(!seen.has(key)){seen.add(key);equations.push({a:c.a,b:c.b})}
  }
  const r=columns.length,linearBetti=[];for(let p=1;p<=Math.max(0,r-1);p++)linearBetti.push({p,value:p*agChooseNumber(r,p+1)});
  return {direction:direction.slice(),blocks,columns,equations,r,codimension:Math.max(0,r-1),linearBetti}
}
function agScrollBettiTable(scroll,maxP){
  const rows=scroll.linearBetti.filter(x=>x.p<=maxP).map(x=>`<tr><td>${x.p}</td><td>${x.value}</td></tr>`).join('');
  return `<table class="small-table"><thead><tr><th>${mathInline('p')}</th><th>${mathInline('\\beta^{\\mathrm{scroll}}_{p,p+1}=p\\binom{r}{p+1}')}</th></tr></thead><tbody>${rows||'<tr><td colspan="2">No nonzero scroll syzygy.</td></tr>'}</tbody></table>`
}

function agHornJacobianCertificate(gale,sampleCount=6,options={}){
  const r=gale.rank,primes=options.primes||[1000003,1000033],deadline=options.deadline||0,expected=Math.max(0,r-1),results=[];
  const zeroRows=gale.rows.map((row,i)=>row.every(x=>x===0n)?i:-1).filter(i=>i>=0);
  if(zeroRows.length)return {rank:r,expected,results,certified:false,pyramid:true,zeroRows,status:`Zero Gale row(s) ${zeroRows.map(i=>i+1).join(', ')} certify that the configuration is a pyramid; its dual is not a discriminant hypersurface, so no Horn hypersurface-rank certificate is applicable.`};
  if(r<=1)return {rank:r,expected,results,certified:true,pyramid:false,status:'The reduced source has dimension zero, so the expected rank is automatic.'};
  for(const prime of primes){
    let best=0,valid=0;
    for(let trial=0;trial<sampleCount*8&&valid<sampleCount;trial++){
      agCheckDeadline(deadline,'Horn-rank sampling reached its time budget.');
      const u=Array.from({length:r},(_,j)=>{let x=agMod((trial+2)*104729+(j+3)*130363+(trial+1)*(j+1)*15485863,prime);x=agMod(x*x+(trial+5)*(j+7)+32452843,prime);return x||1}),linear=[];let good=true;
      for(const row of gale.rows){let value=0;for(let j=0;j<r;j++)value=agMod(value+Number(agMod(Number(row[j]%BigInt(prime)),prime))*u[j],prime);if(!value){good=false;break}linear.push(value)}
      if(!good)continue;valid++;
      const J=Array.from({length:r},()=>Array(r).fill(0));
      for(let i=0;i<gale.rows.length;i++){
        const inv=agModInv(linear[i],prime),row=gale.rows[i].map(x=>agMod(Number(x%BigInt(prime)),prime));
        for(let j=0;j<r;j++)for(let k=0;k<r;k++)J[j][k]=agMod(J[j][k]+row[j]*row[k]%prime*inv,prime)
      }
      best=Math.max(best,agRankMod(J,prime));if(best===expected)break
    }
    results.push({prime,best,valid})
  }
  const certified=results.some(x=>x.best===expected),status=certified?`Rank ${expected} was attained modulo ${results.filter(x=>x.best===expected).map(x=>x.prime).join(', ')}, certifying a non-defective reduced discriminant over characteristic zero.`:`No sampled point attained rank ${expected}; the observed ranks are not a proof of dual defect.`;
  return {rank:r,expected,results,certified,pyramid:false,status}
}

function agParameterLabel(name,inputHtml){return `<label data-help-decorated="1">${helpLabel(name)}${inputHtml}</label>`}
function agModuleShell(id,title,lead,parameters=''){
  return `<details class="formula ag-module" id="${id}"><summary>${helpLabel(title)}</summary><div class="ag-module-body"><p class="ag-module-lead">${lead}</p><div class="compute-panel ag-compute-panel">${parameters}<button type="button" class="compute-btn" data-ag-compute="${id}">Compute</button></div><div class="compute-result ag-result" data-ag-result="${id}"><p class="subtle">No calculation has been run. This module is computed on demand.</p></div></div></details>`
}
function agLegacyDropdowns(s){
  return `<details class="formula"><summary>Adjoint tower and subcurve polygons</summary><div class="figure-box adjoint-tower-inline">${svgAdjointTower(s.v)}</div><table class="small-table"><thead><tr><th>level</th><th>vertices</th><th>Vol</th><th>B</th><th>I</th><th>N</th><th>polygon search</th></tr></thead><tbody>${algebraicSubcurveRows(s)}</tbody></table></details><details class="formula"><summary>Curve-counting and mirror-symmetry computations</summary>${curveCountingRows(s)}<div class="compute-panel"><label>Mirror period support<select id="period-support"><option value="all" selected>all lattice points</option><option value="vertices">vertices only</option><option value="boundary">boundary lattice points</option></select></label><label>max order<input id="period-order" type="number" min="1" value="16"></label><button type="button" id="compute-period" class="compute-btn">Compute CT period</button><button type="button" id="compute-pf" class="compute-btn">Compute PF / mirror map</button><button type="button" id="stop-pf" class="compute-btn stop-btn">Stop PF</button></div><div id="curve-compute-result" class="compute-result"></div></details>${detail('Background: curves, adjunction, toric MMP and periods',`The adjoint tower repeatedly replaces a polygon by the convex hull of its strict interior lattice points. Width-spectrum rows record the actual primitive directions and slice sizes, not just a direct rewrite of one basic count. The period button computes exact constant terms ${mathInline('\\operatorname{CT}(f^k)')} after translating by the selected interior origin.`)}`
}
function algebraicSection(s){
  const aParams=agParameterLabel('Betti columns','<input id="ag-a-max-p" type="number" min="1" max="30" value="8">')+agParameterLabel('Primitive-binomial degree bound','<input id="ag-a-graver-degree" type="number" min="2" max="12" value="5">');
  const bParams=agParameterLabel('Betti columns','<input id="ag-b-max-p" type="number" min="1" max="30" value="8">');
  const cParams=agParameterLabel('Horn rank samples','<input id="ag-c-horn-samples" type="number" min="2" max="40" value="6">');
  return `<div class="card"><h3>Algebraic geometry and curve-counting data</h3><div class="ag-module-grid">${agModuleShell('ag-module-a','Equations, toric ideals and syzygies','Computes exact generating fibers and minimal Markov-basis rigidity, a user-truncated primitive Graver basis, and graded Koszul homology.',aParams)}${agModuleShell('ag-module-b','Canonical curve, pencils and scrolls','Computes the canonical toric envelope, its requested Koszul columns, and explicit determinantal scroll equations attached to minimum-width pencils.',bParams)}${agModuleShell('ag-module-c','GKZ system and discriminant strata','Constructs circuits, a saturated Gale dual, GKZ operators and discriminants, then seeks an exact modular full-rank certificate for the Horn–Kapranov map.',cParams)}</div>${agLegacyDropdowns(s)}</div>`
}

async function computeAGModuleA(v,root){
  const out=root.querySelector('[data-ag-result="ag-module-a"]');if(!out)return;
  const maxP=agInt(root.querySelector('#ag-a-max-p')?.value,1,30,8),graverDegree=agInt(root.querySelector('#ag-a-graver-degree')?.value,2,12,5),budget=agBudgetMs(10000,[[maxP,8,3000],[graverDegree,5,6000]]),deadline=agNow()+budget;
  out.innerHTML='<p class="empty">Computing generating fibers, primitive binomials and Koszul homology...</p>';await agPause();
  try{
    const A=agSortedPoints(latticePoints(v).all),profile=agAutomaticProfile(A.length),scale=Math.max(1,1+(maxP-8)/4+(graverDegree-5)/3),monomialCap=Math.min(1500000,Math.round(profile.monomialCap*scale));
    const generators=agToricGenerators(A,monomialCap),markov=agMarkovAnalysis(generators),circuits=agCircuits(A,profile.circuitDisplay,Math.min(1200000,Math.round(profile.circuitSubsetCap*scale)));
    const graver=await agTruncatedGraverBasis(A,graverDegree,{deadline,monomialCap,pairCap:Math.min(10000000,Math.round(2000000*scale))});
    const betti=agKoszulBetti(A,{maxQ:2,maxP,columnCap:Math.min(120000,Math.round(profile.koszulColumnCap*scale)),operationCap:Math.min(80000000,Math.round(profile.koszulOperationCap*scale)),deadline,monomialCap,assumeCM:true});
    const equations=generators.all.map(g=>mathInline(`${agBinomialLatex(g,'X')}=0`)),indispensable=markov.indispensable.map(g=>mathInline(`${agBinomialLatex(g,'X')}=0`)),graverRows=graver.primitive.map(g=>mathInline(`${agBinomialLatex(g,'X')}=0`));
    const circuitRows=circuits.circuits.map((c,i)=>`${mathInline(`\\ell^{(${i+1})}=${agVectorLatex(c.full)}`)};&nbsp; ${mathInline(`${agCircuitBinomial(c,'X')}=0`)}`);
    out.innerHTML=`${kvTable([
      ['Minimal toric equations',`${generators.quadrics.length} quadratic and ${generators.cubics.length} essential cubic generator(s)`],
      ['Markov-fiber rigidity',`${markov.fibers.length} generating fiber(s); ${markov.indispensable.length} indispensable binomial(s); ${markov.unique?'unique minimal Markov basis':`${markov.totalBases.toString()} minimal Markov bases`}`],
      ['Truncated Graver basis',`${graver.primitive.length} primitive binomial(s), certified through degree ${graver.completeThrough}${graver.stopped?`; stopped before completing degree ${graver.completeThrough+1} (${graver.reason})`:''}`],
      ['Graded Koszul cohomology',`${agLinearStrandSummary(betti)}; ${agPropertyNReport(betti)}`],
      ['Primitive affine circuits',`${circuits.totalCircuits.toLocaleString()} circuit(s) found${circuits.truncated?' before the adaptive subset guard':''}`]
    ])}<details class="formula"><summary>${helpLabel('Minimal toric equations')}</summary>${agDisplayList(equations,40)}</details><details class="formula"><summary>${helpLabel('Markov-fiber rigidity')}</summary>${agMarkovFiberTable(markov)}<p class="subtle">The exact total counts all spanning-tree choices between lower-move components. A binomial is indispensable precisely in a generating fiber consisting of two singleton components.</p>${indispensable.length?`<h5>Indispensable binomials</h5>${agDisplayList(indispensable,30)}`:''}</details><details class="formula"><summary>${helpLabel('Truncated Graver basis')}</summary>${agDisplayList(graverRows,40)}<p class="subtle">Only degrees at most ${graver.completeThrough} are certified. Primitive means conformally indecomposable; this is stronger than being a circuit or a minimal Markov generator.</p></details><details class="formula"><summary>${helpLabel('Koszul Betti table')}</summary>${agKoszulTableHtml(betti)}<p class="subtle">For the normal polygon semigroup ring, Cohen–Macaulayness justifies the homological ceiling used here.</p></details><details class="formula"><summary>${helpLabel('Primitive affine circuits')}</summary>${agDisplayList(circuitRows,profile.circuitDisplay)}</details><p class="small-note">Requested budget: ${(budget/1000).toFixed(0)} s. Variable order: ${A.map((p,i)=>`${mathInline(`X_{${i+1}}`)}↔${mathInline(agVectorLatex(p))}`).join(', ')}.</p>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}

async function computeAGModuleB(v,root){
  const out=root.querySelector('[data-ag-result="ag-module-b"]');if(!out)return;
  const maxP=agInt(root.querySelector('#ag-b-max-p')?.value,1,30,8),budget=agBudgetMs(10000,[[maxP,8,4000]]),deadline=agNow()+budget;
  out.innerHTML='<p class="empty">Computing canonical-envelope syzygies and determinantal scroll presentations...</p>';await agPause();
  try{
    const pointData=latticePoints(v),canonicalSupport=agSortedPoints(pointData.inn),interiorDimension=interiorPolygonDimensionValue(v),profile=agAutomaticProfile(canonicalSupport.length),scale=Math.max(1,1+(maxP-8)/4);
    const envelope=canonicalSupport.length>=2?agToricGenerators(canonicalSupport,Math.min(1000000,Math.round(profile.monomialCap*scale))):null;
    const canonicalBetti=canonicalSupport.length>=2?agKoszulBetti(canonicalSupport,{maxQ:2,maxP,columnCap:Math.min(100000,Math.round(profile.koszulColumnCap*scale)),operationCap:Math.min(70000000,Math.round(profile.koszulOperationCap*scale)),deadline,assumeCM:true}):null;
    const directions=agMinWidthDirections(v),aut=automorphisms(v),orbits=agDirectionOrbits(directions,aut.maps),pencils=directions.map(u=>agWidthInvariantData(v,u));
    const pencilRows=pencils.map((p,i)=>`<tr><td>${i+1}</td><td>${mathInline(agVectorLatex(p.direction))}</td><td>${mathInline(`\\pi_{${i+1}}=[1:x^{${p.direction[0]}}y^{${p.direction[1]}}]`)}</td><td>${p.degree}</td><td>${mathInline(`(${p.widthInvariants.join(',')})`)}</td><td>${mathInline(`(${p.scrollar.join(',')})`)}</td><td>${interiorDimension===2?(p.complete?'complete':'incomplete'):'not asserted'}</td></tr>`).join('');
    const equations=envelope?envelope.all.map(g=>mathInline(`${agBinomialLatex(g,'Z')}=0`)):[],scrollCards=[];
    if(interiorDimension===2){
      for(let i=0;i<orbits.length;i++){
        agCheckDeadline(deadline,'Scroll-presentation computation reached its time budget.');
        const scroll=agScrollPresentation(canonicalSupport,orbits[i][0]),scrollEquations=scroll.equations.map(g=>mathInline(`${agBinomialLatex(g,'Z')}=0`));
        scrollCards.push(`<div class="decomp-card"><h5>pencil orbit ${i+1}: ${mathInline(`u=${agVectorLatex(scroll.direction)}`)}</h5>${kvTable([['Scroll matrix columns',scroll.r],['Verified determinantal quadrics',scroll.equations.length],['Scroll Eagon–Northcott strand',mathInline(`\\beta^{\\mathrm{scroll}}_{p,p+1}=p\\binom{${scroll.r}}{p+1}`)]])}<details class="formula"><summary>${helpLabel('Determinantal scroll presentations')}</summary>${agDisplayList(scrollEquations,40)}</details><details class="formula"><summary>${helpLabel('Scroll Eagon–Northcott strand')}</summary>${agScrollBettiTable(scroll,maxP)}</details></div>`)
      }
    }
    const envelopeText=!canonicalSupport.length?'empty canonical support':canonicalSupport.length===1?'one canonical monomial':`${envelope.quadrics.length} quadratic and ${envelope.cubics.length} essential cubic equation(s)`;
    out.innerHTML=`${kvTable([
      ['Canonical toric envelope',envelopeText],
      ['Canonical-envelope Koszul syzygies',canonicalBetti?`${agLinearStrandSummary(canonicalBetti)}; ${agPropertyNReport(canonicalBetti)}`:'not defined for an empty or one-point support'],
      ['Minimum-width pencil package',`${pencils.length} exact direction(s) in ${orbits.length} automorphism orbit(s)`],
      ['Determinantal scroll presentations',interiorDimension===2?`${scrollCards.length} orbit representative(s) converted to verified scroll minors`:'not asserted because the interior configuration is not two-dimensional']
    ])}<details class="formula"><summary>${helpLabel('Canonical-envelope equations')}</summary>${agDisplayList(equations,30)}<p class="subtle">These are equations of the toric envelope. A coefficient-specific canonical curve generally has additional equations.</p></details>${canonicalBetti?`<details class="formula"><summary>${helpLabel('Canonical-envelope Koszul table')}</summary>${agKoszulTableHtml(canonicalBetti)}</details>`:''}<details class="formula"><summary>${helpLabel('Minimum-width pencils')}</summary><table class="small-table"><thead><tr><th>#</th><th>direction</th><th>monomial map</th><th>degree</th><th>width invariants</th><th>scrollar invariants</th><th>completeness</th></tr></thead><tbody>${pencilRows||'<tr><td colspan="7">none</td></tr>'}</tbody></table></details><details class="formula"><summary>${helpLabel('Pencil orbits under polygon automorphisms')}</summary>${agDisplayList(orbits.map((orbit,i)=>`${i+1}. ${orbit.map(agVectorLatex).map(mathInline).join(', ')}`),30)}</details>${scrollCards.length?`<details class="formula"><summary>${helpLabel('Determinantal scroll presentations')}</summary>${scrollCards.join('')}</details>`:''}<p class="small-note">Requested budget: ${(budget/1000).toFixed(0)} s.</p>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}

async function computeAGModuleC(v,root){
  const out=root.querySelector('[data-ag-result="ag-module-c"]');if(!out)return;
  const samples=agInt(root.querySelector('#ag-c-horn-samples')?.value,2,40,6),budget=agBudgetMs(10000,[[samples,6,1500]]),deadline=agNow()+budget;
  out.innerHTML='<p class="empty">Computing circuits, a saturated Gale dual and a Horn differential-rank certificate...</p>';await agPause();
  try{
    const A=agSortedPoints(latticePoints(v).all),profile=agAutomaticProfile(A.length),circuits=agCircuits(A,profile.circuitDisplay,profile.circuitSubsetCap),gale=agGaleDual(A),horn=agHornCoordinates(gale,20),rankCertificate=agHornJacobianCertificate(gale,samples,{deadline});
    const generators=agToricGenerators(A,profile.monomialCap),operators=generators.all.map((g,i)=>{const full=g.a.map((x,j)=>x-g.b[j]),c={full};return `${mathInline(`\\ell^{(${i+1})}=${agVectorLatex(full)}`)}<br>${mathInline(agBoxOperator(c,`\\ell^{(${i+1})}`))}`});
    const discriminants=circuits.circuits.map((c,i)=>mathInline(`\\Delta_{C_{${i+1}}}=${agCircuitDiscriminant(c)}`)),eulers=agEulerOperators(A).map(mathInline),hornForms=horn.forms?horn.forms.map((f,i)=>mathInline(`L_{${i+1}}=${f}`)):[],hornCoordinates=horn.coordinates.map(mathInline);
    const rankRows=rankCertificate.results.map(x=>`<tr><td>${mathInline(`\\mathbf F_{${x.prime}}`)}</td><td>${x.valid}</td><td>${x.best}</td><td>${rankCertificate.expected}</td></tr>`).join('');
    out.innerHTML=`${kvTable([
      ['Saturated Gale dual',`${mathInline('\\widetilde A B=0')} verified; rank ${gale.rank}`],
      ['Horn–Kapranov rank certificate',rankCertificate.status],
      ['GKZ Markov-basis box operators',`${operators.length} operator(s) from the complete degree-two/three Markov basis of the polygon configuration`],
      ['Circuit-discriminant strata',`${circuits.totalCircuits.toLocaleString()} circuit(s) found; ${discriminants.length} retained for display`],
      ['Universal Laurent family',mathInline(agUniversalLaurentLatex(A))]
    ])}<details class="formula"><summary>${helpLabel('Saturated Gale dual')}</summary>${agGaleRowsHtml(gale)}</details><details class="formula"><summary>${helpLabel('Horn–Kapranov reduced-discriminant chart')}</summary>${agDisplayList(hornForms,40)}${agDisplayList(hornCoordinates,30)}</details><details class="formula"><summary>${helpLabel('Horn–Kapranov rank certificate')}</summary><table class="small-table"><thead><tr><th>field</th><th>valid samples</th><th>best rank</th><th>expected rank</th></tr></thead><tbody>${rankRows||'<tr><td colspan="4">No positive-dimensional reduced chart.</td></tr>'}</tbody></table><p class="subtle">The matrix tested is ${mathInline('J(u)=B^{\\mathsf T}\\operatorname{diag}((Bu)^{-1})B')}. Full rank is a certificate; failure to attain it is deliberately inconclusive.</p></details><details class="formula"><summary>${helpLabel('GKZ box operators')}</summary>${agDisplayList(operators,50)}</details><details class="formula"><summary>${helpLabel('GKZ Euler operators')}</summary>${agDisplayList(eulers,10)}</details><details class="formula"><summary>${helpLabel('Circuit discriminants')}</summary>${agDisplayList(discriminants,profile.circuitDisplay)}</details><p class="small-note">Requested budget: ${(budget/1000).toFixed(0)} s. Coefficient order: ${A.map((p,i)=>`${mathInline(`c_{${i+1}}`)}↔${mathInline(agVectorLatex(p))}`).join(', ')}.</p>`;
    renderMath(out)
  }catch(error){agError(out,error)}
}

function edgeSplitDecompositionsTimed(v,limit=12,cap=500000,deadline=0){
  v=hull(v);const ev=edgeVectors(v),lens=edges(v),dirs=ev.map((e,i)=>[e[0]/lens[i],e[1]/lens[i]]),n=lens.length,out=[],seen=new Set(),q=Array(n).fill(0);let visited=0,stopped=false,reason='';
  function rec(i,sx,sy){
    if(stopped)return;if((visited++&2047)===0&&deadline&&agNow()>deadline){stopped=true;reason='time budget';return}if(visited>cap){stopped=true;reason='state cap';return}
    if(i===n){if(sx||sy)return;if(q.every(x=>x===0)||q.every((x,j)=>x===lens[j]))return;const r=lens.map((L,j)=>L-q[j]),Q=polygonFromSplitDirs(dirs,q),R=polygonFromSplitDirs(dirs,r);if(!Q||!R)return;if(keyOf(minkowskiSumPoly(Q,R))!==keyOf(v))return;const k=[keyOf(Q),keyOf(R)].sort().join('||');if(seen.has(k))return;seen.add(k);out.push({Q,R,q:q.slice(),r,VolQ:area2(Q),VolR:area2(R)});if(out.length>=limit){stopped=true;reason='requested limit'}return}
    for(let t=0;t<=lens[i];t++){q[i]=t;rec(i+1,sx+dirs[i][0]*t,sy+dirs[i][1]*t);if(stopped)return}
  }
  rec(0,0,0);return {out,stopped,visited,reason}
}
async function computeMinkowskiDecompositions(v,root=document){
  const box=root.querySelector('#minkowski-result');if(!box)return;const limit=agInt(root.querySelector('#minkowski-limit')?.value,1,80,12),budget=agBudgetMs(10000,[[limit,12,1200]]),deadline=agNow()+budget;
  box.innerHTML='<p class="empty">Searching exact segment quotients and full-dimensional edge splittings...</p>';await agPause();
  const P=latticePoints(v),segs=segmentSummands(v,P,limit),remain=Math.max(1,limit-segs.length),stateCap=Math.min(5000000,Math.max(600000,limit*60000)),res=edgeSplitDecompositionsTimed(v,remain,stateCap,deadline),rows=[];
  const matchText=ms=>ms.length?ms.map(r=>`<a href="#" data-open="${r[MIDX.id]}">${id(r[MIDX.id])}</a>`).join(', '):'none found';
  for(let i=0;i<Math.min(segs.length,limit)&&agNow()<deadline;i++){
    const d=segs[i],Q=d.quotientVertices,qIsPolygon=Q.length>=3&&area2(Q)>0,matchesQ=qIsPolygon?await findEquivalent(Q,4):[];
    rows.push(`<div class="decomp-card"><h4>Decomposition ${rows.length+1}: ${mathInline('P=Q+[0,u]')}</h4><div class="visual-grid"><div><h5>quotient summand Q</h5><div class="figure-tools">${copyButton(Q)}</div><div class="figure-box">${qIsPolygon?svgPolygon(Q,[],{large:false}):svgVectorPlot(Q.length===2?[Q[0],Q[1]]:[[0,0]],[1,1],'One-dimensional quotient summand')}</div></div><div><h5>segment summand [0,u]</h5><div class="figure-box">${svgVectorPlot([d.u],[1],'Segment summand')}</div></div></div>${kvTable([['Segment vector \\(u\\)',`(${d.u[0]},${d.u[1]})`],['Normalized mixed intersection',mathInline(`D_Q\\cdot D_R=\\frac{${area2(v)}-${qIsPolygon?area2(Q):0}-0}{2}`)],['Database matches for Q',qIsPolygon?matchText(matchesQ):'one-dimensional; not searched']])}</div>`);await agPause()
  }
  for(let i=0;i<res.out.length&&rows.length<limit&&agNow()<deadline;i++){
    const d=res.out[i],matchesQ=await findEquivalent(d.Q,4),matchesR=await findEquivalent(d.R,4),mixed=(area2(v)-d.VolQ-d.VolR)/2;
    rows.push(`<div class="decomp-card"><h4>Decomposition ${rows.length+1}: ${mathInline('P=Q+R')}</h4><div class="visual-grid"><div><h5>summand Q</h5><div class="figure-tools">${copyButton(d.Q)}</div><div class="figure-box">${svgPolygon(d.Q,[],{large:false})}</div></div><div><h5>summand R</h5><div class="figure-tools">${copyButton(d.R)}</div><div class="figure-box">${svgPolygon(d.R,[],{large:false})}</div></div></div>${kvTable([['Normalized areas',`Vol(Q)=${d.VolQ}, Vol(R)=${d.VolR}`],['Mixed intersection',mathInline(`D_Q\\cdot D_R=${mixed}`)],['Database matches for Q',matchText(matchesQ)],['Database matches for R',matchText(matchesR)]])}</div>`);await agPause()
  }
  if(!rows.length)box.innerHTML=`<p class="empty">No exact Minkowski summand was found. The full-dimensional search visited ${res.visited.toLocaleString()} states${res.stopped?` before stopping at the ${res.reason}`:''}.</p>`;
  else box.innerHTML=`<p class="small-note">Found ${rows.length} exact decomposition(s). Search visited ${res.visited.toLocaleString()} full-dimensional states${res.stopped?` and stopped at the ${res.reason}`:''}; requested budget ${(budget/1000).toFixed(0)} s.</p>${rows.join('')}`;
  attachCopyButtons(box);attachOpenLinks(box);renderMath(box)
}

function minkowskiSection(s,P){
  const equivariantParams=agParameterLabel('Invariant degree bound','<input id="minkowski-invariant-degree" type="number" min="2" max="10" value="4">')+agParameterLabel('Cayley Betti columns','<input id="minkowski-cayley-p" type="number" min="1" max="20" value="5">');
  return `<div class="card"><h3>Minkowski, Cayley and deformation</h3><details class="formula" open><summary>${helpLabel('Minkowski decomposition')}</summary><div class="compute-panel">${agParameterLabel('Maximum decompositions','<input id="minkowski-limit" type="number" min="1" max="80" value="12">')}<button type="button" id="compute-minkowski" class="compute-btn">Recompute Minkowski decompositions</button></div><div id="minkowski-result" class="compute-result"><p class="empty">Preparing the default exact decomposition search...</p></div></details><details class="formula ag-module" id="minkowski-equivariant"><summary>${helpLabel('Equivariant algebra and deformation presentations')}</summary><div class="ag-module-body"><p class="ag-module-lead">Computes automorphism characters and attaches coefficient-convolution, Cayley-ideal, Jacobian-rank and requested Koszul columns to verified Minkowski sums.</p><div class="compute-panel ag-compute-panel">${equivariantParams}<button type="button" class="compute-btn" data-ag-compute="minkowski-equivariant">Compute</button></div><div class="compute-result ag-result" data-ag-result="minkowski-equivariant"><p class="subtle">No calculation has been run. This module is computed on demand.</p></div></div></details></div>`
}

async function computeMinkowskiEquivariant(v,root){
  const out=root.querySelector('[data-ag-result="minkowski-equivariant"]');if(!out)return;
  const maxDegree=agInt(root.querySelector('#minkowski-invariant-degree')?.value,2,10,4),maxP=agInt(root.querySelector('#minkowski-cayley-p')?.value,1,20,5),limit=agInt(root.querySelector('#minkowski-limit')?.value,1,40,12),budget=agBudgetMs(10000,[[maxDegree,4,4000],[maxP,5,3000],[limit,12,1000]]),deadline=agNow()+budget;
  out.innerHTML='<p class="empty">Computing equivariant characters and Minkowski–Cayley deformation packages...</p>';await agPause();
  try{
    const equivariant=agEquivariantCharacters(v,maxDegree),invariantRows=equivariant.invariants.map(r=>`<tr><td>${r.degree}</td><td>${r.symmetric}</td><td>${r.ring}</td><td>${r.ideal}</td></tr>`).join(''),characterRows=equivariant.elements.map((e,i)=>`<tr><td>${i+1}</td><td>${e.T.det}</td><td>${mathInline(agVectorLatex(e.cycles))}</td><td>${e.symmetric[2]}</td><td>${e.ring[2]}</td><td>${e.ideal[2]}</td></tr>`).join('');
    const pointData=latticePoints(v),decompositions=[],seen=new Set();function add(item){if(!item||decompositions.length>=limit||keyOf(minkowskiSumPoly(item.Q,item.R))!==keyOf(v))return;const key=[keyOf(item.Q),keyOf(item.R)].sort().join('||');if(!seen.has(key)){seen.add(key);decompositions.push(item)}}
    add(agZonotopePair(v,pointData));add(agHomotheticPair(v));for(const item of segmentSummands(v,pointData,limit))add({Q:item.quotientVertices,R:[[0,0],item.u],type:'segment summand'});
    const split=edgeSplitDecompositionsTimed(v,Math.max(0,limit-decompositions.length),Math.min(4000000,Math.max(500000,limit*50000)),deadline);for(const item of split.out)add({Q:item.Q,R:item.R,type:'full-dimensional split'});
    const cards=[];
    for(let i=0;i<decompositions.length;i++){
      agCheckDeadline(deadline,'Minkowski–Cayley package reached its time budget.');const d=decompositions[i],support=agCayleySupport(d.Q,d.R),profile=agAutomaticProfile(support.length),lowGenerators=agToricGenerators(support,profile.monomialCap),cayleyEquations=lowGenerators.all.slice(0,16).map(g=>mathInline(`${agMonomialLatex(g.a,'U')}-${agMonomialLatex(g.b,'U')}=0`)),convolution=agMinkowskiConvolutionEquations(d.Q,d.R),convolutionEquations=convolution.equations.slice(0,18).map(mathInline),jacobian=agConvolutionJacobianRanks(convolution);
      const table=agKoszulBetti(support,{maxQ:2,maxP,assumeCM:false,columnCap:Math.min(50000,Math.max(9000,profile.koszulColumnCap*2)),operationCap:Math.min(30000000,Math.max(6500000,profile.koszulOperationCap*2)),deadline});
      cards.push(`<div class="decomp-card"><h5>${i+1}. ${d.type}: ${mathInline('P=Q+R')}</h5>${kvTable([['Coefficient-convolution presentation',`${convolution.entries.length} exact coefficient equation(s)`],['Sampled differential rank',`${jacobian.ranks.join('/')} over ${jacobian.primes.map(p=>mathInline(`\\mathbf F_{${p}}`)).join(', ')}; expected ${jacobian.expected}`],['Cayley low-degree ideal',`${lowGenerators.quadrics.length} quadratic and ${lowGenerators.cubics.length} cubic relation(s)`]])}<details class="formula"><summary>${helpLabel('Coefficient convolution equations')}</summary>${agDisplayList(convolutionEquations,18)}</details><details class="formula"><summary>${helpLabel('Cayley low-degree equations')}</summary>${agDisplayList(cayleyEquations,16)}</details><details class="formula"><summary>${helpLabel('Cayley low-degree Koszul strands')}</summary>${agKoszulTableHtml(table)}<p class="subtle">No Cohen–Macaulay hypothesis is used to truncate at the codimension; only the requested columns are displayed.</p></details></div>`)
    }
    out.innerHTML=`${kvTable([['Equivariant section-ring character',`${equivariant.aut.order} affine lattice automorphism(s), through degree ${maxDegree}`],['Invariant equation spaces',mathInline('\\dim I_d^G=|G|^{-1}\\sum_{g\\in G}\\chi_{I_d}(g)')],['Verified deformation presentations',`${decompositions.length} exact Minkowski package(s)`]])}<details class="formula"><summary>${helpLabel('Invariant Hilbert coefficients')}</summary><table class="small-table"><thead><tr><th>${mathInline('d')}</th><th>${mathInline('\\dim(\\operatorname{Sym}^dH^0(L))^G')}</th><th>${mathInline('\\dim H^0(dL)^G')}</th><th>${mathInline('\\dim I_d^G')}</th></tr></thead><tbody>${invariantRows}</tbody></table></details><details class="formula"><summary>${helpLabel('Automorphism character rows in degree two')}</summary><table class="small-table"><thead><tr><th>element</th><th>det</th><th>cycles</th><th>${mathInline('\\chi_{\\operatorname{Sym}^2}')}</th><th>${mathInline('\\chi_{R_2}')}</th><th>${mathInline('\\chi_{I_2}')}</th></tr></thead><tbody>${characterRows}</tbody></table></details><details class="formula"><summary>${helpLabel('Minkowski–Cayley deformation presentations')}</summary>${cards.join('')||'<p class="subtle">No nontrivial exact decomposition was found within the requested range.</p>'}</details><p class="small-note">Requested budget: ${(budget/1000).toFixed(0)} s.</p>`;renderMath(out)
  }catch(error){agError(out,error)}
}

function attachAlgebraicHeavyControls(v,root=document){
  root.querySelectorAll('[data-ag-compute]').forEach(button=>{if(button.dataset.agBound)return;button.dataset.agBound='1';button.onclick=()=>{const id=button.dataset.agCompute;if(id==='ag-module-a')computeAGModuleA(v,root);else if(id==='ag-module-b')computeAGModuleB(v,root);else if(id==='ag-module-c')computeAGModuleC(v,root);else if(id==='minkowski-equivariant')computeMinkowskiEquivariant(v,root)}})
}
