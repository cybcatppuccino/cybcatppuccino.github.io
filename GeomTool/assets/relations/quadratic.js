/* GeomTool v4.1: dimension-two metric algebra with source-aware reduction.
 * All measurements are in length² units; no ratio is made with zero or
 * near-zero operands. Equal-length substitution is not a new theorem. */
(function(root,factory){const api=factory();root.GeomRelationQuadratic=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const multipliers=[{v:1,label:'1'},{v:2,label:'2'},{v:3,label:'3'},{v:4,label:'4'},{v:1.5,label:'3/2'},{v:4/3,label:'4/3'},{v:Math.SQRT2,label:'√2'},{v:Math.sqrt(3),label:'√3'},{v:(1+Math.sqrt(5))/2,label:'φ'}];
function lengthClasses(c){
 const ls=[...c.segments.map(s=>({key:s.key,value:s.value})),...(c.CAll||c.C).map(q=>({key:'R:'+q.id,value:q.r}))].sort((a,b)=>a.value-b.value||a.key.localeCompare(b.key));
 const groups=[],byKey=new Map();for(const m of ls){let g=groups[groups.length-1];if(!g||!c.nearRatio(m.value,g[0].value,1))groups.push(g=[]);g.push(m);}
 for(const g of groups)for(const m of g)byKey.set(m.key,g[0].key);
 return byKey;
}
function collect(c){
 const metrics=[],add=(key,name,value,type,marks,aliases=[],factors=null)=>{
  if(!Number.isFinite(value)||Math.abs(value)<=c.eps*Math.max(c.localScale,c.eps)*8)return;
  const entry={key,name,value,type,marks,aliases:[name,...aliases],factors};metrics.push(entry);return entry;
 };
 const classes=lengthClasses(c);
 const segs=c.segments;
 for(const s of segs){const name=c.display(s.a)+c.display(s.b);
  add('S:'+s.key,`${name}²`,s.value*s.value,'lengthSquared',s.marks,[`${c.display(s.b)}${c.display(s.a)}²`],[s.key,s.key]);
 }
 for(const q of c.CAll||c.C){const name=c.display(q.id),mark=c.circleMark(q);
  add('R2:'+q.id,`r(${name})²`,q.r*q.r,'radiusSquared',[mark],[`半径${name}的平方`],['R:'+q.id,'R:'+q.id]);
  add('C:'+q.key,`S(${name})`,Math.PI*q.r*q.r,'diskArea',[mark],[`${name}面积`,`圆面积${name}`]);
  const center=(q.centerId&&c.pMap.get(q.centerId))||c.P.find(p=>Math.hypot(p.x-q.x,p.y-q.y)<=c.eps);
  for(const p of c.P){const power=(p.x-q.x)**2+(p.y-q.y)**2-q.r*q.r;
   const m=add('W:'+p.id+'|'+q.key,`Pow(${c.display(p.id)},${name})`,power,'power',[c.pMark(p.id),mark],[`${c.display(p.id)}对${name}的幂`]);
   // CP² − r(c)² = Pow(P,c) is the DEFINITION of point power.
   // It must not appear as a new result merely because a free P was added.
   if(m&&center&&center.id!==p.id){const dk='D:'+[center.id,p.id].sort().join('|');
    m.algebraKey=`difference:${classes.get(dk)||dk}-${classes.get('R:'+q.id)||'R:'+q.id}`;
   }
  }
 }
 for(const t of c.triangles||[]){const name=t.ids.map(c.display).join('');
  add('A:'+t.ids.join('|'),`[${name}]`,t.area,'area',t.ids.map((id,i)=>c.sMark(id,t.ids[(i+1)%3])),[`△${name}面积`]);
 }
 // Binary homogeneous quadratic forms are necessary for Pythagorean,
 // parallelogram and many circle identities. They are not the same as a
 // product; enumerate separately, omitting same-length-class substitutions.
 // The configurable operand cap makes expensive quadratic exploration opt-in
 // without penalizing ordinary relation scans.
 const squared=metrics.filter(m=>m.type==='lengthSquared'||m.type==='radiusSquared');
 const exprCap=Number.isInteger(c.options?.quadraticExpressionLimit)?Math.max(0,c.options.quadraticExpressionLimit):20;
 const chosen=squared.slice(0,exprCap),unit=m=>classes.get(m.factors[0])||m.factors[0];
 for(let i=0;i<chosen.length;i++)for(let j=i+1;j<chosen.length;j++){
  const a=chosen[i],b=chosen[j],A=unit(a),B=unit(b);
  if(A===B)continue;
  const z=add(`PLUS:${a.key}+${b.key}`,`${a.name} + ${b.name}`,a.value+b.value,'sumSquares',[...a.marks,...b.marks],[],null);
  if(z)z.algebraKey=`sum:${[A,B].sort().join('+')}`;
  for(const [left,right] of [[a,b],[b,a]]){
   const w=add(`MINUS:${left.key}-${right.key}`,`${left.name} − ${right.name}`,left.value-right.value,'differenceSquares',[...left.marks,...right.marks],[],null);
   if(w)w.algebraKey=`difference:${unit(left)}-${unit(right)}`;
  }
 }
 // Products of two different nonzero segment lengths. Canonical ordering
 // removes AB*CD vs CD*AB. Limit product enumeration on very large drawings;
 // independent primitive lengths still remain unlimited in the scan.
 const cap=Number.isInteger(c.options?.quadraticProductLimit)?Math.max(0,c.options.quadraticProductLimit):66;
 const selected=segs.slice(0,cap);
 for(let i=0;i<selected.length;i++)for(let j=i+1;j<selected.length;j++){
  const a=selected[i],b=selected[j],nameA=c.display(a.a)+c.display(a.b),nameB=c.display(b.a)+c.display(b.b);
  add('M:'+a.key+'*'+b.key,`${nameA}×${nameB}`,a.value*b.value,'product',[...a.marks,...b.marks],[`${nameA}*${nameB}`,`${nameB}*${nameA}`,`${nameB}·${nameA}`],[a.key,b.key]);
 }
 // Dot products retain sign; positive lengths and areas never smuggle a
 // negative power into a positive equivalence class.
 for(let i=0;i<c.P.length;i++)for(let j=0;j<c.P.length;j++)for(let k=j+1;k<c.P.length;k++){
  const v=c.P[i],a=c.P[j],b=c.P[k];if(v===a||v===b)continue;
  const value=(a.x-v.x)*(b.x-v.x)+(a.y-v.y)*(b.y-v.y);
  add('DOT:'+v.id+'|'+[a.id,b.id].sort().join('|'),`⟨${c.display(v.id)}${c.display(a.id)},${c.display(v.id)}${c.display(b.id)}⟩`,value,'dot',[c.sMark(v.id,a.id),c.sMark(v.id,b.id)],[`vec(${c.display(v.id)}${c.display(a.id)})·vec(${c.display(v.id)}${c.display(b.id)})`]);
 }
 for(const m of metrics)if(m.factors)m.algebraKey=m.factors.map(f=>classes.get(f)||f).sort().join('*');
 return metrics.sort((a,b)=>a.value-b.value||a.key.localeCompare(b.key));
}
// For quadratic expressions sharing a free point, move the free point in two
// noncollinear directions. If the relation survives identically it is an
// algebraic rewriting, not an independent incidence relation. This check
// is conservative and used only for dot/area identities.
function elementaryIdentity(m,n,f,c){
 if(!['dot','area'].includes(m.type)&&!['dot','area'].includes(n.type))return false;
 const keys=q=>q.type==='dot'?q.key.slice(4).split('|'):q.type==='area'?q.key.slice(2).split('|'):[];
 const shared=keys(m).filter(id=>keys(n).includes(id));if(!shared.length)return false;
 function evaluate(q,id,pos){
  const p=k=>k===id?pos:c.pMap.get(k);
  if(q.type==='dot'){const [v,a,b]=q.key.slice(4).split('|'),V=p(v),A=p(a),B=p(b);return (A.x-V.x)*(B.x-V.x)+(A.y-V.y)*(B.y-V.y);}
  if(q.type==='area'){const [a,b,d]=q.key.slice(2).split('|'),A=p(a),B=p(b),D=p(d);return Math.abs((B.x-A.x)*(D.y-A.y)-(B.y-A.y)*(D.x-A.x))/2;}
  return q.value;
 }
 for(const id of shared){const p=c.pMap.get(id),unit=Math.max(.1,c.localScale*.1);if(!p)continue;
  if([[.22,.13],[-.19,.17]].every(([x,y])=>c.nearQuadratic(evaluate(m,id,{x:p.x+x*unit,y:p.y+y*unit}),f*evaluate(n,id,{x:p.x+x*unit,y:p.y+y*unit}))))return true;
 }
 return false;
}
function scan(c){
 const metrics=collect(c),groups=[];
 for(const m of metrics){let g=groups[groups.length-1];if(!g||!c.nearQuadratic(m.value,g[0].value))groups.push(g=[]);g.push(m);}
 const compact=g=>g.length<=6?g.map(m=>m.name).join(' = '):g.slice(0,6).map(m=>m.name).join(' = ')+` …（${g.length}项）`;
 const key=g=>g.map(x=>x.key).sort().join(';');
 const describe=g=>g.map(m=>({name:m.name,key:m.key,type:m.type,aliases:m.aliases,algebraKey:m.algebraKey}));
 const algebraOnly=g=>g.length>1&&g.every(m=>m.algebraKey&&m.algebraKey===g[0].algebraKey);
 const representatives=g=>new Set(g.map(m=>m.algebraKey||m.key)).size;
 function emit(high,low,f){
  const same=low===null,all=[...high,...(low||[])];
  if(same&&algebraOnly(high))return;
  if(!same&&algebraOnly(high)&&algebraOnly(low)&&high[0].algebraKey===low[0].algebraKey)return;
  // A purely algebraic class has just a single source of information even
  // if hundreds of multiply-relabelled products appear in its string.
  const identityMembers=all.filter(m=>m.algebraKey).length;
  const algebraRepetitions=identityMembers-representatives(all.filter(m=>m.algebraKey));
  const title=same?compact(high):`${compact(high)} = ${f.label} × (${compact(low)})`;
  const id=same?'EQ:'+key(high):`R:${key(high)}|${key(low)}|${f.label}`;
  c.add('quadraticRatio',id,title,all.flatMap(m=>m.marks),same?65:76,
   `统一二次量连${same?'等':'比'}；含 ${all.length} 项，代数替换 ${algebraRepetitions} 项`,all.flatMap(m=>m.aliases));
  const e=c.entries[c.entries.length-1];if(e?.id==='quadraticRatio:'+id){
   e.metric={high:describe(high),low:low?describe(low):[],factor:same?1:f.v,factorLabel:same?'1':f.label};
   e.quadraticTypes=[...new Set(all.map(m=>m.type))];
   e.algebraicSubstitutions=algebraRepetitions;
   e.algebraicOnly=all.every(m=>m.algebraKey!==undefined);
   e.relationEvidence=[...new Set(all.map(m=>m.algebraKey||m.type))];
  }
 }
 for(const g of groups)if(g.length>=2)emit(g,null,multipliers[0]);
 // Group-to-group ratios; do not generate a Cartesian product of equivalent
 // metric aliases. Use signed multipliers and preserve inverse search.
 for(let i=0;i<groups.length;i++)for(let j=i+1;j<groups.length;j++){
  const a=groups[i],b=groups[j],hi=Math.abs(a[0].value)>=Math.abs(b[0].value)?a:b,lo=hi===a?b:a;
  const ratio=hi[0].value/lo[0].value;if(!Number.isFinite(ratio)||Math.abs(ratio)<1+2*c.mode.ratio)continue;
  const sign=ratio<0?-1:1,abs=Math.abs(ratio);
  for(const f of multipliers.slice(1))if(c.nearQuadraticRatio(hi[0].value,lo[0].value*sign,f.v)){
   // Suppress only when EVERY pair of witnesses is a universal algebraic
   // identity; one independent mixed equality must survive.
   if(hi.every(m=>lo.every(n=>elementaryIdentity(m,n,sign*f.v,c))))break;
   // Multiplying a known one-dimensional length ratio by the very same
   // length does not create an independent quadratic ratio, even when the
   // common length uses an unrelated, newly added free point.
   if(hi.every(m=>m.type==='product')&&lo.every(n=>n.type==='product')&&
      hi.every(m=>lo.every(n=>m.factors.some(key=>n.factors.includes(key)))))break;
   emit(hi,lo,{v:sign*f.v,label:(sign<0?'−':'')+f.label});break;
  }
 }
}
return {collect,scan,lengthClasses,elementaryIdentity};
});
