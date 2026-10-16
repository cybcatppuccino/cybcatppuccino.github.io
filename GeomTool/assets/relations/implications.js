/* v3.3: explicit, typed implication graph.  A rule is applied only when both
 * source and target are present and their geometric witness IDs coincide.
 * Logical implication is directional; search synonyms survive reductions.
 */
(function(root,factory){const api=factory();root.GeomRelationImplications=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const typeEdges=Object.freeze([
 ['square','rectangle'],['square','rhombus'],['square','parallelogram'],
 ['rectangle','parallelogram'],['rhombus','parallelogram'],
 ['equilateral','isosceles'],
 ['congruentTriangles','similarTriangles']
]);
const normalizedVertices=e=>{
 if(['square','rectangle','rhombus','parallelogram'].includes(e.kind))
  return [...new Set(e.marks.filter(m=>m.type==='segment').flatMap(m=>[m.a,m.b]))].sort().join('|');
 return '';
};
function reduce(entries){
 const remove=new Set(),edges=[];
 const families=new Map();for(const e of entries){const key=normalizedVertices(e);if(key){const f=families.get(key)||new Map();f.set(e.kind,e);families.set(key,f);}}
 for(const [key,f] of families)for(const [strong,weak] of typeEdges){
  const a=f.get(strong),b=f.get(weak);if(!a||!b||a.id===b.id)continue;
  remove.add(b.id);edges.push({from:a.id,to:b.id,reason:strong+' implies '+weak});
  (a.searchTags??=[]).push(b.title,b.kind,...(b.searchTags||[]));
 }
 return {entries:entries.filter(e=>!remove.has(e.id)),removed:remove.size,edges};
}
// Fact nodes are more general than rendered results.  They may be derived
// definitions or consequences not selected in the UI. This is important: a
// non-selected parent must never delete a selected child result.
const consequences={
 square:['rectangle','rhombus','parallelogram','oppositeSidesParallel','allSidesEqual','allAnglesRight','concyclic'],
 rectangle:['parallelogram','oppositeSidesParallel','allAnglesRight','concyclic'],
 rhombus:['parallelogram','oppositeSidesParallel','allSidesEqual','diagonalsPerpendicular'],
 parallelogram:['oppositeSidesParallel','oppositeSidesEqual','diagonalsBisect'],
 equilateral:['isosceles','threeSidesEqual','allAngles60'],
 midpoint:['collinear','equalHalfSegments'],
 perpendicular:['rightAngle'],
 parallel:['sameSideSignedDistance'],
 angleBisector:['equalAdjacentAngles'],
 congruentTriangles:['similarTriangles','equalArea','correspondingSidesEqual','correspondingAnglesEqual'],
 similarTriangles:['proportionalSides','correspondingAnglesEqual'],
 reflection:['equalDistancesFromAxis','congruentReflectedTriangles'],
 pointOnCircle:['equalRadius'],
 radicalAxis:['equalPowers'],
 circleLineTangent:['radiusPerpendicularAtContact'],
 orthocentricSystem:['fourEquivalentOrthocenters']
};
function buildGraph(entries,constructions=[]){
 const nodes=new Map(),edges=[],relationIds=new Set(entries.map(e=>e.id));
 const node=(id,kind,origin)=>{if(!nodes.has(id))nodes.set(id,{id,kind,origin});};
 for(const e of entries){
  node(e.id,e.kind,'detected');const implied=consequences[e.kind]||[];
  const geometry=e.id.slice(e.kind.length+1);
  e.entailment={implies:[]};
  for(const child of implied){const key='F:'+child+':'+geometry;
   node(key,child,'derived');edges.push({from:e.id,to:key,rule:e.kind+' → '+child,category:'implication'});
   e.entailment.implies.push(key);
  }
 }
 for(const c of constructions){
  const id='C:'+c.id,op=c.op;node(id,op,'construction');
  if(op==='midpoint'&&c.inputs?.length===2){
   const [a,b]=c.inputs,mid=c.id,base=[a,b].sort().join('|');
   for(const [type,payload] of [['collinear',[a,b,mid].sort().join('|')],['equalHalfSegments',[[a,mid].sort().join('|'),[b,mid].sort().join('|')].sort().join('=')]]){
    const to='F:'+type+':'+payload;node(to,type,'derived');edges.push({from:id,to,rule:'MIDPOINT_DEF',category:'definition'});
   }
  }
  if(['parallel','parallel3','perpendicular','perpendicular3','foot','foot3','angleBisector','circumcircle','onCircle','onLine'].includes(op)){
   const key='F:construction-'+op+':'+c.id;node(key,op,'derived');edges.push({from:id,to:key,rule:op.toUpperCase()+'_DEF',category:'definition'});
  }
 }
 return {nodes:[...nodes.values()],edges};
}
return {reduce,typeEdges,buildGraph,consequences};
});
