/* v3.4 construction-aware relation attribution and declarative implication DAG.
 * Rules never fabricate a geometric fact: a direct witness is assigned only if
 * its defining operands can be checked against an already detected relation.
 * "Direct" describes why a relation was constructed, not a symbolic proof.
 */
(function(root,factory){const api=factory();root.GeomRelationProvenance=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const key=(a,b)=>[a,b].sort().join('|');
const logic=(typeof globalThis!=='undefined'&&globalThis.GeomConstructionLogic)||(typeof require==='function'?require('./construction-logic.js'):null);
const IMPLICATIONS=Object.freeze({
 square:['rectangle','rhombus','parallelogram','concyclic','perpendicular','fixedAngle'],
 rectangle:['parallelogram','perpendicular','fixedAngle'],
 rhombus:['parallelogram','distanceRatio'],
 equilateral:['distanceRatio','equalAngle','fixedAngle'],
 congruentTriangles:['similarTriangles','equalArea','areaRatio'],
 orthocentricSystem:['specialPoint'],
 midpoint:['collinear','distanceRatio','quadraticRatio'],
 perpendicular:['perpFoot','fixedAngle'],
 quadraticRatio:['areaRatio','equalArea'],
 incircle:['circleLineTangent'],
 specialLine:['angleBisector']
});
function markIds(e){const s=new Set();for(const m of e.marks||[]){if(m.type==='point')s.add(m.id);if(m.type==='segment'){s.add(m.a);s.add(m.b);}if(m.type==='angle'){s.add(m.a);s.add(m.b);s.add(m.v);}}return s;}
function segmentKeys(e){const out=new Set();for(const m of e.marks||[])if(m.type==='segment')out.add('D:'+key(m.a,m.b));return out;}
// Each witness below is operand-aware: a construction never labels an
// unrelated relation merely because its type agrees. Grouped relations may
// contain a direct subfact plus unrelated discoveries and are marked mixed.
function factMatches(f,e,c){
 const o=f.operand||{},marks=e.marks||[],idSet=markIds(e);
 const lineKey=ref=>!ref?null:(ref.type==='pair'?c.lineFamily.get('P:'+key(ref.a,ref.b))||'P:'+key(ref.a,ref.b):c.lineFamily.get('L:'+ref.id)||'L:'+ref.id);
 const lines=new Set(marks.filter(m=>m.type==='line').map(m=>lineKey(m.ref)));
 const fLines=(o.lines||[]).map(lineKey);
 switch(f.kind){
 case 'midpoint':return e.kind==='midpoint'&&idSet.has(o.point)&&o.vertices.every(p=>idSet.has(p));
 case 'collinear':return e.kind==='collinear'&&o.points.every(p=>idSet.has(p));
 case 'orthocentricSystem':return e.kind==='orthocentricSystem'&&o.points.every(p=>idSet.has(p));
 case 'triangleCenter':return e.kind==='specialPoint'&&idSet.has(o.point)&&o.vertices.every(p=>idSet.has(p))&&(e.detail||'').includes(o.center);
 case 'perpendicular':case 'parallel':return e.kind===f.kind&&fLines.length===2&&fLines.every(x=>lines.has(x));
 case 'angleBisector':return ['angleBisector','specialLine'].includes(e.kind)&&lines.has(lineKey(o.line))&&idSet.has(o.vertex)&&o.arms.every(p=>idSet.has(p))&&(e.detail||'').includes('角平分');
 case 'equalSegments':{
  if(e.kind==='equilateral'){const pts=new Set(marks.filter(m=>m.type==='segment').flatMap(m=>[m.a,m.b]));return o.segments.every(s=>s.slice(2).split('|').every(id=>pts.has(id)));}
  if(!['distanceRatio','quadraticRatio'].includes(e.kind)||e.metric?.factor!==1)return false;
  const wanted=e.kind==='quadraticRatio'?o.segments.map(s=>'S:'+s):o.segments;
  return wanted.every(s=>(e.metric.high||[]).some(m=>m.key===s));
 }
 case 'incidenceCircle':return e.kind==='pointOnCircle'&&idSet.has(o.point)&&marks.some(m=>m.type==='circle'&&m.ref?.id===o.circle);
 case 'incidenceLine':return e.kind==='pointOnLine'&&idSet.has(o.point)&&lines.has(lineKey(o.line));
 case 'incidenceCarrier':return ['pointOnLine','pointOnCircle'].includes(e.kind)&&idSet.has(o.point)&&marks.some(m=>(m.type==='line'||m.type==='circle')&&m.ref?.id===o.carrier);
 case 'reflection':return e.kind==='reflection'&&o.points.every(p=>idSet.has(p))&&lines.has(lineKey(o.line));
 case 'tangent':return e.kind==='circleLineTangent'&&lines.has(lineKey({type:'object',id:o.line}))&&marks.some(m=>m.type==='circle'&&m.ref?.id===o.circle);
 default:return false;
 }
}
function attachFacts(entries,facts,c,tag){
 let matches=0;
 for(const e of entries)for(const f of facts){if(!factMatches(f,e,c))continue;
  matches++;tag({e},{id:f.node,label:f.node,op:f.op},f.reason);
  (e.verifiedConsequences??=[]).push({node:f.node,kind:f.kind,operand:f.operand,level:f.level,rule:f.reason});
 }
 return matches;
}
function annotate(entries,snapshot,ctx){
 const nodes=new Map((snapshot.construction||snapshot.constructions||[]).map(n=>[n.id,n]));
 const constructionFacts=logic.makeFacts(snapshot.construction||snapshot.constructions||[]);
 const known=entries.map(e=>({e,pts:markIds(e),segments:segmentKeys(e)}));
 const lineSupport=id=>ctx.lineFamily.get('L:'+id)||'L:'+id;
 const pairSupport=(a,b)=>ctx.lineFamily.get('P:'+key(a,b))||'P:'+key(a,b);
 const getLineMark=e=>(e.marks||[]).filter(m=>m.type==='line').map(m=>m.ref?.type==='pair'?pairSupport(m.ref.a,m.ref.b):lineSupport(m.ref?.id));
 const tag=(item,n,reason)=>{
  const e=item.e;
  const arr=e.constructionWitnesses||(e.constructionWitnesses=[]);
  if(arr.some(w=>w.node===n.id&&w.reason===reason))return;
  arr.push({node:n.id,label:n.label||n.id,op:n.op,reason});
  e.searchTags=[...(e.searchTags||[]),'作图直接关系','定义条件',reason,n.label||n.id];
 };
 const containsDistanceEquality=(e,items)=>{
  if(e.kind!=='distanceRatio'&&e.kind!=='quadraticRatio')return false;
  const metric=e.metric;if(!metric||metric.factor!==1)return false;
  const set=new Set(metric.high.map(m=>m.key));
  return items.every(k=>set.has(k));
 };
 for(const n of nodes.values()){
  const inp=n.inputs||[],[a,b,d]=inp;
  for(const item of known){const e=item.e,ids=item.pts,lines=getLineMark(e);
   switch(n.op){
    case 'midpoint':{
     if(inp.length<2)break;
     const legs=['D:'+key(n.id,a),'D:'+key(n.id,b)],sq=legs.map(x=>'S:'+x);
     if(containsDistanceEquality(e,e.kind==='quadraticRatio'?sq:legs))tag(item,n,'中点两侧相等');
     else if(e.kind==='midpoint'&&[n.id,a,b].every(x=>ids.has(x)))tag(item,n,'中点定义');
     else if(e.kind==='collinear'&&[n.id,a,b].every(x=>ids.has(x)))tag(item,n,'中点共线');
     break;
    }
    case 'parallel':case 'parallel3':{
     if(e.kind==='parallel'&&inp.some(x=>lines.includes(lineSupport(x))))tag(item,n,'平行构造定义');break;
    }
    case 'perpendicular':case 'perpendicular3':case 'perpBisector':{
     if(e.kind==='perpendicular'&&(lines.includes(lineSupport(n.id))||inp.some(x=>lines.includes(lineSupport(x)))))tag(item,n,'垂直构造定义');break;
    }
    case 'foot':case 'foot3':{
     if(e.kind==='perpFoot'&&ids.has(n.id))tag(item,n,'垂足定义');break;
    }
    case 'circle':case 'circleBySegment':case 'circleRadius':case 'circumcircle':case 'diameterCircle':{
     if(e.kind==='pointOnCircle'&&e.marks.some(m=>m.type==='circle'&&m.ref?.id===n.id)&&inp.some(x=>ids.has(x)))tag(item,n,'圆的定义点');
     break;
    }
    case 'onLine':case 'onCircle':{
     if((e.kind==='pointOnLine'||e.kind==='pointOnCircle')&&ids.has(n.id))tag(item,n,'约束点定义');break;
    }
    case 'intersection':case 'secondIntersection':{
     if((e.kind==='pointOnLine'||e.kind==='pointOnCircle')&&ids.has(n.id))tag(item,n,'交点定义');break;
    }
    case 'centroid':case 'circumcenter':case 'incenter':case 'orthocenter':case 'excenter':{
     if(e.kind==='specialPoint'&&ids.has(n.id)&&inp.every(x=>ids.has(x)))tag(item,n,'三角形特殊心定义');break;
    }
    case 'angleBisector':case 'outerAngleBisector':{
     if((e.kind==='angleBisector'||e.kind==='specialLine')&&lines.includes(lineSupport(n.id)))tag(item,n,'角平分线定义');break;
    }
   }
  }
 }
 // Constructor-known incidence: no numerical inference is needed to know
 // that lines AB, AC, AD share their defining endpoint A. This also works
 // with polygon-generated segments, rays and through-point constructions.
 const support=id=>ctx.lineFamily.get(id)||id;
 const anchors=new Map();
 const pushAnchor=(family,vertex)=>{if(!vertex)return;const a=anchors.get(family)||new Set();a.add(vertex);anchors.set(family,a);};
 for(const n of nodes.values()){
  let seed=[];
  if(['line','segment','ray'].includes(n.op))seed=n.inputs.slice(0,2);
  else if(['parallel','perpendicular','parallel3','perpendicular3','tangent','tangentFrom'].includes(n.op))seed=n.inputs.slice(0,1);
  else if(['angleBisector','outerAngleBisector'].includes(n.op))seed=n.inputs.slice(1,2);
  if(!seed.length)continue;
  const fam=support('L:'+n.id);
  for(const id of seed)pushAnchor(fam,id);
 }
 // Virtual pairs are NOT construction anchors: treating arbitrary collinear
 // implicit pairs as anchors would erase genuinely discovered concurrency.
 for(const e of entries){if(e.kind!=='concurrent')continue;
  const pivot=(e.marks||[]).find(m=>m.type==='point')?.id;
  const refs=(e.marks||[]).filter(m=>m.type==='line').map(m=>m.ref);
  if(!pivot||refs.length<3)continue;
  const attached=refs.every(ref=>{
   const family=ref?.type==='pair'?support('P:'+key(ref.a,ref.b)):support('L:'+ref?.id);
   return anchors.get(family)?.has(pivot);
  });
  if(attached)tag({e},{id:pivot,label:pivot,op:'constructionAnchor'},'所有直线均以该点为构造锚点');
 }
 const operandMatches=attachFacts(entries,constructionFacts,ctx,tag);
 const impliedNames={rectangle:'矩形',rhombus:'菱形',parallelogram:'平行四边形',concyclic:'共圆',perpendicular:'垂直',fixedAngle:'特殊角',distanceRatio:'等长及长度比例',quadraticRatio:'平方长度比例',equalAngle:'等角',similarTriangles:'相似',equalArea:'等面积',areaRatio:'面积比例',specialPoint:'特殊点',collinear:'共线',perpFoot:'垂足',circleLineTangent:'圆线相切',angleBisector:'角平分线'};
 const condition=(e,k)=>{
  if(e.kind==='quadraticRatio')return e.quadraticTypes?.includes('area')&&['equalArea','areaRatio'].includes(k);
  if(e.kind==='specialLine'&&k==='angleBisector')return (e.detail||'').includes('角平分线');
  if(e.kind==='perpendicular'&&k==='perpFoot')return false; // A foot requires a named point witness.
  return true;
 };
 for(const e of entries){
  e.origin=e.constructionWitnesses?.length?'construction':'discovered';
  const metric=e.metric;const directSegments=new Set((e.verifiedConsequences||[]).filter(f=>f.kind==='equalSegments').flatMap(f=>f.operand.segments||[]));
  const members=metric?.factor===1?metric.high||[]:[];
  const allCovered=members.length&&members.every(m=>directSegments.has(m.key)||directSegments.has(m.key.replace(/^S:/,'')));
  e.source=e.origin==='construction'?(members.length&&!allCovered?'mixed':'construction'):'discovery';
  e.impliedKinds=(IMPLICATIONS[e.kind]||[]).filter(k=>condition(e,k));
  e.searchTags=[...(e.searchTags||[]),...e.impliedKinds.map(k=>impliedNames[k]||k)];
  if(e.kind==='midpoint'){
   const trio=[...markIds(e)];if(trio.length===3){
    const midpoint=e.marks.find(m=>m.type==='point')?.id;
    const other=trio.filter(id=>id!==midpoint);
    if(midpoint&&other.length===2)e.derivedFacts=[{kind:'equalLength',segments:[[midpoint,other[0]],[midpoint,other[1]]],from:e.id}];
   }
  }
 }
 return {nodes:nodes.size,directCount:entries.filter(e=>e.origin==='construction').length,definitionFacts:constructionFacts.length,operandMatches};
}
function consolidate(entries,c,snapshot={}){
 const removed=new Set();
 // A condition is never discarded because its TYPE is weaker. The stronger
 // witness must be in the same scan and cover the same actual geometric operands.
 const q=entries.filter(e=>e.kind==='quadraticRatio'&&e.metric);
 const seg=(e)=>new Set((e.metric?.high||[]).concat(e.metric?.low||[]).map(m=>m.key));
 const allTypes=e=>new Set(e.quadraticTypes||[]);
 const absorb=(keep,drop,reason)=>{if(keep===drop||removed.has(drop.id))return;
  removed.add(drop.id);keep.searchTags=[...(keep.searchTags||[]),drop.title,drop.kind,...drop.searchTags||[]];
  (keep.entails||(keep.entails=[])).push({id:drop.id,kind:drop.kind,reason});};
 // Witness-aware edges from the DAG. Only suppress a concrete weaker
 // candidate when the stronger candidate explains ALL of its operands.
 const stronger=entries.filter(e=>['square','equilateral','incircle'].includes(e.kind));
 const canonicalLine=ref=>!ref?null:ref.type==='pair'?
  (c.lineFamily.get('P:'+key(ref.a,ref.b))||'P:'+key(ref.a,ref.b)):
  (c.lineFamily.get('L:'+ref.id)||'L:'+ref.id);
 const circleRef=e=>(e.marks||[]).find(m=>m.type==='circle')?.ref;
 const sameCircle=(x,y)=>x&&y&&x.type===y.type&&x.id===y.id;
 for(const strong of stronger){
  const edges=(strong.marks||[]).filter(m=>m.type==='segment');
  const edgeKeys=new Set(edges.map(m=>'D:'+key(m.a,m.b)));
  const squareNames=new Set(edges.flatMap(m=>[m.a,m.b]));
  for(const weak of entries){if(!IMPLICATIONS[strong.kind]?.includes(weak.kind))continue;
   if(weak.kind==='distanceRatio'&&weak.metric?.factor===1&&(strong.kind==='square'||strong.kind==='equilateral')){
    const metric=weak.metric.high||[];
    const keys=metric.filter(m=>m.type==='segment').map(m=>m.key);
    // Preserve equal-length chains involving additional independently
    // constructed objects; they carry genuine information beyond a square.
    if(keys.length===metric.length&&keys.length===edgeKeys.size&&keys.every(k=>edgeKeys.has(k)))
     absorb(strong,weak,'规则图：特殊多边形的边长相等已包含于其定义');
   }
   if(weak.kind==='circleLineTangent'&&strong.kind==='incircle'&&sameCircle(circleRef(strong),circleRef(weak))){
    const mark=weak.marks.find(m=>m.type==='line');const supp=canonicalLine(mark?.ref);
    const boundary=edges.some(m=>canonicalLine({type:'pair',a:m.a,b:m.b})===supp);
    if(boundary)absorb(strong,weak,'规则图：三角形内切圆与其边的切触');
   }
  }
 }
 for(const weak of entries){
  if(weak.kind==='equalArea'||weak.kind==='areaRatio'){
   const triangles=new Set((weak.id.split(':').slice(1).join(':').match(/[^;|:]+/g)||[]).filter(x=>x.includes(',')) .map(x=>'A:'+x.replaceAll(',','|')));
   const matched=q.find(e=>{
    const ids=seg(e);
    return triangles.size>0&&[...triangles].every(k=>ids.has(k))&&allTypes(e).has('area');
   });
   if(matched)absorb(matched,weak,'二次量连比已表示同一面积条件');
  }
  // Squared-only groups are direct algebraic restatements of corresponding
  // equal-length facts; mixed squared-distance/area groups remain interesting.
  if(weak.kind==='quadraticRatio'&&allTypes(weak).size===1&&allTypes(weak).has('lengthSquared')){
   const wanted=[...(weak.metric?.high||[]),...(weak.metric?.low||[])];
   const matched=entries.find(e=>e.kind==='distanceRatio'&&e.metric&&(()=>{
    const values=[...e.metric.high,...e.metric.low];
    const exact=new Set(values.map(x=>'S:'+x.key));
    return wanted.length===values.length&&wanted.every(m=>exact.has(m.key))&&
      Math.abs((e.metric.factor||1)**2-(weak.metric.factor||1))<=c.mode.ratio*3;
   })());
   if(matched)absorb(matched,weak,'平方距离比例由长度比例直接推出');
  }
 }
 // v3.4: construction-aware transitive compression.  A generated center
 // absorbs only *complete* primitive facts already implied by that same
 // construction. Multi-member line/length groups are never removed merely
 // because one subpair is explained.
 const sourceFacts=logic.makeFacts(snapshot.construction||snapshot.constructions||[]);
 const sourceByNode=new Map();for(const fact of sourceFacts){const a=sourceByNode.get(fact.node)||[];a.push(fact);sourceByNode.set(fact.node,a);}
 const lineKey=ref=>!ref?null:ref.type==='pair'?(c.lineFamily.get('P:'+key(ref.a,ref.b))||'P:'+key(ref.a,ref.b)):(c.lineFamily.get('L:'+ref.id)||'L:'+ref.id);
 for(const strong of entries){
  if(removed.has(strong.id)||!['orthocentricSystem','midpoint'].includes(strong.kind))continue;
  const facts=(strong.verifiedConsequences||[]).flatMap(match=>sourceByNode.get(match.node)||[]);
  if(!facts.length)continue;
  for(const weak of entries){if(removed.has(weak.id)||weak.id===strong.id)continue;
   if(strong.kind==='orthocentricSystem'&&weak.kind==='perpendicular'){
    const supports=(weak.marks||[]).filter(m=>m.type==='line').map(m=>lineKey(m.ref));
    if(supports.length!==2)continue;
    const covered=facts.some(f=>f.kind==='perpendicular'&&f.operand.lines.map(lineKey).every(x=>supports.includes(x)));
    if(covered)absorb(strong,weak,'垂心构造蕴含该对高线垂直');
   }
   if(strong.kind==='midpoint'&&weak.kind==='collinear'){
    const pts=new Set((weak.marks||[]).filter(m=>m.type==='point').map(m=>m.id));
    if(pts.size===3&&facts.some(f=>f.kind==='collinear'&&f.operand.points.every(id=>pts.has(id))))(strong.entails??=[]).push({id:weak.id,kind:weak.kind,reason:'中点定义蕴含三点共线'});
   }
   if(strong.kind==='midpoint'&&weak.kind==='distanceRatio'&&weak.metric?.factor===1){
    const items=weak.metric.high||[];
    if(items.length===2&&facts.some(f=>f.kind==='equalSegments'&&f.operand.segments.every(s=>items.some(v=>v.key===s))))(strong.entails??=[]).push({id:weak.id,kind:weak.kind,reason:'中点定义蕴含等长两段'});
   }
  }
 }
 return {entries:entries.filter(e=>!removed.has(e.id)),suppressed:removed.size};
}
// Scores are computed from the full construction DAG and relation incidence,
// not from arbitrary object names or mere creation index. Long dependency
// chains earn a bonus only when their dependent node is involved in few
// independent observations; trivial construction identities are discounted.
function score(entries,snapshot={}){
 const raw=snapshot.construction||snapshot.constructions||[],nodes=new Map(raw.map(n=>[n.id,n]));
 const state=new Map(),depth=new Map();
 function level(id){if(depth.has(id))return depth.get(id);if(state.get(id)==='working')return 0;
  const n=nodes.get(id);if(!n)return 0;state.set(id,'working');
  const answer=['free','real','referencePoint'].includes(n.op)?0:1+Math.max(0,...(n.inputs||[]).filter(x=>nodes.has(x)).map(level));
  state.set(id,'done');depth.set(id,answer);return answer;
 }
 for(const id of nodes.keys())level(id);
 const members=e=>{
  const ids=markIds(e);
  for(const m of e.marks||[]){if(m.ref?.type==='object')ids.add(m.ref.id);if(m.line?.type==='object')ids.add(m.line.id);if(m.circleRef?.type==='object')ids.add(m.circleRef.id);}
  for(const x of [...(e.metric?.high||[]),...(e.metric?.low||[])]){
   if(x.a)ids.add(x.a);if(x.b)ids.add(x.b);if(x.point)ids.add(x.point);
   const tail=String(x.key||'').split(':').slice(1).join(':');
   if(/^(?:D:)?[^:|]+\|[^:|]+$/.test(tail))for(const part of tail.replace(/^D:/,'').split('|'))ids.add(part);
  }
  return [...ids].filter(x=>nodes.has(x));
 };
 const involved=new Map(entries.map(e=>[e.id,members(e)]));
 const frequency=new Map(),pairFrequency=new Map();
 for(const ids of involved.values()){
  for(const id of new Set(ids))frequency.set(id,(frequency.get(id)||0)+1);
  for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
   const k=key(ids[i],ids[j]);pairFrequency.set(k,(pairFrequency.get(k)||0)+1);
  }
 }
 const kindRarity={lineIntersectionCircle:9,circleIntersectionLine:9,multiCircleCommon:10,radicalAxis:10,orthocentricSystem:9,concurrent:9,similarTriangles:8,congruentTriangles:8,quadraticRatio:7,circleCircleTangent:8,circleLineTangent:8};
 const primitives=new Set(['coincident','collinear','pointOnLine','pointOnCircle','midpoint','fixedAngle']);
 for(const e of entries){const ids=involved.get(e.id)||[],derived=ids.filter(id=>(depth.get(id)||0)>0),uniq=new Set(ids);
  const maxDepth=Math.max(0,...derived.map(id=>depth.get(id)));
  const sparse=derived.length?Math.max(...derived.map(id=>1/Math.sqrt(1+(frequency.get(id)||0)))):0;
  const rarePairs=[];for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++)if(derived.includes(ids[i])||derived.includes(ids[j]))rarePairs.push(1/Math.sqrt(1+(pairFrequency.get(key(ids[i],ids[j]))||0)));
  const pairNovelty=rarePairs.length?Math.max(...rarePairs):0;
  const deepest=maxDepth>0?Math.min(1,Math.log2(maxDepth+1)/3.6):0;
  const graphValue=deepest*(0.68*sparse+0.32*pairNovelty);
  const uncommon=kindRarity[e.kind]||0;
  const direct=e.origin==='construction',primitive=primitives.has(e.kind);
  // Constructed but sparse, deep endpoints may be interesting even if the
  // relation kind is elementary. This is distinct from a direct definition.
  const independent=Math.max(1,Math.min(67,22+3*Math.min(6,uniq.size)+23*graphValue+uncommon-(direct?13:0)-(primitive?7:0)));
  const nontrivial=Math.max(1,Math.min(33,Math.round((e.score-45)*.41)+15+uncommon*.6-(direct?7:0)-(primitive?6:0)));
  e.scoreComponents={independentInformation:+independent.toFixed(2),nontriviality:+nontrivial.toFixed(2),dependencyDepth:maxDepth,relationSparsity:+sparse.toFixed(4),graphNovelty:+graphValue.toFixed(4),observedObjects:uniq.size};
  e.score=Math.max(1,Math.min(100,Math.round(independent+nontrivial)));
  // v4 separates a verified numerical statement from its novelty. A
  // theorem already imposed by the construction cannot top the discovery list.
  if(e.source==='construction')e.score=Math.min(e.score,e.kind==='concurrent'?12:22);
  else if(e.source==='mixed')e.score=Math.min(e.score,42);
  else if(e.source==='entailed')e.score=Math.min(e.score,25);
  e.scoreComponents.constructionPenalty=e.source==='construction'?'definition-cap':e.source==='mixed'?'mixed-cap':e.source==='entailed'?'implication-cap':'none';
 }
}
// A read-only explanation graph: definitions C→F, verified operands F→R,
// and witnessed semantic implication R→R.  Graph nodes are not extra rows.
function buildImplicationGraph(entries,snapshot={}){
 const commands=snapshot.construction||snapshot.constructions||[],g=logic.graph(commands);
 const nodes=g.nodes.slice(),edges=g.edges.slice(),known=new Set(nodes.map(n=>n.id));
 const relationNode=(id,kind,hidden=false)=>{const name='R:'+id;if(!known.has(name)){known.add(name);nodes.push({id:name,kind,hidden});}return name;};
 const byFact=new Map();for(let i=0;i<g.facts.length;i++){
  const f=g.facts[i],k=JSON.stringify([f.node,f.kind,f.operand]);const a=byFact.get(k)||[];a.push('F:'+i);byFact.set(k,a);
 }
 for(const e of entries){const name=relationNode(e.id,e.kind);
  for(const w of e.verifiedConsequences||[]){const k=JSON.stringify([w.node,w.kind,w.operand]);
   for(const id of byFact.get(k)||[])edges.push({from:id,to:name,rule:'VERIFIED_OPERANDS',category:'witness'});
  }
  for(const target of e.entails||[])edges.push({from:name,to:relationNode(target.id,target.kind,true),rule:target.reason,category:'entails'});
 }
 return {nodes,edges};
}
return {annotate,consolidate,score,buildImplicationGraph,IMPLICATIONS};
});
