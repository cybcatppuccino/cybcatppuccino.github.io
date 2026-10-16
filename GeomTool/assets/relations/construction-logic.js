/* v3.4 — operand-aware consequences of Kernel construction commands.
 * These are causal facts, never fabricated rendered relations. Facts provide
 * witnesses only when an actually detected numeric predicate agrees.
 * IDs are graph IDs, not display labels; no coordinates are mutated.
 */
(function(root,factory){const api=factory();root.GeomConstructionLogic=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const pair=(a,b)=>[a,b].sort().join('|');
const seg=(a,b)=>'D:'+pair(a,b);
const pLine=(a,b)=>({type:'pair',a,b});
const objLine=id=>({type:'object',id});
function makeFacts(construction=[]){
 const facts=[];const push=(n,kind,operand,reason,level='definition')=>facts.push({node:n.id,op:n.op,kind,operand,reason,level});
 for(const n of construction){const [a,b,c]=n.inputs||[],id=n.id;
  switch(n.op){
   case 'real':push(n,'onRealAxis',{point:id},'实轴约束');break;
   case 'referencePoint':push(n,'fixedCoordinate',{point:id,x:Number(n.params?.x),y:Number(n.params?.y)},'参考点固定坐标');break;
   case 'polygon':if((n.inputs||[]).length>=3)push(n,'polygonVertices',{polygon:id,vertices:n.inputs.slice()},'多边形定义顶点');break;
   case 'midpoint':if(a&&b){push(n,'midpoint',{point:id,vertices:[a,b]},'中点定义');push(n,'collinear',{points:[a,id,b]},'中点共线');push(n,'equalSegments',{segments:[seg(a,id),seg(id,b)]},'中点两侧相等');}break;
   case 'onLine':if(a)push(n,'incidenceLine',{point:id,line:objLine(a)},'线上点约束');break;
   case 'onCircle':if(a)push(n,'incidenceCircle',{point:id,circle:a},'圆上点约束');break;
   case 'circleCenter':if(a)push(n,'circleCenter',{point:id,circle:a},'圆心定义');break;
   case 'intersection':if(a&&b)for(const carrier of [a,b])push(n,'incidenceCarrier',{point:id,carrier},'交点载体约束');break;
   case 'secondIntersection':if(b&&c)for(const carrier of [b,c])push(n,'incidenceCarrier',{point:id,carrier},'第二交点约束');break;
   case 'line':case 'segment':case 'ray':if(a&&b){for(const vertex of [a,b])push(n,'incidenceLine',{point:vertex,line:objLine(id)},'两点定线的端点在载体上');}break;
   case 'parallel':if(b)push(n,'parallel',{lines:[objLine(id),objLine(b)]},'平行线定义');break;
   case 'parallel3':if(b&&c)push(n,'parallel',{lines:[objLine(id),pLine(b,c)]},'三点平行线定义');break;
   case 'perpendicular':if(b)push(n,'perpendicular',{lines:[objLine(id),objLine(b)]},'垂线定义');break;
   case 'perpendicular3':if(b&&c)push(n,'perpendicular',{lines:[objLine(id),pLine(b,c)]},'三点垂线定义');break;
   case 'perpBisector':if(a&&b){push(n,'perpendicular',{lines:[objLine(id),pLine(a,b)]},'中垂线垂直底边');push(n,'midpointOnLine',{line:id,vertices:[a,b]},'中垂线经过中点');}break;
   case 'foot':if(a&&b){push(n,'incidenceLine',{point:id,line:objLine(b)},'垂足在线上');push(n,'perpendicular',{lines:[pLine(a,id),objLine(b)]},'垂足与原线垂直');}break;
   case 'foot3':if(a&&b&&c){push(n,'collinear',{points:[b,id,c]},'三点垂足共线');push(n,'perpendicular',{lines:[pLine(a,id),pLine(b,c)]},'三点垂足垂直');}break;
   case 'angleBisector':case 'outerAngleBisector':if(a&&b&&c){push(n,'angleBisector',{vertex:b,arms:[a,c],line:objLine(id),outer:n.op==='outerAngleBisector'},'角平分线定义');}break;
   case 'circle':if(a&&b){push(n,'circleCenter',{point:a,circle:id},'给定圆心');push(n,'incidenceCircle',{point:b,circle:id},'圆过定义点');}break;
   case 'circleRadius':if(a){push(n,'circleCenter',{point:a,circle:id},'给定圆心');push(n,'fixedRadius',{circle:id,radius:Number(n.params?.radius)},'给定半径');}break;
   case 'circleBySegment':if(a&&b&&c)push(n,'equalCircleRadius',{circle:id,from:seg(b,c)},'线段转移为圆半径');break;
   case 'circumcircle':if(a&&b&&c)for(const p of [a,b,c])push(n,'incidenceCircle',{point:p,circle:id},'三点定圆');break;
   case 'onCircumcircle':if(a&&b&&c)push(n,'concyclic',{points:[a,b,c,id]},'三点圆约束');break;
   case 'diameterCircle':if(a&&b){for(const p of [a,b])push(n,'incidenceCircle',{point:p,circle:id},'直径圆经过端点');push(n,'diameter',{circle:id,vertices:[a,b]},'直径定义');}break;
   case 'centroid':if(a&&b&&c)push(n,'triangleCenter',{center:'重心',point:id,vertices:[a,b,c]},'三角形重心定义');break;
   case 'circumcenter':if(a&&b&&c){push(n,'triangleCenter',{center:'外心',point:id,vertices:[a,b,c]},'外心定义');push(n,'equalSegments',{segments:[seg(id,a),seg(id,b),seg(id,c)]},'外心到三顶点等距');}break;
   case 'incenter':if(a&&b&&c)push(n,'triangleCenter',{center:'内心',point:id,vertices:[a,b,c]},'内心定义');break;
   case 'excenter':if(a&&b&&c)push(n,'triangleCenter',{center:'旁心',point:id,vertices:[a,b,c]},'旁心定义');break;
   case 'orthocenter':if(a&&b&&c){
    push(n,'triangleCenter',{center:'垂心',point:id,vertices:[a,b,c]},'垂心定义');
    for(const [v,u,w] of [[a,b,c],[b,a,c],[c,a,b]])push(n,'perpendicular',{lines:[pLine(id,v),pLine(u,w)]},'垂心三条高线');
    push(n,'orthocentricSystem',{points:[a,b,c,id]},'四点垂心等价组');
    for(const v of [a,b,c])push(n,'equivalentOrthocenter',{point:v,vertices:[a,b,c,id].filter(x=>x!==v)},'垂心组的另外三种等价表达','consequence');
    // All six unordered pairs of the four-point orthocentric system define
    // three perpendicular line-pair identities; not arbitrary point triples.
    for(const [u,v,w,x] of [[a,b,c,id],[a,c,b,id],[a,id,b,c]])push(n,'perpendicular',{lines:[pLine(u,v),pLine(w,x)]},'四点垂心组的对边垂直','consequence');
   }break;
   case 'equilateral':if(a&&b)push(n,'equalSegments',{segments:[seg(a,b),seg(a,id),seg(b,id)]},'等边三角形三边相等');break;
   case 'translate':if(a&&b&&c)push(n,'equalDirectedVectors',{vectors:[[a,id],[b,c]]},'平移向量相等');break;
   case 'parallelogram':if(a&&b&&c){push(n,'parallel',{lines:[pLine(a,b),pLine(c,id)]},'平行四边形对边平行');push(n,'parallel',{lines:[pLine(a,c),pLine(b,id)]},'平行四边形对边平行');}break;
   case 'reflectPoint':if(a&&b)push(n,'midpoint',{point:b,vertices:[a,id]},'中心对称中心是中点');break;
   case 'reflectLine':if(a&&b){push(n,'reflection',{points:[a,id],line:objLine(b)},'轴对称定义');push(n,'perpendicular',{lines:[pLine(a,id),objLine(b)]},'轴对称连线垂直于轴');}break;
   case 'rotateLeft':case 'rotateRight':case 'rotate':if(a&&b)push(n,'equalSegments',{segments:[seg(a,b),seg(id,b)]},'旋转保长度');break;
   case 'homothety':if(a&&b)push(n,'collinear',{points:[a,b,id]},'位似共线');break;
   case 'tangent':if(a&&b)push(n,'tangent',{point:a,circle:b,line:id},'切线定义');break;
   case 'tangentFrom':if(a&&b)push(n,'tangent',{point:a,circle:b,line:id},'外点切线定义');break;
   // The following commands also have implicit mathematical definitions, but
   // are not expressed as falsely specific numeric equalities here.
   default:break;
  }
 }
 return facts;
}
const graph=(constructions,facts=makeFacts(constructions))=>({
 nodes:[...(constructions||[]).map(n=>({id:'C:'+n.id,op:n.op,kind:'construction'})),...facts.map((f,i)=>({id:'F:'+i,kind:f.kind,operand:f.operand}))],
 edges:facts.map((f,i)=>({from:'C:'+f.node,to:'F:'+i,rule:f.reason,category:f.level})),facts
});
return {makeFacts,graph,pair,seg};
});
