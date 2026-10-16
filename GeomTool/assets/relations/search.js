/* v3.1 semantic querying: metadata aware, symmetric ratios, implied predicates. */
(function(root,factory){const api=factory();root.GeomRelationSearch=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const implied={square:'正方形 矩形 菱形 平行四边形 等长 共圆 全等 对角线 四边形',rectangle:'矩形 平行四边形 对角线相等 共圆 全等 四边形',rhombus:'菱形 平行四边形 四边等长 对角线垂直 全等 四边形',parallelogram:'平行四边形 对边平行 对边相等 对角线互相平分 全等 四边形',kite:'筝形 邻边相等 对称轴 对角线垂直 对角线平分 全等 四边形',equilateral:'等边三角形 等腰三角形 等角 等边',congruentTriangles:'全等 相似 三角形',similarTriangles:'相似 三角形',specialLine:'特殊线 中线 高线 角平分 中垂线',angleBisector:'等角 相等角 角平分线',perpendicular:'垂直 直角 90° 特殊角',distanceRatio:'比例 比值 连等 连比 等长 两点等距 等距',fixedAngle:'特殊角 角度',radicalAxis:'根轴 等幂 点的幂',concurrent:'三线共点 多线共点',multiCircleCommon:'三圆共点 多圆共点'};
const alias={等向量:'平行四边形',对角线垂直:'垂直',等弦:'等距',平行线:'平行',相同角:'等角',同等角:'等角',垂线:'垂直',余角:'互余',补角:'互补',菱形:'菱形',等长:'距离',半径相同:'等半径',};
const normalize=s=>String(s||'').toLowerCase().normalize('NFKC').replace(/[−–—]/g,'-').replace(/[√]/g,'sqrt').replace(/[φΦ]/g,'phi').replace(/\s+/g,'').replace(/[△∆∠]/g,'');
const ratioValue=str=>{
 let s=normalize(str).replace(/π/g,'pi');
 const known={sqrt2:Math.SQRT2,sqrt3:Math.sqrt(3),phi:(1+Math.sqrt(5))/2,pi:Math.PI};if(s.startsWith('-')&&s.slice(1) in known)return -known[s.slice(1)];
 if(s in known)return known[s];
 const frac=s.match(/^([+-]?\d+(?:\.\d+)?)\/([+-]?\d+(?:\.\d+)?)$/);
 if(frac)return Number(frac[1])/Number(frac[2]);
 const num=Number(s);return Number.isFinite(num)?num:NaN;
};
const similarMetric=(term,m)=>m.some(w=>(w.aliases||[w.name]).some(n=>normalize(n)===normalize(term)));
function matchRatio(entry,query){
 const m=query.trim().match(/^\s*(.+?)\s*[:：／/]\s*(.+?)\s*[=＝]\s*([\w.\/()+\-√φΦ−]+)\s*$/);
 if(!m||!entry.metric)return null;
 const value=ratioValue(m[3]);if(!Number.isFinite(value)||Math.abs(value)<1e-12)return false;
 const g=entry.metric,hi=g.high||[],lo=g.low||[];
 const eq=(a,b)=>similarMetric(a,hi)&&similarMetric(b,hi);
 if(g.factor===1)return Math.abs(value-1)<1e-10&&(eq(m[1],m[2])||eq(m[2],m[1]));
 const direct=similarMetric(m[1],hi)&&similarMetric(m[2],lo)&&Math.abs(value-g.factor)<=1e-9*Math.max(1,g.factor);
 const reversed=similarMetric(m[1],lo)&&similarMetric(m[2],hi)&&Math.abs(value-1/g.factor)<=1e-9*Math.max(1,Math.abs(1/g.factor));
 return direct||reversed;
}
function matches(entry,query,label=''){
 const q=String(query||'').trim();if(!q)return true;
 const ratio=matchRatio(entry,q);if(ratio!==null)return ratio;
 const hay=normalize([entry.title,entry.detail,entry.searchText,entry.kind,label,implied[entry.kind]||'',...(entry.searchTags||[])].join(' '));
 const bits=q.split(/[\s,，、;；]+/).filter(Boolean);
 return bits.every(t=>{
  const anglePair=t.match(/^(.+?)[=＝](.+)$/);
  if(anglePair&&entry.angleEqual){
   const n1=normalize(anglePair[1]),n2=normalize(anglePair[2]);
   if((entry.angleEqual||[]).some(n=>normalize(n)===n1)&&(entry.angleEqual||[]).some(n=>normalize(n)===n2))return true;
  }const norm=normalize(t),candidate=normalize(alias[t]||t);
  return hay.includes(norm)||hay.includes(candidate)||((entry.metric?.high||[]).concat(entry.metric?.low||[]).some(m=>[m.name,...m.aliases||[]].some(n=>normalize(n).includes(norm))));
 });
}
return {matches,matchRatio,implied,normalize};
});
