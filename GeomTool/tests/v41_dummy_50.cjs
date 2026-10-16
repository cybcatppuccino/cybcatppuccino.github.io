'use strict';
/* Fifty independently specified tiny diagrams, with a categorical oracle for
 * every scene.  Do not change these expectations to mask regression errors. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js');
const {scan,registry}=require('../assets/relations/detectors.js');
const {matches}=require('../assets/relations/search.js');
const report=[];
const P=(...rows)=>({points:rows.map(([id,x,y])=>({id,x,y})),lines:[],circles:[],construction:[]});
const C=(s,id,cx,cy,r,centerId=null)=>{s.circles.push({id,cx,cy,r,centerId});return s;};
const add=(k,op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
const free=(k,id,x,y)=>add(k,'free',id,[],{x,y});
const make=fn=>{const k=new Kernel();fn(k);const s=k.relationSnapshot();s.construction=k.snapshot().nodes.filter(n=>n.valid).map(n=>({id:n.id,op:n.op,inputs:n.inputs,params:n.params}));return s;};
const select=(r,kind)=>r.entries.filter(e=>e.kind===kind);
const metric=(e,key)=>(e.metric?.high||[]).concat(e.metric?.low||[]).some(m=>m.key===key);
function audit(name,category,s,kinds,oracle){
 assert(report.length<50,'exactly fifty expected');
 const r=scan(s,{kinds,perturb:false});
 try{oracle(r);}catch(error){throw Error(`[${String(report.length+1).padStart(2,'0')}] ${name}: ${error.message}`);}
 for(const e of r.entries){assert(Number.isFinite(e.score)&&e.score>=1&&e.score<=100,`bad score: ${name}`);assert(e.source!=='construction'||e.score<=22,`construction inflated ${name}`);assert(e.source!=='entailed'||e.score<=25,`logical implication inflated ${name}`);}
 report.push({number:report.length+1,name,category,objects:r.objects,relations:r.count,kindsChecked:kinds,passed:true,highest:r.entries[0]?{kind:r.entries[0].kind,score:r.entries[0].score}:null});
}
// 01—10: radius is a FIRST-degree operand and must be causal, not just close.
const radiograph=(mode='on')=>make(k=>{
 free(k,'O',0,0);free(k,'A',4,0);add(k,'circle','c',['O','A']);
 if(mode==='on')add(k,'onCircle','B',['c'],{theta:Math.PI/2});
 else if(mode==='free')free(k,'B',0,4);
 else if(mode==='off')free(k,'B',0,4.3);
});
audit('圆定义点与圆上构造点都等于半径','radius',radiograph(),['distanceRatio','pointOnCircle'],r=>{
 const eq=select(r,'distanceRatio').find(e=>e.metric?.factor===1&&['R:c','D:A|O','D:B|O'].every(k=>metric(e,k)));assert(eq&&eq.source==='construction'&&eq.score<=22);
});
audit('仅给定圆心过点：半径=定义线段','radius',radiograph('none'),['distanceRatio'],r=>assert(select(r,'distanceRatio').some(e=>e.metric?.factor===1&&metric(e,'R:c')&&metric(e,'D:A|O')&&e.source==='construction')));
audit('同圆上的自由点不能把整个半径连等冒称构造','radius',radiograph('free'),['distanceRatio'],r=>{
 const eq=select(r,'distanceRatio').find(e=>e.metric?.factor===1&&metric(e,'R:c')&&metric(e,'D:B|O'));assert(eq&&eq.source==='mixed'&&eq.score<=42);
});
audit('定数值半径恰好等于自由线段并非构造等长','radius',make(k=>{free(k,'O',0,0);free(k,'A',3,0);add(k,'circleRadius','c',['O'],{radius:3});}),['distanceRatio'],r=>{
 const eq=select(r,'distanceRatio').find(e=>e.metric?.factor===1&&metric(e,'R:c')&&metric(e,'D:A|O'));assert(eq&&eq.source!=='construction');
});
audit('线段转移为圆半径的直接事实','radius',make(k=>{free(k,'O',-1,0);free(k,'A',1,2);free(k,'B',4,2);add(k,'circleBySegment','c',['O','A','B']);}),['distanceRatio'],r=>assert(select(r,'distanceRatio').some(e=>e.metric?.factor===1&&metric(e,'R:c')&&metric(e,'D:A|B')&&e.source==='construction')));
audit('两个重合圆仍各自拥有半径量','radius',C(C(P(['A',0,0],['B',3,0]),'c1',0,0,3),'c2',0,0,3),['distanceRatio'],r=>assert(select(r,'distanceRatio').some(e=>e.metric?.factor===1&&metric(e,'R:c1')&&metric(e,'R:c2'))));
audit('不同半径不得被错误等同','radius',C(C(P(['A',0,0],['B',2,0]),'c1',0,0,2),'c2',10,3,5),['distanceRatio'],r=>assert(!select(r,'distanceRatio').some(e=>e.metric?.factor===1&&metric(e,'R:c1')&&metric(e,'R:c2'))));
audit('定数值圆上点：圆心到点等于半径','radius',make(k=>{free(k,'O',1,1);add(k,'circleRadius','c',['O'],{radius:5});add(k,'onCircle','B',['c'],{theta:Math.PI});}),['distanceRatio'],r=>assert(select(r,'distanceRatio').some(e=>e.metric?.factor===1&&metric(e,'R:c')&&metric(e,'D:B|O')&&e.source==='construction')));
audit('独立线段巧合等于半径不得被全归因','radius',make(k=>{free(k,'O',0,0);free(k,'A',4,0);free(k,'B',20,0);free(k,'D',24,0);add(k,'circle','c',['O','A']);}),['distanceRatio'],r=>assert(select(r,'distanceRatio').some(e=>e.metric?.factor===1&&metric(e,'R:c')&&metric(e,'D:B|D')&&e.source!=='construction')));
audit('半径=整段与中点半段两倍的不同量纲比','radius',make(k=>{free(k,'O',0,0);free(k,'A',6,0);add(k,'circle','c',['O','A']);add(k,'midpoint','M',['O','A']);}),['distanceRatio'],r=>assert(select(r,'distanceRatio').some(e=>e.metric?.factor===2&&(metric(e,'R:c')||metric(e,'D:A|O'))&&metric(e,'D:M|O'))));
// 11—20: parallelism, equal area, degeneracy and removal of trapezoids.
const baseArea=(thirdA,thirdB)=>P(['A',0,0],['B',4,0],['C',...thirdA],['D',...thirdB]);
audit('共底且第三顶点连线平行底边：等面积蕴含','parallel-area',baseArea([0,2],[3,2]),['equalArea','parallel'],r=>assert(select(r,'equalArea').some(e=>e.entailedByParallel?.length&&e.score<=25)));
audit('共底等面积但两个顶点在底边两侧：非平行同底高','parallel-area',baseArea([0,2],[3,-2]),['equalArea'],r=>assert(select(r,'equalArea').some(e=>!e.entailedByParallel?.length)));
audit('不等高的三角形不能虚构等面积','parallel-area',baseArea([0,2],[3,3.4]),['equalArea'],r=>assert(!select(r,'equalArea').some(e=>e.entailedByParallel?.length)));
audit('普通梯形不再作为发现类别','trapezoid-removed',P(['A',0,0],['B',4,0],['C',3,2],['D',0,2]),['parallel','trapezoid','isoscelesTrapezoid'],r=>assert(!r.entries.some(e=>/trapezoid/i.test(e.kind))&&registry.every(k=>!['trapezoid','isoscelesTrapezoid'].includes(k.id))));
audit('等腰梯形也不再作为发现类别','trapezoid-removed',P(['A',0,0],['B',4,0],['C',3,2],['D',1,2]),['parallel','trapezoid','isoscelesTrapezoid'],r=>assert(!r.entries.some(e=>/trapezoid/i.test(e.kind))));
audit('平行四边形仍保留，不受去梯形影响','trapezoid-removed',P(['A',0,0],['B',4,0],['C',5,2],['D',1,2]),['parallelogram','trapezoid'],r=>assert(select(r,'parallelogram').length===1&&!r.entries.some(e=>e.kind==='trapezoid')));
audit('同一底边上的三个顶点产生三角形面积簇','parallel-area',P(['A',0,0],['B',4,0],['C',.5,2],['D',2,2],['E',3.5,2]),['equalArea'],r=>assert(select(r,'equalArea').some(e=>e.entailedByParallel?.length>=2&&e.score<=25)));
audit('两个位置分离但面积碰巧相等的三角形','parallel-area',P(['A',0,0],['B',2,0],['C',0,2],['D',10,7],['E',12,7],['F',11,9]),['equalArea'],r=>assert(select(r,'equalArea').some(e=>!e.entailedByParallel?.length)));
audit('圆和平行线混图不能再产出梯形类别','trapezoid-removed',C(P(['A',0,0],['B',4,0],['C',3,2],['D',0,2]),'c',2,1,3),['parallel','distanceRatio','trapezoid'],r=>assert(!r.entries.some(e=>e.kind==='trapezoid')));
audit('没有勾选平行检测时，等面积仍能记录平行蕴含','parallel-area',baseArea([1,5],[3,5]),['equalArea'],r=>assert(select(r,'equalArea').some(e=>e.entailedByParallel?.length&&e.source==='entailed')));
// 21—30: quadratic dimension-two algebra, products, signed values, zero guards.
audit('直角三角形：面积与边长乘积的二倍关系','quadratic',P(['A',0,0],['B',4,0],['C',0,2]),['quadraticRatio'],r=>assert(select(r,'quadraticRatio').some(e=>e.quadraticTypes.includes('area')&&e.quadraticTypes.includes('product'))&&select(r,'quadraticRatio').some(e=>e.metric?.factor===1&&e.quadraticTypes.includes('sumSquares')&&e.quadraticTypes.includes('lengthSquared'))&&select(r,'quadraticRatio').some(e=>e.metric?.factor===1&&e.quadraticTypes.includes('differenceSquares')&&e.quadraticTypes.includes('lengthSquared'))));
audit('AB=AC 的共同乘因子不应单独刷一条强关系','quadratic',P(['A',0,0],['B',2,0],['C',0,2],['D',11,0],['E',14,0]),['quadraticRatio'],r=>assert(!select(r,'quadraticRatio').some(e=>e.quadraticTypes.length===1&&e.quadraticTypes[0]==='product'&&e.algebraicOnly&&e.metric?.factor===1&&e.relationEvidence?.length===1)));
audit('不同四组线段的乘积 AB·CD = EF·GH','quadratic',P(['A',0,0],['B',2,0],['C',20,0],['D',23,0],['E',40,0],['F',41.5,0],['G',60,0],['H',64,0]),['quadraticRatio'],r=>{
 const wanted=['M:D:A|B*D:C|D','M:D:E|F*D:G|H'];assert(select(r,'quadraticRatio').some(e=>e.metric?.factor===1&&wanted.every(w=>metric(e,w))));
});
audit('平方比 2 可被正向与逆向查询','quadratic',P(['A',0,0],['B',Math.SQRT2,0],['C',0,3],['D',1,3]),['quadraticRatio'],r=>{
 const e=select(r,'quadraticRatio').find(e=>e.metric?.factor===2&&metric(e,'S:D:A|B')&&metric(e,'S:D:C|D'));assert(e&&matches(e,'AB²:CD²=2')&&matches(e,'CD²:AB²=1/2'));
});
audit('圆内负点幂与圆外正点幂可以有负倍率','quadratic',C(P(['P',3,0],['Q',Math.sqrt(33),0]),'c',0,0,5),['quadraticRatio'],r=>assert(select(r,'quadraticRatio').some(e=>e.metric?.factor===-2&&e.quadraticTypes.includes('power'))));
audit('圆上零点幂不得作为零分母产生比值','degenerate',C(P(['P',5,0],['Q',3,0]),'c',0,0,5),['quadraticRatio'],r=>assert(!select(r,'quadraticRatio').some(e=>metric(e,'W:P|C:c'))));
audit('半径平方正确归入二次量且可检索','quadratic',C(P(['O',0,0],['A',3,0],['B',0,4]),'c',0,0,3,'O'),['quadraticRatio'],r=>assert(select(r,'quadraticRatio').some(e=>metric(e,'R2:c'))));
audit('平方根倍率用于两个线段乘积','quadratic',P(['A',0,0],['B',Math.SQRT2,0],['C',10,0],['D',11,0],['E',20,0],['F',21,0],['G',30,0],['H',31,0]),['quadraticRatio'],r=>assert(select(r,'quadraticRatio').some(e=>Math.abs(e.metric?.factor-Math.SQRT2)<1e-10&&e.quadraticTypes.includes('product'))));
audit('二次面积 2[IJK] 的倍率保留','quadratic',P(['A',0,0],['B',4,0],['C',0,2],['D',10,0],['E',12,0],['F',10,2]),['quadraticRatio'],r=>assert(select(r,'quadraticRatio').some(e=>e.metric?.factor===2&&e.quadraticTypes.includes('area'))));
audit('随机非特殊位置不应误报黄金比例二次关系','degenerate',P(['A',.13,.45],['B',1.39,2.71],['C',-1.61,.53],['D',3.21,-.79]),['quadraticRatio'],r=>assert(!select(r,'quadraticRatio').some(e=>Math.abs(Math.abs(e.metric?.factor||0)-(1+Math.sqrt(5))/2)<1e-11)));
// 31—40: quadrilateral fact closure must preserve all vertex correspondence.
const qPts={square:P(['A',0,0],['B',2,0],['C',2,2],['D',0,2]),rectangle:P(['A',0,0],['B',4,0],['C',4,2],['D',0,2]),rhombus:P(['A',0,2],['B',3,0],['C',0,-2],['D',-3,0]),parallelogram:P(['A',0,0],['B',4,0],['C',5,2],['D',1,2]),kite:P(['A',0,3],['B',-2,0],['C',0,-1],['D',2,0])};
audit('正方形：四边相等、对角线、圆、全等均有子事实','quadrilateral',qPts.square,['square'],r=>{const e=select(r,'square')[0];assert(e&&['equalLength','parallel','rightAngle','concyclic','congruentTriangles','perpendicular','diagonalsBisect','equalArea','angleBisector','supplementaryAngles'].every(k=>e.derivedFacts.some(f=>f.kind===k)));assert(matches(e,'AB=BC')&&matches(e,'矩形'));});
audit('矩形：对角线相等、共圆及四直角','quadrilateral',qPts.rectangle,['rectangle'],r=>{const e=select(r,'rectangle')[0];assert(e&&e.derivedFacts.some(f=>f.kind==='concyclic')&&e.derivedFacts.some(f=>f.kind==='equalLength'&&f.alias==='AC=BD')&&e.derivedFacts.filter(f=>f.kind==='rightAngle').length===4&&e.derivedFacts.filter(f=>f.kind==='supplementaryAngles').length===4);});
audit('非方形菱形：边等长且对角线垂直','quadrilateral',qPts.rhombus,['rhombus','square'],r=>{const e=select(r,'rhombus')[0];assert(e&&!select(r,'square').length&&e.derivedFacts.some(f=>f.kind==='perpendicular')&&e.derivedFacts.filter(f=>f.kind==='equalLength').length>=4&&e.derivedFacts.filter(f=>f.kind==='angleBisector').length===4);});
audit('一般平行四边形：两组平行、对边等长及对角线平分','quadrilateral',qPts.parallelogram,['parallelogram'],r=>{const e=select(r,'parallelogram')[0];assert(e&&e.derivedFacts.filter(f=>f.kind==='parallel').length===2&&e.derivedFacts.filter(f=>f.kind==='equalLength').length===2&&e.derivedFacts.some(f=>f.kind==='diagonalsBisect')&&e.derivedFacts.filter(f=>f.kind==='equalArea').length===2&&e.derivedFacts.filter(f=>f.kind==='equalAngle').length===2);});
audit('第一种筝形：对称轴平分并且分割三角形全等','quadrilateral',qPts.kite,['kite'],r=>{const e=select(r,'kite')[0];assert(e&&e.derivedFacts.some(f=>f.kind==='diagonalBisector')&&e.derivedFacts.some(f=>f.kind==='congruentTriangles')&&e.derivedFacts.filter(f=>f.kind==='equalLength').length===2&&e.derivedFacts.filter(f=>f.kind==='angleBisector').length===2&&e.derivedFacts.some(f=>f.kind==='equalArea'));});
audit('第二种筝形顶点顺序：仍正确定位对称对角线','quadrilateral',P(['A',-3,0],['B',0,2],['C',1,0],['D',0,-2]),['kite'],r=>{const e=select(r,'kite')[0];assert(e&&e.derivedFacts.some(f=>f.kind==='diagonalBisector')&&e.derivedFacts.filter(f=>f.kind==='equalLength').length===2&&e.derivedFacts.filter(f=>f.kind==='angleBisector').length===2&&e.derivedFacts.some(f=>f.kind==='equalArea'));});
audit('仅扫描正方形时仍能通过蕴含检索矩形和菱形','quadrilateral',qPts.square,['square'],r=>{const e=select(r,'square')[0];assert(e&&matches(e,'矩形')&&matches(e,'菱形')&&matches(e,'对角线'));});
audit('边长的直接子关系被四边形吸收后仍能搜索','quadrilateral',qPts.rectangle,['rectangle','distanceRatio'],r=>{const e=select(r,'rectangle')[0];assert(e&&matches(e,'AB=CD')&&e.entails?.some(f=>f.kind==='distanceRatio'));});
audit('构造得到的平行四边形自身是条件而非高分发现','construction',make(k=>{free(k,'A',0,0);free(k,'B',4,0);free(k,'C',1,2);add(k,'parallelogram','D',['A','B','C']);}),['parallelogram','parallel'],r=>{const e=select(r,'parallelogram')[0];assert(e&&e.source==='construction'&&e.score<=22);});
audit('不规则凸四边形不应误报为筝形/矩形/菱形','quadrilateral',P(['A',0,0],['B',4,0],['C',3,2],['D',-1,1]),['kite','rectangle','rhombus','square'],r=>assert(!r.entries.some(e=>['kite','rectangle','rhombus','square'].includes(e.kind))));
// 41—45: harmonic division must not depend on alphabetical labelling.
audit('调和点列基础构图识别','harmonic',P(['A',0,0],['B',2,0],['C',.5,0],['D',-1,0]),['harmonic','quadraticRatio'],r=>{const e=select(r,'harmonic')[0];assert(e&&e.derivedFacts.some(f=>f.kind==='equalProduct'&&f.alias==='AC·BD=AD·BC')&&matches(e,'二次量连等'));});
audit('改换字母顺序后交比为负一仍能识别','harmonic',P(['C',0,0],['D',2,0],['B',.5,0],['A',-1,0]),['harmonic'],r=>assert(select(r,'harmonic').length===1));
audit('普通四共线点并非必然调和','harmonic',P(['A',0,0],['B',1,0],['C',2,0],['D',4,0]),['harmonic'],r=>assert(select(r,'harmonic').length===0));
audit('接近但不精确的调和点列不得蒙混通过','harmonic',P(['A',0,0],['B',2,0],['C',.5001,0],['D',-1,0]),['harmonic'],r=>assert(select(r,'harmonic').length===0));
audit('调和四点加第五点仍只对应它自己的四点组','harmonic',P(['A',0,0],['B',2,0],['C',.5,0],['D',-1,0],['E',7.25,0]),['harmonic'],r=>assert(select(r,'harmonic').some(e=>e.id==='harmonic:A|B|C|D')));
// 46—50: constraints, nontrivial theorems, and pathological inputs.
audit('三条同端点构造线共点属于定义条件低分','score',make(k=>{for(const [id,x,y] of [['A',0,0],['B',4,0],['C',-1,3],['D',-2,-2]])free(k,id,x,y);for(const id of ['B','C','D'])add(k,'line','l'+id,['A',id]);}),['concurrent'],r=>{const e=select(r,'concurrent').find(e=>e.title.includes('A'));assert(e&&e.source==='construction'&&e.score<=12);});
audit('三条高线汇交仍为非平凡而非同锚点定义','score',make(k=>{for(const [id,x,y] of [['A',0,0],['B',4,0],['C',1.1,3]])free(k,id,x,y);add(k,'orthocenter','H',['A','B','C']);for(const [id,ins] of [['hA',['A','B','C']],['hB',['B','A','C']],['hC',['C','A','B']]])add(k,'perpendicular3',id,ins);}),['concurrent'],r=>{const e=select(r,'concurrent').find(e=>e.title.includes('H'));assert(e&&e.source!=='construction'&&e.score>12);});
audit('中点直接定义的长度等式评分低','score',make(k=>{free(k,'A',0,0);free(k,'B',4,0);add(k,'midpoint','M',['A','B']);}),['midpoint','distanceRatio'],r=>{const e=select(r,'distanceRatio').find(e=>e.metric?.factor===1&&metric(e,'D:A|M')&&metric(e,'D:B|M'));assert(e&&e.source==='construction'&&e.score<=22);});
audit('零半径与退化线段不进入半径及二次比值','degenerate',C(P(['A',0,0],['B',0,0],['C',1,0]),'c',0,0,0),['distanceRatio','quadraticRatio'],r=>assert(!r.entries.some(e=>JSON.stringify(e.metric||{}).includes('R:c')||JSON.stringify(e.metric||{}).includes('S:D:A|B'))));
audit('平行构造产生的同底等面积关系不得抢高分','score',make(k=>{free(k,'A',0,0);free(k,'B',4,0);free(k,'C',0,3);add(k,'line','base',['A','B']);add(k,'parallel','top',['C','base']);add(k,'onLine','D',['top'],{t:2});}),['equalArea','parallel'],r=>{const e=select(r,'equalArea').find(e=>e.entailedByParallel?.length);assert(e&&e.score<=25);});
assert.equal(report.length,50,'exact 50 dumb diagrams');
const summary={schema:'geomtool.v4.1.50-scenario-audit/v1',source:'executed kernel and detector numerical snapshots',scenes:report.length,passed:report.filter(x=>x.passed).length,results:report};
const out=path.join(__dirname,'../docs/V41_DUMMY_50_AUDIT.json');fs.writeFileSync(out,JSON.stringify(summary,null,2));
console.log('PASS v4.1:',report.length,'independently specified dumb diagrams, including source, implication, degeneracy, ranking, search and algebraic operands');
