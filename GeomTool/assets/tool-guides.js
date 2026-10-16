/* Unique labeled geometric construction sketches and step-by-step tool guides. */
(function(root,factory){const api=factory();root.GeomToolGuides=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const guides={
 select:[['点击对象查看其属性；按住活动点即可沿自由度拖动','在画布空白处拖动以平移，滚轮缩放；按空格可强制平移'],'选择、移动与平移整合在一个工具中'],
 smartPoint:[['移动光标：预览虚像点，邻近的载体高亮','单击一次：空白作自由点；线上/圆上作约束点；交会附近作固定分支交点'],'自动识别放点位置，永远单击一次即可完成'],
 pan:[['在画板上按住并拖动','松开鼠标以完成平移'],'改变视图，不改变几何坐标'],
 free:[['在画板任意位置点击以放置 A','继续点击以放置 B 等自由点'],'创建可自由拖动的点 A'],
 onCarrier:[['点击直线 l 或圆 c 的相应位置','拖动新点可沿载体移动'],'在 l 或 c 上建立受约束点 P'],
 similarPoint:[['依次点击 A、B、C','依次点击 D、E，产生 F'],'按对应 A↔D、B↔E、C↔F 构造同向相似点，允许共线三角形'],
 equalAngleLine:[['依次点击 A、B、C 确定有向角 ABC','依次点击 D、E，构造过 E 的直线 l'],'∠ABC = ∠DEl（模 180°），l 经过 E'],
 midpoint:[['第一步：点击点 A','第二步：点击点 B'],'生成 AB 的中点 M'],
 foot:[['点击点 P','点击直线 l'],'生成 P 到 l 的垂足 H'],
 centroid:[['依次点击 A、B、C 三个顶点','完成第三次点击'],'生成 △ABC 的重心 G'],
 circumcenter:[['依次点击 A、B、C 三个顶点','完成第三次点击'],'生成 △ABC 的外心 O'],
 incenter:[['依次点击 A、B、C 三个顶点','完成第三次点击'],'生成 △ABC 的内心 I'],
 orthocenter:[['依次点击 A、B、C 三个顶点','完成第三次点击'],'生成 △ABC 的垂心 H'],
 excenter:[['先点击 A，再点击 B','最后点击 C'],'生成 △ABC 的 A 旁心 J'],
 parallelogram:[['依次点击 A、B、C','令 AB 与 CD 平行'],'生成平行四边形的第四个顶点 D（B+C−A）'],
 line:[['点击点 A','点击点 B'],'生成经过 A、B 的无限直线 l'],
 segment:[['点击端点 A','点击端点 B'],'生成线段 AB'],
 ray:[['点击起点 A','点击方向点 B'],'生成从 A 出发经过 B 的射线'],
 parallel:[['点击直线外的点 P','点击参考直线 l'],'生成经过 P 且平行 l 的直线 m'],
 perpendicular:[['点击点 P','点击直线 l'],'生成经过 P 且垂直 l 的直线 h'],
 perpBisector:[['点击端点 A','点击端点 B'],'生成线段 AB 的中垂线'],
 angleBisector:[['点击点 A','点击角顶点 B，再点击点 C'],'生成 ∠ABC 的内角平分线'],
 tangent:[['点击位于圆 c 上的点 P','点击圆 c'],'生成圆 c 在 P 处的切线'],
 tangentFrom:[['点击圆外的点 P','点击圆 c；可配置分支 0/1'],'生成从 P 引出的指定分支切线'],
 parallel3:[['点击过线点 P','再依次点击 A、B'],'生成经过 P 且平行 AB 的直线'],
 perpendicular3:[['点击过线点 P','再依次点击 A、B'],'生成经过 P 且垂直 AB 的直线'],
 outerAngleBisector:[['依次点击 A、B、C','B 是角顶点'],'生成 ∠ABC 的外角平分线'],
 circle:[['点击圆心 O','点击圆上的点 A'],'生成以 O 为圆心、OA 为半径的圆'],
 circleRadius:[['设置左侧半径数值 r','点击圆心 O'],'生成半径 r 的圆'],
 circumcircle:[['依次点击 A、B、C 三个点','完成第三次点击'],'生成经过 A、B、C 的外接圆'],
 polygon:[['依次点击 A、B、C …（至少三点）','双击画布或按 Enter 完成'],'生成可选中的多边形'],
 circleBySegment:[['点击圆心 O','依次点击 A、B，指定半径 AB'],'生成以 O 为圆心、AB 为半径的圆'],
 diameterCircle:[['点击直径端点 A','点击直径端点 B'],'生成以 AB 为直径的圆'],
 equilateral:[['点击端点 A','点击端点 B；可选择方向分支'],'生成满足 △ABC 为等边三角形的 C'],
 foot3:[['点击点 P','依次点击 A、B 定义直线'],'生成 P 到 AB 的垂足 H'],
 reflectPoint:[['点击原点 P','点击对称中心 O'],'生成 P 关于 O 的中心对称点 P′'],
 reflectLine:[['点击原点 P','点击对称轴 l'],'生成 P 关于 l 的轴对称点 P′'],
 translate:[['点击被移动点 P','点击起点 A，再点击终点 B'],'生成 P′=P+(B−A)'],
 rotate:[['设置旋转角度（度）','点击点 P，再点击中心 O'],'生成绕 O 旋转指定角度的 P′'],
 homothety:[['设置位似比例 k','点击点 P，再点击中心 O'],'生成以 O 为中心、比例为 k 的位似点 P′'],
 rotateLeft:[['点击点 P','点击旋转中心 O'],'生成逆时针旋转 90° 的 P′'],
 rotateRight:[['点击点 P','点击旋转中心 O'],'生成顺时针旋转 90° 的 P′']
};
const pos={A:[22,72],B:[118,72],C:[70,12],P:[35,20],O:[70,50],M:[70,72],H:[35,72],D:[118,20],X:[70,44],Q:[100,33],I:[70,51],G:[70,50],J:[105,102],L:[18,51],R:[122,51],T:[99,19],U:[70,81]};
const path=(a,b,cls='')=>`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" class="${cls}"/>`;
const L=(a,b,cls='')=>path(pos[a]||a,pos[b]||b,cls);
const P=(name,p=pos[name],out=false)=>`<circle cx="${p[0]}" cy="${p[1]}" r="3.5" class="${out?'output':'sample-dot'}"/><text x="${p[0]+6}" y="${p[1]-7}" class="${out?'output-label':''}">${name}</text>`;
const C=(c,r,cls='')=>`<circle cx="${c[0]}" cy="${c[1]}" r="${r}" class="${cls}"/>`;
const poly=(str,cls='')=>`<polygon points="${str}" class="${cls}"/>`;
const guideLine=(a,b)=>L(a,b,'guide-line');
const output=(a,b)=>L(a,b,'result-line');
const A=()=>P('A')+P('B'),triangle=()=>L('A','B')+L('B','C')+L('C','A')+P('A')+P('B')+P('C');
function drawing(id){
 switch(id){
 case 'select':return triangle()+C([70,48],12,'result-line')+P('M',undefined,true);
 case 'pan':return `<path d="M18 49H119m-9-8 9 8-9 8M70 16v63m-8-9 8 9 8-9" class="result-line"/>`+P('O');
 case 'smartPoint':return L([8,68],[133,25])+C([96,35],24)+P('P',[73,46],true)+P('A',[20,20])+P('X',[112,51],true);
 case 'free':return P('A')+P('B',[102,23],true);
 case 'onCarrier':return L([8,68],[133,25])+P('P',[79,44],true);
 case 'similarPoint':return L([12,70],[48,18])+L([48,18],[76,70])+L([76,70],[12,70])+L([84,78],[101,30])+L([101,30],[130,78])+L([130,78],[84,78],'result-line')+P('A',[12,70])+P('B',[48,18])+P('C',[76,70])+P('D',[84,78])+P('E',[101,30])+P('F',[130,78],true);
 case 'equalAngleLine':return L([16,68],[56,30])+L([56,30],[92,63])+L([10,24],[110,77])+output([78,5],[129,83])+P('A',[16,68])+P('B',[56,30])+P('C',[92,63])+P('D',[10,24])+P('E',[110,77]);
 case 'midpoint':return L('A','B')+A()+P('M',undefined,true);
 case 'foot':case 'foot3':return L('A','B')+guideLine('P','H')+A()+P('P')+P('H',undefined,true)+`<path d="M35 64h8v8" class="detail-line-svg"/>`;
 case 'centroid':return triangle()+guideLine('C','M')+P('M')+P('G',[70,52],true);
 case 'circumcenter':return triangle()+C([70,61.2],49.2,'guide-circle')+P('O',[70,61.2],true);
 case 'incenter':return triangle()+L('B',[70,49],'guide-line')+L('C',[70,49],'guide-line')+P('I',[70,49],true);
 case 'orthocenter':return triangle()+guideLine('C','M')+guideLine('A',[97,12])+P('H',[70,33.6],true);
 case 'excenter':return L([20,76],[88,76])+L([88,76],[54,32])+L([54,32],[20,76])+guideLine([20,76],[109.6,32])+guideLine([88,76],[109.6,32])+P('A',[20,76])+P('B',[88,76])+P('C',[54,32])+P('J',[109.6,32],true);
 case 'parallelogram':return poly('20,70 75,70 97,20 42,20','guide-poly')+L([20,70],[75,70])+output([75,70],[97,20])+output([42,20],[97,20])+P('A',[20,70])+P('B',[75,70])+P('C',[42,20])+P('D',[97,20],true);
 case 'line':return L([4,80],[136,24],'result-line')+P('A',[27,70])+P('B',[106,36]);
 case 'segment':return L('A','B','result-line')+A();
 case 'ray':return L('A',[132,72],'result-line')+`<path d="m125 66 7 6-7 6" class="result-line"/>`+A();
 case 'parallel':return L([8,73],[131,73])+output([7,30],[132,30])+P('P',[40,30],true);
 case 'perpendicular':return L([8,72],[132,72])+output([52,6],[52,87])+P('P',[52,22],true);
 case 'perpBisector':return L('A','B')+output([70,7],[70,85])+A()+P('M');
 case 'angleBisector':return L('B','A')+L('B','C')+output('B',[31,30])+P('A')+P('B')+P('C');
 case 'outerAngleBisector':{
  const b=pos.B,a=pos.A,c=pos.C,u=[a[0]-b[0],a[1]-b[1]],v=[c[0]-b[0],c[1]-b[1]];
  const ul=Math.hypot(...u),vl=Math.hypot(...v),d=[u[0]/ul-v[0]/vl,u[1]/ul-v[1]/vl];
  return L('B','A')+L('B','C')+output([b[0]-d[0]*24,b[1]-d[1]*24],[b[0]+d[0]*24,b[1]+d[1]*24])+P('A')+P('B')+P('C');
 }
 case 'parallel3':return L('A','B')+output([7,20],[133,20])+A()+P('P',[35,20],true);
 case 'perpendicular3':return L('A','B')+output([70,4],[70,89])+A()+P('P',[70,20],true);
 case 'tangent':return C([70,48],31)+output([11,17],[130,17])+P('P',[70,17],true)+P('O',[70,48]);
 case 'tangentFrom':return C([83,49],24)+L([11,20],[84.65,24.98],'result-line')+P('P',[11,20])+P('T',[84.65,24.98],true);
 case 'circle':return C([70,48],34,'result-line')+P('O',[70,48])+P('A',[104,48]);
 case 'circleRadius':return C([70,48],34,'result-line')+P('O',[70,48])+L([70,48],[104,48],'guide-line')+`<text x="86" y="43">r</text>`;
 case 'circumcircle':return C([70,48],40,'result-line')+P('A',[36,69])+P('B',[104,69])+P('C',[70,8]);
 case 'polygon':return poly('19,64 49,12 105,18 123,71 67,84','result-poly')+P('A',[19,64])+P('B',[49,12])+P('C',[105,18])+P('D',[123,71]);
 case 'circleBySegment':return C([70,48],30,'result-line')+P('O',[70,48])+L([11,82],[41,82],'guide-line')+P('A',[11,82])+P('B',[41,82]);
 case 'diameterCircle':return C([70,48],39,'result-line')+L([31,48],[109,48],'guide-line')+P('A',[31,48])+P('B',[109,48]);
 case 'equilateral':return L([38,72],[102,72])+output([38,72],[70,16.57])+output([102,72],[70,16.57])+P('A',[38,72])+P('B',[102,72])+P('C',[70,16.57],true);
 case 'reflectPoint':return guideLine([25,24],[115,72])+P('P',[25,24])+P('O',[70,48])+P('P′',[115,72],true);
 case 'reflectLine':return L([70,4],[70,88])+guideLine([29,32],[111,32])+P('P',[29,32])+P('P′',[111,32],true);
 case 'translate':return L([21,76],[68,76],'guide-line')+`<path d="m61 70 7 6-7 6" class="guide-line"/>`+P('A',[21,76])+P('B',[68,76])+P('P',[38,23])+P('P′',[85,23],true)+output([38,23],[85,23]);
 case 'rotate':return C([65,49],36,'guide-circle')+P('O',[65,49])+P('P',[101,49])+P('P′',[86,20],true)+output([65,49],[101,49])+output([65,49],[86,20]);
 case 'homothety':return output([37,45],[127,45])+P('O',[37,45])+P('P',[72,45])+P('P′',[125,45],true);
 case 'rotateLeft':return P('O',[61,55])+P('P',[106,55])+P('P′',[61,10],true)+output([61,55],[106,55])+output([61,55],[61,10]);
 case 'rotateRight':return P('O',[61,39])+P('P',[106,39])+P('P′',[61,84],true)+output([61,39],[106,39])+output([61,39],[61,84]);
 default:return triangle();
 }
}
function sketch(id,large=false){return `<svg class="tool-sketch ${large?'large':''}" viewBox="0 0 140 92" role="img" aria-label="${id} 作图示意"><g class="sketch-geometry">${drawing(id)}</g></svg>`;}
function guide(id){const g=guides[id]||[[`依次选取所需对象`,'完成构造'],`生成 ${id}`];return {steps:g[0],effect:g[1],svg:sketch(id,true)};}
return {sketch,guide,guides};
});
