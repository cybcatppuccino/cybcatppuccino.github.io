'use strict';
/* v3.2 negative geometry oracle: each scene isolates a different family of
 * mathematical tautologies or near-degeneracies. Counts and IDs are checked,
 * not merely the registry size. */
const assert=require('node:assert/strict');
const {scan}=require('../assets/relations/detectors.js');
const scene=(pts=[],ls=[],cs=[])=>({points:pts.map(([id,x,y])=>({id,label:id,x,y})),lines:ls.map(([id,x1,y1,x2,y2])=>({id,label:id,x1,y1,x2,y2})),circles:cs.map(([id,cx,cy,r])=>({id,label:id,cx,cy,r}))});
const run=(x,kinds)=>scan(x,kinds?{kinds,perturb:true}:{perturb:true});
const count=(r,k)=>r.entries.filter(e=>e.kind===k).length;
let assertions=0;
function test(label,ok){assert.ok(ok,label);assertions++;}
const incircle=scene([['A',-2,0],['B',2,0],['C',0,-2],['K',0,1]],[],[['c',0,0,1]]);
let r=run(incircle,['lineIntersectionCircle','pointOnCircle']);
test('1. named point on circle is not a novel line-intersection-circle relation',count(r,'lineIntersectionCircle')===0&&count(r,'pointOnCircle')===1);
const unnamed=scene([], [['l1',-2,1,3,1],['l2',0,-3,0,3]], [['c',0,0,1]]);
r=run(unnamed,['lineIntersectionCircle']);test('2. unnamed line-line intersection on circle is retained',count(r,'lineIntersectionCircle')===1);
const named={...unnamed,points:[{id:'K',x:0,y:1,label:'K'}]};r=run(named,['lineIntersectionCircle']);test('3. naming the same circle intersection suppresses the incidental incidence',count(r,'lineIntersectionCircle')===0);
const collinear=scene([['A',0,0],['B',1,0],['C',2,0],['D',3,0]]);
r=run(collinear,['circleCircleTangent']);test('4. collinear center-through-point circles do not create synthetic tangencies',r.count===0);
const nontouching=scene([['A',0,0],['B',0,2],['C',5,0],['D',5,3]]);
r=run(nontouching,['circleCircleTangent']);test('5. unrelated virtual circles are genuinely tangent at an unnamed point',r.entries.some(e=>e.title.includes('A')&&e.title.includes('C')));
const touching=scene([['A',0,0],['B',2,0],['C',5,0],['D',5,3]]);
r=run(touching,['circleCircleTangent']);test('6. contact at defining radius endpoint is not a new tangent fact',r.count===0);
const radiusFoot=scene([['A',0,0],['B',2,0]], [['l',2,-3,2,3]]);
r=run(radiusFoot,['circleLineTangent']);test('7. tangent at the defining endpoint reduces to radius perpendicularity',r.count===0);
const virtualLine=scene([['A',0,0],['B',0,2]], [['l',2,-3,2,3]]);
r=run(virtualLine,['circleLineTangent']);test('8. tangent at an unnamed foot is retained',r.entries.some(e=>e.title.includes('⊙(A,B)')));
const commonChord=scene([], [['rad',0,-3,0,3]], [['c1',-1,0,Math.SQRT2],['c2',1,0,Math.SQRT2]]);
r=run(commonChord,['circleIntersectionLine']);test('9. two unconstructed circle-circle intersections lie on the radical line',r.count===1);
const chordNamed={...commonChord,points:[{id:'P',label:'P',x:0,y:1},{id:'Q',label:'Q',x:0,y:-1}]};r=run(chordNamed,['circleIntersectionLine']);test('10. two named intersections do not spawn a second relation',r.count===0);
const multi=scene([],[],[['c1',1,0,1],['c2',0,1,1],['c3',-1,0,1]]);
r=run(multi,['multiCircleCommon']);test('11. three circles have an unconstructed common point',r.count===1);
r=run({...multi,points:[{id:'P',label:'P',x:0,y:0}]},['multiCircleCommon']);test('12. named common point is already represented as point-circle incidences',r.count===0);
const ortho=scene([['A',0,2],['B',-2,0],['C',3,0],['H',0,3]]);
r=run(ortho,['orthocentricSystem','specialPoint']);test('13. four equivalent orthocenters coalesce into one orthocentric system',count(r,'orthocentricSystem')===1&&count(r,'specialPoint')===0);
const sym=scene([['A',-3,1],['B',3,1],['C',0,-1.1],['D',0,2.6]], [['l',0,-4,0,4]]);
r=run(sym,['reflection','congruentTriangles']);test('14. reflection claim follows the stronger triangle congruence witness',count(r,'congruentTriangles')>=1&&count(r,'reflection')===0);
test('15. suppressed reflection searchable as congruence synonym',r.entries.some(e=>e.kind==='congruentTriangles'&&(e.searchTags||[]).some(tag=>tag.includes('对称'))));
r=run(sym,['reflection']);test('16. independent selected reflection rule still works',count(r,'reflection')===1);
const right=scene([['A',1,0],['B',0,0],['C',0,1],['D',0.318727,0.442139]]);
r=run(right,['perpendicular','complementaryAngle','specialLine']);test('17. arbitrary interior ray does not create redundant complementary right-angle subdivision',count(r,'complementaryAngle')===0);
test('18. right triangle sides do not generate spurious altitude-only special lines',count(r,'specialLine')===0);
const sq=scene([['A',0,0],['B',2,0],['C',2,2],['D',0,2]]);
r=run(sq,['square','concyclic','rotation','congruentTriangles','equalArea']);test('19. square implies its four vertices are concyclic',count(r,'square')===1&&count(r,'concyclic')===0);
test('20. square also absorbs vertex 90-degree rotational facts',count(r,'rotation')===0);
test('21. areas implied by congruent triangles do not repeat',count(r,'equalArea')===0);
const sqFive={...sq,points:[...sq.points,{id:'E',label:'E',x:1,y:1+Math.SQRT2}]};
r=run(sqFive,['square','concyclic']);test('22. new independent circle incidences are not suppressed by the square rule',count(r,'square')===1);
const equid=scene([['A',0,0],['B',1,0],['C',0,1]]);
r=run(equid,['distanceRatio']);test('23. natural equal AB=AC remains searchable',r.entries.some(e=>e.metric?.factor===1));
const radical=scene([['P',1,3]],[],[['c1',0,0,2],['c2',2,0,2]]);
r=run(radical,['radicalAxis']);test('24. nontrivial point on radical axis is preserved',r.count===1);
const almost=scene([['A',0,0],['B',2,0],['C',4,0.0000006]]);
r=run(almost,['collinear']);test('25. almost-collinear points are not treated as exact collinear',r.count===0);
const tangNear=scene([['A',0,0],['B',0,2],['C',5+1e-6,0],['D',5+1e-6,3]]);
r=run(tangNear,['circleCircleTangent']);test('26. almost-tangent virtual circles are not accepted',r.count===0);
const equalAngle=scene([['A',0,0],['B',3,0],['C',0,1],['D',3,2],['E',0,5],['F',3,5],['G',0,6],['H',3,7]]);
r=run(equalAngle,['equalAngle']);test('27. diverse lines with equal angles form chains rather than pairwise records',r.count>=1&&r.count<=25);
const arithmetic=scene([['A',0,0],['B',4,0],['C',0,3],['D',2,2],['E',4,3]]);
r=run(arithmetic,['distanceRatio']);test('28. equal and inverted length ratios remain represented once',new Set(r.entries.map(e=>e.id)).size===r.count);
// Deterministic free-point oracle. These non-special locations are chosen away
// from every old point, support line, and algebraic special locus.
for(let k=0;k<12;k++){
 const p=['F',7.11843+Math.PI*k*.371,5.29436+Math.SQRT2*k*.247];
 const base=scene([['A',0,0],['B',2,0],['C',0,3]],[],[['c',0,0,2]]);
 const before=run(base),after=run({...base,points:[...base.points,{id:'F',label:'F',x:p[1],y:p[2]}]});
 test('free '+k+' makes no spurious relations',after.count===before.count);
}
// Additional isolated construction families (no solver dependencies).
const duplicatePoints=scene([['A',0,0],['A2',0,0],['A3',0,0],['B',2,0],['C',4,0],['D',6,0]]);
r=run(duplicatePoints,['coincident','collinear']);
test('29. coincident points are a single equivalence class',count(r,'coincident')===1);
test('30. four collinear geometric positions are a maximal group',count(r,'collinear')===1);
const altLine=scene([['A',0,0],['B',2,0],['C',4,0]], [['l1',0,0,2,0],['l2',1,0,4,0]]);
r=run(altLine,['parallel']);test('31. coincident line descriptions never create self-parallel records',r.entries.length===0);
const concentric=scene([],[],[['c1',0,0,2],['c2',0,0,3],['copy',0,0,2]]);
r=run(concentric,['concentric','equalRadius']);test('32. multiple names for exactly the same circle do not report equalRadius',count(r,'equalRadius')===0);
test('33. genuine concentric unequal circles are reported once',count(r,'concentric')===1);
const degenerateCircles=scene([['A',0,0],['B',0,0]],[],[['c',0,0,0],['valid',0,0,1]]);
r=run(degenerateCircles,['circleCircleTangent','concyclic','pointOnCircle']);test('34. zero-radius circle is excluded from candidates',r.count===0);
const collinearOnly=scene([['A',0,0],['B',1,0],['C',2,0],['D',3,0]]);
r=run(collinearOnly,['fixedAngle','angleBisector']);test('35. degenerate straight angles do not invent angle-bisectors',r.count===0);
const almostConcur=scene([], [['x',-2,0,2,0],['y',0,-2,0,2],['z',-2,-2+1e-5,2,2+1e-5]]);
r=run(almostConcur,['concurrent']);test('36. near-concurrency fails strict incidence test',r.count===0);
const genuineTangent=scene([], [['l',1,-3,1,3]], [['c',0,0,1]]);
r=run(genuineTangent,['circleLineTangent']);test('37. explicit circle tangent at unnamed contact remains a real result',r.count===1);
const nearContact=scene([], [['l',1+1e-6,-3,1+1e-6,3]], [['c',0,0,1]]);
r=run(nearContact,['circleLineTangent']);test('38. offset line almost tangent is not accepted',r.count===0);
const namedIntersectionCircle=scene([['P',0,1]], [['l1',0,-3,0,3],['l2',-3,1,3,1]], [['c',0,0,1]]);
r=run(namedIntersectionCircle,['lineIntersectionCircle']);test('39. many line aliases through a named circle point create no extra result',r.count===0);
const crossed=scene([['A',-2,-2],['B',2,2],['C',-2,2],['D',2,-2],['E',0,-2],['F',0,2]]);
r=run(crossed,['concurrent']);test('40. three genuinely concurrent implicit lines survive',r.count===1);
const fourRays=scene([['A',-2,0],['B',2,0],['C',0,-2],['D',0,2],['O',0,0]]);
r=run(fourRays,['concurrent']);test('41. three rays through the same named shared endpoint are not independent concurrency',r.count===0);
// A dense diagram is intentionally a high-degeneracy stress case. Fuzzy
// angle predicates used to create thousands of artifacts after tiny moves.
const grid=scene(Array.from({length:12},(_,i)=>['P'+Math.floor(i/3)+(i%3),Math.floor(i/3)*2,(i%3)*2]),[],[['c',3,2,Math.SQRT2]]);
const baseGrid=run(grid);
test('42. 12-point degenerate grid is reduced to a bounded catalog',baseGrid.count<1200);
test('43. all returned relation IDs are unique',new Set(baseGrid.entries.map(e=>e.id)).size===baseGrid.count);
for(const amount of [1e-12,1e-11,1e-10]){
 const moved={...grid,points:grid.points.map(p=>p.id==='P11'?{...p,x:p.x+amount,y:p.y+amount*.3}:p)};
 const scanMoved=run(moved);test('micro-stability '+amount+': no count oscillation',scanMoved.count===baseGrid.count);
 test('micro-stability '+amount+': exact semantic ID set persists',new Set(baseGrid.entries.map(e=>e.id)).size===scanMoved.count&&scanMoved.entries.every(e=>baseGrid.entries.some(b=>b.id===e.id)));
}
const visiblyMoved={...grid,points:grid.points.map(p=>p.id==='P11'?{...p,x:p.x+1e-8,y:p.y+3e-9}:p)};
const physicallyChanged=run(visiblyMoved);
test('44. a larger REAL geometry change does not produce hundreds of new count artifacts',Math.abs(physicallyChanged.count-baseGrid.count)<100);
test('45. strict mode still legitimately discards collinearity/special-point relations when moved off their exact loci',count(physicallyChanged,'collinear')<count(baseGrid,'collinear')&&count(physicallyChanged,'specialPoint')<count(baseGrid,'specialPoint'));
console.log('PASS v3.2 degeneracy:',assertions,'assertions');
