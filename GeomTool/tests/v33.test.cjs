'use strict';
/* Targeted numeric, provenance and implication regressions for v3.3. */
const assert=require('node:assert/strict');
const {scan,registry}=require('../assets/relations/detectors.js');
const {matches}=require('../assets/relations/search.js');
const {makeContext}=require('../assets/relations/core.js');
const {Kernel,runText}=require('../assets/kernel.js');
const {IMPLICATIONS}=require('../assets/relations/provenance.js');
const fs=require('node:fs');
let n=0;const ok=(c,t)=>{assert.ok(c,t);n++;};
const P=(a,x,y)=>({id:a,label:a,x,y});
const scene=(pts,lines=[],circles=[])=>({points:pts.map(p=>P(...p)),lines:lines.map(([id,x1,y1,x2,y2])=>({id,label:id,x1,y1,x2,y2})),circles:circles.map(([id,cx,cy,r])=>({id,label:id,cx,cy,r}))});
const run=(s,kinds,perturb=false)=>scan(s,{kinds,perturb});
const kinds=registry.map(r=>r.id);
ok(kinds.includes('quadraticRatio'),'unified degree-two detector exposed');
ok(!registry.find(r=>r.id==='quadraticRatio').enabled,'expensive second-order detector is opt-in');
const original=scene([['A',0,0],['B',4,0],['C',0,3]]);
const frozen=JSON.stringify(original);
const raw=run(original,['distanceRatio']);ok(raw.perturbations===0,'micro-perturbation off by default');
const tested=run(original,['distanceRatio'],true);ok(tested.perturbations===2,'micro-perturbation on performs two internal passes');
ok(JSON.stringify(original)===frozen,'no coordinates or other snapshot fields mutated');
ok(raw.tolerance==='strict'&&tested.tolerance==='strict','precision mode always strict');
// SSS checks both the verdict and CORRESPONDENCE, not just sorted side lengths.
const twins=scene([['A',0,0],['B',3,0],['C',0,4],['D',8,1],['E',8,5],['F',11,1]]);
const cong=run(twins,['congruentTriangles']).entries.find(e=>e.title.includes('ABC')&&e.title.includes('D'));
ok(!!cong,'congruent 3-4-5 triangles discovered');
ok(cong.title.includes('△DFE')&&!cong.title.includes('△DEF'),`vertex order correctly matched: ${cong?.title}`);
ok(cong.correspondences?.length>=2,'vertex mappings stored');
const scaled=scene([['A',0,0],['B',3,0],['C',0,4],['D',8,1],['E',8,9],['F',14,1]]);
const sim=run(scaled,['similarTriangles']).entries.find(e=>e.title.includes('ABC')&&e.title.includes('D'));
ok(sim?.title.includes('△DFE'),'scaled reflected triangle uses matching vertices');
ok(Math.abs(sim.correspondences.find(x=>x.ids.includes('D')).scale-2)<1e-10,'similarity scale stored');
const almost=scene([['A',0,0],['B',3,0],['C',0,4],['D',8,1],['E',8,9.00001],['F',14,1]]);
ok(!run(almost,['similarTriangles']).count,'near-but-not-similar triangle is rejected');
// Separate dimensional units: position, angle and degree-two residuals.
for(const scale of [1e-6,1,1e6]){
 const s=scene([['A',0,0],['B',4*scale,0],['C',2*scale,0]]);
 ok(run(s,['collinear']).count===1,`exact collinearity scale ${scale}`);
 const d=scene([['A',0,0],['B',4*scale,0],['C',2*scale,1e-6*scale]]);
 ok(run(d,['collinear']).count===0,`near-collinearity rejects at scale ${scale}`);
 const c=makeContext(s);ok(c.eps/scale<1e-7,`point tolerance scales with ${scale}`);
}
const withOutlier=scene([['A',0,0],['B',2,0],['C',1,1e-5],['X',1e12,1e12]]);
ok(run(withOutlier,['collinear']).count===0,'remote outlier does not loosen local-line tolerance');
const exactRight=scene([['A',1,0],['O',0,0],['B',0,1]]);
ok(run(exactRight,['fixedAngle']).entries.some(e=>e.title.includes('90°')),'exact 90° angle retained');
const bentRight=scene([['A',1,0],['O',0,0],['B',1e-6,1]]);
ok(!run(bentRight,['fixedAngle']).entries.some(e=>e.title.includes('90°')),'near 90° is not rounded to 90°');
const sq=scene([['A',0,0],['B',Math.SQRT2,0],['C',0,1],['D',1,1]]);
const sqResult=run(sq,['quadraticRatio']);
ok(sqResult.entries.some(e=>e.metric?.factor===2&&e.quadraticTypes.includes('lengthSquared')),'AB²:CD²=2 represented');
const ratio=sqResult.entries.find(e=>e.metric?.factor===2&&e.quadraticTypes.includes('lengthSquared')&&matches(e,'AB²:CD²=2','二次量连比'));
ok(!!ratio,'squared length ratio matches correctly via search');
ok(matches(ratio,'CD²:AB²=1/2','二次量连比'),'inverse squared length ratio matches');
const powers=scene([['P',3,0],['Q',Math.sqrt(17),0]],[],[['c',0,0,5]]);
const powerFacts=run(powers,['quadraticRatio']).entries;
ok(powerFacts.some(e=>e.metric?.factor===2&&e.metric.high.some(m=>m.type==='power')&&e.metric.low.some(m=>m.type==='power')),'inside-circle negative powers have valid positive ratios');
const negative=scene([['P',3,0],['Q',Math.sqrt(33),0]],[],[['c',0,0,5]]);
ok(run(negative,['quadraticRatio']).entries.some(e=>e.metric?.factor<0&&e.quadraticTypes.includes('power')),'opposite signed point powers retain negative ratios');
const areas=scene([['A',0,0],['B',4,0],['C',0,2],['D',0,1]]);
ok(run(areas,['quadraticRatio']).entries.some(e=>e.metric?.factor===2&&e.quadraticTypes.includes('area')),'areas participate in dimension-two ratios');
const free=scene([['A',0,0],['B',2,0],['C',0,3]],[],[['c',0,0,2]]);
const ext=scene([['A',0,0],['B',2,0],['C',0,3],['F',7.11843,5.29436]],[],[['c',0,0,2]]);
const a=run(free,['quadraticRatio']),b=run(ext,['quadraticRatio']);
ok(a.count===b.count,'moving unrelated free point does not create tautological dot-area ratios');
// Derived fact attribution is based on the actual constructor dependency graph.
const k=new Kernel();for(const cmd of ['point A 0 0','point B 4 0','midpoint M A B'])runText(k,cmd);
const snap=k.snapshot();const pnodes=[],nmeta=[];
for(const node of snap.nodes)if(node.valid){nmeta.push({id:node.id,label:node.label,op:node.op,inputs:node.inputs});if(node.geom?.kind==='point')pnodes.push({id:node.id,label:node.label,x:node.geom.x,y:node.geom.y});}
const constructed={points:pnodes,construction:nmeta};
const base=run(constructed,['distanceRatio','midpoint','collinear']);
ok(base.entries.some(e=>e.kind==='midpoint'&&e.origin==='construction'),'midpoint explicitly traced to construction');
const equal=base.entries.find(e=>e.kind==='distanceRatio'&&e.metric?.factor===1);
ok(equal?.constructionWitnesses?.some(w=>w.op==='midpoint'&&w.reason.includes('相等')),'AM=BM marked as consequence of midpoint construction');
ok(equal.searchTags.includes('定义条件'),'constructor-derived fact searchable by definition');
ok(base.constructionFacts>=2,'direct construction facts counted');
const without=run({points:pnodes},['distanceRatio']);
ok(without.entries.every(e=>e.origin==='discovered'),'same coordinates without dependency have no invented proof');
ok(IMPLICATIONS.square.includes('rhombus')&&IMPLICATIONS.congruentTriangles.includes('equalArea'),'semantic implication DAG exported');
const square=scene([['A',0,0],['B',2,0],['C',2,2],['D',0,2]]);
const quad=run(square,['square']).entries[0];
ok(quad?.impliedKinds?.includes('rhombus')&&quad.searchTags.includes('菱形'),'geometric DAG makes square queryable as rhombus');
// Dense degenerate diagram, exact identities persist after small perturbations.
const grid=scene(Array.from({length:12},(_,i)=>['P'+Math.floor(i/3)+(i%3),Math.floor(i/3)*2,(i%3)*2]),[],[['c',3,2,Math.SQRT2]]);
const before=run(grid);for(const d of [1e-12,1e-11,1e-10]){
 const moved={...grid,points:grid.points.map(p=>p.id==='P11'?{...p,x:p.x+d,y:p.y+d*.3}:p)};
 const after=run(moved);ok(after.count===before.count,`12-vertex grid stable at ${d}`);
 ok(after.entries.every(e=>before.entries.some(q=>q.id===e.id)),`semantic IDs stable at ${d}`);
}
const ui=fs.readFileSync(require('node:path').join(__dirname,'../index.html'),'utf8');
ok(ui.includes('id="radar-perturb"')&&!ui.includes('严格精度 · 双向微扰复验'),'only requested perturbation checkbox added');
console.log(`PASS v3.3: ${n} focused assertions; ${kinds.length} registered relation kinds`);
