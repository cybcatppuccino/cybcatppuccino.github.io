/* GeomTool v8.2 — deterministic, constraint-preserving numeric layout refinement.
 * Each saved keyframe is an ACCEPTED descent step. The slider travels along these
 * keyframes by cumulative path length with smootherstep easing (not a straight
 * interpolation from the starting drawing to the end). No network dependencies.
 */
(function(root,factory){const api=factory(root.GeomKernel||(typeof require==='function'?require('./kernel.js'):null));if(typeof module==='object'&&module.exports)module.exports=api;root.GeomPretty=api;})(typeof globalThis!=='undefined'?globalThis:this,function(lib){
'use strict';
const movable=new Set(['free','real','onLine','onCircle']);
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
function controls(kernel){
 const all=[];
 for(const n of kernel.list()){
  if(!n.valid||!movable.has(n.op)||n.style.system||n.style.visible===false)continue;
  const geom=n.geom;let scale=1;
  if(n.op==='onLine')scale=Math.max(.01,Math.hypot(kernel.geom(n.inputs[0]).d.x,kernel.geom(n.inputs[0]).d.y));
  if(n.op==='onCircle')scale=Math.max(.01,kernel.geom(n.inputs[0]).r);
  for(const field of n.op==='free'?['x','y']:n.op==='real'?['x']:n.op==='onLine'?['t']:['theta'])all.push({id:n.id,field,scale});
 }
 return all;
}
function optimize(kernel, options={}){
 if(options.constraints?.length)return optimizeConstrained(kernel,options);
 const strength=clamp(Number(options.strength)||65,5,100);
 const maxSteps=clamp(Number(options.iterations)||55,1,120);
 const trial=new lib.Kernel(kernel.serialize());
 const dims=controls(trial), initial=dims.map(d=>trial.node(d.id).params[d.field]*d.scale);
 if(!dims.length)return {frames:[[]],dims,energyStart:0,energyEnd:0,steps:0,strength};
 // Include dependent points, not just draggable handles: constraints must hold.
 const validIds=trial.list().filter(n=>n.valid).map(n=>n.id);
 const initialPoints=trial.list().filter(n=>n.valid&&n.geom?.kind==='point'&&!n.style.system).map(n=>({x:n.geom.x,y:n.geom.y}));
 const anchor=initialPoints.length?{
  x:initialPoints.reduce((s,p)=>s+p.x,0)/initialPoints.length,
  y:initialPoints.reduce((s,p)=>s+p.y,0)/initialPoints.length
 }:{x:0,y:0};
 const scope=Math.max(1,...initialPoints.map(p=>Math.hypot(p.x-anchor.x,p.y-anchor.y)));
 const desired=0.75+strength*.009;
 function set(u){for(let i=0;i<dims.length;i++){const d=dims[i];trial.node(d.id).params[d.field]=u[i]/d.scale;}trial.rebuild();}
 function energy(u){
  set(u);
  if(validIds.some(id=>!trial.node(id).valid))return 1e15;
  const points=trial.list().filter(n=>n.valid&&n.geom?.kind==='point'&&!n.style.system).map(n=>n.geom);
  if(!points.length)return 0;
  let e=0,mx=0,my=0;
  for(const p of points){mx+=p.x;my+=p.y;}
  mx/=points.length;my/=points.length;
  let vx=0,vy=0;
  for(const p of points){vx+=(p.x-mx)**2;vy+=(p.y-my)**2;}
  vx/=points.length;vy/=points.length;
  const ratio=Math.log((vx+.26)/(vy+.26));
  e+=(.32+strength*.008)*ratio*ratio;
  // Nonlinear short-distance repulsion, plus gentle distance compression.
  for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){
   const dx=points[i].x-points[j].x,dy=points[i].y-points[j].y;
   const d=Math.hypot(dx,dy),near=Math.max(0,desired-d),far=Math.max(0,d-7.5);
   e+=3.2*near*near+.006*far*far;
  }
  // Discourage very flat polygons and three-point triangles.
  for(const n of trial.list()){
   if(!n.valid||n.style.system)continue;
   if(n.geom?.kind==='polygon'){
    const ps=n.geom.points,perimeter=ps.reduce((s,p,i)=>s+Math.hypot(p.x-ps[(i+1)%ps.length].x,p.y-ps[(i+1)%ps.length].y),0);
    const twice=Math.abs(ps.reduce((s,p,i)=>s+p.x*ps[(i+1)%ps.length].y-p.y*ps[(i+1)%ps.length].x,0));
    const quality=twice/(perimeter*perimeter+1e-6);
    e+=Math.max(0,.045-quality)**2*230;
   }
  }
  e+=.025*((mx-anchor.x)**2+(my-anchor.y)**2);
  for(let k=0;k<u.length;k++)e+=(0.028+(.012*(100-strength)/100))*((u[k]-initial[k])/Math.max(1,scope*.5))**2;
  return Number.isFinite(e)?e:1e18;
 }
 let u=[...initial],value=energy(u),start=value;
 const frames=[[...initial]],energies=[value];
 for(let step=0;step<maxSteps;step++){
  const grad=[];
  for(let k=0;k<u.length;k++){
   const h=.004*Math.max(1,scope*.12),plus=[...u],minus=[...u];plus[k]+=h;minus[k]-=h;
   grad.push((energy(plus)-energy(minus))/(2*h));
  }
  set(u);
  const maxGrad=Math.max(...grad.map(Math.abs));if(!Number.isFinite(maxGrad)||maxGrad<1e-5)break;
  const base=Math.min(.34,.62/maxGrad),maxDistance=.35;
  let accepted=null,score=value;
  for(let t=0;t<9;t++){
   const alpha=base*Math.pow(.5,t),candidate=u.map((v,i)=>v-clamp(alpha*grad[i],-maxDistance,maxDistance));
   const test=energy(candidate);
   if(test<value-1e-8){accepted=candidate;score=test;break;}
  }
  if(!accepted)break;
  u=accepted;value=score;frames.push([...u]);energies.push(value);
 }
 set(u);
 return {frames,energies,dims,initial:[...initial],energyStart:start,energyEnd:value,steps:frames.length-1,strength};
}
// v5.3: feasibility-first constrained optimization. In the constraint stage,
// damped Gauss-Newton finds a nearby realization, with deterministic multiple
// starting points when a singular initial sketch prevents convergence. After
// feasibility, guarded beauty descent NEVER accepts a step violating the goals.
function optimizeConstrained(kernel,options={}){
 const Cond=(typeof globalThis!=='undefined'&&globalThis.GeomPrettyConstraints)||(typeof require==='function'?require('./pretty-constraints.js'):null);
 if(!Cond)throw Error('条件求解模块未加载');
 const cs=options.constraints, strength=clamp(Number(options.strength)||65,5,100);
 const trial=new lib.Kernel(kernel.serialize()),dims=controls(trial);
 const initial=dims.map(d=>trial.node(d.id).params[d.field]*d.scale),orig=initial.slice();
 const validIds=trial.list().filter(n=>n.valid).map(n=>n.id);
 const originalPs=trial.list().filter(n=>n.valid&&n.geom?.kind==='point'&&!n.style?.system).map(n=>n.geom);
 const scope=Math.max(1,...originalPs.map(p=>Math.hypot(p.x,p.y)));
 // Numerical target is deliberately much stricter than the relation detector's
 // 1e-9-scale tolerances. ~1e-15 is reachable only in well-conditioned scenes.
 const tol=Number.isFinite(options.tolerance)&&options.tolerance>0?options.tolerance:5e-13;
 const wantedSteps=clamp(Number(options.iterations)||130,20,240);
 const frames=[initial.slice()],energies=[];
 function set(u){for(let i=0;i<dims.length;i++){const d=dims[i];trial.node(d.id).params[d.field]=u[i]/d.scale;}trial.rebuild();}
 function info(u){set(u);if(validIds.some(id=>!trial.node(id).valid))return {loss:1e16,max:1e8,satisfied:0,res:Array(residualCount).fill(1e8)};
  const stats=Cond.assess(cs,trial,tol),res=stats.flatMap(x=>x.residuals.map(v=>Number.isFinite(v)?clamp(v,-1e8,1e8):1e8));
  return {res,loss:res.reduce((s,r)=>s+r*r,0),max:stats.reduce((x,a)=>Math.max(x,a.error),0),satisfied:stats.filter(x=>x.satisfied).length,stats};
 }
 const residualCount=cs.flatMap(c=>Cond.residuals(c,trial)).length;
 function prettyCost(u){set(u);if(validIds.some(id=>!trial.node(id).valid))return 1e12;
  const ps=trial.list().filter(n=>n.valid&&n.geom?.kind==='point'&&!n.style.system).map(n=>n.geom);
  let e=0,mx=0,my=0,vx=0,vy=0;
  for(const p of ps){mx+=p.x;my+=p.y;}mx/=ps.length||1;my/=ps.length||1;
  for(const p of ps){vx+=(p.x-mx)**2;vy+=(p.y-my)**2;}vx/=ps.length||1;vy/=ps.length||1;
  e+=(.32+strength*.008)*Math.log((vx+.26)/(vy+.26))**2;
  const desired=.75+strength*.009;
  for(let i=0;i<ps.length;i++)for(let j=i+1;j<ps.length;j++){
   const d=Math.hypot(ps[i].x-ps[j].x,ps[i].y-ps[j].y);
   e+=3.2*Math.max(0,desired-d)**2+.006*Math.max(0,d-7.5)**2;
  }
  e+=.025*(mx*mx+my*my);
  e+=u.reduce((v,x,i)=>v+.04*((x-orig[i])/Math.max(1,scope*.5))**2,0);
  return Number.isFinite(e)?e:1e12;
 }
 function solve(a,b){ // dense positive-definite system with pivoting
  const n=b.length,mat=a.map((r,i)=>[...r,b[i]]);
  for(let k=0;k<n;k++){
   let pivot=k;for(let j=k+1;j<n;j++)if(Math.abs(mat[j][k])>Math.abs(mat[pivot][k]))pivot=j;
   if(Math.abs(mat[pivot][k])<1e-18)return null;
   [mat[k],mat[pivot]]=[mat[pivot],mat[k]];
   const div=mat[k][k];for(let j=k;j<=n;j++)mat[k][j]/=div;
   for(let i=0;i<n;i++)if(i!==k){const m=mat[i][k];for(let j=k;j<=n;j++)mat[i][j]-=m*mat[k][j];}
  }return mat.map(r=>r[n]);
 }
 function jacobian(u,base,accurate=false){
  const n=u.length,m=base.res.length,J=Array.from({length:m},()=>Array(n).fill(0));
  for(let j=0;j<n;j++){
   const h=(accurate?7e-5:1.3e-5)*Math.max(1,Math.abs(u[j])*.05),p=u.slice(),q=u.slice();p[j]+=h;q[j]-=h;
   const plus=info(p).res,minus=info(q).res;
   let p2=null,q2=null;
   if(accurate){p[j]=u[j]+2*h;q[j]=u[j]-2*h;p2=info(p).res;q2=info(q).res;}
   for(let row=0;row<m;row++){
    const diff=accurate?(-p2[row]+8*plus[row]-8*minus[row]+q2[row])/(12*h):(plus[row]-minus[row])/(2*h);
    J[row][j]=clamp(Number.isFinite(diff)?diff:0,-1e8,1e8);
   }
  }
  set(u);return J;
 }
 let u=initial.slice(),state=info(u),best={u:u.slice(),state},lambda=.003;
 energies.push(state.loss);
 // Smooth feasibility loss drives intermediate steps. "Best" is chosen by
 // number of satisfied conditions first, then numerical violation.
 const choose=(candidate,datum)=>datum.satisfied>candidate.state.satisfied||
  (datum.satisfied===candidate.state.satisfied&&(datum.max<candidate.state.max-1e-14||
   (Math.abs(datum.max-candidate.state.max)<=1e-14&&datum.loss<candidate.state.loss-1e-29)));
 const accept=(next,st)=>{u=next;state=st;frames.push(u.slice());energies.push(state.loss);if(choose(best,st))best={u:u.slice(),state:st};};
 const maxRestarts=options.restarts===0?0:4;
 let rounds=0;
 for(let restart=0;restart<=maxRestarts&&dims.length;restart++){
  if(restart){
   // Virtual-only deterministic restart (NOT applied to user's canvas).
   // Deterministic local perturbation: improves recovery from zero Jacobians
   // in tangency, cyclicity and initially coincident-point configurations.
   const size=[0,.12,.33,.72,1.25][restart];u=best.u.map((v,j)=>v+size*Math.sin((j+1)*2.39996+restart*1.379));state=info(u);lambda=.04;
  }
  let stalled=0;
  for(let step=0;step<wantedSteps;step++){
   rounds++;
   if(state.max<=tol){if(choose(best,state))best={u:u.slice(),state};break;}
   const J=jacobian(u,state,state.max<1e-7),n=u.length,m=state.res.length;
   const h=Array.from({length:n},()=>Array(n).fill(0)),g=Array(n).fill(0);
   for(let i=0;i<n;i++)for(let r=0;r<m;r++){
    const x=J[r][i];g[i]-=x*state.res[r];
    for(let j=0;j<=i;j++)h[i][j]+=x*J[r][j];
   }
   for(let i=0;i<n;i++)for(let j=0;j<i;j++)h[j][i]=h[i][j];
   let winner=null;
   for(let attempt=0;attempt<15;attempt++){
    const damping=lambda*Math.pow(4,attempt);
    const matrix=h.map((r,i)=>r.map((v,j)=>v+(i===j?damping:0)));
    const stepVec=solve(matrix,g);if(!stepVec)continue;
    const length=Math.hypot(...stepVec),cap=Math.max(.2,Math.min(1.1,scope*.13));
    const factor=Math.min(1,cap/(length||1));const next=u.map((v,j)=>v+stepVec[j]*factor);
    const st=info(next);
    if(st.loss<state.loss*(1-1e-9)-1e-31){winner={next,st};lambda=Math.max(1e-16,damping*.2);break;}
   }
   if(!winner){lambda*=8;if(++stalled>=5)break;continue;}
   stalled=0;accept(winner.next,winner.st);
  }
  if(best.state.satisfied===cs.length)break;
 }
 // Final feasibility polish: small, minimum-norm Newton corrections using
 // an accurate symmetric Jacobian, line-search and machine-scale regularizer.
 // The polish never accepts a worse residual or modifies the live drawing.
 if(dims.length&&best.state.max<1e-5){
  u=best.u.slice();state=info(u);
  for(let iteration=0;iteration<35;iteration++){
   if(state.max<=Math.max(1e-15,Math.min(tol*.002,5e-15)))break;
   const J=jacobian(u,state,true),n=u.length,m=state.res.length;
   const normal=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>
    J.reduce((v,r)=>v+r[i]*r[j],0)));
   const gradient=Array.from({length:n},(_,i)=>-J.reduce((v,r,j)=>v+r[i]*state.res[j],0));
   let chosen=null;
   for(const regularizer of [1e-14,1e-12,1e-10,1e-8,1e-6]){
    const step=solve(normal.map((r,i)=>r.map((v,j)=>v+(i===j?regularizer:0))),gradient);
    if(!step||step.some(x=>!Number.isFinite(x)))continue;
    for(let alpha=1;alpha>=1/128;alpha/=2){
     const next=u.map((x,i)=>x+alpha*step[i]),st=info(next);
     if(st.max<=state.max+2e-16&&st.loss<state.loss*(1-1e-11)-1e-32){chosen={next,st};break;}
    }
    if(chosen)break;
   }
   if(!chosen)break;
   accept(chosen.next,chosen.st);
  }
 }
 // Restore best count first (for inconsistent systems), then best violation.
 // If not fully feasible, keep the most satisfied configuration and never
 // sacrifice constraints for mere visual quality.
 u=best.u.slice();state=info(u);
 if(Math.hypot(...u.map((v,j)=>v-frames[frames.length-1][j]))>1e-12){frames.push(u.slice());energies.push(state.loss);}
 const feasibility={satisfied:state.satisfied,total:cs.length,maxError:state.max,loss:state.loss,details:state.stats||Cond.assess(cs,trial,tol)};
 const prettyBefore=prettyCost(u);
 if(state.satisfied===cs.length&&dims.length){
  let appearance=prettyBefore;
  const beautyRounds=clamp(Math.round(wantedSteps*.55),8,80);
  for(let step=0;step<beautyRounds;step++){
   const grad=[];
   for(let i=0;i<u.length;i++){
    const h=.002*Math.max(1,scope*.1),p=u.slice(),q=u.slice();p[i]+=h;q[i]-=h;
    grad.push((prettyCost(p)-prettyCost(q))/(2*h));
   }
   const n=u.length,J=jacobian(u,state),m=state.res.length;
   // Damped null-space projection (J'J + eps I)^-1 eps grad. This
   // keeps the visual adjustment tangential to the active constraints.
   const h=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>
    J.reduce((sum,r)=>sum+r[i]*r[j],0)+(i===j?1e-5:0)));
   const direction=solve(h,grad.map(g=>g*1e-5))||grad;
   const maxGrad=Math.max(...direction.map(Math.abs));if(!Number.isFinite(maxGrad)||maxGrad<1e-8)break;
   let winner=null;
   for(let t=0;t<12;t++){
    const alpha=Math.min(.18,.16/maxGrad)*Math.pow(.5,t);
    let next=u.map((v,i)=>v-clamp(alpha*direction[i],-.2,.2));
    // Correct normal drift with min-norm Gauss-Newton steps before accepting.
    let st=info(next);
    for(let correction=0;correction<12&&st.max>Math.min(tol*.2,2e-14);correction++){
     const jc=jacobian(next,st,st.max<1e-7),a=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>
      jc.reduce((sum,r)=>sum+r[i]*r[j],0)+(i===j?1e-13:0)));
     const rhs=Array.from({length:n},(_,i)=>-jc.reduce((sum,r,z)=>sum+r[i]*st.res[z],0));
     const delta=solve(a,rhs);if(!delta)break;
     next=next.map((x,i)=>x+clamp(delta[i],-.25,.25));st=info(next);
    }
    if(st.satisfied!==cs.length||st.max>tol)continue;
    const newLook=prettyCost(next);
    if(newLook<appearance-1e-8){winner={next,st,newLook};break;}
   }
   if(!winner)break;
   accept(winner.next,winner.st);appearance=winner.newLook;
  }
 }
 set(u);
 return {frames,energies,dims,initial,steps:frames.length-1,strength,
  energyStart:energies[0],energyEnd:state.loss,feasibility:{...feasibility,satisfied:state.satisfied,maxError:state.max,loss:state.loss,details:state.stats||Cond.assess(cs,trial,tol),tolerance:tol},
  beautyStart:prettyBefore,beautyEnd:prettyCost(u),rounds,constrained:true};
}
function at(plan,progress){
 if(!plan?.dims?.length)return [];
 const frames=plan.frames;if(frames.length===1)return [...frames[0]];
 const t=clamp(progress/100,0,1),ease=t*t*t*(t*(t*6-15)+10);
 const cumul=[0];for(let i=1;i<frames.length;i++){
  const d=Math.hypot(...frames[i].map((v,j)=>v-frames[i-1][j]));cumul.push(cumul[i-1]+d);
 }
 const target=ease*cumul[cumul.length-1];let k=1;while(k<cumul.length-1&&cumul[k]<target)k++;
 const f=(target-cumul[k-1])/Math.max(1e-12,cumul[k]-cumul[k-1]);
 return frames[k-1].map((v,j)=>v+(frames[k][j]-v)*f);
}
function apply(kernel,plan,progress){
 if(!plan?.dims?.length)return false;
 const u=at(plan,progress);
 for(let i=0;i<plan.dims.length;i++){
  const d=plan.dims[i],node=kernel.node(d.id);
  if(!node)return false;
  node.params[d.field]=u[i]/d.scale;
 }
 kernel.rebuild();kernel._emit('preview');return true;
}
return {optimize,apply,at,controls};
});
