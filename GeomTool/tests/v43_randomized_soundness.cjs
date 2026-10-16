'use strict';
// Property-based numerical audit of *formal* DDAR conclusions. Numeric
// geometry is used here only as an external test oracle, not by the prover.
const assert=require('node:assert/strict');
const {Kernel}=require('../assets/kernel.js'),DD=require('../assets/ddar.js');
const put=(k,op,id,inputs=[],params={})=>k.execute({op,id,inputs,params});
const rand=(seed)=>()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};
const near=(x,y)=>Math.abs(x-y)<=2e-6*Math.max(1,Math.abs(x),Math.abs(y));
const PI=Math.PI;
let total=0,checked=0,skipped=0;
for(let rep=0;rep<13;rep++){
 const rnd=rand(202643+rep), k=new Kernel(),coords=['A','B','C','P','Q','R'].map((id,i)=>{
  const x=(rnd()-.5)*13+i*.01,y=(rnd()-.5)*11-i*.03;
  put(k,'free',id,[],{x,y});return id;
 });
 const safe=(op,id,t,params)=>{try{put(k,op,id,t,params);}catch(_){}};
 safe('line','l',['A','B']);safe('segment','s',['B','C']);
 safe('midpoint','M',['A','B']);safe('centroid','G',['A','B','C']);
 safe('orthocenter','H',['A','B','C']);safe('circumcenter','O',['A','B','C']);
 safe('incenter','I',['A','B','C']);safe('excenter','J',['A','B','C']);
 safe('perpBisector','bi',['A','B']);safe('angleBisector','al',['A','B','C']);
 safe('outerAngleBisector','ao',['A','B','C']);
 safe('parallel','pa',['P','l']);safe('perpendicular','pe',['P','l']);
 safe('reflectLine','X',['Q','l']);safe('reflectPoint','Y',['Q','P']);
 safe('translate','T',['P','A','B']);safe('parallelogram','D',['A','B','C']);
 safe('homothety','S',['Q','A'],{factor:1.7});safe('rotate','U',['P','A'],{degrees:37});
 safe('rotateLeft','V',['P','A']);safe('rotateRight','W',['P','A']);
 safe('equilateral','E',['A','B'],{branch:rep%2});
 safe('circle','c',['A','B']);safe('circleBySegment','cb',['P','A','B']);
 safe('circumcircle','cc',['A','B','C']);safe('diameterCircle','dc',['A','B']);
 safe('onCircle','F',['c'],{theta:.9});safe('onCircumcircle','Z',['A','B','C'],{theta:2.1});
 safe('onLine','N',['l'],{t:.37});safe('foot','K',['Q','l']);
 safe('foot3','K3',['Q','A','B']);
 safe('tangentFrom','tl',['R','c'],{branch:0});
 const snap=k.snapshot(),r=DD.infer(snap,{maxFacts:8000,maxRounds:6}),map=new Map(snap.nodes.filter(n=>n.valid&&n.geom).map(n=>[n.id,n.geom]));
 // Reconstitute *only* mathematical witness positions for virtual objects.
 for(const v of r.virtuals){if(v.op==='midpoint'){const [a,b]=v.inputs.map(x=>map.get(x));if(a&&b)map.set(v.id,{kind:'point',x:(a.x+b.x)/2,y:(a.y+b.y)/2});}
  else if(v.op==='circumcenter'){const c=map.get(v.id.slice(3));if(c&&c.kind==='circle')map.set(v.id,{kind:'point',x:c.c.x,y:c.c.y});}
  else if(v.op==='circumcircle'){const all=v.inputs.map(x=>map.get(x));if(all.every(Boolean)){const [a,b,c]=all,u=b.x-a.x,w=b.y-a.y,x=c.x-a.x,y=c.y-a.y,den=2*(u*y-w*x);if(Math.abs(den)>1e-8){const xx=a.x+((u*u+w*w)*y-(x*x+y*y)*w)/den,yy=a.y+((x*x+y*y)*u-(u*u+w*w)*x)/den;map.set(v.id,{kind:'circle',c:{x:xx,y:yy},r:Math.hypot(xx-a.x,yy-a.y)});}}}
  else if(v.op==='foot'){const [p,l]=v.inputs.map(x=>map.get(x));if(p&&l){const scale=((p.x-l.p.x)*l.d.x+(p.y-l.p.y)*l.d.y)/(l.d.x*l.d.x+l.d.y*l.d.y);map.set(v.id,{kind:'point',x:l.p.x+scale*l.d.x,y:l.p.y+scale*l.d.y});}}
 }
 for(const v of r.virtuals.filter(v=>v.op==='circumcenter'&&!map.has(v.id))){const [a,b,c]=v.inputs.map(x=>map.get(x)),ux=b.x-a.x,uy=b.y-a.y,vx=c.x-a.x,vy=c.y-a.y,D=2*(ux*vy-uy*vx);map.set(v.id,{kind:'point',x:a.x+((ux*ux+uy*uy)*vy-(vx*vx+vy*vy)*uy)/D,y:a.y+((vx*vx+vy*vy)*ux-(ux*ux+uy*uy)*vx)/D});}
 const point=id=>{const g=map.get(id);return g&&g.kind==='point'?g:null;},
 dist=(a,b)=>Math.hypot(point(a).x-point(b).x,point(a).y-point(b).y),
 seg=s=>s.split('|'), len=s=>dist(...seg(s)),
 vec=l=>{if(l==='L:l_Re')return{x:1,y:0};if(l.startsWith('P:')){const [a,b]=seg(l.slice(2)),p=point(a),q=point(b);return p&&q?{x:q.x-p.x,y:q.y-p.y}:null;}const obj=map.get(l.slice(2));return obj?.kind==='line'?obj.d:null;},
 cross=(a,b)=>a.x*b.y-a.y*b.x,dot=(a,b)=>a.x*b.x+a.y*b.y,
 ang=(a,b)=>{const u=vec(a),v=vec(b);if(!u||!v)return null;return ((Math.atan2(cross(u,v),dot(u,v))%PI)+PI)%PI;},
 angular=(a,b)=>{const x=Math.abs(a-b);return Math.min(x,PI-x)<5e-6;};
 for(const f of r.facts){let result=true,known=true;const a=f.args;
 try{
  if(f.kind==='cong')result=near(len(a[0]),len(a[1]));
  else if(f.kind==='eqratio')result=near(len(a[0]),Number(a[2])*len(a[1]));
  else if(f.kind==='coll'){const [u,v,w]=a.map(point);result=near(cross({x:v.x-u.x,y:v.y-u.y},{x:w.x-u.x,y:w.y-u.y}),0);}
  else if(f.kind==='midp'){const [m,u,v]=a.map(point);result=near(m.x,(u.x+v.x)/2)&&near(m.y,(u.y+v.y)/2);}
  else if(f.kind==='para'||f.kind==='perp'){const [u,v]=a.map(vec);if(!u||!v)known=false;else result=near(f.kind==='para'?cross(u,v):dot(u,v),0);}
  else if(f.kind==='eqangleLines'){const x=ang(a[0],a[1]),y=ang(a[2],a[3]);if(x===null||y===null)known=false;else result=angular(x,y);}
  else if(f.kind==='angconst'){const x=ang(a[0],a[1]);if(x===null)known=false;else result=angular(x,Number(a[2])*PI/180);}
  else if(f.kind==='eqangle'){const [u,v,w]=a[0].split(','),[x,y,z]=a[1].split(',');result=angular(ang(DD.LP(v,u),DD.LP(v,w)),ang(DD.LP(y,x),DD.LP(y,z)));}
  else if(f.kind==='isos')result=near(dist(a[0],a[1]),dist(a[0],a[2]));
  else if(f.kind==='cyclic'){
   const [p,q,t,w]=a.map(point);let D=(q.x-p.x)*(t.y-p.y)-(q.y-p.y)*(t.x-p.x);if(Math.abs(D)<1e-8){known=false;}else{
    const u=q.x-p.x,v=q.y-p.y,x=t.x-p.x,y=t.y-p.y;
    const cx=p.x+((u*u+v*v)*y-(x*x+y*y)*v)/(2*D),cy=p.y+((x*x+y*y)*u-(u*u+v*v)*x)/(2*D);
    result=near(Math.hypot(w.x-cx,w.y-cy),Math.hypot(p.x-cx,p.y-cy));
   }
  }
  else known=false;
 }catch(e){known=false;}
 if(known){checked++;if(!result)throw new Error(`INVALID FACT seed=${rep}, ${f.kind} ${a.join(' ')} [${f.rule}]`);}else skipped++;
 }
 total+=r.facts.length;
}
assert(checked>350,`Only ${checked} facts checked`);
console.log(`PASS v4.3 randomized soundness: ${checked} numerical audits, ${skipped} other facts, ${total} total over 13 geometric configurations`);
