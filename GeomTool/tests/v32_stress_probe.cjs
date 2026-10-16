const {scan}=require('../assets/relations/detectors.js');
const S={points:[],lines:[],circles:[]};
for(let x=0;x<4;x++)for(let y=0;y<3;y++)S.points.push({id:'P'+x+y,label:'P'+x+y,x:x*2,y:y*2});
S.circles.push({id:'c',cx:3,cy:2,r:Math.SQRT2});
const digest=s=>{const t=Date.now(),r=scan(s);console.log('points',s.points.length,'relations',r.count,'unstable',r.unstableRejected,'semantic',r.semanticRemoved,'ms',Date.now()-t,'kinds',Object.entries(Object.groupBy(r.entries,e=>e.kind)).map(([k,v])=>k+':'+v.length).join(' '));return r;};
const r=digest(S);for(const move of [1e-12,1e-11,1e-10,1e-8]){let v={...S,points:S.points.map(p=>p.id==='P11'?{...p,x:p.x+move,y:p.y+move*.3}:p)};let rr=digest(v);const a=new Map(r.entries.map(e=>[e.id,e])),b=new Map(rr.entries.map(e=>[e.id,e]));const removed=[...a.values()].filter(e=>!b.has(e.id)),added=[...b.values()].filter(e=>!a.has(e.id));console.log('move',move,'removed',removed.length,'added',added.length,'samplesRemoved',removed.slice(0,12).map(e=>e.kind+':'+e.title),'new',added.slice(0,5).map(e=>e.title));}
