const {scan}=require('../assets/relations/detectors.js');
const s=(a,ls=[],cs=[])=>({points:a.map(([id,x,y])=>({id,label:id,x,y})),lines:ls.map(([id,x1,y1,x2,y2])=>({id,label:id,x1,y1,x2,y2})),circles:cs.map(([id,cx,cy,r])=>({id,label:id,cx,cy,r}))});
const tests={
 'circle-on-intersection':s([['A',-3,0],['B',3,0],['K',0,0],['C',0,1],['D',1,0]],[],[['c',0,0,1]]),
 'collinear-tangent':s([['A',0,0],['B',1,0],['C',2,0],['D',3,0],['E',1,2]]),
 'orthocentric':s([['A',0,0],['B',-1,-1],['C',1,-1],['D',0,1]]),
 'symmetric':s([['A',-1,1],['B',1,1],['C',0,0],['D',0,2]], [['l',0,-3,0,3]]),
 'right-interior':s([['A',1,0],['B',0,0],['C',0,1],['D',0.3,0.4]]),
 'empty-free':s([['A',0,0],['B',2,0],['C',0,3]],[],[['c',0,0,2]]),
 'square':s([['A',0,0],['B',2,0],['C',2,2],['D',0,2],['E',1,1]]),
 'chord-shared':s([['A',0,0],['B',1,0],['C',0,1],['D',1,1]],[],[['c',0,0,1]])
};
for(const [name,sc] of Object.entries(tests)){const st=Date.now(),res=scan(sc);console.log('\n'+name,'count',res.count,'unstable',res.unstableRejected,'time',Date.now()-st);for(const [kind,arr] of Object.entries(Object.groupBy(res.entries,e=>e.kind))){console.log(' ',kind,arr.length,arr.slice(0,3).map(e=>e.title).join(' | '));}}
const t=tests['empty-free'],base=scan(t);for(const free of [['F',11.238942,8.291451],['F',0.2,0.1]]){const m=scan({...t,points:[...t.points,{id:free[0],x:free[1],y:free[2]}]}); console.log('free',free,'delta',m.count-base.count,'new',m.entries.filter(e=>!base.entries.some(b=>b.id===e.id)).map(e=>e.kind+':'+e.title).slice(0,50));}
