/* GeomTool v8 — display-only source diagram annotations.
 * Only reads existing Kernel coordinates. No _create, execute, serialize, or
 * proof/compiler calls: these strokes never become geometric assumptions.
 */
(function(root,factory){const api=factory();root.GeomLibraryVisuals=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
function compile(entry){
 const marks=[],seen=new Set(),add=(kind,args)=>{if(!args.every(x=>typeof x==='string'&&x))return;const key=kind+':'+args.join('|');if(!seen.has(key)){seen.add(key);marks.push({kind,args});}};
 const seg=(a,b)=>{if(a!==b)add('segment',[a,b]);};
 const line=(a,b)=>{if(a!==b)add('line',[a,b]);};
 const circ=(o,p)=>{if(o!==p)add('circle',[o,p]);};
 const circum=(a,b,c)=>add('circumcircle',[a,b,c]);
 const triangle=(a,b,c)=>{if(a!==b&&b!==c&&a!==c)add('polygon',[a,b,c]);};
 const poly=ids=>{if(ids.length>=3)add('polygon',ids);else if(ids.length===2)seg(...ids);};
 const normal=(op,a,outs)=>{
  const o=outs[0];
  switch(op){
   case 'triangle':case 'acute_triangle':case 'ieq_triangle':case 'triangle12':case 'r_triangle':case 'risos':case 'iso_triangle':poly(outs);break;
   case 'quadrangle':case 'pentagon':case 'rectangle':case 'square':case 'isquare':case 'eq_trapezoid':case 'trapezoid':poly(outs);break;
   case 'segment':case 'between':case 'between_bound':if(a.length===2)seg(...a);else if(outs.length===3)seg(outs[0],outs[2]);break;
   case 'midpoint':case 'on_bline':case 'iso_triangle_vertex':if(a.length>=2){seg(a[0],a[1]);if(op==='on_bline')add('perpBisector',a.slice(0,2));}break;
   case 'foot':if(a.length>=3){line(a[1],a[2]);seg(a[0],o);add('right',[o,a[0],a[1]]);}break;
   case 'on_tline':if(a.length===3){line(a[1],a[2]);add('perpendicular',[a[0],a[1],a[2]]);if(a[0]!==o)seg(a[0],o);}break;
   case 'on_pline':case 'on_pline0':if(a.length===3){seg(a[1],a[2]);add('parallel',[a[0],a[1],a[2]]);}break;
   case 'intersection_ll':if(a.length===4){line(a[0],a[1]);line(a[2],a[3]);}break;
   case 'intersection_lc':if(a.length===3){line(a[0],a[2]);circ(a[1],a[2]);}break;
   case 'intersection_cc':if(a.length===3){circ(a[0],a[2]);circ(a[1],a[2]);}break;
   case 'intersection_lp':case 'intersection_lt':if(a.length===5){line(a[0],a[1]);add(op==='intersection_lp'?'parallel':'perpendicular',[a[2],a[3],a[4]]);}break;
   case 'intersection_pp':case 'intersection_tt':if(a.length===6){add(op==='intersection_pp'?'parallel':'perpendicular',a.slice(0,3));add(op==='intersection_pp'?'parallel':'perpendicular',a.slice(3,6));}break;
   case 'circumcenter':case 'circle':if(a.length===3){triangle(...a);circum(...a);for(const x of a)seg(o,x);}break;
   case 'on_circum':if(a.length===3){triangle(...a);circum(...a);}break;
   case 'ninepoints':if(a.length===3){triangle(...a);circum(...outs.slice(0,3));}break;
   case 'incenter':case 'excenter':if(a.length===3){triangle(...a);for(const x of a)seg(o,x);add('angle',[a[0],a[1],a[2]]);}break;
   case 'incenter2':case 'excenter2':if(a.length===3){triangle(...a);for(const x of a)seg(outs[3],x);if(outs.length===4)for(let j=0;j<3;j++)seg(outs[j],outs[3]);}break;
   case 'centroid':if(a.length===3){triangle(...a);for(const x of a)seg(o,x);}break;
   case 'orthocenter':if(a.length===3){triangle(...a);for(const x of a)seg(x,o);}break;
   case 'angle_bisector':if(a.length===3){seg(a[1],a[0]);seg(a[1],a[2]);if(o)seg(a[1],o);add('angle',a);}break;
   case 'angle_mirror':if(a.length===3){seg(a[1],a[0]);seg(a[1],a[2]);if(o)seg(a[1],o);add('angle',a);}break;
   case 'on_line':if(a.length===2)line(...a);break;
   case 'on_circle':if(a.length===2)circ(...a);break;
   case 'on_dia':if(a.length===2){seg(...a);add('diameter',a);}break;
   case 'eqdistance':if(a.length===3){seg(a[1],a[2]);add('circleSegment',a);}break;
   case 'on_aline':if(a.length===5){line(a[0],a[1]);line(a[2],a[3]);seg(a[4],o);add('angle',[a[0],a[1],a[2]]);}break;
   case 'on_aline0':if(a.length===7){line(a[0],a[1]);line(a[2],a[3]);line(a[4],a[5]);seg(a[6],o);}break;
   case 's_angle':if(a.length>=3){seg(a[0],a[1]);seg(a[1],o);}break;
   case 'eq_triangle':if(a.length>=2){triangle(a[0],a[1],o);}break;
   case 'psquare':case 'nsquare':if(a.length===2){triangle(a[0],a[1],o);}break;
   case 'parallelogram':if(a.length>=3)poly([...a.slice(0,3),o]);break;
   case 'shift':if(a.length===3){seg(a[1],a[2]);seg(a[0],o);}break;
   case 'trisegment':if(a.length===2){seg(...a);for(const x of outs)seg(a[0],x);}break;
   case 'lconst':case 'l2const':case 'rconst':if(a.length>=2&&o)add('segment',[a[0],o]);break;
   case 'cc_tangent':case 'cc_itangent':if(a.length===4){circ(a[0],a[1]);circ(a[2],a[3]);for(let j=0;j+1<outs.length;j+=2)seg(outs[j],outs[j+1]);}break;
   default:break;
  }
 };
 if(entry?.kind==='coordinates'){
  // Coordinate-only predicates are original assumptions, not unique constructible recipes.
  // For readability they can indicate their witnesses without creating nodes.
  for(const raw of entry.predicates||[]){const t=String(raw).trim().split(/\s+/),op=t.shift(),a=t;
   if(['coll','collinear'].includes(op)&&a.length>=3)line(a[0],a[1]);
   if(['cyclic','concyclic'].includes(op)&&a.length>=4)circum(...a.slice(0,3));
   if(op==='midp'&&a.length>=3)seg(a[1],a[2]);
   if(['para','perp','cong'].includes(op)&&a.length>=4){seg(a[0],a[1]);seg(a[2],a[3]);}
  }
 }else for(const s of entry?.steps||[]){
  for(const op of s.operations||[])normal(op.op,op.args||[],s.outputs||[]);
 }
 return marks.slice(0,1600);
}
function draw(ctx,view,marks,lookup,theme='light'){
 if(!marks?.length)return;
 const p=id=>{const g=lookup(id);return g?.kind==='point'&&Number.isFinite(g.x)&&Number.isFinite(g.y)?g:null;};
 const screen=g=>({x:view.ox+g.x*view.scale,y:view.oy-g.y*view.scale});
 const segment=(a,b)=>{if(!a||!b)return;const u=screen(a),v=screen(b);ctx.beginPath();ctx.moveTo(u.x,u.y);ctx.lineTo(v.x,v.y);ctx.stroke();};
 const line=(a,b)=>{if(!a||!b||Math.hypot(b.x-a.x,b.y-a.y)<1e-9)return;const u=screen(a),v=screen(b),len=Math.hypot(v.x-u.x,v.y-u.y),dx=(v.x-u.x)/len,dy=(v.y-u.y)/len;
  ctx.beginPath();ctx.moveTo(u.x-dx*4000,u.y-dy*4000);ctx.lineTo(u.x+dx*4000,u.y+dy*4000);ctx.stroke();};
 const circle=(center,r)=>{if(!center||!(r>1e-9)||!Number.isFinite(r))return;const v=screen(center);ctx.beginPath();ctx.arc(v.x,v.y,r*view.scale,0,Math.PI*2);ctx.stroke();};
 const circum=(a,b,c)=>{if(!a||!b||!c)return;const u=b.x-a.x,v=b.y-a.y,w=c.x-a.x,z=c.y-a.y,det=2*(u*z-v*w);if(Math.abs(det)<1e-10)return;const A=u*u+v*v,B=w*w+z*z,center={x:a.x+(A*z-B*v)/det,y:a.y+(B*u-A*w)/det};circle(center,Math.hypot(a.x-center.x,a.y-center.y));};
 ctx.save();ctx.lineCap='round';ctx.lineWidth=1.25;ctx.strokeStyle=theme==='dark'?'#60788f':'#a3b4c6';ctx.setLineDash([3,5]);
 for(const m of marks){const a=m.args.map(p);try{switch(m.kind){
  case 'segment':segment(a[0],a[1]);break;
  case 'line':line(a[0],a[1]);break;
  case 'circle':if(a[0]&&a[1])circle(a[0],Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y));break;
  case 'circleSegment':if(a.every(Boolean))circle(a[0],Math.hypot(a[1].x-a[2].x,a[1].y-a[2].y));break;
  case 'circumcircle':circum(...a);break;
  case 'diameter':if(a.every(Boolean)){const center={x:(a[0].x+a[1].x)/2,y:(a[0].y+a[1].y)/2};circle(center,Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y)/2);}break;
  case 'polygon':if(a.length>=3&&a.every(Boolean)){ctx.save();ctx.setLineDash([]);ctx.beginPath();a.forEach((x,i)=>{const s=screen(x);if(i)ctx.lineTo(s.x,s.y);else ctx.moveTo(s.x,s.y);});ctx.closePath();ctx.fillStyle=theme==='dark'?'#70a9cd0b':'#5d98bb0b';ctx.fill();ctx.stroke();ctx.restore();}break;
  case 'perpBisector':if(a.every(Boolean)){const mid={x:(a[0].x+a[1].x)/2,y:(a[0].y+a[1].y)/2},d={x:a[0].y-a[1].y,y:a[1].x-a[0].x};line(mid,{x:mid.x+d.x,y:mid.y+d.y});}break;
  case 'parallel':case 'perpendicular':if(a.every(Boolean)){const v={x:a[2].x-a[1].x,y:a[2].y-a[1].y};const d=m.kind==='parallel'?v:{x:-v.y,y:v.x};line(a[0],{x:a[0].x+d.x,y:a[0].y+d.y});}break;
  case 'angle':if(a.length>=3&&a.every(Boolean)){const o=screen(a[1]),u=screen(a[0]),v=screen(a[2]);const x=Math.atan2(u.y-o.y,u.x-o.x),y=Math.atan2(v.y-o.y,v.x-o.x);let delta=y-x;while(delta>Math.PI)delta-=2*Math.PI;while(delta< -Math.PI)delta+=2*Math.PI;ctx.beginPath();ctx.arc(o.x,o.y,20,x,x+delta,delta<0);ctx.stroke();}break;
  case 'right':if(a[0]&&a[1]&&a[2]){const q=screen(a[0]),x=screen(a[1]),y=screen(a[2]);const ux=x.x-q.x,uy=x.y-q.y,vx=y.x-q.x,vy=y.y-q.y,L=Math.hypot(ux,uy),M=Math.hypot(vx,vy);if(L>8&&M>8){const s=10,xx=ux/L*s,xy=uy/L*s,yx=vx/M*s,yy=vy/M*s;ctx.beginPath();ctx.moveTo(q.x+xx,q.y+xy);ctx.lineTo(q.x+xx+yx,q.y+xy+yy);ctx.lineTo(q.x+yx,q.y+yy);ctx.stroke();}}break;
  default:break;
 }}catch(_){/* incomplete/skipped source construction: display what is defined */}}
 ctx.restore();
}
return {compile,draw};
});
