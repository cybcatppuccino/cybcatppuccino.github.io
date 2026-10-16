/* GeomTool v2.2 — declarative, dynamic relation annotations.
 * Segment witnesses are drawn whether or not the underlying segments exist.
 * Angle arcs use directed BA/BC rays (not arbitrary infinite-line directions).
 */
(function(root,factory){const api=factory();root.GeomRelationOverlay=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const valid=p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y);
function makeResolver(lookup){
 const point=id=>{const q=lookup(id);return q?.kind==='point'&&valid(q)?q:null;};
 const line=ref=>{if(!ref)return null;if(ref.type==='pair'){const a=point(ref.a),b=point(ref.b);return a&&b?{p:a,d:{x:b.x-a.x,y:b.y-a.y},extent:'segment'}:null;}
 const q=lookup(ref.id);return q?.kind==='line'&&valid(q.p)&&valid(q.d)?q:null;};
 const circle=ref=>{if(!ref)return null;if(ref.type==='centerThrough'){const a=point(ref.a),b=point(ref.b);return a&&b?{c:a,r:Math.hypot(a.x-b.x,a.y-b.y)}:null;}
 const q=lookup(ref.id);return q?.kind==='circle'&&valid(q.c)?q:null;};
 return {point,line,circle};
}
function circum(a,b,c){const x=b.x-a.x,y=b.y-a.y,u=c.x-a.x,v=c.y-a.y,d=2*(x*v-y*u);if(Math.abs(d)<1e-12*Math.max(1,Math.hypot(x,y)*Math.hypot(u,v)))return null;const e=x*x+y*y,f=u*u+v*v;const cx=a.x+(e*v-f*y)/d,cy=a.y+(f*x-e*u)/d;return {c:{x:cx,y:cy},r:Math.hypot(cx-a.x,cy-a.y)};}
function draw(ctx,view,entry,lookup,theme='light'){
 if(!entry)return {ok:false,missing:0};
 const resolve=makeResolver(lookup),s=p=>({x:view.ox+p.x*view.scale,y:view.oy-p.y*view.scale});
 const ink=theme==='dark'?'#52d8d0':'#068d9a',light=theme==='dark'?'#a6faf2':'#117282';
 let missing=0,drawn=0;
 ctx.save();ctx.lineCap='round';ctx.lineJoin='round';ctx.strokeStyle=ink;ctx.fillStyle=ink;ctx.lineWidth=2.4;
 function stroke(a,b,dash=[6,5]){const x=s(a),y=s(b);if(Math.hypot(x.x-y.x,x.y-y.y)<0.01)return;ctx.save();ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(x.x,x.y);ctx.lineTo(y.x,y.y);ctx.stroke();ctx.restore();drawn++;}
 function dot(p,r=5){const x=s(p);ctx.save();ctx.setLineDash([]);ctx.beginPath();ctx.arc(x.x,x.y,r,0,2*Math.PI);ctx.fillStyle=theme==='dark'?'#182636':'#fff';ctx.fill();ctx.stroke();ctx.restore();drawn++;}
 function lineDraw(g){const len=Math.hypot(g.d.x,g.d.y);if(len<=1e-10)return;const u={x:g.d.x/len,y:g.d.y/len};let a=g.p,b={x:g.p.x+g.d.x,y:g.p.y+g.d.y};if(g.extent==='line'){a={x:g.p.x-u.x*5000,y:g.p.y-u.y*5000};b={x:g.p.x+u.x*5000,y:g.p.y+u.y*5000};}else if(g.extent==='ray')b={x:g.p.x+u.x*5000,y:g.p.y+u.y*5000};stroke(a,b);}
 function lineAngleDraw(mark){
  const g=resolve.line(mark.a),h=resolve.line(mark.b);if(!g||!h)return false;
  const x=g.d,y=h.d,det=x.x*y.y-x.y*y.x;
  if(Math.abs(det)<=1e-10*Math.hypot(x.x,x.y)*Math.hypot(y.x,y.y))return false;
  const f={x:h.p.x-g.p.x,y:h.p.y-g.p.y},t=(f.x*y.y-f.y*y.x)/det;
  const center={x:g.p.x+t*x.x,y:g.p.y+t*x.y},z=s(center);
  // Do not create a giant arc across the canvas if the intersection is far away.
  if(z.x < -80||z.y < -80||z.x>ctx.canvas.width/(view.dpr||1)+80||z.y>ctx.canvas.height/(view.dpr||1)+80)return true;
  const xlen=Math.hypot(x.x,x.y),ylen=Math.hypot(y.x,y.y);
  let u={x:x.x/xlen,y:x.y/xlen},v={x:y.x/ylen,y:y.y/ylen};
  if(u.x*v.x+u.y*v.y<0)v={x:-v.x,y:-v.y};
  // Pick the acute sector facing the visible endpoints, when available.
  const midpoint=q=>({x:q.p.x+q.d.x*.5,y:q.p.y+q.d.y*.5});
  const p=midpoint(g),q=midpoint(h),toward={x:(p.x+q.x)*.5-center.x,y:(p.y+q.y)*.5-center.y};
  if((u.x+v.x)*toward.x+(u.y+v.y)*toward.y<0){u={x:-u.x,y:-u.y};v={x:-v.x,y:-v.y};}
  const aa=Math.atan2(u.y,u.x),bb=Math.atan2(v.y,v.x);let delta=bb-aa;
  while(delta>Math.PI)delta-=2*Math.PI;while(delta< -Math.PI)delta+=2*Math.PI;
  const r=25;ctx.save();ctx.setLineDash([]);ctx.lineWidth=2.6;ctx.beginPath();ctx.arc(z.x,z.y,r,-aa,-aa-delta,delta>0);ctx.stroke();
  if(mark.label){const m=aa+delta/2,tx=z.x+Math.cos(m)*(r+19),ty=z.y-Math.sin(m)*(r+19);
   ctx.font='600 11px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';const width=ctx.measureText(mark.label).width+10;
   ctx.fillStyle=theme==='dark'?'#1a2e3d':'#fff';ctx.fillRect(tx-width/2,ty-9,width,18);ctx.fillStyle=light;ctx.fillText(mark.label,tx,ty);
  }ctx.restore();drawn++;return true;
 }
 function circleDraw(q){if(!(q.r>0)||!valid(q.c))return;const cc=s(q.c);ctx.save();ctx.setLineDash([7,5]);ctx.beginPath();ctx.arc(cc.x,cc.y,q.r*view.scale,0,2*Math.PI);ctx.stroke();ctx.restore();drawn++;}
 function angleDraw(mark){const a=resolve.point(mark.a),b=resolve.point(mark.b),v=resolve.point(mark.v);if(!a||!b||!v)return false;const angA=Math.atan2(a.y-v.y,a.x-v.x),angB=Math.atan2(b.y-v.y,b.x-v.x);let delta=angB-angA;while(delta>Math.PI)delta-=Math.PI*2;while(delta< -Math.PI)delta+=Math.PI*2;
 if(!Number.isFinite(delta)||Math.abs(delta)<1e-8)return false;
 const sv=s(v),sa=s(a),sb=s(b),ra=Math.hypot(sa.x-sv.x,sa.y-sv.y),rb=Math.hypot(sb.x-sv.x,sb.y-sv.y);
 const r=Math.max(12,Math.min(37,Math.min(ra,rb)*.31));ctx.save();ctx.setLineDash([]);ctx.lineWidth=2.4;ctx.beginPath();ctx.arc(sv.x,sv.y,r,-angA,-angA-delta,delta>0);ctx.stroke();ctx.restore();
 if(mark.label){const mid=angA+delta/2,labelR=r+19;const lx=sv.x+Math.cos(mid)*labelR,ly=sv.y-Math.sin(mid)*labelR;ctx.save();ctx.font='600 11px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';const tx=String(mark.label),w=ctx.measureText(tx).width+10;ctx.fillStyle=theme==='dark'?'#1a2e3d':'#fff';ctx.fillRect(lx-w/2,ly-8,w,16);ctx.fillStyle=light;ctx.fillText(tx,lx,ly);ctx.restore();}
 drawn++;return true;
 }
 for(const m of entry.marks||[]){try{
  if(m.type==='point'){const p=resolve.point(m.id);p?dot(p):missing++;}
  else if(m.type==='segment'){const a=resolve.point(m.a),b=resolve.point(m.b);a&&b?stroke(a,b):missing++;}
  else if(m.type==='line'){const g=resolve.line(m.ref);g?lineDraw(g):missing++;}
  else if(m.type==='circle'){const q=resolve.circle(m.ref);q?circleDraw(q):missing++;}
  else if(m.type==='circumcircle'){const v=m.ids.map(resolve.point);if(v.every(Boolean)){const cc=circum(...v);cc?circleDraw(cc):missing++;}else missing++;}
  else if(m.type==='angle'){if(!angleDraw(m))missing++;}
  else if(m.type==='lineAngle'){lineAngleDraw(m);}
  else if(m.type==='coordinate'){dot(m,4);}
  else if(m.type==='radicalAxis'){
   const a=resolve.circle(m.a),b=resolve.circle(m.b);if(!a||!b){missing++;continue;}
   const dx=b.c.x-a.c.x,dy=b.c.y-a.c.y,len2=dx*dx+dy*dy;
   if(len2<=1e-20){missing++;continue;}
   const powerA=a.c.x*a.c.x+a.c.y*a.c.y-a.r*a.r;
   const powerB=b.c.x*b.c.x+b.c.y*b.c.y-b.r*b.r;
   const t=(powerB-powerA)/(2*len2),pt={x:t*dx,y:t*dy};
   lineDraw({p:pt,d:{x:-dy,y:dx},extent:'line'});
  }
  else if(m.type==='tangentFoot'){const q=resolve.circle(m.circleRef),l=resolve.line(m.lineRef);if(!q||!l){missing++;continue;}const d=l.d,t=((q.c.x-l.p.x)*d.x+(q.c.y-l.p.y)*d.y)/(d.x*d.x+d.y*d.y);const f={x:l.p.x+t*d.x,y:l.p.y+t*d.y};
   // A virtual CD is initially shown as segment CD; extend its supporting
   // line to the tangency point if the contact lies outside that segment.
   if(l.extent==='segment'&&(t<0||t>1))stroke(t<0?l.p:{x:l.p.x+d.x,y:l.p.y+d.y},f,[3,6]);
   stroke(q.c,f,[3,5]);dot(f,4);}
  else if(m.type==='distance'){const p=resolve.point(m.point),l=resolve.line(m.line);if(!p||!l){missing++;continue;}
   const d=l.d,t=((p.x-l.p.x)*d.x+(p.y-l.p.y)*d.y)/(d.x*d.x+d.y*d.y),f={x:l.p.x+t*d.x,y:l.p.y+t*d.y};
   lineDraw(l);stroke(p,f,[3,4]);dot(f,3.3);
  }
 }catch(e){missing++;}}
 ctx.restore();return {ok:drawn>0&&missing===0,missing,drawn};
}
return {draw,makeResolver};
});
