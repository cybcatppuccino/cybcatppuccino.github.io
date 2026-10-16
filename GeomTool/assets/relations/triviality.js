/* v3.2: geometry-first intrinsic / degenerate configuration rejection.
 * These checks run during candidate generation, before the expensive
 * three-pass numerical stability confirmation. No dependence on chosen UI kinds.
 */
(function(root,factory){const api=factory();root.GeomRelationTriviality=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const contactGuard=(ctx,r)=>Math.max(ctx.eps*8,Math.sqrt(2*Math.max(r,ctx.eps)*ctx.eps)*1.1);
const locate=(ctx,x,ys)=>ctx.P.some(p=>dist(p,x)<=ctx.eps&&(ys||[]).every(c=>Math.abs(dist(p,{x:c.x,y:c.y})-c.r)<=ctx.eps));
const anchoredCirclePoint=(ctx,c,p)=>{
 if(c.kind!=='implicit')return false;
 const candidates=[{a:c.a,b:c.b},...(c.aliases||[]),...(c.virtualAliases||[])];
 return candidates.some(w=>{const a=ctx.pMap.get(w.a),b=ctx.pMap.get(w.b);return a&&b&&(dist(p,a)<=contactGuard(ctx,c.r)||dist(p,b)<=contactGuard(ctx,c.r));});
};
function circleContact(a,b,external){
 const dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);if(!d)return null;
 // External contact lies along c1 -> c2. For internal contact, the
 // smaller circle is internally tangent on the near/far side as needed.
 const x=external?a.r:((a.r>=b.r)?a.r:-a.r);
 return {x:a.x+x*dx/d,y:a.y+x*dy/d};
}
function intrinsicCircleTangent(ctx,a,b,external){
 if(a.kind!=='implicit'||b.kind!=='implicit')return false;
 const anchored=x=>[x.a,x.b,...(x.aliases||[]).flatMap(v=>[v.a,v.b]),...(x.virtualAliases||[]).flatMap(v=>[v.a,v.b])];
 const aa=new Set(anchored(a)),bb=new Set(anchored(b));
 if([...aa].some(id=>bb.has(id)))return true; // shared defining data
 const contact=circleContact(a,b,external);
 return contact&&(anchoredCirclePoint(ctx,a,contact)||anchoredCirclePoint(ctx,b,contact));
}
function intrinsicLineTangent(ctx,q,line,foot){
 // The radius endpoint was used to DEFINE a virtual circle, and the
 // tangent support passes through that endpoint. Its tangency says no more
 // than the existing perpendicular-radius fact.
 if(q.kind!=='implicit')return false;
 const aliases=[{a:q.a,b:q.b},...(q.aliases||[]),...(q.virtualAliases||[])];
 return aliases.some(({a,b})=>{
  const p=ctx.pMap.get(b);return p&&dist(p,foot)<=contactGuard(ctx,q.r)&&ctx.distPointLine(p,line)<=contactGuard(ctx,q.r);
 });
}
function isNamedIncidence(ctx,at,circles){
 const guard=ctx.eps*64;return ctx.P.some(p=>dist(p,at)<=guard&&circles.every(q=>Math.abs(dist(p,{x:q.x,y:q.y})-q.r)<=guard));
}
function intrinsicComplement(ctx,a,b){
 // For angle(La,Lb)+angle(Lb,Lc)=90, La perpendicular Lc is the
 // complete, more primitive fact. Suppress all its arbitrary subdivisions.
 if(!a?.a||!b?.a||!a?.b||!b?.b)return false;
 for(const [sharedA,otherA] of [[a.a,a.b],[a.b,a.a]])for(const [sharedB,otherB] of [[b.a,b.b],[b.b,b.a]]){
  if(ctx.aligned(sharedA,sharedB)&&ctx.perpendicular(otherA,otherB))return true;
 }
 return false;
}
function intrinsicSupplement(ctx,a,b){
 if(!a?.v||!b?.v)return false;
 if(!ctx._cachedRayLines)ctx._cachedRayLines=new Map(ctx.L.map(l=>[l.key,l]));
 const line=(v,p)=>{
  const k='P:'+ [v,p].sort().join('|');
  return ctx._cachedRayLines.get(ctx.lineFamily.get(k)||k);
 };
 const x=[line(a.v,a.a),line(a.v,a.b)],y=[line(b.v,b.a),line(b.v,b.b)];
 if(x.some(v=>!v)||y.some(v=>!v))return false;
 return (ctx.aligned(x[0],y[0])&&ctx.aligned(x[1],y[1]))||
        (ctx.aligned(x[0],y[1])&&ctx.aligned(x[1],y[0]));
}
function intrinsicEqualAngle(ctx,a,b){
 // An equality between two exactly identical unoriented angle supports is a
 // self-equality, even if they are written via different vertices or aliases.
 const lines=q=>q?.v?null:[q?.a?.key,q?.b?.key].sort().join('|');
 return lines(a)!==null&&lines(a)===lines(b);
}
return {dist,anchoredCirclePoint,intrinsicCircleTangent,intrinsicLineTangent,isNamedIncidence,intrinsicComplement,intrinsicSupplement,intrinsicEqualAngle};
});
