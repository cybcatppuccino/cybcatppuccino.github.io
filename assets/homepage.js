(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const rootEl = document.documentElement;
  const forms = Array.isArray(window.NEWFORMS) ? window.NEWFORMS : [];
  const polyLibrary = window.POLYHEDRA_LIBRARY && typeof window.POLYHEDRA_LIBRARY === 'object' ? window.POLYHEDRA_LIBRARY : {objects:[],sourceCounts:{}};
  const polyCatalog = Array.isArray(polyLibrary.objects) ? polyLibrary.objects.map(o => ({...o,treeCount:String(o.trees ?? '—'),netCount:String(o.nets ?? o.netCount ?? '—')})) : [];
  const offlineCatalogSize = polyCatalog.length;
  window.CYBCAT_POLYHEDRA = polyCatalog;

  /* ---------- theme ---------- */
  const themeButton = $('themeToggle');
  const viewers = [];
  function isDark(){ return rootEl.dataset.theme === 'dark'; }
  function updateTheme(){
    const dark = isDark();
    $('modeIcon').textContent = dark ? '☼' : '☾';
    themeButton.setAttribute('aria-pressed', String(dark));
    themeButton.setAttribute('aria-label', dark ? 'Switch to day mode' : 'Switch to night mode');
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = dark ? '#151d1f' : '#f5f3ed';
    viewers.forEach(v => v.draw());
  }
  themeButton.addEventListener('click', () => {
    rootEl.dataset.theme = isDark() ? 'light' : 'dark';
    try { localStorage.setItem('cybcat-theme', rootEl.dataset.theme); } catch (_) {}
    updateTheme();
  });

  /* ---------- random newform ---------- */
  const cumulative = [];
  let totalWeight = 0, previousForm = -1;
  forms.forEach(f => {
    totalWeight += 1 / Math.pow(Math.max(1, Number(f.level) || 1), .35);
    cumulative.push(totalWeight);
  });
  function nextFormIndex(){
    if (!forms.length) return -1;
    const target = Math.random() * totalWeight;
    let lo = 0, hi = forms.length - 1;
    while (lo < hi){ const mid = (lo + hi) >>> 1; if (cumulative[mid] < target) lo = mid + 1; else hi = mid; }
    if (lo === previousForm && forms.length > 1) lo = (lo + 1) % forms.length;
    previousForm = lo;
    return lo;
  }
  function cleanNumber(value){
    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);
    if (Math.abs(n) < 1e-12) return '0';
    return n.toPrecision(8).replace(/(?:\.0+|(\.\d*?)0+)$/, '$1');
  }
  function expansion(coeffs){
    const terms = [];
    for (let i = 1; i < coeffs.length && terms.length < 9; i++){
      const a = Number(coeffs[i]); if (!Number.isFinite(a) || a === 0) continue;
      const mono = i === 1 ? 'q' : `q^${i}`;
      const body = Math.abs(a) === 1 ? mono : `${Math.abs(a)}${mono}`;
      terms.push(terms.length ? `${a < 0 ? ' − ' : ' + '}${body}` : `${a < 0 ? '−' : ''}${body}`);
    }
    return terms.length ? terms.join('') : '0';
  }
  function renderNewform(){
    const index = nextFormIndex();
    if (index < 0){ $('nfLabel').textContent = '—'; $('nfQexp').textContent = 'archive unavailable'; return; }
    const f = forms[index];
    $('nfLabel').textContent = `${f.level}.${f.weight}.${f.id}`;
    $('nfWeight').textContent = `k ${f.weight}`;
    $('nfQexp').textContent = `f(q) = ${expansion(f.coeffs || [])} + …`;
    const box = $('nfValues'); box.replaceChildren();
    [['Lhalf','L(½)',f.weight===1],['L1','L(1)',true],['L32','L(3/2)',f.weight===3],['L2','L(2)',f.weight===4]].forEach(([key,label,allow]) => {
      if (!allow || !Object.prototype.hasOwnProperty.call(f,key)) return;
      const cell = document.createElement('div'); cell.className = 'newform-value';
      const small = document.createElement('small'); small.textContent = label;
      const code = document.createElement('code'); code.textContent = cleanNumber(f[key]);
      cell.append(small,code); box.append(cell);
    });
  }
  $('newformNext').addEventListener('click', renderNewform);
  renderNewform();

  /* ---------- polyhedron catalogue ---------- */
  const vlen = p => Math.hypot(p[0],p[1],p[2]);
  const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const sub = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const mul = (a,k) => [a[0]*k,a[1]*k,a[2]*k];
  const unit = a => { const n=vlen(a)||1; return mul(a,1/n); };

  /* ---------- manual Canvas2D viewer ---------- */
  function average(arr){return arr.reduce((a,b)=>a+b,0)/Math.max(1,arr.length);}
  function faceHue(n){const map={3:8,4:206,5:142,6:42,7:325,8:268,9:174,10:186,11:302,12:92};return map[n]??((n*47+17)%360);}
  function lerp3(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
  function newell(poly){
    let nx=0,ny=0,nz=0;
    for(let i=0;i<poly.length;i++){
      const a=poly[i],b=poly[(i+1)%poly.length];
      nx+=(a[1]-b[1])*(a[2]+b[2]);
      ny+=(a[2]-b[2])*(a[0]+b[0]);
      nz+=(a[0]-b[0])*(a[1]+b[1]);
    }
    const q=Math.hypot(nx,ny,nz)||1;
    return [nx/q,ny/q,nz/q];
  }
  function pointInPolygon(x,y,poly){
    let inside=false;
    for(let i=0,j=poly.length-1;i<poly.length;j=i++){
      const xi=poly[i][0],yi=poly[i][1],xj=poly[j][0],yj=poly[j][1];
      const hit=((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/((yj-yi)||1e-12)+xi);
      if(hit)inside=!inside;
    }
    return inside;
  }
  class PolyViewer{
    constructor(canvas){
      this.canvas=canvas;
      this.ctx=canvas.getContext('2d',{alpha:true});
      this.model=null;
      this.ax=-.44;
      this.ay=.62;
      this.drag=null;
      this.w=0;
      this.h=0;
      this.dpr=1;
      this.camDist=7.142857142857143; // matches the previous perspective strength 1 / 0.14
      this.scaleFactor=.39;
      this.saturation=72;
      this.opacity=48;
      this.lastVisibility={visible:0,hidden:0};
      canvas.addEventListener('pointerdown',e=>this.down(e));
      canvas.addEventListener('pointermove',e=>this.move(e));
      canvas.addEventListener('pointerup',e=>this.up(e));
      canvas.addEventListener('pointercancel',()=>this.drag=null);
      new ResizeObserver(()=>this.resize()).observe(canvas);
      viewers.push(this);
    }
    setModel(m){
      this.model=m;
      this.ax=-.42;
      this.ay=.58;
      if(m&&!m._viewerAdjacency){
        const edgeToFaces=new Map();
        const vertexToFaces=Array.from({length:m.points.length},()=>[]);
        m.faces.forEach((face,fi)=>{
          face.forEach(v=>{ if (vertexToFaces[v]) vertexToFaces[v].push(fi); });
          for(let i=0;i<face.length;i++){
            const a=face[i],b=face[(i+1)%face.length],x=Math.min(a,b),y=Math.max(a,b),k=`${x},${y}`;
            if(!edgeToFaces.has(k))edgeToFaces.set(k,[]);
            edgeToFaces.get(k).push(fi);
          }
        });
        m._viewerAdjacency={edgeToFaces,vertexToFaces};
      }
      this.draw();
    }
    resize(){
      const r=this.canvas.getBoundingClientRect();
      if(!r.width||!r.height)return;
      this.dpr=Math.min(devicePixelRatio||1,2);
      this.w=r.width;this.h=r.height;
      this.canvas.width=Math.max(1,Math.round(r.width*this.dpr));
      this.canvas.height=Math.max(1,Math.round(r.height*this.dpr));
      this.ctx.setTransform(this.dpr,0,0,this.dpr,0,0);
      this.draw();
    }
    down(e){ if(e.button!==0)return; this.canvas.setPointerCapture?.(e.pointerId); this.drag={id:e.pointerId,x:e.clientX,y:e.clientY}; }
    move(e){
      if(!this.drag||e.pointerId!==this.drag.id)return;
      const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;
      // Keep the previous hand feel; primary v7 fix is the occlusion logic.
      this.ay-=dx*.011;
      this.ax+=dy*.011;
      this.drag.x=e.clientX; this.drag.y=e.clientY; this.draw();
    }
    up(e){ if(this.drag?.id===e.pointerId)this.drag=null; }
    setSaturation(value){ this.saturation=Math.max(0,Math.min(100,Number(value)||0)); this.draw(); }
    setOpacity(value){ this.opacity=Math.max(0,Math.min(100,Number(value)||0)); this.draw(); }
    rotateView(p){
      const cx=Math.cos(this.ax),sx=Math.sin(this.ax),cy=Math.cos(this.ay),sy=Math.sin(this.ay);
      const x=p[0],y=p[1],z=p[2]||0;
      const yy=y*cx-z*sx,zz=y*sx+z*cx,xx=x*cy+zz*sy,z2=-x*sy+zz*cy;
      return [xx,yy,z2];
    }
    project(v){
      const depth=this.camDist/(this.camDist+v[2]);
      const scale=Math.min(this.w,this.h)*this.scaleFactor;
      return [this.w/2+v[0]*scale*depth,this.h/2+v[1]*scale*depth,v[2],depth];
    }
    prepareFaces(view,screen){
      return this.model.faces.map((face,idx)=>{
        const vv=face.map(i=>view[i]).filter(Boolean);
        const ss=face.map(i=>screen[i]).filter(Boolean);
        if(vv.length<3||ss.length<3)return null;
        let n=newell(vv);
        const centroid=vv.reduce((q,p)=>[q[0]+p[0],q[1]+p[1],q[2]+p[2]],[0,0,0]).map(x=>x/vv.length);
        if(dot(n,centroid)<0)n=mul(n,-1);
        const d=dot(n,vv[0]);
        const origin=vv[0];
        const camera=[0,0,-this.camDist];
        const facing=dot(n,sub(camera,centroid))>0;
        const u=unit(sub(vv[1],origin));
        const v=cross(n,u);
        const poly2=vv.map(p=>{const rel=sub(p,origin);return [dot(rel,u),dot(rel,v)];});
        let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
        ss.forEach(p=>{ if(p[0]<minX)minX=p[0]; if(p[0]>maxX)maxX=p[0]; if(p[1]<minY)minY=p[1]; if(p[1]>maxY)maxY=p[1]; });
        return {face,idx,vv,ss,n,d,origin,u,v,poly2,centroid,facing,z:average(vv.map(p=>p[2])),minX,maxX,minY,maxY};
      }).filter(Boolean);
    }
    pointInFace(point,face){
      const rel=sub(point,face.origin);
      return pointInPolygon(dot(rel,face.u),dot(rel,face.v),face.poly2);
    }
    isOccluded(sample,faces,skipFaces=[]){
      const camera=[0,0,-this.camDist];
      const dir=sub(sample.v,camera);
      const sx=sample.s?.[0],sy=sample.s?.[1];
      for(const f of faces){
        if(skipFaces.includes(f.idx))continue;
        if(Number.isFinite(sx)&&Number.isFinite(sy)){
          if(sx<f.minX-.5||sx>f.maxX+.5||sy<f.minY-.5||sy>f.maxY+.5)continue;
        }
        const den=dot(f.n,dir);
        if(Math.abs(den)<1e-10)continue;
        const t=(f.d-dot(f.n,camera))/den;
        // Only an intersection strictly between the camera and the sample can occlude the sample.
        if(t<=1e-5||t>=1-1e-5)continue;
        const hit=[camera[0]+dir[0]*t,camera[1]+dir[1]*t,camera[2]+dir[2]*t];
        if(this.pointInFace(hit,f))return true;
      }
      return false;
    }
    draw(){
      if(!this.model||!this.w||!this.h)return;
      const ctx=this.ctx,m=this.model,dark=isDark();
      const view=m.points.map(p=>this.rotateView(p));
      const pts=view.map(p=>this.project(p));
      const faces=this.prepareFaces(view,pts);
      const adj=m._viewerAdjacency||{edgeToFaces:new Map(),vertexToFaces:[]};
      ctx.clearRect(0,0,this.w,this.h);
      ctx.lineJoin='round';
      ctx.lineCap='round';

      const hiddenSeg=[],visibleSeg=[];
      const steps=faces.length>90?4:faces.length>50?5:8;
      for(const [a,b] of m.edges){
        const A=view[a],B=view[b];
        if(!A||!B)continue;
        const edgeKey=`${Math.min(a,b)},${Math.max(a,b)}`;
        const skip=adj.edgeToFaces.get(edgeKey)||[];
        for(let j=0;j<steps;j++){
          const t0=j/steps,t1=(j+1)/steps,tm=(j+.5)/steps;
          const v0=lerp3(A,B,t0),v1=lerp3(A,B,t1),vm=lerp3(A,B,tm);
          const s0=this.project(v0),s1=this.project(v1),sm=this.project(vm);
          (this.isOccluded({v:vm,s:sm},faces,skip)?hiddenSeg:visibleSeg).push([s0,s1]);
        }
      }
      this.lastVisibility={visible:visibleSeg.length,hidden:hiddenSeg.length};
      this.canvas.dataset.visibleSegments=String(visibleSeg.length);
      this.canvas.dataset.hiddenSegments=String(hiddenSeg.length);

      if(hiddenSeg.length){
        ctx.beginPath();
        for(const [a,b] of hiddenSeg){ ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); }
        ctx.strokeStyle=dark?'#d7dcd5':'#2b332f';
        const o=this.opacity/100;
        ctx.globalAlpha=.16*Math.pow(o,1.65);
        ctx.lineWidth=.56;
        ctx.stroke();
      }

      // Smaller z is closer to the camera, so paint larger-z faces first.
      // α now controls ONLY rear/occluded structure. Front-facing visible faces keep
      // a stable colour/opacity at every α value; lowering α removes visual bleed-through
      // from the back without washing out the front surface.
      const minFaceZ=Math.min(...faces.map(f=>f.z)),maxFaceZ=Math.max(...faces.map(f=>f.z)),faceRange=Math.max(1e-6,maxFaceZ-minFaceZ);
      const rearVisibility=this.opacity/100;
      const frontAlpha=.235;
      const rearAlpha=frontAlpha*Math.pow(rearVisibility,1.35);
      this.lastFaceAlpha={front:frontAlpha,rear:rearAlpha};
      faces.slice().sort((a,b)=>b.z-a.z).forEach(f=>{
        const hue=faceHue(f.face.length);
        const near=(maxFaceZ-f.z)/faceRange;
        const screenCentroid=this.project(f.centroid);
        const occluded=this.isOccluded({v:f.centroid,s:screenCentroid},faces,[f.idx]);
        const rear=occluded||!f.facing;
        const saturation=Math.max(0,Math.min(100,this.saturation*(rear?.74:(.94+.06*near))));
        if(rear&&rearAlpha<.001)return;
        ctx.beginPath();
        ctx.moveTo(f.ss[0][0],f.ss[0][1]);
        for(let i=1;i<f.ss.length;i++)ctx.lineTo(f.ss[i][0],f.ss[i][1]);
        ctx.closePath();
        ctx.fillStyle=`hsl(${hue} ${saturation.toFixed(1)}% ${dark?61:58}%)`;
        ctx.globalAlpha=rear?rearAlpha:frontAlpha;
        ctx.fill('evenodd');
      });

      if(visibleSeg.length){
        ctx.beginPath();
        for(const [a,b] of visibleSeg){ ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); }
        ctx.strokeStyle=dark?'#e7e9e3':'#1e2522';
        ctx.globalAlpha=.82;
        ctx.lineWidth=1.32;
        ctx.stroke();
      }

      const dotFill=dark?'rgba(20,27,29,.9)':'rgba(247,245,240,.94)';
      const dotStroke=dark?'rgba(239,241,235,.94)':'rgba(24,30,27,.92)';
      const hiddenDots=[],visibleDots=[];
      view.forEach((v,i)=>{
        const s=pts[i],skip=adj.vertexToFaces[i]||[];
        (this.isOccluded({v,s},faces,skip)?hiddenDots:visibleDots).push(s);
      });
      const hiddenDotAlpha=.28*Math.pow(this.opacity/100,1.65);
      for(const [list,alpha,r,lw] of [[hiddenDots,hiddenDotAlpha,1.65,.5],[visibleDots,.96,2.5,.9]]){
        ctx.globalAlpha=alpha;
        for(const p of list){
          ctx.beginPath();
          ctx.arc(p[0],p[1],r,0,Math.PI*2);
          ctx.fillStyle=dotFill;
          ctx.fill();
          ctx.lineWidth=lw;
          ctx.strokeStyle=dotStroke;
          ctx.stroke();
        }
      }
      ctx.globalAlpha=1;
    }
  }
  const solidViewer=new PolyViewer($('solidCanvas'));
  window.CYBCAT_VIEWER=solidViewer;

  function exactCount(value){
    const raw=String(value??'—').replace(/,/g,'');
    return /^\d+$/.test(raw)?raw:String(value??'—');
  }
  function fitNetNumbers(){
    const badge=$('solidNetBadge');
    if(!badge)return;
    const maxWidth=Math.max(40,badge.clientWidth-16);
    for(const id of ['solidTreeCount','solidNetCount']){
      const el=$(id);if(!el)continue;
      el.style.fontSize='';
      let size=8.8;
      while(el.scrollWidth>maxWidth&&size>4.2){size-=.2;el.style.fontSize=`${size.toFixed(1)}px`;}
    }
  }
  function randomSolidIndex(){
    if(!polyCatalog.length)return -1;
    return Math.floor(Math.random()*polyCatalog.length);
  }
  function renderSolid(){
    const i=randomSolidIndex();
    if(i<0)return;
    const m=polyCatalog[i],trees=exactCount(m.treeCount),display=exactCount(m.netCount);
    const nameEl=$('solidName');
    nameEl.textContent=m.name;
    nameEl.title=m.name;
    nameEl.classList.toggle('name-long',m.name.length>48);
    nameEl.classList.toggle('name-very-long',m.name.length>78);
    $('solidTreeCount').textContent=trees;
    $('solidNetCount').textContent=display;
    $('solidNetBadge').title=`${trees} spanning trees / ${display} symmetry-reduced`;
    $('solidCard').dataset.source=m.source;
    requestAnimationFrame(fitNetNumbers);
    solidViewer.setModel(m);
  }
  $('solidNext').addEventListener('click',renderSolid);
  const saturationControl=$('solidSaturation'),opacityControl=$('solidOpacity');
  saturationControl?.addEventListener('input',e=>solidViewer.setSaturation(e.target.value));
  opacityControl?.addEventListener('input',e=>solidViewer.setOpacity(e.target.value));
  if(saturationControl)solidViewer.saturation=Number(saturationControl.value)||72;
  if(opacityControl)solidViewer.opacity=Number(opacityControl.value)||48;
  const netBadge=$('solidNetBadge');
  if(netBadge)new ResizeObserver(()=>fitNetNumbers()).observe(netBadge);
  renderSolid();

  window.CYBCAT_CATALOG_INFO = {total:polyCatalog.length,offline:offlineCatalogSize,...(polyLibrary.sourceCounts||{})};
  updateTheme();
})();
