(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const rootEl = document.documentElement;
  const forms = Array.isArray(window.NEWFORMS) ? window.NEWFORMS : [];
  const polyLibrary = window.POLYHEDRA_LIBRARY && typeof window.POLYHEDRA_LIBRARY === 'object' ? window.POLYHEDRA_LIBRARY : {objects:[],sourceCounts:{}};
  const reflexiveLibrary = window.REFLEXIVE_LIBRARY && typeof window.REFLEXIVE_LIBRARY === 'object' ? window.REFLEXIVE_LIBRARY : null;
  // Reuse the loaded models, including their geometry: no duplicate catalogue or geometry.
  const polyCatalog = Array.isArray(polyLibrary.objects) ? polyLibrary.objects : [];
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

  /* ---------- newforms: weighted unfiltered random, indexed navigation & search ---------- */
  // Keep the previous 1 / level^0.35 random distribution, with a compact typed CDF.
  const cumulative = new Float64Array(forms.length);
  let totalWeight = 0, currentForm = -1, formMatches = null;
  for (let i = 0; i < forms.length; i++) {
    totalWeight += 1 / Math.pow(Math.max(1, Number(forms[i].level) || 1), .35);
    cumulative[i] = totalWeight;
  }
  const formLabel = f => `${f.level}.${f.weight}.${f.id}`;
  function weightedRandomForm() {
    if (!forms.length) return -1;
    const target = Math.random() * totalWeight;
    let lo = 0, hi = forms.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cumulative[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    // Preserve the original one-step no-immediate-repeat behavior.
    return lo === currentForm && forms.length > 1 ? (lo + 1) % forms.length : lo;
  }
  function cleanNumber(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);
    if (Math.abs(n) < 1e-12) return '0';
    return n.toPrecision(8).replace(/(?:\.0+|(\.\d*?)0+)$/, '$1');
  }
  function expansion(coeffs) {
    const terms = [];
    const end = Math.min(100, coeffs.length - 1);
    for (let i = 1; i <= end; i++) {
      const a = Number(coeffs[i]);
      if (!Number.isFinite(a) || a === 0) continue;
      const mono = i === 1 ? 'q' : `q^${i}`;
      const term = `${Math.abs(a) === 1 ? '' : Math.abs(a)}${mono}`;
      terms.push(terms.length ? `${a < 0 ? ' − ' : ' + '}${term}` : `${a < 0 ? '−' : ''}${term}`);
    }
    // Never pretend a truncated source is verified through q^100.
    return (terms.join('') || '0') + (end === 100 ? ' + O(q^101)' : ' + …');
  }

  // Four value elements created once and reused on every change, avoiding DOM churn.
  const valueSpecs = [['Lhalf','L(½)',1],['L1','L(1)',0],['L32','L(3/2)',3],['L2','L(2)',4]];
  const valueCells = valueSpecs.map(([key, label, weight]) => {
    const cell = document.createElement('div'); cell.className = 'newform-value';
    const small = document.createElement('small'); small.textContent = label;
    const code = document.createElement('code');
    cell.append(small, code);
    $('nfValues').append(cell);
    return {cell, code, key, weight};
  });
  function setFormButtons(disabled) {
    for (const id of ['newformPrev','newformForward','newformNext']) $(id).disabled = disabled;
  }
  function renderNewform(index) {
    currentForm = index;
    if (index < 0) {
      $('nfLabel').textContent = forms.length ? 'No matches' : '—';
      $('nfOldLabel').textContent = '';
      $('nfQexp').textContent = forms.length ? 'No matching newform' : 'archive unavailable';
      for (const v of valueCells) v.cell.hidden = true;
      setFormButtons(true);
      return;
    }
    setFormButtons(false);
    const f = forms[index];
    // Show only coefficients and LMFDB labels verified against the supplied
    // source traces; every homepage newform has a q^100 expansion.
    $('nfLabel').textContent = f.lmfdb_label || 'LMFDB —';
    $('nfOldLabel').textContent = formLabel(f);
    $('nfOldLabel').title = 'PFTool label';
    $('nfQexp').textContent = `f(q) = ${expansion(f.coeffs || [])}`;
    $('nfQexp').scrollTop = 0;
    for (const v of valueCells) {
      const visible = (!v.weight || v.weight === f.weight) && Object.prototype.hasOwnProperty.call(f, v.key);
      v.cell.hidden = !visible;
      if (visible) v.code.textContent = cleanNumber(f[v.key]);
    }
  }
  function randomNewform() {
    if (formMatches) {
      renderNewform(formMatches.length ? formMatches[Math.floor(Math.random() * formMatches.length)] : -1);
    } else renderNewform(weightedRandomForm());
  }
  function moveNewform(delta) {
    if (formMatches) {
      const n = formMatches.length;
      if (!n) return;
      const position = formMatches.indexOf(currentForm);
      renderNewform(formMatches[((position < 0 ? 0 : position) + delta + n) % n]);
    } else if (forms.length) {
      renderNewform((currentForm + delta + forms.length) % forms.length);
    }
  }

  /* ---------- matching: direct labels, q coefficients, IDs, names, counts, typo tolerance ---------- */
  function normalizeQuery(s) {
    return String(s).normalize('NFKC').toLowerCase()
      .replace(/[−–—‐‑]/g, '-').replace(/\s+/g, ' ').trim();
  }
  function compact(s) { return normalizeQuery(s).replace(/[\s_*]/g, ''); }
  function parseQSearch(query) {
    const q = compact(query);
    const m = /^([+-]?\d*)q(?:\^?(\d{1,3}))?$/.exec(q);
    if (!m) return null;
    const n = m[2] ? Number(m[2]) : 1;
    if (n < 1 || n > 100) return {power:n, coefficient:null, outside:true};
    let coefficient = null;
    if (m[1]) coefficient = m[1] === '-' ? -1 : m[1] === '+' ? 1 : Number(m[1]);
    return {power:n, coefficient, outside:false};
  }
  function textMatchScore(candidate, query) {
    if (!candidate) return -1;
    const c = compact(candidate), q = compact(query);
    if (!q) return -1;
    if (c === q) return 0;
    if (c.startsWith(q)) return 2;
    if (c.includes(q)) return 5;
    return -1;
  }
  // Bounded Levenshtein only on a no-direct-results fallback; no large search index.
  function editDistanceWithin(a, b, bound) {
    if (Math.abs(a.length - b.length) > bound) return false;
    let prev = Array.from({length:b.length+1}, (_,i) => i);
    let curr = new Array(b.length+1);
    for (let i=1; i<=a.length; i++) {
      curr[0] = i;
      let min = curr[0];
      for (let j=1; j<=b.length; j++) {
        curr[j] = Math.min(prev[j]+1, curr[j-1]+1, prev[j-1]+(a[i-1] === b[j-1] ? 0 : 1));
        if (curr[j] < min) min = curr[j];
      }
      if (min > bound) return false;
      [prev,curr] = [curr,prev];
    }
    return prev[b.length] <= bound;
  }
  function fuzzyMatch(haystack, query) {
    const q = compact(query);
    if (q.length < 4) return false;
    const bound = q.length >= 9 ? 2 : 1;
    const words = normalizeQuery(haystack).split(/[^\p{L}\p{N}.]+/u);
    return words.some(word => word && editDistanceWithin(compact(word), q, bound));
  }
  function scoreForm(f, query, qSearch) {
    if (qSearch) {
      if (qSearch.outside || !f.coeffs || qSearch.power >= f.coeffs.length) return -1;
      const a = f.coeffs[qSearch.power];
      return Number.isFinite(Number(a)) && (qSearch.coefficient === null ? Number(a) !== 0 : Number(a) === qSearch.coefficient) ? 0 : -1;
    }
    const old = formLabel(f);
    const oldScore = textMatchScore(old, query);
    const officialScore = textMatchScore(f.lmfdb_label, query);
    return oldScore < 0 ? officialScore : officialScore < 0 ? oldScore : Math.min(oldScore, officialScore);
  }
  function scorePoly(m, query) {
    const q = compact(query);
    const id = compact(m.id || '');
    const key = compact(m.key || '');
    if (id === q || key === q || (/^\d+$/.test(q) && (id === 'j'+q || key === 'j'+q))) return 0;
    const fields = [m.name, m.id, m.key, m.source, m.mccooeyFile, m.rpolyFileId, m.uniformNumber,
      m.trees, m.nets, m.netCount, m.symmetryOrder];
    let best = Infinity;
    for (const field of fields) {
      if (field == null) continue;
      const score = textMatchScore(String(field).replace(/,/g,''), query);
      if (score >= 0 && score < best) best = score;
    }
    if (best < Infinity) return best;
    // Multiword searches can match separate fields (e.g. 'johnson 25').
    const terms = normalizeQuery(query).split(/\s+/);
    if (terms.length > 1) {
      const text = fields.filter(x=>x != null).join(' ').toLowerCase();
      if (terms.every(t=>text.includes(t))) return 8;
    }
    return -1;
  }
  function findMatches(items, query, score, fuzzy) {
    const ranked = [];
    for (let i = 0; i < items.length; i++) {
      const rank = score(items[i], query);
      if (rank >= 0) ranked.push([rank,i]);
    }
    if (!ranked.length && fuzzy) {
      for (let i = 0; i < items.length; i++) if (fuzzy(items[i], query)) ranked.push([20,i]);
    }
    ranked.sort((a,b)=>a[0]-b[0] || a[1]-b[1]);
    return ranked.map(v => v[1]);
  }
  function refreshNewformSearch() {
    // Synchronous on every input; never debounce or require Enter.
    const query = normalizeQuery($('newformSearch').value);
    if (!query) {
      formMatches = null;
      renderNewform(currentForm >= 0 ? currentForm : weightedRandomForm());
      return;
    }
    const qSearch = parseQSearch(query);
    formMatches = findMatches(forms, query, (f,s)=>scoreForm(f,s,qSearch),
      qSearch ? null : (f,s)=>fuzzyMatch(`${formLabel(f)} ${f.lmfdb_label||''}`,s));
    // Input edits restart navigation at the first current match. This makes a
    // narrowed/expanded result set visible at once, even when the previous
    // selection remains a valid match.
    renderNewform(formMatches[0] ?? -1);
  }
  $('newformSearch').addEventListener('input', refreshNewformSearch);
  $('newformSearch').addEventListener('search', refreshNewformSearch);
  $('newformPrev').addEventListener('click',()=>moveNewform(-1));
  $('newformForward').addEventListener('click',()=>moveNewform(1));
  $('newformNext').addEventListener('click',randomNewform);
  randomNewform();

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
      // SO(3) orientation, represented by a unit quaternion (w, x, y, z).
      // Every drag increment is left-composed in the CURRENT screen frame.
      this.orientation=[1,0,0,0];
      this.rotationMatrix=[1,0,0,0,1,0,0,0,1];
      this.resetOrientation(-.44,.62);
      this.drag=null;
      this.w=0;
      this.h=0;
      this.dpr=1;
      this.camDist=7.142857142857143; // matches the previous perspective strength 1 / 0.14
      this.scaleFactor=.39;
      // Reflexive polytopes use a fixed orthographic camera: rotation never zooms.
      this.orthographic=canvas.id==='reflexiveCanvas';
      this.viewCenter=[0,0,0];
      this.fitRadius=1;
      this.fitScale=1;
      this.viewOffsetX=0;
      this.viewOffsetY=0;
      this.axesHalfLength=1;
      this.saturation=50;
      this.opacity=33;
      this.lastVisibility={visible:0,hidden:0};
      canvas.addEventListener('pointerdown',e=>this.down(e));
      canvas.addEventListener('pointermove',e=>this.move(e));
      canvas.addEventListener('pointerup',e=>this.up(e));
      canvas.addEventListener('pointercancel',()=>this.drag=null);
      new ResizeObserver(()=>this.resize()).observe(canvas);
      viewers.push(this);
    }
    setModel(m, preserveOrientation=false){
      this.model=m;
      if (!m) { if (this.w && this.h) this.ctx.clearRect(0,0,this.w,this.h); return; }
      if(!preserveOrientation)this.resetOrientation(-.42,.58);
      this.updateProjectionFit();
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
      this.updateProjectionFit();
      this.draw();
    }
    updateProjectionFit(){
      if(!this.orthographic||!this.model||!this.w||!this.h)return;
      const m=this.model;
      const points=m.latticePoints?.length?m.latticePoints:m.points;
      const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
      for(const p of points){
        for(let j=0;j<3;j++){
          low[j]=Math.min(low[j],p[j]);
          high[j]=Math.max(high[j],p[j]);
        }
      }
      // The centre and radius depend on the selected model, NOT its orientation.
      this.viewCenter=low.map((v,j)=>(v+high[j])/2);
      this.axesHalfLength=Math.max(1,Math.min(3,Math.max(...low.map(Math.abs),...high.map(Math.abs))));
      let radius=0;
      for(const p of points){
        radius=Math.max(radius,Math.hypot(...sub(p,this.viewCenter)));
      }
      // Include the unlabelled x/y/z axis ends in the guaranteed fit.
      for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
        const end=[0,0,0];end[axis]=sign*this.axesHalfLength;
        radius=Math.max(radius,Math.hypot(...sub(end,this.viewCenter)));
      }
      this.fitRadius=Math.max(radius,1e-6);
      const pad=Math.max(10,Math.min(18,Math.min(this.w,this.h)*.045));
      const controlsReserve=this.w>=170?48:32;
      const usableWidth=Math.max(1,this.w-controlsReserve-2*pad);
      const usableHeight=Math.max(1,this.h-2*pad);
      this.fitScale=Math.min(usableWidth,usableHeight)/(2*this.fitRadius);
      this.viewOffsetX=pad+usableWidth/2;
      this.viewOffsetY=this.h/2;
    }
    // Quaternion multiplication: q1*q2 applies q2 first, then q1.
    multiplyRotations(a,b){
      const [w,x,y,z]=a,[v,i,j,k]=b;
      return [w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v];
    }
    storeOrientation(q){
      // Renormalize on every step so arbitrarily long drags remain in SO(3).
      const norm=Math.hypot(...q)||1;
      const [w,x,y,z]=q.map(v=>v/norm);
      this.orientation=[w,x,y,z];
      this.rotationMatrix=[
        1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)
      ];
    }
    resetOrientation(pitch,yaw){
      // Match the previous default view: R_y(yaw) * R_x(pitch).
      const qx=[Math.cos(pitch/2),Math.sin(pitch/2),0,0];
      const qy=[Math.cos(yaw/2),0,Math.sin(yaw/2),0];
      this.storeOrientation(this.multiplyRotations(qy,qx));
    }
    rotateScreen(dx,dy){
      // The drag axis (dy,-dx,0) is expressed in CAMERA coordinates.
      // Thus R_new = exp([omega]_x) * R_current, rather than changing
      // Euler angles about the original model axes (which causes gimbal lock).
      const travel=Math.hypot(dx,dy);
      if(!travel)return;
      const halfAngle=.011*travel/2;
      const f=Math.sin(halfAngle)/travel;
      const delta=[Math.cos(halfAngle),dy*f,-dx*f,0];
      this.storeOrientation(this.multiplyRotations(delta,this.orientation));
    }
    down(e){ if(e.button!==0)return; this.canvas.setPointerCapture?.(e.pointerId); this.drag={id:e.pointerId,x:e.clientX,y:e.clientY}; }
    move(e){
      if(!this.drag||e.pointerId!==this.drag.id)return;
      const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;
      this.rotateScreen(dx,dy);
      this.drag.x=e.clientX; this.drag.y=e.clientY; this.draw();
    }
    up(e){ if(this.drag?.id===e.pointerId)this.drag=null; }
    setSaturation(value){ this.saturation=Math.max(0,Math.min(100,Number(value)||0)); this.draw(); }
    setOpacity(value){ this.opacity=Math.max(0,Math.min(100,Number(value)||0)); this.draw(); }
    rotateView(p){
      const center=this.orthographic?this.viewCenter:[0,0,0];
      const x=p[0]-center[0],y=p[1]-center[1],z=(p[2]||0)-center[2];
      const r=this.rotationMatrix;
      return [r[0]*x+r[1]*y+r[2]*z,r[3]*x+r[4]*y+r[5]*z,r[6]*x+r[7]*y+r[8]*z];
    }
    project(v){
      if(this.orthographic){
        return [this.viewOffsetX+v[0]*this.fitScale,this.viewOffsetY+v[1]*this.fitScale,v[2],1];
      }
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
        const interior=this.orthographic?this.rotateView([0,0,0]):[0,0,0];
        if(dot(n,sub(centroid,interior))<0)n=mul(n,-1);
        const d=dot(n,vv[0]);
        const origin=vv[0];
        const camera=[0,0,-this.camDist];
        const facing=this.orthographic?n[2]<-1e-9:dot(n,sub(camera,centroid))>0;
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
        let hit;
        if(this.orthographic){
          // Orthographic rays have fixed x/y, travelling from negative z.
          if(Math.abs(f.n[2])<1e-10)continue;
          const z=(f.d-f.n[0]*sample.v[0]-f.n[1]*sample.v[1])/f.n[2];
          if(z>=sample.v[2]-1e-5)continue;
          hit=[sample.v[0],sample.v[1],z];
        }else{
          const den=dot(f.n,dir);
          if(Math.abs(den)<1e-10)continue;
          const t=(f.d-dot(f.n,camera))/den;
          // Only an intersection strictly between the camera and the sample occludes it.
          if(t<=1e-5||t>=1-1e-5)continue;
          hit=[camera[0]+dir[0]*t,camera[1]+dir[1]*t,camera[2]+dir[2]*t];
        }
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
      // α=0: a fully opaque body; α=1: no face coating, all geometry visible.
      // The visibility multiplier is used for EVERY covered line and point,
      // including the axis portions inside the body and the exterior grid dots
      // seen through the body. No constant alpha floor remains at α=0.
      const transparency=this.opacity/100;
      const coveredVisibility=Math.pow(transparency,1.35);
      ctx.clearRect(0,0,this.w,this.h);
      ctx.lineJoin='round';
      ctx.lineCap='round';

      // Paint polygon faces from back to front, before all wireframe elements.
      // Front face opacity reaches exactly one when α=0 and exactly zero at 100.
      // Back faces vanish at both endpoints; at partial transparency they add
      // only a faint internal tint without showing through an opaque front.
      const minFaceZ=Math.min(...faces.map(f=>f.z)),maxFaceZ=Math.max(...faces.map(f=>f.z));
      const faceRange=Math.max(1e-6,maxFaceZ-minFaceZ);
      const frontAlpha=1-transparency;
      const rearAlpha=.18*transparency*(1-transparency);
      this.lastFaceAlpha={front:frontAlpha,rear:rearAlpha};
      for(const f of faces.slice().sort((a,b)=>b.z-a.z)){
        const near=(maxFaceZ-f.z)/faceRange;
        const screenCentroid=this.project(f.centroid);
        const rear=!f.facing||this.isOccluded({v:f.centroid,s:screenCentroid},faces,[f.idx]);
        const alpha=rear?rearAlpha:frontAlpha;
        if(alpha<=0)continue;
        const hue=m.reflexive?((f.idx*137.508+23)%360):faceHue(f.face.length);
        const saturation=Math.max(0,Math.min(100,this.saturation*(rear?.74:(.94+.06*near))));
        ctx.beginPath();ctx.moveTo(f.ss[0][0],f.ss[0][1]);
        for(let i=1;i<f.ss.length;i++)ctx.lineTo(f.ss[i][0],f.ss[i][1]);
        ctx.closePath();
        ctx.fillStyle=`hsl(${hue} ${saturation.toFixed(1)}% ${dark?61:58}%)`;
        ctx.globalAlpha=alpha;
        ctx.fill('evenodd');
      }

      // Sample the projected and depth components of every segment. A segment
      // crossing the silhouette can be visible outside the solid yet invisible
      // inside: use fine fragments instead of classifying its entire midpoint.
      const divideLine=(a,b,skip,shown,covered,pixelStep=12)=>{
        const s0=this.project(a),s1=this.project(b);
        const screenLength=Math.hypot(s1[0]-s0[0],s1[1]-s0[1]);
        const depthLength=Math.abs(b[2]-a[2])*(this.orthographic?this.fitScale:Math.min(this.w,this.h)*.32);
        const steps=Math.max(1,Math.min(90,Math.ceil(Math.max(screenLength,depthLength)/pixelStep)));
        for(let j=0;j<steps;j++){
          const left=lerp3(a,b,j/steps),right=lerp3(a,b,(j+1)/steps);
          const mid=lerp3(a,b,(j+.5)/steps);
          const sink=this.isOccluded({v:mid,s:this.project(mid)},faces,skip)?covered:shown;
          sink.push([this.project(left),this.project(right)]);
        }
      };
      const strokeLines=(parts,color,width,alpha)=>{
        if(!parts.length||alpha<=0)return;
        ctx.beginPath();
        for(const [a,b] of parts){ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}
        ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();
      };
      const edgeShown=[],edgeCovered=[];
      for(const [a,b] of m.edges){
        if(!view[a]||!view[b])continue;
        const key=`${Math.min(a,b)},${Math.max(a,b)}`;
        divideLine(view[a],view[b],adj.edgeToFaces.get(key)||[],edgeShown,edgeCovered);
      }
      this.lastVisibility={visible:edgeShown.length,hidden:edgeCovered.length};
      this.canvas.dataset.visibleSegments=String(edgeShown.length);
      this.canvas.dataset.hiddenSegments=String(edgeCovered.length);
      // Draw covered segments first, and always with a true zero opacity when α=0.
      const edgeColor=dark?'#e7e9e3':'#1e2522';
      strokeLines(edgeCovered,edgeColor,1.15,.85*coveredVisibility);
      strokeLines(edgeShown,edgeColor,1.32,.85);

      if(this.orthographic){
        // Bare x/y/z axes: uncovered sections remain visible, covered sections
        // gradually disappear in lockstep with every other occluded element.
        const length=this.axesHalfLength;
        const axisColors=dark?['#f18f88','#84d5aa','#85b9fa']:['#bb5454','#31936c','#437fc4'];
        for(let j=0;j<3;j++){
          const a=[0,0,0],b=[0,0,0];a[j]=-length;b[j]=length;
          const shown=[],covered=[];
          divideLine(this.rotateView(a),this.rotateView(b),[],shown,covered,8);
          strokeLines(covered,axisColors[j],1.2,.76*coveredVisibility);
          strokeLines(shown,axisColors[j],1.2,.76);
        }
      }

      if(m.reflexive&&Array.isArray(m.latticePoints)){
        if(Array.isArray(m.latticeLinks)&&m.latticeLinks.length){
          // These links connect only the boundary/interior lattice set V.
          const shown=[],covered=[];
          const rotated=m.latticePoints.map(p=>this.rotateView(p));
          for(const [ia,ib] of m.latticeLinks){
            divideLine(rotated[ia],rotated[ib],[],shown,covered,12);
          }
          const linkColor=dark?'#aad2c0':'#4b806c';
          strokeLines(covered,linkColor,.55,.42*coveredVisibility);
          strokeLines(shown,linkColor,.55,.42);
        }
        const colors=dark
          ? ['#e9bd75','#79cbe6','#a0e1a5','#91aba3']
          : ['#9b5e2f','#287f9e','#27834f','#6f8d81'];
        const dotStroke=dark?'rgba(15,21,22,.94)':'rgba(250,249,245,.96)';
        const hiddenByType=[0,0,0,0],shownByType=[0,0,0,0];
        // Draw covered dots first, then front dots. Crucially, types 2 (inside)
        // and 3 (exterior neighbours) undergo the SAME surface occlusion test.
        const visible=[],covered=[];
        for(const p of m.latticePoints){
          const v=this.rotateView(p),s=this.project(v),type=p[3];
          const skip=type===0?(adj.vertexToFaces[p[4]]||[]):[];
          const point={s,type};
          const occluded=this.isOccluded({v,s},faces,skip);
          (occluded?covered:visible).push(point);
          (occluded?hiddenByType:shownByType)[type]++;
        }
        this.lastGridVisibility={visible:shownByType,hidden:hiddenByType};
        for(const [group,factor] of [[covered,coveredVisibility],[visible,1]]){
          if(factor===0)continue;
          for(const {s,type} of group){
            const base=type===3?.47:type===2?.89:.98;
            ctx.globalAlpha=base*factor;
            ctx.beginPath();
            ctx.arc(s[0],s[1],type===0?2.65:type===1?1.75:type===2?2.35:1.25,0,Math.PI*2);
            ctx.fillStyle=colors[type];ctx.fill();
            ctx.lineWidth=type===0?.9:.65;
            ctx.strokeStyle=dotStroke;ctx.stroke();
          }
        }
      }else{
        const dotFill=dark?'rgba(20,27,29,.9)':'rgba(247,245,240,.94)';
        const dotStroke=dark?'rgba(239,241,235,.94)':'rgba(24,30,27,.92)';
        const covered=[],shown=[];
        view.forEach((v,i)=>{
          const s=pts[i],skip=adj.vertexToFaces[i]||[];
          (this.isOccluded({v,s},faces,skip)?covered:shown).push(s);
        });
        for(const [group,alpha] of [[covered,.96*coveredVisibility],[shown,.96]]){
          if(alpha<=0)continue;
          ctx.globalAlpha=alpha;
          for(const s of group){
            ctx.beginPath();ctx.arc(s[0],s[1],2.5,0,Math.PI*2);
            ctx.fillStyle=dotFill;ctx.fill();
            ctx.lineWidth=.9;ctx.strokeStyle=dotStroke;ctx.stroke();
          }
        }
      }
      this.lastTransparency={value:this.opacity,coveredVisibility,faceOpacity:frontAlpha};
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
  // Unbiased selection across *all* loaded polyhedra, not across source groups.
  // Rejection sampling removes modulo bias; secure randomness when available.
  const randomWord = new Uint32Array(1);
  function uniformIndex(count) {
    if (count < 1) return -1;
    if (globalThis.crypto?.getRandomValues && count <= 0x100000000) {
      const range = 0x100000000;
      const ceiling = range - range % count;
      do { crypto.getRandomValues(randomWord); } while (randomWord[0] >= ceiling);
      return randomWord[0] % count;
    }
    return Math.floor(Math.random() * count);
  }
  let currentSolid = -1, solidMatches = null;
  function setSolidButtons(disabled) {
    for (const id of ['solidPrev','solidForward','solidNext']) $(id).disabled = disabled;
  }
  function renderSolid(index) {
    currentSolid = index;
    if (index < 0) {
      const noData = !polyCatalog.length;
      $('solidName').textContent = noData ? 'No polyhedron available' : 'No matching polyhedron';
      $('solidName').title = '';
      $('solidTreeCount').textContent = '—';
      $('solidNetCount').textContent = '—';
      $('solidVEF').textContent = '—';
      $('solidCard').dataset.source = '';
      solidViewer.setModel(null);
      setSolidButtons(true);
      return;
    }
    setSolidButtons(false);
    const m = polyCatalog[index];
    const trees = exactCount(m.trees ?? m.treeCount);
    const nets = exactCount(m.nets ?? m.netCount);
    const nameEl = $('solidName');
    nameEl.textContent = m.name;
    nameEl.title = m.name;
    nameEl.classList.toggle('name-long',m.name.length>48);
    nameEl.classList.toggle('name-very-long',m.name.length>78);
    $('solidTreeCount').textContent = trees;
    $('solidNetCount').textContent = nets;
    $('solidNetBadge').title = `${trees} spanning trees / ${nets} symmetry-reduced`;
    $('solidVEF').textContent = `V${m.points?.length ?? 0} E${m.edges?.length ?? 0} F${m.faces?.length ?? 0}`;
    $('solidCard').dataset.source = m.source;
    requestAnimationFrame(fitNetNumbers);
    solidViewer.setModel(m);
  }
  function randomSolid() {
    if (solidMatches) renderSolid(solidMatches.length ? solidMatches[uniformIndex(solidMatches.length)] : -1);
    else renderSolid(uniformIndex(polyCatalog.length));
  }
  function moveSolid(delta) {
    if (solidMatches) {
      const n = solidMatches.length;
      if (!n) return;
      const position = solidMatches.indexOf(currentSolid);
      renderSolid(solidMatches[((position < 0 ? 0 : position) + delta + n) % n]);
    } else if (polyCatalog.length) renderSolid((currentSolid + delta + polyCatalog.length) % polyCatalog.length);
  }
  function refreshSolidSearch() {
    // Synchronous live search: current result and navigation scope both update.
    const query = normalizeQuery($('solidSearch').value);
    if (!query) {
      solidMatches = null;
      renderSolid(currentSolid >= 0 ? currentSolid : uniformIndex(polyCatalog.length));
      return;
    }
    solidMatches = findMatches(polyCatalog, query, scorePoly, (m,s)=>fuzzyMatch(`${m.name} ${m.id} ${m.key||''}`,s));
    renderSolid(solidMatches[0] ?? -1);
  }
  $('solidSearch').addEventListener('input', refreshSolidSearch);
  $('solidSearch').addEventListener('search', refreshSolidSearch);
  $('solidPrev').addEventListener('click',()=>moveSolid(-1));
  $('solidForward').addEventListener('click',()=>moveSolid(1));
  $('solidNext').addEventListener('click',randomSolid);
  const saturationControl=$('solidSaturation'),opacityControl=$('solidOpacity');
  saturationControl?.addEventListener('input',e=>solidViewer.setSaturation(e.target.value));
  opacityControl?.addEventListener('input',e=>solidViewer.setOpacity(e.target.value));
  if(saturationControl)solidViewer.saturation=Number(saturationControl.value);
  if(opacityControl)solidViewer.opacity=Number(opacityControl.value);
  const netBadge=$('solidNetBadge');
  if(netBadge)new ResizeObserver(()=>fitNetNumbers()).observe(netBadge);
  randomSolid();


  /* ---------- 3D reflexive-polytope catalogue ---------- */
  const reflexiveCount=reflexiveLibrary?.count||0;
  const reflexiveModels=new Array(reflexiveCount);
  const reflexivePredecessors=Array.from({length:reflexiveCount},()=>[]);
  const reflexiveSymmetryNames=['C1','C2','C3','C2×C2','C2×C2×C2','S3','D4','D6','D4×C2','D6×C2','S4','S4×C2'];
  // Exact minimizers are precomputed in the lightweight reflexive-sl3.js asset.
  const reflexiveSL3=window.REFLEXIVE_SL3;
  const optimizedReflexiveModels=new Array(reflexiveCount);

  if(reflexiveLibrary){
    for(let i=0;i<reflexiveCount;i++){
      const successors=reflexiveLibrary.getSuccessors(i);
      for(let j=0;j<successors.length;j++)reflexivePredecessors[successors[j]].push(i);
    }
  }
  function reflexiveModel(index){
    if(!reflexiveLibrary||index<0||index>=reflexiveCount)return null;
    if(!reflexiveModels[index])reflexiveModels[index]=reflexiveLibrary.getPoly(index);
    return reflexiveModels[index];
  }
  function gcdPair(a,b){
    a=Math.abs(Math.trunc(a));b=Math.abs(Math.trunc(b));
    while(b){const t=a%b;a=b;b=t;}
    return a;
  }
  function gcd3(a,b,c){return gcdPair(gcdPair(a,b),c)||1;}
  function pointKey(x,y,z){return `${x},${y},${z}`;}
  function ensureReflexiveGeometry(m){
    if(m._reflexiveGeometry)return m._reflexiveGeometry;
    const planes=[];
    for(const face of m.faces){
      const a=m.points[face[0]],b=m.points[face[1]],c=m.points[face[2]];
      let n=cross(sub(b,a),sub(c,a)).map(Math.round);
      let d=dot(n,a);
      if(d<0){n=mul(n,-1);d=-d;}
      const g=gcd3(n[0],n[1],n[2]);
      n=n.map(x=>x/g);d=Math.round(d/g);
      planes.push({n,d});
    }
    const edgeSet=new Set(),edges=[];
    for(const face of m.faces){
      for(let i=0;i<face.length;i++){
        const a=face[i],b=face[(i+1)%face.length];
        const x=Math.min(a,b),y=Math.max(a,b),key=`${x},${y}`;
        if(edgeSet.has(key))continue;
        edgeSet.add(key);edges.push([x,y]);
      }
    }
    m.edges=edges;
    m._reflexiveGeometry={planes,edges};
    return m._reflexiveGeometry;
  }
  function reflexiveLatticeStats(m){
    if(m._reflexiveLatticeStats)return m._reflexiveLatticeStats;
    const {planes,edges}=ensureReflexiveGeometry(m);
    const vertexSet=new Set(m.points.map(p=>pointKey(p[0],p[1],p[2])));
    const edgePointSet=new Set();
    for(const [ia,ib] of edges){
      const a=m.points[ia],b=m.points[ib];
      const dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2];
      const g=gcd3(dx,dy,dz);
      for(let t=1;t<g;t++){
        edgePointSet.add(pointKey(a[0]+(dx/g)*t,a[1]+(dy/g)*t,a[2]+(dz/g)*t));
      }
    }
    const min=[0,1,2].map(j=>Math.min(...m.points.map(p=>p[j])));
    const max=[0,1,2].map(j=>Math.max(...m.points.map(p=>p[j])));
    const counts=[0,0,0,0],latticePoints=[];
    for(let x=min[0];x<=max[0];x++)for(let y=min[1];y<=max[1];y++)for(let z=min[2];z<=max[2];z++){
      let inside=true,boundary=false;
      for(const plane of planes){
        const value=plane.n[0]*x+plane.n[1]*y+plane.n[2]*z;
        if(value>plane.d){inside=false;break;}
        if(value===plane.d)boundary=true;
      }
      if(!inside)continue;
      const key=pointKey(x,y,z);
      const type=vertexSet.has(key)?0:edgePointSet.has(key)?1:boundary?2:3;
      counts[type]++;
      latticePoints.push([x,y,z,type]);
    }
    m._reflexiveLatticeStats={counts,total:counts.reduce((a,b)=>a+b,0),points:latticePoints};
    return m._reflexiveLatticeStats;
  }
  function buildReflexiveLatticePoints(m){
    if(m._displayLatticePoints)return m._displayLatticePoints;
    const stats=reflexiveLatticeStats(m);
    const vertexIndex=new Map(m.points.map((p,i)=>[pointKey(p[0],p[1],p[2]),i]));
    const insideIndex=new Map();
    const display=[];
    for(const p of stats.points){
      const key=pointKey(p[0],p[1],p[2]);
      insideIndex.set(key,display.length);
      const type=p[3]===0?0:p[3]===3?2:1;
      display.push([p[0],p[1],p[2],type,vertexIndex.get(key)??-1]);
    }

    // The a × b × c grid is the smallest axis-aligned integer box
    // containing the polytope, endpoints included; do not add an outer layer.
    const min=[0,1,2].map(j=>Math.min(...m.points.map(p=>p[j])));
    const max=[0,1,2].map(j=>Math.max(...m.points.map(p=>p[j])));
    // EXACT Chebyshev-distance <= 1 dilation of V, clipped to that box.
    // Unlike expanded supporting planes this has no false-positive grid points.
    const exterior=new Set();
    for(const p of stats.points){
      for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){
        const x=p[0]+dx,y=p[1]+dy,z=p[2]+dz;
        if(x<min[0]||x>max[0]||y<min[1]||y>max[1]||z<min[2]||z>max[2])continue;
        const key=pointKey(x,y,z);
        if(!insideIndex.has(key)&&!exterior.has(key)){
          exterior.add(key);
          display.push([x,y,z,3,-1]);
        }
      }
    }
    // Distance exactly one in Euclidean space: the six orthogonal unit
    // neighbours. Positive directions avoid duplicate thin connections.
    const latticeLinks=[];
    for(const p of stats.points){
      const a=insideIndex.get(pointKey(p[0],p[1],p[2]));
      for(let axis=0;axis<3;axis++){
        const q=p.slice(0,3);q[axis]++;
        const b=insideIndex.get(pointKey(q[0],q[1],q[2]));
        if(b!==undefined)latticeLinks.push([a,b]);
      }
    }
    m.latticeLinks=latticeLinks;
    m._displayLatticePoints=display;
    return display;
  }
  function ehrhartPolynomial(M){
    const pieces=[];
    const coefficients=[[M-3,3,3],[M-3,2,2],[M+9,6,1],[1,1,0]];
    for(const [numerator,denominator,power] of coefficients){
      if(numerator===0)continue; // Never display zero terms.
      const divisor=gcdPair(numerator,denominator);
      const n=Math.abs(numerator)/divisor,d=denominator/divisor;
      const sign=numerator<0?'-':pieces.length?'+':'';
      // Parenthesize *every* noninteger coefficient; ±1 is implicit on k terms.
      const coefficient=power>0&&n===d?'':d===1?String(n):`(${n}/${d})`;
      const monomial=power===0?'':power===1?'k':`k^${power}`;
      pieces.push(`${sign}${coefficient}${monomial}`);
    }
    return pieces.join('')||'0';
  }
  function reflexiveDisplayStats(index,m){
    if(m._reflexiveDisplayStats)return m._reflexiveDisplayStats;
    const MStats=reflexiveLatticeStats(m);
    const dualIndex=reflexiveLibrary.dual[index];
    const dualModel=reflexiveModel(dualIndex);
    const NStats=reflexiveLatticeStats(dualModel);
    const successors=reflexiveLibrary.getSuccessors(index);
    const predecessors=reflexivePredecessors[index];
    const allTri=m.faces.every(face=>face.length===3);
    const degree=new Array(m.points.length).fill(0);
    for(const [a,b] of m.edges){degree[a]++;degree[b]++;}
    const simple=degree.every(d=>d===3);
    let smooth=allTri;
    if(smooth){
      for(const face of m.faces){
        const a=m.points[face[0]],b=m.points[face[1]],c=m.points[face[2]];
        if(Math.abs(dot(a,cross(b,c)))!==1){smooth=false;break;}
      }
    }
    const maxmin=successors.length===0&&predecessors.length===0?'Maximal & Minimal':
      successors.length===0?'Maximal':predecessors.length===0?'Minimal':'No';
    m._reflexiveDisplayStats={
      id:index+1,
      M:MStats.total,MCounts:MStats.counts,
      N:NStats.total,NCounts:NStats.counts,
      volume:2*MStats.total-6,
      ehrhart:ehrhartPolynomial(MStats.total),
      picard:m.picard,cor:m.cor,
      symmetry:reflexiveSymmetryNames[m.symmetry]||'—',
      sections:reflexiveLibrary.getSections(index),
      simp:allTri?'Simplicial':simple?'Simple':'No',
      maxmin,
      smooth,
      terminal:MStats.counts[1]+MStats.counts[2]===0,
      selfdual:dualIndex===index
    };
    return m._reflexiveDisplayStats;
  }
  function renderReflexiveStats(stats){
    const columns=[
      [
        ['ID',String(stats.id)],
        ['M Lattice',`${stats.M} (${stats.MCounts.join('+')})`],
        ['N Lattice',`${stats.N} (${stats.NCounts.join('+')})`],
        ['Volume',String(stats.volume)],
        ['Ehrhart |kP∩M|',stats.ehrhart]
      ],
      [
        ['Picard',String(stats.picard)],
        ['Cor',String(stats.cor)],
        ['Symmetry',stats.symmetry],
        ['Reflsection',stats.sections.length?stats.sections.join(', '):'—']
      ],
      [
        ['Simp',stats.simp],
        ['MaxMin',stats.maxmin],
        ['Smooth',stats.smooth?'True':'False'],
        ['Terminal',stats.terminal?'True':'False'],
        ['Selfdual',stats.selfdual?'True':'False']
      ]
    ];
    const ids=['reflexiveStatsA','reflexiveStatsB','reflexiveStatsC'];
    for(let i=0;i<3;i++){
      const column=$(ids[i]);column.replaceChildren();
      for(const [label,value] of columns[i]){
        const row=document.createElement('div');row.className='reflexive-stat';
        const small=document.createElement('small');small.textContent=label;
        const code=document.createElement('code');code.textContent=value;
        row.append(small,code);column.append(row);
      }
    }
  }
  function optimizedReflexive(index){
    if(!reflexiveSL3||reflexiveSL3.count!==reflexiveCount)return null;
    if(optimizedReflexiveModels[index])return optimizedReflexiveModels[index];
    const matrix=reflexiveSL3.matrixAt(index),original=reflexiveModel(index);
    if(!matrix||!original)return null;
    const determinant=matrix[0]*(matrix[4]*matrix[8]-matrix[5]*matrix[7])
      -matrix[1]*(matrix[3]*matrix[8]-matrix[5]*matrix[6])
      +matrix[2]*(matrix[3]*matrix[7]-matrix[4]*matrix[6]);
    if(determinant!==1)throw new Error(`Invalid SL(3,Z) matrix for ID ${index+1}`);
    const points=original.points.map(p=>[
      matrix[0]*p[0]+matrix[1]*p[1]+matrix[2]*p[2],
      matrix[3]*p[0]+matrix[4]*p[1]+matrix[5]*p[2],
      matrix[6]*p[0]+matrix[7]*p[1]+matrix[8]*p[2]
    ]);
    // Keep the original face incidence unchanged, and derive fresh lattice,
    // plane and edge information from the transformed integer coordinates.
    const transformed={id:original.id,points,faces:original.faces,reflexive:true};
    transformed.latticePoints=buildReflexiveLatticePoints(transformed);
    optimizedReflexiveModels[index]=transformed;
    return transformed;
  }
  const reflexiveViewer=reflexiveCount?new PolyViewer($('reflexiveCanvas')):null;
  if(reflexiveViewer)window.CYBCAT_REFLEXIVE_VIEWER=reflexiveViewer;
  let currentReflexive=-1,reducedReflexive=true;
  const sl3Button=$('reflexiveSL3');
  function showSL3State(){
    if(!sl3Button)return;
    sl3Button.setAttribute('aria-pressed',String(reducedReflexive));
    sl3Button.title=reducedReflexive?'Restore original lattice coordinates':'SL(3,ℤ): minimize vertex-pair diameter, then box a ≥ b ≥ c';
    sl3Button.setAttribute('aria-label',reducedReflexive?'Restore original integer coordinates':'Optimize lattice using SL(3,Z)');
  }
  function setReflexiveButtons(disabled){
    for(const id of ['reflexivePrev','reflexiveForward','reflexiveNext','reflexiveMirror','reflexiveUp','reflexiveDown','reflexiveSL3'])$(id).disabled=disabled;
    if(!disabled){
      $('reflexiveUp').disabled=reflexiveLibrary.getSuccessors(currentReflexive).length===0;
      $('reflexiveDown').disabled=reflexivePredecessors[currentReflexive].length===0;
    }
  }
  function renderReflexive(index){
    currentReflexive=index;
    showSL3State();
    if(index<0||index>=reflexiveCount){
      for(const id of ['reflexiveStatsA','reflexiveStatsB','reflexiveStatsC'])$(id).replaceChildren();
      reflexiveViewer?.setModel(null);
      setReflexiveButtons(true);
      return;
    }
    const m=reflexiveModel(index);
    const stats=reflexiveDisplayStats(index,m);
    m.reflexive=true;
    m.latticePoints=buildReflexiveLatticePoints(m);
    renderReflexiveStats(stats);
    // Changing the polytope must not change the user-selected reduction mode.
    const visibleModel=reducedReflexive?optimizedReflexive(index):m;
    reflexiveViewer?.setModel(visibleModel||m);
    setReflexiveButtons(false);
  }
  function toggleSL3Reduction(){
    if(currentReflexive<0||currentReflexive>=reflexiveCount)return;
    const next=!reducedReflexive;
    const model=next?optimizedReflexive(currentReflexive):reflexiveModel(currentReflexive);
    if(!model)return;
    if(!next){model.reflexive=true;model.latticePoints=buildReflexiveLatticePoints(model);}
    reducedReflexive=next;
    // Preserve camera angles when comparing the two lattice coordinate systems.
    reflexiveViewer?.setModel(model,true);
    showSL3State();
  }
  function setReflexiveSearch(value){$('reflexiveSearch').value=value==null?'':String(value);}
  function randomReflexive(){
    if(!reflexiveCount)return;
    setReflexiveSearch('');
    renderReflexive(uniformIndex(reflexiveCount));
  }
  function moveReflexive(delta){
    if(!reflexiveCount)return;
    setReflexiveSearch('');
    renderReflexive((currentReflexive<0?0:(currentReflexive+delta+reflexiveCount)%reflexiveCount));
  }
  function mirrorReflexive(){
    if(currentReflexive<0)return;
    const target=reflexiveLibrary.dual[currentReflexive];
    setReflexiveSearch(target+1);
    renderReflexive(target);
  }
  function stepReflexive(direction){
    if(currentReflexive<0)return;
    const choices=direction>0?reflexiveLibrary.getSuccessors(currentReflexive):reflexivePredecessors[currentReflexive];
    if(!choices.length)return;
    const target=choices[uniformIndex(choices.length)];
    setReflexiveSearch(target+1);
    renderReflexive(target);
  }
  function refreshReflexiveSearch(){
    const raw=$('reflexiveSearch').value.trim();
    if(!raw){if(currentReflexive<0)randomReflexive();return;}
    if(!/^\d+$/.test(raw)){renderReflexive(-1);return;}
    const id=Number(raw);
    renderReflexive(Number.isSafeInteger(id)&&id>=1&&id<=reflexiveCount?id-1:-1);
  }
  $('reflexiveSearch').addEventListener('input',refreshReflexiveSearch);
  $('reflexiveSearch').addEventListener('search',refreshReflexiveSearch);
  $('reflexivePrev').addEventListener('click',()=>moveReflexive(-1));
  $('reflexiveForward').addEventListener('click',()=>moveReflexive(1));
  $('reflexiveNext').addEventListener('click',randomReflexive);
  $('reflexiveMirror').addEventListener('click',mirrorReflexive);
  $('reflexiveUp').addEventListener('click',()=>stepReflexive(1));
  $('reflexiveDown').addEventListener('click',()=>stepReflexive(-1));
  sl3Button?.addEventListener('click',toggleSL3Reduction);
  const reflexiveSaturation=$('reflexiveSaturation'),reflexiveOpacity=$('reflexiveOpacity');
  reflexiveSaturation?.addEventListener('input',e=>reflexiveViewer?.setSaturation(e.target.value));
  reflexiveOpacity?.addEventListener('input',e=>reflexiveViewer?.setOpacity(e.target.value));
  if(reflexiveViewer){
    reflexiveViewer.saturation=Number(reflexiveSaturation?.value??50);
    reflexiveViewer.opacity=Number(reflexiveOpacity?.value??33);
    randomReflexive();
  }

  window.CYBCAT_CATALOG_INFO = {total:polyCatalog.length,offline:offlineCatalogSize,...(polyLibrary.sourceCounts||{})};
  updateTheme();
})();
