'use strict';
function id(n){return 'LP'+String(n).padStart(6,'0')}
function gcd(a,b){a=Math.abs(Math.trunc(a)); b=Math.abs(Math.trunc(b)); while(b){let t=a%b; a=b; b=t} return a}
function lcm(a,b){return a/gcd(a,b)*b}
function det(a,b){return a[0]*b[1]-a[1]*b[0]}
function cross(o,a,b){return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])}
function hull(P){P=[...new Set(P.map(p=>p.join(',')))].map(s=>s.split(',').map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]); if(P.length<=1)return P; let L=[],U=[]; for(const p of P){while(L.length>1&&cross(L.at(-2),L.at(-1),p)<=0)L.pop();L.push(p)} for(let i=P.length-1;i>=0;i--){let p=P[i];while(U.length>1&&cross(U.at(-2),U.at(-1),p)<=0)U.pop();U.push(p)} return L.slice(0,-1).concat(U.slice(0,-1))}
function bbox(P){return{minX:Math.min(...P.map(p=>p[0])),maxX:Math.max(...P.map(p=>p[0])),minY:Math.min(...P.map(p=>p[1])),maxY:Math.max(...P.map(p=>p[1]))}}
function polygonGridInfo(P){let v=hull(P), b=bbox(v), w=Math.ceil(b.maxX)-Math.floor(b.minX)+1, h=Math.ceil(b.maxY)-Math.floor(b.minY)+1, cells=Math.max(1,w)*Math.max(1,h); return {bbox:b,width:w,height:h,cells,tooLarge:w>LARGE_GRID_SIDE_LIMIT||h>LARGE_GRID_SIDE_LIMIT||cells>LARGE_GRID_CELL_LIMIT}}
function largeGridMessage(info){return `grid ${info.width}×${info.height} (${info.cells.toLocaleString()} lattice positions) exceeds the ${LARGE_GRID_SIDE_LIMIT}×${LARGE_GRID_SIDE_LIMIT} display/copy limit`}
function maybeLargeGridNote(info){return info&&info.tooLarge?`<p class="small-note">Large-grid guard: ${largeGridMessage(info)}. Grid lines, lattice-point enumeration and copy-to-drawn-grid are skipped for this derived polygon.</p>`:''}
function norm(P){let v=hull(P); if(!v.length)return []; let b=bbox(v); return hull(v.map(p=>[p[0]-b.minX,p[1]-b.minY]))}
function cyc(P){let v=norm(P), n=v.length, best=null, bestv=null; for(const arr of [v,v.slice().reverse()]) for(let i=0;i<n;i++){let q=arr.slice(i).concat(arr.slice(0,i)), s=JSON.stringify(q); if(best===null||s<best){best=s;bestv=q}} return bestv||v}
function keyOf(P){return cyc(P).map(p=>p.join(',')).join(';')}
function exactKeyOf(P){return hull(P).map(p=>p.join(',')).join(';')}
function area2(v){return Math.abs(v.reduce((s,p,i)=>s+det(p,v[(i+1)%v.length]),0))}
function signedArea2(v){return v.reduce((s,p,i)=>s+det(p,v[(i+1)%v.length]),0)}
function edges(v){return v.map((p,i)=>{let q=v[(i+1)%v.length]; return gcd(q[0]-p[0],q[1]-p[1])})}
function cycleSeq(a){let n=a.length,b=null; for(const arr of [a,a.slice().reverse()]) for(let i=0;i<n;i++){let s=arr.slice(i).concat(arr.slice(0,i)).join(','); if(b===null||s<b)b=s} return b}
function width(v,m){let vals=v.map(p=>m[0]*p[0]+m[1]*p[1]); return Math.max(...vals)-Math.min(...vals)}
function normals(v){return v.map((p,i)=>{let q=v[(i+1)%v.length],dx=q[0]-p[0],dy=q[1]-p[1],l=gcd(dx,dy);return[dy/l,-dx/l]})}
function mults(N){return N.map((n,i)=>Math.abs(det(N[(i-1+N.length)%N.length],n)))}
function dualDirs(v,S){
  // Exact bounded enumeration of primitive dual directions m with width_m(P) <= S.
  // If d1,d2 are independent vertex differences, then |<d_i,m>| <= S.  Cramer's
  // rule therefore gives a rigorous integer bounding box for m; no floating polar
  // vertices or tolerance decisions enter the enumeration.
  v=hull(v); S=Math.max(0,Math.trunc(S));
  let D=[];
  for(let i=0;i<v.length;i++)for(let j=i+1;j<v.length;j++){
    let d=[v[i][0]-v[j][0],v[i][1]-v[j][1]];
    if(d[0]||d[1])D.push(d)
  }
  let best=null;
  for(let i=0;i<D.length;i++)for(let j=i+1;j<D.length;j++){
    let q=Math.abs(det(D[i],D[j]));
    if(q&&(!best||q>best.q))best={d1:D[i],d2:D[j],q}
  }
  if(!best)return [];
  let {d1,d2,q}=best;
  let A=Math.ceil(S*(Math.abs(d1[1])+Math.abs(d2[1]))/q);
  let B=Math.ceil(S*(Math.abs(d1[0])+Math.abs(d2[0]))/q);
  let R=[];
  for(let a=-A;a<=A;a++)for(let b=-B;b<=B;b++){
    if(!(a||b)||gcd(a,b)!==1)continue;
    if(width(v,[a,b])<=S)R.push([a,b])
  }
  return R
}
function canonicalDualDirection(m){let a=m[0],b=m[1]; if(a<0||(a===0&&b<0)){a=-a;b=-b} if(Object.is(a,-0))a=0; if(Object.is(b,-0))b=0; return [a,b]}
function latticeWidthData(v){
  v=hull(v);
  let facet=normals(v), upper=Math.min(...facet.map(n=>width(v,n)));
  let dirs=dualDirs(v,upper), lw=upper;
  for(const m of dirs)lw=Math.min(lw,width(v,m));
  let seen=new Set(), minimum=[];
  for(const m of dirs)if(width(v,m)===lw){let c=canonicalDualDirection(m),k=c.join(','); if(!seen.has(k)){seen.add(k);minimum.push(c)}}
  minimum.sort((a,b)=>Math.abs(a[0])+Math.abs(a[1])-Math.abs(b[0])-Math.abs(b[1])||a[0]-b[0]||a[1]-b[1]);
  return {width:lw,directions:minimum,enumerationBound:upper,candidateCount:dirs.length}
}
function optDisplay(v){
  v=hull(v);
  let S0=Math.max(width(v,[1,0]),width(v,[0,1])), D=dualDirs(v,S0), W=new Map(D.map(m=>[m.join(','),width(v,m)]));
  let bestScore=null,bv=null,basis=null;
  function better(sc,best){if(!best)return true; for(let i=0;i<5;i++){if(sc[i]!==best[i])return sc[i]<best[i]} return sc[5]<best[5]}
  for(const m of D)for(const n of D){let de=m[0]*n[1]-m[1]*n[0]; if(Math.abs(de)!==1)continue; let w1=W.get(m.join(',')),w2=W.get(n.join(',')); let tv=v.map(p=>[m[0]*p[0]+m[1]*p[1],n[0]*p[0]+n[1]*p[1]]); let cv=cyc(tv), sc=[Math.max(w1,w2),w1,w2,w1*w2,w1+w2,JSON.stringify(cv)]; if(better(sc,bestScore)){bestScore=sc;bv=cv;basis=[m,n]}}
  return {v:bv||cyc(v),basis:basis||[[1,0],[0,1]],score:bestScore}
}
function optKey(v){return optDisplay(v).v.map(p=>p.join(',')).join(';')}
function onbd(p,v){return v.some((a,i)=>{let b=v[(i+1)%v.length];return cross(a,b,p)===0&&Math.min(a[0],b[0])<=p[0]&&p[0]<=Math.max(a[0],b[0])&&Math.min(a[1],b[1])<=p[1]&&p[1]<=Math.max(a[1],b[1])})}
function inside(p,v){let c=false; for(let i=0,j=v.length-1;i<v.length;j=i++){let a=v[i],b=v[j]; if(((a[1]>p[1])!==(b[1]>p[1]))&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])c=!c} return c}
function clonePoints(P){return P.map(p=>[p[0],p[1]])}
function clonePointData(P){return {bd:clonePoints(P.bd),inn:clonePoints(P.inn),all:clonePoints(P.all)}}
function latticePoints(v){let key=exactKeyOf(v), hit=MEMO.latticePoints.get(key); if(hit)return clonePointData(hit); let hv=hull(v), b=bbox(hv), bd=[], inn=[]; for(let x=b.minX;x<=b.maxX;x++)for(let y=b.minY;y<=b.maxY;y++){let p=[x,y]; if(onbd(p,hv))bd.push(p); else if(inside(p,hv))inn.push(p)} let out={bd,inn,all:bd.concat(inn)}; MEMO.latticePoints.set(key,clonePointData(out)); return out}
function primitiveContent(v){let p0=v[0],g=0; for(const p of v.slice(1)){g=gcd(g,p[0]-p0[0]);g=gcd(g,p[1]-p0[1])} return g||1}
function primitivePolygonData(v){v=hull(v); let c=primitiveContent(v); if(c<=1)return {poly:v,content:1,changed:false,anchor:v[0]}; let a=v[0], poly=cyc(v.map(p=>[(p[0]-a[0])/c,(p[1]-a[1])/c])); return {poly,content:c,changed:true,anchor:a}}
function primitivePolygon(v){return primitivePolygonData(v).poly}
function centralFromPoints(P){let S=new Set(P.map(p=>p.join(','))), b=bbox(P), sx=b.minX+b.maxX, sy=b.minY+b.maxY; return P.every(p=>S.has((sx-p[0])+','+(sy-p[1])))}
function stats(v){let key=exactKeyOf(v), hit=MEMO.stats.get(key); if(hit)return {...hit,v:clonePoints(hit.v),e:hit.e.slice(),Ns:clonePoints(hit.Ns),mm:hit.mm.slice(),box:hit.box.slice()}; v=hull(v); let e=edges(v), Vol=area2(v), B=e.reduce((a,b)=>a+b,0), I=(Vol-B+2)/2, N=B+I, Ns=normals(v), mm=mults(Ns), b=bbox(v), lw=latticeWidthData(v).width; let out={v,V:v.length,e,edge:cycleSeq(e),Vol,B,I,N,area:Vol/2,Ns,mm,mult:cycleSeq(mm),smooth:mm.every(x=>x===1),content:primitiveContent(v),lw,box:[b.maxX-b.minX,b.maxY-b.minY],boxSq:Math.max(b.maxX-b.minX,b.maxY-b.minY),primEdges:e.filter(x=>x===1).length,maxEdge:Math.max(...e),minEdge:Math.min(...e)}; MEMO.stats.set(key,{...out,v:clonePoints(out.v),e:out.e.slice(),Ns:clonePoints(out.Ns),mm:out.mm.slice(),box:out.box.slice()}); return out}
function signatureFromStats(s){return `${s.Vol}|${s.V}|${s.B}|${s.I}|${s.edge}|${s.content}|${s.mult}`}
function codegree(s){return s.I>0?1:(s.B>3?2:3)}
function parseScalarInput(x){
  if(x===''||x===undefined||x===null)return null;
  if(typeof x==='number')return Number.isFinite(x)?x:null;
  let t=String(x).trim().replace(/−/g,'-');
  if(!t)return null;
  if(/^[-+]?\d+(?:\.\d+)?\s*\/\s*[-+]?\d+(?:\.\d+)?$/.test(t)){
    let [a,b]=t.split('/').map(z=>Number(z.trim()));
    return Number.isFinite(a)&&Number.isFinite(b)&&b!==0?a/b:null;
  }
  let n=Number(t); return Number.isFinite(n)?n:null;
}
function parseIntegerInput(x){let n=parseScalarInput(x); return n===null||!Number.isFinite(n)?null:Math.trunc(n)}
function nearly(a,b){return Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(a),Math.abs(b))}
function cmp(a,op,b){let v=parseScalarInput(b); if(v===null)return true; op=op||'='; if(op==='≤'||op==='<=')return a<=v||nearly(a,v); if(op==='≥'||op==='>=')return a>=v||nearly(a,v); if(op==='<')return a<v&&!nearly(a,v); if(op==='>')return a>v&&!nearly(a,v); if(op==='≠'||op==='!='||op==='ne')return !nearly(a,v); return nearly(a,v)}
function hasVal(x){return x!==''&&x!==undefined&&x!==null}
function boolFilter(actual,want){return want===''||want===undefined||want===null||String(actual)===String(want)}
function ehrhartValueFromRow(r,k){return (r[MIDX.Vol]*k*k+r[MIDX.B]*k)/2+1}
function hStar1FromRow(r){return r[MIDX.N]-3}
function rowCodegree(r){return r[MIDX.I]>0?1:(r[MIDX.B]>3?2:3)}
function rowDegree(r){return 3-rowCodegree(r)}
function rowMultArray(r){return String(r[MIDX.mult]||'').split(',').filter(Boolean).map(Number)}
function cycleMatches(hay,needle){return !hasVal(needle)||String(hay||'').replace(/\s+/g,'')===String(needle).replace(/\s+/g,'')}
function srcName(bit){return bit===3?'S+L':bit===2?'LDP':bit===1?'small':'unknown'}
function unpackDict(packed,arr){return typeof packed==='string'?packed.split('|'):(arr||[])}
function decodeMetaChunk(arr){if(MANIFEST&&(MANIFEST.metaEncoding==='uvarintB64v1'||MANIFEST.metaEncoding==='uvarintBinV2'))return decodeMetaVar(arr,MANIFEST.metaRowLength||20); if(typeof arr!=='string')return arr; arr=arr.trim(); if(!arr)return []; return arr.split(';').map(r=>r.split(',').map(x=>parseInt(x,36)))}
function unflatVertices(a){if(!a.length||Array.isArray(a[0]))return a; let v=[]; for(let i=0;i<a.length;i+=2)v.push([a[i],a[i+1]]); return v}
function decodeSource(code,srcFlag){if(Array.isArray(code)){if(srcFlag===1)return [`S:v${code[0]}:${code[1]}`]; if(srcFlag===2){let deg=String(code[4])+(code[5]&&code[5]!==1?`/${code[5]}`:''); return [`L:i${code[0]}:ID${code[1]}:order${code[2]}:degree${deg}`]}} if(typeof code==='string')return [code]; return []}
function parseB36(x){return x&&x[0]==='-'?-parseInt(x.slice(1),36):parseInt(x,36)}
function b64Bytes(txt){let bin=atob(txt.trim()), a=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i); return a}
function rawBytes(raw){if(raw instanceof Uint8Array)return raw; if(raw instanceof ArrayBuffer)return new Uint8Array(raw); return b64Bytes(raw)}
function readUVar(a,st){let x=0,s=0,b; do{b=a[st.i++]; if(b===undefined)throw new Error('Truncated varint data'); x+=(b&127)*2**s; s+=7; if(s>53)throw new Error('Varint exceeds JavaScript safe integer range')}while(b&128); return x}
function unzig(x){return (x>>>1)^-(x&1)}
function decodeMetaVar(raw,rowLen){let a=rawBytes(raw), st={i:0}, rows=readUVar(a,st), out=[]; for(let r=0;r<rows;r++){let row=[]; for(let j=0;j<rowLen;j++)row.push(readUVar(a,st)); out.push(row)} return out}
function decodeDataChunkVar(raw){let a=rawBytes(raw), st={i:0}, rows=readUVar(a,st), out=[]; for(let r=0;r<rows;r++){let nv=readUVar(a,st), flat=[]; for(let i=0;i<nv;i++)flat.push(unzig(readUVar(a,st))); let ns=readUVar(a,st), src=[]; for(let i=0;i<ns;i++)src.push(unzig(readUVar(a,st))); out.push([flat,src])} return out}
function decodePackedDataLine(line){let [vpart,spart='']=line.split('|'), flat=vpart?vpart.split('.').filter(Boolean).map(parseB36):[], src=spart?spart.split('.').filter(Boolean).map(parseB36):[]; return [flat,src]}
function initDrawGrid(){let host=$('#point-grid'); if(host&&!host.dataset.ready)makeGrid()}
async function load(){initDrawGrid(); MANIFEST=await fetch('data/manifest.json').then(r=>r.json()); let sum=$('#summary'); if(sum){sum.innerHTML=''; sum.hidden=true} EDGE_DICT=unpackDict(MANIFEST.edgeDictPacked,MANIFEST.edgeDict); MULT_DICT=unpackDict(MANIFEST.multDictPacked,MANIFEST.multDict); let loaded=0; for(const ch of MANIFEST.metaChunks){let raw=await fetch('data/'+ch.file).then(r=>MANIFEST.metaEncoding==='uvarintBinV2'?r.arrayBuffer():((MANIFEST.metaEncoding==='base36csvtxt'||MANIFEST.metaEncoding==='uvarintB64v1')?r.text():r.json())); let arr=decodeMetaChunk(raw); for(const row of arr){if(MANIFEST.metaPacked){row[MIDX.edge]=EDGE_DICT[row[MIDX.edge]]; row[MIDX.mult]=MULT_DICT[row[MIDX.mult]]} META.push(row); BY_ID.set(row[MIDX.id],row); let sig=[row[MIDX.Vol],row[MIDX.V],row[MIDX.B],row[MIDX.I],row[MIDX.edge],row[MIDX.content],row[MIDX.mult]].join('|'); if(!BUCKET.has(sig))BUCKET.set(sig,[]); BUCKET.get(sig).push(row)} loaded+=arr.length; $('#load-status').textContent=`Loaded ${loaded.toLocaleString()} metadata rows...`; await new Promise(r=>setTimeout(r,0)) } $('#load-status').textContent=`Ready: ${META.length.toLocaleString()} searchable records.`; renderMath(document.body); $('#search-status').textContent='Ready. Enter at least one search condition. Display limit defaults to 100.'}
async function fetchRecord(row){if(row&&row.__virtual)return {rid:row[MIDX.id],v:clonePoints(row.__vertices),src:['drawn polygon; no database record'],srcCount:0,virtual:true}; let cno=row[MIDX.cno], off=row[MIDX.off], ch=MANIFEST.dataChunks[cno-1], f=ch.file; if(!DATA_CACHE.has(f)){if(MANIFEST.dataPacked==='flatSourceTxtV2'){let txt=await fetch('data/'+f).then(r=>r.text()); DATA_CACHE.set(f,txt.trim()?txt.trim().split('\n'):[])}else if(MANIFEST.dataPacked==='flatSourceVarB64V3'){let txt=await fetch('data/'+f).then(r=>r.text()); DATA_CACHE.set(f,decodeDataChunkVar(txt))}else if(MANIFEST.dataPacked==='flatSourceVarBinV4'){let buf=await fetch('data/'+f).then(r=>r.arrayBuffer()); DATA_CACHE.set(f,decodeDataChunkVar(buf))}else{DATA_CACHE.set(f, await fetch('data/'+f).then(r=>r.json()))}} let raw=DATA_CACHE.get(f)[off], rec=MANIFEST.dataPacked==='flatSourceTxtV2'?decodePackedDataLine(raw):raw; if(MANIFEST.dataPacked==='flatSourceV1'||MANIFEST.dataPacked==='flatSourceTxtV2'||MANIFEST.dataPacked==='flatSourceVarB64V3'||MANIFEST.dataPacked==='flatSourceVarBinV4')return {rid:row[MIDX.id],v:unflatVertices(rec[0]),src:decodeSource(rec[1],row[MIDX.src]),srcCount:1}; return {rid:rec[0],v:rec[1],src:rec[2]||[],srcCount:rec[3]||((rec[2]||[]).length)}}
