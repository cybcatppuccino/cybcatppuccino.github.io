const fs = require('fs');
const vm = require('vm');
function assert(cond,msg){ if(!cond) throw new Error(msg); }
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
let context;
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, defaultChecked:true, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, matches(){ return false; },
    querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{
    const src=String(script.src || script.getAttribute?.('src') || '');
    for(const lvl of [4,5,6]) if(src.includes(`ries-hypdata-v11_9_2-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-hypdata-v11_9_2-level${lvl}.js`,'utf8'), context);
    if(script.text) vm.runInContext(String(script.text), context);
    if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
  console, performance:{ now:(()=>{let t=0; return()=>{t+=1; return t;};})() }, setTimeout, clearTimeout,
  requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
  URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
  atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js']) vm.runInContext(fs.readFileSync(f,'utf8'), context);

const T=context.__RIES_HYPDATA_TEST__;
assert(T && typeof T.ensureHypDataLoaded==='function', 'hypdata hooks missing');
assert(T.RIES_HYPDATA_ASSET_LEVELS[0].url.includes('ries-hypdata-v11_9_2-level4.js'), 'v11.9.2 hypdata level4 URL missing');
assert(fs.readFileSync('ries.html','utf8').includes('RIES <em>v11.9.2</em>') && fs.readFileSync('ries.html','utf8').includes('ries-script.js?v=11.9.2'), 'v11.9.2 page/cache-buster missing');

function bytes(b64){ return Buffer.from(b64,'base64'); }
function u32(b64){ const b=bytes(b64), out=[]; for(let i=0;i<b.length;i+=4) out.push(b.readUInt32LE(i)); return out; }
function u8(b64){ return [...bytes(b64)]; }
function f64(b64){ const b=bytes(b64), out=[]; for(let i=0;i<b.length;i+=8) out.push(b.readDoubleLE(i)); return out; }

(async()=>{
  assert(await T.ensureHypDataLoaded({stage:1,label:'hypergeometric pFq database',phase:'test'}), 'level4 did not load');
  assert(context.RIES_HYPDATA_V1192_CHUNKS && context.RIES_HYPDATA_V1192_CHUNKS[0], 'v11.9.2 chunk global missing');
  const ch=context.RIES_HYPDATA_V1192_CHUNKS[0];
  assert(ch.version==='11.9.2' && ch.rows===29631 && ch.realCompB64, 'level4 payload mismatch');
  assert(ch.mkBlob.includes('P|0|1/12,1/12|-1/2|-1'), 'data.zip 2F1 grid row missing from level4');
  const comps=u8(ch.realCompB64);
  assert(comps.includes(1) && comps.includes(2), 'real/imag scalar projection component codes missing');
  const vals=f64(ch.realValuesB64), rows=u32(ch.realRowB64);
  const idx=comps.findIndex(c=>c===1 && Number.isFinite(vals[comps.indexOf(c)]) && Math.abs(vals[comps.indexOf(c)])>1e-6);
  const k=comps.findIndex((c,i)=>c===1 && Number.isFinite(vals[i]) && Math.abs(vals[i])>1e-6);
  assert(k>=0, 'no Re(H) projection available for smoke test');
  const target=vals[k];
  const out=await T.hypDataRowsAsync({target,complexTarget:false,level:4,sortMode:'confidence',modules:{hypData:true},moduleLimits:{hypData:20}});
  assert(out.length>0, 'level4 projection target did not search');
  assert(out.some(r => String(r.latex||'').includes('\\operatorname{Re}') || String(r.valueHtml||'').includes('Re(H)')), 'Re(H) LaTeX/display not found in projection hit');

  assert(await T.ensureHypDataLoaded({stage:3,label:'hypergeometric pFq database',phase:'test'}), 'level6 did not load');
  assert(T.hypDataLoadedChunks(3).length===3, 'level6 should have all cumulative chunks');
  const out6=await T.hypDataRowsAsync({target,complexTarget:false,level:6,sortMode:'confidence',modules:{hypData:true},moduleLimits:{hypData:20}});
  assert(out6.length>0, 'level6 cumulative search missed level4 data/projection target');
  console.log('PASS RIES v11.9.2 hypdata merge, Re/Im projections, and cumulative level4/5/6 search smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
