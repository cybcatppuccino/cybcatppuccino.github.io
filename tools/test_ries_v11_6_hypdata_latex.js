const fs = require('fs');
const vm = require('vm');
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
const context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{
  const src=String(script.src || script.getAttribute?.('src') || '');
  for(const lvl of [4,5,6]) if(src.includes(`ries-hypdata-v11_5_2-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-hypdata-v11_5_2-level${lvl}.js`,'utf8'), context);
  if(script.text) vm.runInContext(String(script.text), context);
  if(typeof script.onload==='function') script.onload(); return script;
} }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
console, performance:{ now:(()=>{let t=0; return()=>{t+=1; return t;};})() }, setTimeout, clearTimeout,
requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);
const T=context.__RIES_HYPDATA_TEST__;
if(!T || typeof T.ensureHypDataLoaded!=='function') throw new Error('hypdata hooks missing');
const mkLatex=T.hypDataMkLatex('P|0|1/2,1/2|1|1/2');
if(!mkLatex.includes('F_{1}') || !mkLatex.includes('\\frac{1}{2}')) throw new Error('hypergeom mk latex renderer should keep backslash fractions');
function firstRealTarget(payload){ const values=Buffer.from(payload.realValuesB64,'base64'); for(let i=0;i<values.length/8;i++){ const v=values.readDoubleLE(i*8); if(Number.isFinite(v) && Math.abs(v)>1e-12) return v; } throw new Error('no real target'); }
(async()=>{
  await T.ensureHypDataLoaded({stage:1,label:'hypergeometric pFq database',phase:'test'});
  const target=firstRealTarget(context.RIES_HYPDATA_V1152_CHUNKS[0]);
  const rows=await T.hypDataRowsAsync({target,complexTarget:false,level:4,sortMode:'confidence', modules:{hypData:true}, hypDataOptions:{depth1:true,depth2:true,depth3:true,multSimple:true,multGamma:true,multDeep:true}, moduleLimits:{hypData:5}, stageBudgets:{hypData1Ms:1000,hypData2Ms:5000,hypData3Ms:50000}});
  if(!rows.length) throw new Error('expected at least one hypdata row');
  if(!String(rows[0].latex).includes('\\approx')) throw new Error('hypdata row latex should contain escaped \\approx');
  if(!String(rows[0].valueHtml).includes('\\(') || !String(rows[0].valueHtml).includes('\\)')) throw new Error('hypdata valueHtml should keep LaTeX delimiters');
  if(String(rows[0].valueHtml).includes('xapprox')) throw new Error('hypdata valueHtml should not drop backslash in approx');
  console.log('PASS RIES v11.6 hypergeom latex escaping smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
