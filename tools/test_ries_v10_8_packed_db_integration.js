
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){
  return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
    classList:{contains(){return false},add(){},remove(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){}, insertAdjacentElement(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(name){return this.src||''}};
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='181'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='1'; els.shortEffort.value='1'; els.limit.value='12'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
vm.createContext(sandbox);
function loadAsset(){ vm.runInContext(fs.readFileSync('assets/shortform100k.js','utf8'), sandbox); }
const head={appendChild(script){ if(script.src && /assets\/shortform100k\.js/.test(script.src)){ loadAsset(); if(script.onload) setTimeout(script.onload,0); } return script; }};
sandbox.document={
  getElementById:id=>els[id]||null,
  querySelectorAll:(sel)=>[], querySelector:()=>fakeEl('qs'),
  createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}, appendChild:head.appendChild}, head, documentElement:head
};
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
let src=fs.readFileSync('ries-script.js','utf8');
src=src.replace('window.__RIES_PRECISION_TEST__ = {', 'window.__RIES_DB_TEST__ = { isShortformDbReady, ensureShortformDbLoaded, staticShortformRows, compactLiteralD };\n      window.__RIES_PRECISION_TEST__ = {');
vm.runInContext(src, sandbox);
(async()=>{
  if(sandbox.__RIES_DB_TEST__.isShortformDbReady()) throw new Error('DB should not be loaded before ensureShortformDbLoaded in this harness.');
  const loaded=await sandbox.__RIES_DB_TEST__.ensureShortformDbLoaded();
  if(!loaded) throw new Error('ensureShortformDbLoaded returned false.');
  if(!sandbox.RIES_SHORTFORM_100K_PACKED || sandbox.RIES_SHORTFORM_100K_PACKED.version!=='10.8.1') throw new Error('Packed DB was not installed with v10.8.1.');
  const got181=sandbox.RIES_SHORTFORM_100K_PACKED.get(181);
  if(!got181.some(e=>e === '(4+6!)/4' || e === '5!+61' || e === '181')) throw new Error('Packed get(181) unexpected: '+JSON.stringify(got181));
  const got139=sandbox.RIES_SHORTFORM_100K_PACKED.get(139);
  if(got139.includes('4!-5-5!')) throw new Error('Ambiguous invalid source expression was stored unparenthesized.');
  if(!got139.some(e=>e === '4!-(5-5!)' || e === '19+5!' || e === '139')) throw new Error('Packed get(139) unexpected: '+JSON.stringify(got139));
  const settings=sandbox.readSettings();
  const rows=sandbox.__RIES_DB_TEST__.staticShortformRows(settings);
  if(!rows.some(r=>/precomputed shortform:/.test(r.candidate))) throw new Error('staticShortformRows did not call packed DB for 181.');
  const ce=sandbox.__RIES_DB_TEST__.compactLiteralD(181);
  if(!ce || !/181|5!\+61|6!/.test(ce.s)) throw new Error('compactLiteralD did not use packed DB for <=100000 constants: '+JSON.stringify(ce));
  console.log('PASS RIES v10.8.1 packed DB integration test');
})().catch(err=>{ console.error(err); process.exit(1); });
