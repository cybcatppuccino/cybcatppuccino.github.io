// RIES v8.3 integer lazy shortform smoke test.
// Run from repository root: node tools/test_ries_v8_3_integer_lazy_shortform.js
const fs = require('fs');
const vm = require('vm');
let listenerCount = 0;
let loadedShortform = false;
function fakeEl(id){
  return {
    id, tagName:String(id).toUpperCase(), value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(child){ if(child && child.textContent) this.innerHTML += child.textContent; return child; }, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null},
    insertAdjacentElement(){}, parentNode:null
  };
}
const ids=['resultBody','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='0'; els.limit.value='8';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
els.target.value='100000';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox;
sandbox.navigator={clipboard:null};
const document = {
  readyState:'complete',
  getElementById:id=>els[id]||null,
  querySelectorAll:()=>[],
  querySelector:()=>fakeEl('qs'),
  createElement:tag=>{
    const el=fakeEl(tag);
    el.getAttribute=(name)=>el[name] || '';
    return el;
  },
  addEventListener(){ listenerCount++; },
  body:{contains:()=>true, prepend(){}, appendChild(child){ return child; }},
  documentElement:{appendChild(child){ return child; }},
  head:{appendChild(child){
    if(child && /shortform100k\.js/.test(child.src||'')){
      vm.runInContext(fs.readFileSync('assets/shortform100k.js','utf8'), sandbox);
      loadedShortform = true;
      setTimeout(()=>child.onload && child.onload(), 0);
    }
    return child;
  }}
};
sandbox.document=document;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
(async()=>{
  await sandbox.solve();
  if(!loadedShortform) throw new Error('shortform100k.js was not lazy-loaded for integer solve.');
  if(!sandbox.RIES_SHORTFORM_100K) throw new Error('shortform database global is unavailable after lazy load.');
  if(!/10\^5|precomputed shortform|database/.test(els.resultBody.innerHTML + els.resultBody.textContent)) throw new Error('Integer solve did not render expected shortform/database output.');
  console.log('PASS RIES v8.3 integer lazy shortform smoke test');
  process.exit(0);
})().catch(err=>{ console.error(err); process.exit(1); });
