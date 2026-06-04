// RIES v8.1 startup guard smoke test.
// Run from repository root: node tools/test_ries_v8_1_startup.js
const fs = require('fs');
const vm = require('vm');
let listenerCount = 0;
function fakeEl(id){
  return {
    id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(){}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}
  };
}
const ids=['resultBody','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='5';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox;
sandbox.navigator={clipboard:null};
sandbox.document={
  getElementById:id=>els[id]||null,
  querySelectorAll:()=>[],
  querySelector:()=>fakeEl('qs'),
  createElement:tag=>fakeEl(tag),
  addEventListener(){ listenerCount++; },
  body:{contains:()=>true, prepend(){} }
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
if(!els.resultBody.innerHTML.includes('Enter a target')) throw new Error('Initial result row was not rendered.');
if(listenerCount < 5) throw new Error('Expected UI listeners were not attached.');
console.log('PASS RIES v8.1 startup smoke test');
