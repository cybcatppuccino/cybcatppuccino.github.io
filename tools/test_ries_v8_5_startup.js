// RIES v8.5 startup/static-deploy smoke test.
// Run from repository root: node tools/test_ries_v8_4_startup.js
const fs = require('fs');
const vm = require('vm');
if(!fs.existsSync('.nojekyll')) throw new Error('Missing .nojekyll for static GitHub Pages deploy.');
let listenerCount = 0;
function fakeEl(id){
  return {
    id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(){}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}
  };
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const html = fs.readFileSync('ries.html','utf8');
for(const token of ['RIES <em>v9.1</em>','id="resultTools"','id="sortConfidenceBtn"','script defer src="ries-script.js"']){
  if(!html.includes(token)) throw new Error(`ries.html is missing expected v9 token: ${token}`);
}
if(/<th>formula<\/th>|<th>form \(N\.k\.\#\)<\/th>|<th>q-expansion<\/th>/.test(html)) throw new Error('v9 should not have standalone formula/form/q-expansion result headers.');
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
  body:{contains:()=>true, prepend(){}}
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
if(!els.resultBody.innerHTML.includes('Enter a target')) throw new Error('Initial result row was not rendered.');
if(listenerCount < 7) throw new Error('Expected v8.5 UI listeners, including sort buttons, were not attached.');
console.log('PASS RIES v8.5 startup/static-deploy smoke test');
