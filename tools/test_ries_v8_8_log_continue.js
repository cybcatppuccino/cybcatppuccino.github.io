// RIES v8.8 log|c| continuation smoke test.
// Run from repository root: node tools/test_ries_v8_8_log_continue.js
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES v8.8')) throw new Error('ries.html was not updated to v8.8.');
for(const token of ['log(11)', 'label:\'e\'', 'logContinuationRemovalOrder', 'logContinuationBasisRows']){
  if(!fs.readFileSync('ries-script.js','utf8').includes(token)) throw new Error(`missing expected v8.8 log-continuation token: ${token}`);
}
let listenerCount = 0;
function fakeEl(id){
  return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
    classList:{contains(){return false},add(){},remove(){}}, addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(){}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}};
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='5';
els.logHeight.value='400'; els.logPrecision.value=''; els.logSlack.value='2';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){ listenerCount++; }, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const logTest=sandbox.window.__RIES_LOG_TEST__;
if(!logTest) throw new Error('v8.8 log test hook was not exposed.');
const idsSeen=logTest.logConstants.map(c=>c.id);
for(const id of ['log11','e']) if(!idsSeen.includes(id)) throw new Error(`missing optional ${id}`);
if(logTest.logConstants.find(c=>c.id==='log11').default) throw new Error('log(11) should be optional, not default.');
if(logTest.logConstants.find(c=>c.id==='e').default) throw new Error('e should be optional, not default.');
const order=logTest.logContinuationRemovalOrder.join(',');
if(order!=='loglog3,loglogpi,log5,loglog2') throw new Error(`unexpected removal order: ${order}`);
if(logTest.logContinueEffort({level:4})!==0) throw new Error('default level should keep original log implementation.');
if(logTest.logContinueEffort({level:5})!==1) throw new Error('first Continue should enumerate one optional candidate.');
if(logTest.logContinueEffort({level:6})!==2) throw new Error('second Continue should enumerate two optional candidates.');
console.log('PASS RIES v8.8 log continuation smoke test');
