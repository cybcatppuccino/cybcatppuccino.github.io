// RIES v8.4 result-display regression test.
// Run from repository root: node tools/test_ries_v8_4_results_display.js
const fs = require('fs');
const vm = require('vm');
let listenerCount = 0;
function fakeEl(id){
  return {
    id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(child){ return child; }, prepend(){},
    querySelector(sel){ return fakeEl(sel); }, querySelectorAll(){ return []; }, getContext(){ return {}; }, closest(){ return null }, getAttribute(){ return ''; }
  };
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='5'; els.target.value='1.23456789012345';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
const headRow=fakeEl('thead-row');
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox;
sandbox.navigator={clipboard:null};
sandbox.document={
  getElementById:id=>els[id]||null,
  querySelector:sel=>sel==='.data thead tr' ? headRow : fakeEl(sel),
  querySelectorAll:()=>[],
  createElement:tag=>fakeEl(tag),
  addEventListener(){ listenerCount++; },
  body:{contains:()=>true, prepend(){}}
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const settings=sandbox.readSettings();
const rows=[
  {candidate:'L-quadratic #1: x = √(2·L(f,1))', latex:'x=\\sqrt{2L(f,1)}', value:'L(f,1) ≈ 0.7; quadratic', err:1e-14, score:5, lfuncCategory:'quadratic', lfuncEntryKey:'11.2.1:L1', lfuncLabel:'L(f,1)', modForm:{level:11, weight:2, index:1, code:'11.2.1'}, qLatex:'f(q)=q-q^{2}+O(q^{3})'},
  {candidate:'L-rational #1: x = 2·L(f,1)', latex:'x=2L(f,1)', value:'L(f,1) ≈ 0.7; rational', err:1e-15, score:1, lfuncCategory:'rational', lfuncEntryKey:'11.2.1:L1', lfuncLabel:'L(f,1)', modForm:{level:11, weight:2, index:1, code:'11.2.1'}, qLatex:'f(q)=q-q^{2}+O(q^{3})'},
  {candidate:'RIES equation: x^2 = 2', value:'x = 1.41421356237', err:1e-11}
];
const deduped=sandbox.dedupeEquivalentRows(rows, settings);
if(deduped.length!==2) throw new Error(`Expected L-equivalent rows to collapse to 2 total rows, got ${deduped.length}.`);
if(!deduped.some(r=>/^L-rational/.test(r.candidate))) throw new Error('The simpler/better rational L row should be kept over the equivalent quadratic row.');
sandbox.renderRows(deduped,{final:true, allRows:deduped, discoveryRows:deduped, settings, sorted:false});
if(/<th>formula<\/th>|<th>form \(N\.k\.\#\)<\/th>|<th>q-expansion<\/th>/.test(headRow.innerHTML)) throw new Error('formula/form/q-expansion must not be standalone result columns in v8.4.');
if(!/result-meta-label">formula/.test(els.resultBody.innerHTML) || !/result-meta-label">form/.test(els.resultBody.innerHTML) || !/result-meta-label">q-expansion/.test(els.resultBody.innerHTML)) throw new Error('formula/form/q-expansion metadata was not moved into the value/root cell.');
if(els.resultTools.hidden) throw new Error('Final result tools should be visible after a final render.');
console.log('PASS RIES v8.4 result display/dedupe smoke test');
