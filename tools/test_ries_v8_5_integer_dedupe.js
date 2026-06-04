// RIES v8.5 integer equivalence / de-duplication smoke test.
// Run from repository root: node tools/test_ries_v8_5_integer_dedupe.js
const fs = require('fs');
const vm = require('vm');
let listenerCount = 0;
function fakeEl(id){
  return {
    id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(child){ return child; }, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}
  };
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='4'; els.limit.value='10';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false; els.target.value='30731041173';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox;
sandbox.navigator={clipboard:null};
sandbox.document={
  getElementById:id=>els[id]||null,
  querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){ listenerCount++; },
  body:{contains:()=>true, prepend(){}}
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function row(expr){ return {candidate:`database: ${expr}`, value:'exact = 30731041173', copyValue:'30731041173', feature:sandbox.exprFeature(expr), digits:sandbox.digitCountExpr(expr), beauty:sandbox.shortRank({s:expr,ops:2,depth:1}), ops:2}; }
const equivalentPowers=[row('40/17·6^13'), row('20/51·6^14'), row('15/68·72^6')];
const picked=sandbox.selectDigitShortforms(equivalentPowers, 5);
if(picked.length!==1 || !picked[0].candidate.includes('40/17·6^13')) throw new Error('Equivalent rational power forms should collapse to the lowest digit-sum representative.');
const k1=sandbox.candidateEquivalenceKey(row('floor(40/17·6^13)-3570'));
const k2=sandbox.candidateEquivalenceKey(row('ceil(20/51·6^14)-3571'));
if(k1!==k2) throw new Error('floor(R)+k and ceil(equivalent R)+(k-1) should share an equivalence key.');
const s=sandbox.simplifyDExprIfBetter(sandbox.makeDExpr(16n/4n*(2n**35n), '16/4·2^35', 'test', 3, 2));
if(s.s!=='2^37') throw new Error(`Expected 16/4·2^35 to simplify to 2^37, got ${s.s}`);
if(!/rawAbsForDb<=100000n/.test(fs.readFileSync('ries-script.js','utf8'))) throw new Error('Large integers should not synchronously load the 100k shortform table.');
console.log('PASS RIES v8.5 integer dedupe/large-continue smoke test');
