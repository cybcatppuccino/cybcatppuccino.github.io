// RIES v8.6 L-function scheduling and LaTeX smoke test.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=false; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null}; sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const testApi=sandbox.__RIES_LFUNC_TEST__;
const first=testApi.LFUNC_MONOMIALS.slice(0,4).map(m=>`${m.i},${m.j}`);
const expected=['1,0','1,1','1,-1','-1,0'];
if(first.join('|')!==expected.join('|')) throw new Error(`Low-level L-function monomial order changed: ${first.join('|')}`);
const cfg0=testApi.lfuncEffortConfig(0, 100000, 10000, 12);
const cfg5=testApi.lfuncEffortConfig(5, 100000, 10000, 24);
if(!(cfg0.quadBound < cfg5.quadBound && cfg0.maxLogComplexity < cfg5.maxLogComplexity && cfg0.highLog===false && cfg5.highLog===true)) throw new Error('L-function effort config should keep low level simple and unlock high-level searches later.');
const loConsts=testApi.lfuncLogConstants(false,0), hiConsts=testApi.lfuncLogConstants(false,5);
if(!(loConsts.every(c=>c.max<=1) && hiConsts.some(c=>c.max>=6))) throw new Error('Low-level log catalog should be much smaller than high-level catalog.');
const latex=sandbox.lfuncFormulaLatex('2^(-2)·3^(2)·5^(5/3)/L(f,1)');
if(latex.includes('^(') || latex.includes('^{2}')) throw new Error(`LaTeX exponent simplification failed: ${latex}`);
if(!latex.includes('2^{-2}') || !latex.includes('3^2') || !latex.includes('5^{5/3}') || !latex.includes('\\,')) throw new Error(`Unexpected L-function LaTeX output: ${latex}`);
console.log('PASS RIES v8.6 L-function low-level scheduling and LaTeX smoke test');
