// RIES v9.5 pretty confidence sorting smoke test.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='3331.921045'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='5'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null}; sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);

function settingsFor(raw, target){ return {raw, normalizedRaw:raw, target}; }

const logRows=[
  {candidate:'log|c| linear relation: x ≈ e^(-17/8) * 2^(43/16) * 3^(9/2) * 5^(19/8) * exp(π*(-1/8))', value:'x = 3331.921045; product = 3331.92104522; terms 5; height 72', err:6.6e-11, height:72n, terms:5, score:1},
  {candidate:'log|c| linear relation: x ≈ 5 * π^(6) * log(2)', value:'x = 3331.921045; product = 3331.921044738; terms 3; height 6', err:7.88e-11, height:6n, terms:3, score:2}
];
let sorted=sandbox.confidenceSortedRows(logRows, settingsFor('3331.921045', 3331.921045));
if(!/5 \* π\^\(6\) \* log\(2\)/.test(sorted[0].candidate)) throw new Error('Sparse low-height log relation should outrank dense relation with only slightly smaller error.');

const algRows=[
  {candidate:'irreducible algebraic: 80x^5 + 718x^4 − 356x^3 − 1803x^2 − 381x − 1230 = 0', value:'x = -9.196375281581; |P(input)| ≈ 6.166*10^-13', err:6e-15, degree:5, height:4268n, score:1},
  {candidate:'irreducible algebraic: 153x^5 + 1607x^4 + 1879x^3 + 407x^2 + 416x + 626 = 0', value:'x = -9.196375281581; |P(input)| ≈ 8.223*10^-13', err:7e-15, degree:5, height:5088n, score:2},
  {candidate:'irreducible algebraic: 3x^3 + 19x^2 + 3x + 754 = 0', value:'x = -9.196375281581; |P(input)| ≈ 1.919*10^-12', err:5.33e-14, degree:3, height:779n, score:3}
];
sorted=sandbox.confidenceSortedRows(algRows, settingsFor('-9.196375281581', -9.196375281581));
if(!/^irreducible algebraic: 3x\^3/.test(sorted[0].candidate)) throw new Error('Compact cubic should outrank large quintics when precision loss is modest.');

const lRows=[
  {candidate:'L-rational #2: x = 2857*L(f,1)/(313*π)', value:'relative residual 1e-19', err:1e-19, score:0.000001, lfuncCategory:'rational', lfuncFormula:'2857*L(f,1)/(313*π)', modForm:{code:'1.2.1'}},
  {candidate:'L-rational #1: x = L(f,1)/4', value:'x/L(f,1) ≈ 1/4; relative residual 6.54e-18', err:6.54e-18, score:0.00001, lfuncCategory:'rational', lfuncFormula:'L(f,1)/4', modForm:{code:'594.4.2'}}
];
sorted=sandbox.confidenceSortedRows(lRows, settingsFor('2.29848605816074', 2.29848605816074));
if(!/L\(f,1\)\/4/.test(sorted[0].candidate)) throw new Error('Small-coefficient L-rational formula should outrank a much longer one when both verify well.');
if(sorted.length!==logRows.length && sorted.length!==2) throw new Error('Sort must not drop rows.');
console.log('PASS RIES v9.5 pretty confidence sorting smoke test');
