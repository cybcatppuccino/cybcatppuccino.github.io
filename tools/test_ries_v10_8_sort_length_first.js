// RIES v10 sorting regression length-first confidence smoke test.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='1.154067477233'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='5'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null}; sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function settingsFor(raw, target){ return {raw, normalizedRaw:raw, target}; }

const gammaLogRows=[
  {candidate:'log|c| linear relation: x ≈ e^(-17/8) * 2^(43/16) * 3^(9/2) * 5^(19/8) * exp(π*(-1/8))', value:'x = 1.154067477233; product = 1.1540674772330001; terms 5; height 72', err:1e-16, height:72n, terms:5},
  {candidate:'log|c| linear relation: x ≈ π^(-1) * Γ(1/4)', value:'x = 1.154067477233; product = 1.154067477233; terms 2; height 2; direct sparse', err:3.41e-14, height:2n, terms:2},
];
let sorted=sandbox.confidenceSortedRows(gammaLogRows, settingsFor('1.154067477233', 1.154067477233));
if(!/π\^\(-1\) \* Γ\(1\/4\)/.test(sorted[0].candidate)) throw new Error('Short Γ(1/4)/π log relation must outrank dense high-precision artefact.');

const riesLogRows=[
  {candidate:'RIES equation: (√(x))² = exp(π)', value:'x = 23.14069263278', err:2e-12},
  {candidate:'log|c| linear relation: x ≈ exp(π)', value:'x = 23.1406926327; product = 23.14069263278; terms 1; height 1; direct sparse', err:8e-11, height:1n, terms:1},
  {candidate:'log|c| linear relation: x ≈ 2^(11/7) * 3^(13/8) * 5^(-17/9) * exp(π)', value:'terms 4; height 79', err:2e-13, height:79n, terms:4},
];
sorted=sandbox.confidenceSortedRows(riesLogRows, settingsFor('23.1406926327', 23.1406926327));
if(!/exp\(π\)/.test(sorted[0].candidate) || /2\^/.test(sorted[0].candidate)) throw new Error('Short exp(pi) explanation should be first among accepted candidates.');

const lRows=[
  {candidate:'L-rational #2: x = 2857*L(f,1)/(313*π)', value:'relative residual 1e-20', err:1e-20, lfuncCategory:'rational', modForm:{code:'1.2.1'}},
  {candidate:'L-rational #1: x = L(f,1)', value:'relative residual 5.397e-20', err:5.397e-20, lfuncCategory:'rational', modForm:{code:'2.2.1'}},
];
sorted=sandbox.confidenceSortedRows(lRows, settingsFor('2.29848605816074', 2.29848605816074));
if(!/x = L\(f,1\)$/.test(sorted[0].candidate)) throw new Error('x=L(f,1) must outrank longer small-residual L-rational formulas.');

const html=fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v10.8.1</em>')) throw new Error('ries.html version should be v10.8.1.');
const src=fs.readFileSync('ries-script.js','utf8');
if(!/renderFinalDefault[\s\S]*sorted:true/.test(src)) throw new Error('final display must default to confidence sorted order.');
console.log('PASS RIES v10 length-first confidence sorting smoke test');
