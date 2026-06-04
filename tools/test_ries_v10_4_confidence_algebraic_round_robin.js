// RIES v10.4 confidence sorting regression: irreducible algebraic rows must
// participate in length-first module round-robin instead of forming a precision block.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='23.1406926327'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='5'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null}; sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function settingsFor(raw, target){ return {raw, normalizedRaw:raw, target}; }
function cat(r){
  const c=String(r.candidate||'');
  if(/^log/.test(c)) return 'log';
  if(r.lfuncCategory) return 'lfunc';
  if(/algebraic/.test(c)) return 'alg';
  return 'other';
}
const rows=[
  {candidate:'irreducible algebraic: 123456789*x^8 - 987654321*x^7 + 135791357*x^6 - 246802468*x^5 + 11235813*x^4 - 314159*x^3 + 271828*x^2 - 161803*x + 1', value:'root', err:0, degree:8, height:987654321n, score:1},
  {candidate:'irreducible algebraic: x^2-2', value:'root', err:0, degree:2, height:2n, score:2},
  {candidate:'exact decimal input algebraic: x^3-x-1', value:'root', err:0, degree:3, height:1n, score:3},
  {candidate:'log|c| linear relation: x ≈ exp(π)', value:'terms 1; height 1', err:8e-11, height:1n, terms:1},
  {candidate:'log|c| linear relation: x ≈ 2^(11/7) * 3^(13/8) * 5^(-17/9) * exp(π)', value:'terms 4; height 79', err:2e-13, height:79n, terms:4},
  {candidate:'L-rational #1: x = L(f,1)', value:'relative residual 5e-10', err:5e-10, lfuncCategory:'rational', modForm:{code:'2.2.1'}},
  {candidate:'L-rational #2: x = 2857*L(f,1)/(313*π)', value:'relative residual 1e-12', err:1e-12, lfuncCategory:'rational', modForm:{code:'1.2.1'}},
];
const sorted=sandbox.confidenceSortedRows(rows, settingsFor('23.1406926327', 23.1406926327));
const cats=sorted.map(cat);
const firstLog=cats.indexOf('log');
const firstL=cats.indexOf('lfunc');
const firstAlg=cats.indexOf('alg');
if(firstLog < 0 || firstL < 0 || firstAlg < 0) throw new Error('Expected log, L-function and algebraic rows to be preserved.');
const firstLongAlg=sorted.findIndex(r=>/123456789\*x\^8/.test(r.candidate));
if(!(firstLog < firstLongAlg && firstL < firstLongAlg)) throw new Error(`Short log/L-function rows must precede longer exact algebraic rows, got ${cats.join(',')}`);
const secondLog=sorted.findIndex(r=>/2\^\(11\/7\)/.test(r.candidate));
const secondL=sorted.findIndex(r=>/2857\*L/.test(r.candidate));
if(firstLongAlg < secondLog || firstLongAlg < secondL) throw new Error('Algebraic rows should be sorted internally by length; long exact algebraic must not jump ahead of second-pass shorter rows.');
const algRows=sorted.filter(r=>cat(r)==='alg');
if(!/x\^2-2/.test(algRows[0].candidate)) throw new Error('Algebraic module must sort its own candidates by visible length/compactness first.');
if(!fs.readFileSync('ries.html','utf8').includes('RIES <em>v10.4</em>')) throw new Error('ries.html version should be v10.4.');
console.log('PASS RIES v10.4 algebraic confidence round-robin sorting test');
