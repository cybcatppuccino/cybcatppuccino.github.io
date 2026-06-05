// RIES v10.6 regression: decimal Möbius module finds fractional-linear
// relations for x, e^x, and log|x|, returns at most five length-sorted rows,
// and participates as a distinct confidence module.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},insertAdjacentElement(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}};}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.logHeight.value='400';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
const ratio=(Math.PI+Math.E)/(Math.PI-Math.E);
let rows=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(settingsFor(ratio.toPrecision(16)));
if(!rows.length || !/\(π \+ e\)\/\(π − e\)/.test(rows[0].candidate)) throw new Error('direct Möbius ratio missing: '+rows.map(r=>r.candidate).join(' | '));
if(rows.length>5) throw new Error('Möbius module must return at most five rows.');
for(let i=1;i<rows.length;i++){
  if(sandbox.resultLengthFirstScore(rows[i]) < sandbox.resultLengthFirstScore(rows[i-1])-1e-9) throw new Error('Möbius rows not length-sorted.');
}
rows=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(settingsFor(Math.log(ratio).toPrecision(16)));
if(!rows.some(r=>/^Möbius relation: x ≈ log\(/.test(r.candidate) && /\\log\\left/.test(r.latex))) throw new Error('exp(x) Möbius transform missing/latex bad: '+rows.map(r=>r.candidate+' latex='+r.latex).join(' | '));
rows=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(settingsFor((-Math.exp(ratio)).toPrecision(16)));
if(!rows.some(r=>/^Möbius relation: x ≈ −exp\(/.test(r.candidate) && /^x \\approx -\\exp/.test(r.latex))) throw new Error('log(abs(x)) negative Möbius transform missing/latex bad: '+rows.map(r=>r.candidate+' latex='+r.latex).join(' | '));
const direct=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(settingsFor(ratio.toPrecision(16),4));
const tri=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(settingsFor(ratio.toPrecision(16),5));
if(!(tri.length>=direct.length)) throw new Error('continue effort should not reduce Möbius candidate count.');
const sorted=sandbox.confidenceSortedRows([
  {candidate:'irreducible algebraic: 9x^8 + 8x^7 − 7x + 1 = 0', degree:8, height:9n, err:1e-13},
  {candidate:'irreducible algebraic: 7x^7 + 6x^6 − 5x + 1 = 0', degree:7, height:9n, err:1e-13},
  direct[0],
  {candidate:'log|c| linear relation: x ≈ π/e', value:'x = 1; product = 1; terms 2; height 1', terms:2, height:1n, err:1e-13}
], settingsFor(ratio.toPrecision(16)));
const labels=sorted.map(r=>/^Möbius/.test(r.candidate)?'mobius':(/^log\|c\|/.test(r.candidate)?'log':(/algebraic/.test(r.candidate)?'alg':'other')));
if(labels.indexOf('mobius')<0 || labels.indexOf('mobius')>labels.indexOf('alg', labels.indexOf('alg')+1)) throw new Error('Möbius module head not round-robin visible: '+labels.join(','));
if(!fs.readFileSync('ries.html','utf8').includes('RIES <em>v11.3.1</em>')) throw new Error('ries.html version should be v11.3.1.');
console.log('PASS RIES v10.6 decimal Möbius module test');
