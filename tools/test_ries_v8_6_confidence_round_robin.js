// RIES v8.6 confidence sorting smoke test.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='1.23456'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null}; sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const settings=sandbox.readSettings();
const rows=[
  {candidate:'RIES equation: x = 1.23456', value:'root', err:2e-6},
  {candidate:'RIES equation: x = 1.23', value:'root', err:1e-2},
  {candidate:'L-rational #1: x = L(f,1)', value:'L', err:1e-8, score:1, lfuncCategory:'rational'},
  {candidate:'L-rational #2: x = 2·L(f,1)', value:'L', err:1e-3, score:2, lfuncCategory:'rational'},
  {candidate:'algebraic: x^2-2', value:'root', err:1e-7, degree:2, height:2n, score:1},
  {candidate:'algebraic: 99*x^2-2', value:'root', err:1e-4, degree:2, height:99n, score:20}
];
const sorted=sandbox.confidenceSortedRows(rows, settings);
const cat=c=>/^RIES/.test(c)?'ries':/^L-rational/.test(c)?'lfunc':/^algebraic/.test(c)?'alg':'other';
const firstThree=sorted.slice(0,3).map(r=>cat(r.candidate));
if(new Set(firstThree).size!==3) throw new Error(`Sort should show each module's best result before second results, got ${firstThree.join(',')}`);
const riesRows=sorted.filter(r=>cat(r.candidate)==='ries');
if(!/1\.23456/.test(riesRows[0].candidate)) throw new Error('Typed-precision RIES hit should rank ahead of weaker RIES rows.');
if(sorted.length!==rows.length) throw new Error('Confidence sort must keep every input row.');
console.log('PASS RIES v8.6 confidence round-robin sorting smoke test');
