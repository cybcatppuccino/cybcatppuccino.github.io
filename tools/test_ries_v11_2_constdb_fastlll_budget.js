const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){},toggle(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},insertAdjacentElement(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}};}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=false; els.doLog.checked=false; els.logHeight.value='400';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), sandbox);
const source=fs.readFileSync('ries-script.js','utf8');
if(!source.includes('function constDbFastLLLReduce')) throw new Error('missing constant-DB-only fast LLL reducer');
vm.runInContext(source, sandbox);
const T=sandbox.__RIES_CONSTDB_TEST__;
if(T.constantDbBudgetMs(4,16)!==10000) throw new Error('level 4 constant DB budget should be 10000');
if(T.constantDbBudgetMs(5,16)!==30000) throw new Error('level 5 constant DB budget should be 30000');
if(T.constantDbBudgetMs(6,16)!==100000) throw new Error('level 6 constant DB budget should be 100000');
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
const rows=T.constantDbRows(settingsFor('2.386110381167886',4));
if(!rows.some(r => /exp\(\(−12 \+ 4·c \+ c\^2\)\/12\)/.test(r.candidate||'') && /c = π/.test(r.valueHtml||''))){
  throw new Error('missing exp((-12+4π+π²)/12) hit: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory).join('\n'));
}
const rel=T.constDbFindAlgebraicRatioLLL(Math.pow(2,1/5),5,15,24,Date.now()+200,1e-12);
if(!rel || rel.degree<5) throw new Error('constant DB algebraic LLL failed on fifth-root relation');
console.log('PASS RIES v11.2 constant DB fast LLL budgets and core hit');
process.exit(0);
