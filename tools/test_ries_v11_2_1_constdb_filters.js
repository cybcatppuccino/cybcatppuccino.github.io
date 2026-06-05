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
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const T=sandbox.__RIES_CONSTDB_TEST__;
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
let st=settingsFor('2.386110381167886',4);
const variants=T.constDbTransformRows(st);
const kinds=variants.map(v=>v.kind).join(',');
if(kinds!=='pow1,exp,log,powm1,pow2') throw new Error('unexpected constant DB transform order: '+kinds);
const tan=T.constantDbRecords().find(c=>c.id==='basic_tan_1_8');
if(!tan || !T.constDbIsPriorityNoiseConstant(tan)) throw new Error('tan(pi/8) should be priority-noise filtered');
if(T.constDbPriorityRelationRecords(T.constantDbRecords()).some(c=>c.id==='basic_tan_1_8')) throw new Error('tan(pi/8) still present in priority relation records');
if(T.constDbRelationUsesTargetNontrivially([{id:'b'},{id:'bc'},{id:'bdivc'}],[2n,1n,-1n])) throw new Error('b*(2+c-1/c)=0 should be filtered as target-independent');
if(!T.constDbRelationUsesTargetNontrivially([{id:'b'},{id:'c'}],[1n,-1n])) throw new Error('b-c=0 should remain useful');
const rows=T.constantDbRows(st);
if(!rows.some(r => /exp\(\(−12 \+ 4·c \+ c\^2\)\/12\)/.test(r.candidate||'') && /c = π/.test(r.valueHtml||''))){
  throw new Error('regression: missing exp((-12+4π+π²)/12) hit');
}
if(rows.some(r => /2·x \+ x·c .*x\/c = 0/.test(r.candidate||'') || /tan\(1π\/8\)/.test(r.valueHtml||''))){
  throw new Error('target-independent tan(pi/8) relation leaked into result rows');
}
console.log('PASS RIES v11.2.1 constant DB transform/filter regression');
process.exit(0);
