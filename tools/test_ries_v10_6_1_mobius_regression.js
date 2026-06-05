// RIES v10.6.1 regression: Möbius decimal module runs independently and
// catches simple pair identities in the initial search pass.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},insertAdjacentElement(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}};}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=false; els.doLog.checked=false; els.logHeight.value='400';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
let s=settingsFor('1.577215664901533',4);
if(!sandbox.__RIES_MOBIUS_TEST__.shouldRunMobiusRows(s)) throw new Error('Möbius should run even when doLog/doAlg are unchecked.');
let rows=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(s);
if(!rows.some(r=>r.candidate==='Möbius relation: x ≈ 1 + γ' && /\\gamma/.test(r.latex))) throw new Error('gamma+1 Möbius hit missing: '+rows.map(r=>r.candidate).join(' | '));
if(rows.some(r=>/(2·γ\)\/2|3·γ\)\/3|4·γ\)\/4)/.test(r.candidate))) throw new Error('coefficient-of-1 formatting/gcd regression: '+rows.map(r=>r.candidate).join(' | '));
s=settingsFor('6.1337074062362276',4);
rows=sandbox.__RIES_MOBIUS_TEST__.mobiusRelationRows(s);
if(!rows.some(r=>r.candidate==='Möbius relation: x ≈ exp(π/√3)' && r.latex==='x \\approx \\exp\\left(\\frac{\\pi}{\\sqrt{3}}\\right)')) throw new Error('exp(pi/sqrt(3)) Möbius hit missing/latex bad: '+rows.map(r=>r.candidate+' latex='+r.latex).join(' | '));
if(rows.some(r=>/2·π\/2·√3|3·π\/3·√3|4·π\/4·√3/.test(r.candidate))) throw new Error('duplicate scaled pi/sqrt3 rows were not normalized: '+rows.map(r=>r.candidate).join(' | '));
if(!fs.readFileSync('ries.html','utf8').includes('RIES <em>v11.4.3</em>')) throw new Error('ries.html version should be v11.4.3.');
console.log('PASS RIES v10.6.1 Möbius regression test');
