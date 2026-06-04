// RIES v10.9.2 regression: prioritized constant database algebraic multiples of π
// and eight-row constant-DB output contract.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},insertAdjacentElement(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}};}
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
if(sandbox.RIES_CONSTANT_DB_300_VERSION !== '10.9.2') throw new Error('constant DB version should be 10.9.2');
const html=fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v10.9.2</em>')) throw new Error('ries.html version should be v10.9.2');
if(!html.includes('assets/constantdb300.js?v=10.9.2')) throw new Error('constant DB cache-buster should be v10.9.2');
if(!html.includes('ries-script.js?v=10.9.2')) throw new Error('ries-script cache-buster should be v10.9.2');
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
const T=sandbox.__RIES_CONSTDB_TEST__;
if(T.constantDbBudgetMs(4,16) < 3200) throw new Error('direct constant DB budget is too small');
if(T.constantDbBudgetMs(5,16) < 5200) throw new Error('first-continue constant DB budget is too small');
function rowsFor(v, level=4){ return T.constantDbRows(settingsFor(String(v),level)); }
function assertEight(rows, name){ if(rows.length!==8) throw new Error(`${name}: expected 8 constant DB rows, got ${rows.length}`); }
function assertCleanLatex(rows, name){
  for(const r of rows){
    const latex=String(r.latex||'');
    if(/[\f\r\b]/.test(latex)) throw new Error(`${name}: bad control escape in latex ${JSON.stringify(latex)}`);
    if((latex.includes('alpha c') && !latex.includes('\\alpha c')) || (latex.includes('frac{') && !latex.includes('\\frac{'))) throw new Error(`${name}: lost latex backslash ${JSON.stringify(latex)}`);
  }
}
let rows=rowsFor('-2.143596015846163',4);
assertEight(rows,'cubic-root-times-pi');
assertCleanLatex(rows,'cubic-root-times-pi');
if(!rows.some(r => /c = π/.test(r.valueHtml||'') && /degree-3 ratio b\/c/.test(r.constantDbCategory||'') && /α\^3 \+ α \+ 1 = 0/.test(r.constantDbCategory||''))){
  throw new Error('missing α·π where α^3+α+1=0: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory+' | '+r.valueHtml).join('\n'));
}
rows=rowsFor(String(Math.cbrt(2)*Math.PI),4);
assertEight(rows,'cubert2-times-pi');
if(!rows.some(r => /c = π/.test(r.valueHtml||'') && /degree-3 ratio b\/c/.test(r.constantDbCategory||'') && /α\^3 − 2 = 0/.test(r.constantDbCategory||''))){
  throw new Error('missing cbrt(2)*π cubic result: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory).join('\n'));
}
rows=rowsFor(String(((1+Math.sqrt(5))/2)*Math.PI),4);
assertEight(rows,'phi-times-pi');
if(!rows.some(r => /c = π/.test(r.valueHtml||'') && /degree-2 ratio b\/c/.test(r.constantDbCategory||''))){
  throw new Error('missing φ*π quadratic result: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory+' | '+r.valueHtml).join('\n'));
}
console.log('PASS RIES v10.9.2 constant database budget, algebraic π, and 8-row output');
