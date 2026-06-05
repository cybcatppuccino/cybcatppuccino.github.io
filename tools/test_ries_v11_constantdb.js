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
if(sandbox.RIES_CONSTANT_DB_300_VERSION !== '11') throw new Error('constant DB version should be 11');
const html=fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v11.1</em>')) throw new Error('ries.html visible version should be v11.1');
if(!html.includes('assets/constantdb300.js?v=11.1')) throw new Error('constant DB cache-buster should be v11.1');
if(!html.includes('ries-script.js?v=11.1')) throw new Error('ries-script cache-buster should be v11.1');
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
const T=sandbox.__RIES_CONSTDB_TEST__;
if(T.constantDbBudgetMs(4,16) < 5000) throw new Error('level 4 constant DB budget is too small');
if(T.constantDbBudgetMs(5,16) < 10000) throw new Error('level 5 constant DB budget is too small');
if(T.constantDbBudgetMs(6,16) < 30000) throw new Error('level 6 constant DB budget is too small');
if(T.riesLevelModuleBudgetMs(4)!==5000 || T.riesLevelModuleBudgetMs(5)!==10000 || T.riesLevelModuleBudgetMs(6)!==30000) throw new Error('module budget table mismatch');
const s10=settingsFor('1.2345678901',4);
if(T.typedDecimalScaleDigits(s10)!==10) throw new Error('decimal scale should be 10');
if(T.constDbMaxRelativeError(s10) > 1.01e-7 || T.constDbMaxRelativeError(s10) < 0.99e-7) throw new Error('constant DB precision gate should be 1e-7 for 10 decimal places');
function rowsFor(v, level=4){ return T.constantDbRows(settingsFor(String(v),level)); }
function assertCleanLatex(rows, name){
  for(const r of rows){
    const latex=String(r.latex||'');
    if(/[\f\r\b\x07]/.test(latex)) throw new Error(`${name}: bad control escape in latex ${JSON.stringify(latex)}`);
    if(latex.includes('alpha c') && !latex.includes('\\alpha c')) throw new Error(`${name}: lost alpha backslash ${JSON.stringify(latex)}`);
    if(latex.includes('frac{') && !latex.includes('\\frac{')) throw new Error(`${name}: lost frac backslash ${JSON.stringify(latex)}`);
  }
}
let rows=rowsFor('-2.143596015846163',4);
assertCleanLatex(rows,'cubic-root-times-pi');
if(rows.some(r => /nearest rational multiple of database constant/.test(r.constantDbCategory||''))) throw new Error('weak fallback row should not be returned for cubic-root-times-pi');
const piCubic=rows.find(r => /c = π/.test(r.valueHtml||'') && /degree-3 ratio b\/c/.test(r.constantDbCategory||'') && /α\^3 \+ α \+ 1 = 0/.test(r.constantDbCategory||''));
if(!piCubic) throw new Error('missing α·π where α^3+α+1=0: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory+' | '+r.valueHtml).join('\n'));
if(!/\\alpha c/.test(piCubic.latex)) throw new Error('pi cubic latex lost alpha: '+JSON.stringify(piCubic.latex));
if((piCubic.err||0)>1e-12) throw new Error('pi cubic err should be near zero, got '+piCubic.err);
rows=rowsFor(String(Math.cbrt(2)*Math.PI),4);
if(!rows.some(r => /c = π/.test(r.valueHtml||'') && /degree-3 ratio b\/c/.test(r.constantDbCategory||'') && /α\^3 − 2 = 0/.test(r.constantDbCategory||''))){
  throw new Error('missing cbrt(2)*π cubic result: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory).join('\n'));
}
rows=rowsFor(String(((1+Math.sqrt(5))/2)*Math.PI),4);
if(!rows.some(r => /c = π/.test(r.valueHtml||'') && /degree-2 ratio b\/c/.test(r.constantDbCategory||''))){
  throw new Error('missing φ*π quadratic result: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory+' | '+r.valueHtml).join('\n'));
}

rows=rowsFor('0.03876817960292',4);
assertCleanLatex(rows,'false-positive-guard');
if(rows.some(r => /α\^3 = 0/.test(r.constantDbCategory||'') || /√\(α·c\)/.test(r.candidate||''))){
  throw new Error('regressed false-positive guard for α^3=0: '+rows.map(r=>r.candidate+' | '+r.constantDbCategory).join('\n'));
}
const s18=settingsFor('1.234567890123456789',4);
if(T.typedInputPrecisionForDouble(s18)!==15) throw new Error('double precision cap should be 15 for 18-digit Number modules');
if(T.constDbMaxRelativeError(s18) > 1.01e-12 || T.constDbMaxRelativeError(s18) < 0.99e-12) throw new Error('constant DB double-based precision gate should be capped at 1e-12 for 18-digit input');
if(typeof T.constantDbRowsAsync !== 'function') throw new Error('missing cooperative async constant DB test hook');
console.log('PASS RIES v11.1 constant database budgets, precision gate, latex, algebraic π, and false-positive guard tests');
