// RIES v10.8.1 regression: LaTeX backslash escaping in low-precision constant DB rows.
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
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
if(sandbox.RIES_CONSTANT_DB_300_VERSION !== '10.8.1') throw new Error('constant DB version should be 10.8.1');
const html=fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v10.8.1</em>')) throw new Error('ries.html version should be v10.8.1.');
if(!html.includes('assets/constantdb300.js?v=10.8.1')) throw new Error('constant DB cache-buster should be v10.8.1');
if(!html.includes('ries-script.js?v=10.8.1')) throw new Error('ries-script cache-buster should be v10.8.1');
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
function rowsFor(v){ return sandbox.__RIES_CONSTDB_TEST__.constantDbRows(settingsFor(String(v),4)); }
function assertCleanLatex(row){
  if(!row) throw new Error('missing row');
  const latex = row.latex || '';
  if(/[\f\r\b]/.test(latex)) throw new Error('LaTeX contains JS control escape: '+JSON.stringify(latex));
  if(latex.includes('x approx') || (latex.includes('frac{') && !latex.includes('\\frac{')) || latex.includes('logleft') || latex.includes('right)') && !latex.includes('\\right)')) throw new Error('LaTeX lost backslashes: '+JSON.stringify(latex));
}
let rows = rowsFor(3*Math.PI/16);
let fracRow = rows.find(r => r.latex === 'x \\approx \\frac{3\\,c}{16}');
assertCleanLatex(fracRow);
rows = rowsFor(6+Math.PI+Math.PI*Math.PI);
let polyRow = rows.find(r => r.latex === 'x \\approx 6 + c + c^{2}');
assertCleanLatex(polyRow);
rows = rowsFor(Math.log(6+Math.PI+Math.PI*Math.PI));
let logRow = rows.find(r => r.latex === 'x \\approx \\log\\left(6 + c + c^{2}\\right)');
assertCleanLatex(logRow);
// Spot-check older non-constant-DB LaTeX helpers still produce escaped LaTeX.
const exprLatex = sandbox.__RIES_INTEGER_TEST__.exprToLatex('(1+2)/3');
if(!/\\frac\{1\+2\}\{3\}/.test(exprLatex)) throw new Error('exprToLatex fraction rendering regressed: '+exprLatex);
const special = sandbox.__RIES_PRECISION_TEST__.specialDecimalConstantRows(settingsFor('3.625609908221908311930685155867672002995167682880065467433377', 4), 1);
if(!special.some(r => r.latex === 'x=\\Gamma(1/4)')) throw new Error('special constant Gamma LaTeX lost backslash: '+special.map(r=>r.latex).join(' | '));
console.log('PASS RIES v10.8.1 LaTeX escaping regression');
