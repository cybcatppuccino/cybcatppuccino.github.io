// RIES v10.7.1 regression: low-precision constant database module.
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
if(sandbox.RIES_CONSTANT_DB_300_VERSION !== '10.7.1') throw new Error('constant DB version should be 10.7.1');
if(!Array.isArray(sandbox.RIES_CONSTANT_DB_300) || sandbox.RIES_CONSTANT_DB_300.length!==300) throw new Error('constant DB should have 300 records');
const gen = sandbox.RIES_CONSTANT_DB_300.filter(r=>r.source==='generated110').map(r=>r.label);
if(gen.some(x=>/^ζ\((2|4|6|8|10|12|14|16|18|20)\)$/.test(x))) throw new Error('generated list contains even zeta value');
if(gen.some(x=>/^√/.test(x))) throw new Error('generated list contains standalone quadratic radical');
for(const x of ['exp(√2)','exp(√3)','exp(√5)','exp(√6)','exp(√7)','exp(√10)']) if(!gen.includes(x)) throw new Error('missing '+x);
for(const x of ['Γ(1/5)Γ(2/5)','Γ(1/7)Γ(2/7)Γ(4/7)']) if(!gen.includes(x)) throw new Error('missing '+x);
if(!sandbox.RIES_CONSTANT_DB_300.some(r=>/ζ\(3\)/.test(r.label))) throw new Error('ζ(3) should be present via uploaded Apéry constant');
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
let s=settingsFor('3.141592653589793',4);
if(!sandbox.__RIES_CONSTDB_TEST__.shouldRunConstantDbRows(s)) throw new Error('constant DB should run on <=20 digit decimal');
let rows=sandbox.__RIES_CONSTDB_TEST__.constantDbRows(s);
let piHit=rows.find(r=>/constant database: x ≈ c/.test(r.candidate) && /c = π/.test(r.valueHtml||''));
if(!piHit) throw new Error('pi constant DB hit should use c in formula and identify c in details: '+rows.map(r=>r.candidate+' html='+r.valueHtml).join(' | '));
if(/operatorname\{/.test(piHit.latex) || /EulerMascheroni/.test(piHit.latex) || /\\pi/.test(piHit.latex)) throw new Error('constant DB formula latex should not spell out c notation: '+piHit.latex);
// Uploaded-190 record: Dottie number; value cell should preserve English description.
s=settingsFor('0.7390851332151606',4);
rows=sandbox.__RIES_CONSTDB_TEST__.constantDbRows(s);
const dottie=rows.find(r=>/Dottie/.test(r.valueHtml||''));
if(!dottie) throw new Error('uploaded Dottie constant hit missing: '+rows.map(r=>r.candidate).join(' | '));
if(!/constant database: x ≈ c/.test(dottie.candidate)) throw new Error('Dottie formula should use c: '+dottie.candidate);
if(!/The only real solution of x = cos\(x\)/.test(dottie.valueHtml||'')) throw new Error('uploaded English description missing: '+(dottie.valueHtml||''));
// Mobius subtest in the constant database: x = gamma+1 should be caught as 1 + c.
s=settingsFor('1.577215664901533',4);
rows=sandbox.__RIES_CONSTDB_TEST__.constantDbRows(s);
const gammaRow=rows.find(r=>/constant database: x ≈ 1 \+ c/.test(r.candidate) && /Euler-Mascheroni/.test(r.valueHtml||''));
if(!gammaRow) throw new Error('constant DB gamma+1 hit missing or not using c: '+rows.map(r=>r.candidate+' html='+r.valueHtml).join(' | '));
const sorted=sandbox.confidenceSortedRows(rows.concat([{candidate:'Möbius relation: x ≈ 1 + γ', latex:'x \\approx 1+\\gamma', err:0, mobiusCategory:'direct'}]), s);
if(!sorted.length) throw new Error('confidence sorting returned no rows');
const html=fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v10.7.1</em>')) throw new Error('ries.html version should be v10.7.1.');
if(!html.includes('assets/constantdb300.js?v=10.7.1')) throw new Error('ries.html should load constantdb300.js v10.7.1');
console.log('PASS RIES v10.7.1 constant database module');
